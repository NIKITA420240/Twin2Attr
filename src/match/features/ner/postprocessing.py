"""Convert word-level predictions into cleaned product entities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pymorphy3
import torch
import torch.nn.functional as F

from .entities import NerEntity


@dataclass(slots=True)
class _Run:
    batch: int
    class_id: int
    start: int
    end: int
    score: float | None


class NerPostprocessor:
    PREPOSITIONS = {
        "для",
        "от",
        "на",
        "с",
        "со",
        "из",
        "по",
        "к",
        "в",
        "во",
        "под",
        "над",
        "при",
        "без",
    }
    IMPOSSIBLE_STANDALONE = PREPOSITIONS | {"и", "или", "а", "но"}

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        cluster_centers: dict[str, torch.Tensor] | None = None,
        device: torch.device,
    ) -> None:
        self.class_names = tuple(class_names)
        self.label2id = {label: index for index, label in enumerate(class_names)}
        self.device = device
        self.o_id = self.label2id["O"]
        self.type_id = self.label2id.get("тип", -1)
        self.article_id = self.label2id.get("артикул", -1)
        self.no_pure_number = {
            self.label2id[label]
            for label in (
                "бренд",
                "тип",
                "материал",
                "цвет",
                "назначение",
            )
            if label in self.label2id
        }
        self.relabel_targets = {
            self.label2id[label]
            for label in (
                "цвет",
                "материал",
                "бренд",
                "модель",
                "артикул",
            )
            if label in self.label2id
        }
        self.min_similarity = {
            self.label2id[label]: value
            for label, value in {
                "цвет": 0.72,
                "материал": 0.62,
                "бренд": 0.74,
                "модель": 0.78,
                "артикул": 0.82,
            }.items()
            if label in self.label2id
        }
        self.min_gap = {
            self.label2id[label]: value
            for label, value in {
                "цвет": 0.20,
                "материал": 0.16,
                "бренд": 0.20,
                "модель": 0.25,
                "артикул": 0.30,
            }.items()
            if label in self.label2id
        }
        self.relabel_target_mask = torch.zeros(
            len(self.class_names),
            dtype=torch.bool,
            device=self.device,
        )
        self.relabel_target_mask[list(self.relabel_targets)] = True
        self.minimum_similarity = torch.full(
            (len(self.class_names),),
            torch.inf,
            device=self.device,
        )
        self.minimum_gap = torch.full_like(self.minimum_similarity, torch.inf)
        for class_id, value in self.min_similarity.items():
            self.minimum_similarity[class_id] = value
        for class_id, value in self.min_gap.items():
            self.minimum_gap[class_id] = value
        self.centers, self.center_class_ids = self._prepare_centers(
            cluster_centers or {}
        )
        self.morph = pymorphy3.MorphAnalyzer()

    def _prepare_centers(
        self,
        centers: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        parts: list[torch.Tensor] = []
        class_ids: list[int] = []
        for label, values in centers.items():
            if label not in self.label2id or not len(values):
                continue
            normalized = F.normalize(values.float(), p=2, dim=1)
            parts.append(normalized)
            class_ids.extend([self.label2id[label]] * len(normalized))
        if not parts:
            return None, None
        return (
            torch.cat(parts).to(self.device),
            torch.tensor(class_ids, dtype=torch.long, device=self.device),
        )

    @lru_cache(maxsize=100_000)
    def _part_of_speech(self, word: str) -> str | None:
        return self.morph.parse(word.casefold())[0].tag.POS

    def _extract_runs(
        self,
        class_ids: torch.Tensor,
        probabilities: torch.Tensor,
        word_lengths: torch.Tensor,
    ) -> list[_Run]:
        ids = class_ids.cpu().numpy()
        scores = probabilities.cpu().numpy()
        lengths = word_lengths.cpu().numpy()
        runs: list[_Run] = []
        for batch, length in enumerate(lengths):
            row = ids[batch, :length]
            if not len(row):
                continue
            boundaries = np.flatnonzero(
                np.r_[True, row[1:] != row[:-1], True]
            )
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                class_id = int(row[start])
                if class_id == self.o_id:
                    continue
                run_score = float(scores[batch, start:end, class_id].mean())
                runs.append(_Run(batch, class_id, int(start), int(end), run_score))
        return runs

    def _rule_cleanup(
        self,
        texts: Sequence[str],
        spans: Sequence[Sequence[tuple[int, int]]],
        runs: Sequence[_Run],
    ) -> list[_Run]:
        cleaned: list[_Run] = []
        for run in runs:
            segment_start: int | None = None
            for word_index in range(run.start, run.end + 1):
                valid = False
                if word_index < run.end:
                    start, end = spans[run.batch][word_index]
                    word = texts[run.batch][start:end]
                    valid = not (
                        (run.class_id in self.no_pure_number and word.isdigit())
                        or (run.class_id == self.article_id and word.isalpha())
                    )
                if valid:
                    segment_start = (
                        word_index if segment_start is None else segment_start
                    )
                    continue
                if segment_start is None:
                    continue
                segment_end = word_index
                if segment_end - segment_start == 1:
                    start, end = spans[run.batch][segment_start]
                    if (
                        texts[run.batch][start:end].casefold()
                        in self.IMPOSSIBLE_STANDALONE
                    ):
                        segment_start = None
                        continue
                cleaned.append(
                    _Run(
                        run.batch,
                        run.class_id,
                        segment_start,
                        segment_end,
                        run.score,
                    )
                )
                segment_start = None
        return cleaned

    def _semantic_cleanup(
        self,
        semantic_hidden: torch.Tensor,
        runs: Sequence[_Run],
    ) -> list[_Run]:
        if not runs or self.centers is None or self.center_class_ids is None:
            return list(runs)
        coordinates = torch.tensor(
            [
                (run.batch, run.class_id, run.start, run.end)
                for run in runs
            ],
            dtype=torch.long,
            device=self.device,
        )
        batch_ids = coordinates[:, 0]
        predicted_ids = coordinates[:, 1]
        starts = coordinates[:, 2]
        ends = coordinates[:, 3]
        prefix = F.pad(semantic_hidden.float().cumsum(dim=1), (0, 0, 1, 0))
        embeddings = prefix[batch_ids, ends] - prefix[batch_ids, starts]
        embeddings /= (ends - starts).float().unsqueeze(1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarities = embeddings @ self.centers.T
        class_scores = torch.full(
            (len(runs), len(self.class_names)),
            -torch.inf,
            device=self.device,
        )
        center_classes = self.center_class_ids[None, :].expand(len(runs), -1)
        class_scores.scatter_reduce_(
            1,
            center_classes,
            similarities,
            reduce="amax",
            include_self=True,
        )
        own_scores = class_scores.gather(1, predicted_ids[:, None]).squeeze(1)
        best_scores, best_ids = class_scores.max(dim=1)
        top_two = class_scores.topk(min(2, class_scores.size(1)), dim=1).values
        second_scores = (
            top_two[:, 1]
            if top_two.size(1) == 2
            else torch.full_like(best_scores, -torch.inf)
        )
        gaps = best_scores - own_scores
        dominance = best_scores - second_scores
        own_available = torch.isfinite(own_scores)
        agrees = best_ids == predicted_ids
        relabel = (
            own_available
            & ~agrees
            & self.relabel_target_mask[best_ids]
            & (best_scores >= self.minimum_similarity[best_ids])
            & (gaps >= self.minimum_gap[best_ids])
            & (dominance >= 0.06)
        )
        new_ids = predicted_ids.clone()
        new_ids[relabel] = best_ids[relabel]
        undecided = own_available & ~agrees & ~relabel
        type_reject = (
            undecided
            & (predicted_ids == self.type_id)
            & self.relabel_target_mask[best_ids]
            & (gaps >= 0.24)
        )
        general_reject = undecided & ~type_reject & (gaps >= 0.20)
        low_own_reject = (
            undecided
            & ~type_reject
            & ~general_reject
            & (own_scores < 0.40)
            & (gaps >= 0.12)
        )
        keep = ~(type_reject | general_reject | low_own_reject)
        keep_values = keep.cpu().tolist()
        class_values = new_ids.cpu().tolist()
        return [
            _Run(
                run.batch,
                int(class_id),
                run.start,
                run.end,
                None if int(class_id) != run.class_id else run.score,
            )
            for run, should_keep, class_id in zip(
                runs,
                keep_values,
                class_values,
            )
            if should_keep
        ]

    def _expand(
        self,
        texts: Sequence[str],
        spans: Sequence[Sequence[tuple[int, int]]],
        word_lengths: torch.Tensor,
        runs: Sequence[_Run],
    ) -> list[_Run]:
        lengths = word_lengths.cpu().tolist()
        expanded: list[_Run] = []
        for run in runs:
            end = run.end
            words = [
                texts[run.batch][start:stop].casefold()
                for start, stop in spans[run.batch]
            ]
            while end < lengths[run.batch]:
                previous = words[end - 1]
                following = words[end]
                if previous in self.PREPOSITIONS or (
                    self._part_of_speech(previous) == "ADJF"
                    and self._part_of_speech(following) == "NOUN"
                ):
                    end += 1
                    continue
                break
            expanded.append(
                _Run(run.batch, run.class_id, run.start, end, run.score)
            )
        return expanded

    @torch.inference_mode()
    def process(
        self,
        texts: Sequence[str],
        spans: Sequence[Sequence[tuple[int, int]]],
        logits: torch.Tensor,
        semantic_hidden: torch.Tensor,
        word_lengths: torch.Tensor,
    ) -> list[list[NerEntity]]:
        probabilities = logits.softmax(dim=-1)
        runs = self._extract_runs(logits.argmax(dim=-1), probabilities, word_lengths)
        runs = self._rule_cleanup(texts, spans, runs)
        runs = self._semantic_cleanup(semantic_hidden, runs)
        runs = self._expand(texts, spans, word_lengths, runs)
        result: list[list[NerEntity]] = [[] for _ in texts]
        for run in runs:
            start = spans[run.batch][run.start][0]
            end = spans[run.batch][run.end - 1][1]
            result[run.batch].append(
                NerEntity(
                    label=self.class_names[run.class_id],
                    text=texts[run.batch][start:end],
                    start=start,
                    end=end,
                    score=run.score,
                )
            )
        return result


__all__ = ["NerPostprocessor"]
