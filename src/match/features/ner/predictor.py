"""Tokenization, batching and inference for the word-level NER model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .entities import NerEntity
from .model import WordNERModel
from .postprocessing import NerPostprocessor


@dataclass(slots=True)
class WordNerPredictor:
    model: WordNERModel
    tokenizer: PreTrainedTokenizerBase
    postprocessor: NerPostprocessor
    device: torch.device
    batch_size: int = 512
    max_length: int = 100
    use_amp: bool = True

    WORD_RE = re.compile(r"\w+(?:[./+\-']\w+)*", flags=re.UNICODE)

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_length < 8:
            raise ValueError("batch_size must be positive and max_length at least 8")
        self.use_amp = self.use_amp and self.device.type == "cuda"
        self.model.to(self.device).eval()

    def _word_spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in self.WORD_RE.finditer(text)]

    def _align(
        self,
        offsets: torch.Tensor,
        spans_batch: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        word_ids = torch.full(offsets.shape[:2], -1, dtype=torch.long)
        subword_positions = torch.zeros(offsets.shape[:2], dtype=torch.long)
        for batch, spans in enumerate(spans_batch):
            word_index = 0
            previous_word = -1
            position = 0
            for token_index, (start, end) in enumerate(offsets[batch].tolist()):
                if start == end:
                    continue
                while word_index < len(spans) and spans[word_index][1] <= start:
                    word_index += 1
                if word_index >= len(spans):
                    break
                word_start, word_end = spans[word_index]
                if start < word_end and end > word_start:
                    word_ids[batch, token_index] = word_index
                    if word_index == previous_word:
                        position += 1
                    else:
                        previous_word = word_index
                        position = 0
                    subword_positions[batch, token_index] = position
        return word_ids, subword_positions

    @torch.inference_mode()
    def _extract_batch(self, texts: Sequence[str]) -> list[list[NerEntity]]:
        spans = [self._word_spans(text) for text in texts]
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        word_ids, subword_positions = self._align(offsets, spans)
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        word_ids = word_ids.to(self.device)
        subword_positions = subword_positions.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            contextual, semantic, lengths = self.model.encode_words(
                inputs["input_ids"],
                inputs["attention_mask"],
                word_ids,
                subword_positions,
                inputs.get("token_type_ids"),
            )
            logits = self.model.classifier(contextual)
        return self.postprocessor.process(texts, spans, logits, semantic, lengths)

    def extract(self, texts: Sequence[str]) -> list[list[NerEntity]]:
        values = ["" if text is None else str(text) for text in texts]
        if not values:
            return []
        results: list[list[NerEntity]] = []
        index = 0
        current_batch_size = min(self.batch_size, len(values))
        while index < len(values):
            chunk = values[index : index + current_batch_size]
            try:
                results.extend(self._extract_batch(chunk))
                index += len(chunk)
            except torch.cuda.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
        return results


__all__ = ["WordNerPredictor"]
