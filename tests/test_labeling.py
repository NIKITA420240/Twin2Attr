import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import polars as pl

from match.config import (
    DatasetLabelingSettings,
    LlmLabelingSettings,
    load_app_config_file,
)
from match.labeling import (
    LlmLabelingResult,
    MatchLabel,
    _run_labeling_rounds,
    labeling_fingerprint,
)
from match.paths import PROJECT_ROOT
from match.workflows.label import label_dataset, select_unlabeled_pairs


def _llm_settings(**overrides) -> LlmLabelingSettings:
    values = {
        "base_url": "https://llm.example.test",
        "model": "test/model",
        "token_env": "TEST_LLM_TOKEN",
        "temperature": 0.0,
        "verify_ssl": True,
        "request_batch_size": 64,
        "max_concurrency": 64,
        "min_concurrency": 32,
        "max_attempts": 1,
        "max_rounds": 3,
        "retry_base_seconds": 0.0,
        "max_prompt_chars": 6000,
    }
    values.update(overrides)
    return LlmLabelingSettings(**values)


class LabelingSelectionTests(unittest.TestCase):
    def test_filters_boundaries_and_excludes_canonical_existing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "matches.parquet"
            pl.DataFrame(
                {
                    "id1": [1, 2, 3, 5, 7, 9],
                    "id2": [2, 1, 4, 6, 8, 10],
                    "target": [0.55, 0.55, 0.54, 0.56, 0.53, 0.57],
                }
            ).write_parquet(source_path)
            settings = DatasetLabelingSettings(
                source_matches_path=source_path,
                items_path=root / "items.parquet",
                output_path=root / "annotations.parquet",
                score_column="target",
                lower_p=0.54,
                upper_p=0.56,
                sample_size=10,
                seed=42,
                checkpoint_every_batches=10,
                llm=_llm_settings(),
            )
            existing = pl.DataFrame(
                {
                    "pair_left": [1],
                    "pair_right": [2],
                    "labeling_fingerprint": ["current"],
                }
            )

            result = select_unlabeled_pairs(
                settings,
                fingerprint="current",
                existing_annotations=existing,
            )

            self.assertEqual(result.source_rows, 6)
            self.assertEqual(result.interval_rows, 4)
            self.assertEqual(result.interval_unique_pairs, 3)
            self.assertEqual(result.already_labeled_rows, 1)
            self.assertEqual(result.remaining_rows, 2)
            self.assertEqual(
                set(result.pairs.select("pair_left", "pair_right").iter_rows()),
                {(3, 4), (5, 6)},
            )

    def test_repeated_selection_advances_to_a_disjoint_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "matches.parquet"
            pl.DataFrame(
                {
                    "id1": [1, 3, 5],
                    "id2": [2, 4, 6],
                    "target": [0.55, 0.55, 0.55],
                }
            ).write_parquet(source_path)
            settings = DatasetLabelingSettings(
                source_matches_path=source_path,
                items_path=root / "items.parquet",
                output_path=root / "annotations.parquet",
                score_column="target",
                lower_p=0.54,
                upper_p=0.56,
                sample_size=1,
                seed=42,
                checkpoint_every_batches=10,
                llm=_llm_settings(),
            )
            first = select_unlabeled_pairs(
                settings,
                fingerprint="current",
                existing_annotations=None,
            )
            completed = first.pairs.select("pair_left", "pair_right").with_columns(
                pl.lit("current").alias("labeling_fingerprint")
            )

            second = select_unlabeled_pairs(
                settings,
                fingerprint="current",
                existing_annotations=completed,
            )

            self.assertNotEqual(
                first.pairs.select("pair_left", "pair_right").row(0),
                second.pairs.select("pair_left", "pair_right").row(0),
            )


class _RoundChain:
    def __init__(self) -> None:
        self.concurrency: list[int] = []
        self.call_index = 0

    def batch(self, prompts, *, config, return_exceptions):
        self.concurrency.append(config["max_concurrency"])
        responses = (
            [
                MatchLabel(match_probability=0.9, reason="first"),
                RuntimeError("rate"),
                RuntimeError("rate"),
            ],
            [
                MatchLabel(match_probability=0.2, reason="second"),
                RuntimeError("rate"),
            ],
            [MatchLabel(match_probability=0.8, reason="third")],
        )[self.call_index]
        self.call_index += 1
        self.assertions(prompts, responses, return_exceptions)
        return responses

    @staticmethod
    def assertions(prompts, responses, return_exceptions) -> None:
        if len(prompts) != len(responses) or not return_exceptions:
            raise AssertionError("Unexpected batch invocation")


class LabelingRoundTests(unittest.TestCase):
    def test_retries_only_failures_and_halves_concurrency(self) -> None:
        pairs = pl.DataFrame(
            {
                "_label_row_id": [0, 1, 2],
                "id1": [1, 3, 5],
                "id2": [2, 4, 6],
                "pair_left": [1, 3, 5],
                "pair_right": [2, 4, 6],
                "source_score": [0.55, 0.55, 0.55],
                "category1": ["a", "b", "c"],
                "category2": ["a", "b", "c"],
                "name1": ["one", "three", "five"],
                "name2": ["two", "four", "six"],
                "attributes1": ["{}", "{}", "{}"],
                "attributes2": ["{}", "{}", "{}"],
            }
        )
        chain = _RoundChain()
        checkpoints: list[pl.DataFrame] = []

        result = _run_labeling_rounds(
            chain,
            pairs,
            _llm_settings(),
            fingerprint="fingerprint",
            checkpoint_every_batches=1,
            checkpoint=checkpoints.append,
        )

        self.assertEqual(chain.concurrency, [64, 32, 32])
        self.assertEqual(result.successful_rows, 3)
        self.assertEqual(result.failed_rows, 0)
        self.assertEqual(result.rounds_completed, 3)
        self.assertEqual(sum(frame.height for frame in checkpoints), 3)
        annotations = pl.concat(checkpoints)
        self.assertEqual(
            annotations.get_column("llm_score").to_list(),
            [0.9, 0.2, 0.8],
        )
        self.assertEqual(
            annotations.get_column("labeling_fingerprint").unique().to_list(),
            ["fingerprint"],
        )


class LabelingWorkflowTests(unittest.TestCase):
    def test_repeated_runs_extend_one_annotation_file_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "matches.parquet"
            items_path = root / "items.parquet"
            output_path = root / "annotations" / "annotations.parquet"
            pl.DataFrame(
                {
                    "id1": [1, 3],
                    "id2": [2, 4],
                    "target": [0.55, 0.55],
                }
            ).write_parquet(source_path)
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "name": ["one", "two", "three", "four"],
                    "category": ["a", "a", "b", "b"],
                    "attributes": ["{}", "{}", "{}", "{}"],
                }
            ).write_parquet(items_path)
            base = load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml"
            )
            settings = replace(
                base.labeling,
                source_matches_path=source_path,
                items_path=items_path,
                output_path=output_path,
                sample_size=1,
                checkpoint_every_batches=1,
            )
            config = replace(
                base,
                labeling=settings,
                logging=replace(base.logging, file=None),
            )

            def fake_labeler(
                pairs,
                llm_settings,
                *,
                checkpoint_every_batches,
                checkpoint,
            ):
                del checkpoint_every_batches
                checkpoint(
                    pairs.select(
                        "id1",
                        "id2",
                        "pair_left",
                        "pair_right",
                        "source_score",
                        "category1",
                        "category2",
                    ).with_columns(
                        pl.lit(0.75).alias("llm_score"),
                        pl.lit("test").alias("reason"),
                        pl.lit(llm_settings.model).alias("llm_model"),
                        pl.lit(labeling_fingerprint(llm_settings)).alias(
                            "labeling_fingerprint"
                        ),
                        pl.lit("2026-08-29T00:00:00+00:00").alias(
                            "labeled_at"
                        ),
                    )
                )
                return LlmLabelingResult(pairs.height, 0, 1, ())

            first = label_dataset(config, labeler=fake_labeler)
            second = label_dataset(config, labeler=fake_labeler)
            third = label_dataset(config, labeler=fake_labeler)

            self.assertEqual(first.output_rows, 1)
            self.assertEqual(second.output_rows, 2)
            self.assertEqual(third.selected_rows, 0)
            self.assertEqual(third.output_rows, 2)
            self.assertEqual(pl.read_parquet(output_path).height, 2)
            self.assertEqual(
                list(output_path.parent.glob("*.parquet")),
                [output_path],
            )


if __name__ == "__main__":
    unittest.main()
