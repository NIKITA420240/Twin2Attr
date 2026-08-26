import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import polars as pl
from transformers import BertConfig, BertModel, BertTokenizerFast

from match.analysis.attribute_importance import (
    analyze_transformer_attribute_importance,
)
from match.config import load_app_config_file
from match.models.transformer.head import PoolingHeadConfig
from match.models.transformer.model import model_factory
from match.pair_encoding import encode_prepared_pair_with_attributes
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair


def _pair() -> PreparedPair:
    left = PreparedCard(
        1,
        "phone",
        "phones",
        (("brand", "acme"), ("color", "black")),
    )
    right = PreparedCard(
        2,
        "phone",
        "phones",
        (("brand", "acme"), ("memory", "large")),
    )
    return PreparedPair(left, right, 1, "phones")


class AttributeImportanceAnalysisTests(unittest.TestCase):
    def _checkpoint(self, root: Path):
        source = root / "source"
        trained = root / "trained"
        source.mkdir()
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "name",
            "category",
            "phone",
            "phones",
            "brand",
            "color",
            "memory",
            "acme",
            "black",
            "large",
        ]
        (source / "vocab.txt").write_text(
            "\n".join(vocabulary),
            encoding="utf-8",
        )
        tokenizer = BertTokenizerFast(
            vocab_file=str(source / "vocab.txt"),
        )
        tokenizer.save_pretrained(source)
        BertModel(
            BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=64,
            )
        ).save_pretrained(source)
        model = model_factory(
            str(source),
            tokenizer,
            use_field_tokens=True,
            head_type="pooling",
            head_config=PoolingHeadConfig(
                poolings=("cls", "attention"),
                mlp_hidden_dims=(),
                dropout=0.0,
                attention_hidden_dim=4,
            ),
        )()
        model.config.match_max_length = 48
        model.config.match_use_field_tokens = True
        model.config.match_max_attribute_value_tokens = 8
        model.save_pretrained(trained)
        tokenizer.save_pretrained(trained)
        return trained, tokenizer

    def test_encoding_preserves_attribute_token_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, tokenizer = self._checkpoint(Path(directory))

            encoded = encode_prepared_pair_with_attributes(
                tokenizer,
                _pair(),
                max_length=48,
                max_attribute_value_tokens=8,
            )

        self.assertEqual(
            len(encoded.token_attributes),
            len(encoded.inputs["input_ids"]),
        )
        self.assertEqual(encoded.present_attribute_sections["brand"], 2)
        self.assertEqual(encoded.included_attribute_sections["brand"], 2)
        self.assertIn("color", encoded.token_attributes)
        self.assertIn("memory", encoded.token_attributes)

    def test_analyzer_writes_parquet_and_metadata_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _ = self._checkpoint(root)
            output = root / "analysis"
            base = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")
            config = replace(
                base,
                runtime=replace(base.runtime, device="cpu"),
                model_description=replace(
                    base.model_description,
                    transformer=replace(
                        base.model_description.transformer,
                        artifact_dir=checkpoint,
                        batch_size=2,
                    ),
                ),
                analysis_models=replace(
                    base.analysis_models,
                    attribute_importance=replace(
                        base.analysis_models.attribute_importance,
                        output_dir=output,
                        min_occurrences=2,
                    ),
                ),
            )

            result = analyze_transformer_attribute_importance(
                config,
                [_pair(), _pair()],
            )
            frame = pl.read_parquet(result.output_path)
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(result.sample_rows, 2)
        self.assertIn("brand", frame.get_column("attribute").to_list())
        self.assertEqual(set(frame.get_column("scope")), {"global", "category"})
        self.assertTrue((frame.get_column("importance") > 0).all())
        self.assertEqual(metadata["max_length"], 48)
        self.assertEqual(metadata["model"], "transformer")


if __name__ == "__main__":
    unittest.main()
