import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from match.models.transformer.factory import _load_components


class TransformerFactoryContractTests(unittest.TestCase):
    def test_prompted_artifact_loads_tokenizer_with_remote_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 16,
                        "num_labels": 1,
                        "match_profile": "prompted_binary_reranker",
                        "match_head_type": "native",
                        "match_num_logits": 1,
                        "match_probability_transform": "sigmoid",
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained"
            ) as load_tokenizer:
                tokenizer, config = _load_components(model_directory)

        self.assertIs(tokenizer, load_tokenizer.return_value)
        self.assertEqual(config["match_num_logits"], 1)
        load_tokenizer.assert_called_once_with(
            model_directory,
            use_fast=True,
            trust_remote_code=True,
        )


if __name__ == "__main__":
    unittest.main()
