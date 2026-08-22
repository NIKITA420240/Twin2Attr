import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig

from match.features.ner.config import WordNerModelConfig
from match.features.ner.model import WordNERModel


class WordNerModelTests(unittest.TestCase):
    def test_produces_one_prediction_per_aligned_word(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            BertConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
            ).save_pretrained(model_dir)
            config = WordNerModelConfig(
                model_name="unused-local-model",
                num_attention_heads=2,
                pos_dim=4,
                attention_hidden=8,
                sequence_heads=2,
            )
            model = WordNERModel(config, architecture_source=model_dir).eval()
            input_ids = torch.tensor([[1, 4, 5, 6, 2]])
            attention_mask = torch.ones_like(input_ids)
            word_ids = torch.tensor([[-1, 0, 0, 1, -1]])
            positions = torch.tensor([[0, 0, 1, 0, 0]])

            contextual, semantic, lengths = model.encode_words(
                input_ids,
                attention_mask,
                word_ids,
                positions,
            )
            output = model(
                input_ids,
                attention_mask,
                word_ids,
                positions,
            )

        self.assertEqual(contextual.shape, (1, 2, 16))
        self.assertEqual(semantic.shape, (1, 2, 16))
        self.assertEqual(lengths.tolist(), [2])
        self.assertEqual(output.logits.shape, (1, 2, 9))


if __name__ == "__main__":
    unittest.main()
