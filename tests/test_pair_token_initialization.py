import unittest

import torch
from torch import nn

from match.pair_encoding import initialize_pair_special_token_embeddings


class FakeTokenizer:
    def __init__(self) -> None:
        self._ids = {
            "[KEY]": (8,),
            "[VAL]": (9,),
            "key": (1,),
            "attribute": (2,),
            "value": (3,),
            "значение": (4, 5),
            "unknown": (0,),
        }
        self.all_special_ids = (0, 8, 9)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.assertFalse(add_special_tokens)
        return list(self._ids[text])

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return next(
            token
            for token, ids in self._ids.items()
            if ids == (token_id,)
        )

    def assertFalse(self, value: bool) -> None:
        if value:
            raise AssertionError("special tokens must not be added to seed texts")


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(10, 2)
        with torch.no_grad():
            self.embeddings.weight.copy_(
                torch.arange(20, dtype=torch.float32).reshape(10, 2)
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings


class PairTokenInitializationTests(unittest.TestCase):
    def test_initializes_each_field_token_from_its_seed_subwords(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        before = model.embeddings.weight.detach().clone()

        sources = initialize_pair_special_token_embeddings(
            tokenizer,
            model,
            key_seed_texts=("key", "attribute"),
            value_seed_texts=("value", "значение"),
        )

        self.assertEqual(sources, {"[KEY]": (1, 2), "[VAL]": (3, 4, 5)})
        torch.testing.assert_close(
            model.embeddings.weight[8], before[[1, 2]].mean(dim=0)
        )
        torch.testing.assert_close(
            model.embeddings.weight[9], before[[3, 4, 5]].mean(dim=0)
        )

    def test_rejects_seed_texts_without_non_special_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "no usable non-special tokens"):
            initialize_pair_special_token_embeddings(
                FakeTokenizer(),
                FakeModel(),
                key_seed_texts=("unknown",),
                value_seed_texts=("value",),
            )


if __name__ == "__main__":
    unittest.main()
