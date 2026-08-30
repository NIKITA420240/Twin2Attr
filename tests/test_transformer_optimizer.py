import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from match.models.transformer.optimizer import (
    LearningRateMultipliers,
    build_transformer_optimizer,
    freeze_backbone_except_last_layers,
    restrict_word_embedding_updates,
)


class FakeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(16, 8)
        self.LayerNorm = nn.LayerNorm(8)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = FakeEmbeddings()
        self.encoder = FakeEncoder()
        self.pooler = nn.Linear(8, 8)


class FakeDistilTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])


class FakeDistilBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = FakeEmbeddings()
        self.transformer = FakeDistilTransformer()


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        self.head = nn.Linear(8, 2)

    @property
    def base_model(self) -> nn.Module:
        return self.backbone

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.embeddings.word_embeddings


class FakeDistilModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeDistilBackbone()


class TransformerOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        self.optimizer = build_transformer_optimizer(
            self.model,
            backbone_lr=1e-5,
            multipliers=LearningRateMultipliers(
                embeddings=0.5,
                head=3.0,
                layerwise_decay=0.8,
            ),
            weight_decay=0.01,
        )

    def test_assigns_every_trainable_parameter_once(self) -> None:
        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(optimizer_parameters), len(trainable_parameters))
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in trainable_parameters},
        )

    def test_uses_expected_learning_rates(self) -> None:
        rates = {
            group["group_name"]: group["lr"]
            for group in self.optimizer.param_groups
        }
        self.assertAlmostEqual(rates["embeddings.decay"], 5e-6)
        self.assertAlmostEqual(rates["backbone.layer.0.decay"], 6.4e-6)
        self.assertAlmostEqual(rates["backbone.layer.1.decay"], 8e-6)
        self.assertAlmostEqual(rates["backbone.layer.2.decay"], 1e-5)
        self.assertAlmostEqual(rates["backbone.other.decay"], 1e-5)
        self.assertAlmostEqual(rates["head.decay"], 3e-5)

    def test_excludes_bias_and_layer_norm_from_weight_decay(self) -> None:
        no_decay_groups = [
            group
            for group in self.optimizer.param_groups
            if group["group_name"].endswith(".no_decay")
        ]
        self.assertTrue(no_decay_groups)
        self.assertTrue(
            all(group["weight_decay"] == 0.0 for group in no_decay_groups)
        )

    def test_scales_ratios_from_configured_learning_rates(self) -> None:
        multipliers = LearningRateMultipliers.from_learning_rates(
            embeddings_lr=5e-6,
            backbone_lr=1e-5,
            head_lr=3e-5,
            layerwise_decay=1.0,
        )
        self.assertEqual(multipliers.embeddings, 0.5)
        self.assertEqual(multipliers.head, 3.0)

    def test_supports_distilbert_transformer_layers(self) -> None:
        optimizer = build_transformer_optimizer(
            FakeDistilModel(),
            backbone_lr=1e-5,
            multipliers=LearningRateMultipliers(layerwise_decay=0.8),
            weight_decay=0.01,
        )
        group_names = {group["group_name"] for group in optimizer.param_groups}
        self.assertIn("backbone.layer.0.decay", group_names)
        self.assertIn("backbone.layer.1.decay", group_names)

    def test_fused_optimizer_falls_back_on_cpu(self) -> None:
        optimizer = build_transformer_optimizer(
            FakeModel(),
            backbone_lr=1e-5,
            multipliers=LearningRateMultipliers(),
            weight_decay=0.01,
            fused=True,
        )

        self.assertIsNot(optimizer.defaults.get("fused"), True)

    def test_enables_fused_optimizer_for_cuda_parameters(self) -> None:
        model = FakeModel()
        adamw_signature = inspect.signature(torch.optim.AdamW)
        with (
            patch.object(
                torch.nn.Parameter,
                "device",
                new_callable=lambda: property(
                    lambda _self: torch.device("cuda")
                ),
            ),
            patch("match.models.transformer.optimizer.torch.optim.AdamW") as adamw,
            patch(
                "match.models.transformer.optimizer.inspect.signature",
                return_value=adamw_signature,
            ),
        ):
            build_transformer_optimizer(
                model,
                backbone_lr=1e-5,
                multipliers=LearningRateMultipliers(),
                weight_decay=0.01,
                fused=True,
            )

        self.assertTrue(adamw.call_args.kwargs["fused"])

    def test_updates_only_selected_word_embedding_rows(self) -> None:
        model = FakeModel()
        restrict_word_embedding_updates(model, [14, 15])
        optimizer = build_transformer_optimizer(
            model,
            backbone_lr=1e-5,
            multipliers=LearningRateMultipliers(embeddings=0.5),
            weight_decay=0.01,
        )
        embeddings = model.get_input_embeddings().weight
        before = embeddings.detach().clone()

        embeddings.sum().backward()
        optimizer.step()

        torch.testing.assert_close(embeddings[:14], before[:14], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(embeddings[14:], before[14:]))
        restricted_group = next(
            group
            for group in optimizer.param_groups
            if group["group_name"] == "new_token_embeddings.no_decay"
        )
        self.assertEqual(restricted_group["lr"], 5e-6)
        self.assertEqual(restricted_group["weight_decay"], 0.0)

    def test_trains_only_last_backbone_layers_and_requested_word_embeddings(self) -> None:
        model = FakeModel()

        trainable_layer_ids = freeze_backbone_except_last_layers(
            model,
            2,
            train_input_word_embeddings=True,
        )

        self.assertEqual(trainable_layer_ids, (1, 2))
        self.assertTrue(model.get_input_embeddings().weight.requires_grad)
        self.assertFalse(model.backbone.embeddings.LayerNorm.weight.requires_grad)
        self.assertFalse(model.backbone.encoder.layer[0].weight.requires_grad)
        self.assertTrue(model.backbone.encoder.layer[1].weight.requires_grad)
        self.assertTrue(model.backbone.encoder.layer[2].weight.requires_grad)
        self.assertFalse(model.backbone.pooler.weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)

    def test_freezes_embeddings_by_default_when_training_top_layers(self) -> None:
        model = FakeModel()

        trainable_layer_ids = freeze_backbone_except_last_layers(model, 2)

        self.assertEqual(trainable_layer_ids, (1, 2))
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)
        self.assertFalse(model.backbone.embeddings.LayerNorm.weight.requires_grad)
        self.assertFalse(model.backbone.encoder.layer[0].weight.requires_grad)
        self.assertTrue(model.backbone.encoder.layer[1].weight.requires_grad)
        self.assertTrue(model.backbone.encoder.layer[2].weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
