import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import torch
from transformers import BertTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

from match.config import load_app_config_file
from match.models.transformer.metrics import compute_pr_auc, positive_probabilities
from match.models.transformer.model import WeightedSequenceTrainer, model_factory
from match.models.transformer.config import SequenceClassifierConfig
from match.models.transformer.head import PoolingHeadConfig
from match.models.transformer.nemotron import (
    NemotronAttentionConfig,
    NemotronAttentionSequenceClassifier,
    NemotronTypedFusionSequenceClassifier,
)
from match.models.transformer.profile import (
    TransformerArtifactContract,
    TransformerRuntimeContract,
)
from match.models.transformer.training import train_sequence_classifier
from match.models.transformer.predictor import TransformerPredictor
from match.models.transformer.onnx_runtime import OnnxRuntimeTransformerExecutor
from match.models.transformer.onnx_export import export_transformer_to_onnx
from match.pair_encoding import PairEncodingCollator, serialize_prompted_pair
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair
from match.pair_features import TypedAttributeOptions, typed_attribute_feature_names
from match.submission import create_submission
from match.workflows.initialize import initialize


def _pair() -> PreparedPair:
    return PreparedPair(
        PreparedCard(1, "Red chair", "Furniture", (("color", "red"),)),
        PreparedCard(2, "Crimson chair", "Furniture", (("color", "crimson"),)),
        1,
        "Furniture",
    )


class _OneLogitModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> SequenceClassifierOutput:
        return SequenceClassifierOutput(logits=input_ids.float())


class _PromptTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def num_special_tokens_to_add(self, *, pair=False):
        return 3 if pair else 2

    def __call__(self, texts, **kwargs):
        self.texts = list(texts)
        max_length = kwargs["max_length"]
        rows = [list(range(1, min(len(text.split()), max_length) + 1)) for text in texts]
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def pad(self, rows, *, padding, max_length=None, return_tensors):
        del padding, return_tensors
        width = max_length or max(len(row["input_ids"]) for row in rows)
        return {
            name: torch.tensor(
                [row[name] + [0] * (width - len(row[name])) for row in rows]
            )
            for name in rows[0]
        }


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        del attention_mask, return_dict, kwargs
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


class NemotronProfileTests(unittest.TestCase):
    @staticmethod
    def _tiny_remote_checkpoint(root: Path) -> Path:
        module_directory = root / "module"
        module_directory.mkdir()
        source = root / f"source_{root.name.removeprefix('tmp')}"
        source.mkdir()
        module_path = module_directory / "tiny_bidirectional.py"
        module_path.write_text(
            '''
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

class TinyBidirectionalConfig(PretrainedConfig):
    model_type = "tiny_bidirectional"
    def __init__(self, hidden_size=8, vocab_size=32, **kwargs):
        kwargs.setdefault("num_labels", 1)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

class TinyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size)])
    def get_input_embeddings(self):
        return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        del attention_mask, return_dict, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return BaseModelOutput(last_hidden_state=hidden)

class TinyBidirectionalForSequenceClassification(PreTrainedModel):
    config_class = TinyBidirectionalConfig
    base_model_prefix = "model"
    def __init__(self, config):
        super().__init__(config)
        self.model = TinyBackbone(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return SequenceClassifierOutput(logits=self.score(pooled))

TinyBidirectionalConfig.register_for_auto_class("AutoConfig")
TinyBidirectionalForSequenceClassification.register_for_auto_class(
    "AutoModelForSequenceClassification"
)
'''.lstrip(),
            encoding="utf-8",
        )
        specification = importlib.util.spec_from_file_location(
            "tiny_bidirectional", module_path
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        config = module.TinyBidirectionalConfig()
        module.TinyBidirectionalForSequenceClassification(config).save_pretrained(
            source
        )
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "question",
            "passage",
            "name",
            "category",
            "chair",
            "phone",
            "red",
            "blue",
        ]
        (source / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
        BertTokenizerFast(vocab_file=str(source / "vocab.txt")).save_pretrained(
            source
        )
        return source

    def test_loads_prompted_native_profile(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            [
                "model_description.transformer.profile=prompted_binary_reranker",
                "model_description.transformer.head.type=native",
                "model_description.transformer.pair_encoding.use_field_tokens=false",
            ],
        )

        transformer = config.model_description.transformer
        self.assertEqual(transformer.profile, "prompted_binary_reranker")
        self.assertEqual(transformer.head.type, "native")

    def test_rejects_cls_for_attention_pooling(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot use cls"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "model_description.transformer.profile=prompted_binary_reranker",
                    "model_description.transformer.head.type=attention_pooling",
                    "model_description.transformer.pair_encoding.use_field_tokens=false",
                ],
            )

    def test_serializes_expected_prompt(self) -> None:
        prompt = serialize_prompted_pair(_pair(), use_field_tokens=False)
        self.assertEqual(
            prompt,
            "question:name: Red chair category: Furniture color: red\n\n"
            "passage:name: Crimson chair category: Furniture color: crimson",
        )

    def test_collator_tokenizes_prompt_as_one_sequence(self) -> None:
        tokenizer = _PromptTokenizer()
        batch = PairEncodingCollator(
            tokenizer,
            32,
            use_field_tokens=False,
            max_attribute_value_tokens=None,
            profile="prompted_binary_reranker",
        )([_pair()])

        self.assertEqual(
            tokenizer.texts,
            [serialize_prompted_pair(_pair(), use_field_tokens=False)],
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 14))
        self.assertEqual(batch["labels"].tolist(), [1])

    def test_attention_wrapper_returns_one_logit(self) -> None:
        config = NemotronAttentionConfig(
            hidden_size=8,
            head_config={
                "poolings": ["mean", "attention"],
                "mlp_hidden_dims": [4],
                "dropout": 0.0,
                "attention_hidden_dim": 4,
                "attention_num_heads": 1,
            },
        )
        model = NemotronAttentionSequenceClassifier(config, _Backbone())
        output = model(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        )
        self.assertEqual(tuple(output.logits.shape), (2, 1))

    def test_attention_artifact_round_trip_preserves_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_remote_checkpoint(root)
            with patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(root / "modules_cache"),
            ):
                model = NemotronAttentionSequenceClassifier.from_backbone_pretrained(
                    str(source),
                    head_config=PoolingHeadConfig(
                        poolings=("mean", "attention"),
                        mlp_hidden_dims=(4,),
                        dropout=0.0,
                        attention_hidden_dim=4,
                    ),
                ).eval()
                inputs = {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                }
                with torch.inference_mode():
                    expected = model(**inputs).logits
                artifact = root / "artifact"
                model.save_pretrained(artifact)
                restored = NemotronAttentionSequenceClassifier.from_artifact(
                    artifact
                ).eval()
                with torch.inference_mode():
                    actual = restored(**inputs).logits

        torch.testing.assert_close(actual, expected)

    def test_typed_fusion_starts_as_native_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_remote_checkpoint(root)
            options = TypedAttributeOptions(enabled=True)
            feature_names = typed_attribute_feature_names(
                enabled_types=options.enabled_types
            )
            with patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(root / "modules_cache"),
            ):
                model = NemotronTypedFusionSequenceClassifier.from_backbone_pretrained(
                    str(source),
                    head_config=PoolingHeadConfig(
                        typed_hidden_dims=(8, 4),
                        dropout=0.0,
                    ),
                    typed_feature_count=len(feature_names),
                ).eval()
                inputs = {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                }
                typed_features = torch.randn(2, len(feature_names))
                with torch.inference_mode():
                    native = model.native_model(**inputs).logits
                    expected = model(
                        **inputs,
                        typed_features=typed_features,
                    ).logits
                torch.testing.assert_close(expected, native)

                TransformerRuntimeContract(
                    output=TransformerArtifactContract.for_training(
                        profile="prompted_binary_reranker",
                        head_type="typed_attribute_fusion",
                        num_logits=1,
                    ),
                    hidden_size=8,
                    max_length=24,
                    use_field_tokens=False,
                    max_attribute_value_chars=256,
                    max_attribute_value_tokens=16,
                    typed_attribute_options=options,
                    typed_feature_names=feature_names,
                    typed_feature_schema_version=1,
                ).apply_encoding_to(model.config)
                artifact = root / "typed_artifact"
                model.save_pretrained(artifact)
                restored = NemotronTypedFusionSequenceClassifier.from_artifact(
                    artifact
                ).eval()
                with torch.inference_mode():
                    actual = restored(
                        **inputs,
                        typed_features=typed_features,
                    ).logits

        torch.testing.assert_close(actual, expected)

    def test_typed_fusion_trains_with_token_cache_and_predicts(self) -> None:
        if not all(
            importlib.util.find_spec(name) is not None
            for name in ("onnx", "onnxruntime")
        ):
            self.skipTest("optional ONNX dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_remote_checkpoint(root)
            artifact = root / "typed_trained"
            pairs = [
                PreparedPair(
                    PreparedCard(1, "red chair", "chair", (("модель", "A1"),)),
                    PreparedCard(2, "red chair", "chair", (("модель", "A1"),)),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(3, "phone", "phone", (("модель", "P1"),)),
                    PreparedCard(4, "chair", "chair", (("модель", "C9"),)),
                    0,
                    "phone",
                ),
                PreparedPair(
                    PreparedCard(5, "chair", "chair", (("цвет", "red"),)),
                    PreparedCard(6, "red chair", "chair", (("цвет", "red"),)),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(7, "phone", "phone", (("вес", "1 kg"),)),
                    PreparedCard(8, "chair", "chair", (("вес", "8 kg"),)),
                    0,
                    "phone",
                ),
            ]
            validation = [
                PreparedPair(
                    PreparedCard(9, "chair", "chair", (("модель", "A1"),)),
                    PreparedCard(10, "chair", "chair", (("модель", "A1"),)),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(11, "phone", "phone", (("цвет", "black"),)),
                    PreparedCard(12, "chair", "chair", (("цвет", "red"),)),
                    0,
                    "chair",
                ),
            ]
            cache = root / "modules_cache"
            with patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(cache),
            ):
                train_sequence_classifier(
                    pairs,
                    validation,
                    SequenceClassifierConfig(
                        str(source),
                        profile="prompted_binary_reranker",
                        head_type="typed_attribute_fusion",
                        head_config=PoolingHeadConfig(
                            typed_hidden_dims=(8, 4),
                            dropout=0.0,
                        ),
                        typed_attribute_options=TypedAttributeOptions(enabled=True),
                        use_field_tokens=False,
                        max_epochs=1,
                        hpo_trials=1,
                        train_batch_size=2,
                        eval_batch_size=2,
                        max_length=24,
                        token_cache_enabled=True,
                        token_cache_directory=root / "token_cache",
                        onnx_export_enabled=True,
                        onnx_precision="float32",
                    ),
                    output_dir=artifact,
                )
                predictor = TransformerPredictor.load(
                    artifact,
                    batch_size=2,
                    pin_memory=False,
                    device="cpu",
                )
                torch_logits = predictor.predict_pair_logits(validation)
                model_config = json.loads(
                    (artifact / "config.json").read_text(encoding="utf-8")
                )
                onnx_metadata = json.loads(
                    (artifact / "onnx" / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                onnx_predictor = TransformerPredictor(
                    predictor.tokenizer,
                    OnnxRuntimeTransformerExecutor(
                        model_directory=artifact,
                        model_config=model_config,
                        provider="cpu",
                        io_binding=False,
                    ),
                    batch_size=2,
                    pin_memory=False,
                )
                onnx_logits = onnx_predictor.predict_pair_logits(validation)

        self.assertEqual(onnx_logits.shape, (2, 1))
        self.assertTrue(np.isfinite(onnx_logits).all())
        self.assertEqual(
            onnx_metadata["classifier_input_names"],
            ["input_ids", "attention_mask", "typed_features"],
        )
        self.assertEqual(onnx_metadata["typed_feature_count"], 52)
        np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-5, rtol=1e-5)

    def test_attention_train_onnx_and_submission_path(self) -> None:
        if not all(
            importlib.util.find_spec(name) is not None
            for name in ("onnx", "onnxruntime")
        ):
            self.skipTest("optional ONNX dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_remote_checkpoint(root)
            artifact = root / "artifact"
            train_pairs = [
                PreparedPair(
                    PreparedCard(1, "red chair", "chair", ()),
                    PreparedCard(2, "red chair", "chair", ()),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(3, "phone", "phone", ()),
                    PreparedCard(4, "blue chair", "chair", ()),
                    0,
                    "phone",
                ),
                PreparedPair(
                    PreparedCard(5, "chair", "chair", ()),
                    PreparedCard(6, "red chair", "chair", ()),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(7, "phone", "phone", ()),
                    PreparedCard(8, "chair", "chair", ()),
                    0,
                    "phone",
                ),
            ]
            validation_pairs = [
                PreparedPair(
                    PreparedCard(9, "red chair", "chair", ()),
                    PreparedCard(10, "chair", "chair", ()),
                    1,
                    "chair",
                ),
                PreparedPair(
                    PreparedCard(11, "phone", "chair", ()),
                    PreparedCard(12, "chair", "chair", ()),
                    0,
                    "chair",
                ),
            ]
            cache = root / "modules_cache"
            with patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(cache),
            ):
                train_sequence_classifier(
                    train_pairs,
                    validation_pairs,
                    SequenceClassifierConfig(
                        str(source),
                        profile="prompted_binary_reranker",
                        head_type="attention_pooling",
                        head_config=PoolingHeadConfig(
                            poolings=("mean", "attention"),
                            mlp_hidden_dims=(4,),
                            dropout=0.0,
                            attention_hidden_dim=4,
                        ),
                        use_field_tokens=False,
                        max_epochs=1,
                        hpo_trials=1,
                        train_batch_size=2,
                        eval_batch_size=2,
                        max_length=24,
                        onnx_export_enabled=True,
                        onnx_precision="float32",
                    ),
                    output_dir=artifact,
                )

                items_path = root / "items.parquet"
                matches_path = root / "matches.parquet"
                output_path = root / "submission.csv"
                pl.DataFrame(
                    {
                        "id": [1, 2, 3, 4],
                        "name": ["red chair", "chair", "phone", "blue chair"],
                        "attributes": ["{}"] * 4,
                        "category": ["chair"] * 4,
                    }
                ).write_parquet(items_path)
                pl.DataFrame({"id1": [1, 3], "id2": [2, 4]}).write_parquet(
                    matches_path
                )
                solution_path = root / "solution.json"
                solution_path.write_text(
                    json.dumps(
                        {
                            "predictor": "transformer",
                            "model_directory": "artifact",
                            "backend": "onnxruntime",
                            "batch_size": 2,
                            "pin_memory": False,
                            "onnxruntime": {
                                "provider": "cpu",
                                "io_binding": False,
                                "graph_optimization": "all",
                                "fallback_to_pytorch": False,
                            },
                            "onnx_artifacts": {
                                "enabled": True,
                                "classifier_path": "onnx/classifier.onnx",
                                "encoder_path": "onnx/encoder.onnx",
                                "precision": "float32",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                result = create_submission(
                    items_path,
                    matches_path,
                    output_path,
                    solution_path=solution_path,
                )

        self.assertEqual(result.height, 2)
        probabilities = result.get_column("predict").to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_native_initialize_onnx_and_submission_path(self) -> None:
        if not all(
            importlib.util.find_spec(name) is not None
            for name in ("onnx", "onnxruntime")
        ):
            self.skipTest("optional ONNX dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_remote_checkpoint(root)
            artifact = root / "native_artifact"
            base = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")
            transformer = replace(
                base.model_description.transformer,
                profile="prompted_binary_reranker",
                pretrained_model_path=str(source),
                artifact_dir=artifact,
                pair_encoding=replace(
                    base.model_description.transformer.pair_encoding,
                    use_field_tokens=False,
                    max_length=24,
                ),
                head=replace(
                    base.model_description.transformer.head,
                    type="native",
                ),
                export=replace(
                    base.model_description.transformer.export,
                    onnx=replace(
                        base.model_description.transformer.export.onnx,
                        enabled=True,
                        precision="float32",
                    ),
                ),
            )
            inference_transformer = replace(
                base.inference.transformer,
                backend="onnxruntime",
                dtype="float32",
                num_workers=0,
                pin_memory=False,
                length_bucketing=replace(
                    base.inference.transformer.length_bucketing,
                    enabled=False,
                ),
                onnxruntime=replace(
                    base.inference.transformer.onnxruntime,
                    provider="cpu",
                    io_binding=False,
                    fallback_to_pytorch=False,
                ),
            )
            config = replace(
                base,
                training=replace(
                    base.training,
                    model="transformer",
                    resolved_config_path=root / "resolved.yaml",
                    solution_path=root / "solution.json",
                ),
                inference=replace(
                    base.inference,
                    model="transformer",
                    solution_path=root / "solution.json",
                    transformer=inference_transformer,
                ),
                model_description=replace(
                    base.model_description,
                    transformer=transformer,
                ),
            )
            cache = root / "modules_cache"
            with patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(cache),
            ):
                initialized = initialize(config)
                export_transformer_to_onnx(artifact, precision="float32")
                solution = json.loads(
                    initialized.solution_path.read_text(encoding="utf-8")
                )
                solution["features"]["normalization"]["enabled"] = False
                initialized.solution_path.write_text(
                    json.dumps(solution), encoding="utf-8"
                )

                items_path = root / "native_items.parquet"
                matches_path = root / "native_matches.parquet"
                output_path = root / "native_submission.csv"
                pl.DataFrame(
                    {
                        "id": [1, 2],
                        "name": ["red chair", "chair"],
                        "attributes": ["{}", "{}"],
                        "category": ["chair", "chair"],
                    }
                ).write_parquet(items_path)
                pl.DataFrame({"id1": [1], "id2": [2]}).write_parquet(
                    matches_path
                )
                result = create_submission(
                    items_path,
                    matches_path,
                    output_path,
                    solution_path=initialized.solution_path,
                )

        self.assertEqual(result.height, 1)
        probability = result.get_column("predict").item()
        self.assertTrue(0.0 <= probability <= 1.0)

    def test_one_logit_probability_and_metric_use_sigmoid(self) -> None:
        logits = np.asarray([[-2.0], [2.0]], dtype=np.float32)
        probabilities = positive_probabilities(logits)
        np.testing.assert_allclose(
            probabilities,
            [0.11920292, 0.88079708],
            rtol=1e-6,
        )
        self.assertEqual(compute_pr_auc((logits, np.asarray([0, 1]))), {"pr_auc": 1.0})

    def test_weighted_trainer_uses_binary_cross_entropy(self) -> None:
        trainer = SimpleNamespace(class_weights=torch.tensor([1.0, 1.0]))
        inputs = {
            "input_ids": torch.tensor([[-2.0], [2.0]]),
            "labels": torch.tensor([0, 1]),
            "sample_weights": torch.ones(2),
        }
        loss = WeightedSequenceTrainer.compute_loss(
            trainer,
            _OneLogitModel(),
            inputs,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-2.0, 2.0]),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(loss, expected)

    def test_native_factory_preserves_one_logit_head(self) -> None:
        native = Mock()
        native.config.num_labels = 1
        tokenizer = Mock()
        with patch(
            "match.models.transformer.construction."
            "AutoModelForSequenceClassification.from_pretrained",
            return_value=native,
        ) as load:
            restored = model_factory(
                "nemotron",
                tokenizer,
                use_field_tokens=False,
                profile="prompted_binary_reranker",
                head_type="native",
                attention_implementation="sdpa",
            )()

        self.assertIs(restored, native)
        load.assert_called_once_with(
            "nemotron",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.assertEqual(native.config.match_num_logits, 1)
        self.assertEqual(native.config.match_probability_transform, "sigmoid")


if __name__ == "__main__":
    unittest.main()
