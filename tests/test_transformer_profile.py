import unittest
from types import SimpleNamespace

import numpy as np

from match.pair_features import TypedAttributeOptions, typed_attribute_feature_names
from match.models.transformer.profile import (
    PROMPTED_BINARY_RERANKER_PROFILE,
    SEQUENCE_CLASSIFIER_PROFILE,
    TransformerArtifactContract,
    TransformerRuntimeContract,
    validate_profile_head,
)


class TransformerArtifactContractTests(unittest.TestCase):
    def test_legacy_config_defaults_to_two_logit_softmax(self) -> None:
        contract = TransformerArtifactContract.from_config(
            SimpleNamespace(num_labels=2)
        )

        self.assertEqual(contract.profile, SEQUENCE_CLASSIFIER_PROFILE)
        self.assertEqual(contract.num_logits, 2)
        self.assertEqual(contract.probability_transform, "softmax")
        np.testing.assert_allclose(
            contract.positive_probabilities(np.asarray([[0.0, 1.0]])),
            [0.73105858],
        )

    def test_prompted_contract_uses_stable_sigmoid_and_mean_pooling(self) -> None:
        contract = TransformerArtifactContract.for_training(
            profile=PROMPTED_BINARY_RERANKER_PROFILE,
            head_type="native",
            num_logits=1,
        )

        probabilities = contract.positive_probabilities(
            np.asarray([[-1000.0], [0.0], [1000.0]])
        )
        np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(
            contract.logit_margin(np.asarray([[-2.0], [3.0]])),
            [-2.0, 3.0],
        )
        self.assertEqual(contract.encoder_pooling, "mean")

    def test_prompted_contract_rejects_two_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one logit"):
            TransformerArtifactContract.for_training(
                profile=PROMPTED_BINARY_RERANKER_PROFILE,
                head_type="native",
                num_logits=2,
            )

    def test_heads_are_validated_by_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid for profile"):
            validate_profile_head(SEQUENCE_CLASSIFIER_PROFILE, "native")

    def test_runtime_contract_reads_nested_backbone_and_legacy_defaults(self) -> None:
        contract = TransformerRuntimeContract.from_config(
            {
                "num_labels": 2,
                "backbone_config": {"hidden_size": 24},
                "match_max_length": 128,
            }
        )

        self.assertEqual(contract.hidden_size, 24)
        self.assertEqual(contract.max_length, 128)
        self.assertTrue(contract.use_field_tokens)
        self.assertEqual(contract.max_attribute_value_tokens, 32)

    def test_typed_runtime_contract_round_trips_persisted_schema(self) -> None:
        options = TypedAttributeOptions(enabled=True)
        names = typed_attribute_feature_names(enabled_types=options.enabled_types)
        config = SimpleNamespace()
        contract = TransformerRuntimeContract(
            output=TransformerArtifactContract.for_training(
                profile=SEQUENCE_CLASSIFIER_PROFILE,
                head_type="typed_attribute_fusion",
                num_logits=2,
            ),
            hidden_size=24,
            max_length=128,
            use_field_tokens=True,
            max_attribute_value_chars=None,
            max_attribute_value_tokens=32,
            typed_attribute_options=options,
            typed_feature_names=names,
            typed_feature_schema_version=1,
        )

        contract.output.apply_to(config)
        contract.apply_encoding_to(config)
        config.hidden_size = contract.hidden_size

        self.assertEqual(TransformerRuntimeContract.from_config(config), contract)

    def test_typed_runtime_contract_rejects_unknown_schema_version(self) -> None:
        options = TypedAttributeOptions(enabled=True)
        with self.assertRaisesRegex(ValueError, "schema version 1"):
            TransformerRuntimeContract(
                output=TransformerArtifactContract.for_training(
                    profile=SEQUENCE_CLASSIFIER_PROFILE,
                    head_type="typed_attribute_fusion",
                    num_logits=2,
                ),
                hidden_size=24,
                max_length=128,
                use_field_tokens=True,
                max_attribute_value_chars=None,
                max_attribute_value_tokens=32,
                typed_attribute_options=options,
                typed_feature_names=typed_attribute_feature_names(
                    enabled_types=options.enabled_types
                ),
                typed_feature_schema_version=2,
            )


if __name__ == "__main__":
    unittest.main()
