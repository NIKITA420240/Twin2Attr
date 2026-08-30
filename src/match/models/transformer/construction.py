"""Construction policies for legacy and prompted Transformer classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...pair_encoding import (
    add_pair_special_tokens,
    initialize_pair_special_token_embeddings,
    pair_special_token_ids,
)
from .head import (
    GatedResidualFusionSequenceClassifier,
    HybridSequenceClassifier,
    PoolingHeadConfig,
    PoolingSequenceClassifier,
)
from .nemotron import (
    NemotronAttentionSequenceClassifier,
    NemotronGatedResidualFusionSequenceClassifier,
    NemotronTypedFusionSequenceClassifier,
)
from .optimizer import (
    freeze_backbone_except_last_layers,
    restrict_word_embedding_updates,
)
from .profile import (
    SEQUENCE_CLASSIFIER_PROFILE,
    TransformerArtifactContract,
    is_prompted_profile,
    validate_profile_head,
)


def _build_classifier(
    model_path: str,
    *,
    profile: str,
    head_type: str,
    head_config: PoolingHeadConfig | None,
    typed_feature_count: int,
    attention_implementation: str,
) -> PreTrainedModel:
    attention_kwargs = (
        {}
        if attention_implementation == "auto"
        else {"attn_implementation": attention_implementation}
    )
    if is_prompted_profile(profile) and head_type == "native":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            **attention_kwargs,
        )
        if int(model.config.num_labels) != 1:
            raise ValueError(
                "prompted_binary_reranker native model must expose one logit"
            )
        return model
    if is_prompted_profile(profile) and head_type == "attention_pooling":
        return NemotronAttentionSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config
            or PoolingHeadConfig(poolings=("mean", "attention")),
            attention_implementation=attention_implementation,
        )
    if is_prompted_profile(profile) and head_type == "typed_attribute_fusion":
        return NemotronTypedFusionSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config or PoolingHeadConfig(),
            typed_feature_count=typed_feature_count,
            attention_implementation=attention_implementation,
        )
    if is_prompted_profile(profile) and head_type == "gated_residual_fusion":
        return NemotronGatedResidualFusionSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config
            or PoolingHeadConfig(poolings=("mean", "attention")),
            typed_feature_count=typed_feature_count,
            attention_implementation=attention_implementation,
        )
    if head_type == "gated_residual_fusion":
        return GatedResidualFusionSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config
            or PoolingHeadConfig(poolings=("mean", "attention")),
            typed_feature_count=typed_feature_count,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            attention_implementation=attention_implementation,
        )
    if head_type == "hybrid":
        return HybridSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config
            or PoolingHeadConfig(poolings=("attention",)),
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            attention_implementation=attention_implementation,
        )
    if head_type == "pooling":
        return PoolingSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config or PoolingHeadConfig(),
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            attention_implementation=attention_implementation,
        )
    if head_type == "default":
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            ignore_mismatched_sizes=True,
            **attention_kwargs,
        )
    raise AssertionError("unreachable Transformer head type")


def model_factory(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
    special_token_initialization_enabled: bool = False,
    special_token_key_seed_texts: Sequence[str] = (),
    special_token_value_seed_texts: Sequence[str] = (),
    train_new_token_embeddings_only: bool = False,
    train_last_n_layers: int | None = None,
    head_type: str = "default",
    head_config: PoolingHeadConfig | None = None,
    typed_feature_count: int = 0,
    profile: str = SEQUENCE_CLASSIFIER_PROFILE,
    attention_implementation: str = "auto",
):
    """Return the callback expected by Hugging Face ``Trainer.model_init``."""
    profile, head_type = validate_profile_head(profile, head_type)

    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        model = _build_classifier(
            model_path,
            profile=profile,
            head_type=head_type,
            head_config=head_config,
            typed_feature_count=typed_feature_count,
            attention_implementation=attention_implementation,
        )
        contract = TransformerArtifactContract.for_training(
            profile=profile,
            head_type=head_type,
            num_logits=int(model.config.num_labels),
        )
        # Legacy sequence classifiers historically had no match_* metadata at
        # initialization time. Keep that observable behavior; the training
        # boundary persists a complete contract before saving the artifact.
        if contract.uses_prompted_pairs:
            contract.apply_to(model.config)
            model.config.use_cache = False
            model.base_model.config.use_cache = False
        if use_field_tokens:
            add_pair_special_tokens(tokenizer, model)
            if special_token_initialization_enabled:
                initialize_pair_special_token_embeddings(
                    tokenizer,
                    model,
                    key_seed_texts=special_token_key_seed_texts,
                    value_seed_texts=special_token_value_seed_texts,
                )
            if train_new_token_embeddings_only:
                restrict_word_embedding_updates(
                    model,
                    pair_special_token_ids(tokenizer),
                )
        if train_last_n_layers is not None:
            freeze_backbone_except_last_layers(
                model,
                train_last_n_layers,
                train_input_word_embeddings=train_new_token_embeddings_only,
            )
        return model

    return initialize_model


__all__ = ["model_factory"]
