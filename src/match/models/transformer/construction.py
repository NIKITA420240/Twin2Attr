"""Construction policies for legacy and prompted Transformer classifiers."""

from __future__ import annotations

from typing import Any

from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...pair_encoding import add_pair_special_tokens, pair_special_token_ids
from .head import PoolingHeadConfig, PoolingSequenceClassifier
from .nemotron import NemotronAttentionSequenceClassifier
from .optimizer import (
    freeze_backbone_except_last_layers,
    restrict_word_embedding_updates,
)
from .profile import (
    SEQUENCE_CLASSIFIER_PROFILE,
    TransformerArtifactContract,
    is_nemotron_profile,
    is_qwen3_reranker_profile,
    validate_profile_head,
)


def _build_classifier(
    model_path: str,
    *,
    profile: str,
    head_type: str,
    head_config: PoolingHeadConfig | None,
) -> PreTrainedModel:
    if is_qwen3_reranker_profile(profile):
        raise ValueError(
            "qwen3_reranker model construction requires the yes/no scoring wrapper"
        )
    if is_nemotron_profile(profile) and head_type == "native":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if int(model.config.num_labels) != 1:
            raise ValueError(
                "prompted_binary_reranker native model must expose one logit"
            )
        return model
    if is_nemotron_profile(profile) and head_type == "attention_pooling":
        return NemotronAttentionSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config
            or PoolingHeadConfig(poolings=("mean", "attention")),
        )
    if head_type == "pooling":
        return PoolingSequenceClassifier.from_backbone_pretrained(
            model_path,
            head_config=head_config or PoolingHeadConfig(),
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
        )
    if head_type == "default":
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            ignore_mismatched_sizes=True,
        )
    raise AssertionError("unreachable Transformer head type")


def model_factory(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
    train_new_token_embeddings_only: bool = False,
    train_last_n_layers: int | None = None,
    head_type: str = "default",
    head_config: PoolingHeadConfig | None = None,
    profile: str = SEQUENCE_CLASSIFIER_PROFILE,
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
