"""Create an inference-ready Transformer artifact without fitting it."""

from __future__ import annotations

import json
from dataclasses import replace

from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from ..config import AppConfig, save_app_config
from ..models.artifacts import TrainingArtifacts, save_solution_manifest
from ..models.transformer.construction import model_factory
from ..models.transformer.head import PoolingHeadConfig
from ..models.transformer.profile import (
    TransformerArtifactContract,
    TransformerRuntimeContract,
    is_qwen3_reranker_profile,
    is_prompted_profile,
    requires_trust_remote_code,
)
from ..models.transformer.qwen3 import qwen3_score_token_ids
from ._common import workflow_logging


DEFAULT_INITIALIZED_MAX_LENGTH = 128


def _initialization_max_length(config: AppConfig) -> int:
    encoding = config.model_description.transformer.pair_encoding
    configured = encoding.max_length
    if configured is not None:
        return configured
    return min(DEFAULT_INITIALIZED_MAX_LENGTH, encoding.hard_cap)


def initialize(config: AppConfig) -> TrainingArtifacts:
    """Initialize and save the configured Transformer with an untrained head."""
    if config.training.model != "transformer":
        raise ValueError(
            "initialize supports only training.model=transformer; "
            f"got {config.training.model!r}"
        )

    output_path = config.model_description.transformer.artifact_dir.resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Transformer artifact directory is not empty: {output_path}. "
            "Move it away or choose another "
            "model_description.transformer.artifact_dir."
        )

    parameters = config.model_description.transformer
    encoding = parameters.pair_encoding
    max_length = _initialization_max_length(config)
    head_config = PoolingHeadConfig(
        poolings=parameters.head.poolings,
        mlp_hidden_dims=parameters.head.mlp_hidden_dims,
        dropout=parameters.head.dropout,
        attention_hidden_dim=parameters.head.attention_hidden_dim,
        attention_num_heads=parameters.head.attention_num_heads,
    )

    with workflow_logging(config, workflow_name="initialize"):
        if is_qwen3_reranker_profile(parameters.profile):
            logger.info("Creating lightweight Qwen3 zero-shot reranker artifact")
            tokenizer = AutoTokenizer.from_pretrained(
                parameters.pretrained_model_path,
                trust_remote_code=False,
                fix_mistral_regex=True,
            )
            tokenizer.padding_side = "left"
            no_token_id, yes_token_id = qwen3_score_token_ids(tokenizer)
            model_config = AutoConfig.from_pretrained(
                parameters.pretrained_model_path,
                trust_remote_code=False,
            )
            output_contract = TransformerArtifactContract.for_training(
                profile=parameters.profile,
                head_type=parameters.head.type,
                num_logits=1,
            )
            output_contract.apply_to(model_config)
            TransformerRuntimeContract(
                output=output_contract,
                hidden_size=int(model_config.hidden_size),
                max_length=max_length,
                use_field_tokens=encoding.use_field_tokens,
                max_attribute_value_chars=encoding.max_attribute_value_chars,
                max_attribute_value_tokens=encoding.max_attribute_value_tokens,
            ).apply_encoding_to(model_config)
            model_config.use_cache = False
            model_config.match_initialized_only = True
            model_config.match_source_model_path = parameters.pretrained_model_path
            model_config.match_no_token_id = no_token_id
            model_config.match_yes_token_id = yes_token_id

            output_path.mkdir(parents=True, exist_ok=True)
            model_config.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            metadata = {
                "trained": False,
                "warning": "Pretrained Qwen3 yes/no scoring preserved.",
                "source_model": parameters.pretrained_model_path,
                "profile": parameters.profile,
                "head_type": parameters.head.type,
                "max_length": max_length,
                "use_field_tokens": encoding.use_field_tokens,
                "max_attribute_value_chars": encoding.max_attribute_value_chars,
                "max_attribute_value_tokens": encoding.max_attribute_value_tokens,
                "no_token_id": no_token_id,
                "yes_token_id": yes_token_id,
            }
            (output_path / "initialization_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=output_path,
            )
            save_app_config(config, config.training.resolved_config_path)
            solution_path = save_solution_manifest(config, artifacts)
            logger.info("Initialized Qwen3 artifact saved to {!s}", output_path)
            return replace(
                artifacts,
                resolved_config_path=config.training.resolved_config_path,
                solution_path=solution_path,
            )
        if (
            is_prompted_profile(parameters.profile)
            and parameters.head.type == "attention_pooling"
        ):
            raise ValueError(
                "attention_pooling is randomly initialized and requires train; "
                "initialize supports only the native Nemotron head"
            )
        if is_prompted_profile(parameters.profile):
            logger.info("Preserving the pretrained native Nemotron score head")
        else:
            logger.warning(
                "Initializing Transformer artifact without training; classifier "
                "predictions will be random"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            parameters.pretrained_model_path,
            trust_remote_code=requires_trust_remote_code(parameters.profile),
        )
        model = model_factory(
            parameters.pretrained_model_path,
            tokenizer,
            use_field_tokens=encoding.use_field_tokens,
            head_type=parameters.head.type,
            head_config=head_config,
            profile=parameters.profile,
        )()
        runtime_contract = TransformerRuntimeContract.from_config(model.config)
        TransformerRuntimeContract(
            output=runtime_contract.output,
            hidden_size=runtime_contract.hidden_size,
            max_length=max_length,
            use_field_tokens=encoding.use_field_tokens,
            max_attribute_value_chars=encoding.max_attribute_value_chars,
            max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        ).apply_encoding_to(model.config)
        model.config.match_initialized_only = True

        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        metadata = {
            "trained": False,
            "warning": (
                "Pretrained native Nemotron head preserved for zero-shot inference."
                if is_prompted_profile(parameters.profile)
                else "Classifier head is initialized but has not been trained."
            ),
            "source_model": parameters.pretrained_model_path,
            "profile": parameters.profile,
            "head_type": parameters.head.type,
            "head_config": head_config.to_dict(),
            "max_length": max_length,
            "use_field_tokens": encoding.use_field_tokens,
            "max_attribute_value_chars": encoding.max_attribute_value_chars,
            "max_attribute_value_tokens": encoding.max_attribute_value_tokens,
        }
        (output_path / "initialization_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        artifacts = TrainingArtifacts(
            predictor="transformer",
            transformer_dir=output_path,
        )
        save_app_config(config, config.training.resolved_config_path)
        solution_path = save_solution_manifest(config, artifacts)
        logger.info("Initialized Transformer artifact saved to {!s}", output_path)
        return replace(
            artifacts,
            resolved_config_path=config.training.resolved_config_path,
            solution_path=solution_path,
        )


__all__ = ["DEFAULT_INITIALIZED_MAX_LENGTH", "initialize"]
