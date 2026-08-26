"""Common contracts, artifacts and lazily loaded model implementations."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BoostingPredictor": (".boosting.predictor", "BoostingPredictor"),
    "BoostingTrainer": (".boosting.training", "BoostingTrainer"),
    "CascadePredictor": (".cascade.predictor", "CascadePredictor"),
    "FusionPredictor": (".fusion.predictor", "FusionPredictor"),
    "FusionTrainer": (".fusion.training", "FusionTrainer"),
    "MatchPredictor": (".contracts", "MatchPredictor"),
    "MaxPoolingPredictor": (".maxpooling.predictor", "MaxPoolingPredictor"),
    "MaxPoolingTrainer": (".maxpooling.training", "MaxPoolingTrainer"),
    "ModelTrainer": (".contracts", "ModelTrainer"),
    "PairEncoder": (".contracts", "PairEncoder"),
    "PredictionBatch": (".contracts", "PredictionBatch"),
    "StackingPredictor": (".stacking.predictor", "StackingPredictor"),
    "StackingTrainer": (".stacking.training", "StackingTrainer"),
    "TrainingArtifacts": (".artifacts", "TrainingArtifacts"),
    "TransformerPredictor": (".transformer.predictor", "TransformerPredictor"),
    "TransformerTrainer": (".transformer.training", "TransformerTrainer"),
    "build_predictor": (".factory", "build_predictor"),
    "build_trainer": (".factory", "build_trainer"),
    "save_solution_manifest": (".artifacts", "save_solution_manifest"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
