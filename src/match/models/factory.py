"""Construction of training and inference strategies from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .contracts import MatchPredictor, ModelTrainer

if TYPE_CHECKING:
    from ..config import AppConfig
    from .transformer.predictor import TransformerPredictor


def build_trainer(config: AppConfig) -> ModelTrainer:
    """Create the trainer selected by ``training.model``."""
    if config.training.model == "transformer":
        from .transformer.training import TransformerTrainer

        return TransformerTrainer(config)
    if config.training.model == "maxpooling":
        from .maxpooling.training import MaxPoolingTrainer

        return MaxPoolingTrainer(config)
    if config.training.model == "fusion":
        from .fusion.training import FusionTrainer
        from .maxpooling.training import MaxPoolingTrainer
        from .transformer.training import TransformerTrainer

        return FusionTrainer(
            config=config,
            transformer=TransformerTrainer(config),
            maxpooling=MaxPoolingTrainer(config),
        )
    if config.training.model == "boosting":
        from .boosting.training import BoostingTrainer

        return BoostingTrainer(config)
    if config.training.model == "stacking":
        from .stacking.training import StackingTrainer
        from .transformer.training import TransformerTrainer

        return StackingTrainer(
            config=config,
            transformer=TransformerTrainer(config),
        )
    raise ValueError(f"Unsupported training model: {config.training.model!r}")


def _artifact_path(value: Any, *, root: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"solution field {name!r} must contain a path")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def build_predictor(
    solution: Mapping[str, Any],
    solution_root: Path,
) -> MatchPredictor:
    """Create the predictor described by a packaged solution manifest."""
    predictor_name = str(solution.get("predictor", "transformer"))
    device = solution.get("device")
    transformer: TransformerPredictor | None = None
    if predictor_name in {"transformer", "fusion"}:
        from .transformer.predictor import TransformerPredictor

        transformer = TransformerPredictor.load(
            _artifact_path(
                solution.get("model_directory"),
                root=solution_root,
                name="model_directory",
            ),
            batch_size=int(solution.get("batch_size", 64)),
            device=device,
        )
        if predictor_name == "transformer":
            return transformer

    maxpooling: Any | None = None
    if predictor_name in {"maxpooling", "fusion"}:
        from .maxpooling.predictor import MaxPoolingPredictor

        maxpooling = MaxPoolingPredictor.load(
            _artifact_path(
                solution.get("maxpooling_path"),
                root=solution_root,
                name="maxpooling_path",
            ),
            batch_size=int(solution.get("maxpooling_batch_size", 512)),
            device=device,
        )
        if predictor_name == "maxpooling":
            return maxpooling

    if predictor_name == "fusion":
        from .fusion.predictor import FusionPredictor

        if transformer is None or maxpooling is None:
            raise RuntimeError("fusion predictor requires both pair encoders")
        return FusionPredictor.load(
            _artifact_path(
                solution.get("fusion_path"),
                root=solution_root,
                name="fusion_path",
            ),
            transformer=transformer,
            maxpooling=maxpooling,
            batch_size=int(solution.get("fusion_batch_size", 512)),
            device=device,
        )
    if predictor_name == "boosting":
        from .boosting.predictor import BoostingPredictor

        return BoostingPredictor.load(
            _artifact_path(
                solution.get("boosting_directory"),
                root=solution_root,
                name="boosting_directory",
            ),
            thread_count=int(solution.get("boosting_thread_count", -1)),
        )
    if predictor_name == "cascade":
        from .boosting.predictor import BoostingPredictor
        from .cascade.predictor import CascadePredictor
        from .transformer.predictor import TransformerPredictor

        fast_name = str(solution.get("fast_model", "boosting"))
        main_name = str(solution.get("main_model", "transformer"))
        if fast_name != "boosting" or main_name != "transformer":
            raise ValueError(
                "current cascade supports fast_model='boosting' and "
                "main_model='transformer'"
            )
        fast_model = BoostingPredictor.load(
            _artifact_path(
                solution.get("boosting_directory"),
                root=solution_root,
                name="boosting_directory",
            ),
            thread_count=int(solution.get("boosting_thread_count", -1)),
        )
        main_model = TransformerPredictor.load(
            _artifact_path(
                solution.get("model_directory"),
                root=solution_root,
                name="model_directory",
            ),
            batch_size=int(solution.get("batch_size", 64)),
            device=device,
        )
        return CascadePredictor(
            fast_model,
            main_model,
            negative_threshold=float(solution.get("negative_threshold", 0.01)),
            positive_threshold=float(solution.get("positive_threshold", 0.99)),
        )
    if predictor_name == "stacking":
        from .stacking.predictor import StackingPredictor
        from .transformer.predictor import TransformerPredictor

        if str(solution.get("base_model")) != "transformer":
            raise ValueError("stacking base_model must be 'transformer'")
        if str(solution.get("stacking_model")) != "boosting":
            raise ValueError("stacking stacking_model must be 'boosting'")
        transformer = TransformerPredictor.load(
            _artifact_path(
                solution.get("model_directory"),
                root=solution_root,
                name="model_directory",
            ),
            batch_size=int(solution.get("batch_size", 64)),
            device=device,
        )
        return StackingPredictor.load(
            _artifact_path(
                solution.get("stacking_directory"),
                root=solution_root,
                name="stacking_directory",
            ),
            transformer=transformer,
            thread_count=int(solution.get("stacking_thread_count", -1)),
        )
    raise ValueError(f"Unsupported predictor in solution.json: {predictor_name!r}")


__all__ = ["build_predictor", "build_trainer"]
