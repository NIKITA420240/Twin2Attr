from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from match.paths import resolve_project_path
from match.utils import predict_pipeline


def _resolve_path(value: str) -> Path:
    return resolve_project_path(value)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    items_path = _resolve_path(cfg.items_path)
    matches_path = _resolve_path(cfg.matches_path)
    output_path = _resolve_path(cfg.output_path)

    logger.info("Starting product matching")
    predict_pipeline(
        data_path=items_path,
        match_path=matches_path,
        model_path=_resolve_path(cfg.model_path),
        logreg_path=_resolve_path(cfg.classifier_path),
        output_csv_path=output_path,
        device=cfg.device,
        batch_size=cfg.batch_size,
        backend=cfg.backend,
    )


if __name__ == "__main__":
    main()
