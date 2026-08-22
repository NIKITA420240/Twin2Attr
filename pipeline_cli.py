"""Hydra command-line entry point for the Twin2Attr pipeline."""

import hydra
from omegaconf import DictConfig

from match.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="pipeline")
def main(config: DictConfig) -> None:
    run_pipeline(config)


if __name__ == "__main__":
    main()
