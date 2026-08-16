import argparse

from loguru import logger

from match.paths import resolve_project_path
from match.utils import predict_pipeline

CLASSIFIER_PATH = "baseline_logreg_l12.joblib"
MODEL_CE_PATH = "models/cross-encoder-ms-marco-MiniLM-L12-v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match product-card pairs")
    parser.add_argument(
        "--items_path",
        "--items-path",
        "-i",
        dest="items_path",
        required=True,
        help="Path to the product data file",
    )
    parser.add_argument(
        "--matches_path",
        "--matches-path",
        "-m",
        dest="matches_path",
        required=True,
        help="Path to the prepared product-pair file",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "-o",
        dest="output_path",
        required=True,
        help="Path where predictions will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Starting product matching")
    predict_pipeline(
        data_path=resolve_project_path(args.items_path),
        match_path=resolve_project_path(args.matches_path),
        model_path=resolve_project_path(MODEL_CE_PATH),
        logreg_path=resolve_project_path(CLASSIFIER_PATH),
        output_csv_path=resolve_project_path(args.output_path),
        batch_size=512,
    )


if __name__ == "__main__":
    main()
