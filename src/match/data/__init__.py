"""Loading and preparation primitives shared by Twin2Attr workflows."""

from .loading import TrainingMatchPaths, load_training_matches, read_parquet
from .preparation import TrainingData, prepare_pair_rows, prepare_training_data

__all__ = [
    "TrainingData",
    "TrainingMatchPaths",
    "load_training_matches",
    "prepare_pair_rows",
    "prepare_training_data",
    "read_parquet",
]
