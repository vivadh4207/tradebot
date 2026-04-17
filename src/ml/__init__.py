"""Machine-learning primitives for the LSTM signal.

Heavy imports (torch) are lazy so `import src.ml` works on machines without
PyTorch installed (Mac, small VPS). Training only runs on the Jetson.
"""
from .features import build_feature_matrix, feature_columns, FeatureStats
from .checkpoint import CheckpointMeta, save_checkpoint, load_checkpoint

__all__ = [
    "build_feature_matrix", "feature_columns", "FeatureStats",
    "CheckpointMeta", "save_checkpoint", "load_checkpoint",
]
