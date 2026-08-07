"""2D-match extraction, visualization and pose-AUC evaluation for VGGT-Omega.

Turns the model's raw per-frame depth + camera predictions for an image pair into
2D-2D pixel correspondences, with two extractors, match/feature visualization, and a
ScanNet-1500 relative-pose AUC benchmark.
"""

from .correspondences import (
    MatchConfig,
    MatchResult,
    match_depth_warp,
    match_features,
    match_mutual_nn_3d,
)
from .features import query_patch_correlation
from .inference import PairPrediction, run_pair
from .pose_eval import (
    direct_relative_pose,
    estimate_relative_pose,
    pose_auc,
    relative_pose_error,
)
from .visualize import draw_matches, visualize_correlation

__all__ = [
    "PairPrediction",
    "run_pair",
    "MatchConfig",
    "MatchResult",
    "match_mutual_nn_3d",
    "match_depth_warp",
    "match_features",
    "query_patch_correlation",
    "estimate_relative_pose",
    "relative_pose_error",
    "pose_auc",
    "direct_relative_pose",
    "draw_matches",
    "visualize_correlation",
]
