"""Two-view relative-pose recovery and AUC evaluation.

Follows the standard SuperGlue / LoFTR ScanNet-1500 protocol: recover the relative
pose from 2D-2D correspondences via the essential matrix + RANSAC, measure the
rotation and translation-direction error against the ground-truth ``T_0to1`` (which
maps points from camera 0 to camera 1), and report the pose AUC at several
angular thresholds.
"""

from __future__ import annotations

import cv2
import numpy as np

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0 and removed later;
# support both (compute the fallback lazily so a missing ``trapz`` doesn't raise).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

__all__ = [
    "estimate_relative_pose",
    "relative_pose_error",
    "pose_auc",
    "direct_relative_pose",
]


def _normalize_keypoints(kpts: np.ndarray, K: np.ndarray) -> np.ndarray:
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    out = np.empty_like(kpts, dtype=np.float64)
    out[:, 0] = (kpts[:, 0] - cx) / fx
    out[:, 1] = (kpts[:, 1] - cy) / fy
    return out


def estimate_relative_pose(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    thresh_px: float = 0.5,
    conf: float = 0.99999,
):
    """Recover ``(R, t, inlier_mask)`` mapping camera 0 -> camera 1 from matches.

    Points are normalized by their own intrinsics so ``K0`` and ``K1`` may differ.
    ``t`` is a unit translation direction (essential matrices are scale-free).
    Returns ``None`` when the problem is degenerate (too few points / no model).
    """
    if len(mkpts0) < 5:
        return None

    kpts0 = _normalize_keypoints(np.asarray(mkpts0, dtype=np.float64), K0)
    kpts1 = _normalize_keypoints(np.asarray(mkpts1, dtype=np.float64), K1)

    # Threshold is in normalized-camera units; scale the pixel threshold by the
    # mean focal length across both cameras.
    mean_focal = np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
    norm_thresh = thresh_px / mean_focal

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), method=cv2.RANSAC, prob=conf, threshold=norm_thresh
    )
    if E is None or E.shape[0] < 3:
        return None

    # findEssentialMat may return several stacked 3x3 candidates; keep the one that
    # recovers the most points in front of both cameras.
    best = None
    for i in range(0, E.shape[0], 3):
        E_candidate = E[i : i + 3]
        if E_candidate.shape != (3, 3):
            continue
        mask_arg = mask.copy() if mask is not None else None
        n_inliers, R, t, pose_mask = cv2.recoverPose(E_candidate, kpts0, kpts1, np.eye(3), mask=mask_arg)
        inliers = pose_mask.ravel() > 0 if pose_mask is not None else np.ones(len(kpts0), dtype=bool)
        if best is None or n_inliers > best[0]:
            best = (n_inliers, R, t[:, 0], inliers)

    if best is None:
        return None
    return best[1], best[2], best[3]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def relative_pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Return ``(translation_error_deg, rotation_error_deg)``.

    Translation is compared as a direction and folded into ``[0, 90]`` to absorb
    the sign ambiguity of an essential-matrix translation.
    """
    t_gt = T_0to1[:3, 3]
    t_err = _angle_between(t, t_gt)
    t_err = min(t_err, 180.0 - t_err)

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(R_gt.T @ R) - 1.0) / 2.0
    R_err = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return t_err, R_err


def pose_auc(errors, thresholds=(5, 10, 20)) -> dict:
    """Pose AUC (as percentages) over angular error thresholds (LoFTR convention).

    ``errors`` are per-pair ``max(t_err, R_err)`` in degrees; failed pairs should be
    passed in as a large error (e.g. 180) so they count as misses.
    """
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    if len(errors) == 0:
        return {thr: 0.0 for thr in thresholds}

    recall = np.arange(1, len(errors) + 1) / len(errors)
    aucs = {}
    for thr in thresholds:
        idx = np.searchsorted(errors, thr, side="right")
        e = np.concatenate([errors[:idx], [thr]])
        r = np.concatenate([recall[:idx], [recall[idx - 1] if idx > 0 else 0.0]])
        aucs[thr] = float(_trapz(r, x=e) / thr * 100.0)
    return aucs


def direct_relative_pose(extr0: np.ndarray, extr1: np.ndarray):
    """Relative pose ``T_0to1`` composed from two predicted 3x4 extrinsics.

    Extrinsics are camera-from-world, so ``T_0to1 = E1 @ inv(E0)`` maps points from
    camera 0 to camera 1. Returns ``(R, t)`` for use as a direct-prediction baseline.
    """
    E0 = np.eye(4)
    E0[:3] = extr0
    E1 = np.eye(4)
    E1[:3] = extr1
    T = E1 @ np.linalg.inv(E0)
    return T[:3, :3], T[:3, 3]
