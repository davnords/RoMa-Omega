"""Offline synthetic self-test for the matching / pose / AUC math.

Builds two cameras looking at a sphere (a *non-planar* surface, so the essential
matrix is well-posed), renders exact depth maps, and checks that:

* both correspondence extractors return geometrically consistent 3D matches,
* ``estimate_relative_pose`` recovers the ground-truth relative pose from them,
* the direct-pose baseline is exact, and
* ``relative_pose_error`` / ``pose_auc`` behave at zero error.

Runs on CPU with no model or checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch

from . import geometry
from .correspondences import MatchConfig, match_depth_warp, match_mutual_nn_3d
from .inference import PairPrediction
from .pose_eval import direct_relative_pose, estimate_relative_pose, pose_auc, relative_pose_error


def _look_at(eye, target, world_up=(0.0, -1.0, 0.0)):
    """OpenCV camera-from-world (R, t) looking from ``eye`` at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)  # camera +z (forward)
    r = np.cross(f, np.asarray(world_up, dtype=np.float64))
    r /= np.linalg.norm(r)  # camera +x (right)
    d = np.cross(f, r)  # camera +y (down)
    R_c2w = np.stack([r, d, f], axis=1)
    if np.linalg.det(R_c2w) < 0:
        R_c2w[:, 0] *= -1
    R = R_c2w.T  # world-to-camera
    t = -R @ eye
    return R, t


def _render_sphere_depth(K, R, t, center, radius, height, width):
    """Exact depth map of a sphere; invalid (missed) pixels get depth 0."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dir_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=np.float64)], axis=-1)
    R_c2w = R.T
    eye = -R.T @ t
    w = dir_cam @ R_c2w.T  # world-space ray directions

    oc = eye - center
    a = np.einsum("hwi,hwi->hw", w, w)
    b = 2.0 * (w @ oc)
    c = oc @ oc - radius ** 2
    disc = b * b - 4 * a * c
    hit = disc >= 0
    sqrt_disc = np.sqrt(np.clip(disc, 0, None))
    s = (-b - sqrt_disc) / (2 * a)
    hit &= s > 0

    world = eye[None, None] + s[..., None] * w
    cam = np.einsum("ij,hwj->hwi", R, world) + t[None, None]
    depth = np.where(hit & (cam[..., 2] > 0), cam[..., 2], 0.0)
    return depth.astype(np.float32)


def build_synthetic_pair(height=256, width=320, patch_size=16):
    fx = fy = 150.0
    K = np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]], dtype=np.float64)
    center = np.array([0.0, 0.0, 3.0])
    radius = 1.3

    R0, t0 = _look_at(np.array([0.0, 0.0, 0.0]), center)
    R1, t1 = _look_at(np.array([0.5, 0.15, 0.1]), center)
    extr = np.stack([np.concatenate([R0, t0[:, None]], axis=1), np.concatenate([R1, t1[:, None]], axis=1)])

    d0 = _render_sphere_depth(K, R0, t0, center, radius, height, width)
    d1 = _render_sphere_depth(K, R1, t1, center, radius, height, width)
    depth = np.stack([d0, d1])
    conf = np.where(depth > 0, 2.0, 0.0).astype(np.float32)
    intr = np.stack([K, K]).astype(np.float64)

    world_points = np.stack(
        [
            geometry.unproject_depth(
                torch.from_numpy(depth[i]),
                torch.from_numpy(intr[i]).float(),
                torch.from_numpy(extr[i]).float(),
            ).numpy()
            for i in range(2)
        ]
    )
    # Mark missed-ray points as non-finite so they are filtered from matching.
    world_points[depth <= 0] = np.nan

    gh, gw = height // patch_size, width // patch_size
    ys, xs = np.meshgrid(np.linspace(0, 1, gh), np.linspace(0, 1, gw), indexing="ij")
    grad = np.stack([ys, xs, 0.5 * (ys + xs)], axis=-1)
    images = np.stack([grad, grad[:, ::-1]]).astype(np.float32)
    images = np.stack([_upscale(images[i], height, width) for i in range(2)])
    patch_feats = np.zeros((2, gh, gw, 8), dtype=np.float32)

    pred = PairPrediction(
        images=images,
        depth=depth,
        depth_conf=conf,
        extrinsics=extr.astype(np.float32),
        intrinsics=intr.astype(np.float32),
        world_points=world_points.astype(np.float32),
        patch_feats=patch_feats,
        patch_size=patch_size,
    )
    T_0to1 = np.eye(4)
    E0 = np.eye(4); E0[:3] = extr[0]
    E1 = np.eye(4); E1[:3] = extr[1]
    T_0to1 = E1 @ np.linalg.inv(E0)
    return pred, K, T_0to1


def _upscale(img, height, width):
    gh, gw = img.shape[:2]
    yi = (np.linspace(0, gh - 1, height)).astype(int)
    xi = (np.linspace(0, gw - 1, width)).astype(int)
    return img[yi][:, xi]


def _geometric_consistency(pred, result):
    """Median 3D distance between matched surface points (independent check)."""
    if len(result) == 0:
        return np.inf
    p0 = result.mkpts0.astype(int)
    p1 = result.mkpts1.astype(int)
    A = pred.world_points[0][p0[:, 1], p0[:, 0]]
    B = pred.world_points[1][p1[:, 1], p1[:, 0]]
    return float(np.nanmedian(np.linalg.norm(A - B, axis=-1)))


def run_self_test() -> bool:
    print("=== matching self-test (synthetic sphere, CPU) ===")
    pred, K, T_0to1 = build_synthetic_pair()
    cfg = MatchConfig(stride=2, conf_thresh=1.0, nn_rel_thresh=0.03, depth_rel_thresh=0.02, cycle_px_thresh=2.0)

    ok = True

    # Direct-pose baseline must be essentially exact (residual is float32 storage).
    R_d, t_d = direct_relative_pose(pred.extrinsics[0], pred.extrinsics[1])
    t_err_d, R_err_d = relative_pose_error(T_0to1, R_d, t_d)
    print(f"[direct baseline] t_err={t_err_d:.3e} deg, R_err={R_err_d:.3e} deg")
    ok &= t_err_d < 0.05 and R_err_d < 0.05

    # AUC at zero error must be 100.
    aucs0 = pose_auc([0.0] * 10)
    print(f"[pose_auc @0] {aucs0}")
    ok &= all(abs(v - 100.0) < 1e-6 for v in aucs0.values())

    for name, result in [
        ("mutual_nn_3d", match_mutual_nn_3d(pred, cfg)),
        ("depth_warp", match_depth_warp(pred, cfg, device="cpu")),
    ]:
        n = len(result)
        consistency = _geometric_consistency(pred, result)
        est = estimate_relative_pose(result.mkpts0, result.mkpts1, K, K, thresh_px=0.5)
        if est is None:
            print(f"[{name}] FAILED to estimate pose ({n} matches)")
            ok = False
            continue
        R, t, inliers = est
        t_err, R_err = relative_pose_error(T_0to1, R, t)
        print(
            f"[{name}] matches={n:5d} 3d_consistency={consistency:.4f} "
            f"t_err={t_err:6.2f} deg R_err={R_err:6.2f} deg inliers={int(inliers.sum())}"
        )
        ok &= n > 50
        ok &= consistency < 0.05
        # depth_warp is sub-pixel exact; mutual_nn_3d is limited to pixel-grid
        # resolution, so it carries a small but non-zero pose floor.
        tol_t, tol_R = (0.5, 0.5) if name == "depth_warp" else (5.0, 5.0)
        ok &= t_err < tol_t and R_err < tol_R

    print("=== SELF-TEST", "PASSED" if ok else "FAILED", "===")
    return ok
