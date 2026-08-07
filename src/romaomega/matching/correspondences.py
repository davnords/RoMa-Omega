"""Extract 2D-2D correspondences from VGGT-Omega's raw 3D predictions.

Two independent methods, both operating in the preprocessed image frame:

1. ``match_mutual_nn_3d`` -- unproject both depth maps into a shared world frame and
   reciprocally (mutual-nearest-neighbour) match the two point clouds, keeping pairs
   whose 3D distance is small relative to the scene scale. This is our own solution
   (SciPy KD-trees); it deliberately does not depend on the vendored ``dust3r`` repo.

2. ``match_depth_warp`` -- warp each source pixel through its predicted depth+camera
   into the other view, accept it when the reprojected depth agrees with the other
   view's predicted depth, and keep only cycle-consistent (mutual) matches.

Both return a :class:`MatchResult` with pixel coordinates ``(x, y)`` in each image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry

__all__ = [
    "MatchConfig",
    "MatchResult",
    "match_mutual_nn_3d",
    "match_depth_warp",
    "match_features",
]


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _nearest_neighbors(query: torch.Tensor, database: torch.Tensor, chunk: int = 8192):
    """For each row of ``query`` return ``(min_dist, argmin_index)`` in ``database``.

    Chunked over the query dimension to bound memory. Runs on ``query.device``.
    """
    n = query.shape[0]
    dists = torch.empty(n, device=query.device, dtype=query.dtype)
    idx = torch.empty(n, device=query.device, dtype=torch.long)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(query[start:end], database)  # (c, M)
        dmin, amin = d.min(dim=1)
        dists[start:end] = dmin
        idx[start:end] = amin
    return dists, idx


@dataclass
class MatchConfig:
    # Shared subsampling / filtering.
    stride: int = 8  # sample source pixels on a grid of this stride
    conf_thresh: float = 1.0  # keep pixels with depth_conf >= this (conf is >= 1)

    # Method 1 (mutual NN in 3D): keep pairs with 3D distance below this fraction of
    # the robust scene scale.
    nn_rel_thresh: float = 0.02

    # Method 2 (depth-check warp).
    depth_rel_thresh: float = 0.05  # |z_reproj - z_other| / z_other tolerance
    cycle_px_thresh: float = 3.0  # forward-backward pixel round-trip tolerance

    # Method 3 (feature / descriptor matching).
    feature_center: bool = True  # subtract common-mode patch feature before cosine
    feature_ratio_thresh: float | None = None  # Lowe ratio test (None disables)


@dataclass
class MatchResult:
    mkpts0: np.ndarray  # (K, 2) pixel (x, y) in image 0
    mkpts1: np.ndarray  # (K, 2) pixel (x, y) in image 1
    scores: np.ndarray  # (K,) higher is better
    method: str

    def __len__(self) -> int:
        return len(self.mkpts0)


def _sampled_pixels(height: int, width: int, stride: int) -> np.ndarray:
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.int64)  # (M, 2) xy


def _empty_result(method: str) -> MatchResult:
    return MatchResult(
        mkpts0=np.zeros((0, 2)), mkpts1=np.zeros((0, 2)), scores=np.zeros((0,)), method=method
    )


def match_mutual_nn_3d(pred, cfg: MatchConfig | None = None, device: str | None = None) -> MatchResult:
    """Mutual nearest neighbours between the two predicted world-point clouds.

    Implemented with chunked ``torch.cdist`` so it runs directly on the GPU tensors
    (no SciPy / host round-trip), which suits VGGT-Omega's dense predictions.
    """
    cfg = cfg or MatchConfig()
    device = _resolve_device(device)
    height, width = pred.hw
    pix = _sampled_pixels(height, width, cfg.stride)  # (M, 2)

    def collect(i):
        rows, cols = pix[:, 1], pix[:, 0]
        P = pred.world_points[i][rows, cols]  # (M, 3)
        conf = pred.depth_conf[i][rows, cols]
        depth = pred.depth[i][rows, cols]
        valid = np.isfinite(P).all(axis=1) & (conf >= cfg.conf_thresh) & (depth > 0)
        return P[valid], pix[valid].astype(np.float64)

    P0_np, pix0 = collect(0)
    P1_np, pix1 = collect(1)
    if len(P0_np) < 1 or len(P1_np) < 1:
        return _empty_result("mutual_nn_3d")

    P0 = torch.as_tensor(P0_np, dtype=torch.float32, device=device)
    P1 = torch.as_tensor(P1_np, dtype=torch.float32, device=device)

    dist01, nn01 = _nearest_neighbors(P0, P1)  # nearest P1 for each P0
    _, nn10 = _nearest_neighbors(P1, P0)  # nearest P0 for each P1

    idx0 = torch.arange(len(P0), device=device)
    mutual = nn10[nn01] == idx0
    i0 = idx0[mutual]
    i1 = nn01[mutual]
    dist = dist01[mutual]

    scale = geometry.scene_scale(torch.cat([P0, P1], dim=0)).item()
    thr = cfg.nn_rel_thresh * max(scale, 1e-9)
    keep = dist < thr
    i0 = i0[keep].cpu().numpy()
    i1 = i1[keep].cpu().numpy()
    dist = dist[keep].cpu().numpy()
    if len(i0) == 0:
        return _empty_result("mutual_nn_3d")

    scores = np.clip(1.0 - dist / thr, 0.0, 1.0)
    return MatchResult(mkpts0=pix0[i0], mkpts1=pix1[i1], scores=scores, method="mutual_nn_3d")


def _warp(uv_src, depth_src, K_src, E_src, K_dst, E_dst):
    """Warp source pixels into the destination view.

    Returns ``(uv_dst, z_reproj)`` where ``z_reproj`` is the depth of the source
    point in the destination camera.
    """
    z = geometry.sample_map(depth_src, uv_src, mode="nearest")  # (M,)
    dirs = geometry.backproject_dirs(K_src, uv_src)  # (M, 3)
    cam = dirs * z[:, None]
    world = (cam - E_src[:3, 3]) @ E_src[:3, :3]  # R^T (X_cam - t)
    uv_dst, z_reproj, _ = geometry.project_points(K_dst, E_dst, world)
    return uv_dst, z_reproj


def match_depth_warp(pred, cfg: MatchConfig | None = None, device: str | None = None) -> MatchResult:
    """Reciprocal depth-consistency warp between the two views."""
    cfg = cfg or MatchConfig()
    device = _resolve_device(device)
    height, width = pred.hw

    depth = torch.as_tensor(pred.depth, dtype=torch.float32, device=device)  # (2, H, W)
    conf = torch.as_tensor(pred.depth_conf, dtype=torch.float32, device=device)
    K = torch.as_tensor(pred.intrinsics, dtype=torch.float32, device=device)  # (2, 3, 3)
    E = torch.as_tensor(pred.extrinsics, dtype=torch.float32, device=device)  # (2, 3, 4)

    pix = _sampled_pixels(height, width, cfg.stride)
    uv0 = torch.as_tensor(pix, dtype=torch.float32, device=device)  # (M, 2)

    conf0 = geometry.sample_map(conf[0], uv0, mode="nearest")
    depth0 = geometry.sample_map(depth[0], uv0, mode="nearest")
    valid0 = (conf0 >= cfg.conf_thresh) & (depth0 > 0)
    uv0 = uv0[valid0]
    if uv0.shape[0] == 0:
        return _empty_result("depth_warp")

    # Forward warp 0 -> 1.
    uv1, z1_reproj = _warp(uv0, depth[0], K[0], E[0], K[1], E[1])
    inb1 = geometry.in_bounds(uv1, height, width)
    depth1_at = geometry.sample_map(depth[1], uv1, mode="bilinear")
    rel_err_fwd = (z1_reproj - depth1_at).abs() / depth1_at.clamp(min=1e-6)
    fwd_ok = inb1 & (z1_reproj > 0) & (depth1_at > 0) & (rel_err_fwd < cfg.depth_rel_thresh)

    # Backward warp 1 -> 0 for cycle consistency.
    uv0_rt, _ = _warp(uv1, depth[1], K[1], E[1], K[0], E[0])
    cycle_err = (uv0_rt - uv0).norm(dim=-1)
    cycle_ok = cycle_err < cfg.cycle_px_thresh

    keep = fwd_ok & cycle_ok
    if keep.sum().item() == 0:
        return _empty_result("depth_warp")

    mkpts0 = uv0[keep].cpu().numpy()
    mkpts1 = uv1[keep].cpu().numpy()
    scores = torch.clamp(1.0 - rel_err_fwd[keep] / cfg.depth_rel_thresh, 0.0, 1.0).cpu().numpy()
    return MatchResult(mkpts0=mkpts0, mkpts1=mkpts1, scores=scores, method="depth_warp")


def _patch_centers(gh: int, gw: int, patch_size: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    cx = xs.ravel() * patch_size + patch_size / 2.0
    cy = ys.ravel() * patch_size + patch_size / 2.0
    return np.stack([cx, cy], axis=-1)  # (N, 2) xy


def match_features(
    pred, cfg: MatchConfig | None = None, layer: int = 11, device: str | None = None
) -> MatchResult:
    """Mutual-nearest-neighbour matching of per-patch aggregator descriptors.

    Unlike the geometry matchers, this uses only the encoder features at a chosen
    cached layer (``pred.patch_feats_by_layer[layer]``), so it is an *independent*
    correspondence signal. Matches are at patch centres (16 px grid), so they are
    coarse; a Lowe ratio test (``cfg.feature_ratio_thresh``) can prune ambiguous ones.
    """
    cfg = cfg or MatchConfig()
    device = _resolve_device(device)

    if pred.patch_feats_by_layer is None or layer not in pred.patch_feats_by_layer:
        raise ValueError(
            f"layer {layer} not in patch_feats_by_layer; run run_pair(..., feature_layers=[...])."
        )

    feats = torch.as_tensor(pred.patch_feats_by_layer[layer], dtype=torch.float32, device=device)
    _, gh, gw, C = feats.shape
    f0 = feats[0].reshape(-1, C)
    f1 = feats[1].reshape(-1, C)

    if cfg.feature_center:
        mean_feat = feats.reshape(-1, C).mean(dim=0)
        f0 = f0 - mean_feat
        f1 = f1 - mean_feat

    f0 = F.normalize(f0, dim=-1)
    f1 = F.normalize(f1, dim=-1)
    sim = f0 @ f1.T  # (N0, N1) cosine similarity

    s01, nn01 = sim.max(dim=1)  # best patch in image 1 for each patch in image 0
    _, nn10 = sim.max(dim=0)  # best patch in image 0 for each patch in image 1
    idx0 = torch.arange(sim.shape[0], device=device)
    keep = nn10[nn01] == idx0  # mutual nearest neighbours

    if cfg.feature_ratio_thresh is not None and sim.shape[1] >= 2:
        top2 = sim.topk(2, dim=1).values
        dist_best = 1.0 - top2[:, 0]
        dist_second = (1.0 - top2[:, 1]).clamp(min=1e-6)
        keep = keep & (dist_best <= cfg.feature_ratio_thresh * dist_second)

    i0 = idx0[keep].cpu().numpy()
    i1 = nn01[keep].cpu().numpy()
    if len(i0) == 0:
        return _empty_result(f"features@L{layer}")

    centers = _patch_centers(gh, gw, pred.patch_size)
    scores = s01[keep].cpu().numpy()
    return MatchResult(mkpts0=centers[i0], mkpts1=centers[i1], scores=scores, method=f"features@L{layer}")
