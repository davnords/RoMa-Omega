"""Torch geometry helpers for turning VGGT-Omega depth + camera predictions into
2D/3D correspondences.

All functions operate in the *preprocessed image frame* (the H x W the model saw)
and follow the model's conventions:

* Extrinsics are 3x4 camera-from-world matrices in OpenCV coordinates
  (``X_cam = R @ X_world + t``), matching ``encoding_to_camera``.
* Intrinsics are 3x3 pinhole matrices at the preprocessed resolution.
* Pixel coordinates are ``(x, y)`` with the pixel *centre* at integer coordinates,
  so principal point ``(cx, cy)`` is applied directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "pixel_grid",
    "backproject_dirs",
    "unproject_depth",
    "project_points",
    "sample_map",
    "scene_scale",
]


def pixel_grid(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return an ``(H*W, 2)`` tensor of pixel-centre ``(x, y)`` coordinates."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def backproject_dirs(K: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Camera-ray directions (z = 1) for pixel coordinates ``uv`` (``(N, 2)``)."""
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """Unproject a depth map into world coordinates.

    ``depth`` is ``(H, W)``; returns world points ``(H, W, 3)`` using
    ``X_world = R^T (X_cam - t)`` (mirrors ``demo_gradio.unproject_depth_map_to_point_map``).
    """
    height, width = depth.shape
    uv = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    dirs = backproject_dirs(K, uv)
    cam_pts = dirs * depth.reshape(-1, 1)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    world = (cam_pts - t) @ R  # (X_cam - t) @ R == R^T (X_cam - t) for row vectors
    return world.reshape(height, width, 3)


def project_points(K: torch.Tensor, extrinsic: torch.Tensor, world: torch.Tensor):
    """Project world points ``(N, 3)`` into a camera.

    Returns ``(uv, depth, cam_pts)`` where ``uv`` is ``(N, 2)`` pixel coordinates,
    ``depth`` is the camera-space z (``(N,)``) and ``cam_pts`` is ``(N, 3)``.
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam = world @ R.T + t  # R @ X_world + t for row vectors
    z = cam[:, 2]
    z_safe = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    u = K[0, 0] * cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z_safe + K[1, 2]
    return torch.stack([u, v], dim=-1), z, cam


def sample_map(value_map: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample ``value_map`` at pixel coordinates ``uv`` (``(N, 2)``).

    ``value_map`` is ``(H, W)`` or ``(H, W, C)``; returns ``(N,)`` or ``(N, C)``.
    Uses ``align_corners=True`` so integer pixels map exactly, with border padding.
    """
    height, width = value_map.shape[:2]
    gx = uv[:, 0] / max(width - 1, 1) * 2.0 - 1.0
    gy = uv[:, 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

    if value_map.dim() == 2:
        inp = value_map[None, None]
    else:
        inp = value_map.permute(2, 0, 1)[None]

    sampled = F.grid_sample(
        inp.float(), grid.float(), mode=mode, align_corners=True, padding_mode="border"
    )
    sampled = sampled[0, :, :, 0].transpose(0, 1)  # (N, C)
    if value_map.dim() == 2:
        return sampled[:, 0]
    return sampled


def in_bounds(uv: torch.Tensor, height: int, width: int, margin: float = 0.0) -> torch.Tensor:
    """Boolean mask of pixel coordinates that lie inside the image."""
    return (
        (uv[:, 0] >= margin)
        & (uv[:, 0] <= width - 1 - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] <= height - 1 - margin)
    )


def scene_scale(points: torch.Tensor) -> torch.Tensor:
    """Robust scene scale: median distance of points to their median centre."""
    center = points.median(dim=0).values
    return (points - center).norm(dim=-1).median()
