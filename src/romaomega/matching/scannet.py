"""ScanNet-1500 test-pair loading and intrinsics handling.

The dataset (symlinked at ``scannet_test_1500/``) stores:

* ``test.npz``: ``name (1500, 4) uint16`` = ``[scene_num, sub, stem0, stem1]`` and
  ``rel_pose (1500, 12)`` = flattened 3x4 relative pose. Verified against the
  per-frame poses: ``rel_pose == inv(P1) @ P0 == T_0to1`` (maps camera 0 -> camera 1),
  where ``pose/{stem}.txt`` are 4x4 camera-to-world matrices.
* ``intrinsics.npz``: per-scene 3x3 intrinsics at 640x480 (= the colour intrinsics
  scaled down).
* Per scene: ``color/{stem}.jpg`` (1296x968), ``intrinsic/intrinsic_color.txt``.

Matches are extracted in the model's preprocessed frame, so ``gt_K_in_preprocessed_frame``
replays the ``load_and_preprocess_images`` crop+resize to bring the ground-truth
colour intrinsics into that same frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Native ScanNet colour resolution (width, height).
SCANNET_COLOR_WH = (1296, 968)

__all__ = [
    "ScanNetPair",
    "load_pairs",
    "image_path",
    "load_color_intrinsics",
    "preprocess_transform",
    "adjust_intrinsics",
    "gt_K_in_preprocessed_frame",
]


@dataclass
class ScanNetPair:
    scene: str
    stem0: int
    stem1: int
    T_0to1: np.ndarray  # (4, 4), maps points from camera 0 to camera 1

    def image_paths(self, root: str) -> tuple[str, str]:
        return image_path(root, self.scene, self.stem0), image_path(root, self.scene, self.stem1)


def load_pairs(root: str) -> list[ScanNetPair]:
    """Load all 1500 evaluation pairs from ``<root>/test.npz``."""
    data = np.load(os.path.join(root, "test.npz"), allow_pickle=True)
    names = data["name"]
    rel_poses = data["rel_pose"]

    pairs = []
    for name, rel in zip(names, rel_poses):
        scene = f"scene{int(name[0]):04d}_{int(name[1]):02d}"
        T = np.eye(4)
        T[:3] = np.asarray(rel, dtype=np.float64).reshape(3, 4)
        pairs.append(ScanNetPair(scene=scene, stem0=int(name[2]), stem1=int(name[3]), T_0to1=T))
    return pairs


def image_path(root: str, scene: str, stem: int) -> str:
    return os.path.join(root, scene, "color", f"{stem}.jpg")


def load_color_intrinsics(root: str, scene: str) -> np.ndarray:
    """3x3 colour-camera intrinsics at native 1296x968 resolution."""
    path = os.path.join(root, scene, "intrinsic", "intrinsic_color.txt")
    return np.loadtxt(path)[:3, :3].astype(np.float64)


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio, image_resolution, patch_size):
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = max(patch_size, int(np.round(image_resolution / aspect_ratio / patch_size)) * patch_size)
    else:
        width = image_resolution
        height = max(patch_size, int(np.round(image_resolution * aspect_ratio / patch_size)) * patch_size)
    return height, width


def preprocess_transform(
    orig_w: int,
    orig_h: int,
    resolution: int,
    mode: str = "balanced",
    patch_size: int = 16,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
):
    """Replicate ``load_and_preprocess_images`` geometry for one image.

    Returns ``(crop_left, crop_top, crop_w, crop_h, target_w, target_h)`` describing
    the centre-crop (to bring the aspect ratio into ``[0.5, 2.0]``) followed by the
    resize. For ScanNet's 1296x968 the aspect ratio is already in range, so the crop
    is a no-op and this reduces to a pure resize.
    """
    aspect_ratio = orig_h / max(orig_w, 1)
    crop_left, crop_top, crop_w, crop_h = 0, 0, orig_w, orig_h

    if aspect_ratio < min_aspect_ratio:
        crop_w = min(orig_w, max(1, int(round(orig_h / min_aspect_ratio))))
        crop_left = max((orig_w - crop_w) // 2, 0)
    elif aspect_ratio > max_aspect_ratio:
        crop_h = min(orig_h, max(1, int(round(orig_w * max_aspect_ratio))))
        crop_top = max((orig_h - crop_h) // 2, 0)

    cropped_aspect = crop_h / max(crop_w, 1)
    if mode == "balanced":
        target_h, target_w = _balanced_target_shape(cropped_aspect, resolution, patch_size)
    else:
        target_h, target_w = _max_size_target_shape(cropped_aspect, resolution, patch_size)

    return crop_left, crop_top, crop_w, crop_h, target_w, target_h


def adjust_intrinsics(K: np.ndarray, transform) -> np.ndarray:
    """Map intrinsics through a ``preprocess_transform`` (crop then resize)."""
    crop_left, crop_top, crop_w, crop_h, target_w, target_h = transform
    K_new = K.astype(np.float64).copy()
    # Crop shifts the principal point.
    K_new[0, 2] -= crop_left
    K_new[1, 2] -= crop_top
    # Resize scales focal length and principal point anisotropically.
    sx = target_w / crop_w
    sy = target_h / crop_h
    K_new[0, 0] *= sx
    K_new[0, 2] *= sx
    K_new[1, 1] *= sy
    K_new[1, 2] *= sy
    return K_new


def gt_K_in_preprocessed_frame(
    K_orig: np.ndarray,
    orig_wh: tuple[int, int] = SCANNET_COLOR_WH,
    resolution: int = 512,
    mode: str = "balanced",
    patch_size: int = 16,
) -> np.ndarray:
    """Ground-truth colour intrinsics expressed in the model's preprocessed frame."""
    transform = preprocess_transform(orig_wh[0], orig_wh[1], resolution, mode, patch_size)
    return adjust_intrinsics(K_orig, transform)
