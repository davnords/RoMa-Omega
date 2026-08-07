"""Single-pass inference for an image pair.

``run_pair`` runs the VGGT-Omega aggregator once and then the camera and depth
heads, reusing the model's own submodules (so we do not touch the released package
and do not pay for a second forward pass). It also reads the aggregator's final
cached layer to expose per-patch features for correlation visualization.

Everything downstream (matching, pose, visualization) consumes the returned
``PairPrediction``, whose fields live in the model's preprocessed image frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from .geometry import unproject_depth

__all__ = ["PairPrediction", "run_pair"]


@dataclass
class PairPrediction:
    """Model predictions for a pair, in the preprocessed image frame (numpy)."""

    images: np.ndarray  # (2, H, W, 3) float in [0, 1]
    depth: np.ndarray  # (2, H, W)
    depth_conf: np.ndarray  # (2, H, W)
    extrinsics: np.ndarray  # (2, 3, 4) camera-from-world, OpenCV
    intrinsics: np.ndarray  # (2, 3, 3)
    world_points: np.ndarray  # (2, H, W, 3)
    patch_feats: np.ndarray  # (2, gh, gw, C) — the selected feature_layer
    patch_size: int
    image_paths: tuple[str, str] | None = None
    patch_feats_by_layer: dict[int, np.ndarray] | None = None  # cached-layer -> (2, gh, gw, C)

    @property
    def hw(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[2]


@torch.no_grad()
def run_pair(
    model,
    image_paths,
    resolution: int = 512,
    mode: str = "balanced",
    device: str = "cuda",
    frames_chunk_size: int | None = 8,
    feature_layer: int = -1,
    feature_layers: list[int] | None = None,
) -> PairPrediction:
    """Run VGGT-Omega on two images and return a :class:`PairPrediction`.

    ``model`` is a ``VGGTOmega`` with the camera and depth heads enabled.
    ``feature_layer`` selects which cached aggregator layer populates
    ``patch_feats`` (used for correlation visualization). Only the DenseHead
    intermediate layers are cached (indices 4, 11, 17, 23); ``-1`` is the last (23).
    Earlier layers are usually more discriminative for dense correspondence.
    ``feature_layers`` additionally caches those layers in ``patch_feats_by_layer``
    (extracted in the same forward pass), for descriptor matching across layers.
    """
    image_paths = list(image_paths)
    if len(image_paths) != 2:
        raise ValueError(f"run_pair expects exactly 2 images, got {len(image_paths)}")

    images = load_and_preprocess_images(image_paths, image_resolution=resolution, mode=mode).to(device)
    imgs = images.unsqueeze(0)  # (1, 2, 3, H, W)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
        tokens_list, patch_token_start = model.aggregator(imgs)

    if model.camera_head is None or model.dense_head is None:
        raise ValueError("run_pair requires a model with camera and depth heads enabled.")

    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        pose_enc = model.camera_head(tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(
            tokens_list, images=imgs, patch_token_start=patch_token_start, frames_chunk_size=frames_chunk_size
        )

    height, width = imgs.shape[-2:]
    extrinsics, intrinsics = encoding_to_camera(pose_enc, (height, width))

    depth = depth[0, ..., 0].float()  # (2, H, W)
    depth_conf = depth_conf[0].float()  # (2, H, W)
    extrinsics = extrinsics[0].float()  # (2, 3, 4)
    intrinsics = intrinsics[0].float()  # (2, 3, 3)

    world_points = torch.stack(
        [unproject_depth(depth[i], intrinsics[i], extrinsics[i]) for i in range(2)], dim=0
    )

    # Per-patch features from the requested cached aggregator layer(s).
    patch_size = getattr(model.aggregator, "patch_size", 16)
    gh, gw = height // patch_size, width // patch_size
    cached_indices = [i for i, t in enumerate(tokens_list) if t is not None]
    resolved_layer = cached_indices[-1] if feature_layer == -1 else feature_layer

    layers_to_extract = {resolved_layer} | set(feature_layers or [])
    patch_feats_by_layer = {}
    for idx in sorted(layers_to_extract):
        tokens = tokens_list[idx]
        if tokens is None:
            raise ValueError(f"Aggregator did not cache layer {idx}; cached layers are {cached_indices}.")
        ft = tokens[0, :, patch_token_start:].float()  # (2, gh*gw, C)
        patch_feats_by_layer[idx] = ft.reshape(2, gh, gw, ft.shape[-1]).cpu().numpy()
    patch_feats = patch_feats_by_layer[resolved_layer]

    images_hwc = images.permute(0, 2, 3, 1).float().cpu().numpy()

    return PairPrediction(
        images=images_hwc,
        depth=depth.cpu().numpy(),
        depth_conf=depth_conf.cpu().numpy(),
        extrinsics=extrinsics.cpu().numpy(),
        intrinsics=intrinsics.cpu().numpy(),
        world_points=world_points.cpu().numpy(),
        patch_feats=patch_feats,  # already numpy
        patch_size=patch_size,
        image_paths=(image_paths[0], image_paths[1]),
        patch_feats_by_layer=patch_feats_by_layer,
    )
