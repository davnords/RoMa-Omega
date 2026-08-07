"""Query-patch feature correlation for VGGT-Omega encoder features.

Given the per-patch aggregator features of an image pair (see
``inference.PairPrediction.patch_feats``), pick a query patch in one image and
compute its cosine similarity to every patch in the other image. Because the
aggregator mixes information across frames, this reflects the correspondence
signal the model actually carries.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["query_patch_correlation"]


def query_patch_correlation(
    patch_feats: np.ndarray,
    query_xy,
    patch_size: int = 16,
    src: int = 0,
    dst: int = 1,
    upsample_hw: tuple[int, int] | None = None,
    center: bool = True,
):
    """Cosine-similarity heatmap of a query patch against the other image.

    Args:
        patch_feats: ``(2, gh, gw, C)`` per-patch features for the pair.
        query_xy: ``(x, y)`` pixel in the ``src`` image; mapped to its patch.
        patch_size: pixels per patch (16 for VGGT-Omega).
        src / dst: which image holds the query / is searched.
        upsample_hw: if given, bilinearly upsample the ``(gh, gw)`` map to this
            ``(H, W)`` for overlaying on the full-resolution image.
        center: subtract the mean patch feature (over both images) before computing
            cosine similarity. The aggregator's features carry a large common-mode
            component that otherwise makes every patch look similar; centering removes
            it and sharpens the correlation peak. Strongly recommended.

    Returns:
        ``(corr_map, query_patch_xy, best_xy)`` where ``corr_map`` is a float array
        in ``[-1, 1]`` (shape ``(gh, gw)`` or ``upsample_hw``), ``query_patch_xy`` is
        the ``(x, y)`` pixel centre of the chosen query patch, and ``best_xy`` is the
        ``(x, y)`` pixel centre of the most similar patch in ``dst``.
    """
    feats = torch.as_tensor(np.asarray(patch_feats), dtype=torch.float32)
    fa = feats[src]  # (gh, gw, C)
    fb = feats[dst]
    gh, gw, _ = fa.shape

    if center:
        mean_feat = feats[[src, dst]].reshape(-1, feats.shape[-1]).mean(dim=0)
        fa = fa - mean_feat
        fb = fb - mean_feat

    qx = int(np.clip(int(query_xy[0]) // patch_size, 0, gw - 1))
    qy = int(np.clip(int(query_xy[1]) // patch_size, 0, gh - 1))

    q = F.normalize(fa[qy, qx], dim=-1)
    fb_norm = F.normalize(fb, dim=-1)
    corr = fb_norm @ q  # (gh, gw), cosine similarity in [-1, 1]

    best_idx = int(torch.argmax(corr))
    best_y, best_x = divmod(best_idx, gw)
    best_xy = (best_x * patch_size + patch_size / 2.0, best_y * patch_size + patch_size / 2.0)
    query_patch_xy = (qx * patch_size + patch_size / 2.0, qy * patch_size + patch_size / 2.0)

    if upsample_hw is not None:
        corr = F.interpolate(
            corr[None, None], size=upsample_hw, mode="bilinear", align_corners=False
        )[0, 0]

    return corr.cpu().numpy(), query_patch_xy, best_xy
