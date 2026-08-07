"""Match and feature-correlation visualization (headless matplotlib)."""

from __future__ import annotations

import os

import numpy as np


def _get_plt():
    """Lazy matplotlib import with a headless backend, so importing this module (and
    running matching / evaluation) does not require matplotlib to be installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Accept ``(H, W, 3)`` or ``(3, H, W)`` in [0, 1] or [0, 255] -> ``(H, W, 3)`` uint8."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _side_by_side(img0: np.ndarray, img1: np.ndarray, gap: int = 10):
    """Concatenate two images horizontally on a white gap. Returns ``(canvas, x_offset1)``."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    height = max(h0, h1)
    canvas = np.full((height, w0 + gap + w1, 3), 255, dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap : w0 + gap + w1] = img1
    return canvas, w0 + gap


def draw_matches(
    images,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    out_path: str,
    scores: np.ndarray | None = None,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 200,
    title: str | None = None,
    dpi: int = 150,
):
    """Draw correspondence lines between an image pair and save to ``out_path``.

    Lines are green/red for RANSAC inliers/outliers when ``inlier_mask`` is given,
    otherwise coloured by ``scores`` (viridis). At most ``max_draw`` lines are drawn.
    """
    plt = _get_plt()
    img0 = _to_uint8_image(images[0])
    img1 = _to_uint8_image(images[1])
    canvas, x_off = _side_by_side(img0, img1)

    mkpts0 = np.asarray(mkpts0, dtype=np.float64).reshape(-1, 2)
    mkpts1 = np.asarray(mkpts1, dtype=np.float64).reshape(-1, 2)
    n = len(mkpts0)

    if n > max_draw:
        sel = np.random.default_rng(0).choice(n, size=max_draw, replace=False)
    else:
        sel = np.arange(n)

    if inlier_mask is not None:
        inlier_mask = np.asarray(inlier_mask).astype(bool)
        colors = np.where(inlier_mask[sel], "lime", "red")
    elif scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        cmap = plt.get_cmap("viridis")
        rng = (np.nanmin(scores), np.nanmax(scores)) if len(scores) else (0.0, 1.0)
        denom = max(rng[1] - rng[0], 1e-9)
        colors = [cmap((scores[i] - rng[0]) / denom) for i in sel]
    else:
        colors = ["yellow"] * len(sel)

    fig, ax = plt.subplots(figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi), dpi=dpi)
    ax.imshow(canvas)
    ax.axis("off")

    for j, i in enumerate(sel):
        x0, y0 = mkpts0[i]
        x1, y1 = mkpts1[i]
        c = colors[j] if isinstance(colors, list) else colors[j]
        ax.plot([x0, x1 + x_off], [y0, y1], color=c, linewidth=0.5, alpha=0.7)
        ax.scatter([x0, x1 + x_off], [y0, y1], color=c, s=2)

    label = title or ""
    n_inl = int(inlier_mask.sum()) if inlier_mask is not None else None
    label += f"  ({n} matches" + (f", {n_inl} inliers" if n_inl is not None else "") + ")"
    ax.set_title(label.strip(), fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


def visualize_correlation(
    images,
    query_xy,
    corr_map: np.ndarray,
    out_path: str,
    best_xy=None,
    src: int = 0,
    dst: int = 1,
    dpi: int = 150,
):
    """Show the query location (left) and its correlation heatmap (right)."""
    plt = _get_plt()
    img_src = _to_uint8_image(images[src])
    img_dst = _to_uint8_image(images[dst])
    corr = np.asarray(corr_map, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    axes[0].imshow(img_src)
    axes[0].scatter([query_xy[0]], [query_xy[1]], c="red", s=80, marker="x", linewidths=2)
    axes[0].set_title(f"query @ ({int(query_xy[0])}, {int(query_xy[1])}) in image {src}")
    axes[0].axis("off")

    axes[1].imshow(img_dst)
    heat = axes[1].imshow(
        corr, cmap="turbo", alpha=0.6, vmin=-1.0, vmax=1.0,
        extent=(0, img_dst.shape[1], img_dst.shape[0], 0),
    )
    if best_xy is not None:
        axes[1].scatter([best_xy[0]], [best_xy[1]], c="red", s=80, marker="+", linewidths=2)
    axes[1].set_title(f"feature correlation in image {dst}")
    axes[1].axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04, label="cosine similarity")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path
