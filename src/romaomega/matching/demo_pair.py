"""Single-pair demo: extract, visualize and (optionally) score 2D matches.

Given a checkpoint and two images -- either explicit paths or a ScanNet pair -- this
runs VGGT-Omega once and writes, for each matching method, a match visualization and
a feature-correlation map for a query point.

Usage:
    # explicit images
    python -m matching.demo_pair --checkpoint ckpt.pt --images a.jpg b.jpg --query 300 200

    # a ScanNet pair (also reports pose error vs. ground truth)
    # (use a real ScanNet-1500 test pair so the two views actually overlap)
    python -m matching.demo_pair --checkpoint ckpt.pt \
        --scene scene0721_00 --stem0 375 --stem1 480
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running both as a module (`python -m matching.demo_pair`) and as a plain
# script (`python matching/demo_pair.py`) by putting the repo root on the path.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matching import scannet
from matching.correspondences import MatchConfig, match_depth_warp, match_mutual_nn_3d
from matching.features import query_patch_correlation
from matching.pose_eval import estimate_relative_pose, relative_pose_error
from matching.visualize import draw_matches, visualize_correlation

MATCHERS = {"mutual_nn_3d": match_mutual_nn_3d, "depth_warp": match_depth_warp}


def _resolve_images(args):
    """Return ``(img0, img1, T_0to1_or_None, gt_scene_or_None)``."""
    if args.images:
        if len(args.images) != 2:
            raise SystemExit("--images expects exactly two paths")
        return args.images[0], args.images[1], None, None
    if args.scene and args.stem0 is not None and args.stem1 is not None:
        img0 = scannet.image_path(args.scannet_root, args.scene, args.stem0)
        img1 = scannet.image_path(args.scannet_root, args.scene, args.stem1)
        # Look up ground-truth relative pose for this pair, if present.
        T = None
        for pair in scannet.load_pairs(args.scannet_root):
            if pair.scene == args.scene and pair.stem0 == args.stem0 and pair.stem1 == args.stem1:
                T = pair.T_0to1
                break
        return img0, img1, T, args.scene
    raise SystemExit("Provide either --images A B or --scene/--stem0/--stem1")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", help="two image paths")
    p.add_argument("--scene", type=str)
    p.add_argument("--stem0", type=int)
    p.add_argument("--stem1", type=int)
    p.add_argument("--scannet-root", type=str, default="scannet_test_1500")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable-alignment", action="store_true")
    p.add_argument("--methods", type=str, default="mutual_nn_3d,depth_warp")
    p.add_argument("--query", type=float, nargs=2, default=None, help="query pixel (x y) for correlation")
    p.add_argument(
        "--feature-layer", type=int, default=11,
        help="cached aggregator layer for the correlation map (4/11/17/23 or -1); "
        "11 is the most discriminative for dense correspondence, 23 the most semantic",
    )
    p.add_argument("--ransac-px", type=float, default=0.5)
    p.add_argument("--out", type=str, default="demo_outputs/matching_demo")
    # MatchConfig knobs.
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--conf-thresh", type=float, default=1.0)
    p.add_argument("--nn-rel-thresh", type=float, default=0.02)
    p.add_argument("--depth-rel-thresh", type=float, default=0.05)
    p.add_argument("--cycle-px-thresh", type=float, default=3.0)
    args = p.parse_args()

    from matching.inference import run_pair
    from matching.model_io import load_model

    img0, img1, T_0to1, gt_scene = _resolve_images(args)
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.checkpoint, enable_alignment=args.enable_alignment, device=args.device)
    pred = run_pair(
        model, (img0, img1), resolution=args.resolution, device=args.device, feature_layer=args.feature_layer
    )
    height, width = pred.hw
    print(f"Preprocessed frame: {height}x{width}")

    cfg = MatchConfig(
        stride=args.stride,
        conf_thresh=args.conf_thresh,
        nn_rel_thresh=args.nn_rel_thresh,
        depth_rel_thresh=args.depth_rel_thresh,
        cycle_px_thresh=args.cycle_px_thresh,
    )

    K_gt = None
    if gt_scene is not None:
        K_orig = scannet.load_color_intrinsics(args.scannet_root, gt_scene)
        K_gt = scannet.gt_K_in_preprocessed_frame(K_orig, scannet.SCANNET_COLOR_WH, args.resolution)
        print(f"Predicted focal (px): fx0={pred.intrinsics[0][0,0]:.1f}  GT focal: fx={K_gt[0,0]:.1f}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        result = MATCHERS[method](pred, cfg, device=args.device)
        inlier_mask = None
        pose_msg = ""
        est = estimate_relative_pose(
            result.mkpts0, result.mkpts1, pred.intrinsics[0], pred.intrinsics[1], thresh_px=args.ransac_px
        )
        if est is not None:
            _, _, inlier_mask = est
            if T_0to1 is not None:
                R, t, _ = est
                t_err, R_err = relative_pose_error(T_0to1, R, t)
                pose_msg = f" | pose err (pred K): t={t_err:.2f} deg, R={R_err:.2f} deg"

        out_path = os.path.join(args.out, f"matches_{method}.png")
        draw_matches(
            pred.images, result.mkpts0, result.mkpts1, out_path,
            scores=result.scores, inlier_mask=inlier_mask, title=method,
        )
        print(f"[{method}] {len(result)} matches -> {out_path}{pose_msg}")
        if len(result) == 0:
            print(
                f"  (no matches: the two views may barely overlap. Use a real "
                f"ScanNet-1500 test pair, or relax thresholds, e.g. "
                f"--depth-rel-thresh 0.1 --nn-rel-thresh 0.05 --cycle-px-thresh 5)"
            )

    # Feature-correlation visualization for a query patch.
    query = tuple(args.query) if args.query is not None else (width / 2.0, height / 2.0)
    corr, q_xy, best_xy = query_patch_correlation(
        pred.patch_feats, query, patch_size=pred.patch_size, src=0, dst=1, upsample_hw=(height, width)
    )
    corr_path = os.path.join(args.out, "correlation.png")
    visualize_correlation(pred.images, q_xy, corr, corr_path, best_xy=best_xy, src=0, dst=1)
    print(f"[correlation] query {q_xy} -> best {best_xy}; saved {corr_path}")


if __name__ == "__main__":
    main()
