"""ScanNet-1500 relative-pose AUC benchmark for VGGT-Omega match extraction.

For every test pair we run the model once, extract 2D correspondences with each
method, recover the relative pose via the essential matrix under both predicted and
ground-truth intrinsics, and accumulate the pose error. We also evaluate the pose
the model predicts directly (composed from its two extrinsics) as a reference.

Usage:
    python -m matching.run_eval --checkpoint path/to/vggt_omega_1b_512.pt
    python -m matching.run_eval --self-test          # offline math check, no model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Allow running both as a module (`python -m matching.run_eval`) and as a plain
# script (`python matching/run_eval.py`) by putting the repo root on the path.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matching import scannet
from matching.correspondences import (
    MatchConfig,
    match_depth_warp,
    match_features,
    match_mutual_nn_3d,
)
from matching.pose_eval import direct_relative_pose, estimate_relative_pose, pose_auc, relative_pose_error

MATCHERS = {"mutual_nn_3d": match_mutual_nn_3d, "depth_warp": match_depth_warp}
FAILED_POSE_ERROR = 180.0  # degrees, counts as a miss in the AUC


def _pose_error(est, T_0to1):
    if est is None:
        return FAILED_POSE_ERROR
    R, t, _ = est
    t_err, R_err = relative_pose_error(T_0to1, R, t)
    return max(t_err, R_err)


def evaluate(args) -> dict:
    from matching.inference import run_pair  # imported lazily so --self-test needs no model deps
    from matching.model_io import load_model

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    intrinsics_modes = ["pred", "gt"] if args.intrinsics == "both" else [args.intrinsics]
    thresholds = tuple(int(t) for t in args.thresholds.split(","))
    feature_layers = [int(x) for x in args.feature_layers.split(",") if x.strip()] if args.feature_layers else []

    cfg = MatchConfig(
        stride=args.stride,
        conf_thresh=args.conf_thresh,
        nn_rel_thresh=args.nn_rel_thresh,
        depth_rel_thresh=args.depth_rel_thresh,
        cycle_px_thresh=args.cycle_px_thresh,
        feature_ratio_thresh=args.feature_ratio_thresh,
    )

    model = load_model(args.checkpoint, enable_alignment=args.enable_alignment, device=args.device)

    pairs = scannet.load_pairs(args.scannet_root)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    # Task names: geometry matchers first, then the feature matcher per layer.
    task_names = list(methods) + [f"features@L{L}" for L in feature_layers]
    print(f"Evaluating {len(pairs)} pairs | tasks={task_names} | intrinsics={intrinsics_modes}")

    errors = {(name, intr): [] for name in task_names for intr in intrinsics_modes}
    errors[("direct", "pred")] = []
    match_counts = {name: [] for name in task_names}

    def run_matcher(name, pred):
        if name.startswith("features@L"):
            return match_features(pred, cfg, layer=int(name.split("L")[1]), device=args.device)
        return MATCHERS[name](pred, cfg, device=args.device)

    # Cache per-scene GT intrinsics in the preprocessed frame.
    gt_K_cache: dict[str, np.ndarray] = {}
    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    for idx, pair in enumerate(pairs):
        img0, img1 = pair.image_paths(args.scannet_root)
        if not (os.path.exists(img0) and os.path.exists(img1)):
            print(f"[skip] missing images for {pair.scene} {pair.stem0}/{pair.stem1}")
            continue

        try:
            pred = run_pair(
                model, (img0, img1), resolution=args.resolution, device=args.device,
                feature_layers=feature_layers,
            )
        except Exception as exc:  # keep the sweep going on a single bad pair
            print(f"[skip] {pair.scene} {pair.stem0}/{pair.stem1}: {exc}")
            continue

        if pair.scene not in gt_K_cache:
            K_orig = scannet.load_color_intrinsics(args.scannet_root, pair.scene)
            gt_K_cache[pair.scene] = scannet.gt_K_in_preprocessed_frame(
                K_orig, scannet.SCANNET_COLOR_WH, args.resolution
            )
        K_pred0, K_pred1 = pred.intrinsics[0], pred.intrinsics[1]
        K_gt = gt_K_cache[pair.scene]

        for name in task_names:
            result = run_matcher(name, pred)
            match_counts[name].append(len(result))
            # Feature matches sit on the 16px patch grid, so they need a looser RANSAC
            # threshold than the sub-pixel geometry matchers for a fair comparison.
            rpx = args.feature_ransac_px if name.startswith("features@L") else args.ransac_px
            for intr in intrinsics_modes:
                if intr == "pred":
                    K0, K1 = K_pred0, K_pred1
                else:
                    K0, K1 = K_gt, K_gt
                est = estimate_relative_pose(result.mkpts0, result.mkpts1, K0, K1, thresh_px=rpx)
                errors[(name, intr)].append(_pose_error(est, pair.T_0to1))

        R_d, t_d = direct_relative_pose(pred.extrinsics[0], pred.extrinsics[1])
        t_err, R_err = relative_pose_error(pair.T_0to1, R_d, t_d)
        errors[("direct", "pred")].append(max(t_err, R_err))

        if (idx + 1) % args.log_every == 0:
            elapsed = time.time() - start
            print(f"  {idx + 1}/{len(pairs)} pairs ({elapsed:.1f}s, {elapsed / (idx + 1):.2f}s/pair)")

    summary = _summarize(errors, match_counts, thresholds)
    _print_table(summary, thresholds)

    out_json = os.path.join(args.out, "auc_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "thresholds": list(thresholds),
                "n_pairs": len(pairs),
                "results": summary,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_json}")
    return summary


def _summarize(errors, match_counts, thresholds) -> dict:
    summary = {}
    for key, errs in errors.items():
        if not errs:
            continue
        method, intr = key
        aucs = pose_auc(errs, thresholds)
        row = {f"auc@{t}": aucs[t] for t in thresholds}
        row["n_pairs"] = len(errs)
        if method in match_counts and match_counts[method]:
            row["mean_matches"] = float(np.mean(match_counts[method]))
        summary[f"{method}/{intr}"] = row
    return summary


def _print_table(summary, thresholds):
    header = f"{'method/intrinsics':<24}" + "".join(f"AUC@{t:<7}" for t in thresholds) + f"{'mean_matches':>14}"
    print("\n" + header)
    print("-" * len(header))
    for name, row in summary.items():
        line = f"{name:<24}"
        line += "".join(f"{row[f'auc@{t}']:<10.2f}" for t in thresholds)
        mm = row.get("mean_matches")
        line += f"{mm:>14.1f}" if mm is not None else f"{'-':>14}"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="run the offline synthetic math check and exit")
    p.add_argument("--checkpoint", type=str, help="path to a VGGT-Omega checkpoint (.pt)")
    p.add_argument("--scannet-root", type=str, default="scannet_test_1500")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable-alignment", action="store_true", help="for the 256 text-aligned checkpoint")
    p.add_argument("--methods", type=str, default="mutual_nn_3d,depth_warp")
    p.add_argument(
        "--feature-layers", type=str, default="4,11,17,23",
        help="cached aggregator layers to evaluate the descriptor matcher on (comma-separated; empty to skip)",
    )
    p.add_argument("--feature-ratio-thresh", type=float, default=None, help="Lowe ratio test for feature matching")
    p.add_argument("--intrinsics", type=str, default="both", choices=["pred", "gt", "both"])
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--thresholds", type=str, default="5,10,20")
    p.add_argument("--ransac-px", type=float, default=0.5, help="RANSAC threshold (px) for geometry matchers")
    p.add_argument(
        "--feature-ransac-px", type=float, default=6.0,
        help="RANSAC threshold (px) for feature matchers (looser: matches sit on the 16px patch grid)",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default="demo_outputs/matching")
    # MatchConfig knobs.
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--conf-thresh", type=float, default=1.0)
    p.add_argument("--nn-rel-thresh", type=float, default=0.02)
    p.add_argument("--depth-rel-thresh", type=float, default=0.05)
    p.add_argument("--cycle-px-thresh", type=float, default=3.0)
    return p


def main():
    args = build_parser().parse_args()

    if args.self_test:
        from matching.selftest import run_self_test

        raise SystemExit(0 if run_self_test() else 1)

    if not args.checkpoint:
        build_parser().error("--checkpoint is required (or pass --self-test)")

    evaluate(args)


if __name__ == "__main__":
    main()
