# `matching/` — 2D correspondences & pose AUC from VGGT-Omega predictions

Turns VGGT-Omega's raw per-frame **depth + camera** predictions for an image pair
into **2D–2D pixel correspondences**, visualizes them, and benchmarks them on the
ScanNet-1500 relative-pose task. Self-contained; imports the `vggt_omega` package
but does not modify it (and does not depend on the vendored `dust3r/`).

## Three correspondence extractors

The first two operate in the model's *preprocessed image frame* using its own
predicted extrinsics/intrinsics (OpenCV, camera-from-world), so they are internally
consistent. The third uses only encoder features and is geometry-independent.

1. **`match_mutual_nn_3d`** (geometry) — unproject both depth maps to a shared world
   frame and reciprocally (mutual nearest-neighbour) match the two point clouds with
   chunked GPU `torch.cdist`; keep pairs whose 3D distance is small relative to the
   scene scale. Coarse (limited to pixel-grid resolution) but assumption-free.
2. **`match_depth_warp`** (geometry) — warp each source pixel through its predicted
   depth+camera into the other view, accept it when the reprojected depth agrees with
   the other view's predicted depth, and keep only cycle-consistent (mutual) matches.
   Sub-pixel accurate; the strongest method.
3. **`match_features`** (descriptors) — mutual-NN matching of per-patch aggregator
   features from a chosen cached layer (4/11/17/23), optionally with a Lowe ratio
   test. Independent of the predicted geometry, but coarse (matches sit on the 16px
   patch grid) and much weaker — see the layer comparison below.

Because the two geometry matchers are derived purely from the predicted depth+camera,
the pose recovered from them essentially reproduces the model's *directly predicted*
pose (their AUC ≈ the `direct` baseline). They measure the **self-consistency** of the
3D predictions. `match_features` is the only extractor with an independent signal.

## Quick start

```python
from matching import run_pair, match_mutual_nn_3d, match_depth_warp, MatchConfig
from matching.model_io import load_model

model = load_model("vggt_omega_1b_512.pt")                 # VGGTOmega on CUDA
pred = run_pair(model, ("a.jpg", "b.jpg"), resolution=512) # single forward pass
res = match_depth_warp(pred, MatchConfig(stride=8))        # -> mkpts0, mkpts1, scores
```

## CLIs

```bash
# offline math check (synthetic sphere; no checkpoint / GPU needed)
python -m matching.run_eval --self-test

# single pair: match viz + feature-correlation viz (+ pose error if ScanNet)
# NOTE: use a real ScanNet-1500 test pair (see test.npz) so the views overlap;
# arbitrary frame numbers from the same scene may not, giving 0 matches.
python -m matching.demo_pair --checkpoint ckpt.pt \
    --scene scene0721_00 --stem0 375 --stem1 480 --query 300 200
# the correlation map reads from aggregator layer 11 by default (most discriminative
# for correspondence); try --feature-layer 4/17/23 to compare.

# full ScanNet-1500 AUC benchmark: geometry matchers + feature matcher per layer,
# under pred/GT intrinsics, plus the direct-pose baseline
python -m matching.run_eval --checkpoint ckpt.pt              # all 1500 pairs
python -m matching.run_eval --checkpoint ckpt.pt --max-pairs 50   # smoke test
# geometry only (skip the feature matcher):
python -m matching.run_eval --checkpoint ckpt.pt --feature-layers ""
```

`run_eval` reports pose AUC@{5,10,20} for every task under **predicted** and
**ground-truth** intrinsics, plus a **direct-pose baseline** (AUC of the pose the
model predicts directly, composed from its two extrinsics), and the mean match count.
Feature matches sit on the 16px patch grid, so they use a looser RANSAC threshold
(`--feature-ransac-px`, default 6) than the sub-pixel geometry matchers (`--ransac-px`,
default 0.5); `--feature-layers` (default `4,11,17,23`) selects which layers to score.

## Modules

| file | contents |
|------|----------|
| `geometry.py` | torch unproject / project / warp / sampling helpers |
| `inference.py` | `run_pair` → `PairPrediction` (depth, camera, world points, patch feats) |
| `correspondences.py` | the two matchers + `MatchConfig` / `MatchResult` |
| `features.py` | query-patch cosine correlation map |
| `visualize.py` | `draw_matches`, `visualize_correlation` (lazy matplotlib) |
| `pose_eval.py` | essential-matrix pose, `relative_pose_error`, `pose_auc`, direct baseline |
| `scannet.py` | ScanNet-1500 pairs / intrinsics + preprocessed-frame GT `K` |
| `model_io.py` | build + load a `VGGTOmega` checkpoint |
| `run_eval.py` / `demo_pair.py` | the two CLIs; `selftest.py` the offline check |

## Requirements & CUDA note

Needs `matplotlib` (viz) and `opencv-python` (pose) on top of the base deps; the
matchers themselves are pure torch. VGGT-Omega is **CUDA-only** — the torch build in
the environment you run the model from must match the GPU driver (e.g. a cu124/cu128
build for driver 550; a cu130 build reports `torch.cuda.is_available() == False`).
The `--self-test` path runs on CPU and needs neither a GPU nor a checkpoint.
