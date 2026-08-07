"""The two runnable matchers for this release: RoMa-Omega (dense) and VGGT-Omega
(zero-shot sparse). Both implement the BenchmarkMatcher interface consumed by
experiments/eval_sparse.py.
"""

from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image

from romaomega.device import device
from romaomega.types import BenchmarkMatcher
from romaomega.vggtroma import VGGTRoMa


class RoMaOmegaMatcher(BenchmarkMatcher):
    """Wraps a dense model (RoMa-Omega) exposing .match()/.sample()/.to_pixel_coordinates()
    behind the sparse-correspondence BenchmarkMatcher interface used by eval_sparse.py."""

    def __init__(self, model=None):
        self.model = model if model is not None else VGGTRoMa()

    @property
    def name(self) -> str:
        return "roma_omega"

    @torch.inference_mode()
    def match(self, img_A_path: str, img_B_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.model.match(img_A_path, img_B_path)
        matches_etc = self.model.sample(preds, 5000)
        matches = matches_etc[0]

        im_A = Image.open(img_A_path)
        im_B = Image.open(img_B_path)
        w1, h1 = im_A.size
        w2, h2 = im_B.size

        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H_A=h1, W_A=w1, H_B=h2, W_B=w2)
        kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
        return kpts1, kpts2

    @property
    def offset(self) -> float:
        return 0.5

    @property
    def get_type(self):
        return "dense"


class VGGTOmegaMatcher(BenchmarkMatcher):
    """Pairwise matcher built on VGGT-Omega's predicted depth + camera.

    Wraps the self-contained ``romaomega.matching`` module: a single forward pass gives
    per-frame depth + camera, from which ``match_depth_warp`` (the strongest, sub-pixel
    extractor) produces 2D-2D correspondences in the model's *preprocessed* image frame.
    ``match`` maps those back to original pixel coordinates so the benchmarks (WxBS, ...)
    can consume them directly.
    """

    def __init__(
        self,
        checkpoint: str = "vggtomega.pt",
        resolution: int = 512,
        mode: str = "balanced",
        stride: int = 8,
    ):
        import sys
        if "third_party/vggtomega" not in sys.path:
            sys.path.append("third_party/vggtomega")
        from romaomega.matching import run_pair, match_depth_warp, MatchConfig
        from romaomega.matching.model_io import load_model

        self._run_pair = run_pair
        self._match_depth_warp = match_depth_warp
        self._match_cfg = MatchConfig(stride=stride)
        self.resolution = resolution
        self.mode = mode
        self.patch_size = 16
        self.model = load_model(checkpoint, device=str(device))

    @property
    def name(self) -> str:
        return "vggt_omega"

    def _to_original_coords(self, pts_xy, img_path, Hc: int, Wc: int):
        """Map (x, y) from the padded preprocessed frame back to original pixels.

        Inverts the three preprocessing steps `load_and_preprocess_images` applies
        per image: aspect-ratio center-crop -> bicubic resize to a patch-multiple
        target -> center-pad to the batch's common size ``(Hc, Wc)``. Returns the
        original-pixel coordinates plus a boolean mask marking points that lie
        inside the real (non-padded) image region — matches landing in the white
        pad border are spurious and must be dropped.
        """
        from vggt_omega.utils.load_fn import (
            _crop_to_supported_aspect_ratio,
            _balanced_target_shape,
            _max_size_target_shape,
            _load_rgb_image,
        )

        img = _load_rgb_image(img_path)
        W0, H0 = img.size
        cropped = _crop_to_supported_aspect_ratio(img)
        cw, ch = cropped.size
        # Only one of width/height crop can trigger; the other offset is 0.
        left = max((W0 - cw) // 2, 0)
        top = max((H0 - ch) // 2, 0)

        aspect_ratio = ch / max(cw, 1)
        if self.mode == "balanced":
            th, tw = _balanced_target_shape(aspect_ratio, self.resolution, self.patch_size)
        else:
            th, tw = _max_size_target_shape(aspect_ratio, self.resolution, self.patch_size)

        pad_left = (Wc - tw) // 2
        pad_top = (Hc - th) // 2

        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        px, py = pts[:, 0], pts[:, 1]
        valid = (px >= pad_left) & (px <= pad_left + tw) & (py >= pad_top) & (py <= pad_top + th)
        x = (px - pad_left) * cw / tw + left
        y = (py - pad_top) * ch / th + top
        return np.stack([x, y], axis=-1), valid

    @torch.inference_mode()
    def match(self, img_A_path: str, img_B_path: str) -> Tuple[np.ndarray, np.ndarray]:
        pred = self._run_pair(
            self.model,
            (img_A_path, img_B_path),
            resolution=self.resolution,
            mode=self.mode,
            device=str(device),
        )
        res = self._match_depth_warp(pred, self._match_cfg)
        Hc, Wc = pred.hw
        mkpts0, valid0 = self._to_original_coords(res.mkpts0, img_A_path, Hc, Wc)
        mkpts1, valid1 = self._to_original_coords(res.mkpts1, img_B_path, Hc, Wc)
        keep = valid0 & valid1  # drop matches that touch either image's pad border
        return mkpts0[keep], mkpts1[keep]

    @property
    def offset(self) -> float:
        return 0.0

    @property
    def get_type(self):
        return "sparse"


MODEL_REGISTRY: dict[str, Callable[[], BenchmarkMatcher]] = {
    "roma_omega": RoMaOmegaMatcher,
    "vggt_omega": VGGTOmegaMatcher,
}
