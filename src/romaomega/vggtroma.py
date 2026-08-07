from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from collections import OrderedDict
from typing import Literal


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import logging
from romaomega.device import device
from romaomega.features import FineFeatures
from romaomega.geometry import (
    bhwc_grid_sample,
    bhwc_interpolate,
    get_normalized_grid,
    prec_mat_from_prec_params,
    to_pixel,
    kde,
)
from romaomega.io import check_not_i16
from romaomega.types import Setting, ImageLike, Model, _load_cfg

from .matcher import Matcher
from .refiner import Refiners

logger = logging.getLogger(__name__)

# Public release checkpoint (EMA weights, matcher + refiners). The default
# VGGTRoMa() call downloads this automatically, so "full pipeline" eval just works.
ROMAOMEGA_CHECKPOINT_URL = (
    "https://github.com/davnords/storage/releases/download/romaomega/romaomega.pth"
)


def _strip_module_prefix(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """EMA state dicts (avg_weights.pth / the released checkpoint) key everything as
    "module.<...>" plus an "n_averaged" buffer; strip both here."""
    return {
        n.replace("module.", ""): w
        for n, w in weights.items()
        if n.startswith("module.")
    }


def _load_run_weights(run: Path, use_avg: bool) -> dict[str, torch.Tensor]:
    """Load a training run's weights (as written by experiments/train_refiners.py)."""
    if use_avg:
        weights = torch.load(run / "avg_weights.pth", map_location=device)
        return _strip_module_prefix(weights)
    return torch.load(run / "weights.pth", map_location=device)


def _enable_vggt_linalg():
    # VGGT's matcher hard-requires highest matmul precision; magma is the linalg
    # backend the dense pipeline uses.
    torch.set_float32_matmul_precision("highest")
    # torch.backends.cuda.preferred_linalg_library(torch._C._LinalgBackend.Magma)


def _interpolate_warp_and_confidence(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    zero_out_precision: bool,
):
    warp = bhwc_interpolate(
        warp.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    if zero_out_precision:
        # delta at 4 is absolute, and if we
        # for the second pass we therefore can't use first pred.
        # overlap is fine since it's relative to matcher pred.
        confidence[..., 1:] = 0.0

    confidence = bhwc_interpolate(
        confidence.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return warp, confidence


def _map_confidence(*, confidence: torch.Tensor, threshold: float | None):
    overlap = confidence[..., :1].sigmoid()
    if threshold is not None:
        overlap[overlap > threshold] = 1.0
    if confidence.shape[-1] >= 4:
        precision = prec_mat_from_prec_params(confidence[..., 1:4])
    else:
        # Coarse matcher emits overlap only (no precision params). Return a
        # placeholder so match()'s downstream .clone()/reshape stays valid;
        # callers needing real precision use the refiner outputs.
        precision = confidence.new_zeros(*confidence.shape[:-1], 2, 2)
    return overlap, precision


class VGGTRoMa(Model):
    @dataclass(frozen=True)
    class Cfg:
        matcher: Matcher.Cfg = Matcher.Cfg()
        refiners: Refiners.Cfg = Refiners.Cfg()
        refiner_features: FineFeatures.Cfg = FineFeatures.Cfg()
        anchor_width: int = 512
        anchor_height: int = 512
        setting: Setting = "base"
        stage: Literal["matcher", "refiners", "inference"] = "inference"
        compile: bool = False

    # settings
    H_lr: int
    W_lr: int
    H_hr: int | None
    W_hr: int | None
    bidirectional: bool
    threshold: float | None
    balanced_sampling: bool

    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        download_default = cfg is None
        if cfg is None:
            # Default to the released RoMa-Omega checkpoint (matcher + refiners),
            # running the full bidirectional hi-res pipeline.
            cfg = VGGTRoMa.Cfg(
                stage="inference",
                compile=os.environ.get("COMPILE", "1") == "1",
            )
            _enable_vggt_linalg()
            self.apply_setting("precise")
        else:
            self.bidirectional = False
        self.matcher = Matcher(cfg.matcher)
        self.cfg = cfg
        self.stage = cfg.stage
        self.anchor_width = cfg.anchor_width
        self.anchor_height = cfg.anchor_height
        if cfg.stage != "matcher":
            self.refiners = Refiners(cfg.refiners)
            self.refiner_features = FineFeatures(cfg.refiner_features)
        else:
            self.refiners = None
            self.refiner_features = None
        self.to(device)
        # self.name = cfg.name
        if download_default:
            weights = torch.hub.load_state_dict_from_url(
                ROMAOMEGA_CHECKPOINT_URL, map_location=device
            )
            weights = _strip_module_prefix(weights)
            missing, unexpected = self.load_state_dict(weights, strict=False)
            if missing or unexpected:
                logger.warning(
                    f"[{self.name}] load_state_dict missing={missing} "
                    f"unexpected={unexpected}"
                )
        if cfg.compile:
            logger.info(f"Compiling {self.name}...")
            self.compile()
        logger.info(f"{self.name} initialized.")

    @classmethod
    def load_trained(
        cls,
        run_path: str | Path,
        *,
        setting: Setting = "precise",
        compile: bool = False,
        use_avg: bool = True,
    ) -> "VGGTRoMa":
        """Build VGGT-RoMa (matcher + refiners) from your own training run dir
        (as written by experiments/train_refiners.py) and put it in full-pipeline
        inference mode. compile defaults to off: benchmark pairs vary in size,
        which otherwise triggers constant torch.compile recompiles."""
        _enable_vggt_linalg()
        run = Path(run_path)
        serialized = json.load(open(run / "cfg.json"))["model"]
        cfg = replace(
            _load_cfg(cls.Cfg, serialized, strict=False),
            stage="inference",
            compile=compile,
        )
        model = cls(cfg)
        weights = _load_run_weights(run, use_avg)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing or unexpected:
            logger.warning(
                f"[{model.name}] load_state_dict missing={missing} "
                f"unexpected={unexpected}"
            )
        model.apply_setting(setting)
        return model.eval()

    def apply_setting(self, setting: Setting):
        if setting in ["mega1500", "scannet1500", "wxbs", "satast"]:
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1024
            self.W_hr = 1024
            self.bidirectional = True
            self.threshold = 0.05
            self.balanced_sampling = True
        elif setting == "turbo":
            self.H_lr = 320
            self.W_lr = 320
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "fast":
            self.H_lr = 512
            self.W_lr = 512
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "base":
            self.H_lr = 640
            self.W_lr = 640
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "precise":
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1280
            self.W_hr = 1280
            self.bidirectional = True
            self.threshold = None
            self.balanced_sampling = True
        else:
            raise TypeError(f"Invalid setting: {setting}")

    def forward(
        self,
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("Float32 matmul precision must be set to highest")
        if self.training:
            assert not self.bidirectional, (
                "Bidirectional matching is not supported in training"
            )
        # assumes images between [0, 1]
        # init preds
        predictions = OrderedDict()

        # match feats
        matcher_output = self.matcher(
            img_A_lr, img_B_lr, bidirectional=self.bidirectional
        )
        # return matcher_output
        predictions["matcher"] = matcher_output
        if self.stage == "matcher":
            if not self.training:
                predictions["warp_AB"] = matcher_output["warp_AB"]
                predictions["confidence_AB"] = matcher_output["confidence_AB"]
                if self.bidirectional:
                    predictions["warp_BA"] = matcher_output["warp_BA"]
                    predictions["confidence_BA"] = matcher_output["confidence_BA"]
                else:
                    predictions["warp_BA"] = None
                    predictions["confidence_BA"] = None
            return predictions
        assert self.refiners is not None and self.refiner_features is not None, (
            "Refiners and refiner features must be initialized"
        )
        warp_AB, confidence_AB = (
            matcher_output["warp_AB"],
            matcher_output["confidence_AB"],
        )
        if self.bidirectional:
            warp_BA, confidence_BA = (
                matcher_output["warp_BA"],
                matcher_output["confidence_BA"],
            )
        else:
            warp_BA = None
            confidence_BA = None
        # refine warp, maybe twice (if hr is available)
        for stage, (img_A, img_B) in enumerate(
            zip([img_A_lr, img_A_hr], [img_B_lr, img_B_hr])
        ):
            if img_A is None or img_B is None:
                continue
            B, C, H, W = img_A.shape
            scale_factor = torch.tensor(
                (W / self.anchor_width, H / self.anchor_height), device=device
            )
            refiner_features_A = self.refiner_features(img_A)
            refiner_features_B = self.refiner_features(img_B)
            for patch_size_str, refiner in self.refiners.items():
                patch_size = int(patch_size_str)
                zero_out_precision = (
                    img_A_hr is not None and patch_size == 4 and stage == 1
                )
                warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                    warp=warp_AB,
                    confidence=confidence_AB,
                    H=H,
                    W=W,
                    patch_size=patch_size,
                    zero_out_precision=zero_out_precision,
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                        warp=warp_BA,
                        confidence=confidence_BA,
                        H=H,
                        W=W,
                        patch_size=patch_size,
                        zero_out_precision=zero_out_precision,
                    )

                f_patch_A = refiner_features_A[patch_size]
                f_patch_B = refiner_features_B[patch_size]
                refiner_output_AB = refiner(
                    f_A=f_patch_A,
                    f_B=f_patch_B,
                    prev_warp=warp_AB,
                    prev_confidence=confidence_AB,
                    scale_factor=scale_factor,
                )
                if self.bidirectional:
                    refiner_output_BA = refiner(
                        f_A=f_patch_B,
                        f_B=f_patch_A,
                        prev_warp=warp_BA,
                        prev_confidence=confidence_BA,
                        scale_factor=scale_factor,
                    )
                else:
                    refiner_output_BA = None
                if self.training:
                    # Supervise every refiner stage. The cascade detaches the
                    # warp between stages (refiner.py), so each stage's params
                    # only receive gradient from their own output. Rename to the
                    # keys the loss reads (warp_AB / confidence_AB).
                    predictions[f"refiner_{patch_size}_AB"] = {
                        "warp_AB": refiner_output_AB["warp"],
                        "confidence_AB": refiner_output_AB["confidence"],
                    }
                else:
                    predictions[f"refiner_{patch_size}_AB"] = refiner_output_AB
                    predictions[f"refiner_{patch_size}_BA"] = refiner_output_BA
                warp_AB, confidence_AB = (
                    refiner_output_AB["warp"],
                    refiner_output_AB["confidence"],
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = (
                        refiner_output_BA["warp"],
                        refiner_output_BA["confidence"],
                    )
            if not self.training:
                predictions["warp_AB"] = warp_AB
                predictions["confidence_AB"] = confidence_AB
                if self.bidirectional:
                    predictions["warp_BA"] = warp_BA
                    predictions["confidence_BA"] = confidence_BA
                else:
                    predictions["warp_BA"] = None
                    predictions["confidence_BA"] = None
        return predictions

    def _load_image(self, img_like: ImageLike) -> torch.Tensor:
        if isinstance(img_like, str) or isinstance(img_like, Path):
            img_pil = Image.open(img_like)
            check_not_i16(img_pil)
            img_pil = img_pil.convert("RGB")
            img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, Image.Image):
            img = torch.from_numpy(np.array(img_like)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, np.ndarray):
            assert img_like.shape[-1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = torch.from_numpy(img_like).permute(2, 0, 1).to(device)
        elif isinstance(img_like, torch.Tensor):
            assert img_like.shape[1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = img_like
        else:
            raise ValueError(f"Unsupported image type: {type(img_like)}")

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    @torch.inference_mode()
    def match(
        self,
        img_like_A: ImageLike,
        img_like_B: ImageLike,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        img_A = self._load_image(img_like_A)
        img_B = self._load_image(img_like_B)

        img_A_lr = F.interpolate(
            img_A,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        img_B_lr = F.interpolate(
            img_B,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        if self.H_hr is not None and self.W_hr is not None:
            img_A_hr = F.interpolate(
                img_A,
                size=(self.H_hr, self.W_hr),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            img_B_hr = F.interpolate(
                img_B,
                size=(self.H_hr, self.W_hr),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        else:
            img_A_hr = None
            img_B_hr = None

        preds = self(img_A_lr, img_B_lr, img_A_hr=img_A_hr, img_B_hr=img_B_hr)

        warp_AB = preds["warp_AB"]
        confidence_AB = preds["confidence_AB"]
        warp_BA = preds["warp_BA"]
        confidence_BA = preds["confidence_BA"]
        overlap_AB, precision_AB = _map_confidence(
            confidence=confidence_AB, threshold=self.threshold
        )
        if self.bidirectional:
            overlap_BA, precision_BA = _map_confidence(
                confidence=confidence_BA, threshold=self.threshold
            )
        else:
            overlap_BA = None
            precision_BA = None

        preds = {
            "warp_AB": warp_AB.clone(),
            "confidence_AB": confidence_AB.clone(),
            "overlap_AB": overlap_AB.clone(),
            "precision_AB": precision_AB.clone(),
            "warp_BA": warp_BA.clone() if warp_BA is not None else None,
            "confidence_BA": confidence_BA.clone()
            if confidence_BA is not None
            else None,
            "overlap_BA": overlap_BA.clone() if overlap_BA is not None else None,
            "precision_BA": precision_BA.clone() if precision_BA is not None else None,
        }
        return preds

    def sample(self, preds: dict[str, torch.Tensor], num_corresp: int):
        warp = preds["warp_AB"]
        overlap_AB = preds["overlap_AB"]
        precision_AB = preds["precision_AB"] if "precision_AB" in preds else None

        warp = warp[0]
        overlap_AB = overlap_AB[0].reshape(-1)
        if precision_AB is not None:
            precision_AB = precision_AB[0]

        H_A, W_A, two = warp.shape
        grid = get_normalized_grid(1, H_A, W_A)[0]
        matches_AB = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
        if self.bidirectional:
            overlap_BA = preds["overlap_BA"]
            warp_BA = preds["warp_BA"]

            precision_BA = preds["precision_BA"] if "precision_BA" in preds else None
            warp_BA = warp_BA[0]
            overlap_BA = overlap_BA[0]
            if precision_BA is not None:
                precision_BA = precision_BA[0]

            if precision_BA is not None and precision_AB is not None:
                precision_A = bhwc_grid_sample(
                    precision_BA[None].reshape(1, H_A, W_A, -1),
                    warp[None],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(H_A, W_A, 2, 2)
                precision_B = bhwc_grid_sample(
                    precision_AB[None].reshape(1, H_A, W_A, -1),
                    warp_BA[None],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(H_A, W_A, 2, 2)
                precision_fwd = torch.stack(
                    (precision_A, precision_AB), dim=-3
                ).reshape(-1, 2, 2, 2)
                precision_bwd = torch.stack(
                    (precision_BA, precision_B), dim=-3
                ).reshape(-1, 2, 2, 2)
                precision = torch.cat((precision_fwd, precision_bwd), dim=0)
            else:
                precision = None
            # let's hope H_A is equal to H_B
            grid = get_normalized_grid(1, H_A, W_A)[0]
            matches_BA = torch.cat((warp_BA, grid), dim=-1).reshape(-1, 4)
            overlap = torch.cat((overlap_AB.reshape(-1), overlap_BA.reshape(-1)), dim=0)
            matches = torch.cat((matches_AB, matches_BA), dim=0)
        else:
            matches = matches_AB
            overlap = overlap_AB.reshape(-1)
            precision = precision_AB.reshape(-1, 2, 2)

        expansion_factor = 4
        overlap = overlap * matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
        corresp_inds = torch.multinomial(
            overlap, expansion_factor * num_corresp, replacement=False
        )
        sampled_matches = matches[corresp_inds]
        sampled_overlap = overlap[corresp_inds]
        if precision is not None:
            sampled_precision = precision[corresp_inds]
        else:
            sampled_precision = None
        # return sampled_matches, sampled_confidence
        density = kde(sampled_matches)

        p = 1 / (density + 1)
        p[density < 10] = (
            1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        )
        balanced_samples = torch.multinomial(
            p, num_samples=min(num_corresp, len(sampled_overlap)), replacement=False
        )
        return (
            sampled_matches[balanced_samples],
            sampled_overlap[balanced_samples],
            sampled_precision[balanced_samples][:, 0]
            if sampled_precision is not None
            else None,
            sampled_precision[balanced_samples][:, 1]
            if sampled_precision is not None
            else None,
        )

    @classmethod
    def prec_map_coordinates(
        cls, precision: torch.Tensor, *, H_in: int, W_in: int, H_out: int, W_out: int
    ):
        W_ratio = W_in / W_out
        H_ratio = H_in / H_out
        ratio = torch.tensor([W_ratio, H_ratio], device=precision.device)
        precision = precision * ratio[None, :, None] * ratio[None, None, :]
        return precision

    @classmethod
    def to_pixel_coordinates(
        cls, warp: torch.Tensor, *, H_A: int, W_A: int, H_B: int, W_B: int
    ):
        return to_pixel(warp[..., :2], H=H_A, W=W_A), to_pixel(
            warp[..., 2:], H=H_B, W=W_B
        )

    @classmethod
    def match_keypoints(
        cls,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        warp: torch.Tensor,
        certainty: torch.Tensor,
        return_tuple: bool = True,
        return_inds: bool = False,
        max_dist: float = 0.005,
        cert_th: float = 0.0,
    ):
        x_AB = bhwc_grid_sample(
            warp,
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0]
        cert_AB = bhwc_grid_sample(
            certainty,
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0]
        D = torch.cdist(x_AB, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_AB > cert_th)
            * (D < max_dist),
            as_tuple=True,
        )

        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)
    def vis(self, preds: dict[str, torch.Tensor], img_like_A: ImageLike, img_like_B: ImageLike):
        warp_AB = preds["warp_AB"]
        overlap_AB = preds["overlap_AB"]
        # precision_AB = preds["precision_AB"] if "precision_AB" in preds else None
        warp_BA = preds["warp_BA"]
        overlap_BA = preds["overlap_BA"] if "overlap_BA" in preds else None
        # precision_BA = preds["precision_BA"] if "precision_BA" in preds else None
        img_A = self._load_image(img_like_A).permute(0, 2, 3, 1)
        img_B = self._load_image(img_like_B).permute(0, 2, 3, 1)
        img_A = bhwc_interpolate(img_A, size=(warp_AB.shape[1], warp_AB.shape[2]), mode="bicubic", align_corners=False, antialias=True)
        img_B = bhwc_interpolate(img_B, size=(warp_BA.shape[1], warp_BA.shape[2]), mode="bicubic", align_corners=False, antialias=True)
        img_A_transfer = bhwc_grid_sample(img_B, warp_AB, mode="bilinear", align_corners=False)
        img_B_transfer = bhwc_grid_sample(img_A, warp_BA, mode="bilinear", align_corners=False)
        white_img_A = torch.ones_like(img_A)
        white_img_B = torch.ones_like(img_B)
        vis_img_A = overlap_AB * img_A_transfer + (1 - overlap_AB) * white_img_A
        vis_img_B = overlap_BA * img_B_transfer + (1 - overlap_BA) * white_img_B

        return vis_img_A, vis_img_B

