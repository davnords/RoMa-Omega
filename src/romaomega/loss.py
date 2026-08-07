import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
import wandb
from romaomega.dense.romav3 import RoMaV3
from romaomega.types import Batch, ConfidenceMode
from romaomega.geometry import (
    get_normalized_grid,
    bhwc_grid_sample_with_nearest_exact_fallback,
    bhwc_interpolate_with_nearest_exact_fallback,
    compute_gt_warp_from_batch,
)
from romaomega.distrib import is_main_process
from romaomega.device import device
from romaomega.geometry import prec_mat_from_prec_params


def _attn_nll_loss_list(
    *,
    attn_logits_list_AB: list[torch.Tensor] | torch.Tensor,
    warp: torch.Tensor,
    overlap: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(attn_logits_list_AB, torch.Tensor):
        attn_logits_list_AB = [attn_logits_list_AB]
    return sum(
        attn_nll_loss(
            attn_logits_AB=attn_logits_AB,
            warp=warp,
            overlap=overlap,
            sample_mask=sample_mask,
        )
        for attn_logits_AB in attn_logits_list_AB
    )


def attn_nll_loss(
    *,
    attn_logits_AB: torch.Tensor,
    warp: torch.Tensor,
    overlap: torch.Tensor,
    reduction: Literal["batch", "pair", "all"] = "all",
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H_patch_A, W_patch_A, H_patch_B, W_patch_B = attn_logits_AB.shape
    grid_patch_A = get_normalized_grid(B, H_patch_A, W_patch_A)
    warp_patch = bhwc_grid_sample_with_nearest_exact_fallback(
        x=warp,
        grid=grid_patch_A,
        mode="bilinear",
        align_corners=False,
        rtol=None,
        atol=None,
    )
    overlap_patch = bhwc_grid_sample_with_nearest_exact_fallback(
        x=overlap,
        grid=grid_patch_A,
        mode="bilinear",
        align_corners=False,
        rtol=None,
        atol=None,
    )
    grid_patch_B = get_normalized_grid(B, H_patch_B, W_patch_B)
    bool_overlap_patch = overlap_patch > 0.5
    attn_logits_AB = attn_logits_AB.reshape(
        B, H_patch_A, W_patch_A, H_patch_B * W_patch_B
    )
    attn_log_probs_AB = F.log_softmax(attn_logits_AB, dim=-1)
    closest_bin_B = (
        (warp_patch[..., None, None, :] - grid_patch_B[:, None, None])
        .reshape(B, H_patch_A, W_patch_A, H_patch_B * W_patch_B, 2)
        .norm(dim=-1)
        .argmin(dim=-1, keepdim=True)
    )
    within_frame = warp_patch.abs().amax(dim=-1, keepdim=True) < 1.0
    covisible = bool_overlap_patch.logical_and(within_frame)
    if sample_mask is not None:
        covisible = covisible.logical_and(sample_mask[:, None, None, None])
    best_match_log_probs = torch.take_along_dim(
        attn_log_probs_AB, closest_bin_B, dim=-1
    )

    if reduction == "batch":
        nll_loss = -best_match_log_probs[covisible].mean()
    elif reduction == "pair":
        nlls = [-best_match_log_probs[b][covisible[b]].sum() for b in range(B)]
        if len(nlls) == 0:
            return torch.tensor(0.0, device=device)
        nll_loss = sum(nlls) / len(nlls)
    elif reduction == "all":
        nll_loss = -best_match_log_probs[covisible].sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return nll_loss


def compute_precision_loss(
    *,
    P_params: torch.Tensor,
    residual: torch.Tensor,
    bool_overlap_cmp: torch.Tensor,
    B: int,
    H: int,
    W: int,
    use_prec_loss_threshold: float,
) -> torch.Tensor:
    assert P_params.shape[-1] == 3, "P_params must have last dim 3 (a, b, c)"
    H_warp, W_warp = residual.shape[1:3]
    # mul with H_warp to get the scale in pixels
    pixel_residuals = (
        residual * torch.tensor([W, H], device=device)[None, None, None] / 2
    )
    prec_nll = gaussian_nll_2d(pixel_residuals, P_params)
    epe = pixel_residuals.norm(dim=-1, keepdim=True)
    bool_use_prec_loss = bool_overlap_cmp.logical_and(epe < use_prec_loss_threshold)
    prec_loss = prec_nll[bool_use_prec_loss].sum() / (B * H_warp * W_warp)
    return prec_loss


def gaussian_nll_2d(
    residuals: torch.Tensor,
    P_params: torch.Tensor,
) -> torch.Tensor:
    P = prec_mat_from_prec_params(P_params)
    quad = (
        torch.einsum(
            "bhwd, bhwdc, bhwc -> bhw", residuals.detach(), P, residuals.detach()
        )
        / 2
    )
    # P = [[a, b], [b, c]] is PD (built from a Cholesky factor in the refiner),
    # so det = a*c - b^2 > 0 and log|det| = log(det). Use the 2x2 closed form
    # instead of torch.linalg.slogdet: it is cheaper and avoids the batched
    # linalg CUDA kernel.
    a, b, c = P_params[..., 0], P_params[..., 1], P_params[..., 2]
    det = (a * c - b * b).clamp_min(1e-12)
    logdet_term = -0.5 * torch.log(det) + math.log(2 * math.pi)

    # logdet_term = -0.5 * torch.linalg.slogdet(P).logabsdet + math.log(2 * math.pi)   
    nll = quad + logdet_term
    if nll.isnan().any():
        print("nll is nan", flush=True)
    return nll[..., None]


class Loss(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        pow: float = 0.5
        eps: float = 1e-3
        attn_nll: bool = True
        confidence_mode: ConfidenceMode = "covis"
        precision_pred: bool = True
        use_prec_loss_threshold: float = 8.0  # in pixels
        warp_losses_high_quality_only: bool = False
        warp_loss_weight: float = 1.0
        attn_nll_weight: float = 1.0
        overlap_loss_weight: float = 0.01  # like roma
        prec_loss_weight: float = 0.001
        rel_depth_error_threshold: float = 0.05
        gt_warp_fwd_bwd_error_threshold: float = 5e-3
        matcher_smooth_target: bool = True
        gt_overlap_dilation: int = 0
        # warp_cycle_interp_atol: float | None = None
        # depth_cycle_interp_rtol: float | None = None
        # warp_down_interp_atol: float | None = None
        use_cycle_error_for_depth_covis: bool = False
        local_neighbourhood_size: int = 1

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.confidence_mode == "covis", "confidence_mode must be covis"
        assert self.cfg.rel_depth_error_threshold is not None, (
            "rel_depth_error_threshold must be provided"
        )

    def forward(
        self, *, batch: Batch, model: RoMaV3, step: int
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        B, three, H_A, W_A = batch.img_A.shape
        B, three, H_B, W_B = batch.img_B.shape

        high_quality_mask = torch.tensor(
            [q == "high" for q in batch.quality], device=device
        )
        warp_loss_sample_mask = (
            high_quality_mask
            if self.cfg.warp_losses_high_quality_only
            else torch.ones_like(high_quality_mask, dtype=torch.bool)
        )

        gt_warp = compute_gt_warp_from_batch(
            batch,
            depth_error_threshold=self.cfg.rel_depth_error_threshold,
            flow_error_threshold=self.cfg.gt_warp_fwd_bwd_error_threshold,
            local_neighbourhood_size=self.cfg.local_neighbourhood_size,
        )
        warp_AB = gt_warp.warp
        overlap = gt_warp.covis
        valid = gt_warp.valid

        bool_overlap = (overlap > 0.5).logical_and(batch.mask_AB)

        H, W = warp_AB.shape[1:3]
        assert three == 3, "Image must have 3 channels"
        predictions = model(batch.img_A, batch.img_B)
        loss = torch.tensor(0.0, device=device)
        stats = {}
        for key, output in predictions.items():
            warp_pred, confidence_pred = output["warp_AB"], output["confidence_AB"]
            B, H_warp, W_warp, two = warp_pred.shape
            scale = H_A / H_warp
            assert two == 2, "Warp must have 2 channels"
            if "attn_AB" in output and self.cfg.attn_nll:
                B, H_patch_A, W_patch_A, _, _ = output["attn_logits_AB"][0].shape
                nll_loss = _attn_nll_loss_list(
                    attn_logits_list_AB=output["attn_logits_AB"],
                    warp=warp_AB,
                    overlap=overlap,
                    sample_mask=warp_loss_sample_mask,
                ) / (B * H_patch_A * W_patch_A)
                if is_main_process():
                    wandb.log(
                        {f"nll-loss-{key}": nll_loss},
                        step=step,
                    )
                loss = loss + self.cfg.attn_nll_weight * nll_loss
            if H_warp != H and W_warp != W:
                # TODO: this should be same as nearest-exact for images divisible by strides?
                warp_cmp = bhwc_interpolate_with_nearest_exact_fallback(
                    warp_AB,
                    (H_warp, W_warp),
                    mode="bilinear",
                    align_corners=False,
                    rtol=None,
                    # atol=self.cfg.warp_down_interp_atol,
                )
                # TODO: bilinear with 0.5
                overlap_cmp = bhwc_interpolate_with_nearest_exact_fallback(
                    overlap,
                    (H_warp, W_warp),
                    mode="bilinear",
                    align_corners=False,
                    rtol=None,
                    # atol=self.cfg.warp_down_interp_atol,
                )
                bool_overlap_cmp = overlap_cmp > 0.5
                valid_cmp = bhwc_interpolate_with_nearest_exact_fallback(
                    valid.float(),
                    (H_warp, W_warp),
                    mode="nearest-exact",
                    rtol=None,
                    # atol=self.cfg.warp_down_interp_atol,
                ).bool()
            else:
                warp_cmp = warp_AB
                overlap_cmp = overlap
                bool_overlap_cmp = bool_overlap
            warp_err_mask = bool_overlap_cmp

            residual = warp_pred - warp_cmp
            sqr_warp_err = (residual**2).sum(dim=-1, keepdim=True)
            warp_err = sqr_warp_err.detach() ** 0.5
            # follows roma
            a = self.cfg.pow
            cs = self.cfg.eps
            if key == "matcher" and self.cfg.matcher_smooth_target:
                cs = cs * 16
            if key == "refiner":
                cs = cs * scale
            robust_warp_err = cs**a * (sqr_warp_err / cs**2 + 1**2) ** (a / 2)
            if self.cfg.gt_overlap_dilation > 0:
                dilated_warp_err_mask = (
                    F.max_pool2d(
                        warp_err_mask[:, None, ..., 0].float(),
                        kernel_size=self.cfg.gt_overlap_dilation,
                        stride=1,
                        padding=self.cfg.gt_overlap_dilation // 2,
                    )[:, 0, ..., None]
                    .bool()
                    .logical_and(valid_cmp)
                )
                warp_err_mask[high_quality_mask] = dilated_warp_err_mask[
                    high_quality_mask
                ]

            warp_err_mask = warp_err_mask.logical_and(
                warp_loss_sample_mask[:, None, None, None]
            )
            warp_loss = robust_warp_err[warp_err_mask].sum() / (B * H_warp * W_warp)
            loss += self.cfg.warp_loss_weight * warp_loss

            overlap_pred = confidence_pred[..., :1]
            if self.cfg.precision_pred and (confidence_pred.shape[-1] == 4):
                bool_overlap_cmp_warp_losses = bool_overlap_cmp.logical_and(
                    warp_loss_sample_mask[:, None, None, None]
                )
                prec_loss = compute_precision_loss(
                    P_params=confidence_pred[..., 1:4],
                    residual=residual,
                    bool_overlap_cmp=bool_overlap_cmp_warp_losses,
                    B=B,
                    H=H_B,
                    W=W_B,
                    use_prec_loss_threshold=self.cfg.use_prec_loss_threshold,
                )
                loss = loss + self.cfg.prec_loss_weight * prec_loss
                if is_main_process():
                    wandb.log(
                        {f"precision-nll-loss-{key}": prec_loss.item()}, step=step
                    )
            else:
                overlap_pred = confidence_pred
            overlap_loss = F.binary_cross_entropy_with_logits(overlap_pred, overlap_cmp)
            if is_main_process():
                bool_overlap_cmp_warp_losses = bool_overlap_cmp.logical_and(
                    warp_loss_sample_mask[:, None, None, None]
                )
                epe_vals = warp_err[bool_overlap_cmp_warp_losses]
                train_epe = (
                    epe_vals.mean()
                    if epe_vals.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                wandb.log(
                    {f"train-epe-{key}": train_epe.detach().item()},
                    step=step,
                )
                wandb.log(
                    {
                        f"overlap-loss-{key}": overlap_loss.item(),
                        f"warp-loss-{key}": warp_loss.item(),
                    },
                    step=step,
                )
            loss += self.cfg.overlap_loss_weight * overlap_loss
            stats[key] = {
                "robust_warp_err": robust_warp_err.detach(),
                "warp_err": warp_err.detach(),
                "overlap_loss": overlap_loss.detach(),
                "warp_loss": warp_loss.detach(),
                "bool_overlap_cmp": bool_overlap_cmp.detach(),
                "warp_cmp": warp_cmp.detach(),
                "warp_pred": warp_pred.detach(),
            }
        return loss, stats
