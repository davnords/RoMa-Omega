from dataclasses import dataclass
from typing import Literal
import torch
import torch.utils.data
import tqdm

from romaomega.device import device
from romaomega.geometry import (
    warp_and_depth_consistency_from_depths,
    bhwc_interpolate,
    warp_and_overlap_from_flows,
)
from romaomega.dense.romav3 import RoMaV3
from romaomega.types import Batch, ConfidenceMode
from romaomega.loss import attn_nll_loss
from romaomega.random import set_seed


def _reduce_metric(
    x: torch.Tensor, mask: torch.Tensor, reduction: Literal["batch", "pair", "all"]
):
    if reduction == "batch":
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        return x[mask].float().mean()
    elif reduction == "pair":
        # thr of 50 covisible follows UFM
        means = [
            x[b][mask[b]].float().mean()
            for b in range(x.shape[0])
            if mask[b].sum() > 50
        ]
        if len(means) == 0:
            return torch.tensor(0.0, device=x.device)
        return sum(means) / len(means)
    elif reduction == "all":
        return x.float().sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def _count_covisible(mask: torch.Tensor, reduction: Literal["batch", "pair", "all"]):
    B = mask.shape[0]
    if reduction == "batch":
        return torch.tensor(1.0, device=mask.device)
    elif reduction == "pair":
        # thr of 50 covisible follows UFM
        return mask.reshape(B, -1).sum(dim=1).ge(50).float().mean()
    elif reduction == "all":
        return mask.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class DenseBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        batch_size: int = 8
        rel_depth_error_threshold: float = 0.05
        warp_cycle_error: float = 5e-3
        num_workers: int = 8
        confidence_mode: ConfidenceMode = "covis"
        attn_nll: bool = True
        seed: int | None = None
        reduction: Literal["batch", "pair", "all"] = "batch"

    # Set by subclasses
    dataset: torch.utils.data.Dataset
    prefix: str

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        if cfg.seed is not None:
            set_seed(cfg.seed)
        assert cfg.confidence_mode == "covis", "confidence_mode must be covis"

    @torch.no_grad()
    def __call__(self, model: RoMaV3, step: int):
        model.eval()
        nll_tot = torch.zeros(1)
        epe_tot = torch.zeros(1)
        pck_1_tot = torch.zeros(1)
        pck_3_tot = torch.zeros(1)
        pck_5_tot = torch.zeros(1)
        pck_32_tot = torch.zeros(1)
        bias_x_tot = torch.zeros(1)
        bias_y_tot = torch.zeros(1)
        covis_pixels_tot = torch.zeros(1)
        bias_covis_pixels_tot = torch.zeros(1)
        B = self.cfg.batch_size
        # Use a fixed generator so shuffle order is repeatable across runs
        if self.cfg.seed is not None:
            # TODO: needed?
            def _worker_init_fn(worker_id):
                set_seed(self.cfg.seed + worker_id)

            set_seed(self.cfg.seed)
            # assert self.cfg.num_workers == 0, "Not deterministic if num_workers > 0"
            generator = torch.Generator()
            generator.manual_seed(self.cfg.seed)
        else:
            _worker_init_fn = None
            generator = None
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=B,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            generator=generator,
            worker_init_fn=_worker_init_fn,
            collate_fn=Batch.collate,
        )

        pck_10s = []
        for idx, batch in enumerate[Batch](
            pbar := tqdm.tqdm(dataloader, miniters=10, desc=f"{self.prefix} eval")
        ):
            batch = batch.to(device)
            H, W = batch.img_A.shape[-2:]
            with torch.inference_mode():
                predictions = model(batch.img_A, batch.img_B)
            warp_pred = predictions["warp_AB"]
            if batch.source[0] == "flow":
                result_fwd_bwd = warp_and_overlap_from_flows(
                    flow_AB=batch.flow_AB,
                    flow_BA=batch.flow_BA,
                    error_threshold=self.cfg.warp_cycle_error,
                )
                warp = result_fwd_bwd.warp
                overlap = result_fwd_bwd.covis.logical_and(batch.mask_AB).float()
            else:
                result = warp_and_depth_consistency_from_depths(
                    depth_A=batch.depth_A,
                    depth_B=batch.depth_B,
                    K_A=batch.K_A,
                    K_B=batch.K_B,
                    T_AB=batch.T_AB,
                    rel_depth_error_threshold=self.cfg.rel_depth_error_threshold,
                )
                warp = result.warp
                # Select overlap based on mode: covis for "covis" mode, valid for "frame" mode
                if self.cfg.confidence_mode == "covis":
                    overlap = result.covis
                elif self.cfg.confidence_mode == "frame":
                    overlap = result.valid
                else:
                    # For "positive" mode, fall back to valid (closest equivalent)
                    overlap = result.valid
                overlap = overlap.float()
            if self.cfg.attn_nll and "attn_AB" in predictions.get("matcher", {}):
                attn_logits_AB = predictions["matcher"]["attn_logits_AB"]
                nll = attn_nll_loss(
                    attn_logits_AB=attn_logits_AB[-1],
                    warp=warp,
                    overlap=overlap,
                    reduction=self.cfg.reduction,
                )
            else:
                nll = torch.tensor(0.0, device=device)
            warp_pred = bhwc_interpolate(warp_pred, size=(H, W))
            WH = torch.tensor((W, H), device=device)
            residuals = (warp_pred - warp).mul(WH[None, None, None] / 2)
            error = residuals.norm(dim=-1, keepdim=True)
            bool_overlap = overlap > 0.5
            # error_for_consistent_depth = error[bool_overlap]
            residual_mask = residuals.norm(dim=-1, keepdim=True) < 1
            residuals_x = residuals[..., :1]
            residuals_y = residuals[..., 1:]

            epe = _reduce_metric(error, bool_overlap, self.cfg.reduction)
            # covis_bce = covis_bce.sum()
            pck_1 = _reduce_metric(error < 1.0, bool_overlap, self.cfg.reduction)
            pck_3 = _reduce_metric(error < 3.0, bool_overlap, self.cfg.reduction)
            pck_5 = _reduce_metric(error < 5.0, bool_overlap, self.cfg.reduction)

            B = error.shape[0]
            for b in range(B):
                mask_b = bool_overlap[b]
                if mask_b.sum() == 0:
                    continue
                correct = (error[b] < 10.0)[mask_b].float().mean()
                pck_10s.append(correct.item())

            pck_32 = _reduce_metric(error < 32.0, bool_overlap, self.cfg.reduction)
            covis_pixels = _count_covisible(bool_overlap, self.cfg.reduction)
            bias_covis_pixels = _count_covisible(
                bool_overlap * residual_mask, self.cfg.reduction
            )
            bias_x = _reduce_metric(
                residuals_x, bool_overlap * residual_mask, self.cfg.reduction
            )
            bias_y = _reduce_metric(
                residuals_y, bool_overlap * residual_mask, self.cfg.reduction
            )
            # if epe/covis_pixels > 10:
            #     print(f"epe > 10: {epe/covis_pixels}, {covis_pixels/(B*H*W)}")

            (
                epe_tot,
                pck_1_tot,
                pck_3_tot,
                pck_5_tot,
                pck_32_tot,
                bias_x_tot,
                bias_y_tot,
                # covis_bce_tot,
                covis_pixels_tot,
                bias_covis_pixels_tot,
                nll_tot,
            ) = (
                epe_tot + epe.cpu(),
                pck_1_tot + pck_1.cpu(),
                pck_3_tot + pck_3.cpu(),
                pck_5_tot + pck_5.cpu(),
                pck_32_tot + pck_32.cpu(),
                bias_x_tot + bias_x.cpu(),
                bias_y_tot + bias_y.cpu(),
                # covis_bce_tot + covis_bce.cpu(),
                covis_pixels_tot + covis_pixels.cpu(),
                bias_covis_pixels_tot + bias_covis_pixels.cpu(),
                nll_tot + nll.cpu(),
            )
            pbar.set_postfix(
                epe=f"{(epe_tot.item() / covis_pixels_tot.item()):.2f}",
                bias_x=f"{(bias_x_tot.item() / bias_covis_pixels_tot.item()):.2f}",
                bias_y=f"{(bias_y_tot.item() / bias_covis_pixels_tot.item()):.2f}",
            )
        epe = epe_tot.item() / covis_pixels_tot.item()
        bias_x = bias_x_tot.item() / bias_covis_pixels_tot.item()
        bias_y = bias_y_tot.item() / bias_covis_pixels_tot.item()
        nll = nll_tot.item() / covis_pixels_tot.item()
        # covis_bce = covis_bce_tot.item() / covis_pixels_tot.item()
        pck_1 = pck_1_tot.item() / covis_pixels_tot.item()
        pck_3 = pck_3_tot.item() / covis_pixels_tot.item()
        pck_5 = pck_5_tot.item() / covis_pixels_tot.item()
        pck_32 = pck_32_tot.item() / covis_pixels_tot.item()

        return {
            f"{self.prefix}_epe": epe,
            f"{self.prefix}_bias_x": bias_x,
            f"{self.prefix}_bias_y": bias_y,
            f"{self.prefix}_nll": nll,
            f"{self.prefix}_pck_1": pck_1,
            f"{self.prefix}_pck_3": pck_3,
            f"{self.prefix}_pck_5": pck_5,
            f"{self.prefix}_pck_32": pck_32,
        }
