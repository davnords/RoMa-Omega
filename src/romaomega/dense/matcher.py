import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal
from romaomega.geometry import get_normalized_grid, bhwc_interpolate
from einops import einsum
from romaomega.device import device
from romaomega.vit import ViTModel, vit_from_name
from romaomega.types import HeadType, MatcherStyle
from romaomega.heads import Head
from romaomega.dpt import DPTHead


def _normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True)


def _cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    f_A = _normalize(f_A, dim=-1)
    f_B = _normalize(f_B, dim=-1)
    res = einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")
    return res


def _compute_match_embeddings_list(
    *,
    f_list_A: list[torch.Tensor],
    f_list_B: list[torch.Tensor],
    pos_emb_grid: torch.Tensor,
    temp: float,
    B: int,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    attn_logits_list_AB = []
    attn_list_AB = []
    match_emb_list_AB = []
    for f_A, f_B in zip(f_list_A, f_list_B):
        attn_logits_AB, attn_AB, match_emb = _compute_match_embeddings(
            f_A=f_A,
            f_B=f_B,
            pos_emb_grid=pos_emb_grid,
            temp=temp,
            B=B,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
        )
        attn_logits_list_AB.append(attn_logits_AB)
        attn_list_AB.append(attn_AB)
        match_emb_list_AB.append(match_emb)
    return attn_logits_list_AB, attn_list_AB, match_emb_list_AB


def _compute_match_embeddings(
    *,
    f_A: torch.Tensor,
    f_B: torch.Tensor,
    pos_emb_grid: torch.Tensor,
    temp: float,
    B: int,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
) -> torch.Tensor:
    attn_logits_AB = (1 / temp * _cosine_similarity(f_A, f_B)).reshape(
        B, H_A * W_A, H_B * W_B
    )
    attn_AB = torch.softmax(attn_logits_AB, dim=2)
    attn_AB = attn_AB.reshape(B, H_A, W_A, H_B, W_B)

    match_emb = einsum(
        attn_AB, pos_emb_grid, "B H_A W_A H_B W_B, B H_B W_B D -> B H_A W_A D"
    )
    attn_logits_AB = attn_logits_AB.reshape(B, H_A, W_A, H_B, W_B)
    # TODO: hmmmmmmmmmmmmmm
    return attn_logits_AB, attn_AB, match_emb


def _compute_head_preds(
    *,
    f_list_A: list[torch.Tensor],
    match_emb_list_AB: list[torch.Tensor],
    f_mv_list_A: list[torch.Tensor],
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    head: DPTHead,
) -> torch.Tensor:
    B, H_A, W_A, D = f_mv_list_A[0].shape
    if f_list_A[-1].shape != f_mv_list_A[0].shape:
        for i in range(len(f_list_A)):
            f_list_A[i] = bhwc_interpolate(
                f_list_A[i], (H_A, W_A), mode="bicubic", align_corners=False
            )
    head_input = []
    n = len(f_list_A)
    m = len(f_mv_list_A)
    if max(n, m) > 2 and len(match_emb_list_AB) == 1:
        match_emb_list_AB = 2 * match_emb_list_AB
    k = len(match_emb_list_AB)
    for i in range(1, max(n, m) + 1):
        f_mono = f_list_A[n - i] if n - i >= 0 else 0
        f_mv = f_mv_list_A[m - i] if m - i >= 0 else 0
        match_emb = match_emb_list_AB[k - i] if k - i >= 0 else 0
        head_input.append(f_mono + f_mv + match_emb)
    head_input = list[torch.Tensor](reversed(head_input))
    warp_and_confidence = head(head_input, img_A=img_A, img_B=img_B)
    warp = warp_and_confidence[:, :, :, :2]
    confidence = warp_and_confidence[:, :, :, 2:]
    return warp, confidence


class _LinearHead(nn.Module):
    def __init__(self, dim: int, warp_dim: int, confidence_dim: int):
        super().__init__()
        self.head = nn.Linear(dim, warp_dim + confidence_dim)

    def forward(
        self, head_input: list[torch.Tensor], img_A: torch.Tensor, img_B: torch.Tensor
    ):
        warp_and_confidence = self.head(head_input[-1])
        return warp_and_confidence


class Matcher(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        mv_vit: ViTModel = "vit_base"
        mv_vit_use_rope: bool = True
        mv_vit_position_mode: Literal["same"] = "same"
        mv_vit_attention_mode: Literal["alternating"] = "alternating"
        mv_vit_init_weights: str | None = None
        head: HeadType = "dpt-no-pos"
        # NOTE: 0.2 in RoMa
        temp: float = 0.1
        # NOTE: 8 in RoMa
        scale: float = 1
        dim: int = 1024
        warp_dim: int = 2
        confidence_dim: int = 1
        num_feature_layers: int = 2
        mv_feature_layers: list[int] = field(default_factory=lambda: [11])
        feat_dim: int = 1024
        pos_emb_dim: int = 1024
        enable_amp: bool = True
        style: MatcherStyle = "romav3"
        upsampler_A: str | None = None
        down_ratio: int = 4
        # ufm uses pos embedings for view B and no attn
        pos_embed_rope_rescale_coords: float | None = None
        early_pos_embed_f_B: bool = False

    def __init__(self, cfg: Cfg):
        super().__init__()
        # https://github.com/facebookresearch/dinov3/blob/54694f7627fd815f62a5dcc82944ffa6153bbb76/dinov3/eval/depth/models/encoder.py#L33
        # 2 5 8 11
        assert 11 in cfg.mv_feature_layers, "11 is the first feature layer for dinov3"
        self.cfg = cfg
        if cfg.mv_vit_init_weights:
            if cfg.mv_vit_init_weights == "mum_vitb":
                assert cfg.mv_vit == "vit_base_mum_decoder", (
                    "Mum init should also use mum encoder"
                )
            else:
                assert cfg.mv_vit == "vit_large_half_dinov3", (
                    "mv_vit_init_weights only supports vit_large_half_dinov3"
                )
                assert cfg.num_feature_layers == 1, (
                    "mv_vit_init_weights only supports 1 feature layer"
                )
        if cfg.feat_dim != cfg.dim:
            self.feat_dim_to_dim = nn.Linear(cfg.feat_dim, cfg.dim)
        else:
            self.feat_dim_to_dim = nn.Identity()
        if cfg.pos_emb_dim != cfg.dim:
            self.pos_emb_dim_to_dim = nn.Linear(cfg.pos_emb_dim, cfg.dim)
        else:
            self.pos_emb_dim_to_dim = nn.Identity()
        omega = 2 * torch.pi * torch.randn(cfg.dim // 2, 2)
        self.omega = nn.Buffer(omega)
        self.scale = nn.Buffer(torch.tensor(cfg.scale))

        self.temp = nn.Buffer(torch.tensor(cfg.temp))
        self.mv_vit = vit_from_name(
            cfg.mv_vit,
            device=device,
            in_dim=cfg.feat_dim * cfg.num_feature_layers,
            out_dim=cfg.dim,
            multiview=True,
            use_rope=cfg.mv_vit_use_rope,
            mv_position_mode=cfg.mv_vit_position_mode,
            mv_attention_mode=cfg.mv_vit_attention_mode,
            pos_embed_rope_rescale_coords=cfg.pos_embed_rope_rescale_coords,
        )
        assert (len(self.mv_vit.blocks) - 1) in cfg.mv_feature_layers, (
            f"mv_feature_layers must contain the last feature layer of the mv_vit. Got {cfg.mv_feature_layers} and expected {(len(self.mv_vit.blocks) - 1)}"
        )
        if cfg.mv_vit_init_weights:
            if cfg.mv_vit_init_weights == "mum_vitb":
                from romaomega.vit.mum_vit import load_and_format_mum_ckpt

                ckpt = load_and_format_mum_ckpt()
            else:
                ckpt = torch.load(cfg.mv_vit_init_weights, map_location=device)
            self.mv_vit.load_state_dict(ckpt, strict=False)

        upsampled_patch_size = 8 if cfg.upsampler_A is not None else 16
        self.head = Head(
            head_type=cfg.head,
            dim=cfg.dim,
            warp_dim=cfg.warp_dim,
            confidence_dim=cfg.confidence_dim,
            down_ratio=cfg.down_ratio,
            patch_size=upsampled_patch_size,
        )
        if cfg.upsampler_A is not None:
            if cfg.upsampler_A == "loftup":
                from romaomega.upsamplers import LoftUp

                self.upsampler_A = LoftUp(cfg.dim)
            elif cfg.upsampler_A == "bilinear":
                self.upsampler_A = lambda x, img: bhwc_interpolate(
                    x, img.shape[-2:], mode="bilinear", align_corners=False
                )
            elif cfg.upsampler_A == "bicubic":
                self.upsampler_A = lambda x, img: bhwc_interpolate(
                    x, img.shape[-2:], mode="bicubic", align_corners=False
                )
            else:
                raise ValueError(f"Upsampler {cfg.upsampler_A} not implemented")

    def forward(
        self,
        f_list_A: list[torch.Tensor],
        f_list_B: list[torch.Tensor],
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        bidirectional: bool,
    ):
        preds = {}
        f_A = torch.cat(f_list_A, dim=-1)
        f_B = torch.cat(f_list_B, dim=-1)
        B, H_A, W_A, D_feat = f_A.shape
        B, H_B, W_B, D_feat = f_B.shape
        assert D_feat == self.cfg.feat_dim * self.cfg.num_feature_layers, (
            f"Feature dimension mismatch with cfg.feat_dim {self.cfg.feat_dim} and cfg.num_feature_layers {self.cfg.num_feature_layers}"
            + f"Got {D_feat=} and expected {self.cfg.feat_dim * self.cfg.num_feature_layers=}"
        )
        x = get_normalized_grid(B, H_B, W_B)
        x_emb = nn.functional.linear(
            x.reshape(B, H_B * W_B, 2), self.scale * self.omega
        ).reshape(B, H_B, W_B, -1)
        pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)
        if self.cfg.early_pos_embed_f_B:
            pos_emb_grid_B = pos_emb_grid.repeat(
                1, 1, 1, D_feat // pos_emb_grid.shape[-1]
            )
            f_B = f_B + pos_emb_grid_B

        with torch.autocast(device.type, torch.bfloat16, enabled=self.cfg.enable_amp):
            assert self.mv_vit is not None
            f_mv_list_AB = self.mv_vit.get_intermediate_layers(
                torch.stack((f_A, f_B), dim=1), n=self.cfg.mv_feature_layers
            )
            f_mv_list_A = [f[:, 0] for f in f_mv_list_AB]
            f_mv_list_B = [f[:, 1] for f in f_mv_list_AB]
            if self.cfg.upsampler_A is not None:
                H_img, W_img = img_A.shape[-2:]
                down_img_A = F.interpolate(
                    img_A,
                    (H_img // 8, W_img // 8),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                f_mv_list_A = [self.upsampler_A(f, down_img_A) for f in f_mv_list_A]

            B, H_A, W_A, _ = f_mv_list_A[0].shape

            # assert H_A == H_B and W_A == W_B, "H_A and W_A must be equal to H_B and W_B"
            attn_logits_list_AB, attn_list_AB, match_emb_list_AB = (
                _compute_match_embeddings_list(
                    f_list_A=f_mv_list_A,
                    f_list_B=f_mv_list_B,
                    pos_emb_grid=pos_emb_grid,
                    temp=self.temp,
                    B=B,
                    H_A=H_A,
                    W_A=W_A,
                    H_B=H_B,
                    W_B=W_B,
                )
            )
            if bidirectional:
                attn_logits_list_BA, attn_list_BA, match_emb_list_BA = (
                    _compute_match_embeddings_list(
                        f_list_A=f_mv_list_B,
                        f_list_B=f_mv_list_A,
                        pos_emb_grid=pos_emb_grid,
                        temp=self.temp,
                        B=B,
                        H_A=H_B,
                        W_A=W_B,
                        H_B=H_A,
                        W_B=W_A,
                    )
                )
            f_list_A = [
                self.feat_dim_to_dim(
                    f.reshape(B, H_A * W_A, self.cfg.feat_dim)
                ).reshape(B, H_A, W_A, self.cfg.dim)
                for f in f_list_A
            ]
            match_emb_list_AB = [
                self.pos_emb_dim_to_dim(
                    e.reshape(B, H_A * W_A, self.cfg.pos_emb_dim)
                ).reshape(B, H_A, W_A, self.cfg.dim)
                for e in match_emb_list_AB
            ]
        warp_AB, confidence_AB = _compute_head_preds(
            f_list_A=f_list_A,
            match_emb_list_AB=match_emb_list_AB,
            f_mv_list_A=f_mv_list_A,
            img_A=img_A,
            img_B=img_B,
            head=self.head,
        )

        if bidirectional:
            with torch.autocast(
                device.type, torch.bfloat16, enabled=self.cfg.enable_amp
            ):
                f_list_B = [
                    self.feat_dim_to_dim(
                        f.reshape(B, H_B * W_B, self.cfg.feat_dim)
                    ).reshape(B, H_B, W_B, self.cfg.dim)
                    for f in f_list_B
                ]
                match_emb_list_BA = [
                    self.pos_emb_dim_to_dim(
                        e.reshape(B, H_B * W_B, self.cfg.pos_emb_dim)
                    ).reshape(B, H_B, W_B, self.cfg.dim)
                    for e in match_emb_list_BA
                ]

            warp_BA, confidence_BA = _compute_head_preds(
                f_list_A=f_list_B,
                match_emb_list_AB=match_emb_list_BA,
                f_mv_list_A=f_mv_list_B,
                img_A=img_B,
                img_B=img_A,
                head=self.head,
            )
        else:
            # match_emb_BA = None
            attn_list_BA = None
            attn_logits_list_BA = None
            warp_BA = None
            confidence_BA = None

        preds["attn_logits_AB"] = attn_logits_list_AB
        preds["attn_AB"] = attn_list_AB
        preds["warp_AB"] = warp_AB
        preds["confidence_AB"] = confidence_AB

        preds["attn_logits_BA"] = attn_logits_list_BA
        preds["attn_BA"] = attn_list_BA
        preds["warp_BA"] = warp_BA
        preds["confidence_BA"] = confidence_BA
        return preds
