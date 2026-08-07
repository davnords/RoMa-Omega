import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from romaomega.device import device
from romaomega.features import Descriptor
from romaomega.geometry import get_normalized_grid
from romaomega.heads import Head
from romaomega.types import HeadType
from romaomega.vit.block import SelfAttentionBlock
from romaomega.vit.rope import RopePositionEmbedding


class _FrozenRef:
    """Holds a frozen module *without* registering it as an ``nn.Module``
    submodule.

    Assigning this (a plain object, not a Module/Parameter/Tensor) as an
    attribute means PyTorch never tracks the wrapped module, so it is excluded
    from ``parameters()`` (optimizer), ``state_dict()`` (checkpoints, including
    EMA ``avg_weights``) and ``.to()/.train()`` propagation. ``__deepcopy__``
    returns the same instance so ``AveragedModel`` (which deep-copies the model)
    shares the weights instead of duplicating ~1B frozen params in memory.
    """

    def __init__(self, module: nn.Module):
        self.module = module

    def __deepcopy__(self, memo):
        return self


def _normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True)


def _cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    f_A = _normalize(f_A, dim=-1)
    f_B = _normalize(f_B, dim=-1)
    res = einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")
    return res


def _compute_match_embeddings(
    *,
    f_A: torch.Tensor,
    f_B: torch.Tensor,
    pos_emb_grid: torch.Tensor,
    temp: torch.Tensor,
    B: int,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft-argmax style match embedding.

    Correlates every location in A against every location in B (cosine
    similarity), softmaxes over B, then aggregates B's positional embedding to
    get, for each location in A, an embedding of where it points to in B.
    """
    attn_logits_AB = (1 / temp * _cosine_similarity(f_A, f_B)).reshape(
        B, H_A * W_A, H_B * W_B
    )
    attn_AB = torch.softmax(attn_logits_AB, dim=2)
    attn_AB = attn_AB.reshape(B, H_A, W_A, H_B, W_B)
    match_emb = einsum(
        attn_AB, pos_emb_grid, "B H_A W_A H_B W_B, B H_B W_B D -> B H_A W_A D"
    )
    attn_logits_AB = attn_logits_AB.reshape(B, H_A, W_A, H_B, W_B)
    return attn_logits_AB, attn_AB, match_emb


def _load_vggt_omega(
    checkpoint: str | None, embed_dim: int, patch_size: int
) -> nn.Module:
    """Build VGGT-Omega with only the aggregator (no camera/depth/text heads).

    Loading the full checkpoint with strict=False populates the aggregator and
    silently ignores the disabled head weights.
    """
    if "third_party/vggtomega" not in sys.path:
        sys.path.append("third_party/vggtomega")
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(
        patch_size=patch_size,
        embed_dim=embed_dim,
        enable_camera=False,
        enable_depth=False,
        enable_alignment=False,
    )
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        missing, _ = model.load_state_dict(state_dict, strict=False)
        assert not missing, f"missing keys loading {checkpoint}: {missing}"
    return model


class Matcher(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        # VGGT-Omega (frozen) acts as the multiview encoder/transformer.
        vggt_checkpoint: str = "vggtomega.pt"
        vggt_embed_dim: int = 1024
        vggt_frozen: bool = True
        patch_size: int = 16
        # Resolution VGGT-Omega runs at, regardless of the outer pipeline
        # resolution. The released checkpoint is trained at 512 (~32x32 tokens),
        # so we resize inputs to ~vggt_resolution**2 area (AR preserved) here.
        vggt_resolution: int = 512
        head: HeadType = "dpt-no-pos"
        # NOTE: 0.2 in RoMa
        temp: float = 0.1
        # NOTE: 8 in RoMa
        scale: float = 1
        # Dimensionality of the Fourier match embedding. When it equals work_dim
        # the match embedding is added to the features with no projection (like
        # romav3's identity pos_emb_dim_to_dim); otherwise it is projected to
        # work_dim by a learned linear.
        pos_emb_dim: int = 1024
        warp_dim: int = 2
        confidence_dim: int = 1
        down_ratio: int = 4
        enable_amp: bool = True
        # Trainable cross-view transformer applied to the deepest VGGT feature
        # *before* correlation, so the matched descriptors are trainable (the
        # backbone stays frozen). Mirrors romav3's mv_vit recipe: project the wide
        # VGGT feature DOWN to global_dim, reason over all tokens of both images at
        # that narrow width for num_global_layers layers, then project back UP and
        # add as a residual. Head count is set so head dim == 64.
        num_global_layers: int = 12
        # Working width of the cross-view transformer (romav3's vit_base is 768).
        global_dim: int = 768
        # Attention pattern of the cross-view refiner, mirroring romav3's mv_vit:
        #   "global"      -> every layer attends over the union of both views'
        #                    tokens (the original VGGT-RoMa behaviour).
        #   "alternating" -> even layers attend globally (cross-view), odd layers
        #                    attend framewise (each view only sees its own tokens),
        #                    exactly like the dense matcher's mv_attention_mode.
        global_attention_mode: Literal["global", "alternating"] = "global"
        # Apply axial RoPE on the framewise layers of the alternating refiner
        # (the global/cross-view layers get no rope, like the dense mv_vit, since
        # positions aren't defined across the concatenated views). No effect when
        # global_attention_mode == "global".
        global_use_rope: bool = True

        # --- Ablation ladder knobs (defaults reproduce the VGGT-RoMa model) ---
        # Feature backbone: frozen VGGT-Omega (default) or a DINOv3 descriptor
        # (the romav3 backbone), for the romav3 -> VGGT-RoMa ablation.
        backbone: Literal["vggt", "dinov3"] = "vggt"
        # DINOv3 descriptor config (only used when backbone == "dinov3").
        descriptor: Descriptor.Cfg = field(default_factory=Descriptor.Cfg)
        # Width at which correlation + the DPT head run. None -> the backbone's
        # feature width (VGGT: 2048). Set to 1024 to match romav3 (down-projects
        # the refined feature, like romav3's mv_vit output_projector).
        work_dim: int | None = 1024
        # Which backbone cached layers feed the DPT head (indices into the layer
        # list, shallow->deep). None -> all (VGGT: 4 distinct scales). Pass 2
        # indices (e.g. (2, 3)) for the romav3-style 2-scale head that the DPT
        # duplicates into its 4 fusion levels.
        head_feature_layers: tuple[int, ...] | None = field(
            default_factory=lambda: (2, 3)
        )

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg

        # --- Frozen backbone: produces a list of per-view feature layers ---
        # Held in _FrozenRef so the frozen weights stay out of the optimizer,
        # checkpoints and EMA. Access via self.vggt / self.descriptor.
        self._vggt = None
        self._descriptor = None
        if cfg.backbone == "vggt":
            # Each cached aggregator layer is cat([frame_tokens, global_tokens]).
            self.feat_dim = 2 * cfg.vggt_embed_dim
            vggt = _load_vggt_omega(
                cfg.vggt_checkpoint, cfg.vggt_embed_dim, cfg.patch_size
            ).to(device)
            if cfg.vggt_frozen:
                vggt.eval()
                for p in vggt.parameters():
                    p.requires_grad_(False)
            self._vggt = _FrozenRef(vggt)
        elif cfg.backbone == "dinov3":
            descriptor = Descriptor(cfg.descriptor)
            descriptor.eval()
            for p in descriptor.parameters():
                p.requires_grad_(False)
            self.feat_dim = 1024
            self._descriptor = _FrozenRef(descriptor)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        # Width at which correlation + the DPT head run. Down-projects the (wide)
        # backbone feature when work_dim < feat_dim, like romav3's mv_vit output
        # projector; Identity when equal so the VGGT default is unchanged.
        self.work_dim = cfg.work_dim if cfg.work_dim is not None else self.feat_dim
        self.to_work = (
            nn.Linear(self.feat_dim, self.work_dim)
            if self.work_dim != self.feat_dim
            else nn.Identity()
        )

        # Fourier positional embedding (over view B's grid) for match embeddings,
        # plus a learnable projection into the working feature space so the warp
        # hint can be added to the features at a sensible scale.
        omega = 2 * torch.pi * torch.randn(cfg.pos_emb_dim // 2, 2)
        self.omega = nn.Buffer(omega)
        self.scale = nn.Buffer(torch.tensor(cfg.scale))
        self.temp = nn.Buffer(torch.tensor(cfg.temp))
        self.match_emb_to_dim = (
            nn.Identity()
            if cfg.pos_emb_dim == self.work_dim
            else nn.Linear(cfg.pos_emb_dim, self.work_dim)
        )

        # Cross-view transformer refiner over the deepest backbone feature.
        # romav3-style: down-project (feat_dim -> global_dim), reason at the narrow
        # width, up-project back, residual. The up-projection is zero-initialized
        # so the refiner is an identity at step 0 (correlation starts from the raw
        # frozen VGGT feature) and only learns to *add* improvements.
        #
        # num_global_layers == 0 is the "no trainable refinement" ablation: skip
        # the refiner entirely so it adds zero dead params. _refine_deepest then
        # short-circuits to passing the raw feature through.
        if cfg.num_global_layers > 0:
            num_heads = cfg.global_dim // 64
            self.global_in = nn.Linear(self.feat_dim, cfg.global_dim)
            self.global_refiner = nn.ModuleList(
                [
                    SelfAttentionBlock(dim=cfg.global_dim, num_heads=num_heads)
                    for _ in range(cfg.num_global_layers)
                ]
            )
            self.global_out = nn.Linear(cfg.global_dim, self.feat_dim)
            nn.init.zeros_(self.global_out.weight)
            nn.init.zeros_(self.global_out.bias)

            # Axial RoPE for the framewise layers of the alternating refiner. Built
            # once and re-evaluated per (H, W) in _refine_deepest. Defaults match
            # the mv_vit (base=100, normalize_coords="separate").
            self.global_rope = (
                RopePositionEmbedding(
                    cfg.global_dim, num_heads=num_heads, device=device
                )
                if cfg.global_attention_mode == "alternating" and cfg.global_use_rope
                else None
            )
        else:
            self.global_in = None
            self.global_refiner = None
            self.global_out = None
            self.global_rope = None

        self.head = Head(
            head_type=cfg.head,
            dim=self.work_dim,
            warp_dim=cfg.warp_dim,
            confidence_dim=cfg.confidence_dim,
            down_ratio=cfg.down_ratio,
            patch_size=cfg.patch_size,
        )

    @property
    def vggt(self) -> nn.Module:
        assert self._vggt is not None, "vggt backbone is not in use"
        return self._vggt.module

    @property
    def descriptor(self) -> nn.Module:
        assert self._descriptor is not None, "dinov3 backbone is not in use"
        return self._descriptor.module

    def _resize_for_vggt(self, img: torch.Tensor) -> torch.Tensor:
        """Disabled: run VGGT at the full pipeline resolution. Input sides must be
        a multiple of patch_size (VGGT will error otherwise)."""
        return img

    def _vggt_features(
        self, img_A: torch.Tensor, img_B: torch.Tensor
    ) -> tuple[list[torch.Tensor], int, int]:
        """Run the frozen aggregator and return the cached layers' patch tokens.

        Returns a list (one entry per cached layer, shallow->deep) of tensors of
        shape (B, 2, H_p, W_p, feat_dim), plus the patch grid size. Images are
        resized to VGGT's training resolution first.
        """
        B = img_A.shape[0]
        img_A = self._resize_for_vggt(img_A)
        img_B = self._resize_for_vggt(img_B)
        H_p = img_A.shape[-2] // self.cfg.patch_size
        W_p = img_A.shape[-1] // self.cfg.patch_size
        images = torch.stack((img_A, img_B), dim=1)  # (B, 2, 3, H, W)

        grad_ctx = torch.no_grad() if self.cfg.vggt_frozen else nullcontext()
        with grad_ctx:
            aggregated_tokens_list, patch_token_start = self.vggt.aggregator(images)

        feats = []
        for tokens in aggregated_tokens_list:
            if tokens is None:
                continue
            patch_tokens = tokens[:, :, patch_token_start:, :]
            feats.append(patch_tokens.reshape(B, 2, H_p, W_p, self.feat_dim))
        return feats, H_p, W_p

    def _descriptor_features(
        self, img_A: torch.Tensor, img_B: torch.Tensor
    ) -> tuple[list[torch.Tensor], int, int]:
        """Run the frozen DINOv3 descriptor on each image and return its per-layer
        features stacked over the two views: list (shallow->deep) of tensors of
        shape (B, 2, H, W, feat_dim), plus the patch grid size."""
        with torch.no_grad():
            feats_A = self.descriptor(img_A)  # list of (B, H, W, D)
            feats_B = self.descriptor(img_B)
        feats = [torch.stack((a, b), dim=1) for a, b in zip(feats_A, feats_B)]
        H, W = feats_A[0].shape[1], feats_A[0].shape[2]
        return feats, H, W

    def _extract_features(
        self, img_A: torch.Tensor, img_B: torch.Tensor
    ) -> tuple[list[torch.Tensor], int, int]:
        if self.cfg.backbone == "vggt":
            return self._vggt_features(img_A, img_B)
        return self._descriptor_features(img_A, img_B)

    def _assemble_head_input(
        self,
        *,
        f_list_src: list[torch.Tensor],
        f_corr_src: torch.Tensor,
        f_corr_tgt: torch.Tensor,
        pos_emb_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        B, H_s, W_s, _ = f_corr_src.shape
        _, H_t, W_t, _ = f_corr_tgt.shape
        attn_logits, attn, match_emb = _compute_match_embeddings(
            f_A=f_corr_src,
            f_B=f_corr_tgt,
            pos_emb_grid=pos_emb_grid,
            temp=self.temp,
            B=B,
            H_A=H_s,
            W_A=W_s,
            H_B=H_t,
            W_B=W_t,
        )
        # Multi-scale VGGT features for the DPT head; inject the (projected)
        # match embedding at the two deepest scales. romav3 feeds the head two
        # scales with the match signal on the deeper one, which the DPT then
        # duplicates into its two deepest fusion levels; we mirror that coverage
        # here by adding the match embedding to the last two head-input scales.
        head_input = list(f_list_src)
        match_proj = self.match_emb_to_dim(match_emb)
        head_input[-1] = head_input[-1] + match_proj
        if len(head_input) >= 2:
            head_input[-2] = head_input[-2] + match_proj
        return attn_logits, attn, head_input

    def _refine_deepest(
        self, feat_A: torch.Tensor, feat_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine the deepest VGGT feature of both views jointly with the global
        attention layers, *before* correlation, so the matched descriptors are
        trainable (the frozen backbone stays frozen).

        romav3-style dimensionality: project the wide VGGT feature DOWN to
        global_dim, run the cross-view transformer at that narrow width, project
        back UP to feat_dim, and add as a residual (the up-projection is
        zero-initialized, so at step 0 this returns the input unchanged).

        Attention pattern depends on cfg.global_attention_mode:
          "global"      -> every layer is full/global over the union of A's and
                           B's tokens (A attends to B and vice versa), i.e. a
                           cross-view transformer like romav3's mv_vit.
          "alternating" -> even layers attend globally, odd layers attend
                           framewise (each view only sees its own tokens, with
                           RoPE when global_use_rope), exactly like the dense
                           matcher's mv_attention_mode == "alternating".
        Same weights for both views.

        When num_global_layers == 0 the refiner is disabled (no params built), so
        this returns the raw features unchanged.
        """
        if self.global_refiner is None:
            return feat_A, feat_B
        B, H, W, D = feat_A.shape
        a = self.global_in(feat_A.reshape(B, H * W, D))
        b = self.global_in(feat_B.reshape(B, H * W, D))
        x = torch.cat((a, b), dim=1)  # (B, 2*H*W, global_dim): all tokens, both views
        if self.cfg.global_attention_mode == "alternating":
            rope = self.global_rope(H=H, W=W) if self.global_rope is not None else None
            for idx, blk in enumerate(self.global_refiner):
                if idx % 2 == 1:
                    # Framewise: split the views apart so attention stays within a
                    # single view, apply rope, then merge back.
                    x = rearrange(x, "B (V N) D -> (B V) N D", V=2)
                    x = blk(x, rope)
                    x = rearrange(x, "(B V) N D -> B (V N) D", V=2)
                else:
                    x = blk(x)  # global / cross-view, no rope
        else:
            for blk in self.global_refiner:
                x = blk(x)
        x = self.global_out(x)  # back up to feat_dim; zero-init -> 0 at step 0
        d_A, d_B = x[:, : H * W], x[:, H * W :]
        refined_A = feat_A + d_A.reshape(B, H, W, D)
        refined_B = feat_B + d_B.reshape(B, H, W, D)
        return refined_A, refined_B

    def _run_head(
        self,
        head_input: list[torch.Tensor],
        img_src: torch.Tensor,
        img_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        warp_and_confidence = self.head(head_input, img_A=img_src, img_B=img_tgt)
        warp = warp_and_confidence[:, :, :, :2]
        confidence = warp_and_confidence[:, :, :, 2:]
        return warp, confidence

    def forward(
        self,
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        bidirectional: bool,
    ):
        preds = {}
        with torch.autocast(device.type, torch.bfloat16, enabled=self.cfg.enable_amp):
            feats, H, W = self._extract_features(img_A, img_B)
            f_list_A = [f[:, 0] for f in feats]
            f_list_B = [f[:, 1] for f in feats]
            # Refine the deepest backbone feature of both views jointly (cross-view
            # global attention) *before* correlation, so the matched descriptors
            # are trainable. The refined feature is then used for both the
            # correlation and the deepest head input (like romav3's mv_vit).
            f_list_A[-1], f_list_B[-1] = self._refine_deepest(
                f_list_A[-1], f_list_B[-1]
            )
            # Project to the working width for correlation + the DPT head (Identity
            # when work_dim == feat_dim). The refined deepest feature drives
            # correlation; head_feature_layers selects which scales feed the head.
            f_corr_A = self.to_work(f_list_A[-1])
            f_corr_B = self.to_work(f_list_B[-1])
            head_idxs = (
                self.cfg.head_feature_layers
                if self.cfg.head_feature_layers is not None and self.cfg.backbone == "vggt"
                else range(len(feats))
            )
            head_layers_A = [self.to_work(f_list_A[i]) for i in head_idxs]
            head_layers_B = [self.to_work(f_list_B[i]) for i in head_idxs]

            B = img_A.shape[0]
            x = get_normalized_grid(B, H, W)
            x_emb = nn.functional.linear(
                x.reshape(B, H * W, 2), self.scale * self.omega
            ).reshape(B, H, W, -1)
            pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

            attn_logits_AB, attn_AB, head_input_AB = self._assemble_head_input(
                f_list_src=head_layers_A,
                f_corr_src=f_corr_A,
                f_corr_tgt=f_corr_B,
                pos_emb_grid=pos_emb_grid,
            )
            if bidirectional:
                attn_logits_BA, attn_BA, head_input_BA = self._assemble_head_input(
                    f_list_src=head_layers_B,
                    f_corr_src=f_corr_B,
                    f_corr_tgt=f_corr_A,
                    pos_emb_grid=pos_emb_grid,
                )
            else:
                attn_logits_BA = None
                attn_BA = None
                head_input_BA = None

        # Run the head outside autocast (like romav3) so warp/confidence come out
        # in float32; otherwise the head's final conv emits bfloat16 and every
        # downstream grid_sample / flow_to_color on the warp fails.
        warp_AB, confidence_AB = self._run_head(head_input_AB, img_A, img_B)
        if bidirectional:
            warp_BA, confidence_BA = self._run_head(head_input_BA, img_B, img_A)
        else:
            warp_BA = None
            confidence_BA = None

        preds["attn_logits_AB"] = [attn_logits_AB]
        preds["attn_AB"] = [attn_AB]
        preds["warp_AB"] = warp_AB
        preds["confidence_AB"] = confidence_AB

        preds["attn_logits_BA"] = [attn_logits_BA] if attn_logits_BA is not None else None
        preds["attn_BA"] = [attn_BA] if attn_BA is not None else None
        preds["warp_BA"] = warp_BA
        preds["confidence_BA"] = confidence_BA
        return preds
