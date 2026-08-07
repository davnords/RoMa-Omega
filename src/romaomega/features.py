from dataclasses import dataclass, field
import math
from typing import Any, Callable
import torch
from einops import rearrange
from romaomega.normalizers import imagenet, inception
from romaomega.types import Normalizer, DescriptorName, FineFeaturesType
from romaomega.device import device
from torch import nn
from functools import partial
import torchvision.models as models
from torch.nn import functional as F
from torch import Tensor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Flux2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        from romaomega.flux2_ae import (
            AutoEncoderParams,
            ResnetBlock,
            AttnBlock,
            Downsample,
        )  # AutoEncoder, swish
        import huggingface_hub
        import safetensors.torch as safetensors

        ae_path = huggingface_hub.hf_hub_download(
            "black-forest-labs/FLUX.2-dev", filename="ae.safetensors"
        )
        weights = safetensors.load_file(ae_path)
        params = AutoEncoderParams()

        self.quant_conv = torch.nn.Conv2d(
            2 * params.z_channels, 2 * params.z_channels, 1
        )
        self.ch = params.ch
        self.num_resolutions = len(params.ch_mult)
        self.num_res_blocks = params.num_res_blocks
        self.resolution = params.resolution
        self.in_channels = params.in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            params.in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = params.resolution
        in_ch_mult = (1,) + tuple(params.ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * params.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * params.z_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

        encoder_and_bn_state_dict = {
            k.replace("encoder.", ""): v
            for k, v in weights.items()
            if k.startswith("encoder.") or k.startswith("bn.")
        }
        self.load_state_dict(encoder_and_bn_state_dict)

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def get_intermediate_layers(self, x: torch.Tensor, *, n: int) -> torch.Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        mean = torch.chunk(h, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        B, C, H, W = z.shape
        return [z.permute(0, 2, 3, 1).reshape(B, H * W, C)]


def wrap_with_normalize(
    forward: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
):
    def wrapped_forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        with (
            torch.autocast(device.type, torch.bfloat16, enabled=enable_amp),
            torch.set_grad_enabled(not frozen),
        ):
            if self.training and frozen:
                self.eval()
            B, C, H, W = img.shape
            assert C == 3, f"Image must have 3 channels, but got shape {img.shape=}"
            img_n = normalizer(img)
            H = H // patch_size
            W = W // patch_size
            raw_outs = forward(img_n)
            maybe_feat_normalizer = (
                F.normalize if normalize_feats else lambda x, dim=-1: x
            )
            return [
                maybe_feat_normalizer(
                    rearrange(x, "B (H W) D -> B H W D", H=H, W=W), dim=-1
                )
                for x in raw_outs
            ]

    return wrapped_forward


def wrap_model(
    model: nn.Module,
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
    func: Any,
):
    if enable_amp and frozen:  # if training we want params in fp32
        model = model.to(torch.bfloat16)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.frozen = frozen
    type(model).forward = wrap_with_normalize(
        func,
        normalizer=normalizer,
        patch_size=patch_size,
        enable_amp=enable_amp,
        frozen=frozen,
        normalize_feats=normalize_feats,
    )
    return model


def _get_layers(layers, model):
    return [b if b > 0 else len(model.blocks) + b for b in layers]


class Descriptor:
    @dataclass(frozen=True)
    class Cfg:
        name: DescriptorName = "dinov3_vitl16"
        enable_amp: bool = True
        frozen: bool = True
        normalize_feats: bool = False
        layer_idx: list[int] = field(
            default_factory=lambda: [11, 17]
        )  # [4, 11, 17, 23] for dinov3 style
        weights_path: str | None = "dinov3_vitl16_pretrain_lvd1689m-08c60483.pth"

    def __new__(cls, cfg: Cfg) -> nn.Module:
        partial_wrap = partial(
            wrap_model,
            enable_amp=cfg.enable_amp,
            frozen=cfg.frozen,
            normalize_feats=cfg.normalize_feats,
        )
        match cfg.name:
            case "dinov3_vitl16":
                normalizer = imagenet
                dinov3_vitl16: nn.Module = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa",
                    model="dinov3_vitl16",
                    pretrained=cfg.weights_path is not None,
                    weights=cfg.weights_path,
                    skip_validation=True,
                ).to(device)
                layers = _get_layers(cfg.layer_idx, dinov3_vitl16)
                return partial_wrap(
                    dinov3_vitl16,
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(dinov3_vitl16.get_intermediate_layers, n=layers),
                )
            case "dinov2_vitl14":
                normalizer = imagenet

                dinov2_vit14: nn.Module = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vitl14"
                ).to(device)
                dinov2_vit14.mask_token = None
                layers = _get_layers(cfg.layer_idx, dinov2_vit14)
                return partial_wrap(
                    dinov2_vit14,
                    normalizer=normalizer,
                    patch_size=14,
                    func=partial(dinov2_vit14.get_intermediate_layers, n=layers),
                )
            case "mum_vitl16":
                from .vit.mum_vit import mum_vitl16

                normalizer = imagenet
                mum = mum_vitl16()
                layers = _get_layers(cfg.layer_idx, mum)
                return partial_wrap(
                    mum,
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(mum.get_intermediate_layers, n=layers),
                )
            case "flux2":
                normalizer = inception
                flux2 = Flux2()
                return partial_wrap(
                    flux2,
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(flux2.get_intermediate_layers, n=[-1]),
                )
            case _:
                raise ValueError(f"Unknown descriptor name: {cfg.name}")


class VGG(nn.Module):
    def forward(self, x):
        x = imagenet(x)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale = scale * 2
                x = layer(x)
            return feats


class VGG19(VGG):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        if patch_size not in [8]:
            raise NotImplementedError(
                f"VGG19 is not supported for patch size {patch_size}"
            )
        last_layer = {8: 28}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[
                :last_layer
            ]
        )


class VGG19BN(VGG):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        last_layer = {1: 7, 2: 14, 4: 27, 8: 40, 16: 52}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features[
                :last_layer
            ]
        )


class FineFeatures(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        type: FineFeaturesType = "vgg19bn"
        patch_size: int = 4

    def __new__(cls, cfg: Cfg):
        match cfg.type:
            case "vgg19":
                return VGG19(cfg.patch_size)
            case "vgg19bn":
                return VGG19BN(cfg.patch_size)
            case "flux2":
                raise NotImplementedError("Flux2 is not supported for fine features")
                return Flux2(cfg.patch_size)
            case _:
                raise ValueError(f"Unknown refiner features type: {cfg.type}")
