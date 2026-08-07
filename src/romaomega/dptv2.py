from dataclasses import dataclass, field
from functools import partial

import torch
from romaomega.device import device
import torch.nn as nn

from typing import Literal

BlockType = Literal["dpt", "flux", "roma"]


class DPTV2Head(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        features: int = 256
        out_channels: list[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
        align_corners: bool = True
        groups: int = 1
        block_type: BlockType = "dpt"
        resize_type: Literal["conv-transpose", "bilinear"] = "conv-transpose"

    def __init__(
        self,
        cfg: Cfg,
        *,
        dim_in: int,
        out_dim: int,
        patch_size: int,
        down_ratio: int,
    ) -> None:
        super().__init__()
        self.out_channels = cfg.out_channels
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.align_corners = cfg.align_corners
        self.norm = nn.LayerNorm(dim_in)
        out_channels = cfg.out_channels

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in cfg.out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        if cfg.resize_type == "conv-transpose":
            self.resize_layers = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_channels=out_channels[0],
                        out_channels=out_channels[0],
                        kernel_size=4,
                        stride=4,
                        padding=0,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=out_channels[1],
                        out_channels=out_channels[1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3],
                        out_channels=out_channels[3],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ]
            )
        elif cfg.resize_type == "bilinear":
            self.resize_layers = [
                partial(
                    custom_interpolate,
                    mode="bilinear",
                    align_corners=cfg.align_corners,
                    size_or_scale=4,
                ),
                partial(
                    custom_interpolate,
                    mode="bilinear",
                    align_corners=cfg.align_corners,
                    size_or_scale=2,
                ),
                nn.Identity(),
                partial(
                    custom_interpolate,
                    mode="bilinear",
                    align_corners=cfg.align_corners,
                    size_or_scale=1 / 2,
                ),
            ]
        else:
            raise ValueError(f"Invalid resize type: {cfg.resize_type}")

        features = cfg.features
        groups = cfg.groups
        align_corners = cfg.align_corners
        self.scratch = _make_scratch(
            in_shape=out_channels, out_shape=features, groups=groups, expand=False
        )
        make_fusion_block = partial(
            FeatureFusionBlock,
            features=features,
            groups=groups,
            align_corners=align_corners,
            has_residual=True,
            block_type=cfg.block_type,
            bn=False,
            expand=False,
            deconv=False,
        )
        self.scratch.refinenet1 = make_fusion_block()
        self.scratch.refinenet2 = make_fusion_block()
        self.scratch.refinenet3 = make_fusion_block()
        self.scratch.refinenet4 = make_fusion_block(has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv2_in_channels = head_features_1 // 2

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                conv2_in_channels,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        *,
        img_A: torch.Tensor | None = None,
        img_B: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tokens, torch.Tensor):
            B, H, W, D = tokens.shape
        else:
            B, H, W, D = tokens[0].shape
        return self._forward_impl(tokens, patch_h=H, patch_w=W)

    def _forward_impl(
        self,
        aggregated_tokens_list_or_tokens: list[torch.Tensor] | torch.Tensor,
        *,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            assert not isinstance(aggregated_tokens_list_or_tokens, torch.Tensor), (
                "aggregated_tokens_list_or_tokens should be a list of tensors"
            )
            aggregated_tokens_list = aggregated_tokens_list_or_tokens
            if len(aggregated_tokens_list) != len(self.out_channels):
                assert len(self.out_channels) % len(aggregated_tokens_list) == 0, (
                    "out_channels must be a multiple of intermediate_layer_idx"
                )
                factor = len(self.out_channels) // len(aggregated_tokens_list)
                aggregated_tokens_list = [
                    x
                    for xs in [
                        [aggregated_tokens_list[i]] * factor
                        for i in range(len(aggregated_tokens_list))
                    ]
                    for x in xs
                ]

            B = aggregated_tokens_list[0].shape[0]

            # H, W = patch_h * self.patch_size, patch_w * self.patch_size

            out = []

            for dpt_idx in range(len(self.out_channels)):
                x = aggregated_tokens_list[dpt_idx]

                x = x.reshape(B, -1, x.shape[-1])

                x = self.norm(x)

                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w)
                )

                x = self.projects[dpt_idx](x)
                x = self.resize_layers[dpt_idx](x)

                out.append(x)

            # Fuse features from multiple layers.
            out = self.scratch_forward(out)
            # Interpolate fused output to match target image resolution.
            out = custom_interpolate(
                out,
                (
                    int(patch_h * self.patch_size / self.down_ratio),
                    int(patch_w * self.patch_size / self.down_ratio),
                ),
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # float it for precision
        out = out.float()
        out = self.scratch.output_conv2(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size_or_scale=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(
            out, layer_3_rn, size_or_scale=layer_2_rn.shape[2:]
        )
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(
            out, layer_2_rn, size_or_scale=layer_1_rn.shape[2:]
        )
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn, size_or_scale=2)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_scratch(
    *,
    in_shape: list[int],
    out_shape: int,
    groups: int,
    expand: bool,
) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        activation: nn.Module,
        bn: bool,
        groups: int,
    ) -> None:
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RoMaBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_1_depthwise = nn.Conv2d(
            dim, dim, kernel_size=5, stride=1, padding=2, groups=dim
        )
        self.conv_1_pointwise = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv_2_depthwise = nn.Conv2d(
            dim, dim, kernel_size=5, stride=1, padding=2, groups=dim
        )
        self.conv_2_pointwise = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x
        h = self.conv_1_depthwise(h)
        h = self.norm1(h)
        h = self.relu(h)
        h = self.conv_1_pointwise(h)

        h = self.conv_2_depthwise(h)
        h = self.norm2(h)
        h = self.relu(h)
        h = self.conv_2_pointwise(h)
        return x + h


class FluxResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        deconv: bool,
        bn: bool,
        expand: bool,
        align_corners: bool,
        has_residual: bool,
        groups: int,
        block_type: Literal["dpt", "flux"],
    ) -> None:
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=self.groups,
        )
        self.has_residual = has_residual
        if block_type == "dpt":
            activation = nn.ReLU(inplace=True)
            if has_residual:
                self.resConfUnit1 = ResidualConvUnit(
                    features=features, activation=activation, bn=bn, groups=self.groups
                )

            self.resConfUnit2 = ResidualConvUnit(
                features=features, activation=activation, bn=bn, groups=self.groups
            )
        elif block_type == "flux":
            activation = swish
            if has_residual:
                self.resConfUnit1 = FluxResnetBlock(features, features)

            self.resConfUnit2 = FluxResnetBlock(features, features)
        elif block_type == "roma":
            if has_residual:
                self.resConfUnit1 = RoMaBlock(features)

            self.resConfUnit2 = RoMaBlock(features)
        else:
            raise ValueError(f"Invalid block type: {block_type}")

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size_or_scale: int | tuple[int, int]) -> torch.Tensor:
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = custom_interpolate(
            output, size_or_scale, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size_or_scale: int | float | tuple[int, int],
    *,
    mode: str,
    align_corners: bool,
) -> torch.Tensor:
    if isinstance(size_or_scale, int) or isinstance(size_or_scale, float):
        size = (int(x.shape[-2] * size_or_scale), int(x.shape[-1] * size_or_scale))
    else:
        size = size_or_scale

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk, size=size, mode=mode, align_corners=align_corners
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(
            x, size=size, mode=mode, align_corners=align_corners
        )
