from romaomega.types import HeadType
from romaomega.dpt import DPTHead
from romaomega.dptv2 import DPTV2Head
# from romaomega.linear import _LinearHead
# from romaomega.conv import ConvHead


class Head:
    def __new__(
        cls,
        *,
        head_type: HeadType,
        dim: int,
        warp_dim: int,
        confidence_dim: int,
        down_ratio: int,
        patch_size: int,
    ):
        if head_type == "dpt-no-pos":
            return DPTHead(
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        elif head_type == "dptv2-baseline":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(block_type="dpt"),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        elif head_type == "dptv2-flux":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(block_type="flux"),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        elif head_type == "dptv2-roma":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(block_type="roma"),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )

        elif head_type == "dptv2-bilinear":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(resize_type="bilinear"),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        elif head_type == "dptv2-bilinear-no-align-corners":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(resize_type="bilinear", align_corners=False),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        elif head_type == "dptv2-no-align-corners":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(align_corners=False),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )

        elif head_type == "dptv2-flux-bilinear":
            return DPTV2Head(
                cfg=DPTV2Head.Cfg(resize_type="bilinear"),
                dim_in=dim,
                out_dim=warp_dim + confidence_dim,
                down_ratio=down_ratio,
                patch_size=patch_size,
            )
        else:
            raise ValueError(f"Head {head_type} not implemented")
