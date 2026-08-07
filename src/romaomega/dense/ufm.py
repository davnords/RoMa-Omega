import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
from romaomega.types import ImageLike
from romaomega.io import check_not_i16
from romaomega.geometry import get_pixel_grid, to_normalized
from romaomega.device import device
from romaomega import RoMaV3


class UFMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        import sys

        sys.path.append("third_party/UFM")
        sys.path.append("third_party/UniCeption")
        # Load the base model (for general use)
        # from uniflowmatch.models.ufm import UniFlowMatchConfidence
        # model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

        # Or load the refinement model (for higher accuracy)
        from uniflowmatch.models.ufm import (
            UniFlowMatchClassificationRefinement,  # type: ignore
        )

        model = UniFlowMatchClassificationRefinement.from_pretrained(
            "infinity1096/UFM-Refine"
        )

        # Set the model to evaluation mode
        model.eval()
        self.model = model
        self.to(device)
        self.bidirectional = True
        # Load images using cv2 or PIL

    def _load_image(self, img_like: ImageLike) -> torch.Tensor:
        if isinstance(img_like, str) or isinstance(img_like, Path):
            img_pil = Image.open(img_like)
            check_not_i16(img_pil)
            img_pil = img_pil.convert("RGB")
            img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).to(device)
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

    def match(self, img_like_A: ImageLike, img_like_B: ImageLike):
        # return RoMav3.match(self, img_like_A, img_like_B)
        self.eval()
        bidirectional = self.bidirectional
        with torch.inference_mode():
            img_A = self._load_image(img_like_A)
            img_B = self._load_image(img_like_B)
            # img_A = F.interpolate(img_A, (420, 560), mode="bicubic", align_corners=False, antialias=True)
            # img_B = F.interpolate(img_B, (420, 560), mode="bicubic", align_corners=False, antialias=True)
            assert img_A is not None and img_B is not None
            # img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            # img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            # img_A = torch.from_numpy(img_A).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            # img_B = torch.from_numpy(img_B).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            preds_AB = self(img_A, img_B)
            warp = preds_AB["warp_AB"]
            overlap = preds_AB["confidence_AB"][..., 0]
            # overlap[overlap > 0.2] = 1.0
            if bidirectional:
                preds_BA = self(img_B, img_A)
                warp_BA = preds_BA["warp_AB"]
                overlap_BA = preds_BA["confidence_AB"][..., 0]
                # overlap_BA[overlap_BA > 0.2] = 1.0
            # from PIL import Image
            # Image.fromarray(flow_to_color(warp[0].cpu().numpy())).save("vis/ufm_warp.png")
        preds = {"warp_AB": warp.clone(), "overlap_AB": overlap.clone()}
        if bidirectional:
            preds["warp_BA"] = warp_BA.clone()
            preds["overlap_BA"] = overlap_BA.clone()

        return preds

    def sample(self, preds: dict[str, torch.Tensor], num_corresp: int):
        return RoMaV3.sample(self, preds, num_corresp)

    def to_pixel_coordinates(
        self, warp: torch.Tensor, H_A: int, W_A: int, H_B: int, W_B: int
    ):
        return RoMaV3.to_pixel_coordinates(warp, H_A=H_A, W_A=W_A, H_B=H_B, W_B=W_B)

    @torch.no_grad()
    def forward(self, source_image, target_image, **kwargs):
        assert not self.training
        source_image = F.interpolate(
            source_image,
            (420, 560),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        target_image = F.interpolate(
            target_image,
            (420, 560),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        B, C, H_A, W_A = source_image.shape
        B, C, H_B, W_B = target_image.shape
        # assert H_A == H_B and W_A == W_B, "Source and target images must have the same shape"
        # Predict correspondences
        result = self.model.predict_correspondences_batched(
            source_image=source_image,
            target_image=target_image,
            data_norm_type="dummy",
        )

        flow = result.flow.flow_output
        covisibility = result.covisibility.mask[..., None]
        flow = flow.permute(0, 2, 3, 1)
        H, W = flow.shape[1:3]
        pixel_warp = flow + get_pixel_grid(1, H=H_A, W=W_A)
        normalized_warp = to_normalized(pixel_warp, H=H_B, W=W_B)

        return {"warp_AB": normalized_warp, "confidence_AB": covisibility}
