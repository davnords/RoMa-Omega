from __future__ import annotations
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import Literal, Protocol
import numpy as np
from PIL import Image
import json
import logging
import sys

logger = logging.getLogger(__name__)


@dataclass
class Warp:
    warp: torch.Tensor
    covis: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    error: torch.Tensor | None = None


@dataclass
class Batch:
    img_A: torch.Tensor
    img_B: torch.Tensor
    depth_A: torch.Tensor
    depth_B: torch.Tensor
    img_A_path: Path
    img_B_path: Path
    source: GTSource | list[GTSource]
    flow_AB: torch.Tensor
    flow_BA: torch.Tensor
    mask_AB: torch.Tensor
    mask_BA: torch.Tensor
    quality: Quality | list[Quality]
    # Optional sparse pixel correspondences: (N, 4) = (x_A, y_A, x_B, y_B) in pixels.
    correspondences_AB: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4))
    )
    K_A: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    K_B: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    pose_A: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    pose_B: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    T_AB: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    num_corresp: int | list[int] = 0

    def to(self, device: torch.device) -> Batch:
        # Preserve dynamic batch type (subclasses may add more tensor fields).
        data = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            else:
                data[k] = v
        return Batch(**data)

    def swap_AB(self) -> Batch:
        """Return a "reversed" batch (B->A) useful for symmetric GT computation."""
        return Batch(
            img_A=self.img_B,
            img_B=self.img_A,
            depth_A=self.depth_B,
            depth_B=self.depth_A,
            K_A=self.K_B,
            K_B=self.K_A,
            pose_A=self.pose_B,
            pose_B=self.pose_A,
            T_AB=torch.linalg.inv(self.T_AB),
            img_A_path=self.img_B_path,
            img_B_path=self.img_A_path,
            source=self.source,
            flow_AB=self.flow_BA,
            flow_BA=self.flow_AB,
            mask_AB=self.mask_BA,
            mask_BA=self.mask_AB,
            quality=self.quality,
            correspondences_AB=self.correspondences_AB[..., [2, 3, 0, 1]],
            num_corresp=self.num_corresp,
        )

    @classmethod
    def collate(cls, samples: list[Batch]) -> Batch:
        keys = samples[0].__dict__.keys()
        batch = {}
        for k in keys:
            if k == "correspondences_AB":
                corresps = 2 * torch.rand((len(samples), MAX_NUM_SPARSE_CORRESP, 4)) - 1
                for i, s in enumerate(samples):
                    corresps[i, : s.num_corresp] = s.correspondences_AB
                batch[k] = corresps
            elif isinstance(samples[0].__dict__[k], torch.Tensor):
                batch[k] = torch.stack([s.__dict__[k] for s in samples])
            else:
                batch[k] = [s.__dict__[k] for s in samples]
        return Batch(**batch)


def _load_cfg(cls, data: dict, *, strict: bool):
    annotations = getattr(cls, "__annotations__", {})
    kwargs = {}
    for k, v in data.items():
        if k not in annotations:
            if strict:
                raise ValueError(f"Attribute {k} not found in {cls.__name__}")
            else:
                logger.warning(f"Attribute {k} not found in {cls.__name__}")
            continue
        annot_type_or_type_name = cls.__dataclass_fields__[k].type
        if isinstance(annot_type_or_type_name, str):
            # Resolve the annotation string in the config class's OWN module, not
            # this one. With `from __future__ import annotations` the field type
            # is a bare string like "Matcher.Cfg"; eval'ing it here would pick up
            # unrelated names defined in types.py (e.g. the Matcher Protocol).
            annot_type = eval(
                annot_type_or_type_name, vars(sys.modules[cls.__module__])
            )
        else:
            annot_type = annot_type_or_type_name
        if hasattr(annot_type, "__annotations__"):
            kwargs[k] = _load_cfg(annot_type, v, strict=strict)
        else:
            kwargs[k] = v

    return cls(**kwargs)


class Model(nn.Module):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def from_run(
        cls, run_path: str | Path, strict: bool = True, load_avg_weights: bool = True
    ) -> Model:
        from romaomega.device import device

        run_path = Path(run_path)
        with open(run_path / "cfg.json", "r") as f:
            serialized_cfg = json.load(f)["model"]

        if load_avg_weights:
            if not (run_path / "avg_weights.pth").exists():
                raise FileNotFoundError(
                    f"{load_avg_weights=}, but average weights file not found at {run_path / 'avg_weights.pth'}"
                )
            weights = torch.load(run_path / "avg_weights.pth", map_location=device)
            weights = {
                n.replace("module.", ""): w
                for n, w in weights.items()
                if n.startswith("module.")
            }
        else:
            weights = torch.load(run_path / "weights.pth", map_location=device)
        return cls.from_serialized_cfg_and_weights(
            serialized_cfg, weights, strict=strict
        )

    @classmethod
    def from_serialized_cfg_and_weights(
        cls, serialized_cfg: dict, weights: dict[str, torch.Tensor], strict: bool = True
    ) -> Model:
        model = cls.from_serialized_cfg(serialized_cfg, strict=strict)
        missing, unexpected = model.load_state_dict(weights, strict=strict)
        if not strict:
            for k in missing:
                logger.warning(f"Missing key {k}")
            for k in unexpected:
                logger.warning(f"Unexpected key {k}")
        return model

    @classmethod
    def from_serialized_cfg(cls, serialized_cfg: dict, strict: bool = True) -> Model:
        cfg = _load_cfg(cls.Cfg, serialized_cfg, strict=strict)
        return cls.from_cfg(cfg)

    @classmethod
    def from_cfg(cls, cfg) -> Model:
        from romaomega.device import device

        return cls(cfg).to(device)


Quality = Literal["high", "low"]
GTSource = Literal["depth", "flow", "sparse"]
HeadType = Literal[
    "dpt-no-pos",
    "dptv2-baseline",
    "dptv2-flux",
    "dptv2-roma",
    "dptv2-bilinear",
    "dptv2-bilinear-no-align-corners",
    "dptv2-no-align-corners",
    "dptv2-flux-bilinear",
]
SampleMode = Literal["frame_distance", "overlap"]
ConfidenceMode = Literal["covis", "frame", "positive"]
NormType = Literal["batch"]
RefinersType = Literal["roma-4-pow2"]
MatcherStyle = Literal["romav3"]
DescriptorName = Literal["dinov3_vitl16", "dinov2_vitl14", "mum_vitl16", "flux2"]
Normalizer = Literal["imagenet", "inception"]  # Callable[[torch.Tensor], torch.Tensor]
OptimizerName = Literal["adamw"]
ImageLike = torch.Tensor | np.ndarray | str | Path | Image.Image
Setting = Literal[
    "mega1500", "scannet1500", "wxbs", "satast", "base", "precise", "turbo", "fast"
]
FineFeaturesType = Literal["vgg19", "vgg19bn", "flux2"]
MAX_NUM_SPARSE_CORRESP = 64


class Matcher(Protocol):
    """Protocol defining the interface for feature matchers."""

    def match(
        self, im_A_path: str | Path, im_B_path: str | Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Match features between two images.
        """
        ...

    def match_keypoints(
        self,
        keypoints_A: torch.Tensor,
        keypoints_B: torch.Tensor,
        warp: torch.Tensor,
        certainty: torch.Tensor,
        return_tuple: bool = False,
    ) -> torch.Tensor:
        """
        Match keypoints between two images.
        """
        ...

    def to_pixel_coordinates(
        self, matches: torch.Tensor, h1: int, w1: int, h2: int, w2: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert matches to pixel coordinates.
        """


class Detector(Protocol):
    """Protocol defining the interface for keypoint detectors."""

    @property
    def topleft(self) -> float:
        """Top-left coordinate offset for keypoint normalization."""
        ...

    def detect_from_path(
        self,
        im_path: str | Path,
        *,
        num_keypoints: int,
        return_dense_probs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Detect keypoints from an image file path.

        Args:
            im_path: Path to the image file
            num_keypoints: Number of keypoints to detect
            return_dense_probs: Whether to return dense probability maps

        Returns:
            Dictionary containing:
                - "keypoints": torch.Tensor of shape (B, N, 2) with normalized coordinates [-1, 1]
                - "keypoint_probs": torch.Tensor of shape (B, N) with confidence scores
                - "dense_probs": torch.Tensor of shape (B, H, W), optional if return_dense_probs=True
        """
        ...
    def detect(self, batch: Batch, *, num_keypoints: int, return_dense_probs: bool = False) -> dict[str, torch.Tensor]:
        """
        Detect keypoints from a batch of images.
        """
        ...

    def to_pixel_coords(
        self, normalized_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            normalized_coords: Tensor of shape (..., 2) with coordinates in [-1, 1]
            h: Image height in pixels
            w: Image width in pixels

        Returns:
            Tensor of shape (..., 2) with pixel coordinates
        """
        ...

class SparseMatcher(Protocol):
    """Protocol defining the interface for sparse matchers, e.g. LightGlue, SuperGlue and DualSoftMaxMatcher."""
    
    def match(
        self, keypoints_A: torch.Tensor, description_A: torch.Tensor, keypoints_B: torch.Tensor, description_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match features between two images.
        """
        ...

    def to_pixel_coords(
        self, normalized_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            normalized_coords: Tensor of shape (..., 2) with coordinates in [-1, 1]
            h: Image height in pixels
            w: Image width in pixels

        Returns:
            Tensor of shape (..., 2) with pixel coordinates
        """
        ...

class Descriptor(Protocol):
    """Protocol defining the interface for feature descriptors, e.g. DeDoDe and SIFT"""

    def describe_keypoints_from_path(self, im_path: str, keypoints: torch.Tensor, H:int=784, W:int=784) -> dict[str, torch.Tensor]:
        ...


from abc import ABC, abstractmethod
from typing import Tuple
class BenchmarkMatcher(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def match(self, img_A_path: str, img_B_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def get_type(self) -> Literal["dense", "sparse"]:
        pass

    @property
    @abstractmethod
    def offset(self) -> float:
        pass