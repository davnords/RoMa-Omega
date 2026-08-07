from dataclasses import dataclass, field
import random
from glob import glob
import os.path as osp
from typing import Any, Literal
from .transforms import Transform
import torch.utils.data
import torch
from romaomega.types import Batch
from .read_flow import read_gen
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger("__name__")


class FlyingThings3D(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["TRAIN", "TEST"]
        data_root: str = "data/FlyingThings3D_opticalflows"
        # Dataset-specific transform overrides (merged with global transform at setup time)
        transform_overrides: dict[str, Any] = field(default_factory=dict)
        scene_names: list[str] | None = None
        weight: int = 1_000_000
        fraction: float = 1.0

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg):
        super().__init__()
        assert cfg.split in ["TRAIN", "TEST"], "split must be TRAIN or TEST"
        self.cfg = cfg
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.transform = Transform(transform_cfg)
        self.dstypes: list[str] = ["frames_cleanpass", "frames_finalpass"]

        image_dirs = sorted(
            glob(osp.join(cfg.data_root, self.dstypes[0], f"{cfg.split}/*/*"))
        )
        flow_dirs = sorted(
            glob(osp.join(cfg.data_root, f"optical_flow/{cfg.split}/*/*"))
        )
        for cam in tqdm(
            ["left"], desc=f"Loading {type(self).__name__} {cfg.split} scenes..."
        ):
            image_dirs_c = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs_c_into_future = sorted(
                [osp.join(f, "into_future", cam) for f in flow_dirs]
            )
            flow_dirs_c_into_past = sorted(
                [osp.join(f, "into_past", cam) for f in flow_dirs]
            )

            for idir, fdir_fwd, fdir_bwd in zip(
                image_dirs_c, flow_dirs_c_into_future, flow_dirs_c_into_past
            ):
                images = sorted(glob(osp.join(idir, "*.png")))
                flows_fwd = sorted(glob(osp.join(fdir_fwd, "*.pfm")))
                flows_bwd = sorted(glob(osp.join(fdir_bwd, "*.pfm")))
                for i in range(len(flows_fwd) - 1):
                    self.image_list += [(images[i], images[i + 1])]
                    # other way around?
                    self.flow_list += [(flows_fwd[i], flows_bwd[i + 1])]
                    # self.flow_list += [(flows_bwd[i], flows_fwd[i+1])]

        if cfg.fraction < 1.0:
            n = max(1, round(len(self.image_list) * cfg.fraction))
            self.image_list = self.image_list[:n]
            self.flow_list = self.flow_list[:n]
        logger.info(f"Built {len(self.image_list)} FlyingThings3D pairs.")

    def __len__(self):
        return self.cfg.weight

    def load_img_and_flow(self, img_path: str, flow_path: str, dstype: str):
        resolved_img_path = img_path.replace(self.dstypes[0], dstype)
        img_pil = Image.open(resolved_img_path)
        width, height = img_pil.size
        resolved_flow_path = flow_path.replace(self.dstypes[0], dstype)
        flow = read_gen(resolved_flow_path)
        scale_pixel_to_normalized = torch.tensor([2 / (width), 2 / (height)])[
            None, None
        ]
        flow_tensor = torch.from_numpy(flow) * scale_pixel_to_normalized
        return img_pil, flow_tensor

    def __getitem__(self, idx):
        try:
            return self.load_item(idx)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.load_item(idx % len(self.image_list))

    def load_item(self, idx):
        idx = idx % len(self.image_list)
        img_A_path, img_B_path = self.image_list[idx]
        flow_fwd_path, flow_bwd_path = self.flow_list[idx]
        # 50% to take cleanpass or finalpass
        dstype = random.choice(self.dstypes)
        if random.random() < 0.5:
            # swap order of images and flows
            img_A_path, img_B_path = img_B_path, img_A_path
            flow_fwd_path, flow_bwd_path = flow_bwd_path, flow_fwd_path
        img_A_pil, flow_AB = self.load_img_and_flow(img_A_path, flow_fwd_path, dstype)
        img_B_pil, flow_BA = self.load_img_and_flow(img_B_path, flow_bwd_path, dstype)
        ret = self.transform(
            pil_img_A=img_A_pil,
            pil_img_B=img_B_pil,
            flow_AB=flow_AB,
            flow_BA=flow_BA,
        )
        return Batch(
            **ret,
            img_A_path=img_A_path,
            img_B_path=img_B_path,
            source="flow",
            quality="high",
        )
