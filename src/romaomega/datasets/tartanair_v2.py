from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal
import torch
from PIL import Image
import numpy as np
import OpenEXR
from romaomega.datasets.transforms import Transform
from romaomega.types import Batch
from tqdm import tqdm
import logging

logger = logging.getLogger("__name__")


def _read_exr_depth(path: Path) -> torch.Tensor:
    with OpenEXR.File(path.as_posix()) as file:
        depth = file.channels()["Y"].pixels.astype(np.float32)
        # follow https://github.com/facebookresearch/map-anything/blob/5ffcb617601a40687e0daf06cd625959adeb4472/mapanything/datasets/wai/tav2_wb.py#L128C13-L128C80
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.from_numpy(depth)
    return out


class TartanAirV2Scene(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, transform_cfg: Transform.Cfg):
        self.data_root = data_root
        self.pair_ids = sorted(
            [
                Path(p).with_suffix("").name.split("_")[0]
                for p in os.listdir(data_root / "images")
            ]
        )
        self.transform = Transform(transform_cfg)

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, idx):
        try:
            return self.load_item(idx)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.load_item(idx % len(self.pair_ids))

    def load_item(self, idx):
        pair_id = self.pair_ids[idx]
        image_paths = [
            self.data_root / "images" / f"{pair_id}_{i}.png" for i in range(2)
        ]
        depth_paths = [
            self.data_root / "depth" / f"{pair_id}_{i}.exr" for i in range(2)
        ]
        intrinsic_paths = [
            self.data_root / "camera_params" / f"{pair_id}_{i}.npy" for i in range(2)
        ]
        pose_paths = [self.data_root / "poses" / f"{pair_id}_{i}.npy" for i in range(2)]
        # intrinsics likely is off by 0.5 in x and y (e.g. 319.5 for imgsize 640), suggests that we should add 0.5
        K_A = torch.from_numpy(np.load(intrinsic_paths[0]))
        K_B = torch.from_numpy(np.load(intrinsic_paths[1]))
        # likely UFM uses the original intrinsics, so adding here might cause issues for them...
        K_A[:2, 2] += 0.5
        K_B[:2, 2] += 0.5

        ret = self.transform(
            pil_img_A=Image.open(image_paths[0]),
            pil_img_B=Image.open(image_paths[1]),
            depth_A=_read_exr_depth(depth_paths[0]),
            depth_B=_read_exr_depth(depth_paths[1]),
            K_A=K_A,
            K_B=K_B,
        )
        # poses are c2w (see https://github.com/facebookresearch/map-anything/blob/5ffcb617601a40687e0daf06cd625959adeb4472/mapanything/datasets/wai/tav2_wb.py#L125)
        pose_c2w_A = torch.from_numpy(np.load(pose_paths[0]))
        pose_c2w_B = torch.from_numpy(np.load(pose_paths[1]))
        T_AB = pose_c2w_B.inverse() @ pose_c2w_A
        return Batch(
            **ret,
            pose_A=pose_c2w_A.inverse(),
            pose_B=pose_c2w_B.inverse(),
            T_AB=T_AB,
            img_A_path=image_paths[0],
            img_B_path=image_paths[1],
            source="depth",
            quality="high",
        )


class TartanAirV2(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "test"]
        data_root: str = "data/tartanair-v2/ufm-split/tav2_wb"
        # Dataset-specific transform overrides (merged with global transform at setup time)
        transform_overrides: dict[str, Any] = field(default_factory=dict)
        weight: int = 1_000_000
        fraction: float = 1.0

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg):
        self.cfg = cfg
        self.transform_cfg = transform_cfg
        # TODO: use split_scene_list from mapanything

        self.val_split_scenes = ["EndofTheWorld", "HongKong", "WesternDesertTown"]

        # Test split
        self.test_split_scenes = [
            "DesertGasStation",
            "OldScandinavia",
            "PolarSciFi",
            "Sewerage",
            "Supermarket",
        ]
        self.scenes: list[TartanAirV2Scene] = []
        for scene in tqdm(
            sorted(os.listdir(cfg.data_root)),
            desc=f"Loading {type(self).__name__} {cfg.split} scenes...",
        ):
            if scene in self.val_split_scenes and cfg.split != "val":
                continue
            if scene in self.test_split_scenes and cfg.split != "test":
                continue
            if (
                scene not in self.val_split_scenes
                and scene not in self.test_split_scenes
                and cfg.split != "train"
            ):
                continue
            self.scenes.append(
                TartanAirV2Scene(Path(cfg.data_root) / scene, self.transform_cfg)
            )
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[:max(1, round(len(self.scenes) * cfg.fraction))]
        self.cum_weights = torch.linspace(1 / len(self.scenes), 1, len(self.scenes))
        logger.info(
            f"Built {len(self.scenes)} TartanAirV2 scenes with a total of {sum([len(scene) for scene in self.scenes])} pairs."
        )

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx: int) -> Batch:
        assert self.cum_weights is not None
        assert self.scenes is not None
        scene_index = torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self))
        scene = self.scenes[scene_index]
        pair_idx = np.random.choice(len(scene))
        return scene[pair_idx]
