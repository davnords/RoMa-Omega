import random
from dataclasses import dataclass, field
import torch.utils.data
import torch
from pathlib import Path
from typing import Any, Literal
import numpy as np
from PIL import Image
from romaomega.datasets.transforms import Transform
import json
import OpenEXR
from romaomega.types import Batch
from tqdm import tqdm
import logging

logger = logging.getLogger("__name__")


def read_exr_depth(path: Path) -> torch.Tensor:
    with OpenEXR.File(path.as_posix()) as file:
        depth = file.channels()["Y"].pixels.astype(np.float32)
        out = torch.from_numpy(depth)
    return out


class ScanNetPlusPlusScene(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        transform: Transform,
        use_pairs_and_overlaps: bool,
    ):
        super().__init__()
        self.data_root = data_root
        self.meta_info = json.load(
            open(data_root / "dslr" / "nerfstudio" / "transforms_undistorted.json", "r")
        )
        # scannet++/scannetpp/data/0a5c013435/
        self.transform = transform
        self.frame_map = {
            frame["file_path"]: frame for frame in self.meta_info["frames"]
        }
        if use_pairs_and_overlaps:
            scene_name = data_root.name
            pairs_and_overlaps = np.load(
                data_root / ".." / ".." / "overlaps" / f"{scene_name}.npy",
                allow_pickle=True,
            ).item()
            pairs = pairs_and_overlaps["pairs"]
            broken_pairs = np.logical_or(pairs[:, 0] == 0, pairs[:, 1] == 0)
            if broken_pairs.sum() > 0:
                pass  # print(f"Found {broken_pairs.sum()} broken pairs in {scene_name}")
            self.pairs = pairs[~broken_pairs]

            self.overlaps = pairs_and_overlaps["overlaps"][~broken_pairs]
        else:
            self.pairs = None
            self.overlaps = None

    def __len__(self):
        return (
            len(self.meta_info["frames"]) ** 2
            if self.pairs is None
            else len(self.pairs)
        )

    def load_frame(self, frame: dict):
        im_path = (
            self.data_root / "dslr" / "resized_undistorted_images" / frame["file_path"]
        )
        depth_path = self.data_root / "rendered_depth" / (frame["file_path"] + ".exr")
        try:
            depth = read_exr_depth(depth_path)
        except Exception as e:
            if depth_path.exists():
                import shutil

                trash_location = Path("trash") / depth_path
                trash_location.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(depth_path, trash_location)
            print(f"Error reading exr depth: {e}")
            print(depth_path)
            raise e
        im = Image.open(im_path)
        K = torch.zeros((3, 3))
        K[0, 0] = self.meta_info["fl_x"]
        K[1, 1] = self.meta_info["fl_y"]
        K[0, 2] = self.meta_info["cx"]
        K[1, 2] = self.meta_info["cy"]
        K[2, 2] = 1.0
        pose_c2w_opengl = torch.tensor(frame["transform_matrix"]).float()
        pose_c2w_colmap = (
            pose_c2w_opengl
            @ torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            ).float()
        )
        pose_w2c_colmap = pose_c2w_colmap.inverse()
        return im, depth, K, pose_w2c_colmap, im_path

    def __getitem__(self, idx: int) -> Batch:
        try:
            return self.load_item(idx)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.load_item(idx % len(self.meta_info["frames"]))

    def load_item(self, idx):
        if self.pairs is not None:
            name_A, name_B = self.pairs[idx]
            frame_A = self.frame_map[name_A]
            frame_B = self.frame_map[name_B]
        else:
            N = len(self.meta_info["frames"])
            idx_A = idx // N
            idx_B = idx % N
            #  idx_A == idx_B:
            #     return random.choice(self)
            frame_A = self.meta_info["frames"][idx_A]
            frame_B = self.meta_info["frames"][idx_B]
        try:
            im_A, depth_A, K_A, pose_A, im_path_A = self.load_frame(frame_A)
            im_B, depth_B, K_B, pose_B, im_path_B = self.load_frame(frame_B)
        except Exception as e:
            print(f"Error loading frame: {e}")
            print(frame_A)
            print(frame_B)
            return random.choice(self)
        ret = self.transform(
            pil_img_A=im_A,
            pil_img_B=im_B,
            depth_A=depth_A,
            depth_B=depth_B,
            K_A=K_A,
            K_B=K_B,
        )
        return Batch(
            **ret,
            pose_A=pose_A,
            pose_B=pose_B,
            T_AB=pose_B @ pose_A.inverse(),
            img_A_path=im_path_A,
            img_B_path=im_path_B,
            source="depth",
            quality="low",
        )


class ScanNetPlusPlus(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "val", "test"]
        data_root: str = "data/scannet++/data_download/scannetpp/data"
        # Dataset-specific transform overrides (merged with global transform at setup time)
        transform_overrides: dict[str, Any] = field(default_factory=dict)
        weight: int = 1_000_000
        use_pairs_and_overlaps: bool = True
        fraction: float = 1.0

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg):
        super().__init__()
        self.data_root = Path(cfg.data_root)
        split_root = self.data_root / ".." / "splits"
        split_file = {
            "train": "nvs_sem_train.txt",
            "val": "nvs_sem_val.txt",
            "test": "nvs_test.txt",
        }[cfg.split]
        split_scenes = open(split_root / split_file).read().splitlines()
        self.cfg = cfg
        transform = Transform(transform_cfg)
        self.scenes = []
        for scene_root in tqdm(
            sorted((self.data_root).iterdir()),
            desc=f"Loading {type(self).__name__} {cfg.split} scenes...",
        ):
            if scene_root.name in split_scenes:
                try:
                    scene = ScanNetPlusPlusScene(
                        scene_root,
                        transform=transform,
                        use_pairs_and_overlaps=cfg.use_pairs_and_overlaps,
                    )
                    self.scenes.append(scene)
                except Exception as e:
                    print(f"Error loading scene {scene_root.name}: {e}")
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[:max(1, round(len(self.scenes) * cfg.fraction))]
        logger.info(
            f"Built {len(self.scenes)} ScanNet++ scenes with a total of {sum([len(scene) for scene in self.scenes])} pairs."
        )

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx: int):
        idx = idx % len(self.scenes)
        return random.choice(self.scenes[idx])
