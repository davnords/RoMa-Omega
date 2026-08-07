import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from romaomega.datasets.colmap_dataset import (
    COLMAPDataset,
    COLMAPScene,
)
from romaomega.datasets.transforms import Transform
from typing import Literal
from tqdm import tqdm


def _load_colmap_depth(depth_file):
    with open(depth_file, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


class MapFreeScene(COLMAPScene):
    def load_depth(self, depth_path: str | Path) -> torch.Tensor:
        if isinstance(depth_path, Path):
            depth_path = depth_path.as_posix()
        bin_path = depth_path + ".geometric.bin"
        depth = _load_colmap_depth(bin_path)
        return torch.from_numpy(depth)


class MapFree(COLMAPDataset[MapFreeScene]):
    @dataclass(frozen=True)
    class Cfg(COLMAPDataset.Cfg):
        split: Literal["train", "val"]
        data_root: str = "data/map-free"

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg) -> None:
        super().__init__(cfg=cfg, transform_cfg=transform_cfg)
        if cfg.split == "train":
            self.min_overlap = 0.01
            self.max_overlap = 1.0
        elif cfg.split == "val":
            self.min_overlap = 0.01
            self.max_overlap = 1.0
        else:
            raise TypeError(f"Split {cfg.split} not available")
        self.scenes = self.build_scenes(cfg.split, 0.01, 1.0, 100_000)
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[:max(1, round(len(self.scenes) * cfg.fraction))]
        self.cum_weights = torch.linspace(1 / len(self.scenes), 1, len(self.scenes))

    def build_scenes(
        self,
        scene_split: str,
        min_overlap: float,
        max_overlap: float,
        max_num_pairs: int,
    ) -> list[MapFreeScene]:
        self.broken_scenes = set(["s00000_seq0.npy", "s00369_seq0.npy"])
        self.val_scenes = set(["s00015.npy", "s00022.npy"])
        sequence = "seq0"
        scenes: list[MapFreeScene] = []
        scene_names = sorted(self.all_scenes)
        for scene_name in tqdm(
            scene_names, desc=f"Loading {type(self).__name__} {scene_split} scenes..."
        ):
            if sequence not in scene_name:
                continue
            if scene_name in self.broken_scenes:
                continue
            scene_name_no_partial = scene_name[:-9] + scene_name[-4:]
            if scene_split == "train" and scene_name_no_partial in self.val_scenes:
                continue
            elif scene_split == "val" and scene_name_no_partial not in self.val_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                self.scene_info_root / scene_name, allow_pickle=True
            ).item()
            scenes.append(
                MapFreeScene(
                    self.data_root,
                    scene_info,
                    min_overlap=min_overlap,
                    scene_name=scene_name,
                    max_overlap=max_overlap,
                    max_num_pairs=max_num_pairs,
                    transform_cfg=self.transform_cfg,
                    use_sky_mask=self.cfg.use_sky_mask,
                )
            )
        return scenes
