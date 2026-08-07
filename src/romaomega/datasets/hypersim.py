from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Literal
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from romaomega.datasets.transforms import Transform
from romaomega.geometry import get_pixel_grid, to_homogeneous
from romaomega.logging import logger
from romaomega.types import Batch, SampleMode
import OpenEXR


def homog_pixel_grid(*, H: int, W: int) -> torch.Tensor:
    return (
        to_homogeneous(
            get_pixel_grid(
                1,
                H=H,
                W=W,
                overload_device=torch.device("cpu"),
            )
        ).reshape(-1, 3)
        # .T
        # .numpy()
    )


def _read_exr_depth(path: Path) -> np.ndarray:
    with OpenEXR.File(path.as_posix()) as file:
        depth = file.channels()["Y"].pixels.astype(np.float32)
        # follow https://github.com/facebookresearch/map-anything/blob/5ffcb617601a40687e0daf06cd625959adeb4472/mapanything/datasets/wai/tav2_wb.py#L128C13-L128C80
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def _write_exr_depth(path: Path, depth: np.ndarray) -> None:
    channels = {"Y": depth}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}

    with OpenEXR.File(header, channels) as file:
        file.write(path.as_posix())
        # file.channels()['Y'].pixels = depth


class HyperSimScene(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_root: Path,
        scene: str,
        K: np.ndarray,
        R: np.ndarray,
        sample_mode: SampleMode,
        max_frame_distance: int | None,
        overlaps: np.ndarray | None,
        overlap_threshold: float | None,
        transform_cfg: Transform.Cfg,
        use_opt_lighting: bool,
    ) -> None:
        super().__init__()
        if transform_cfg.shake_t != 0:
            raise ValueError("Shake t must be 0")
        self.transform = Transform(transform_cfg)
        self.max_frame_distance = max_frame_distance
        self.scene_root = data_root / scene
        metadata_scene = self.scene_root / "_detail" / "metadata_scene.csv"
        # get first camera name
        camera_name = sorted((self.scene_root / "_detail").glob("cam_*"))[0].name
        df = pd.read_csv(metadata_scene)
        self.meters_per_asset = df.loc[
            df["parameter_name"] == "meters_per_asset_unit", "parameter_value"
        ].iloc[0]  # ty: ignore[non-subscriptable]
        self.intrinsic = K
        self.R = R
        self.overlaps = overlaps
        self.overlap_threshold = overlap_threshold
        self.use_opt_lighting = use_opt_lighting
        self.image_paths = sorted(
            glob(
                (
                    self.scene_root
                    / "images"
                    / f"scene_{camera_name}_final_preview"
                    / (
                        "frame.*.color.jpg"
                        if not use_opt_lighting
                        else "frame.*.color_opt.jpg"
                    )
                ).as_posix()
            )
        )
        self.distance_paths = sorted(
            glob(
                (
                    self.scene_root
                    / "images"
                    / f"scene_{camera_name}_geometry_hdf5"
                    / "frame.*.depth_meters.hdf5"
                ).as_posix()
            )
        )
        self.distance_paths = {int(dp.split(".")[-3]): dp for dp in self.distance_paths}
        self.image_paths = {int(ip.split(".")[-3]): ip for ip in self.image_paths}
        self.image_ids = set(self.distance_paths.keys()).intersection(
            self.image_paths.keys()
        )
        if len(self.image_ids) == 0:
            raise ValueError("No shared image/depth paths.")

        camera_root = self.scene_root / "_detail" / camera_name
        camera_positions_hdf5_file = camera_root / "camera_keyframe_positions.hdf5"
        camera_orientations_hdf5_file = (
            camera_root / "camera_keyframe_orientations.hdf5"
        )
        with (
            h5py.File(camera_positions_hdf5_file, "r") as h5_pos,
            h5py.File(camera_orientations_hdf5_file, "r") as h5_rots,
        ):  # type: ignore
            camera_positions: np.ndarray = h5_pos["dataset"][:]  # type: ignore
            rots: np.ndarray = h5_rots["dataset"][:]  # type: ignore
            rots = rots.transpose((0, 2, 1))
            translations = -rots @ camera_positions[..., None]
            self.poses = np.zeros((len(rots), 4, 4))
            self.poses[:, 3, 3] = 1.0
            self.poses[:, :3, :3] = R[None] @ rots
            self.poses[:, :3, 3:] = R[None] @ translations
        # NOTE: cannot do the below assert, because some depths/images are missing
        # Also not so great that the poses are not labeled, but I assume its the same order...
        # assert len(self.distance_paths) == len(self.poses)
        self._cached_grid = None
        self.idx_to_image_id = {
            idx: img_id for idx, img_id in enumerate(self.image_ids)
        }
        self.image_id_to_idx = {
            img_id: idx for idx, img_id in enumerate(self.image_ids)
        }
        self.sample_mode: SampleMode = sample_mode

    def __len__(self):
        return len(self.image_ids)

    def load_distance(self, distance_path) -> np.ndarray:
        # dynamically convert to exr if not already
        exr_path = Path(distance_path).with_suffix(".exr")
        if exr_path.exists():
            try:
                return _read_exr_depth(exr_path)
            except Exception as e:
                logger.warning(f"Error reading exr depth: {e}")
                pass
        with h5py.File(distance_path, "r") as x:
            data: np.ndarray = x["dataset"][:]  # type: ignore
            _write_exr_depth(exr_path, data)
        return data

    def depth_from_distance(
        self, distance: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        H, W = distance.shape[0], distance.shape[1]
        if self._cached_grid is None:
            # TODO: this assumes all images in same scene are same size, are they?
            self._cached_grid = homog_pixel_grid(H=H, W=W)
        rays = self._cached_grid @ torch.linalg.inv(K).mT  # HWx3
        ray_z = rays[..., -1] / torch.linalg.norm(rays, dim=-1)

        # rays = self.grid @ torch.linalg.inv(K).mT   # HWx3
        # return distance.reshape(H, W, 1)
        # ray_z = rays[...,-1] / torch.linalg.norm(rays, dim=-1)
        z = distance.reshape(-1) * ray_z
        return z.reshape(H, W, 1)

    def __getitem__(self, idx: int) -> Batch:
        try:
            return self.load_item(idx)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.load_item(idx % len(self.image_ids))

    def load_item(self, idx):
        # get length of sequence to sample
        # sample random pair with max distance 10 (take care not to sample idx twice)
        _idx1 = idx
        key1 = self.idx_to_image_id[_idx1]
        if self.sample_mode == "frame_distance":
            nearby_image_ids = [
                img_id
                for img_id in self.image_ids.difference([key1])
                if (
                    True
                    if self.max_frame_distance is None
                    else abs(img_id - key1) <= self.max_frame_distance
                )
            ]
        elif self.sample_mode == "overlap":
            assert self.overlaps is not None, (
                "Overlaps must be provided for overlap mode"
            )
            nearby_image_ids = [
                img_id
                for img_id in self.image_ids.difference([key1])
                if self.overlaps[_idx1, self.image_id_to_idx[img_id]]
                > self.overlap_threshold
            ]
        else:
            raise TypeError(f"Unknown sample mode: {self.sample_mode}")
        if len(nearby_image_ids) == 0:
            return self[np.random.choice(len(self))]
        key2 = np.random.choice(nearby_image_ids)

        # all intrinsics are the same
        K_A = torch.tensor(self.intrinsic).reshape(3, 3).float()
        K_B = torch.tensor(self.intrinsic).reshape(3, 3).float()

        # read and compute relative poses
        T_A = torch.tensor(self.poses[key1]).float()
        T_B = torch.tensor(self.poses[key2]).float()
        T_AB = T_B @ torch.linalg.inv(T_A)
        # load images/depths
        im_A_path, im_B_path = (
            Path(self.image_paths[key1]),
            Path(self.image_paths[key2]),
        )
        depth_A_path, depth_B_path = (
            self.distance_paths[key1],
            self.distance_paths[key2],
        )
        pil_img_A = Image.open(im_A_path)
        pil_img_B = Image.open(im_B_path)
        # rescale intrinsics (for tartanair im_A and im_B should always be same size, but keeping it general)
        distance_A = (
            torch.tensor(self.load_distance(depth_A_path)).float()
            / self.meters_per_asset
        )
        distance_B = (
            torch.tensor(self.load_distance(depth_B_path)).float()
            / self.meters_per_asset
        )
        # FIXME
        # return {"im_A": torch.tensor(np.array(pil_im_A)), "im_B": torch.tensor(np.array(pil_im_B)), "depth_A": torch.tensor(np.array(distance_A)), "depth_B": torch.tensor(np.array(distance_B)), "K_A": K_A, "K_B": K_B}

        # Process images
        ret = self.transform(
            pil_img_A=pil_img_A,
            pil_img_B=pil_img_B,
            depth_A=distance_A,
            depth_B=distance_B,
            K_A=K_A,
            K_B=K_B,
        )
        ret["depth_A"] = self.depth_from_distance(ret["depth_A"], ret["K_A"]).float()
        ret["depth_B"] = self.depth_from_distance(ret["depth_B"], ret["K_B"]).float()
        # Set NaN depths to 0
        ret["depth_A"][ret["depth_A"].isnan()] = 0
        ret["depth_B"][ret["depth_B"].isnan()] = 0
        return Batch(
            **ret,
            pose_A=T_A,
            pose_B=T_B,
            T_AB=T_AB,
            img_A_path=im_A_path,
            img_B_path=im_B_path,
            source="depth",
            quality="high",
        )


class HyperSim(Dataset[Batch]):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["train", "val"]
        data_root: str = "data/ml-hypersim/contrib/mikeroberts3000/jupyter"
        # Dataset-specific transform overrides (merged with global transform at setup time)
        transform_overrides: dict[str, Any] = field(default_factory=dict)
        scene_names: list[str] | None = None
        sample_mode: SampleMode = "overlap"
        max_frame_distance: int | None = None
        overlap_path: str | None = "data/ml-hypersim/scene_overlaps_symm_geometric.npy"
        overlap_threshold: float | None = 0.05
        weight: int = 1_000_000
        use_opt_lighting: bool = True  # Changed to True Jan 16 2026
        fraction: float = 1.0

    def __init__(
        self,
        cfg: Cfg,
        transform_cfg: Transform.Cfg,
    ) -> None:
        self.cfg = cfg
        self.transform_cfg = transform_cfg
        self.data_root = cfg.data_root
        self.scenes: list[HyperSimScene] = []
        self.ignore_scenes = [
            "ai_003_001",  # images are completely black
        ]
        data_root = Path(cfg.data_root)
        if cfg.sample_mode == "overlap":
            assert cfg.overlap_path is not None, (
                "Overlap path must be provided for overlap mode"
            )
            overlaps = np.load(cfg.overlap_path, allow_pickle=True).item()
        else:
            overlaps = None
        metadata_camera_parameters_csv_file = (
            data_root / "metadata_camera_parameters.csv"
        )
        df_camera_parameters = pd.read_csv(
            metadata_camera_parameters_csv_file, index_col="scene_name"
        )
        if cfg.scene_names is None:
            if cfg.split == "train":
                scene_names = {f"ai_{i:03d}" for i in range(50)}
            elif cfg.split == "val":
                scene_names = {f"ai_{i:03d}" for i in range(51, 56)}
            else:
                raise ValueError("split not known")
        else:
            scene_names = cfg.scene_names
        for scene_path in tqdm(
            list(data_root.iterdir()),
            desc=f"Loading {type(self).__name__} {cfg.split} scenes...",
        ):
            scene_name = scene_path.name
            if (scene_name[:-4] not in scene_names) and (scene_name not in scene_names):
                continue
            if scene_name in self.ignore_scenes:
                continue
            df_: pd.Series = df_camera_parameters.loc[scene_name]  # type: ignore
            width_pixels = int(df_["settings_output_img_width"])
            height_pixels = int(df_["settings_output_img_height"])

            M_proj = [
                [
                    df_["M_proj_00"],
                    df_["M_proj_01"],
                    df_["M_proj_02"],
                    df_["M_proj_03"],
                ],
                [
                    df_["M_proj_10"],
                    df_["M_proj_11"],
                    df_["M_proj_12"],
                    df_["M_proj_13"],
                ],
                [
                    df_["M_proj_20"],
                    df_["M_proj_21"],
                    df_["M_proj_22"],
                    df_["M_proj_23"],
                ],
                [
                    df_["M_proj_30"],
                    df_["M_proj_31"],
                    df_["M_proj_32"],
                    df_["M_proj_33"],
                ],
            ]
            M_proj = np.array(M_proj)
            M_screen_from_ndc = np.array(
                [
                    [0.5 * (width_pixels), 0, 0, 0.5 * (width_pixels)],
                    [0, -0.5 * (height_pixels), 0, 0.5 * (height_pixels)],
                    [0, 0, 0.5, 0.5],  # doesn't matter
                    [0, 0, 0, 1.0],
                ]
            )
            x = (M_screen_from_ndc @ M_proj)[[0, 1, 3]]
            K, R = cv2.decomposeProjectionMatrix(x)[:2]  # type: ignore
            K = K / K[2, 2]

            try:
                self.scenes.append(
                    HyperSimScene(
                        data_root=data_root,
                        scene=scene_name,
                        K=K,
                        R=R,
                        transform_cfg=self.transform_cfg,
                        sample_mode=cfg.sample_mode,
                        max_frame_distance=cfg.max_frame_distance,
                        overlaps=None if overlaps is None else overlaps[scene_name],
                        overlap_threshold=cfg.overlap_threshold,
                        use_opt_lighting=cfg.use_opt_lighting,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Got exception {e} while making hypersim scene, continuing."
                )
        logger.info(
            f"Built {len(self.scenes)} HyperSim scenes with a total of {sum([len(scene) for scene in self.scenes])} pairs."
        )
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[:max(1, round(len(self.scenes) * cfg.fraction))]
        self.cum_weights = torch.linspace(1 / len(self.scenes), 1, len(self.scenes))

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx: int) -> Batch:
        scene_index = torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self))
        scene = self.scenes[scene_index]
        pair_idx = np.random.choice(range(len(scene)))
        return scene[pair_idx]
