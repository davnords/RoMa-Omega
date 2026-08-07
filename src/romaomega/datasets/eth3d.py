from dataclasses import dataclass
from PIL import Image
import os.path as osp
from typing import Literal
from .transforms import Transform
import torch.utils.data
from romaomega.types import Batch
import numpy as np
import logging
import os

logger = logging.getLogger("__name__")


# Reference code: https://github.com/UniFlowMatch/UFM/blob/benchmark/uniflowmatch/datasets/eth3d.py
# Now we have downloaded the processed data to /mimer/NOBACKUP/groups/3d-dl/ufm_benchmarks which should be jointly accessible for all of us
# However, if you want to download the data yourself, follow the instructions at: https://github.com/UniFlowMatch/UFM/tree/benchmark
class ETH3D(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["test"]
        data_root: str = "/mimer/NOBACKUP/groups/3d-dl/ufm_benchmarks/eth3d_processed"  # Temporary path for convenience, change as needed
        scene_names: list[str] | None = None
        weight: int = 1_000_000

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg):
        super().__init__()
        self.cfg = cfg
        self.transform = Transform(transform_cfg)

        assert cfg.split == "test", "Only 'test' split is supported for ETH3D dataset."
        pairs_path = os.path.join(
            cfg.data_root, "..", "anymap_data_pairs", "eth3d_valid_pairs.npz"
        )
        pairs_data = np.load(pairs_path, allow_pickle=True)
        pairs = pairs_data["data"]
        self.pairs = pairs
        self.scenes = np.unique(self.pairs[:, 0])

        print(
            "Loaded ETH3D with {} image pairs from {} scenes.".format(
                len(self.pairs), len(self.scenes)
            )
        )

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx):
        try:
            return self.load_item(idx)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.load_item(idx % len(self.pairs))

    def load_item(self, idx):
        idx = idx % len(self.pairs)
        scene_name, img1_name, img2_name, score = self.pairs[idx]
        scene_folder = osp.join(self.cfg.data_root, scene_name)

        img_paths = []
        depthmaps = []
        poses = []
        ks = []
        for view_name in [img1_name, img2_name]:
            # Load the RGB image
            img_path = osp.join(scene_folder, "undistorted_images", view_name + ".jpg")
            # Load the depth data
            depth_path = osp.join(
                scene_folder, "undistorted_depths", view_name + ".npy"
            )
            depthmap = np.load(depth_path).astype(np.float32)
            # Load the camera parameters
            params_path = osp.join(
                scene_folder, "undistorted_camera_params", view_name + ".npy"
            )
            fx, fy, cx, cy = np.load(params_path).astype(np.float32)
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(
                np.float32
            )
            # Load the pose
            pose_path = osp.join(scene_folder, "poses", view_name + ".npy")
            w2c_pose = np.load(pose_path).astype(np.float32)

            img_paths.append(img_path)
            depthmaps.append(torch.from_numpy(depthmap))
            poses.append(torch.from_numpy(w2c_pose))
            ks.append(torch.from_numpy(intrinsics))
        ret = self.transform(
            pil_img_A=Image.open(img_paths[0]),
            pil_img_B=Image.open(img_paths[1]),
            depth_A=depthmaps[0],
            depth_B=depthmaps[1],
            K_A=ks[0],
            K_B=ks[1],
        )
        T_A = poses[0].float()
        T_B = poses[1].float()
        T_AB = T_B @ torch.linalg.inv(T_A)
        return Batch(
            **ret,
            pose_A=T_A,
            pose_B=T_B,
            T_AB=T_AB,
            img_A_path=img_paths[0],
            img_B_path=img_paths[1],
            source="depth",
            quality="low",
        )
