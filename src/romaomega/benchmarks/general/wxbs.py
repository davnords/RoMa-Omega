from dataclasses import dataclass

import cv2
import numpy as np

from tqdm import tqdm
from wxbs_benchmark.dataset import WxBSDataset
from wxbs_benchmark.evaluation import evaluate_Fs

from romaomega.types import BenchmarkMatcher

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class WxBSBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        subset: str = "test"
        dataset_path: str = "data/WxBS"
        download: bool = True

    def __init__(self, cfg: Cfg | None = None) -> None  :
        if cfg is None:
            cfg = WxBSBenchmark.Cfg()
        self.subset = cfg.subset
        WxBSDataset.urls["v1.1"][0] = (
            "https://github.com/Parskatt/storage/releases/download/wxbs/WxBS_v1.1.zip"
        )

        def wrap(f):
            def __getitem__(self, idx):
                out = f(self, idx)
                return {
                    **out,
                    "imgfname1": self.pairs[idx][0],
                    "imgfname2": self.pairs[idx][1],
                }

            return __getitem__

        WxBSDataset.__getitem__ = wrap(WxBSDataset.__getitem__)
        self.dataset = WxBSDataset(
            cfg.dataset_path, subset=self.subset, download=cfg.download
        )

    def benchmark(
            self, 
            model: BenchmarkMatcher,
        ):
        Fs = []
        for pair_dict in tqdm(self.dataset):
            kpts1, kpts2 = model.match(pair_dict["imgfname1"], pair_dict["imgfname2"])
            kpts1 = kpts1 - model.offset
            kpts2 = kpts2 - model.offset

            try:
                F, _ = cv2.findFundamentalMat(kpts1, kpts2, cv2.USAC_MAGSAC, 0.25, 0.999, 100000)
            except:
                F = np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0]])

            if F is None:
                F = np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0]])
            Fs.append(F)

        result_dict, thresholds = evaluate_Fs(
            Fs, self.subset
        )

        avg_pck = result_dict["average"] 
        mAA_10px = avg_pck[:11].mean()

        print(f"mAA_10px: {mAA_10px:.4f}")

        return {
            "avg_pck": avg_pck.tolist(),
            "mAA_10px": float(mAA_10px),
        }