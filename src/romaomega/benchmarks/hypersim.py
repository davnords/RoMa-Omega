from dataclasses import dataclass

from romaomega.datasets.hypersim import HyperSim
from romaomega.datasets.transforms import Transform
from romaomega.benchmarks._dense import DenseBenchmark


class HyperSimBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: HyperSim.Cfg = HyperSim.Cfg(
            split="val", weight=1000, sample_mode="frame_distance"
        )
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = HyperSim(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
