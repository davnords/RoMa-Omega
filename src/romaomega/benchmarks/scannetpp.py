from dataclasses import dataclass

from romaomega.datasets.scannetpp import ScanNetPlusPlus
from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.transforms import Transform


class ScanNetPlusPlusBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: ScanNetPlusPlus.Cfg = ScanNetPlusPlus.Cfg(
            split="val",
            weight=1000,
        )
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = ScanNetPlusPlus(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
