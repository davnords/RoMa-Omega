from dataclasses import dataclass

from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.flyingthings import FlyingThings3D
from romaomega.datasets.transforms import Transform


class FlyingThings3DBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: FlyingThings3D.Cfg = FlyingThings3D.Cfg(split="TEST", weight=1000)
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = FlyingThings3D(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
