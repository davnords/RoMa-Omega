from dataclasses import dataclass

from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.aerialmegadepth import AerialMegaDepth
from romaomega.datasets.transforms import Transform


class AerialMegaDepthBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: AerialMegaDepth.Cfg = AerialMegaDepth.Cfg(
            split="dedode_test", weight=1000
        )
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = AerialMegaDepth(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
