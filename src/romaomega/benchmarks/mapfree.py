from dataclasses import dataclass

from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.mapfree import MapFree
from romaomega.datasets.transforms import Transform


class MapFreeBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: MapFree.Cfg = MapFree.Cfg(
            split="val",
            weight=1000,
        )
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = MapFree(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
