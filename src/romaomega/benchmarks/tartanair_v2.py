from dataclasses import dataclass

from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.tartanair_v2 import TartanAirV2
from romaomega.datasets.transforms import Transform


class TartanAirV2Benchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: TartanAirV2.Cfg = TartanAirV2.Cfg(
            split="test",
            weight=1000,
        )
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = TartanAirV2(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
