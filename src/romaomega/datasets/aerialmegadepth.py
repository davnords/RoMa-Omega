from romaomega.datasets.megadepth import MegaDepth
from romaomega.datasets.transforms import Transform
from dataclasses import dataclass


class AerialMegaDepth(MegaDepth):
    @dataclass(frozen=True)
    class Cfg(MegaDepth.Cfg):
        data_root: str = "data/aerialmegadepth"

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg) -> None:
        super().__init__(cfg=cfg, transform_cfg=transform_cfg)
