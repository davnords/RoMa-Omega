from dataclasses import dataclass

from romaomega.benchmarks._dense import DenseBenchmark
from romaomega.datasets.eth3d import ETH3D
from romaomega.datasets.transforms import Transform

# I got this performance when I ran using RoMa v2:
# {'eth3d_test_epe': 6.83216064453125, 'eth3d_test_pck_1': 0.6360906982421874, 'eth3d_test_pck_3': 0.8465414428710938, 'eth3d_test_pck_5': 0.8895974731445313, 'eth3d_test_pck_32': 0.954237548828125, 'eth3d_test_bias_x': 0.03001789474487305, 'eth3d_test_bias_y': -0.014064806938171387, 'eth3d_test_nll': 0.04784785461425781}
# This is slightly worse than the numbers reported in the UFM paper, but still decent.


class ETH3DBenchmark(DenseBenchmark):
    @dataclass(frozen=True)
    class Cfg(DenseBenchmark.Cfg):
        dataset: ETH3D.Cfg = ETH3D.Cfg(
            split="test",
            weight=1000,
        )
        rel_depth_error_threshold: float = 0.05  # 05
        transform_cfg: Transform.BenchmarkCfg = Transform.BenchmarkCfg()

    def __init__(
        self,
        cfg: Cfg,
    ) -> None:
        super().__init__(cfg)
        self.dataset = ETH3D(cfg=cfg.dataset, transform_cfg=cfg.transform_cfg)
        self.prefix = f"{type(self).__name__}_{cfg.dataset.split}"
