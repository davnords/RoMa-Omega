import tyro
import torch

from romaomega.vggtroma import VGGTRoMa
from romaomega.random import set_seed
from romaomega.benchmarks.run_dense_benchmarks import run_dense_benchmarks


def main(
    run_path: str | None = None,
    load_avg_weights: bool = False,
):
    set_seed(1)
    torch.backends.cuda.preferred_linalg_library(torch._C._LinalgBackend.Magma)

    if run_path is not None:
        model = VGGTRoMa.from_run(run_path, strict=False, load_avg_weights=load_avg_weights)
    else:
        model = VGGTRoMa()
    run_dense_benchmarks(model)


if __name__ == "__main__":
    tyro.cli(main)
