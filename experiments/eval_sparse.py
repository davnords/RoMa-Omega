import json
from datetime import date
from pathlib import Path
from typing import Literal

import tyro

from hardmatch import HardMatchBenchmark

from romaomega.matchers import MODEL_REGISTRY
from romaomega.random import set_seed
from romaomega.benchmarks.general import (
    WxBSBenchmark,
    RubikBenchmark,
)


def main(
    name: Literal[
        "vggt_omega",
        "roma_omega",
    ] = "roma_omega",
    benchmark: Literal[
        "wxbs",
        "hardmatch",
        "rubik",
    ] = "wxbs",
):
    set_seed(1)
    Path("data").mkdir(exist_ok=True)

    model = MODEL_REGISTRY[name]()
    if benchmark == "wxbs":
        wxbs = WxBSBenchmark()
        res = wxbs.benchmark(model)
        print(res)
    elif benchmark == "hardmatch":
        hardmatch = HardMatchBenchmark()
        res = hardmatch.benchmark(model)
        print(res)
    elif benchmark == "rubik":
        rubik = RubikBenchmark()
        res = rubik.benchmark(model)
        print(res)
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    out_dir = Path("results") / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_{benchmark}.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
