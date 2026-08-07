from romaomega.benchmarks import (
    AerialMegaDepthBenchmark,
    MapFreeBenchmark,
    TartanAirV2Benchmark,
    MegaDepthBenchmark,
    ScanNetPlusPlusBenchmark,
    FlyingThings3DBenchmark,
)

def run_dense_benchmarks(model):
    benchmarks = [
        AerialMegaDepthBenchmark(AerialMegaDepthBenchmark.Cfg(seed=1337, num_workers=16)),
        MegaDepthBenchmark(MegaDepthBenchmark.Cfg(seed=1337, num_workers=16)),
        TartanAirV2Benchmark(TartanAirV2Benchmark.Cfg(seed=1337, num_workers=8)),
        ScanNetPlusPlusBenchmark(ScanNetPlusPlusBenchmark.Cfg(seed=1337, num_workers=16)),
        FlyingThings3DBenchmark(FlyingThings3DBenchmark.Cfg(seed=1337, num_workers=16)),
        MapFreeBenchmark(MapFreeBenchmark.Cfg(seed=1337, num_workers=16)),
    ]
    results = {}
    for benchmark in benchmarks:
        res = benchmark(model, 0)
        print(res)
        results[benchmark.prefix] = res
    with open("dense_benchmarks_results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
