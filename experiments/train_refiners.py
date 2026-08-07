import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.amp.grad_scaler
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import tyro
from tqdm import tqdm

import romaomega.distrib as romav3
from romaomega.benchmarks.hypersim import HyperSimBenchmark
from romaomega.benchmarks.mega import MegaDepthBenchmark
from romaomega.benchmarks.scannetpp import ScanNetPlusPlusBenchmark
from romaomega.datasets.hypersim import HyperSim
from romaomega.datasets.megadepth import MegaDepth
from romaomega.datasets.scannetpp import ScanNetPlusPlus
from romaomega.datasets.transforms import Transform
import wandb
from romaomega.device import device
from romaomega.distrib import init_distributed, is_main_process
from romaomega.features import FineFeatures
from romaomega.logging import logger
from romaomega.loss import Loss
from romaomega.matcher import Matcher
from romaomega.optimizer import Optimizer, create_warmup_scheduler
from romaomega.refiner import Refiners
from romaomega.vggtroma import VGGTRoMa as RoMav3
from romaomega.run import (
    DenseCfg,
    TrainData,
)
from romaomega.run import (
    dump_run,
    load_run,
    setup_data,
    setup_run,
    train_step,
)


@dataclass(frozen=True)
class Cfg(DenseCfg):
    model: RoMav3.Cfg = RoMav3.Cfg(
        stage="refiners",
        refiners=Refiners.Cfg(confidence_dim=4),
        refiner_features=FineFeatures.Cfg(patch_size=4),
        matcher=Matcher.Cfg(),
    )
    train_data: TrainData = TrainData()
    # Refiners train at the fixed 640x640 refine resolution with shake aug.
    # In the current data pipeline the global transform is merged into every
    # dataset (run._init_train_datasets), so this replaces the old
    # RefineTrainData(transform=RefineCfg(...)) per-dataset pattern.
    transform: Transform.Cfg = Transform.RefineCfg()
    # NOTE: point this at a VGGTRoMa matcher run (its matcher cfg must match
    # cfg.model.matcher below, including vggt_checkpoint).
    matcher_run_path: str = "experiments/vggtroma/runs/vggtroma/2026-06-12_20-28-37/step_300000"
    batch_size: int = 8
    loss: Loss.Cfg = Loss.Cfg()
    strict_matcher_load: bool = False
    # HyperSimBenchmark.transform_cfg defaults to Transform.BenchmarkCfg
    # (640x640, no augmentation), which matches the old per-dataset transform.
    benchmark: HyperSimBenchmark.Cfg = HyperSimBenchmark.Cfg(
        dataset=HyperSim.Cfg(
            split="val",
            weight=500,
            sample_mode="frame_distance",
        )
    )
    benchmark_train: HyperSimBenchmark.Cfg = HyperSimBenchmark.Cfg(
        dataset=HyperSim.Cfg(
            split="train",
            weight=500,
        )
    )
    # Short dense-PCK trackers on the standard mega / scannet eval sets. weight
    # is the number of pairs per eval, kept small so these stay quick.
    benchmark_mega: MegaDepthBenchmark.Cfg = MegaDepthBenchmark.Cfg(
        dataset=MegaDepth.Cfg(
            split="dedode_test",
            weight=200,
        )
    )
    benchmark_scannetpp: ScanNetPlusPlusBenchmark.Cfg = ScanNetPlusPlusBenchmark.Cfg(
        dataset=ScanNetPlusPlus.Cfg(
            split="val",
            weight=200,
        )
    )
    ema_decay: float = 0.999


def main(_cfg: Cfg):
    is_resumed = _cfg.resume_run is not None
    if not is_resumed and not _cfg.dry_run:
        assert _cfg.name is not None, "name is required"
    if _cfg.resume_run is not None:
        cfg, step, weights, optimizer_state = load_run(Path(_cfg.resume_run))
    else:
        cfg = _cfg
        step = 0
        weights = None
        optimizer_state = None
    # distributed / cudnn / matmul precision
    init_distributed()
    torch.set_num_threads(cfg.omp_num_threads)
    logger.info(f"Using {torch.get_num_threads()} OpenMP threads")
    os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("CUDNN benchmark enabled")
    torch.set_float32_matmul_precision("highest")

    # model, optimizer, loss, and grad scaler setup
    matcher_run_path = Path(cfg.matcher_run_path)
    from romaomega.types import _load_cfg as load_cfg

    matcher_cfg = load_cfg(
        type(cfg.model.matcher),
        json.load(open(matcher_run_path / "cfg.json", "r"))["model"]["matcher"],
        strict=cfg.strict_matcher_load,
    )
    # check that current config matches the one we load for weights
    assert cfg.model.matcher == matcher_cfg, (
        f"{cfg.model.matcher} doesn't match loaded cfg {matcher_cfg}"
    )

    matcher_weight_path = matcher_run_path / "weights.pth"
    model = RoMav3(cfg=cfg.model).to(device)
    stage_one_weights = torch.load(matcher_weight_path, map_location=device)
    missing, unexpected = model.load_state_dict(stage_one_weights, strict=False)
    for k in missing:
        assert k.startswith("refiner"), k
    assert not unexpected, unexpected
    for p in model.matcher.parameters():
        p.requires_grad = False

    if is_resumed:
        assert weights is not None
        model.load_state_dict(weights)
        del weights
    d_model = (
        torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=False)
        if romav3.is_distributed()
        else model
    )
    optimizer = Optimizer.from_config(cfg=cfg.opt, model=model)
    if is_resumed:
        assert optimizer_state is not None
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state
    scheduler = create_warmup_scheduler(optimizer, cfg.warmup_steps, last_step=step - 1)
    loss = Loss(cfg=cfg.loss)
    grad_scaler = torch.amp.grad_scaler.GradScaler(
        init_scale=65536,
        growth_interval=2000,
        backoff_factor=1 - 1e-8,
        growth_factor=1 + 1e-8,
    )
    averaged_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True
    )

    # dataset setup
    sampler, loader = setup_data(cfg=cfg)

    # benchmark setup
    benchmarks = [
        HyperSimBenchmark(cfg.benchmark),
        HyperSimBenchmark(cfg.benchmark_train),
        MegaDepthBenchmark(cfg.benchmark_mega),
        ScanNetPlusPlusBenchmark(cfg.benchmark_scannetpp),
    ]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(f"experiments/vggtroma/runs/{cfg.name}/{now}")

    if is_main_process() and not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_run(
            run_dir=run_dir,
            cfg=cfg,
            step=step,
            weights=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )
        if is_resumed:
            with open(run_dir / "resumed_from.txt", "w") as f:
                f.write(str(_cfg.resume_run))
        with open(run_dir / "matcher_step.txt", "w") as f:
            with open(matcher_run_path / "step.txt", "r") as m:
                matcher_step = int(m.read())
                f.write(str(matcher_step))
    wandb.init(
        project="romav3",
        config=asdict(cfg),
        name=cfg.name,
        mode="online"
        if cfg.wandb and not cfg.dry_run and is_main_process()
        else "disabled",
    )
    num_epochs = cfg.num_steps // len(loader) + 1
    epoch = step // len(loader)
    # The model already self-compiles in __init__ (cfg.model.compile / COMPILE=1).
    # Do NOT wrap it in a second torch.compile here -- nesting torch.compile over
    # an already-compiled module causes graph breaks / recompiles and is ~3x
    # slower. Pass the plain d_model (matches the reference train_loop).
    c_d_model = d_model
    for epoch in range(epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in (pbar := tqdm(loader)):
            if step >= cfg.num_steps:
                break
            model.train(True)
            batch = batch.to(device)
            loss_value = train_step(
                batch=batch,
                model=c_d_model,  # type: ignore
                step=step,
                loss=loss,
                grad_scaler=grad_scaler,
                optimizer=optimizer,
                cfg=cfg,
                scheduler=scheduler,
                run_name=cfg.name,
            )
            averaged_model.update_parameters(model)
            if step % cfg.ckpt_interval == 0 and is_main_process() and not cfg.dry_run:
                dump_run(
                    run_dir=run_dir,
                    cfg=cfg,
                    step=step,
                    weights=model.state_dict(),
                    avg_weights=averaged_model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    make_step_dir=(step % cfg.step_dir_interval == 0),
                )
            if (
                step % cfg.eval_interval == 0
                and is_main_process()
                and not cfg.no_test
                and step > 0
            ):
                for benchmark in benchmarks:
                    benchmark_result = benchmark(model=averaged_model, step=step)
                    wandb.log(benchmark_result, step=step)
                    logger.info(f"Benchmark results at step {step}: {benchmark_result}")
            step += 1
            pbar.set_description(f"Loss: {loss_value:.4f}")
        if step >= cfg.num_steps:
            break
    dump_run(
        run_dir=run_dir,
        cfg=cfg,
        step=step,
        weights=model.state_dict(),
        avg_weights=averaged_model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        make_step_dir=False,
    )
    if is_main_process():
        wandb.finish()


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "0"
    cfg = tyro.cli(Cfg)
    try:
        main(cfg)
    except KeyboardInterrupt as e:
        dist.destroy_process_group()
        raise e
