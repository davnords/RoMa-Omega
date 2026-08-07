from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import tyro
import wandb

from romaomega.device import device
from romaomega.distrib import init_distributed, is_main_process, is_distributed
from romaomega.logging import logger
from romaomega.loss import Loss
from romaomega.optimizer import Optimizer, create_warmup_scheduler
from romaomega.vggtroma import VGGTRoMa
from romaomega.run import (
    load_run,
    DenseCfg,
    Benchmarks,
    setup_data,
    setup_run,
    create_grad_scaler,
    create_averaged_model,
    init_wandb,
    init_benchmarks,
    create_run_dir,
    train_loop,
    set_torch_misc,
)


@dataclass(frozen=True)
class MatcherCfg(DenseCfg):
    model: VGGTRoMa.Cfg = VGGTRoMa.Cfg(
        stage="matcher",
    )
    wandb_entity: str = "georgs-team"
    benchmarks: Benchmarks = Benchmarks()


def main(_cfg: MatcherCfg):
    is_resumed = _cfg.resume_run is not None
    if not is_resumed and not _cfg.dry_run:
        assert _cfg.name is not None, "name is required"

    if _cfg.resume_run is not None:
        cfg, step, weights, optimizer_state, scheduler_state, averaged_model_state = load_run(
            Path(_cfg.resume_run)
        )
    else:
        cfg = _cfg
        step = 0
        weights = None
        optimizer_state = None
        scheduler_state = None
        averaged_model_state = None

    # distributed / cudnn / matmul precision
    init_distributed()
    set_torch_misc(cfg)
    # model, optimizer, loss, and grad scaler setup
    model = VGGTRoMa(cfg.model).to(device)
    if is_resumed:
        assert weights is not None
        model.load_state_dict(weights)
        del weights
    d_model = (
        torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=False)
        if is_distributed()
        else model
    )
    optimizer = Optimizer.from_config(cfg=cfg.opt, model=model)
    if is_resumed:
        assert optimizer_state is not None
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state
    scheduler = create_warmup_scheduler(
        optimizer,
        cfg.warmup_steps,
        last_step=step - 1,
        num_steps=cfg.num_steps,
        decay=cfg.lr_decay,
    )
    if is_resumed and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        del scheduler_state
    loss = Loss(cfg=cfg.loss)
    grad_scaler = create_grad_scaler()
    averaged_model = create_averaged_model(model, ema_decay=cfg.ema_decay)
    if is_resumed and averaged_model_state is not None:
        averaged_model.load_state_dict(averaged_model_state)
        del averaged_model_state

    # dataset setup
    sampler, loader = setup_data(cfg=cfg)

    # benchmark setup
    benchmarks = init_benchmarks(cfg.benchmarks) if cfg.benchmarks is not None else None

    # eval callback for benchmarks
    def matcher_eval_callback(eval_model: nn.Module, step: int, run_dir: Path) -> None:
        if benchmarks is not None:
            for benchmark in benchmarks:
                benchmark_result = benchmark(model=eval_model, step=step)
                wandb.log(benchmark_result, step=step)
                logger.info(f"Benchmark results at step {step}: {benchmark_result}")

    # run directory and wandb
    run_dir = create_run_dir("experiments/vggtroma/runs", cfg.name)
    if is_main_process() and not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run, wandb_url = init_wandb(
        entity=cfg.wandb_entity,
        project="romav3",
        config=asdict(cfg),
        name=cfg.name,
        enabled=cfg.wandb,
        dry_run=cfg.dry_run,
    )
    if is_main_process() and not cfg.dry_run:
        setup_run(
            run_dir=run_dir,
            cfg=cfg,
            step=step,
            weights=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            wandb_url=wandb_url,
        )
        if is_resumed:
            with open(run_dir / "resumed_from.txt", "w") as f:
                f.write(str(_cfg.resume_run))

    # training loop
    train_loop(
        cfg=cfg,
        model=model,
        d_model=d_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        grad_scaler=grad_scaler,
        averaged_model=averaged_model,
        sampler=sampler,
        loader=loader,
        run_dir=run_dir,
        step=step,
        eval_callback=matcher_eval_callback,
    )


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "0"
    cfg = tyro.cli(MatcherCfg)
    try:
        main(cfg)
    except KeyboardInterrupt as e:
        dist.destroy_process_group()
        raise e
