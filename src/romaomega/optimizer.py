import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from torch.optim import AdamW
from romaomega.dense.romav3 import RoMaV3

class DualOptimizer(torch.optim.Optimizer):
    def __init__(self, opt1, opt2):
        self.optimizers = [opt1, opt2]
        param_groups = []
        for opt in self.optimizers:
            param_groups.extend(opt.param_groups)
        super().__init__(param_groups, defaults={})
    
    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    def state_dict(self):
        return {'optimizers': [opt.state_dict() for opt in self.optimizers]}
    
    def load_state_dict(self, state_dict):
        for opt, opt_state in zip(self.optimizers, state_dict['optimizers']):
            opt.load_state_dict(opt_state)

class Optimizer(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        lr: float = 4e-4
        weight_decay: float = 0.0
        beta1: float = 0.9
        beta2: float = 0.999
        eps: float = 1e-8
        fused: bool = True
        optimizer: Literal["adamw", "muon"] = "adamw"

    @classmethod
    def from_config(cls, cfg: Cfg, model: RoMaV3) -> torch.optim.Optimizer:
        requires_grad_params = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        if cfg.optimizer == "adamw":
            param_groups = [{"params": requires_grad_params, "lr": cfg.lr}]
            return AdamW(
                param_groups,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                fused=cfg.fused,
            )
        elif cfg.optimizer == "muon":
            # Muon version from KimiK2 from https://arxiv.org/pdf/2502.16982
            # this allows the same learning rate as for Adamw to be used
            from torch.optim import Muon
            # We want to filter out the embedding layer and the final projection
            projector_keywords = ["patch_embed", "head"]
            # Weights to ignore in muon:
            # This checks if the parameter name contains any of the keywords and if so excludes it
            projector_params = set([p for name, p in requires_grad_params if any(k in name for k in projector_keywords)])
            hidden_gains_biases = [p for _, p in requires_grad_params if p.ndim != 2 and p not in projector_params]
            # Weights to use muon for:
            hidden_weights = [p for _, p in requires_grad_params if p.ndim == 2 and p not in projector_params]

            adamw = AdamW(
                hidden_gains_biases+list(projector_params),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                fused=cfg.fused,
            )
            muon = Muon(hidden_weights, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw")
            return DualOptimizer(adamw, muon)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    last_step: int = -1,
    num_steps: int | None = None,
    decay: bool = False,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then optionally cosine-decay to ``min_lr_ratio * lr``.

    With ``decay=False`` this is the original warmup-then-flat schedule, so
    turning decay off restores the previous behaviour exactly.
    """
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if not decay or num_steps is None:
            return 1.0
        progress = (current_step - warmup_steps) / max(1, num_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_step)
