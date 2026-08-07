"""Helpers to build and load a VGGT-Omega model for matching / evaluation."""

from __future__ import annotations

import torch

from vggt_omega.models import VGGTOmega

__all__ = ["load_model"]


def load_model(
    checkpoint: str,
    enable_alignment: bool = False,
    device: str = "cuda",
) -> VGGTOmega:
    """Instantiate ``VGGTOmega`` (camera + depth heads) and load a checkpoint.

    Accepts a raw ``state_dict`` or a dict wrapping it under ``model`` / ``state_dict``.
    ``enable_alignment=True`` is only needed for the 256 text-aligned checkpoint.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment, but VGGT-Omega is CUDA-only. "
            "Use an environment whose torch build matches the driver "
            "(e.g. a cu124/cu128 build for driver 550), or pass --device cpu at your own risk."
        )

    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=enable_alignment)

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_model] {len(missing)} missing keys (first few): {missing[:5]}")
    if unexpected:
        print(f"[load_model] {len(unexpected)} unexpected keys (first few): {unexpected[:5]}")

    return model.to(device).eval()
