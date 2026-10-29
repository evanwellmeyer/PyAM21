"""Shared helpers: device/dtype handling and cached lookup tables.

Conventions of the whole package:
  - Column tensors have shape (ncol, nlev) (or (ncol, nlev+1) at interfaces),
    index 0 at the model top, as in FMS (k = 1 is the top level there too).
  - float64 on CPU reproduces the Fortran reference (default real = real(8)).
    MPS has no float64, so on MPS everything runs in float32.
  - Tables are built once in float64 numpy, exactly as the Fortran builds them,
    and moved to the working device/dtype on first use.
"""
from __future__ import annotations

import numpy as np
import torch


def default_dtype_for(device: torch.device | str) -> torch.dtype:
    """float64 where supported, float32 on MPS."""
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


class TableCache:
    """A float64 numpy table with per-(device, dtype) tensor copies."""

    def __init__(self, array: np.ndarray):
        self.array = np.ascontiguousarray(array, dtype=np.float64)
        self._cache: dict = {}

    def to(self, device, dtype) -> torch.Tensor:
        key = (torch.device(device), dtype)
        t = self._cache.get(key)
        if t is None:
            t = torch.as_tensor(self.array, dtype=dtype, device=device)
            self._cache[key] = t
        return t

    def like(self, ref: torch.Tensor) -> torch.Tensor:
        return self.to(ref.device, ref.dtype)


def fortran_int(x: torch.Tensor) -> torch.Tensor:
    """Fortran INT(): truncation toward zero, as an integer (int64) tensor."""
    return torch.trunc(x).to(torch.int64)


def full_like(ref: torch.Tensor, value: float, shape=None) -> torch.Tensor:
    return torch.full(ref.shape if shape is None else shape, value, dtype=ref.dtype, device=ref.device)


# ---- gradient-safe elementary functions ---------------------------------------------------------
# The physics evaluates many expressions everywhere and selects with torch.where. Where an
# unselected (or a legitimately zero) argument sits on a singularity of the derivative -
# x**p with p < 1 at x = 0, sqrt(0) - the forward value is fine but the backward pass
# forms inf * 0 = NaN, which torch.where does not stop. These helpers return the same
# forward values (bit for bit where the argument is positive) with a zero derivative at
# the singular point.

def spow(x: torch.Tensor, p: float) -> torch.Tensor:
    """x**p for x > 0, 0 for x <= 0; d/dx = 0 at x <= 0."""
    pos = x > 0.0
    return torch.where(pos, torch.where(pos, x, torch.ones_like(x)) ** p, torch.zeros_like(x))


def ssqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt(x) for x > 0, 0 for x <= 0; d/dx = 0 at x <= 0."""
    pos = x > 0.0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), torch.zeros_like(x))
