"""Per-column index helpers used to vectorise level searches of the Fortran schemes.

Levels are addressed with Fortran (1-based) level numbers held in int64 tensors of
shape (ncol,). ``at(a, k)`` reads a(k) per column; ``put``/``add`` write one level per
column where a condition holds. All are differentiable in the values (not indices).
"""
from __future__ import annotations

import torch


def at(a: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """a(k) for each column; a is (ncol, n), k1 is 1-based (ncol,) and must be in 1..n."""
    return a.gather(1, (k1 - 1).clamp(0, a.shape[1] - 1)[:, None]).squeeze(1)


def levels(a: torch.Tensor) -> torch.Tensor:
    """1-based level numbers (1, n) for a (ncol, n) tensor."""
    return torch.arange(1, a.shape[1] + 1, device=a.device)[None, :]


def onehot(a: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    return levels(a) == k1[:, None]


def put(a: torch.Tensor, k1: torch.Tensor, val, cond: torch.Tensor | None = None) -> torch.Tensor:
    m = onehot(a, k1)
    if cond is not None:
        m = m & cond[:, None]
    v = val[:, None] if isinstance(val, torch.Tensor) and val.dim() == 1 else val
    return torch.where(m, v, a)


def add(a: torch.Tensor, k1: torch.Tensor, val: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    m = onehot(a, k1)
    if cond is not None:
        m = m & cond[:, None]
    return torch.where(m, a + val[:, None], a)


def first_true(mask: torch.Tensor, reverse: bool = False):
    """First level (1-based) where mask is true, scanning down (or up if reverse).

    Returns (k1, found); k1 is 0 where nothing was found.
    """
    n = mask.shape[1]
    lev = torch.arange(1, n + 1, device=mask.device).expand_as(mask)
    if reverse:
        cand = torch.where(mask, lev, torch.zeros_like(lev))
        k1 = cand.max(dim=1).values
    else:
        cand = torch.where(mask, lev, torch.full_like(lev, n + 1))
        k1 = cand.min(dim=1).values
        k1 = torch.where(k1 == n + 1, torch.zeros_like(k1), k1)
    return k1, mask.any(dim=1)
