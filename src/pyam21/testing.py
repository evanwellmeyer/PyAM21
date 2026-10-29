"""Reading reference dumps (reference/extra/ref_dump.F90) and comparing results.

A dump file holds named arrays in Fortran order. ``read_dump`` returns numpy arrays
with the Fortran shape. FMS physics arrays are (i, j, k[, n]) with i = j = 1 in the
SCM; ``col`` turns them into the port's (ncol, nlev[, n]) layout.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUMPS = os.path.join(REPO, "reference", "dumps")


def read_dump(path: str) -> dict:
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        name = data[pos:pos + 32].decode().strip()
        pos += 32
        typ, ndim = np.frombuffer(data, "<i4", 2, pos)
        pos += 8
        shape = tuple(int(s) for s in np.frombuffer(data, "<i4", ndim, pos))
        pos += 4 * ndim
        n = int(np.prod(shape)) if ndim else 1
        dt = "<f8" if typ == 1 else "<i4"
        size = 8 if typ == 1 else 4
        a = np.frombuffer(data, dt, n, pos).copy()
        pos += n * size
        out[name] = a.reshape(shape, order="F") if ndim else a[0]
    return out


def dump_files(case: str, tag: str) -> list[tuple[int, int, str]]:
    """Sorted (step, count, path) of every dump with this tag."""
    res = []
    for p in glob.glob(os.path.join(DUMPS, case, f"{tag}_s*_c*.bin")):
        m = re.search(rf"{re.escape(tag)}_s(\d+)_c(\d+)\.bin$", p)
        if m:
            res.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(res)


def dump_pairs(case: str, tag: str):
    """Yield (step, count, inputs, outputs) for tag_in / tag_out dump pairs."""
    outs = {(s, c): p for s, c, p in dump_files(case, tag + "_out")}
    for s, c, p in dump_files(case, tag + "_in"):
        if (s, c) in outs:
            yield s, c, read_dump(p), read_dump(outs[(s, c)])


def col(a, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """(i, j, k, ...) Fortran array -> (i*j, k, ...) tensor; (i, j) -> (i*j,)."""
    a = np.asarray(a)
    if a.ndim >= 2:
        a = a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])
    return torch.as_tensor(a, dtype=dtype, device=device)


def maxrel(x, y, floor: float = 0.0) -> float:
    """max |x - y| / max(max |y|, floor); 0 if both are all zero."""
    x = x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, float)
    y = y.detach().cpu().double().numpy() if isinstance(y, torch.Tensor) else np.asarray(y, float)
    scale = max(np.max(np.abs(y)) if y.size else 0.0, floor)
    d = np.max(np.abs(x - y)) if y.size else 0.0
    return 0.0 if d == 0 else (d / scale if scale > 0 else np.inf)


class DumpRadiation:
    """Radiation stand-in replaying the Fortran's per-step radiation_driver outputs
    (rad_out dumps), to test everything else in the radiation cases."""

    def __init__(self, tag: str):
        self.ref = {s: p for s, c, p in dump_files(tag, "rad_out")}

    def __call__(self, model, ctx):
        r = read_dump(self.ref[model.step_count + 1])
        t = lambda x: torch.as_tensor(col(x), dtype=model.dtype, device=model.device)
        c = lambda x: torch.as_tensor(np.asarray(x).reshape(-1)[:1], dtype=model.dtype,
                                      device=model.device).expand(model.ncol)
        return dict(tdt_rad=t(r["tdt_rad"]), tdtlw=t(r["tdtlw"]), coszen=c(r["coszen"]),
                    flux_sw=c(r["flux_sw_surf"]), flux_lw=c(r["flux_lw_surf"]))
