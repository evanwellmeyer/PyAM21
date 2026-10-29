"""Time and pressure interpolation of the observational SCM forcing datasets
(interp_time / interp_2d_field of scm_astex, scm_arm, scm_mpacea, scm_mpaceb)."""
from __future__ import annotations

import numpy as np
import torch


def tsec(t) -> int:
    return t.days * 86400 + t.seconds


def interp_time(time, times):
    """-> (itime_less, itime_more, weight_less, weight_more), 0-based indices.

    Outside the record: both indices at the end, weight_more = 1 (as in the Fortran).
    Inside: the first later-or-equal layer j >= 1 and j - 1."""
    t = tsec(time)
    ts = [tsec(x) for x in times]
    n = len(ts)
    if t <= ts[0] or t >= ts[-1]:
        i = 0 if t <= ts[0] else n - 1
        return i, i, 0.0, 1.0
    for j in range(1, n):
        if ts[j] >= t:
            wm = float(t - ts[j - 1]) / float(ts[j] - ts[j - 1])
            return j - 1, j, 1.0 - wm, wm
    raise AssertionError


def _prep(plev, less, more, wl, wm):
    plev = np.array(plev, dtype=np.float64)
    less = np.array(less, dtype=np.float64)
    more = np.array(more, dtype=np.float64)
    if plev[0] < plev[1]:
        plev[0], plev[1] = plev[1], plev[0]
        more[1] = more[0]
        less[1] = less[0]
    return plev, wm * more + wl * less


def interp_2d_field(p_interp, plev, less, more=None, wl=1.0, wm=0.0):
    """interp_2d_field: time-weighted profile interpolated linearly in pressure on a
    surface-first profile (plev decreasing), constant beyond the ends. numpy."""
    plev, f = _prep(plev, less, less if more is None else more, wl, wm)
    n = len(plev)
    out = np.empty(len(p_interp))
    for i, p in enumerate(np.asarray(p_interp, dtype=np.float64)):
        if p <= plev[n - 1]:
            out[i] = f[n - 1]
        elif p >= plev[0]:
            out[i] = f[0]
        else:
            for j in range(1, n):
                if plev[j] <= p:
                    pw = (p - plev[j - 1]) / (plev[j] - plev[j - 1])
                    out[i] = pw * f[j] + (1.0 - pw) * f[j - 1]
                    break
    return out


def interp_2d_field_t(p_interp: torch.Tensor, plev, less, more=None, wl=1.0, wm=0.0):
    """Vectorised interp_2d_field for a (ncol, k) pressure tensor."""
    plev, f = _prep(plev, less, less if more is None else more, wl, wm)
    n = len(plev)
    pl = torch.as_tensor(plev, dtype=p_interp.dtype, device=p_interp.device)
    fl = torch.as_tensor(f, dtype=p_interp.dtype, device=p_interp.device)
    j = torch.searchsorted(-pl, -p_interp.contiguous(), right=False).clamp(1, n - 1)
    pw = (p_interp - pl[j - 1]) / (pl[j] - pl[j - 1])
    out = pw * fl[j] + (1.0 - pw) * fl[j - 1]
    out = torch.where(p_interp >= pl[0], fl[0].expand_as(out), out)
    out = torch.where(p_interp <= pl[n - 1], fl[n - 1].expand_as(out), out)
    return out


def scalar(less, more, wl, wm) -> float:
    """interp_2d_field of a constant 'profile' at 500 hPa: the time interpolation
    (the Fortran's wm * more + wl * less, with its operation order)."""
    return wm * more + wl * less
