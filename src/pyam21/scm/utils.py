"""SCM utilities (atmos_scm/tools/scm_utils.F90, siena_201207).

These run once, at initialisation, on scalars, so they are plain float64 Python
(the saturation lookup goes through the same table as the physics).
"""
from __future__ import annotations

import math

import torch

from ..constants import CP_AIR, GRAV, HLV, KAPPA, RDGAS, RVGAS
from ..sat_vapor_pres import get_sat_vapor_pres

P00 = 100000.0

_ZS = (0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852)
_TS = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95)
_LRS = (-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0)
_PS = (1013.25e2, 226.3206e2, 54.7489e2, 8.6802e2, 1.1091e2, 0.6694e2, 0.0396e2, 0.0037e2)


def us_std_atm(p: float) -> tuple[float, float]:
    """US standard atmosphere temperature at pressure p (Pa); qv = 0."""
    nmax = 8
    i = nmax
    while p >= _PS[i - 1] and i >= 2:
        i -= 1
    if i < nmax:
        if _LRS[i - 1] == 0.0:
            dz = -RDGAS * _TS[i - 1] / (1000.0 * GRAV) * math.log(p / _PS[i - 1])
        else:
            dz = -(_TS[i - 1] / _LRS[i - 1]) * (1 - (p / _PS[i - 1]) ** (-RDGAS * _LRS[i - 1] / (1000.0 * GRAV)))
        t = _TS[i - 1] + _LRS[i - 1] * dz
    else:
        t = _TS[i - 1]
    return t, 0.0


def lookup_es_scalar(t: float) -> float:
    return float(get_sat_vapor_pres().lookup_es(torch.tensor([t], dtype=torch.float64))[0])


def _saturated(temp, thetal, rt, pres, ex):
    esat = lookup_es_scalar(temp)
    rsat = RDGAS / RVGAS * esat / (pres - esat)
    return thetal * ex - temp + HLV / CP_AIR * (rt - rsat)


def _rtsec(x1, x2, ya, yb, yc, yd, xacc):
    maxit = 30
    fl = _saturated(x1, ya, yb, yc, yd)
    f = _saturated(x2, ya, yb, yc, yd)
    if abs(fl) < abs(f):
        rts, xl = x1, x2
        fl, f = f, fl
    else:
        xl, rts = x1, x2
    for _ in range(maxit):
        dx = (xl - rts) * f / (f - fl)
        xl = rts
        fl = f
        rts = rts + dx
        f = _saturated(rts, ya, yb, yc, yd)
        if abs(dx) < xacc or f == 0.0:
            return rts
    raise RuntimeError("rtsec: exceed maximum iterations")


def thetal_to_temp(thetal: float, rt: float, pres: float) -> tuple[float, float, float]:
    """Temperature, vapour and liquid specific humidity from (theta_l, total mixing ratio)."""
    ex = (pres / P00) ** KAPPA
    temp_unsat = thetal * ex
    if rt <= 1.0e-10:
        return temp_unsat, rt, 0.0
    temp_sat = _rtsec(temp_unsat - 30.0, temp_unsat + 30.0, thetal, rt, pres, ex, 1.0e-3)
    if temp_sat > temp_unsat:
        esat = lookup_es_scalar(temp_sat)
        rsat = RDGAS / RVGAS * esat / (pres - esat)
        return temp_sat, rsat / (1.0 + rt), (rt - rsat) / (1.0 + rt)
    return temp_unsat, rt / (1.0 + rt), 0.0


def locate(xx, x) -> int:
    """Numerical Recipes bisection; returns j (0..n) as in the Fortran (1-based bracket)."""
    n = len(xx)
    jl, ju = 0, n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if (xx[n - 1] > xx[0]) == (x > xx[jm - 1]):
            jl = jm
        else:
            ju = jm
    return jl
