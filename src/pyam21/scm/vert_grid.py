"""Hybrid sigma-pressure vertical grid of the SCM.

Translation of atmos_scm/tools/scm_vert_grid.F90 (siena_201207). Pressures on
full levels are the Simmons-Burridge log averages of the half levels; levels whose
eta increment is zero are pure pressure levels with a fixed full-level pressure.

Tensors: ps (ncol,), phalf (ncol, nlev+1), pfull (ncol, nlev); index 0 = top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..constants import GRAV, RDGAS, RVGAS

D608 = (RVGAS - RDGAS) / RDGAS
GINV = 1.0 / GRAV
PREF, TREF, GAMMA = 101325.0, 288.0, 0.0065
_EPS = np.finfo(np.float64).eps


@dataclass
class VertGrid:
    """vert_grid_type. Arrays are float64 numpy (they are constants of the grid)."""
    nlev: int
    nplev: int
    eta: np.ndarray      # (nlev+1)  B at half levels
    peta: np.ndarray     # (nlev+1)  A at half levels [Pa]
    deta: np.ndarray
    dpeta: np.ndarray
    apeta: np.ndarray    # full-level pressure of the pure-pressure levels
    wta: np.ndarray
    wtb: np.ndarray
    dfl: np.ndarray
    hybrid: bool
    pzero: bool
    psmin: float


def vert_grid_init(eta_in, peta_in=None) -> VertGrid:
    eta_in = np.asarray(eta_in, dtype=np.float64)
    kx = eta_in.size - 1
    eta = np.empty(kx + 1)
    eta[0] = 0.0
    eta[kx] = 1.0
    eta[1:kx] = eta_in[1:kx]
    peta = np.zeros(kx + 1)
    if peta_in is not None:
        peta[:kx] = np.asarray(peta_in, dtype=np.float64)[:kx]
    deta = eta[1:] - eta[:-1]
    dpeta = peta[1:] - peta[:-1]
    apeta = np.zeros(kx)
    wta = np.zeros(kx)
    wtb = np.zeros(kx)
    hybrid = bool(np.any(peta > 0.0))
    nplev = 0
    for k in range(kx):
        if deta[k] > 0.0:
            break
        nplev = k + 1
    pzero = True
    if nplev >= 1:
        phalf = peta + eta * PREF
        lphalf = np.empty(kx + 1)
        if phalf[0] <= _EPS:
            lphalf[0] = 0.0
            lphalf[1:] = np.log(phalf[1:])
        else:
            lphalf[:] = np.log(phalf)
            pzero = False
        for k in range(kx):
            lpfull = (phalf[k + 1] * lphalf[k + 1] - phalf[k] * lphalf[k]) / (phalf[k + 1] - phalf[k]) - 1.0
            apeta[k] = math.exp(lpfull)
            wtb[k] = lphalf[k + 1] - lpfull
            wta[k] = lpfull - lphalf[k]
        if pzero:
            wta[0] = wtb[0]
    psmin = 0.0
    for k in range(kx):
        if deta[k] > 0.0:
            psmin = max(psmin, -dpeta[k] / deta[k])
    rgog = RDGAS * GAMMA / GRAV
    dfl = np.zeros(kx + 1)
    for k in range(kx):
        pres = peta[k] + eta[k] * PREF
        dfl[k] = GRAV * TREF * (1.0 - (pres / PREF) ** rgog) / GAMMA
    return VertGrid(kx, nplev, eta, peta, deta, dpeta, apeta, wta, wtb, dfl, hybrid, pzero, psmin)


def _col(a: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(a, dtype=ref.dtype, device=ref.device)


def compute_pres_half(vg: VertGrid, ps: torch.Tensor) -> torch.Tensor:
    peta, eta = _col(vg.peta, ps), _col(vg.eta, ps)
    kp = vg.nplev + 1
    ph = peta[None, :] + eta[None, :] * ps[:, None]
    # the first nplev+1 half levels are exactly peta (eta = 0 there)
    ph = torch.cat([peta[None, :kp].expand(ps.shape[0], kp), ph[:, kp:]], dim=1)
    return ph


def compute_pres_depth(vg: VertGrid, ps: torch.Tensor) -> torch.Tensor:
    dpeta, deta = _col(vg.dpeta, ps), _col(vg.deta, ps)
    kp = vg.nplev
    dp = dpeta[None, :] + deta[None, :] * ps[:, None]
    return torch.cat([dpeta[None, :kp].expand(ps.shape[0], kp), dp[:, kp:]], dim=1)


def _log_pressures(vg: VertGrid, ph: torch.Tensor):
    """lph and ph*lph with the model-top convention (0 at a zero-pressure top)."""
    if vg.pzero:
        lph = torch.cat([torch.zeros_like(ph[:, :1]), torch.log(ph[:, 1:])], dim=1)
        phl = torch.cat([torch.zeros_like(ph[:, :1]), ph[:, 1:] * lph[:, 1:]], dim=1)
    else:
        lph = torch.log(ph)
        phl = ph * lph
    return lph, phl


def _sb_stable(a, d):
    """Simmons-Burridge weights of a layer from its top pressure a and depth d, without
    the cancellation of (b ln b - a ln a) / d in low precision:
    wtb = ln b - ln pf = 1 - (a/d) ln(1 + d/a), wta = ln pf - ln a = ln(1 + d/a) - wtb."""
    pos = a > 0.0
    r = d / torch.where(pos, a, torch.ones_like(a))
    l1 = torch.log1p(r)
    wtb = torch.where(pos, 1.0 - l1 / r, torch.ones_like(a))
    wta = torch.where(pos, l1 - wtb, wtb)
    return wta, wtb


def compute_pres_full(vg: VertGrid, ps: torch.Tensor, phalf=None, dpde=None) -> torch.Tensor:
    ph = compute_pres_half(vg, ps) if phalf is None else phalf
    dp = compute_pres_depth(vg, ps) if dpde is None else dpde
    kp = vg.nplev
    if ps.dtype != torch.float64:
        _, wtb = _sb_stable(ph[:, kp:-1], dp[:, kp:])
        pf = ph[:, kp + 1:] * torch.exp(-wtb)
        top = _col(vg.apeta[:kp], ps)[None, :].expand(ps.shape[0], kp)
        return torch.cat([top, pf], dim=1)
    _, phl = _log_pressures(vg, ph)
    pf = torch.exp((phl[:, kp + 1:] - phl[:, kp:-1]) / dp[:, kp:] - 1.0)
    top = _col(vg.apeta[:kp], ps)[None, :].expand(ps.shape[0], kp)
    return torch.cat([top, pf], dim=1)


def compute_pres_weights(vg: VertGrid, phalf: torch.Tensor, pfull: torch.Tensor):
    kp = vg.nplev
    kx = pfull.shape[1]
    if pfull.dtype != torch.float64:
        # logs of ratios near 1 instead of differences of logs near 11.5
        ncol = pfull.shape[0]
        pf = pfull[:, kp:]
        wtb = torch.log1p((phalf[:, kp + 1:] - pf) / pf)
        a = phalf[:, kp:kx]
        pos = a > 0.0
        wta = torch.where(pos, torch.log1p((pf - a) / torch.where(pos, a, torch.ones_like(a))), wtb)
        wtb = torch.cat([_col(vg.wtb[:kp], pfull)[None, :].expand(ncol, kp), wtb], dim=1)
        wta = torch.cat([_col(vg.wta[:kp], pfull)[None, :].expand(ncol, kp), wta], dim=1)
        return wta, wtb
    ks = max(2, kp + 1) if vg.pzero else kp + 1          # 1-based, as in the Fortran
    logph = torch.zeros_like(phalf)
    logph = torch.cat([logph[:, :ks - 1], torch.log(phalf[:, ks - 1:])], dim=1)
    logpf = torch.cat([torch.zeros_like(pfull[:, :kp]), torch.log(pfull[:, kp:])], dim=1)
    ncol = pfull.shape[0]
    wtb_top = _col(vg.wtb[:kp], pfull)[None, :].expand(ncol, kp)
    wtb = torch.cat([wtb_top, logph[:, kp + 1:] - logpf[:, kp:]], dim=1)
    parts = [_col(vg.wta[:kp], pfull)[None, :].expand(ncol, kp)]
    if vg.pzero and kp == 0:                              # wta(1) = wtb(1)
        parts.append(wtb[:, :1])
    parts.append(logpf[:, ks - 1:] - logph[:, ks - 1:kx])
    wta = torch.cat(parts, dim=1)
    return wta, wtb


def compute_geop_height(vg: VertGrid, fssl: torch.Tensor, vtemp, wta, wtb):
    """Returns zfull, zhalf in geopotential (m2 s-2), integrating upward from fssl."""
    klev = vg.nlev
    zb = fssl
    zfull = [None] * klev
    zhalf = [None] * (klev + 1)
    zhalf[klev] = zb
    for k in range(klev - 1, -1, -1):
        rt = RDGAS * vtemp[:, k]
        zt = zb + rt * (wta[:, k] + wtb[:, k])
        zfull[k] = zb + rt * wtb[:, k]
        zb = zt
        zhalf[k] = zb
    return torch.stack(zfull, dim=1), torch.stack(zhalf, dim=1)


def compute_height(vg: VertGrid, fssl, temp, sphum, pfull, phalf):
    """Heights (m) of full and half levels above the surface geopotential fssl/g."""
    wta, wtb = compute_pres_weights(vg, phalf, pfull)
    vtemp = temp * (1.0 + D608 * sphum)
    zf, zh = compute_geop_height(vg, fssl, vtemp, wta, wtb)
    return zf * GINV, zh * GINV


def compute_height_bottom(vg: VertGrid, ps, tbot, qbot, kbot: int | None = None):
    """Height and pressure of the lowest full level (the kbot branch used by the SCM)."""
    kb = vg.nlev if kbot is None else kbot            # 1-based
    pht = vg.peta[kb - 1] + vg.eta[kb - 1] * ps
    phb = vg.peta[kb] + vg.eta[kb] * ps
    rt = GINV * RDGAS * (tbot * (1.0 + D608 * qbot))
    dp = phb - pht
    if torch.is_tensor(tbot) and tbot.dtype != torch.float64:
        _, wtb = _sb_stable(pht if torch.is_tensor(pht) else torch.full_like(tbot, pht), dp)
        return rt * wtb, phb * torch.exp(-wtb)
    lphb = torch.log(phb)
    lpht = torch.log(pht)
    lpf = (phb * lphb - pht * lpht) / dp - 1
    return rt * (lphb - lpf), torch.exp(lpf)
