"""Vertical advection of FMS.

Translation of atmos_shared/vert_advection/vert_advection.F90 (siena_201207).

    rdt = vert_advection(dt, w, dz, r, scheme=..., form=..., flags=...)

w (ncol, nlev+1) is the velocity at half levels (positive downward, toward
increasing k, in the units of dz per second); dz (ncol, nlev) the layer thickness;
r (ncol, nlev) the advected field.  Index 0 is the top, as in the Fortran.

All branches are vectorised over columns and levels with torch.where, including the
multi-layer (CFL > 1) path of the parabolic scheme, so there are no host syncs.
The Fortran CFL statistics (cflmaxc, cflerr) are diagnostics only and are not kept.
"""
from __future__ import annotations

import torch

SECOND_CENTERED = 101
FOURTH_CENTERED = 102
FINITE_VOLUME_LINEAR = 103
FINITE_VOLUME_PARABOLIC = 104
FINITE_VOLUME_PARABOLIC2 = 105
SECOND_CENTERED_WTS = 106
FOURTH_CENTERED_WTS = 107
VAN_LEER_LINEAR = FINITE_VOLUME_LINEAR

FLUX_FORM = 201
ADVECTIVE_FORM = 202

WEIGHTED_TENDENCY = 1
OUTFLOW_BOUNDARY = 2

_SMALL = 1.0e-6


def _sign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fortran SIGN(a, b): |a| with the sign bit of b."""
    return torch.copysign(torch.abs(a), b)


def slope_z(r: torch.Tensor, dz: torch.Tensor, limit: bool = True, linear: bool = True) -> torch.Tensor:
    n = r.shape[1]
    grad = (r[:, 1:] - r[:, :-1]) / (dz[:, 1:] + dz[:, :-1])        # grad(k), k = 2..n  -> index k-2
    if linear:
        mid = (grad[:, 1:] + grad[:, :-1]) * dz[:, 1:-1]
    else:
        mid = ((grad[:, 1:] * (2.0 * dz[:, :-2] + dz[:, 1:-1])
                + grad[:, :-1] * (2.0 * dz[:, 2:] + dz[:, 1:-1]))
               * dz[:, 1:-1] / (dz[:, :-2] + dz[:, 1:-1] + dz[:, 2:]))
    first = 2.0 * grad[:, :1] * dz[:, :1]
    last = 2.0 * grad[:, -1:] * dz[:, -1:]
    slope = torch.cat([first, mid, last], dim=1)
    if limit:
        rmin = torch.minimum(torch.minimum(r[:, :-2], r[:, 1:-1]), r[:, 2:])
        rmax = torch.maximum(torch.maximum(r[:, :-2], r[:, 1:-1]), r[:, 2:])
        s = slope[:, 1:-1]
        lim = torch.minimum(torch.minimum(torch.abs(s), 2.0 * (r[:, 1:-1] - rmin)), 2.0 * (rmax - r[:, 1:-1]))
        s = _sign(torch.ones_like(s), s) * lim
        z = torch.zeros_like(slope[:, :1])
        slope = torch.cat([z, s, z], dim=1)
    assert slope.shape[1] == n
    return slope


def compute_weights(dz: torch.Tensor):
    """zwt(0:3, k) for k = 3..n-1 (1-based); returned as 4 tensors (ncol, n), zero elsewhere."""
    n = dz.shape[1]
    zw = [torch.zeros_like(dz) for _ in range(4)]
    if n < 4:
        return zw
    km2, km1, k0, kp1 = dz[:, 0:n - 3], dz[:, 1:n - 2], dz[:, 2:n - 1], dz[:, 3:n]
    denom1 = 1.0 / (km1 + k0)
    denom2 = 1.0 / (km2 + km1 + k0 + kp1)
    denom3 = 1.0 / (2 * km1 + k0)
    denom4 = 1.0 / (km1 + 2 * k0)
    num3 = km2 + km1
    num4 = k0 + kp1
    x = num3 * denom3 - num4 * denom4
    y = 2.0 * km1 * k0
    w0 = km1 * denom1
    w1 = w0 + x * y * denom1 * denom2
    w2 = km1 * num3 * denom3 * denom2
    w3 = k0 * num4 * denom4 * denom2
    for z, w in zip(zw, (w0, w1, w2, w3)):
        z[:, 2:n - 1] = w
    return zw


def vert_advection(dt: float, w: torch.Tensor, dz: torch.Tensor, r: torch.Tensor, mask=None,
                   scheme: int = VAN_LEER_LINEAR, form: int = FLUX_FORM, flags: int = 0) -> torch.Tensor:
    ncol, n = r.shape
    if w.shape[1] != n + 1:
        raise ValueError("vertical dimension of input arrays inconsistent")
    do_weighted_dt = bool(flags & 1)
    do_outflow_bnd = bool(flags & 2)
    # interface indices (0-based): 0 .. n ; interior interfaces 1 .. n-1
    if do_outflow_bnd:
        f_top = torch.zeros_like(w[:, 0])
        f_bot = torch.zeros_like(w[:, n])
    else:
        f_top = w[:, 0] * r[:, 0]
        f_bot = w[:, n] * r[:, n - 1]
    flux = [None] * (n + 1)
    flux[0], flux[n] = f_top, f_bot

    if scheme == SECOND_CENTERED_WTS:
        wt = dz[:, :-1] / (dz[:, :-1] + dz[:, 1:])
        rst = r[:, :-1] + wt * (r[:, 1:] - r[:, :-1])
        interior = w[:, 1:n] * rst
    elif scheme == SECOND_CENTERED:
        interior = w[:, 1:n] * (0.5 * (r[:, 1:] + r[:, :-1]))
    elif scheme in (FOURTH_CENTERED_WTS, FOURTH_CENTERED):
        # interfaces k = ks+2 .. ke-1 (1-based)  ->  0-based interface i = 2 .. n-2
        if scheme == FOURTH_CENTERED_WTS:
            zwt = compute_weights(dz)
            slp = slope_z(r, dz, limit=False, linear=False)
            i = slice(2, n - 1)
            rst4 = (r[:, 1:n - 2] + zwt[1][:, i] * (r[:, i] - r[:, 1:n - 2])
                    - zwt[2][:, i] * slp[:, i] + zwt[3][:, i] * slp[:, 1:n - 2])
            rst2 = r[:, 1:n - 2] + zwt[0][:, i] * (r[:, i] - r[:, 1:n - 2])
            wt = dz[:, 0] / (dz[:, 0] + dz[:, 1])
            rs_top = r[:, 0] + wt * (r[:, 1] - r[:, 0])
            wt = dz[:, n - 2] / (dz[:, n - 2] + dz[:, n - 1])
            rs_bot = r[:, n - 2] + wt * (r[:, n - 1] - r[:, n - 2])
        else:
            c1, c2 = 7.0 / 12.0, 1.0 / 12.0
            rst4 = c1 * (r[:, 2:n - 1] + r[:, 1:n - 2]) - c2 * (r[:, 3:n] + r[:, 0:n - 3])
            rst2 = 0.5 * (r[:, 2:n - 1] + r[:, 1:n - 2])
            rs_top = 0.5 * (r[:, 1] + r[:, 0])
            rs_bot = 0.5 * (r[:, n - 1] + r[:, n - 2])
        rst = rst4 if mask is None else torch.where(mask[:, 3:n] > _SMALL, rst4, rst2)
        interior = torch.cat([(w[:, 1] * rs_top)[:, None], w[:, 2:n - 1] * rst,
                              (w[:, n - 1] * rs_bot)[:, None]], dim=1)
    elif scheme == FINITE_VOLUME_LINEAR:
        slp = slope_z(r, dz)
        interior = _fv_linear(dt, w, dz, r, slp, 1, n - 1)
        if do_outflow_bnd:
            full = _fv_linear(dt, w, dz, r, slp, 0, n, bounded=True)
            flux[0], flux[n] = full[:, 0], full[:, n]
    elif scheme in (FINITE_VOLUME_PARABOLIC, FINITE_VOLUME_PARABOLIC2):
        interior, fb = _fv_parabolic(dt, w, dz, r, scheme, do_outflow_bnd)
        if do_outflow_bnd:
            flux[0], flux[n] = fb
    else:
        raise ValueError("invalid value for optional argument scheme")

    fl = torch.cat([flux[0][:, None], interior, flux[n][:, None]], dim=1)
    if form == FLUX_FORM:
        rdt = -(fl[:, 1:] - fl[:, :-1])
    elif form == ADVECTIVE_FORM:
        rdt = -(fl[:, 1:] - fl[:, :-1] - r * (w[:, 1:] - w[:, :-1]))
    else:
        raise ValueError("invalid value for optional argument form")
    if not do_weighted_dt:
        rdt = rdt / dz
    return rdt


def _fv_linear(dt, w, dz, r, slp, k0, k1, bounded=False):
    """Linear finite-volume fluxes at interfaces k0..k1 (0-based, inclusive)."""
    n = r.shape[1]
    ks = torch.arange(k0, k1 + 1, device=r.device)
    wk = w[:, k0:k1 + 1]
    up = (ks - 1).clamp(0, n - 1)            # cell above the interface
    dn = ks.clamp(0, n - 1)                  # cell below the interface
    cn_d = dt * wk / dz[:, up]
    rst_d = r[:, up] + 0.5 * slp[:, up] * (1.0 - cn_d)
    cn_u = -dt * wk / dz[:, dn]
    rst_u = r[:, dn] - 0.5 * slp[:, dn] * (1.0 - cn_u)
    down = wk >= 0.0
    f = wk * torch.where(down, rst_d, rst_u)
    if bounded:
        # inflow at the domain edges keeps the initialised (zero) flux
        f = torch.where(down & (ks == 0), torch.zeros_like(f), f)
        f = torch.where(~down & (ks == n), torch.zeros_like(f), f)
    return f


def _fv_parabolic(dt, w, dz, r, scheme, do_outflow_bnd):
    ncol, n = r.shape
    zwt = compute_weights(dz)
    slp = slope_z(r, dz, linear=False)
    # r_left(k) for k = ks+2..ke-1 (1-based) -> 0-based 2..n-2 ; r_right(k-1) = r_left(k)
    i = slice(2, n - 1)
    rl_mid = (r[:, 1:n - 2] + zwt[1][:, i] * (r[:, i] - r[:, 1:n - 2])
              - zwt[2][:, i] * slp[:, i] + zwt[3][:, i] * slp[:, 1:n - 2])
    r_left = torch.zeros_like(r)
    r_right = torch.zeros_like(r)
    r_left = torch.cat([r_left[:, :2], rl_mid, r_left[:, n - 1:]], dim=1)
    r_right = torch.cat([r_right[:, :1], rl_mid, r_right[:, n - 2:]], dim=1)
    # boundary-adjacent and boundary values (order as in the Fortran)
    r_left[:, 1] = r[:, 1] - 0.5 * slp[:, 1]
    r_right[:, n - 2] = r[:, n - 2] + 0.5 * slp[:, n - 2]
    r_left[:, 0] = r[:, 0] - 0.5 * slp[:, 0]
    r_right[:, 0] = r[:, 0] + 0.5 * slp[:, 0]
    r_left[:, n - 1] = r[:, n - 1] - 0.5 * slp[:, n - 1]
    r_right[:, n - 1] = r[:, n - 1] + 0.5 * slp[:, n - 1]

    if scheme == FINITE_VOLUME_PARABOLIC2:
        r_left = r - _sign(torch.minimum(torch.abs(slp), torch.abs(r_left - r)), slp)
        r_right = r + _sign(torch.minimum(torch.abs(slp), torch.abs(r_right - r)), slp)
    else:
        test_1 = (r_right - r) * (r - r_left) <= 0.0
        r_left = torch.where(test_1, r, r_left)
        r_right = torch.where(test_1, r, r_right)
        rm = r_right - r_left
        a = rm * (r - 0.5 * (r_right + r_left))
        b = rm * rm / 6.0
        inner = torch.zeros_like(r, dtype=torch.bool)
        inner[:, 1:n - 1] = True
        new_left = torch.where(inner & (a > b), 3.0 * r - 2.0 * r_right, r_left)
        new_right = torch.where(inner & (a < -b), 3.0 * r - 2.0 * r_left, r_right)
        r_left, r_right = new_left, new_right

    tt = 2.0 / 3.0
    k0, k1 = (0, n) if do_outflow_bnd else (1, n - 1)
    ks = torch.arange(k0, k1 + 1, device=r.device)                   # interfaces (0-based)
    wk = w[:, k0:k1 + 1]
    down = wk >= 0.0
    idx_up0 = (ks - 1).clamp(0, n - 1).expand(ncol, -1)
    idx_dn0 = ks.clamp(0, n - 1).expand(ncol, -1)
    cn_d = dt * wk / torch.gather(dz, 1, idx_up0)
    cn_u = -dt * wk / torch.gather(dz, 1, idx_dn0)
    cn = torch.where(down, cn_d, cn_u)
    dtw = torch.where(down, dt * wk, -dt * wk)
    # multi-layer walk (cn > 1): move kk away from the interface while the swept
    # depth exceeds whole layers (kk stops at the first/last layer)
    kk = torch.where(down, idx_up0, idx_dn0)
    step = torch.where(down, -1, 1)
    # The Fortran stops the downward-flow walk at kk == 1 and the upward-flow walk at
    # kk == ks (also the top layer, so it never triggers there and the walk could run
    # past the bottom). Here the upward walk is also stopped at the bottom layer.
    stop = torch.where(down, torch.zeros_like(kk), torch.full_like(kk, n - 1))
    dzsum = torch.zeros_like(wk)
    rsum = torch.zeros_like(wk)
    active = cn > 1.0
    for _ in range(n):
        dzk = torch.gather(dz, 1, kk)
        go = active & (dzsum + dzk < dtw) & (kk != stop)
        dzsum = torch.where(go, dzsum + dzk, dzsum)
        rsum = torch.where(go, rsum + torch.gather(r, 1, kk), rsum)
        kk = torch.where(go, kk + step, kk)
        active = go
    xx_multi = (dtw - dzsum) / torch.gather(dz, 1, kk)
    xx = torch.where(cn > 1.0, xx_multi, cn)
    rl = torch.gather(r_left, 1, kk)
    rr = torch.gather(r_right, 1, kk)
    rk = torch.gather(r, 1, kk)
    rm = rr - rl
    r6 = 6.0 * (rk - 0.5 * (rr + rl))
    r6 = torch.where(down & (kk == 0), torch.zeros_like(r6), r6)
    r6 = torch.where(~down & (kk == n - 1), torch.zeros_like(r6), r6)
    rst_d = rr - 0.5 * xx * (rm - (1.0 - tt * xx) * r6)
    rst_u = rl + 0.5 * xx * (rm + (1.0 - tt * xx) * r6)
    rst = torch.where(down, rst_d, rst_u)
    multi = cn > 1.0
    rst = torch.where(multi, (xx * rst + rsum) / torch.where(multi, cn, torch.ones_like(cn)), rst)
    f = wk * rst
    if do_outflow_bnd:
        zero = torch.zeros_like(f)
        f = torch.where(down & (ks == 0), zero, f)
        f = torch.where(~down & (ks == n), zero, f)
        return f[:, 1:n], (f[:, 0], f[:, n])
    return f, None
