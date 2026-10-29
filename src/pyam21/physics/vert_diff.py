"""Implicit vertical diffusion (atmos_param/vert_diff/vert_diff.F90, vert_diff_driver.F90).

Backward-Euler tridiagonal diffusion of u, v, dry static energy (t + gz/cp), q and
the other prognostic tracers, split into a downward sweep (gcm_vert_diff_down) that
leaves the lowest level open for the surface coupling and an upward back-substitution
(gcm_vert_diff_up) once the surface has returned delta_t / delta_q.

The persistent arrays of the Fortran (e_global, f_t_global, f_q_global) are returned
by `down` and passed to `up` explicitly. diff_m/diff_t are defined at the interface
above each level (index 0 unused). kbot = nlev.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, GRAV, RDGAS, RVGAS

D608 = (RVGAS - RDGAS) / RDGAS


@dataclass(frozen=True)
class VertDiffConfig:
    """vert_diff_driver_nml."""
    do_conserve_energy: bool = False
    do_mcm_no_neg_q: bool = False
    use_virtual_temp_vert_diff: bool = True
    do_mcm_plev: bool = False
    do_mcm_vert_diff_tq: bool = False

    @classmethod
    def from_nml(cls, nml: dict) -> "VertDiffConfig":
        g = nml.get("vert_diff_driver_nml", {})
        kw = {f: bool(g[f]) for f in cls.__dataclass_fields__ if f in g}
        return cls(**kw)


def compute_mu(p_half):
    return GRAV / (p_half[:, 1:] - p_half[:, :-1])


def compute_nu(c: VertDiffConfig, diff, p_half, p_full, z_full, t, q):
    tt = t * (1.0 + D608 * q) if c.use_virtual_temp_vert_diff else t
    rho_half = 2.0 * p_half[:, 1:-1] / (RDGAS * (tt[:, 1:] + tt[:, :-1]))
    if c.do_mcm_plev:
        nu = GRAV * rho_half * rho_half * diff[:, 1:] / (p_full[:, 1:] - p_full[:, :-1])
    else:
        nu = rho_half * diff[:, 1:] / (z_full[:, :-1] - z_full[:, 1:])
    return torch.cat([torch.zeros_like(nu[:, :1]), nu], dim=1)      # nu(1) undefined in Fortran; 0


def compute_e(delt, mu, nu):
    """Returns e (ncol, nlev-1), b, c, g (ncol, nlev); g/e valid for k = 2..nlev-1."""
    n = mu.shape[1]
    a = torch.cat([-mu[:, :-1] * nu[:, 1:] * delt, torch.zeros_like(mu[:, :1])], dim=1)
    c = torch.cat([torch.zeros_like(mu[:, :1]), -mu[:, 1:] * nu[:, 1:] * delt], dim=1)
    b = 1.0 - a - c
    e = [-a[:, 0] / b[:, 0]]
    g = [torch.zeros_like(b[:, 0])]
    for k in range(1, n - 1):
        gk = 1.0 / (b[:, k] + c[:, k] * e[k - 1])
        g.append(gk)
        e.append(-a[:, k] * gk)
    g.append(torch.zeros_like(b[:, 0]))
    return torch.stack(e, dim=1), b, c, torch.stack(g, dim=1)


def compute_f(dt_xi, b, c, g):
    n = b.shape[1]
    f = [dt_xi[:, 0] / b[:, 0]]
    for k in range(1, n - 1):
        f.append((dt_xi[:, k] - c[:, k] * f[k - 1]) * g[:, k])
    return torch.stack(f, dim=1)


def explicit_tend(mu, nu, xi, dt_xi, dxi=None):
    """dxi: xi[k] - xi[k-1] if already known more accurately than by differencing xi."""
    if dxi is None:
        dxi = xi[:, 1:] - xi[:, :-1]
    fluxx = torch.cat([torch.zeros_like(xi[:, :1]), nu[:, 1:] * dxi], dim=1)
    upper = dt_xi[:, :-1] + mu[:, :-1] * (fluxx[:, 1:] - fluxx[:, :-1])
    last = dt_xi[:, -1:] - mu[:, -1:] * fluxx[:, -1:]
    return torch.cat([upper, last], dim=1)


def vert_diff_up(delt, e, f, delta_xi_n):
    n = e.shape[1] + 1
    out = [None] * n
    out[n - 1] = delta_xi_n / delt
    for k in range(n - 2, -1, -1):
        out[k] = e[:, k] * out[k + 1] + f[:, k]
    return torch.stack(out, dim=1)


def diff_surface(mu_delt, nu, e_n1, f_delt_n1, dflux_datmos, flux, factor, delta_xi):
    fff = 1.0 / factor
    dflux = -nu * (1.0 - e_n1)
    delta_xi = delta_xi + mu_delt * nu * f_delt_n1
    delta_xi = (delta_xi + mu_delt * flux * fff) / (1.0 - mu_delt * (dflux + dflux_datmos * fff))
    flux = flux + dflux_datmos * delta_xi
    return flux, delta_xi


class VertDiff:
    def __init__(self, config: VertDiffConfig, do_vert_diff_tracer=None):
        """do_vert_diff_tracer: bool per tracer (sphum is handled with t, never here)."""
        if config.do_mcm_vert_diff_tq or config.do_mcm_no_neg_q:
            raise NotImplementedError("MCM vertical diffusion options")
        self.c = config
        self.do_vert_diff_tracer = do_vert_diff_tracer

    def _down_2(self, delt, mu, nu, xi_1, xi_2, dt_xi_1, dt_xi_2, dxi_1=None):
        dt11 = explicit_tend(mu, nu, xi_1, dt_xi_1, dxi_1)
        dt22 = explicit_tend(mu, nu, xi_2, dt_xi_2)
        e, b, c, g = compute_e(delt, mu, nu)
        f_1 = compute_f(dt11, b, c, g)
        f_2 = compute_f(dt22, b, c, g)
        return dict(e=e, f_1=f_1, f_2=f_2, mu_delt_n=mu[:, -1] * delt, nu_n=nu[:, -1], e_n1=e[:, -1],
                    f_1_delt_n1=f_1[:, -1] * delt, f_2_delt_n1=f_2[:, -1] * delt,
                    delta_1_n=dt11[:, -1] * delt, delta_2_n=dt22[:, -1] * delt)

    def down(self, delt, u, v, t, q, tr, diff_m, diff_t, p_half, p_full, z_full,
             tau_u, tau_v, dtau_du, dtau_dv, dt_u, dt_v, dt_t, dt_q, dt_tr, sphum: int = 0):
        """gcm_vert_diff_down. tr/dt_tr are (ncol, nlev, ntr) with sphum at index `sphum`.

        Returns updated tendencies, surface stresses, the Surf_diff fields and the
        saved state for `up`.
        """
        c = self.c
        gcp = GRAV / CP_AIR
        tt = t + z_full * gcp
        mu = compute_mu(p_half)
        # momentum
        nu = compute_nu(c, diff_m, p_half, p_full, z_full, t, q)
        dt_u_old, dt_v_old = dt_u, dt_v
        d = self._down_2(delt, mu, nu, u, v, dt_u, dt_v)
        tau_u, delta_u_n = diff_surface(d["mu_delt_n"], d["nu_n"], d["e_n1"], d["f_1_delt_n1"],
                                        dtau_du, tau_u, 1.0, d["delta_1_n"])
        tau_v, delta_v_n = diff_surface(d["mu_delt_n"], d["nu_n"], d["e_n1"], d["f_2_delt_n1"],
                                        dtau_dv, tau_v, 1.0, d["delta_2_n"])
        dt_u = vert_diff_up(delt, d["e"], d["f_1"], delta_u_n)
        dt_v = vert_diff_up(delt, d["e"], d["f_2"], delta_v_n)
        if c.do_conserve_energy:
            du = dt_u - dt_u_old
            dv = dt_v - dt_v_old
            diss = -(1.0 / CP_AIR) * ((u + 0.5 * delt * du) * du + (v + 0.5 * delt * dv) * dv)
            dt_t = dt_t + diss
        else:
            diss = torch.zeros_like(u)
        # heat, moisture, tracers
        nu = compute_nu(c, diff_t, p_half, p_full, z_full, t, q)
        e, b, cc, g = compute_e(delt, mu, nu)
        mu_delt_n = mu[:, -1] * delt
        nu_n = nu[:, -1]
        e_n1 = e[:, -1]
        ntr = tr.shape[2]
        dts = []
        delta_tr = []
        dflux_tr = []
        for n in range(ntr):
            dtn = dt_tr[..., n]
            if n == sphum or (self.do_vert_diff_tracer is not None and not self.do_vert_diff_tracer[n]):
                dts.append(dtn)
                delta_tr.append(torch.zeros_like(mu_delt_n))
                dflux_tr.append(torch.zeros_like(mu_delt_n))
                continue
            dtn = explicit_tend(mu, nu, tr[..., n], dtn)
            f_tr = compute_f(dtn, b, cc, g)
            f_tr_delt_n1 = f_tr[:, -1] * delt
            delta_tr_n = dtn[:, -1] * delt
            delta_tr.append(delta_tr_n + mu_delt_n * nu_n * f_tr_delt_n1)
            dflux_tr.append(-nu_n * (1.0 - e_n1))
            # no surface exchange for these tracers: zero surface flux, complete now
            _, delta_n = diff_surface(mu_delt_n, nu_n, e_n1, f_tr_delt_n1,
                                      torch.zeros_like(mu_delt_n), torch.zeros_like(mu_delt_n), 1.0, delta_tr_n)
            dts.append(vert_diff_up(delt, e, f_tr, delta_n))
        # below float64, difference t and z separately: t + gz/cp (~330 K) differenced across
        # a well-mixed layer loses most of its digits (float64 keeps the Fortran's order)
        dtt = None if t.dtype == torch.float64 else \
            (t[:, 1:] - t[:, :-1]) + (z_full[:, 1:] - z_full[:, :-1]) * gcp
        d2 = self._down_2(delt, mu, nu, tt, q, dt_t, dt_q, dtt)
        delta_t = d2["delta_1_n"] + mu_delt_n * nu_n * d2["f_1_delt_n1"]
        dflux_t = -nu_n * (1.0 - e_n1)
        delta_tr[sphum] = d2["delta_2_n"] + mu_delt_n * nu_n * d2["f_2_delt_n1"]
        dflux_tr[sphum] = -nu_n * (1.0 - e_n1)
        return dict(dt_u=dt_u, dt_v=dt_v, dt_t=dt_t, dt_q=dt_q, dt_tr=torch.stack(dts, dim=-1),
                    tau_u=tau_u, tau_v=tau_v, dissipative_heat=diss,
                    dtmass=mu_delt_n, delta_t=delta_t, dflux_t=dflux_t,
                    delta_tr=torch.stack(delta_tr, dim=-1), dflux_tr=torch.stack(dflux_tr, dim=-1),
                    delta_u=delta_u_n, delta_v=delta_v_n,
                    e_global=d2["e"], f_t_global=d2["f_1"], f_q_global=d2["f_2"])

    def up(self, delt, e_global, f_t_global, f_q_global, delta_t, delta_q):
        """gcm_vert_diff_up: dt_t and dt_q from the surface-updated lowest-level deltas.

        (The Fortran overwrites dt_t/dt_q; they already contain every tendency accumulated
        before the down sweep, through f.)
        """
        dt_t = vert_diff_up(delt, e_global, f_t_global, delta_t)
        dt_q = vert_diff_up(delt, e_global, f_q_global, delta_q)
        return dt_t, dt_q
