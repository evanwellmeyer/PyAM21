"""GCSS ATEX trade-wind cumulus under stratocumulus (atmos_scm/forc/scm_atex.F90, siena_201207).

Stevens et al. (2001): piecewise-linear sounding, constant f = 0.375e-4 geostrophic
forcing, a subsidence / large-scale cooling and drying profile tied to the inversion
height (switched on after 1.5 h), a cloud-top longwave flux 74 exp(-130 LWP), and bulk
surface fluxes from the lowest level (atex_sfclyr).
"""
from __future__ import annotations

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, KAPPA, RDGAS, RVGAS
from ...time_manager import set_date
from ..utils import us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, vadv

P00 = 100000.0
D608 = (RVGAS - RDGAS) / RDGAS


def _pw(z, knots, vals, top):
    """Piecewise-linear profile with the Fortran's knot-by-knot interpolation formulae."""
    zlo = 0.0
    for zk, (a, b) in zip(knots, vals):
        if z < zk:
            return a + (b - a) / (zk - zlo) * (z - zlo) if zlo > 0.0 else a + (b - a) / zk * z
        zlo = zk
    return top


def atex_snd(z: float):
    kn = (150.0, 700.0, 750.0, 1400.0, 1650.0, 4000.0)
    u = _pw(z, kn, ((-11.0, -10.55), (-10.55, -8.9), (-8.9, -8.75), (-8.75, -6.8), (-6.8, -5.75),
                    (-5.75, 1.0)), 1.0)
    v = _pw(z, kn, ((-2.0, -1.9), (-1.9, -1.1), (-1.1, -1.0), (-1.0, -0.14), (-0.14, 0.18), (0.18, 2.75)), 2.75)
    qv = _pw(z, kn + (9000.0,), ((0.013, 0.0125), (0.0125, 0.0125), (0.0125, 0.0115), (0.0115, 0.01025),
                                 (0.01025, 0.0045), (0.0045, 0.0045), (0.0045, 0.0)), 0.0)
    T = _pw(z, kn + (15000.0, 17500.0, 20000.0, 60000.0),
            ((297.0, 295.557), (295.557, 290.265), (290.265, 290.111), (290.111, 285.542), (285.542, 291.541),
             (291.541, 276.2), (278.0, 203.0), (203.0, 194.0), (194.0, 206.0), (206.0, 270.0)), 270.0)
    return u, v, u, v, T + 0.04, qv


class AtexForcing(ScmForcing):
    experiment = "atex"
    NML = "scm_atex_nml"

    def __init__(self, nml, vgrid, lat_rad, lon_rad, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.momentum_scheme = int(b.get("momentum_vert_advec_scheme", 3))
        self.do_rad = bool(b.get("do_rad", True))
        self.do_geo = bool(b.get("do_geo", True))
        self.do_vadv = bool(b.get("do_vadv", True))
        self.init_time = set_date(*[int(x) for x in b.get("init_date", [0, 0, 0, 0, 0, 0])]) \
            if any(int(x) for x in b.get("init_date", [0])) else None
        self.psfc = 1015.0e2
        self.zsfc = 0.0
        self.u_geos = self.v_geos = None
        self.tdt_lw = self.tdt_sw = None

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        f64 = dict(dtype=torch.float64)
        ps = torch.full((1,), self.psfc, **f64)
        elev = torch.full((1,), self.zsfc, **f64)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        std = [us_std_atm(float(pf[0, k])) for k in range(kdim)]
        T = torch.tensor([[s[0] for s in std]], **f64)
        qv = torch.tensor([[s[1] for s in std]], **f64)
        zf, zh = compute_height(vg, GRAV * elev, T, qv, pf, ph)
        u, v, ug, vg_, Tn, qvn = (np.zeros(kdim) for _ in range(6))
        i, maxerror = 0, 1.0
        while maxerror > 0.001 and i < 10:
            i += 1
            for k in range(kdim):
                u[k], v[k], ug[k], vg_[k], Tn[k], qvn[k] = atex_snd(float(zf[0, k]))
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64),
                                      torch.tensor(qvn[None], **f64), pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= 10:
            raise RuntimeError("atex_forc_init: initial condition did not converge")
        u = np.maximum(u, -8.0)
        self.u_geos = self.tensor(ug[None])
        self.v_geos = self.tensor(vg_[None])
        z = np.zeros(kdim)
        tracers = np.stack([qvn, z, z, z], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    def _elapsed(self, time):
        t0 = self.init_time
        d0, s0 = (t0.days, t0.seconds) if t0 is not None else (0, 0)
        return float(time.seconds - s0) + 86400.0 * (time.days - d0)

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        qv, ql = tr[..., SPHUM], tr[..., LIQ_WAT]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.psfc, dtype=T.dtype, device=T.device)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        zf, zh = compute_height(vg, GRAV * state["elev"], T, qv, pf, ph)
        dp = ph[:, 1:] - ph[:, :-1]
        zero = torch.zeros_like(T)
        omega_f = zero
        omega_h = torch.zeros_like(ph)
        dT_lf = dqv_lf = zero
        if self._elapsed(time) >= 5400.0:
            # inversion: the level above the first (from the top) with q_t > 6.5 g/kg
            qt = qv / (1.0 - qv) + ql
            zi = torch.zeros_like(ps)
            found = torch.zeros_like(ps, dtype=torch.bool)
            for k in range(1, kdim):
                hit = (qt[:, k] > 6.5e-3) & ~found
                zi = torch.where(hit, zf[:, k - 1], zi)
                found = found | hit
            zi_ = zi[:, None]
            rho_g = pf / (RDGAS * T) * GRAV
            omega_f = torch.where(zf <= zi_, rho_g * 0.0065 * zf / zi_,
                                  torch.where(zf <= zi_ + 300.0, rho_g * 0.0065 * (1.0 - (zf - zi_) / 300.0), zero))
            zhk, phk = zh[:, 1:kdim], ph[:, 1:kdim]
            rho_gh = phk / (RDGAS * 0.5 * (T[:, 1:] + T[:, :-1])) * GRAV
            zh0 = torch.zeros_like(zhk)
            om = torch.where(zhk <= zi_, rho_gh * 0.0065 * zhk / zi_,
                             torch.where(zhk <= zi_ + 300.0, rho_gh * 0.0065 * (1.0 - (zhk - zi_) / 300.0), zh0))
            zc = torch.zeros_like(ps)[:, None]
            omega_h = torch.cat([zc, om, zc], dim=1)
            ztop = zh[:, :kdim]                           # interface above each level
            dT_lf = torch.where(zf < zi_, -1.1575e-5 * (3.0 - zf / zi_),
                                torch.where(ztop < zi_ + 300.0, -2.315e-5 * (1.0 - (zf - zi_) / 300.0), zero))
            dT_lf = dT_lf * (pf / P00) ** (RDGAS / CP_AIR)
            dqv_lf = torch.where(zf < zi_, -1.58e-8 * (1.0 - zf / zi_), zero)
        dT_rad = zero
        if self.do_rad:
            cols = [torch.zeros_like(ps)]
            for k in range(1, kdim):
                cols.append(cols[-1] + ql[:, k - 1] * dp[:, k - 1] / GRAV)
            klwp = torch.stack(cols, dim=1)
            frad = 74.0 * torch.exp(-130.0 * 1.0 * klwp)
            frad = torch.cat([frad, frad[:, -1:]], dim=1)
            dT_rad = GRAV / CP_AIR * (frad[:, 1:] - frad[:, :-1]) / dp
        self.tdt_lw = dT_rad
        self.tdt_sw = zero
        dT_vadv = zero
        dtr = torch.zeros_like(tr)
        du_vadv = dv_vadv = zero
        if self.do_vadv:
            dT_adi = RDGAS * T * omega_f / CP_AIR / pf
            dT_vadv = vadv(dt, omega_h, dp, T, self.temp_scheme) + dT_adi
            dtr = torch.stack([vadv(dt, omega_h, dp, tr[..., n], self.tracer_scheme)
                               for n in range(tr.shape[-1])], dim=-1)
            if self.momentum_scheme == 1:
                du_vadv = vadv(dt, omega_h, dp, u, 1)
                dv_vadv = vadv(dt, omega_h, dp, v, 1)
        du_geos = dv_geos = zero
        if self.do_geo:
            fcriolis = 0.375e-4
            du_geos = fcriolis * (v - self.v_geos)
            dv_geos = -fcriolis * (u - self.u_geos)
        dtr = torch.cat([(dtr[..., SPHUM] + dqv_lf)[..., None], dtr[..., 1:]], dim=-1)
        return ForcingTendencies(ps=ps, T=dT_vadv + dT_rad + dT_lf, u=du_vadv + du_geos, v=dv_vadv + dv_geos,
                                 tracers=dtr, omega=omega_f, diag=dict(tdt_radf=dT_rad, tdt_lf=dT_lf))

    def add_tdtlw(self, x):
        return x + self.tdt_lw if self.tdt_lw is not None else x

    def add_tdtsw(self, x):
        return x + self.tdt_sw if self.tdt_sw is not None else x

    def specified_surface_flux(self, sfc: dict) -> dict:
        """scm_surface_flux case('atex'): atex_sfclyr bulk fluxes with u* = 0.3."""
        t_atm, q_atm, u_atm, v_atm, p_atm = (sfc[k] for k in ("t_atm", "q_atm", "u_atm", "v_atm", "p_atm"))
        thlm = t_atm / (p_atm / 1000.0e2) ** (RDGAS / CP_AIR)
        ubar = torch.clamp(torch.sqrt(u_atm ** 2 + v_atm ** 2), min=0.25)
        c10, sst = 0.0013, 298.0
        wpthlp = -c10 * ubar * (thlm - sst * (1000.0 / 1015.0) ** KAPPA)
        wprtp = -c10 * ubar * (q_atm - 0.0198293)
        u_star = torch.full_like(t_atm, 0.3)
        w_atm = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=1.0)
        tv = t_atm * (1.0 + D608 * q_atm)
        rho = p_atm / (RDGAS * tv)
        flux_t = wpthlp * rho * CP_AIR
        flux_q = wprtp * rho
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q,
                    flux_u=-u_atm * u_star * u_star / w_atm, flux_v=-v_atm * u_star * u_star / w_atm,
                    w_atm=w_atm, b_star=GRAV / 292.0 * flux_t / (rho * CP_AIR) / u_star,
                    q_star=flux_q / rho / u_star,
                    dtaudu_atm=zero, dtaudv_atm=zero, dhdt_atm=zero, dedq_atm=zero)
