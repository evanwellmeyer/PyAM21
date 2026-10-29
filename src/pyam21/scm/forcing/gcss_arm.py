"""GCSS ARM shallow cumulus over land, 21 June 1997 (atmos_scm/forc/scm_gcss_arm.F90, siena_201207).

Brown et al. (2002): theta_l / q_t sounding, constant geostrophic wind, prescribed
large-scale advective and radiative theta / q_t tendencies below 3 km interpolated in
time, no vertical advection, and prescribed sensible / latent heat fluxes over a
specified land surface with u* from a Monin-Obukhov diagnosis (diag_ustar).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLV, KAPPA, RDGAS, RVGAS
from ...time_manager import set_date, set_time
from ..utils import thetal_to_temp, us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SurfaceSpec, SPHUM

P00 = 1000.0e2
D608 = (RVGAS - RDGAS) / RDGAS
ATHETA = (0.000, 0.000, 0.000, -0.080, -0.160, -0.160)
RTHETA = (-0.125, 0.000, 0.000, 0.000, 0.000, -0.100)
ART = (0.080, 0.020, -0.040, -0.100, -0.160, -0.300)
FLX_TIMES = (41400.0, 55800.0, 64800.0, 68400.0, 77400.0, 86400.0, 93600.0)
FLX_H = (-30.0, 90.0, 140.0, 140.0, 100.0, -10.0, -10.0)
FLX_LE = (5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0)


def _seg(z, knots, v0, slopes_to, top):
    """The Fortran's '<=' piecewise-linear chain."""
    zlo, a = 0.0, v0
    for zk, b in zip(knots, slopes_to):
        if z <= zk:
            return a + (b - a) / (zk - zlo) * (z - zlo) if zlo > 0.0 else a + (b - a) / zk * z
        zlo, a = zk, b
    return top(z)


def gcss_arm_snd(z: float):
    kn = (50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0)
    qt = _seg(z, kn + (5500.0,), 15.2, (15.17, 14.98, 14.8, 14.7, 13.5, 3.0, 3.0),
              lambda z: max(0.0, 3.0 - 8.0e-3 * (z - 5500.0)))
    thl = _seg(z, kn + (5500.0,), 299.0, (301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2),
               lambda z: 343.2 + 0.0e-3 * (z - 5500.0))
    return 10.0, 0.0, 10.0, 0.0, thl, 0.001 * qt


def gcss_arm_sfcflx(time: float):
    t = FLX_TIMES
    if time <= t[0]:
        return FLX_H[0], FLX_LE[0]
    if time >= t[-1]:
        return FLX_H[-1], FLX_LE[-1]
    for i in range(len(t) - 1):
        if t[i] <= time < t[i + 1]:
            a = (time - t[i]) / (t[i + 1] - t[i])
            return (1.0 - a) * FLX_H[i] + a * FLX_H[i + 1], (1.0 - a) * FLX_LE[i] + a * FLX_LE[i + 1]


def diag_ustar(z, bflx, wnd, z0: float):
    """Monin-Obukhov u* from the buoyancy flux (4 fixed-point iterations), vectorised."""
    g, vonk, pi = 9.81, 0.4, 3.141592654
    am, bm = 4.8, 19.3
    lnz = torch.log(z / z0)
    klnz = vonk / lnz
    c1 = pi / 2.0 - 3.0 * math.log(2.0)
    ustar = wnd * klnz
    active = torch.abs(bflx) > 1.0e-6
    bsafe = torch.where(active, bflx, torch.ones_like(bflx))
    for _ in range(4):
        lmo = -ustar ** 3 / (vonk * bsafe)
        zeta = z / lmo
        stable = zeta > 0.0
        huge = stable & (zeta > 1.0e10)
        zs = torch.where(stable & ~huge, zeta, torch.zeros_like(zeta))
        u_st = vonk * wnd / (lnz + am * zs)
        zu = torch.where(stable, torch.zeros_like(zeta), zeta)
        x = torch.sqrt(torch.sqrt(1.0 - bm * zu))
        psi1 = 2.0 * torch.log(1.0 + x) + torch.log(1.0 + x * x) - 2.0 * torch.atan(x) + c1
        u_un = wnd * vonk / (lnz - psi1)
        new = torch.where(huge, torch.full_like(ustar, 1e-10), torch.where(stable, u_st, u_un))
        ustar = torch.where(active, new, ustar)
        active = active & ~huge
    return ustar


class GcssArmForcing(ScmForcing):
    experiment = "gcss_arm"
    NML = "scm_gcss_arm_nml"

    def __init__(self, nml, vgrid, lat_rad, lon_rad, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.do_geo = bool(b.get("do_geo", True))
        self.rad_scheme = str(b.get("rad_scheme", "simplified")).strip()
        t0 = set_date(*[int(x) for x in b.get("init_date", [0] * 6)])
        self.days0 = t0.days
        self.dt_atmos = int(b.get("dt_atmos", 60))
        # the flux routine keeps its own clock, advanced by dt_atmos on every call
        self.flux_time = set_date(*[int(x) for x in b.get("current_date", [0] * 6)]) - set_time(self.dt_atmos)
        self.psfc = 970.0e2
        self.zsfc = 0.0
        self.u_geos = self.v_geos = None
        f = nml.get("scm_forc_nml", {})
        spec = SurfaceSpec()
        if bool(f.get("do_specified_tskin", False)):
            spec.tskin = 300.0
        if bool(f.get("do_specified_albedo", False)):
            spec.albedo_obs = 0.2
        if bool(f.get("do_specified_rough_leng", False)):
            spec.rough_mom, spec.rough_heat = 0.035, 0.035 / 7.4
        self.surface = spec

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        f64 = dict(dtype=torch.float64)
        ps = torch.full((1,), self.psfc, **f64)
        elev = torch.full((1,), self.zsfc, **f64)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        std = [us_std_atm(float(pf[0, k])) for k in range(kdim)]
        zf, zh = compute_height(vg, GRAV * elev, torch.tensor([[s[0] for s in std]], **f64),
                                torch.tensor([[s[1] for s in std]], **f64), pf, ph)
        u, v, ug, vg_, Tn, qvn, qln = (np.zeros(kdim) for _ in range(7))
        i, maxerror = 0, 1.0
        while maxerror > 0.001 and i < 10:
            i += 1
            for k in range(kdim):
                u[k], v[k], ug[k], vg_[k], thl, qt = gcss_arm_snd(float(zf[0, k]))
                t_k, qvn[k], qln[k] = thetal_to_temp(thl, qt, float(pf[0, k]))
                Tn[k] = max(t_k, 200.0)
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64),
                                      torch.tensor(qvn[None], **f64), pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= 10:
            raise RuntimeError("gcss_arm_forc_init: initial condition did not converge")
        self.u_geos = self.tensor(ug[None])
        self.v_geos = self.tensor(vg_[None])
        qa = np.where(qln > 0.0, 1.0, 0.0)
        tracers = np.stack([qvn, qln, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.psfc, dtype=T.dtype, device=T.device)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        zf, zh = compute_height(vg, GRAV * state["elev"], T, tr[..., SPHUM], pf, ph)
        exner = (pf / P00) ** KAPPA
        true_time = time.seconds + 86400.0 * (time.days - self.days0)
        i1 = min(max(math.floor((true_time - 41400.0) / 10800.0) + 1, 1), 5)
        i2 = i1 + 1
        a = (true_time - (41400.0 + 10800.0 * (i1 - 1))) / (10800.0 if i1 < 5 else 9000.0)
        j1, j2 = i1 - 1, i2 - 1
        if self.rad_scheme == "simplified":
            theta_tmp = (1.0 - a) * ATHETA[j1] + a * ATHETA[j2]
            rad_tmp = (1.0 - a) * RTHETA[j1] + a * RTHETA[j2]
        else:
            theta_tmp = (1.0 - a) * (ATHETA[j1] + 0.0) + a * (ATHETA[j2] + 0.0)
            rad_tmp = 0.0
        rt_tmp = (1.0 - a) * ART[j1] + a * ART[j2]
        theta_tmp, rad_tmp, rt_tmp = theta_tmp / 3600.0, rad_tmp / 3600.0, rt_tmp / (3600.0 * 1000.0)
        zi = torch.trunc(zf)                                # select case (int(zf))
        low = (zi >= 0) & (zi <= 999)
        mid = (zi >= 1000) & (zi <= 2999)
        b = torch.where(mid, 1.0 - (zf - 1000.0) / 2000.0, torch.ones_like(zf))
        on = low | mid
        zero = torch.zeros_like(T)
        dqv_lf = torch.where(on, b * rt_tmp, zero)
        dT_lf = torch.where(on, b * (theta_tmp + rad_tmp) * exner, zero)
        dT_rad = torch.where(on, b * rad_tmp * exner, zero)
        du_geos = dv_geos = zero
        if self.do_geo:
            fcriolis = 2.0 * 2.0 * 3.1415926 / 86400.0 * math.sin(self.lat)
            du_geos = fcriolis * (v - self.v_geos)
            dv_geos = -fcriolis * (u - self.u_geos)
        dtr = torch.zeros_like(tr)
        dtr = torch.cat([dqv_lf[..., None], dtr[..., 1:]], dim=-1)
        # (dT_rad is part of dT_lf already and is added once more, as in the Fortran)
        return ForcingTendencies(ps=ps, T=dT_rad + zero + dT_lf, u=du_geos, v=dv_geos, tracers=dtr,
                                 omega=zero, diag=dict(tdt_radf=dT_rad, tdt_lf=dT_lf))

    def specified_surface_flux(self, sfc: dict) -> dict:
        """scm_surface_flux case('gcss_arm'). Only flux_t/q/u/v, u*, b*, q* and the
        derivative terms are replaced; w_atm and dtaudu/dv stay from surface_flux."""
        t_atm, q_atm, u_atm, v_atm, p_atm, z_atm = (sfc[k] for k in ("t_atm", "q_atm", "u_atm", "v_atm",
                                                                      "p_atm", "z_atm"))
        rho = p_atm / (RDGAS * t_atm * (1.0 + D608 * q_atm))
        thlm = t_atm / (p_atm / 1000.0e2) ** (RDGAS / CP_AIR)
        self.flux_time = self.flux_time + set_time(self.dt_atmos)
        true_time = self.flux_time.seconds + 86400.0 * (self.flux_time.days - self.days0)
        H, LE = gcss_arm_sfcflx(true_time)
        flux_t = torch.full_like(t_atm, H)
        flux_q = torch.full_like(t_atm, LE)
        bflx = GRAV / thlm * (flux_t / (CP_AIR * rho))
        ubar = torch.clamp(torch.sqrt(u_atm ** 2 + v_atm ** 2), min=0.25)
        u_star = diag_ustar(z_atm, bflx, ubar, 0.035)
        flux_u = -u_atm * u_star * u_star / ubar
        flux_v = -v_atm * u_star * u_star / ubar
        flux_q = flux_q / HLV
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q, flux_u=flux_u, flux_v=flux_v,
                    b_star=GRAV / thlm * flux_t / (rho * CP_AIR) / u_star, q_star=flux_q / rho / u_star,
                    dhdt_surf=zero, dedt_surf=zero, dedq_surf=zero, dhdt_atm=zero, dedq_atm=zero)
