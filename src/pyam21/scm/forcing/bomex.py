"""BOMEX trade-cumulus case (atmos_scm/forc/scm_bomex.F90, siena_201207).

Siebesma et al. (2003): prescribed subsidence below 2100 m, radiative cooling below
2500 m, large-scale drying below 500 m, geostrophic wind, fixed surface fluxes.
The Fortran's quirks are kept: omega on half levels uses the full-level pressure and
0.0065/2100 in its lowest segment (the full-level omega uses 0.0065/1500).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLV, RDGAS, RVGAS
from ... import vert_advection as va
from ..utils import thetal_to_temp, us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT

P00 = 100000.0

_SCHEMES = {1: va.SECOND_CENTERED, 2: va.FOURTH_CENTERED, 3: va.FINITE_VOLUME_LINEAR,
            4: va.FINITE_VOLUME_PARABOLIC, 5: va.SECOND_CENTERED_WTS, 6: va.FOURTH_CENTERED_WTS}


def bomex_snd(z: float):
    """Initial sounding: u, v, ug, vg, theta_l, qt (mixing ratio, kg/kg)."""
    if z < 700.0:
        u = -8.75
    elif z < 5500.0:
        u = -8.75 + 1.8e-3 * (z - 700)
    else:
        u = 0.0
    v = 0.0
    if z < 520.0:
        qt = 17.0 + (16.3 - 17.0) / (520) * z
    elif z < 1480.0:
        qt = 16.3 + (10.7 - 16.3) / (1480 - 520) * (z - 520)
    elif z < 2000.0:
        qt = 10.7 + (4.2 - 10.7) / (2000 - 1480) * (z - 1480)
    elif z < 5500.0:
        qt = 4.2 - 1.2e-3 * (z - 2000)
    else:
        qt = 0.0
    qt = 0.001 * qt
    qt = qt / (1.0 - qt)
    if z < 520.0:
        thetal = 298.7
    elif z < 1480.0:
        thetal = 298.7 + (302.4 - 298.7) / (1480 - 520) * (z - 520)
    elif z < 2000.0:
        thetal = 302.4 + (308.2 - 302.4) / (2000 - 1480) * (z - 1480)
    else:
        thetal = 308.2 + 3.65e-3 * (z - 2000)
    ug = -10.0 + 1.8e-3 * z if z < 5500.0 else 0.0
    return u, v, ug, 0.0, thetal, qt


class BomexForcing(ScmForcing):
    experiment = "bomex"

    def __init__(self, nml, vgrid, lat_rad, lon_rad, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get("scm_bomex_nml", {})
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.momentum_scheme = int(b.get("momentum_vert_advec_scheme", 3))
        self.do_rad = bool(b.get("do_rad", True))
        self.do_geo = bool(b.get("do_geo", True))
        self.do_vadv = bool(b.get("do_vadv", True))
        self.do_nudge = bool(b.get("do_nudge", True))
        self.psfc = 1015.0e2
        self.zsfc = 0.0
        self.ustar_sfc = float(b.get("ustar_sfc", 0.28))
        self.flux_t_sfc = float(b.get("flux_t_sfc", 9.52))
        self.flux_q_sfc = float(b.get("flux_q_sfc", 153.4))
        self.u_geos = None
        self.v_geos = None

    # --- bomex_forc_init --------------------------------------------------------------------
    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        f64 = dict(dtype=torch.float64)
        ps = torch.full((1,), self.psfc, **f64)
        elev = torch.full((1,), self.zsfc, **f64)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        T = torch.tensor([[us_std_atm(float(pf[0, k]))[0] for k in range(kdim)]], **f64)
        qv = torch.zeros_like(T)
        zf, zh = compute_height(vg, GRAV * elev, T, qv, pf, ph)
        u = np.zeros(kdim); v = np.zeros(kdim); ug = np.zeros(kdim); vgeo = np.zeros(kdim)
        Tn = np.zeros(kdim); qvn = np.zeros(kdim); qln = np.zeros(kdim)
        itmax = 10
        i = 0
        maxerror = 1.0
        while maxerror > 0.001 and i < itmax:
            i += 1
            for k in range(kdim):
                u[k], v[k], ug[k], vgeo[k], thl, qt = bomex_snd(float(zf[0, k]))
                t_snd, qv_snd, ql_snd = thetal_to_temp(thl, qt, float(pf[0, k]))
                Tn[k] = max(t_snd, 200.0)
                qvn[k] = qv_snd
                qln[k] = ql_snd
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64),
                                      torch.tensor(qvn[None], **f64), pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= itmax:
            raise RuntimeError("bomex_forc_init: initial condition did not converge")
        self.u_geos = self.tensor(ug[None])
        self.v_geos = self.tensor(vgeo[None])
        qa = np.where(qln > 0.0, 1.0, 0.0)
        tracers = np.stack([qvn, qln, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]),
                    elev=self.tensor([self.zsfc]))

    # --- update_bomex_forc ------------------------------------------------------------------
    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        qv = tr[..., SPHUM]
        elev = state["elev"]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.psfc, dtype=T.dtype, device=T.device)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        zf, zh = compute_height(vg, GRAV * elev, T, qv, pf, ph)
        zero = torch.zeros_like(T)

        rho_g = pf / (RDGAS * T) * GRAV
        omega_f = torch.where(zf < 1500.0, rho_g * (0.0065 / 1500.0) * zf,
                              torch.where(zf < 2100.0,
                                          rho_g * (0.0065 - (0.0065 / (2100 - 1500)) * (zf - 1500)),
                                          zero))
        # half levels k = 2..kdim (1-based): pressure pf(k), temperature mean of k, k-1
        zhk = zh[:, 1:kdim]
        rho_gh = pf[:, 1:] / (RDGAS * 0.5 * (T[:, 1:] + T[:, :-1])) * GRAV
        zh0 = torch.zeros_like(zhk)
        om_mid = torch.where(zhk < 1500.0, rho_gh * (0.0065 / 2100.0) * zhk,
                             torch.where(zhk < 2100.0,
                                         rho_gh * (0.0065 - (0.0065 / (2100 - 1500)) * (zhk - 1500)),
                                         zh0))
        zc = torch.zeros_like(ps)[:, None]
        omega_h = torch.cat([zc, om_mid, zc], dim=1)

        dp = ph[:, 1:] - ph[:, :-1]
        pi_fac = (pf / P00) ** (RDGAS / CP_AIR)

        dT_rad = zero
        if self.do_rad:
            dT_rad = torch.where(zf < 1500.0, -2.315e-5 * pi_fac,
                                 torch.where(zf < 2500.0,
                                             (-2.315e-5 + 2.315e-5 / (2500.0 - 1500.0) * (zf - 1500.0)) * pi_fac,
                                             zero))
        dqv_lf = torch.where(zf < 300.0, torch.full_like(zf, -1.2e-8),
                             torch.where(zf < 500.0, -(1.2e-8 - 1.2e-8 * (zf - 300.0) / (500.0 - 300.0)), zero))
        dT_vadv = zero
        dtr_vadv = torch.zeros_like(tr)
        du_vadv = zero
        dv_vadv = zero
        if self.do_vadv:
            dT_adi = RDGAS * T * omega_f / CP_AIR / pf
            dT_vadv = va.vert_advection(dt, omega_h, dp, T, scheme=_SCHEMES[self.temp_scheme],
                                        form=va.ADVECTIVE_FORM) + dT_adi
            ts = _SCHEMES[self.tracer_scheme]
            dtr_vadv = torch.stack(
                [va.vert_advection(dt, omega_h, dp, tr[..., n], scheme=ts, form=va.ADVECTIVE_FORM)
                 for n in (SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT)], dim=-1)
            if self.momentum_scheme == 1:
                du_vadv = va.vert_advection(dt, omega_h, dp, u, scheme=va.SECOND_CENTERED, form=va.ADVECTIVE_FORM)
                dv_vadv = va.vert_advection(dt, omega_h, dp, v, scheme=va.SECOND_CENTERED, form=va.ADVECTIVE_FORM)
        du_geos = zero
        dv_geos = zero
        if self.do_geo:
            fcriolis = 2.0 * 2.0 * 3.1415926 / 86400.0 * math.sin(self.lat)
            du_geos = fcriolis * (v - self.v_geos)
            dv_geos = -fcriolis * (u - self.u_geos)
        AU = du_vadv + du_geos + zero + zero
        AV = dv_vadv + dv_geos + zero + zero
        AT = dT_rad + dT_vadv + zero
        Atr = dtr_vadv.clone()
        Atr[..., SPHUM] = dtr_vadv[..., SPHUM] + dqv_lf
        return ForcingTendencies(ps=ps, T=AT, u=AU, v=AV, tracers=Atr, omega=omega_f,
                                 diag=dict(tdt_radf=dT_rad, tdt_vadv=dT_vadv, qvdt_lf=dqv_lf))

    # --- scm_surface_flux, case 'bomex' ------------------------------------------------------
    def specified_surface_flux(self, sfc: dict) -> dict:
        t_atm, q_atm, u_atm, v_atm, p_atm = (sfc[k] for k in ("t_atm", "q_atm", "u_atm", "v_atm", "p_atm"))
        d622 = RDGAS / RVGAS
        d608 = (1.0 - d622) / d622
        tv_atm = t_atm * (1.0 + d608 * q_atm)
        rho = p_atm / (RDGAS * tv_atm)
        u_star = torch.full_like(t_atm, self.ustar_sfc)
        flux_t = torch.full_like(t_atm, self.flux_t_sfc)
        flux_q = torch.full_like(t_atm, self.flux_q_sfc)
        w_atm = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=1.0)
        flux_u = -u_atm * u_star * u_star / w_atm
        flux_v = -v_atm * u_star * u_star / w_atm
        flux_q = flux_q / HLV
        b_star = GRAV / 292.0 * flux_t / (CP_AIR * rho) / u_star
        q_star = flux_q / rho / u_star
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q, flux_u=flux_u, flux_v=flux_v,
                    w_atm=w_atm, b_star=b_star, q_star=q_star,
                    dtaudu_atm=zero, dtaudv_atm=zero, dhdt_atm=zero, dedq_atm=zero)
