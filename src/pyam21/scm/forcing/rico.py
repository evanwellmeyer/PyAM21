"""RICO trade-wind cumulus, 'short' composite case (atmos_scm/forc/scm_rico.F90, siena_201207).

vanZanten et al. (2011): piecewise-linear sounding, subsidence and large-scale
cooling / moistening tendencies prescribed in height, geostrophic wind, and (with
do_specified_flux) the RICO bulk surface-layer formulae (rico_sfclyr) over SST 299.8 K.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, KAPPA, RDGAS, RVGAS
from ...sat_vapor_pres import get_sat_vapor_pres
from ..utils import us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, vadv

P00 = 100000.0
D608 = (RVGAS - RDGAS) / RDGAS


def rico_snd(z: float):
    """rico_snd for configuration 'short'."""
    if z < 4000.0:
        u = -9.9 + (-1.9 + 9.9) / 4000.0 * z
    elif z < 12000.0:
        u = -1.9 + (30.0 + 1.9) / (12000.0 - 4000.0) * (z - 4000.0)
    elif z < 13000.0:
        u = 30.0
    elif z < 20000.0:
        u = 30.0 - 30.0 / (20000.0 - 13000.0) * (z - 13000.0)
    else:
        u = 0.0
    v = -3.8
    if z < 740.0:
        qv = 16.0 + (13.8 - 16.0) / 740.0 * z
    elif z < 3260.0:
        qv = 13.8 + (2.4 - 13.8) / (3260.0 - 740.0) * (z - 740.0)
    elif z < 4000.0:
        qv = 2.4 + (1.8 - 2.4) / (4000.0 - 3260.0) * (z - 3260.0)
    elif z < 9000.0:
        qv = 1.8 + (0 - 1.8) / (10000.0 - 4000.0) * (z - 4000.0)
    else:
        qv = 0.0
    if z < 740.0:
        T = 299.2 + (292.0 - 299.2) / 740.0 * z
    elif z < 4000.0:
        T = 292.0 + (278.0 - 292.0) / (4000.0 - 740.0) * (z - 740.0)
    elif z < 15000.0:
        T = 278.0 + (203.0 - 278.0) / (15000.0 - 4000.0) * (z - 4000.0)
    elif z < 17500.0:
        T = 203.0 + (194.0 - 203.0) / (17500.0 - 15000.0) * (z - 15000.0)
    elif z < 20000.0:
        T = 194.0 + (206.0 - 194.0) / (20000.0 - 17500.0) * (z - 17500.0)
    elif z < 60000.0:
        T = 206.0 + (270.0 - 206.0) / (60000.0 - 20000.0) * (z - 20000.0)
    else:
        T = 270.0
    return u, v, u, v, T, qv


class RicoForcing(ScmForcing):
    experiment = "rico"
    NML = "scm_rico_nml"

    def __init__(self, nml, vgrid, lat_rad, lon_rad, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.configuration = str(b.get("configuration", "short")).strip()
        if self.configuration != "short":
            raise NotImplementedError("RICO configuration 'long' (lsforcings.nc)")
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.momentum_scheme = int(b.get("momentum_vert_advec_scheme", 3))
        self.do_geo = bool(b.get("do_geo", True))
        self.do_vadv = bool(b.get("do_vadv", True))
        self.use_online_sfcflux = bool(nml.get("scm_forc_nml", {}).get("use_rico_online_sfcflux", False))
        self.psfc = 1015.4e2
        self.zsfc = 0.0
        self.u_geos = self.v_geos = None

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
        u, v, ug, vg_, Tn, qvn = (np.zeros(kdim) for _ in range(6))
        i, maxerror = 0, 1.0
        while maxerror > 0.001 and i < 10:
            i += 1
            for k in range(kdim):
                u[k], v[k], ug[k], vg_[k], Tn[k], q = rico_snd(float(zf[0, k]))
                qvn[k] = 0.001 * q
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64),
                                      torch.tensor(qvn[None], **f64), pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= 10:
            raise RuntimeError("rico_forc_init: initial condition did not converge")
        self.u_geos = self.tensor(ug[None])
        self.v_geos = self.tensor(vg_[None])
        z = np.zeros(kdim)
        tracers = np.stack([qvn, z, z, z], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    @staticmethod
    def _w(z):
        """Subsidence velocity (m/s) profile."""
        return torch.where(z < 2260.0, (0.005 / 2260.0) * z,
                           torch.where(z < 4000.0, torch.full_like(z, 0.005),
                                       torch.where(z < 5000.0, 0.005 - (0.005 / (5000 - 4000)) * (z - 4000),
                                                   torch.zeros_like(z))))

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.psfc, dtype=T.dtype, device=T.device)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        zf, zh = compute_height(vg, GRAV * state["elev"], T, tr[..., SPHUM], pf, ph)
        omega_f = pf / (RDGAS * T) * GRAV * self._w(zf)
        omega_f = torch.where(zf < 5000.0, omega_f, torch.zeros_like(omega_f))
        zhk, phk = zh[:, 1:kdim], ph[:, 1:kdim]
        om = phk / (RDGAS * 0.5 * (T[:, 1:] + T[:, :-1])) * GRAV * self._w(zhk)
        om = torch.where(zhk < 5000.0, om, torch.zeros_like(om))
        zc = torch.zeros_like(ps)[:, None]
        omega_h = torch.cat([zc, om, zc], dim=1)
        dp = ph[:, 1:] - ph[:, :-1]
        zero = torch.zeros_like(T)
        dT_lf = torch.where(zf < 4000.0, -2.51 / 86400.0 + (-2.18 + 2.51) / (86400.0 * 4000.0) * zf,
                            torch.where(zf < 5000.0, -2.18 / 86400.0 + 2.18 / (86400.0 * (5000.0 - 4000.0))
                                        * (zf - 4000.0), zero))
        dqv_lf = torch.where(zf < 3000.0, 0.001 * (-1.0 / 86400.0 + (0.345 + 1.0) / (86400.0 * 3000.0) * zf),
                             torch.where(zf < 4000.0, torch.full_like(zf, 0.001 * (0.345 / 86400.0)),
                                         torch.where(zf < 5000.0, 0.001 * (0.345 / 86400.0 + (-0.345) /
                                                                          (86400.0 * (5000.0 - 4000.0)) * (zf - 4000.0)),
                                                     zero)))
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
            fcriolis = 2.0 * 2.0 * 3.1415926 / 86400.0 * math.sin(self.lat)
            du_geos = fcriolis * (v - self.v_geos)
            dv_geos = -fcriolis * (u - self.u_geos)
        dtr = torch.cat([(dtr[..., SPHUM] + dqv_lf)[..., None], dtr[..., 1:]], dim=-1)
        return ForcingTendencies(ps=ps, T=zero + dT_vadv + dT_lf, u=du_vadv + du_geos, v=dv_vadv + dv_geos,
                                 tracers=dtr, omega=omega_f, diag=dict(tdt_lf=dT_lf))

    def specified_surface_flux(self, sfc: dict) -> dict:
        """scm_surface_flux case('rico') with use_rico_online_sfcflux = .false.: rico_sfclyr."""
        if self.use_online_sfcflux:
            raise NotImplementedError("rico_sfcflux_online")
        t_atm, q_atm, u_atm, v_atm, p_atm = (sfc[k] for k in ("t_atm", "q_atm", "u_atm", "v_atm", "p_atm"))
        thlm = t_atm / (p_atm / 1000.0e2) ** (RDGAS / CP_AIR)
        lowest, sst, psfc = 20.08, 299.8, 101540.0
        z0 = 0.00015
        ubar = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=0.25)
        r20 = (math.log(20 / z0)) / (math.log(lowest / z0))
        Cm, Ch, Cq = 0.001229 * r20 * r20, 0.001094 * r20 * r20, 0.001133 * r20 * r20
        qsat = float(get_sat_vapor_pres().compute_qs(torch.tensor([sst], dtype=torch.float64),
                                                     torch.tensor([psfc], dtype=torch.float64))[0])
        wpthlp = -Ch * ubar * (thlm - sst * (P00 / psfc) ** KAPPA)
        wprtp = -Cq * ubar * (q_atm - qsat)
        flux_u = -Cm * ubar * u_atm
        flux_v = -Cm * ubar * v_atm
        w_atm = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=1.0)
        u_star = torch.sqrt(-flux_u * w_atm / u_atm)
        rho = p_atm / (RDGAS * t_atm * (1.0 + D608 * q_atm))
        flux_t = wpthlp * rho * CP_AIR
        flux_q = wprtp * rho
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q, flux_u=flux_u, flux_v=flux_v,
                    b_star=GRAV / 292.0 * flux_t / (rho * CP_AIR) / u_star, q_star=flux_q / rho / u_star,
                    dtaudu_atm=zero, dtaudv_atm=zero, dhdt_atm=zero, dedq_atm=zero)
