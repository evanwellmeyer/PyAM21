"""DYCOMS-II RF02 drizzling stratocumulus (atmos_scm/forc/scm_rf02.F90, siena_201207).

Ackerman et al. (2009): analytic theta_l / q_t sounding, otherwise the RF01 forcing
(divergence below p_omega_zero, geostrophic wind = initial wind, GCSS longwave
parameterisation) with the RF02 variants of the radiation, and fixed surface fluxes
with a prescribed surface air density.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLV
from ..utils import thetal_to_temp, us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .rf01 import RF01Forcing


def rf02_snd(z: float, configuration: str = "base"):
    zi, ztop = 795.0, 2500.0
    q0 = 5.0e-3 if configuration != "dry" else 4.0e-3
    if z <= ztop:
        u = 3.0 + 0.0043 * z
        v = -9.0 + 0.0056 * z
        if z < zi:
            thetal, qt = 288.3, 9.45e-3
        else:
            thetal = 295.0 + (z - zi) ** (1.0 / 3.0)
            qt = q0 - 3.0e-3 * (1.0 - math.exp((zi - z) / 500.0))
    else:
        u = 3.0 + 0.0043 * ztop
        v = -9.0 + 0.0056 * ztop
        thetal = 295.0 + (ztop - zi) ** (1.0 / 3.0) + 3.5e-3 * (z - ztop)
        qt = max(q0 - 3.0e-3 * (1.0 - math.exp((zi - ztop) / 500.0)) - 2.0e-6 * (z - ztop), 0.0)
    return u, v, thetal, qt


class RF02Forcing(RF01Forcing):
    experiment = "rf02"
    NML = "scm_rf02_nml"
    QT_AS_MIXING_RATIO = False
    CLIP_WARMING = True

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path=None, device="cpu", dtype=torch.float64):
        b = nml.get(self.NML, {})
        self.configuration = str(b.get("configuration", "base"))
        self.rho_sfc = float(b.get("rho_sfc", 1.21))
        nml = dict(nml)
        nml[self.NML] = {"flux_t_sfc": 16.0, "flux_q_sfc": 93.0, **b}
        super().__init__(nml, vgrid, lat_rad, lon_rad, case_path, device, dtype)

    def _read_sounding(self, case_path):
        pass

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        f64 = dict(dtype=torch.float64)
        ps = torch.full((1,), self.psfc, **f64)
        elev = torch.full((1,), self.zsfc, **f64)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        T = torch.tensor([[us_std_atm(float(pf[0, k]))[0] for k in range(kdim)]], **f64)
        qv = torch.tensor([[us_std_atm(float(pf[0, k]))[1] for k in range(kdim)]], **f64)
        zf, zh = compute_height(vg, GRAV * elev, T, qv, pf, ph)
        u = np.zeros(kdim); v = np.zeros(kdim); Tn = np.zeros(kdim); qvn = np.zeros(kdim); qln = np.zeros(kdim)
        i, maxerror = 0, 1.0
        while maxerror > 0.001 and i < 10:
            i += 1
            for k in range(kdim):
                u[k], v[k], thl, rt = rf02_snd(float(zf[0, k]), self.configuration)
                t_k, qvn[k], qln[k] = thetal_to_temp(thl, rt, float(pf[0, k]))
                Tn[k] = max(t_k, 200.0)
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64), torch.tensor(qvn[None], **f64),
                                      pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= 10:
            raise RuntimeError("rf02_forc_init: initial condition did not converge")
        qa = np.where(qln > 0.0, 1.0, 0.0)
        self.u_geos = self.tensor(u[None])
        self.v_geos = self.tensor(v[None])
        tracers = np.stack([qvn, qln, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    def specified_surface_flux(self, sfc: dict) -> dict:
        t_atm, u_atm, v_atm = sfc["t_atm"], sfc["u_atm"], sfc["v_atm"]
        rho = torch.full_like(t_atm, self.rho_sfc)
        u_star = torch.full_like(t_atm, self.ustar_sfc)
        flux_t = torch.full_like(t_atm, self.flux_t_sfc)
        flux_q = torch.full_like(t_atm, self.flux_q_sfc) / HLV
        w_atm = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=1.0)
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q,
                    flux_u=-u_atm * u_star * u_star / w_atm, flux_v=-v_atm * u_star * u_star / w_atm,
                    w_atm=w_atm, b_star=GRAV / 292.0 * flux_t / (CP_AIR * rho) / u_star,
                    q_star=flux_q / rho / u_star,
                    dtaudu_atm=zero, dtaudv_atm=zero, dhdt_atm=zero, dedq_atm=zero)
