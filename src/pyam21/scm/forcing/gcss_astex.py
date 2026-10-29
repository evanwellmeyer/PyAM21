"""GCSS ASTEX Lagrangian 1 stratocumulus transition (atmos_scm/forc/gcss_astex.F90, siena_201207).

Hourly-composite sounding 12 of Bretherton's ASTEX Lagrangian 1 interpolated in
pressure, otherwise the RF01 forcing (divergence below p_omega_zero, geostrophic wind
from the sounding, GCSS longwave parameterisation with the inversion at q_t = 10 g/kg)
and constant surface fluxes given directly in kinematic units.
"""
from __future__ import annotations

import numpy as np
import torch

from ...constants import GRAV, KAPPA
from ..utils import locate, us_std_atm
from ..vert_grid import compute_pres_full
from .rf01 import RF01Forcing

P00 = 100000.0


class GcssAstexForcing(RF01Forcing):
    experiment = "gcss_astex"
    NML = "gcss_astex_nml"
    SOUNDING = "gcss_astex_lagr1_h12"
    KSND = 42
    PSFC = 1029.0e2
    ZI_QT = 0.01

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path, device="cpu", dtype=torch.float64):
        b = nml.get(self.NML, {})
        self.configuration = str(b.get("configuration", "gcss")).strip()
        if self.configuration == "dry":
            raise NotImplementedError("gcss_astex configuration = 'dry'")
        super().__init__(nml, vgrid, lat_rad, lon_rad, case_path, device, dtype)
        self.divf = float(b.get("divf", 5.0e-6))
        self.p_omega_zero = float(b.get("p_omega_zero", 85000.0))
        self.flux_tw = float(b.get("flux_tw", 1.0e-2))
        self.flux_qw = float(b.get("flux_qw", 1.0e-5))

    def _read_sounding(self, case_path):
        import os
        rows = []
        with open(os.path.join(case_path, self.SOUNDING)) as f:
            lines = f.read().splitlines()[7:7 + self.KSND]
        for line in lines:
            vals = line.split()
            rows.append([float(x) for x in vals[:9]])
        a = np.array(rows)
        self.p_snd = a[:, 0] * 100.0
        theta = a[:, 1]
        self.qv_snd = a[:, 2] + a[:, 7]
        self.u_snd, self.v_snd, self.ug_snd, self.vg_snd = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        self.ql_snd = np.zeros_like(theta)
        self.T_snd = theta * (self.p_snd / P00) ** KAPPA

    def snd_p(self, p, u, v, ug, vg, T, qv, ql):
        """gcss_astex_snd: outside the sounding the inputs are returned unchanged."""
        k = locate(self.p_snd, p)
        if k < 1 or k >= self.KSND:
            return u, v, ug, vg, T, qv, ql
        k0 = k - 1
        b = (p - self.p_snd[k0]) / (self.p_snd[k0 + 1] - self.p_snd[k0])
        a = 1.0 - b
        lin = lambda x: a * x[k0] + b * x[k0 + 1]
        qv = lin(self.qv_snd) * 0.001
        qv = qv / (1.0 + qv)
        return (lin(self.u_snd), lin(self.v_snd), lin(self.ug_snd), lin(self.vg_snd), lin(self.T_snd), qv,
                lin(self.ql_snd) * 0.001)

    def initial_state(self, time) -> dict:
        kdim = self.vgrid.nlev
        pf = compute_pres_full(self.vgrid, torch.full((1,), self.psfc, dtype=torch.float64))
        u, v, ug, vg_, T, qv, ql = (np.zeros(kdim) for _ in range(7))
        for k in range(kdim):
            p = float(pf[0, k])
            t_std, q_std = us_std_atm(p)
            u[k], v[k], ug[k], vg_[k], T[k], qv[k], ql[k] = self.snd_p(p, 0.0, 0.0, 0.0, 0.0, t_std, q_std, 0.0)
            qv[k] = qv[k] / (1.0 + qv[k])        # (the Fortran converts to specific humidity twice)
        qa = np.where(ql > 0.0, 1.0, 0.0)
        self.u_geos = self.tensor(ug[None])
        self.v_geos = self.tensor(vg_[None])
        tracers = np.stack([qv, ql, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(T[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    def specified_surface_flux(self, sfc: dict) -> dict:
        """scm_surface_flux case('gcss_astex'): kinematic fluxes, no density or latent heat."""
        t_atm, u_atm, v_atm = sfc["t_atm"], sfc["u_atm"], sfc["v_atm"]
        u_star = torch.full_like(t_atm, self.ustar_sfc)
        flux_t = torch.full_like(t_atm, self.flux_tw)
        flux_q = torch.full_like(t_atm, self.flux_qw)
        w_atm = torch.clamp(torch.sqrt(u_atm * u_atm + v_atm * v_atm), min=1.0)
        zero = torch.zeros_like(t_atm)
        return dict(u_star=u_star, flux_t=flux_t, flux_q=flux_q,
                    flux_u=-u_atm * u_star * u_star / w_atm, flux_v=-v_atm * u_star * u_star / w_atm,
                    w_atm=w_atm, b_star=GRAV / 292.0 * flux_t / u_star, q_star=flux_q / u_star,
                    dtaudu_atm=zero, dtaudv_atm=zero, dhdt_atm=zero, dedq_atm=zero)
