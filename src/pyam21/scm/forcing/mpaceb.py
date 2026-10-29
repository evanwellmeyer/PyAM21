"""M-PACE period B, Arctic mixed-phase stratocumulus over open water, 9-10 Oct 2004
(atmos_scm/forc/scm_mpaceb.F90, siena_201207).

Klein et al. (2009): a single initial sounding (also the nudging target for the winds),
divergence 5.8e-6 /s below the observed inversion, prescribed horizontal cooling and
moistening, fixed SST 274.01 K and surface fluxes (136.5 W/m2 sensible, 107.7 W/m2
latent) enforced by iterating the surface-layer scheme (mpaceb_surface_flux_loop).
Condensate above p_cld_zero is evaporated in place every step.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLS, HLV, PI, RDGAS
from ..vert_grid import compute_pres_full, compute_pres_half
from .interp import interp_2d_field, interp_2d_field_t
from .base import (ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT, surface_flux_loop,
                   vadv)

P00 = 100000.0
MISSING = -999.0


def read_sounding(path, nlev=550):
    """formatted.mpaceB.initial.sounding: (f8.3,1X,f5.1,1X,f5.1,1X,f7.3,1X,f7.5,1X,f7.5)."""
    cols = ((0, 8), (9, 14), (15, 20), (21, 28), (29, 36), (37, 44))
    rows = []
    with open(path) as f:
        lines = f.read().splitlines()[4:4 + nlev]
    for line in lines:
        rows.append([float(line[a:b]) for a, b in cols])
    return np.array(rows)


class MpacebForcing(ScmForcing):
    experiment = "mpaceb"
    NML = "scm_mpaceb_nml"
    SOUNDING = "formatted.mpaceB.initial.sounding"
    IVERT = 550

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path, pdamp, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.vert_advec_cond = bool(b.get("vert_advec_cond", True))
        self.wind_relaxation_tau = float(b.get("wind_relaxation_tau", 3.0 * 3600.0))
        self.p_omega_zero = float(b.get("p_omega_zero", 50000.0))
        self.p_cld_zero = float(b.get("p_cld_zero", 50000.0))
        self.pdamp = pdamp
        a = read_sounding(os.path.join(case_path, self.SOUNDING), self.IVERT)
        q, ql = a[:, 4] / 1000.0, a[:, 5] / 1000.0
        self.plev = a[:, 0] * 100.0
        self.u_forc, self.v_forc, self.t_forc = a[:, 1], a[:, 2], a[:, 3]
        self.q_forc = q / (1.0 + q + ql)
        self.ql_forc = ql / (1.0 + q + ql)
        self.ps_forc = 100.0 * 1010.00
        self.div_forc = 5.8 * 1.0e-6
        self.pib_forc = 850.0 * 100.0
        self.sst_forc = 274.01
        f = nml.get("scm_forc_nml", {})
        self.senflux = 136.5
        self.evapflux = 107.7 / HLV
        if bool(f.get("do_specified_tskin", False)):
            self.surface.tskin = self.sst_forc

    def _ext(self, field):
        """Profile extended with a p = 0 level carrying the top value (plev_int, field_int)."""
        return np.append(self.plev, 0.0), np.append(field, field[-1])

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        ps = self.ps_forc
        phalf = np.asarray(vg.peta, dtype=np.float64) + ps * np.asarray(vg.eta, dtype=np.float64)
        lph = np.log(np.maximum(phalf, np.finfo(np.float64).eps))
        pf = np.exp((phalf[1:] * lph[1:] - phalf[:-1] * lph[:-1]) / (phalf[1:] - phalf[:-1]) - 1.0)
        T = interp_2d_field(pf, *self._ext(self.t_forc))
        qv = interp_2d_field(pf, *self._ext(self.q_forc))
        ql = interp_2d_field(pf, *self._ext(self.ql_forc))
        u = interp_2d_field(pf, *self._ext(self.u_forc))
        v = interp_2d_field(pf, *self._ext(self.v_forc))
        qa = np.where(ql > 0.0, 1.0, 0.0)
        tracers = np.stack([qv, ql, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(T[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([ps]), elev=self.tensor([0.0]))

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.ps_forc, dtype=T.dtype, device=T.device)
        pfull = compute_pres_full(vg, ps)
        phalf = compute_pres_half(vg, ps)
        dp = phalf[:, 1:] - phalf[:, :-1]
        # evaporate the condensate above p_cld_zero, in place
        qv, ql, qi, qa = tr[..., SPHUM], tr[..., LIQ_WAT], tr[..., ICE_WAT], tr[..., CLD_AMT]
        hi = pfull < self.p_cld_zero
        zero = torch.zeros_like(T)
        T = torch.where(hi, T - HLV * ql / CP_AIR - HLS * qi / CP_AIR, T)
        qv = torch.where(hi, qv + ql + qi, qv)
        ql = torch.where(hi, zero, ql)
        qi = torch.where(hi, zero, qi)
        qa = torch.where(hi, zero, qa)
        tr = torch.stack([qv, ql, qi, qa], dim=-1)
        pinv = self.pib_forc
        if pinv < self.p_omega_zero:
            raise RuntimeError("update_mpaceb_forc: pinv_obs < p_omega_zero")
        div = self.div_forc
        psc = ps[:, None]
        omega_half = torch.where(phalf > pinv, div * (psc - phalf), div * (psc - pinv) + 0.0 * phalf)
        omega_full = torch.where(pfull > pinv, div * (psc - pfull), div * (psc - pinv) + 0.0 * pfull)
        above = pfull <= self.pdamp
        # temperature
        at_adi = RDGAS * T * omega_full / CP_AIR / pfull
        adv_v = vadv(dt, omega_half, dp, T, self.temp_scheme)
        adv_h = torch.clamp(-15.0 * (1.0 - (psc - pfull) / 21818.0), max=-4.0) / 86400.0
        at_adi = torch.where(above, zero, at_adi)
        adv_v = torch.where(above, zero, adv_v)
        adv_h = torch.where(above, zero, adv_h)
        AT = at_adi + adv_v + adv_h
        # water vapour
        adv_v = vadv(dt, omega_half, dp, qv, self.tracer_scheme)
        adv_h = torch.clamp(-3.0 * (1.0 - (psc - pfull) / 15171.0), max=0.164) / 1000.0 / 86400.0
        AQ = torch.where(above, zero, adv_v) + torch.where(above, zero, adv_h)
        # winds relaxed to the sounding
        tau = max(self.wind_relaxation_tau, float(dt))
        AU = (interp_2d_field_t(pfull, *self._ext(self.u_forc)) - u) / tau
        AV = (interp_2d_field_t(pfull, *self._ext(self.v_forc)) - v) / tau
        AL = AI = AA = zero
        if self.vert_advec_cond:
            AL = zero + vadv(dt, omega_half, dp, ql, self.tracer_scheme)
            AI = zero + vadv(dt, omega_half, dp, qi, self.tracer_scheme)
            AA = zero + vadv(dt, omega_half, dp, qa, self.tracer_scheme)
        return ForcingTendencies(ps=ps, T=AT, u=AU, v=AV, tracers=torch.stack([AQ, AL, AI, AA], dim=-1),
                                 omega=omega_full, diag={}, state=dict(T=T, tracers=tr))

    def specified_surface_flux(self, sfc: dict) -> dict:
        out = surface_flux_loop(sfc["surface_flux"], sfc, self.senflux, self.evapflux)
        zero = torch.zeros_like(sfc["t_atm"])
        out.update(dhdt_surf=zero, dedt_surf=zero, dedq_surf=zero, dhdt_atm=zero, dedq_atm=zero)
        return out
