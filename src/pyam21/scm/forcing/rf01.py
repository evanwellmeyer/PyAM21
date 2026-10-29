"""DYCOMS-II RF01 nocturnal stratocumulus (atmos_scm/forc/scm_rf01.F90, siena_201207).

Stevens et al. (2005): sounding from rf01_idealb.dat.new2, divergence divf below
p_omega_zero, geostrophic wind equal to the initial wind, the GCSS longwave
parameterisation from the liquid water path (applied through AT and, for the
turbulence scheme, as tdt_lw), fixed surface fluxes.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLV, RDGAS, RVGAS
from ..utils import locate, us_std_atm
from ..vert_grid import compute_height, compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT, fixed_width_table, vadv_all

P00 = 100000.0


class RF01Forcing(ScmForcing):
    experiment = "rf01"
    NML = "scm_rf01_nml"
    SOUNDING = "rf01_idealb.dat.new2"
    KSND = 147
    PSFC = 1017.8e2

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.momentum_scheme = int(b.get("momentum_vert_advec_scheme", 3))
        self.divf = float(b.get("divf", 3.75e-6))
        self.p_omega_zero = float(b.get("p_omega_zero", 70000.0))
        self.do_rad = bool(b.get("do_rad", True))
        self.do_geo = bool(b.get("do_geo", True))
        self.do_vadv = bool(b.get("do_vadv", True))
        self.psfc = self.PSFC
        self.zsfc = 0.0
        self.ustar_sfc = float(b.get("ustar_sfc", 0.25))
        self.flux_t_sfc = float(b.get("flux_t_sfc", 15.0))
        self.flux_q_sfc = float(b.get("flux_q_sfc", 115.0))
        self._read_sounding(case_path)
        self.u_geos = self.v_geos = None
        self.tdt_lw = None
        self.tdt_sw = None

    def _read_sounding(self, case_path):
        rows = fixed_width_table(os.path.join(case_path, self.SOUNDING), (9, 8, 7, 7, 8), self.KSND)
        self.z_snd, self.p_snd, self.T_snd, self.qv_snd, self.ql_snd = (np.array(c) for c in zip(*rows))

    def snd(self, z):
        u, v = 7.0, -5.5
        k = locate(self.z_snd, z)
        if k < 1 or k >= self.KSND:
            raise RuntimeError("rf01_snd: z outside the sounding")
        k0 = k - 1
        b = (z - self.z_snd[k0]) / (self.z_snd[k0 + 1] - self.z_snd[k0])
        a = 1.0 - b
        T = a * self.T_snd[k0] + b * self.T_snd[k0 + 1]
        qv = (a * self.qv_snd[k0] + b * self.qv_snd[k0 + 1]) * 0.001
        ql = (a * self.ql_snd[k0] + b * self.ql_snd[k0 + 1]) * 0.001
        return u, v, T, qv, ql

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
                u[k], v[k], Tn[k], qvn[k], qln[k] = self.snd(float(zf[0, k]))
            zfn, zhn = compute_height(vg, GRAV * elev, torch.tensor(Tn[None], **f64), torch.tensor(qvn[None], **f64),
                                      pf, ph)
            maxerror = float(torch.max(torch.abs(zfn - zf)))
            zf, zh = zfn, zhn
        if i >= 10:
            raise RuntimeError("rf01_forc_init: initial condition did not converge")
        qa = np.where(qln > 0.0, 1.0, 0.0)
        self.u_geos = self.tensor(u[None])
        self.v_geos = self.tensor(v[None])
        tracers = np.stack([qvn, qln, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(Tn[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([self.psfc]), elev=self.tensor([self.zsfc]))

    # the GCSS RF01/RF02 longwave parameterisation ------------------------------------------------
    # variants: RF02 finds the inversion from qv + ql and
    # zeroes a warming level lying directly below a level with no radiative tendency
    QT_AS_MIXING_RATIO = True
    ZI_QT = 0.008
    CLIP_WARMING = False

    def radiation(self, T, qv, ql, pf, zf, dp):
        ncol, kdim = T.shape
        klwp = 85.0 * ql * dp / GRAV
        qt = qv / (1.0 - qv) + ql if self.QT_AS_MIXING_RATIO else qv + ql
        # inversion height: first level from the bottom (i = kdim .. 2) with qt < 0.008,
        # interpolated with the level below it
        zi = torch.zeros(ncol, dtype=T.dtype, device=T.device)
        found = torch.zeros(ncol, dtype=torch.bool, device=T.device)
        for i in range(kdim - 1, 0, -1):          # 0-based i = kdim-1 .. 1
            hit = (qt[:, i] < self.ZI_QT) & ~found
            if i + 1 < kdim:
                den = qt[:, i + 1] - qt[:, i]
                z1 = (zf[:, i] * (qt[:, i + 1] - self.ZI_QT) + zf[:, i + 1] * (self.ZI_QT - qt[:, i])) \
                    / torch.where(hit, den, torch.ones_like(den))
                zi = torch.where(hit, z1, zi)
            found = found | hit
        frad = []
        Qz1 = torch.zeros_like(klwp[:, 0])
        for k in range(kdim):
            Qz1 = Qz1 + klwp[:, k]
            Qz2 = torch.zeros_like(Qz1)
            for i in range(k + 1, kdim):
                Qz2 = Qz2 + klwp[:, i]
            dzi = zf[:, k] - zi
            pos = dzi > 0.0
            dz = torch.where(pos, dzi, torch.ones_like(dzi))
            term3 = CP_AIR * self.divf * (0.25 * dz ** (4.0 / 3.0) + zi * dz ** (1.0 / 3.0))
            term3 = term3 * pf[:, k] / (RDGAS * T[:, k])
            term3 = torch.where(pos, term3, torch.zeros_like(term3))   # dzi == 0 gives 0 as well
            frad.append(70.0 * torch.exp(-Qz1) + 22.0 * torch.exp(-Qz2) + term3)
        fr = torch.stack([frad[0]] + frad, dim=1)             # frad(1) = frad(2)
        dT = GRAV / CP_AIR * (fr[:, 1:] - fr[:, :-1]) / dp
        if self.CLIP_WARMING:
            cols = [dT[:, 0]]
            for k in range(1, kdim):
                c = dT[:, k]
                cols.append(torch.where((c > 0.0) & (cols[-1] == 0.0), torch.zeros_like(c), c))
            dT = torch.stack(cols, dim=1)
        return dT

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        qv, ql = tr[..., SPHUM], tr[..., LIQ_WAT]
        ncol, kdim = T.shape
        ps = torch.full((ncol,), self.psfc, dtype=T.dtype, device=T.device)
        pf = compute_pres_full(vg, ps)
        ph = compute_pres_half(vg, ps)
        zf, zh = compute_height(vg, GRAV * state["elev"], T, qv, pf, ph)
        zero = torch.zeros_like(T)
        omega_f = torch.where(pf > self.p_omega_zero, pf / (RDGAS * T) * GRAV * zf * self.divf, zero)
        dp = ph[:, 1:] - ph[:, :-1]
        du_geos = dv_geos = zero
        if self.do_geo:
            fcriolis = 2.0 * 2.0 * 3.1415926 / 86400.0 * math.sin(self.lat)
            du_geos = fcriolis * (v - self.v_geos)
            dv_geos = -fcriolis * (u - self.u_geos)
        dT_rad = self.radiation(T, qv, ql, pf, zf, dp) if self.do_rad else zero
        self.tdt_lw = dT_rad
        self.tdt_sw = zero
        dT_vadv = zero
        dtr = torch.zeros_like(tr)
        du_vadv = dv_vadv = zero
        dT_adi = zero
        if self.do_vadv:
            dT_adi = RDGAS * T * omega_f / CP_AIR / pf
            zhk = zh[:, 1:kdim]
            phk = ph[:, 1:kdim]
            om = torch.where(phk > self.p_omega_zero,
                             phk / (RDGAS * 0.5 * (T[:, 1:] + T[:, :-1])) * GRAV * zhk * self.divf,
                             torch.zeros_like(zhk))
            zc = torch.zeros_like(ps)[:, None]
            omega_h = torch.cat([zc, om, zc], dim=1)
            dT_vadv, dtr, du_vadv, dv_vadv = vadv_all(dt, omega_h, dp, T, tr, u, v, self.temp_scheme,
                                                      self.tracer_scheme, self.momentum_scheme)
        dT_vadv = dT_vadv + dT_adi
        AT = zero + dT_rad + dT_vadv
        AU = du_vadv + du_geos
        AV = dv_vadv + dv_geos
        return ForcingTendencies(ps=ps, T=AT, u=AU, v=AV, tracers=dtr, omega=omega_f,
                                 diag=dict(tdt_radf=dT_rad))

    def add_tdtlw(self, x):
        return x + self.tdt_lw if self.tdt_lw is not None else x

    def add_tdtsw(self, x):
        return x + self.tdt_sw if self.tdt_sw is not None else x

    def specified_surface_flux(self, sfc: dict) -> dict:
        from .bomex import BomexForcing
        return BomexForcing.specified_surface_flux(self, sfc)
