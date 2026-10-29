"""ASTEX Lagrangian 1 or 2, stratocumulus-to-cumulus transition, June 1992
(atmos_scm/forc/scm_astex.F90, siena_201207).

Bretherton's hourly composite soundings along the Lagrangian trajectory (lagr?_hNN),
blended above 100 hPa with the McClatchey mid-latitude summer/winter atmospheres;
divergence below the observed inversion tapering to zero at p_omega_zero; temperature
and humidity nudged to the soundings more than 150 g Pa above the inversion; winds
nudged everywhere; time-varying SST; interactive surface fluxes. Condensate above
p_cld_zero is evaporated in place every step.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLS, HLV, PI, RDGAS
from ...time_manager import get_date, set_date
from ..vert_grid import compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT, vadv
from .interp import interp_2d_field, interp_2d_field_t, interp_time, scalar

P00 = 100000.0
KAPPA_R = RDGAS / CP_AIR


def _read_stand(path, n=35):
    rows = []
    with open(path) as f:
        lines = f.read().splitlines()[4:4 + n]
    for line in lines:
        rows.append([float(x) for x in line.split()[:4]])
    a = np.array(rows)[::-1]                  # stored bottom-up: index 0 = surface
    return a[:, 0] * 100.0, a[:, 1], a[:, 2] / (1.0 + a[:, 2]), a[:, 3] / (1.0 + a[:, 3])


class AstexForcing(ScmForcing):
    experiment = "astex"
    NML = "scm_astex_nml"
    IVERT = 41
    ITIME = {1: 43, 2: 41}

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path, pdamp, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.lagrnum = int(b.get("lagrnum", b.get("LAGRNUM", 1)))
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 3))
        self.vert_advec_cond = bool(b.get("vert_advec_cond", True))
        self.p_omega_zero = float(b.get("p_omega_zero", 70000.0))
        self.p_cld_zero = float(b.get("p_cld_zero", 50000.0))
        self.wind_tau = float(b.get("wind_relaxation_tau", 3.0 * 3600.0))
        self.thermo_tau = float(b.get("thermo_relaxation_tau", 3.0 * 3600.0))
        self.pdamp = pdamp
        self._read(case_path)
        if bool(nml.get("scm_forc_nml", {}).get("do_specified_tskin", False)):
            self.surface.tskin = 300.0             # replaced by the sounding SST at init

    def _read(self, path):
        n, itime = self.IVERT, self.ITIME[self.lagrnum]
        self.p_stand, self.t_stand, self.q_stand = {}, {}, {}
        for idx, name in ((1, "mlw.dat"), (2, "mls.dat")):
            p, t, q, _ = _read_stand(os.path.join(path, name))
            self.p_stand[idx], self.t_stand[idx], self.q_stand[idx] = p, t, q
        pre = f"lagr{self.lagrnum}_h"
        T = np.zeros((n, itime)); Q = np.zeros_like(T); U = np.zeros_like(T); V = np.zeros_like(T)
        QL = np.zeros_like(T)
        plev = np.zeros(n)
        sst, ps, div, lwp = (np.zeros(itime) for _ in range(4))
        times = []
        for t in range(itime):
            with open(os.path.join(path, f"{pre}{t:02d}")) as f:
                lines = f.read().splitlines()
            h = lines[3]
            junday = float(h[1:9])
            sst[t] = float(h[26:33]); ps[t] = float(h[34:40]); div[t] = float(h[41:45])
            lwp[t] = 0.0
            for k in range(n):
                line = lines[6 + k]
                l = n - 1 - k
                plev[l] = float(int(line[1:5]))
                T[l, t] = float(line[5:14]); Q[l, t] = float(line[14:23]); U[l, t] = float(line[23:32])
                V[l, t] = float(line[32:40]); QL[l, t] = float(line[56:64])
                lwp[t] = lwp[t] + QL[l, t] * 1.0e-03 * 10 * 100.0 / GRAV
            hour = 24.0 * (junday - float(int(junday)))
            times.append(set_date(1992, 6, int(junday), int(hour), 0, 0))
        with open(os.path.join(path, f"lagr{self.lagrnum}_scalars")) as f:
            sl = f.read().splitlines()[5:5 + itime]
        self.pib = np.array([float(int(x[66:69])) * 100.0 for x in sl])
        Q, QL = Q / 1000.0, QL / 1000.0
        q = Q / (1.0 + Q + QL)
        ql = QL / (1.0 + Q + QL)
        self.q_forc = np.maximum(4.0e-06, q)
        self.ql_forc = ql
        self.plev = 100.0 * plev
        self.ps_forc = 100.0 * ps
        self.t_forc = T * ((self.plev[:, None] / 100000.0) ** KAPPA_R)
        self.u_forc, self.v_forc = U, V
        self.div_forc = div * 1.0e-06
        self.sst_forc, self.lwp_forc, self.times = sst, lwp, times

    # ------------------------------------------------------------------------------------
    def _with_ps(self, ps, field_l, field_m, first_l, first_m, top="last"):
        """plev_int / field_int of the update: ps, the sounding, and a p = 0 level."""
        pl = np.concatenate([[ps], self.plev, [0.0]])
        tl = field_l[-1] if top == "last" else 0.0
        tm = field_m[-1] if top == "last" else 0.0
        return pl, np.concatenate([[first_l], field_l, [tl]]), np.concatenate([[first_m], field_m, [tm]])

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        il, im, wl, wm = interp_time(time, self.times)
        ps = scalar(self.ps_forc[il], self.ps_forc[im], wl, wm)
        phalf = np.asarray(vg.peta, dtype=np.float64) + ps * np.asarray(vg.eta, dtype=np.float64)
        phalf[0] = 0.0
        pf = 0.5 * (phalf[1:] + phalf[:-1])
        n = self.IVERT
        js = 0
        while self.plev[n - 1] < self.p_stand[1][js]:
            js += 1
        months = get_date(time)[1]
        sw = 1.0 - (float(abs(months - 7))) / 6.0
        pst = self.p_stand[1][js:]

        def blended(forc, stand, first_scale):
            fl = np.concatenate([[forc[0, il] * first_scale], forc[:, il],
                                 (1.0 - sw) * stand[1][js:] + sw * stand[2][js:]])
            fm = np.concatenate([[forc[0, im] * first_scale], forc[:, im],
                                 (1.0 - sw) * stand[1][js:] + sw * stand[2][js:]])
            pl = np.concatenate([[ps], self.plev, pst])
            return interp_2d_field(pf, pl, fl, fm, wl, wm)

        T = blended(self.t_forc, self.t_stand, (ps / self.plev[0]) ** KAPPA_R)
        qv = blended(self.q_forc, self.q_stand, 1.0)
        pl, fl, fm = self._with_ps(ps, self.ql_forc[:, il], self.ql_forc[:, im], self.ql_forc[0, il],
                                   self.ql_forc[0, im], top="zero")
        ql = interp_2d_field(pf, pl, fl, fm, wl, wm)
        pl, fl, fm = self._with_ps(ps, self.u_forc[:, il], self.u_forc[:, im], self.u_forc[0, il], self.u_forc[0, im])
        u = interp_2d_field(pf, pl, fl, fm, wl, wm)
        pl, fl, fm = self._with_ps(ps, self.v_forc[:, il], self.v_forc[:, im], self.v_forc[0, il], self.v_forc[0, im])
        v = interp_2d_field(pf, pl, fl, fm, wl, wm)
        qa = np.where(ql > 0.0, 1.0, 0.0)
        if self.surface.tskin is not None:
            self.surface.tskin = scalar(self.sst_forc[il], self.sst_forc[im], wl, wm)
        lwpf = scalar(self.lwp_forc[il], self.lwp_forc[im], wl, wm)
        lwp = 0.0
        for k in range(kdim):
            lwp = lwp + ql[k] * (phalf[k + 1] - phalf[k]) / GRAV
        if lwp > 0.0 and lwpf > 0.0:
            ql = ql * lwpf / lwp
        tracers = np.stack([qv, ql, np.zeros(kdim), qa], axis=-1)[None]
        return dict(T=self.tensor(T[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([ps]), elev=self.tensor([0.0]))

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        ncol, kdim = T.shape
        il, im, wl, wm = interp_time(time, self.times)
        ps0 = scalar(self.ps_forc[il], self.ps_forc[im], wl, wm)
        ps = torch.full((ncol,), ps0, dtype=T.dtype, device=T.device)
        pfull = compute_pres_full(vg, ps)
        phalf = compute_pres_half(vg, ps)
        dp = phalf[:, 1:] - phalf[:, :-1]
        zero = torch.zeros_like(T)
        qv, ql, qi, qa = tr[..., SPHUM], tr[..., LIQ_WAT], tr[..., ICE_WAT], tr[..., CLD_AMT]
        hi = pfull < self.p_cld_zero
        T = torch.where(hi, T - HLV * ql / CP_AIR - HLS * qi / CP_AIR, T)
        qv = torch.where(hi, qv + ql + qi, qv)
        ql = torch.where(hi, zero, ql)
        qi = torch.where(hi, zero, qi)
        qa = torch.where(hi, zero, qa)
        tr = torch.stack([qv, ql, qi, qa], dim=-1)
        pinv = scalar(self.pib[il], self.pib[im], wl, wm)
        if pinv < self.p_omega_zero:
            raise RuntimeError("update_astex_forc: pinv_obs < p_omega_zero")
        div = scalar(self.div_forc[il], self.div_forc[im], wl, wm)
        p0 = self.p_omega_zero

        def omega(p):
            below = div * (ps0 - p)
            mid = div * (ps0 - pinv) * (p - p0) / (pinv - p0)
            return torch.where(p > pinv, below, torch.where(p > p0, mid, torch.zeros_like(p)))

        omega_half = omega(phalf)
        omega_full = omega(pfull)
        tau_t = max(self.thermo_tau, float(dt))
        tau_w = max(self.wind_tau, float(dt))
        nudge_zone = pfull <= (pinv - GRAV * 150.0)
        # temperature
        AT = RDGAS * T * omega_full / CP_AIR / pfull
        adv = vadv(dt, omega_half, dp, T, self.temp_scheme)
        pl, fl, fm = self._with_ps(ps0, self.t_forc[:, il], self.t_forc[:, im],
                                   self.t_forc[0, il] * ((ps0 / self.plev[0]) ** KAPPA_R),
                                   self.t_forc[0, im] * ((ps0 / self.plev[0]) ** KAPPA_R))
        t_obs = interp_2d_field_t(pfull, pl, fl, fm, wl, wm)
        nudg_t = torch.where(nudge_zone, (t_obs - T) / tau_t, zero)
        AT = torch.where(nudge_zone, zero, AT)
        adv = torch.where(nudge_zone, zero, adv)
        AT = AT + adv + nudg_t
        # water vapour
        adv = vadv(dt, omega_half, dp, qv, self.tracer_scheme)
        pl, fl, fm = self._with_ps(ps0, self.q_forc[:, il], self.q_forc[:, im], self.q_forc[0, il],
                                   self.q_forc[0, im])
        q_obs = interp_2d_field_t(pfull, pl, fl, fm, wl, wm)
        nudg_q = torch.where(nudge_zone, (q_obs - qv) / tau_t, zero)
        adv = torch.where(nudge_zone, zero, adv)
        AQ = adv + nudg_q
        # winds
        pl, fl, fm = self._with_ps(ps0, self.u_forc[:, il], self.u_forc[:, im], self.u_forc[0, il], self.u_forc[0, im])
        AU = (interp_2d_field_t(pfull, pl, fl, fm, wl, wm) - u) / tau_w
        pl, fl, fm = self._with_ps(ps0, self.v_forc[:, il], self.v_forc[:, im], self.v_forc[0, il], self.v_forc[0, im])
        AV = (interp_2d_field_t(pfull, pl, fl, fm, wl, wm) - v) / tau_w
        AL = AI = AA = zero
        if self.vert_advec_cond:
            AL = zero + vadv(dt, omega_half, dp, ql, self.tracer_scheme)
            AI = zero + vadv(dt, omega_half, dp, qi, self.tracer_scheme)
            AA = zero + vadv(dt, omega_half, dp, qa, self.tracer_scheme)
        if self.surface.tskin is not None:
            self.surface.tskin = scalar(self.sst_forc[il], self.sst_forc[im], wl, wm)
        return ForcingTendencies(ps=ps, T=AT, u=AU, v=AV, tracers=torch.stack([AQ, AL, AI, AA], dim=-1),
                                 omega=omega_full, diag=dict(nudg_t=nudg_t, nudg_q=nudg_q),
                                 state=dict(T=T, tracers=tr))
