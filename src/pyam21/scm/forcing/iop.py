"""ARM variational-analysis IOP forcing: ARM SGP summer 1997 and M-PACE period A
(atmos_scm/forc/scm_arm.F90, scm_mpacea.F90, siena_201207; one template).

3-hourly layer and surface fields (layer.dat, surface.dat) with missing values
repaired by rem_miss_var; FORCMETHOD 'D' (the analysis total advective tendencies)
or 'F' (horizontal advection + nudging); analysis omega; winds relaxed to the
analysis; condensate advected by omega; specified land surface: skin temperature from
the observed upward longwave, albedo from the SW fluxes, prescribed sensible and latent
heat fluxes enforced by iterating the surface layer (*_surface_flux_loop).

ARM blends the profiles above the analysis top with the McClatchey mid-latitude
summer/winter atmospheres; M-PACE A with the SHEBA arctic profile above pdamp and zeroes
the forcing above pdamp.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from ...constants import CP_AIR, GRAV, HLV, RDGAS, STEFAN, TFREEZE
from ...time_manager import get_date, set_date
from ..vert_grid import compute_pres_full, compute_pres_half
from .base import ForcingTendencies, ScmForcing, SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT, surface_flux_loop, vadv
from .interp import interp_2d_field, interp_2d_field_t, interp_time, scalar, tsec

P00 = 100000.0
KAPPA_R = RDGAS / CP_AIR


# ---- file readers ------------------------------------------------------------------------------
def _records(lines, pos, n):
    """n values of a (5e15.7) record starting at line pos -> (values, next line)."""
    vals = []
    while len(vals) < n:
        vals.extend(float(x) for x in lines[pos].split())
        pos += 1
    return np.array(vals[:n]), pos


def read_layer(path):
    lines = open(path).read().splitlines()
    itime = int(lines[1][6:9])
    ivert = int(lines[3][5:8])
    pos = 5
    plev, pos = _records(lines, pos, ivert)
    cal = {}
    for name in ("time", "year", "month", "day", "hour", "minute"):
        pos += 1
        cal[name], pos = _records(lines, pos, itime)
    pos += 1
    nvar = int(float(lines[pos].split()[0])); pos += 1
    fields = []
    for i in range(nvar):
        pos += 1                                   # variable name
        mat = np.zeros((ivert, itime))
        for j in range(ivert):
            mat[j], pos = _records(lines, pos, itime)
        fields.append(mat)
    return itime, ivert, plev, cal, fields


def read_surface(path, itime):
    lines = open(path).read().splitlines()
    nfields = int(lines[1][5:8])
    pos = 4
    fields = []
    for i in range(nfields):
        pos += 1
        v, pos = _records(lines, pos, itime)
        fields.append(v)
    return fields


def rem_miss_var(bad_val, times, data, adjlev=None):
    """scm_arm / scm_mpacea rem_miss_var: fill values <= bad_val by linear interpolation
    in time (nearest good neighbours) or, when those are closer, in the vertical.

    data: (nlev_total, ntime). The Fortran's vertical search is unguarded at the ends:
    it reads the neighbouring element of the column-major array (the previous / next
    time's top / bottom level); that is reproduced, and reads outside the array stop
    the search (as a good value would)."""
    data = np.array(data, dtype=np.float64)
    nall, ntime = data.shape
    nlev = nall if adjlev is None else adjlev
    flat = data.T.reshape(-1)                      # column-major storage
    ts = [tsec(t) for t in times]
    new = data.copy()

    def bad_at(j, it):                             # 1-based (j may be out of range)
        idx = (it - 1) * nall + (j - 1)
        if idx < 0 or idx >= flat.size:
            return False
        return flat[idx] <= bad_val

    for jlev in range(1, nlev + 1):
        for it in range(1, ntime + 1):
            if not data[jlev - 1, it - 1] <= bad_val:
                continue
            icur = it
            while data[jlev - 1, icur - 1] <= bad_val:
                icur -= 1
                if icur == 0:
                    break
            il = icur
            icur = it
            while data[jlev - 1, icur - 1] <= bad_val:
                icur += 1
                if icur == ntime + 1:
                    break
            im = icur
            if il == 0 and im <= ntime:
                il = im
            if im == ntime + 1 and il >= 1:
                im = il
            if im == ntime + 1 and il == 0:
                return new
            if im != il:
                wm = float(ts[it - 1] - ts[il - 1]) / float(ts[im - 1] - ts[il - 1])
                wl = 1.0 - wm
            else:
                wm, wl = 1.0, 0.0
            jl = jm = 0
            if nlev > 1:
                jcur = jlev
                while bad_at(jcur, it) and jcur >= 1:
                    jcur -= 1
                jl = jcur
                jcur = jlev
                while bad_at(jcur, it) and jcur <= nlev:
                    jcur += 1
                jm = jcur
                if jl == 0 and jm <= nlev:
                    jl = jm
                if jm == nlev + 1 and jl >= 1:
                    jm = jl
            dlev = abs(jm - jlev) + abs(jl - jlev)
            dtime = abs(im - it) + abs(il - it)
            if dtime <= dlev or nlev == 1 or dlev >= 2 * (nlev + 2):
                new[jlev - 1, it - 1] = wm * data[jlev - 1, im - 1] + wl * data[jlev - 1, il - 1]
            else:
                wm = float(abs(jl - jlev)) / float(dlev)
                wl = float(abs(jm - jlev)) / float(dlev)
                new[jlev - 1, it - 1] = wm * data[jm - 1, it - 1] + wl * data[jl - 1, it - 1]
    return new


def _rem1(bad_val, times, v):
    return rem_miss_var(bad_val, times, v[None, :], adjlev=1)[0]


# ---- forcing ------------------------------------------------------------------------------------
class IopForcing(ScmForcing):
    """Shared by ArmForcing / MpaceaForcing (class attributes select the variant)."""
    NML = ""
    ROUGH_MOM = 0.035
    ROUGH_HEAT = 0.035 / 7.4
    STANDARD = "mclatchey"          # or "arctic"
    ZERO_ABOVE_PDAMP = False
    ARM_ROUGH = False

    def __init__(self, nml, vgrid, lat_rad, lon_rad, case_path, pdamp, device="cpu", dtype=torch.float64):
        super().__init__(nml, vgrid, lat_rad, lon_rad, device, dtype)
        b = nml.get(self.NML, {})
        self.forcmethod = str(b.get("forcmethod", "D")).strip().upper()
        if self.forcmethod not in ("D", "F"):
            raise NotImplementedError(f"FORCMETHOD {self.forcmethod}")
        self.tracer_scheme = int(b.get("tracer_vert_advec_scheme", 3))
        self.temp_scheme = int(b.get("temp_vert_advec_scheme", 1))
        self.vert_advec_cond = bool(b.get("vert_advec_cond", True))
        self.do_qc_adv_loss = bool(b.get("do_qc_adv_loss", False))
        self.do_iop_mean_albedo = bool(b.get("do_iop_mean_albedo", False))
        self.use_given_omega_surface = bool(b.get("use_given_omega_surface", False))
        self.wind_tau = float(b.get("wind_relaxation_tau", 6.0 * 3600.0))
        self.relax_tau = float(b.get("relaxation_tau", 6.0 * 3600.0))
        rough_mom = float(b.get("rough_mom", self.ROUGH_MOM))
        rough_heat = float(b.get("rough_heat", self.ROUGH_HEAT))
        if self.ARM_ROUGH:
            rough_heat = rough_mom / 7.4
        self.pdamp = pdamp
        self._read(case_path)
        f = nml.get("scm_forc_nml", {})
        spec = self.surface
        if bool(f.get("do_specified_rough_leng", False)):
            spec.rough_mom, spec.rough_heat = rough_mom, rough_heat
        self.do_tskin = bool(f.get("do_specified_tskin", False))
        self.do_albedo = bool(f.get("do_specified_albedo", False))
        self.senflux = 10.0
        self.evapflux = 100.0 / HLV
        self.tskin = 300.0
        self.albedo_obs = 0.2

    # ------------------------------------------------------------------------------------
    def _read(self, path):
        itime, ivert, plev, cal, lf = read_layer(os.path.join(path, "layer.dat"))
        self.itime, self.ivert = itime, ivert
        sf = read_surface(os.path.join(path, "surface.dat"), itime)
        nfields = len(sf)
        times = [set_date(int(cal["year"][j]), int(cal["month"][j]), int(cal["day"][j]),
                          int(cal["hour"][j]), int(cal["minute"][j]), 0) for j in range(itime)]
        self.times = times
        if self.STANDARD == "arctic":
            rows = [[float(x) for x in l.split()[:4]]
                    for l in open(os.path.join(path, "arctic_profile.ascii")).read().splitlines()[4:64]]
            a = np.array(rows)
            q = a[:, 2] / 1000.0
            self.p_stand, self.t_stand, self.q_stand = 100.0 * a[:, 0], a[:, 1], q / (1.0 + q)
        else:
            from .astex import _read_stand
            self.p_stand, self.t_stand, self.q_stand = {}, {}, {}
            for idx, name in ((1, "mlw.dat"), (2, "mls.dat")):
                p, t, q, _ = _read_stand(os.path.join(path, name))
                self.p_stand[idx], self.t_stand[idx], self.q_stand[idx] = p, t, q
        t_forc, q_forc, u_forc, v_forc, omega = lf[0], lf[1], lf[2], lf[3], lf[4]
        tadv_hor, qadv_hor, qadv_tot, tadv_tot = lf[6], lf[8], lf[9], lf[12]
        # surface fields (1-based numbering of the Fortran)
        g = lambda i: sf[i - 1].copy()
        lat_flux, sen_flux, ps = g(8), g(9), g(10)
        tair = g(12)
        lwup = g(40) if nfields >= 43 else None
        swup = g(42) if nfields >= 43 else None
        swdn = g(43) if nfields >= 43 else None
        tskin_forc = g(44) if nfields >= 44 else None
        alb_24hr = g(45) if nfields >= 45 else None
        dps_dt_given = g(36) if nfields >= 36 else None
        # missing values
        dum = np.where(t_forc > 1.0, t_forc * (1000.0 / plev[:, None]) ** KAPPA_R, 0.0)
        dum = rem_miss_var(1.0, times, dum)
        t_forc = dum / (1000.0 / plev[:, None]) ** KAPPA_R
        q_forc = rem_miss_var(1.0e-06, times, q_forc)
        u_forc = rem_miss_var(-9998.0, times, u_forc)
        v_forc = rem_miss_var(-9998.0, times, v_forc)
        omega = rem_miss_var(-9998.0, times, omega)
        tadv_tot = rem_miss_var(-9998.0, times, tadv_tot)
        tadv_hor = rem_miss_var(-9998.0, times, tadv_hor)
        qadv_tot = rem_miss_var(-9998.0, times, qadv_tot)
        qadv_hor = rem_miss_var(-9998.0, times, qadv_hor)
        ps = _rem1(-9998.0, times, ps)
        sen_flux = _rem1(-9998.0, times, sen_flux)
        lat_flux = _rem1(-9998.0, times, lat_flux)
        tair = _rem1(-9998.0, times, tair)
        if tskin_forc is not None:
            tskin_forc = _rem1(-9998.0, times, tskin_forc)
        if alb_24hr is not None:
            alb_24hr = _rem1(-8887.0, times, alb_24hr)
        q_forc = q_forc / 1000.0
        q_forc = q_forc / (1.0 + q_forc)
        omega = 100.0 * omega / 3600.0
        plev = 100.0 * plev
        ps = 100.0 * ps
        tadv_hor = tadv_hor / 3600.0
        tadv_tot = (tadv_tot / 3600.0) + tadv_hor
        qadv_hor = qadv_hor / 3600.0 / 1000.0
        qadv_tot = (qadv_tot / 3600.0 / 1000.0) + qadv_hor
        qadv_hor = qadv_hor * (1.0 - q_forc) * (1.0 - q_forc)
        qadv_tot = qadv_tot * (1.0 - q_forc) * (1.0 - q_forc)
        tair = tair + TFREEZE
        if self.use_given_omega_surface and dps_dt_given is not None:
            dps_dt = dps_dt_given * 100.0 / 3600.0
        else:
            dps_dt = np.zeros(itime)
            dps_dt[0] = -9999.0
            for i in range(1, itime - 1):
                dps_dt[i] = (ps[i + 1] - ps[i - 1]) / float(tsec(times[i + 1]) - tsec(times[i - 1]))
            dps_dt[itime - 1] = -9999.0
            dps_dt = _rem1(-9998.0, times, dps_dt)
        self.tau_inv = 2.0 * np.sqrt(u_forc * u_forc + v_forc * v_forc) / (370.0 * 1000.0)
        if alb_24hr is not None:
            albedo = alb_24hr.copy()
            t1 = t2 = 0.0
            for i in range(itime):
                if swdn[i] > 50.0:
                    t1 += alb_24hr[i]; t2 += 1.0
            avg_albedo = t1 / t2
        elif swup is not None and swdn is not None:
            t1 = t2 = 0.0
            albedo = np.full(itime, -0.9)
            for i in range(itime):
                t1 += swup[i]; t2 += swdn[i]
                if swdn[i] > 50.0:
                    albedo[i] = swup[i] / swdn[i]
            avg_albedo = t1 / t2
            albedo = _rem1(-0.89, times, albedo)
        else:
            albedo = np.full(itime, 0.19)
            avg_albedo = 0.19
        if tskin_forc is not None:
            tskin_forc = tskin_forc + TFREEZE
        self.plev, self.ps_forc, self.dps_dt = plev, ps, dps_dt
        self.t_forc, self.q_forc, self.u_forc, self.v_forc = t_forc, q_forc, u_forc, v_forc
        self.omega_forc, self.tadv_tot, self.qadv_tot = omega, tadv_tot, qadv_tot
        self.sen_flux, self.lat_flux, self.tair = sen_flux, lat_flux, tair
        self.lwup, self.tskin_forc = lwup, tskin_forc
        self.albedo_sfc, self.avg_albedo = albedo, avg_albedo

    # ------------------------------------------------------------------------------------
    def _surface_update(self, time):
        il, im, wl, wm = interp_time(time, self.times)
        self.senflux = scalar(self.sen_flux[il], self.sen_flux[im], wl, wm)
        self.evapflux = scalar(self.lat_flux[il], self.lat_flux[im], wl, wm) / HLV
        if self.tskin_forc is not None:
            self.tskin = scalar(self.tskin_forc[il], self.tskin_forc[im], wl, wm)
        elif self.lwup is not None:
            self.tskin = math.sqrt(math.sqrt(scalar(self.lwup[il], self.lwup[im], wl, wm) / STEFAN))
        else:
            self.tskin = scalar(self.tair[il], self.tair[im], wl, wm)
        if self.do_iop_mean_albedo:
            self.albedo_obs = self.avg_albedo
        else:
            self.albedo_obs = scalar(self.albedo_sfc[il], self.albedo_sfc[im], wl, wm)
        if self.do_tskin:
            self.surface.tskin = self.tskin
        if self.do_albedo:
            self.surface.albedo_obs = self.albedo_obs

    def _jstart(self, ps):
        js = 0
        while ps < self.plev[js]:
            js += 1
        return js

    def _col(self, ps, forc, js, il, im, first_l=None, first_m=None):
        """plev_int / fields: ps (with the given first values, default the jstart level),
        the analysis from jstart up, and p = 0 carrying the top value."""
        fl0 = forc[js, il] if first_l is None else first_l
        fm0 = forc[js, im] if first_m is None else first_m
        pl = np.concatenate([[ps], self.plev[js:], [0.0]])
        fl = np.concatenate([[fl0], forc[js:, il], [forc[-1, il]]])
        fm = np.concatenate([[fm0], forc[js:, im], [forc[-1, im]]])
        return pl, fl, fm

    def initial_state(self, time) -> dict:
        vg = self.vgrid
        kdim = vg.nlev
        il, im, wl, wm = interp_time(time, self.times)
        ps = scalar(self.ps_forc[il], self.ps_forc[im], wl, wm)
        phalf = np.asarray(vg.peta, dtype=np.float64) + ps * np.asarray(vg.eta, dtype=np.float64)
        if self.STANDARD == "arctic":
            lph = np.log(np.maximum(phalf, np.finfo(np.float64).eps))
            pf = np.exp((phalf[1:] * lph[1:] - phalf[:-1] * lph[:-1]) / (phalf[1:] - phalf[:-1]) - 1.0)
        else:
            phalf[0] = 0.0
            pf = 0.5 * (phalf[1:] + phalf[:-1])
        js = self._jstart(ps)
        n = self.ivert
        if self.STANDARD == "arctic":
            je = n - 1
            while self.plev[je] < self.pdamp:
                je -= 1
            jss = 0
            while self.pdamp < self.p_stand[jss]:
                jss += 1
            ps_top, t_top, q_top = self.p_stand[jss:], self.t_stand[jss:], self.q_stand[jss:]
        else:
            je = n - 1
            jss = 0
            while self.plev[n - 1] < self.p_stand[1][jss]:
                jss += 1
            months = get_date(time)[1]
            sw = 1.0 - (float(abs(months - 7))) / 6.0
            ps_top = self.p_stand[1][jss:]
            t_top = (1.0 - sw) * self.t_stand[1][jss:] + sw * self.t_stand[2][jss:]
            q_top = (1.0 - sw) * self.q_stand[1][jss:] + sw * self.q_stand[2][jss:]

        def blend(forc, top, scale):
            pl = np.concatenate([[ps], self.plev[js:je + 1], ps_top])
            fl = np.concatenate([[forc[js, il] * scale], forc[js:je + 1, il], top])
            fm = np.concatenate([[forc[js, im] * scale], forc[js:je + 1, im], top])
            return interp_2d_field(pf, pl, fl, fm, wl, wm)

        T = blend(self.t_forc, t_top, (ps / self.plev[js]) ** KAPPA_R)
        qv = blend(self.q_forc, q_top, 1.0)
        u = interp_2d_field(pf, *self._col(ps, self.u_forc, js, il, im), wl, wm)
        v = interp_2d_field(pf, *self._col(ps, self.v_forc, js, il, im), wl, wm)
        self._surface_update(time)
        z = np.zeros(kdim)
        tracers = np.stack([qv, z, z, z], axis=-1)[None]
        return dict(T=self.tensor(T[None]), u=self.tensor(u[None]), v=self.tensor(v[None]),
                    tracers=self.tensor(tracers), ps=self.tensor([ps]), elev=self.tensor([0.0]))

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        if tsec(time) < tsec(self.times[0]) or tsec(time) > tsec(self.times[-1]):
            raise RuntimeError(f"{self.experiment}: time outside the forcing dataset")
        vg = self.vgrid
        T, u, v, tr = state["T"], state["u"], state["v"], state["tracers"]
        qv, ql, qi, qa = tr[..., SPHUM], tr[..., LIQ_WAT], tr[..., ICE_WAT], tr[..., CLD_AMT]
        ncol, kdim = T.shape
        il, im, wl, wm = interp_time(time, self.times)
        ps0 = scalar(self.ps_forc[il], self.ps_forc[im], wl, wm)
        omega_surface = scalar(self.dps_dt[il], self.dps_dt[im], wl, wm)
        ps = torch.full((ncol,), ps0, dtype=T.dtype, device=T.device)
        pfull = compute_pres_full(vg, ps)
        phalf = compute_pres_half(vg, ps)
        dp = phalf[:, 1:] - phalf[:, :-1]
        js = self._jstart(ps0)
        pd = self.pdamp
        zero = torch.zeros_like(T)

        def cut(pl, fl, fm):                      # where (plev_int < pdamp) field = 0
            m = pl < pd
            return pl, np.where(m, 0.0, fl), np.where(m, 0.0, fm)

        col = self._col(ps0, self.omega_forc, js, il, im, self.dps_dt[il], self.dps_dt[im])
        col = cut(*col)
        omega_full = interp_2d_field_t(pfull, *col, wl, wm)
        omega_half = interp_2d_field_t(phalf, *col, wl, wm)
        above = pfull < pd
        tau_t = max(self.relax_tau, float(dt))
        # temperature
        AT = interp_2d_field_t(pfull, *cut(*self._col(ps0, self.tadv_tot, js, il, im, 0.0, 0.0)), wl, wm)
        nudg_t = zero
        if self.forcmethod == "F":
            sc = (ps0 / self.plev[js]) ** KAPPA_R
            t_obs = interp_2d_field_t(pfull, *self._col(ps0, self.t_forc, js, il, im,
                                                        self.t_forc[js, il] * sc, self.t_forc[js, im] * sc), wl, wm)
            nudg_t = (t_obs - T) / tau_t
            AT = AT + nudg_t
        if self.ZERO_ABOVE_PDAMP:
            AT = torch.where(above, zero, AT)
        # water vapour
        AQ = interp_2d_field_t(pfull, *cut(*self._col(ps0, self.qadv_tot, js, il, im, 0.0, 0.0)), wl, wm)
        if self.forcmethod == "F":
            q_obs = interp_2d_field_t(pfull, *self._col(ps0, self.q_forc, js, il, im), wl, wm)
            AQ = q_obs + (q_obs - qv) / tau_t    # (sic: AQ = tmp_ans + nudg_q)
        if self.ZERO_ABOVE_PDAMP:
            AQ = torch.where(above, zero, AQ)
        # winds
        tau_w = max(self.wind_tau, float(dt))
        AU = (interp_2d_field_t(pfull, *self._col(ps0, self.u_forc, js, il, im), wl, wm) - u) / tau_w
        AV = (interp_2d_field_t(pfull, *self._col(ps0, self.v_forc, js, il, im), wl, wm) - v) / tau_w
        if self.ZERO_ABOVE_PDAMP:
            AU = torch.where(above, zero, AU)
            AV = torch.where(above, zero, AV)
        AL = AI = AA = zero
        if self.vert_advec_cond:
            AL = zero + vadv(dt, omega_half, dp, ql, self.tracer_scheme)
            AI = zero + vadv(dt, omega_half, dp, qi, self.tracer_scheme)
            AA = zero + vadv(dt, omega_half, dp, qa, self.tracer_scheme)
        if self.do_qc_adv_loss:
            ti = interp_2d_field_t(pfull, *cut(*self._col(ps0, self.tau_inv, js, il, im)), wl, wm)
            AL, AI, AA = AL - ql * ti, AI - qi * ti, AA - qa * ti
        if self.ZERO_ABOVE_PDAMP:
            AL = torch.where(above, zero, AL)
            AI = torch.where(above, zero, AI)
            AA = torch.where(above, zero, AA)
        self._surface_update(time)
        return ForcingTendencies(ps=ps, T=AT, u=AU, v=AV, tracers=torch.stack([AQ, AL, AI, AA], dim=-1),
                                 omega=omega_full, diag=dict(nudg_t=nudg_t, omega_surface=omega_surface))

    def specified_surface_flux(self, sfc: dict) -> dict:
        out = surface_flux_loop(sfc["surface_flux"], sfc, self.senflux, self.evapflux, land=True)
        zero = torch.zeros_like(sfc["t_atm"])
        out.update(dhdt_surf=zero, dedt_surf=zero, dedq_surf=zero, dhdt_atm=zero, dedq_atm=zero)
        return out


class ArmForcing(IopForcing):
    experiment = "arm"
    NML = "scm_arm_nml"
    ROUGH_MOM, ROUGH_HEAT = 0.035, 0.035 / 7.4
    STANDARD = "mclatchey"
    ZERO_ABOVE_PDAMP = False
    ARM_ROUGH = True


class MpaceaForcing(IopForcing):
    experiment = "mpacea"
    NML = "scm_mpacea_nml"
    ROUGH_MOM, ROUGH_HEAT = 0.001, 0.001
    STANDARD = "arctic"
    ZERO_ABOVE_PDAMP = True
