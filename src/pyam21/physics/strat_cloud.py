"""Tiedtke (1993) prognostic stratiform clouds with Rotstayn (1997) / Klein microphysics.

Translation of atmos_param/strat_cloud/strat_cloud_legacy.F90 (siena_201207), the code
path AM2.1 takes: no PDF clouds, no prognostic droplet number, no dust Bergeron,
Khairoutdinov-Kogan autoconversion off, cloud generator off, overlap = 2 (random
overlap of precipitation with cloud), super_choice supersaturation removal into new
cloud, tracer_advec (omega not added to the ice fall speed).

The level loop runs top-down (as in the Fortran) and carries the clear-sky and in-cloud
rain and snow fluxes; every column is handled at once. Diagnostics are not kept, apart
from the surface precipitation and the 3-D precipitation fluxes.

Inputs are the stratiform tracers after convection (ql, qi, qa), with qrat and ahuco
the convective humidity ratio / area (1 and 0 in AM2.1). Outputs are increments over
the step (ST, SQ, SL, SI, SA) and surface rain/snow in kg/m2 per step, as in the
Fortran.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, DENS_H2O, GRAV, HLF, HLS, HLV, RDGAS, RVGAS, TFREEZE
from ..sat_vapor_pres import get_sat_vapor_pres
from ..core import spow, ssqrt

D608 = (RVGAS - RDGAS) / RDGAS
RHO_ICE = 100.0
ELI = 0.7


@dataclass(frozen=True)
class StratCloudConfig:
    U00: float = 0.80
    u00_profile: bool = False
    rthresh: float = 10.0
    use_kk_auto: bool = False
    U_evap: float = 1.0
    eros_scale: float = 1.0e-06
    eros_choice: bool = False
    eros_scale_c: float = 8.0e-06
    eros_scale_t: float = 5.0e-05
    mc_thresh: float = 0.001
    diff_thresh: float = 1.0
    super_choice: bool = False
    tracer_advec: bool = False
    qmin: float = 1.0e-10
    Dmin: float = 1.0e-08
    efact: float = 0.0
    vfact: float = 1.0
    cfact: float = 1.0
    do_old_snowmelt: bool = False
    iwc_crit: float = 0.0
    vfall_const2: float = 3.29
    vfall_exp2: float = 0.16
    N_land: float = 250.0e6
    N_ocean: float = 100.0e6
    do_pdf_clouds: bool = False
    do_liq_num: bool = False
    do_dust_berg: bool = False
    overlap: int = 2

    @classmethod
    def from_nml(cls, nml: dict) -> "StratCloudConfig":
        g = {k.lower(): v for k, v in nml.get("strat_cloud_nml", {}).items()}
        kw = {}
        for f in cls.__dataclass_fields__:
            if f == "overlap":
                continue
            if f.lower() in g:
                kw[f] = type(getattr(cls, f))(g[f.lower()])
        ov = nml.get("cloud_rad_nml", {}).get("overlap", 2)
        return cls(overlap=int(ov), **kw)


class StratCloud:
    def __init__(self, config: StratCloudConfig):
        c = config
        if c.do_pdf_clouds or c.do_liq_num or c.do_dust_berg or c.use_kk_auto:
            raise NotImplementedError("strat_cloud options beyond the AM2.1 path")
        if c.overlap not in (1, 2):
            raise ValueError("overlap must be 1 or 2")
        self.c = config

    def _xfer(self, qa_mean, qa_mean_lst, a_clr, a_cld, clr, cld):
        """cloud_clear_xfer (cloud generator off)."""
        qmin = self.c.qmin
        z = torch.zeros_like(qa_mean)
        if self.c.overlap == 1:
            cld2clr = torch.minimum(a_cld, torch.clamp(a_cld - qa_mean, min=0.0))
            clr2cld = torch.minimum(torch.clamp(qa_mean - qa_mean_lst, min=0.0), a_clr)
        else:
            cld2clr = torch.minimum(a_cld, torch.clamp(a_cld * (1.0 - qa_mean), min=0.0))
            clr2cld = torch.minimum(torch.clamp(a_clr * qa_mean, min=0.0), a_clr)
        prec_cld2clr = cld * (cld2clr / torch.clamp(a_cld, min=qmin))
        prec_clr2cld = clr * (clr2cld / torch.clamp(a_clr, min=qmin))
        a_clr = a_clr + cld2clr - clr2cld
        a_cld = a_cld - cld2clr + clr2cld
        clr = clr + prec_cld2clr - prec_clr2cld
        cld = cld - prec_cld2clr + prec_clr2cld
        return a_clr, a_cld, clr, cld

    @staticmethod
    def _analytic(C, D, q0, Dmin, dmin_mode):
        """Analytic solution of dq/dt = C - D q over the step (qceqs, qc1s, qcbars)."""
        big = D > Dmin
        Ds = torch.where(big, D, torch.ones_like(D))
        qceqs = C / Ds
        qc1s = qceqs - (qceqs - q0) * torch.exp(-1.0 * Ds)
        qcbars = qceqs - ((qc1s - q0) / Ds)
        return (torch.where(big, qc1s, q0 + C), torch.where(big, qcbars, q0 + 0.5 * C))

    def __call__(self, dtcloud, pfull, phalf, radturbten2, T, qv, ql, qi, qa, omega, Mc, diff_t, LAND,
                 qrat=None, ahuco=None):
        """All 3-D inputs (ncol, nlev) [phalf (ncol, nlev+1)]; LAND (ncol,)."""
        c = self.c
        qmin, Dmin = c.qmin, c.Dmin
        ncol, kdim = T.shape
        svp = get_sat_vapor_pres()
        if qrat is None:
            qrat = torch.ones_like(T)
        if ahuco is None:
            ahuco = torch.zeros_like(T)
        inv_dtcloud = 1.0 / dtcloud
        qs, dqsdT, esat = svp.compute_qs(T, pfull, want_dqsdT=True, want_esat=True)
        A_plus_B = ((HLV / 0.024 / T) * ((HLV / RVGAS / T) - 1.0)) + (RVGAS * T * pfull / 2.21 / esat)
        gamma = dqsdT * (torch.clamp(0.05 * (T - TFREEZE + 20.0), 0.0, 1.0) * HLV
                         + torch.clamp(0.05 * (TFREEZE - T), 0.0, 1.0) * HLS) / CP_AIR
        qpos = qrat > 0.0
        airdens = torch.where(qpos, pfull / (RDGAS * T * (1.0 + (D608 * qv / torch.where(qpos, qrat, torch.ones_like(qrat)))
                                                            - ql - qi)),
                              pfull / (RDGAS * T * (1.0 - ql - qi)))
        N = c.N_land * LAND + c.N_ocean * (1.0 - LAND)
        z = torch.zeros(ncol, dtype=T.dtype, device=T.device)
        rain_cld = rain_clr = a_rain_cld = a_rain_clr = z
        snow_cld = snow_clr = a_snow_cld = a_snow_clr = z
        qa_mean_lst = z
        ps = phalf[:, kdim]
        ST, SQ, SL, SI, SA = ([None] * kdim for _ in range(5))
        rain3d = [z] * (kdim + 1)
        snow3d = [z] * (kdim + 1)
        snowclr3d = [z] * (kdim + 1)
        f_snow_berg = [None] * kdim
        W = torch.where
        # ---- level-local part (no dependence on precipitation from above): all levels at once
        Z = torch.zeros_like(T)
        st2 = sq2 = sl2 = si2 = sa2 = Z
        deltpg2 = (phalf[:, 1:] - phalf[:, :-1]) / GRAV
        qp = qrat > 0.0
        U2 = W(qp, torch.clamp(qv / (W(qp, qrat, torch.ones_like(qrat)) * qs), 0.0, 1.0), Z)
        qa_upd2, qi_upd2, ql_upd2 = qa, qi, ql
        m = qa <= qmin
        sa2 = W(m, sa2 - qa, sa2)
        qa_upd2 = W(m, Z, qa_upd2)
        m = qa_upd2 > U2
        sa2 = W(m, sa2 + U2 - qa_upd2, sa2)
        qa_upd2 = W(m, U2, qa_upd2)
        m = (ql <= qmin) | (qa <= qmin)
        sl2 = W(m, sl2 - ql, sl2)
        sq2 = W(m, sq2 + ql, sq2)
        st2 = W(m, st2 - HLV * ql / CP_AIR, st2)
        ql_upd2 = W(m, Z, ql_upd2)
        m = (qi <= qmin) | (qa <= qmin)
        si2 = W(m, si2 - qi, si2)
        sq2 = W(m, sq2 + qi, sq2)
        st2 = W(m, st2 - HLS * qi / CP_AIR, st2)
        qi_upd2 = W(m, Z, qi_upd2)
        # large-scale saturation change and cloud formation
        dqs_ls = (((omega + GRAV * Mc) / airdens / CP_AIR) + radturbten2) * dtcloud * dqsdT
        U00p = torch.full_like(Z, c.U00)
        if c.u00_profile:
            psc = ps[:, None]
            mp = pfull > 0.8 * psc
            U00p = W(mp, c.U00 + (1.0 - c.U00) * (((pfull - (0.8 * psc)) / (0.2 * psc)) ** 2.0), U00p)
        U00p = U00p + (1.0 - U00p) * ahuco
        omU = torch.clamp(1.0 - U2, min=qmin)
        m = (dqs_ls <= 0.0) & (U2 >= U00p) & (qa_upd2 < 1.0)
        ar = (1.0 + qa_upd2 * gamma) ** 2.0 - (1.0 - qa_upd2) * (1.0 - qa_upd2) * gamma * dqs_ls / qs / omU
        t1 = torch.sqrt(W(m, ar, torch.ones_like(ar))) - (1.0 + qa_upd2 * gamma)
        den = (1.0 - qa_upd2) * (1.0 - qa_upd2) * gamma / qs / omU / 2.0
        t1 = -1.0 * t1 / W(m, den, torch.ones_like(den))
        dqs_ls = W(m, torch.minimum(t1, dqs_ls / (1.0 + 0.5 * (1.0 + qa_upd2) * gamma)),
                   dqs_ls / (1.0 + qa_upd2 * gamma))
        m = (dqs_ls <= 0.0) & (U2 >= U00p) & (qa_upd2 + ahuco <= 1.0)
        da_ls = W(m, -0.5 * (1.0 - qa_upd2 - ahuco) * (1.0 - qa_upd2 - ahuco) * dqs_ls / qs / omU, Z)
        # erosion
        tmp2s = torch.full_like(Z, c.eros_scale)
        if c.eros_choice:
            tmp2s = W(Mc > c.mc_thresh, torch.full_like(Z, c.eros_scale_c), tmp2s)
            diff_next = torch.cat([diff_t[:, 1:kdim], diff_t[:, kdim - 1:kdim]], dim=1)
            tmp2s = W((diff_t[:, :kdim] > c.diff_thresh) | (diff_next > c.diff_thresh),
                      torch.full_like(Z, c.eros_scale_t), tmp2s)
        m = (ql_upd2 > qmin) | (qi_upd2 > qmin)
        sw = W(m, qi_upd2 + ql_upd2, torch.ones_like(Z))
        D_eros2 = qa_upd2 * tmp2s * dtcloud * qs * (1.0 - U2) / sw
        pbot = pfull[:, kdim - 1:kdim]
        D_eros2 = W(pfull > 400.0e2, D_eros2 + c.efact * D_eros2 * ((pbot - pfull) / (pbot - 400.0e2)),
                    D_eros2 + c.efact * D_eros2)
        D_eros2 = W(m, D_eros2, Z)
        # cloud fraction
        C_dts = da_ls / torch.clamp(1.0 - qa_upd2, min=qmin)
        D_dts = D_eros2
        qc0s = qa_upd2
        act = (C_dts > Dmin) | (D_dts > Dmin)
        CD = W(act, C_dts + D_dts, torch.ones_like(Z))
        qceqs = C_dts / CD
        qc1s = qceqs - (qceqs - qc0s) * torch.exp(-1.0 * CD)
        qcbars = qceqs - ((qc1s - qc0s) / CD)
        qc1s = W(act, qc1s, qc0s)
        qcbars = W(act, qcbars, qc0s)
        sa2 = sa2 + qc1s - qc0s
        qa_upd2 = qc1s
        dcond2 = -1.0 * qcbars * dqs_ls
        # phase of condensation / evaporation
        tmp1s = torch.ones_like(Z)
        tmp1s = W((dcond2 >= 0.0) & (T < TFREEZE - 40.0), Z, tmp1s)
        neg = dcond2 < 0.0
        frac_l = torch.minimum(-1.0 * dcond2, ql_upd2) / torch.clamp(-1.0 * dcond2, min=qmin)
        tmp1s = W(neg & (qi_upd2 > qmin), W(ql_upd2 > qmin, frac_l, Z), tmp1s)
        dcond_ice2 = (1.0 - tmp1s) * dcond2
        dcond2 = tmp1s * dcond2
        qa_mean2 = qa_upd2
        ql_mean2 = ql_upd2 + torch.clamp(dcond2, min=0.0)
        qi_mean2 = qi_upd2 + torch.clamp(dcond_ice2, min=0.0)
        qa_prev2 = torch.cat([torch.zeros_like(Z[:, :1]), qa_mean2[:, :-1]], dim=1)
        for j in range(kdim):
            Tj, qvj, qlj, qij, qaj = T[:, j], qv[:, j], ql[:, j], qi[:, j], qa[:, j]
            qsj, gamj, adj, ABj = qs[:, j], gamma[:, j], airdens[:, j], A_plus_B[:, j]
            pfj = pfull[:, j]
            deltpg = deltpg2[:, j]
            st, sq, sl, si, sa = st2[:, j], sq2[:, j], sl2[:, j], si2[:, j], sa2[:, j]
            U = U2[:, j]
            qa_upd, qi_upd, ql_upd = qa_upd2[:, j], qi_upd2[:, j], ql_upd2[:, j]
            D_eros = D_eros2[:, j]
            dcond_ls, dcond_ls_ice = dcond2[:, j], dcond_ice2[:, j]
            qa_mean, ql_mean, qi_mean = qa_mean2[:, j], ql_mean2[:, j], qi_mean2[:, j]
            qa_mean_lst = qa_prev2[:, j]
            # ---- precipitation transfer between clear and cloudy parts
            a_rain_clr, a_rain_cld, rain_clr, rain_cld = self._xfer(qa_mean, qa_mean_lst, a_rain_clr, a_rain_cld,
                                                                    rain_clr, rain_cld)
            a_snow_clr, a_snow_cld, snow_clr, snow_cld = self._xfer(qa_mean, qa_mean_lst, a_snow_clr, a_snow_cld,
                                                                    snow_clr, snow_cld)
            # ---- snow melt
            snow_fact = 2.0 if c.do_old_snowmelt else 0.0
            t1 = CP_AIR * (Tj - TFREEZE - snow_fact) * deltpg * inv_dtcloud / HLF
            t2 = torch.clamp(torch.minimum(snow_clr, t1), min=0.0)
            st = st - HLF * t2 * dtcloud / deltpg / CP_AIR
            rain_clr = rain_clr + t2
            a_rain_clr = W((t2 > 0.0) & (a_snow_clr > qmin), torch.maximum(a_rain_clr, a_snow_clr), a_rain_clr)
            m = (snow_clr < t1) & (a_snow_clr > qmin)
            snow_clr, a_snow_clr = W(m, z, snow_clr - t2), W(m, z, a_snow_clr)
            if not c.do_old_snowmelt:
                t1 = t1 - t2
                t2 = torch.clamp(torch.minimum(snow_cld, t1), min=0.0)
                st = st - HLF * t2 * dtcloud / deltpg / CP_AIR
                rain_cld = rain_cld + t2
                a_rain_cld = W((t2 > 0.0) & (a_snow_cld > qmin), torch.maximum(a_rain_cld, a_snow_cld), a_rain_cld)
                m = (snow_cld < t1) & (a_snow_cld > qmin)
                snow_cld, a_snow_cld = W(m, z, snow_cld - t2), W(m, z, a_snow_cld)
            qi_mean = qi_mean + snow_cld * dtcloud / deltpg
            lamda_f = 1.6 * 10.0 ** (3.0 + 0.023 * (TFREEZE - Tj))
            # ---- liquid sinks: accretion, autoconversion, Bergeron, riming, freezing
            qam = torch.clamp(qa_mean, min=qmin)
            rad_liq = 620350.49 * spow(adj * ql_mean / qam / N / DENS_H2O, 1.0 / 3.0)
            rad_liq = W(qa_mean <= qmin, z, rad_liq)
            D1_dt = (dtcloud * 65.772565 * (a_rain_cld / qam) * (rad_liq * rad_liq / (rad_liq * rad_liq + 20.5))
                     * spow(rain_cld / torch.clamp(a_rain_cld, min=qmin) / DENS_H2O, 7.0 / 9.0))
            tmp1 = 32681.0 * dtcloud * ((N * DENS_H2O) ** (-1.0 / 3.0)) * spow(ql_mean / qam, 4.0 / 3.0)
            tmp2 = torch.clamp(3 * torch.log(torch.clamp(rad_liq, min=qmin) / c.rthresh), min=0.0)
            tmp1 = torch.minimum(tmp1, tmp2)
            D1_dt = D1_dt + tmp1
            a_rain_cld = W(tmp1 > Dmin, qa_mean, a_rain_cld)
            cold = (Tj < TFREEZE) & (ql_mean > qmin) & (qa_mean > qmin)
            qas = W(cold, qa_mean, torch.ones_like(z))
            qls = W(cold, ql_mean, torch.ones_like(z))
            ex = c.cfact * 1000.0 * torch.exp((12.96 * 0.0125 * (TFREEZE - Tj)) - 0.639)
            D2b = (dtcloud * qas * ((ex / adj) ** (2.0 / 3.0)) * 7.8
                   * spow(torch.maximum(qi_mean / qas, 1.0e-12 * ex / adj), 1.0 / 3.0) * 0.0125
                   * (TFREEZE - Tj) / ((700.0 ** (1.0 / 3.0)) * ABj * qls))
            D2_dt = W(cold, D2b, z)
            sum_berg = D2_dt
            mr = (a_snow_cld > qmin) & (ql_mean > qmin) & (qa_mean > qmin)
            if c.do_old_snowmelt:
                mr = mr & (Tj < TFREEZE)
            tmp1 = W(mr, dtcloud * 0.5 * ELI * lamda_f * snow_cld / W(mr, qa_mean, torch.ones_like(z)) / RHO_ICE, z)
            D2_dt = D2_dt + tmp1
            sum_rime = tmp1
            mf = (Tj < TFREEZE - 40.0) & (ql_mean > qmin) & (qa_mean > qmin)
            D2_dt = W(mf, torch.log(W(mf, ql_mean, torch.full_like(z, qmin)) / qmin), D2_dt)
            sum_freeze = W(mf, D2_dt, z)
            sum_rime = W(mf, z, sum_rime)
            sum_berg = W(mf, z, sum_berg)
            qldt_sum = sum_berg + sum_freeze + sum_rime
            f_snow_berg[j] = W(qldt_sum > 0.0, sum_berg / W(qldt_sum > 0.0, qldt_sum, torch.ones_like(z)), z)
            # ---- liquid budget
            qlm = torch.clamp(ql_mean, min=qmin)
            evl = torch.clamp(-1.0 * dcond_ls, min=0.0) / qlm
            C_dts = torch.clamp(dcond_ls, min=0.0)
            D_dts = D1_dt + D2_dt + D_eros + evl
            qc0s = ql_upd
            qc1s, qcbars = self._analytic(C_dts, D_dts, qc0s, Dmin, True)
            sl = sl + qc1s - qc0s
            ql_upd = qc1s
            tmp2 = D_dts * qcbars / torch.clamp(D_dts, min=Dmin)
            st = st + (HLV * torch.clamp(dcond_ls, min=0.0) / CP_AIR) - (HLV * evl * tmp2 / CP_AIR)
            sq = sq - torch.clamp(dcond_ls, min=0.0) + evl * tmp2
            st = st + (HLF * D2_dt - HLV * D_eros) * tmp2 / CP_AIR
            sq = sq + D_eros * tmp2
            rain_cld = rain_cld + D1_dt * tmp2 * deltpg * inv_dtcloud
            tmp3 = tmp2 * D2_dt
            qi_mean = qi_mean + tmp3
            # ---- ice: fall, melting, budget
            iwc = adj * qi_mean / qam
            Vfall = W(iwc >= c.iwc_crit, c.vfact * 3.29 * spow(iwc, 0.16),
                      c.vfact * c.vfall_const2 * spow(iwc, c.vfall_exp2))
            tmp3 = tmp3 + snow_cld * dtcloud / deltpg
            tmp1 = z if c.tracer_advec else omega[:, j]
            mi = (qi_mean > qmin) & (qa_mean > qmin)
            D1_dt = W(mi, torch.clamp(((adj * Vfall) + (tmp1 / GRAV)) * dtcloud / deltpg, min=0.0), z)
            a_snow_cld = W(mi, qa_mean, z)
            snow_cld = W(mi, snow_cld, z)
            tmp1 = CP_AIR * (Tj - TFREEZE) / HLF
            tmp2 = torch.clamp(torch.minimum(qi_mean, tmp1), min=0.0)
            D2_dt = torch.clamp(torch.log(torch.clamp(qi_mean, min=qmin) / torch.clamp(qi_mean - tmp2, min=qmin)),
                                min=0.0)
            a_rain_cld = W(D2_dt > Dmin, qa_mean, a_rain_cld)
            m = (qi_mean < tmp1) & (qi_mean > qmin)
            D1_dt = W(m, z, D1_dt)
            snow_cld = W(m, z, snow_cld)
            a_snow_cld = W(m, z, a_snow_cld)
            qim = torch.clamp(qi_mean, min=qmin)
            evi = torch.clamp(-1.0 * dcond_ls_ice, min=0.0) / qim
            C_dts = tmp3 + torch.clamp(dcond_ls_ice, min=0.0)
            D_dts = D1_dt + D2_dt + D_eros + evi
            qc0s = qi_upd
            qc1s, qcbars = self._analytic(C_dts, D_dts, qc0s, Dmin, True)
            si = si + qc1s - qc0s
            qi_upd = qc1s
            tmp2s = D_dts * qcbars / torch.clamp(D_dts, min=Dmin)
            st = st + HLS * torch.clamp(dcond_ls_ice, min=0.0) / CP_AIR - HLS * evi * tmp2s / CP_AIR
            sq = sq - torch.clamp(dcond_ls_ice, min=0.0) + evi * tmp2s
            st = st - HLS * D_eros * tmp2s / CP_AIR
            sq = sq + D_eros * tmp2s
            snow_cld = D1_dt * tmp2s * deltpg * inv_dtcloud
            st = st - HLF * D2_dt * tmp2s / CP_AIR
            rain_cld = rain_cld + D2_dt * tmp2s * deltpg * inv_dtcloud
            # ---- evaporation of clear-sky rain and sublimation of clear-sky snow
            U_clr = torch.clamp((U - qa_mean) / torch.clamp(1.0 - qa_mean, min=qmin), 0.0, 1.0)
            arc = torch.clamp(a_rain_clr, min=qmin)
            t1 = (56788.636 * dtcloud * spow(rain_clr / arc / DENS_H2O, 11.0 / 18.0) / torch.sqrt(adj) / ABj / qsj)
            t1 = t1 * qsj * (1.0 - U_clr) / (1.0 + 0.5 * t1 * (1.0 + gamj))
            lim = ((1.0 - qa_mean) / arc) * qsj * torch.clamp(c.U_evap - U_clr, min=0.0) / (
                1.0 + (c.U_evap * (1.0 - qa_mean) + qa_mean) * gamj)
            t1 = torch.minimum(t1, lim)
            t1 = t1 * a_rain_clr * deltpg * inv_dtcloud
            t2 = torch.clamp(torch.minimum(rain_clr, t1), min=0.0)
            sq = sq + t2 * dtcloud / deltpg
            st = st - HLV * t2 * dtcloud / deltpg / CP_AIR
            m = (t1 > rain_clr) & (a_rain_clr > qmin)
            rain_clr, a_rain_clr = W(m, z, rain_clr - t2), W(m, z, a_rain_clr)
            asc = torch.clamp(a_snow_clr, min=qmin)
            t1 = dtcloud * (4.0 / 3.14159 / RHO_ICE / adj / ABj / qsj) * spow(snow_clr / asc / 3.29, 1.0 / 1.16) * (
                0.65 * lamda_f * lamda_f
                + 198.92227 * lamda_f * torch.sqrt(adj * lamda_f) * spow(snow_clr / asc, 1.0 / 14.5))
            t1 = t1 * qsj * (1.0 - U_clr) / (1.0 + 0.5 * t1 * (1.0 + gamj))
            lim = ((1.0 - qa_mean) / asc) * qsj * torch.clamp(c.U_evap - U_clr, min=0.0) / (
                1.0 + (c.U_evap * (1.0 - qa_mean) + qa_mean) * gamj)
            t1 = torch.minimum(t1, lim)
            t1 = t1 * a_snow_clr * deltpg * inv_dtcloud
            t2 = torch.clamp(torch.minimum(snow_clr, t1), min=0.0)
            sq = sq + t2 * dtcloud / deltpg
            st = st - HLS * t2 * dtcloud / deltpg / CP_AIR
            m = (t1 > snow_clr) & (a_snow_clr > qmin)
            snow_clr, a_snow_clr = W(m, z, snow_clr - t2), W(m, z, a_snow_clr)
            # ---- no negative vapour
            t1 = qvj + sq
            neg = t1 < 0.0
            vcold = Tj <= TFREEZE - 40.0
            t2a = torch.minimum(-t1, ql_upd)
            t3a = torch.minimum(-t1 - t2a, qi_upd)
            t3b = torch.minimum(-t1, qi_upd)
            t2b = torch.minimum(-t1 - t3b, ql_upd)
            t2 = W(neg, W(vcold, t2a, t2b), z)
            t3 = W(neg, W(vcold, t3a, t3b), z)
            ql_upd = W(neg, ql_upd - t2, ql_upd)
            qi_upd = W(neg, qi_upd - t3, qi_upd)
            sl = W(neg, sl - t2, sl)
            si = W(neg, si - t3, si)
            sq = W(neg, sq + t2 + t3, sq)
            st = W(neg, st - HLV * t2 / CP_AIR - HLS * t3 / CP_AIR, st)
            # ---- remove tiny clouds
            m = ((ql_upd <= qmin) & (qi_upd <= qmin)) | (qa_upd <= qmin)
            sl = W(m, sl - ql_upd, sl)
            si = W(m, si - qi_upd, si)
            sq = W(m, sq + ql_upd + qi_upd, sq)
            st = W(m, st - (HLV * ql_upd + HLS * qi_upd) / CP_AIR, st)
            sa = W(m, sa - qa_upd, sa)
            ql_upd = W(m, z, ql_upd)
            qi_upd = W(m, z, qi_upd)
            qa_upd = W(m, z, qa_upd)
            # ---- remove supersaturation
            tt = Tj + st
            qs2, dq2 = svp.compute_qs(tt, pfj, want_dqsdT=True)
            g2 = dq2 * (torch.clamp(0.05 * (tt - TFREEZE + 20.0), 0.0, 1.0) * HLV
                        + torch.clamp(0.05 * (TFREEZE - tt), 0.0, 1.0) * HLS) / CP_AIR
            sup = torch.clamp(qvj + sq - qs2, min=0.0) / (1.0 + g2)
            sq = sq - sup
            ms = sup > 0.0
            if c.super_choice:
                ice = ms & vcold
                liq = ms & ~vcold
                qi_upd = W(ice, qi_upd + sup, qi_upd)
                si = W(ice, si + sup, si)
                st = W(ice, st + HLS * sup / CP_AIR, st)
                ql_upd = W(liq, ql_upd + sup, ql_upd)
                sl = W(liq, sl + sup, sl)
                st = W(liq, st + HLV * sup / CP_AIR, st)
                sa = W(ms, sa + (1.0 - qa_upd - 0.0), sa)
                qa_upd = W(ms, torch.full_like(z, 1.0 - 0.0), qa_upd)
            else:
                c20 = Tj <= TFREEZE - 20.0
                add_cld = qa_mean * sup * deltpg * inv_dtcloud
                add_clr = (1.0 - qa_mean) * sup * deltpg * inv_dtcloud
                snow_cld = W(ms & c20, snow_cld + add_cld, snow_cld)
                snow_clr = W(ms & c20, snow_clr + add_clr, snow_clr)
                a_snow_cld = W(ms & c20, qa_mean, a_snow_cld)
                a_snow_clr = W(ms & c20, 1.0 - qa_mean, a_snow_clr)
                rain_cld = W(ms & ~c20, rain_cld + add_cld, rain_cld)
                rain_clr = W(ms & ~c20, rain_clr + add_clr, rain_clr)
                a_rain_cld = W(ms & ~c20, qa_mean, a_rain_cld)
                a_rain_clr = W(ms & ~c20, 1.0 - qa_mean, a_rain_clr)
                st = W(ms, st + W(c20, torch.full_like(z, HLS), torch.full_like(z, HLV)) * sup / CP_AIR, st)
            # ---- final realizability
            qlu = qlj + sl
            m = (torch.abs(qlu) <= qmin) & (qvj + sq + qlu > 0.0)
            sl = W(m, -qlj, sl)
            sq = W(m, sq + qlu, sq)
            st = W(m, st - HLV * qlu / CP_AIR, st)
            qiu = qij + si
            m = (torch.abs(qiu) <= qmin) & (qvj + sq + qiu > 0.0)
            si = W(m, -qij, si)
            sq = W(m, sq + qiu, sq)
            st = W(m, st - HLS * qiu / CP_AIR, st)
            qau = qaj + sa
            m = torch.abs(qau) <= qmin
            sa = W(m, -qaj, sa)
            qa_mean_lst = qa_mean
            rain3d[j + 1] = rain_clr + rain_cld
            snow3d[j + 1] = snow_clr + snow_cld
            snowclr3d[j + 1] = snow_clr
            ST[j], SQ[j], SL[j], SI[j], SA[j] = st, sq, sl, si, sa
        S = lambda lst: torch.stack(lst, dim=1)
        return dict(ST=S(ST), SQ=S(SQ), SL=S(SL), SI=S(SI), SA=S(SA),
                    surfrain=dtcloud * (rain_clr + rain_cld), surfsnow=dtcloud * (snow_clr + snow_cld),
                    rain3d=S(rain3d), snow3d=S(snow3d), snowclr3d=S(snowclr3d), f_snow_berg=S(f_snow_berg))
