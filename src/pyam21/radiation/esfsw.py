"""ESF shortwave radiation (sea_esf_rad/esfsw_driver.F90 swresf, siena_201207).

Exponential-sum-fit gas transmissions (H2O and O3 k-distributions, CO2 and O2 from
fitted absorptivities, the Schumann-Runge band in the last band), Rayleigh scattering,
aerosol and cloud optical depths, the delta-Eddington layer properties for the clear
and overcast parts of each layer combined linearly in the cloud fraction (random
overlap), and the adding method for the fluxes. One solar zenith angle per column
(nzens = 1, NSOLWG = 1), one diffuse stream (sw_diff_streams = 1) or four.

Vectorised over columns; the band / frequency / level loops follow the Fortran.
Quantities that the Fortran carries from one band to the next (the CO2 and O2
absorptivities and effective optical depths, the pressure-scaled H2O amount) are
carried the same way.
"""
from __future__ import annotations

import math

import torch

from ..constants import CP_AIR, GRAV
from .esfsw_data import SolarSpectrum

WTMAIR = 2.896440e01
WTMO2 = 3.19988e01
PSTD_MKS = 101325.0
O2MIXRAT = 2.0953e-01
RHOAIR = 1.292269
RADCON_MKS = (GRAV / CP_AIR) * 86400.0
FLAG = 1.0e-99
ONEDI3 = 1.0 / 3.0
TWODI3 = 2.0 / 3.0


def deledd(taustr, omegastr, gstr, cosang, cosangstr, wtstr, nstr4, want_dif=True):
    """Delta-Eddington layer reflection / transmission (all points; the caller masks).

    float64 follows the Fortran expression by expression. In lower precision the
    single-scattering albedo cap 1 - 1e-8 rounds to 1 (a singular layer) and 1/exp(-tau q)
    overflows, so the cap is 1 - 1e-6 and the solution is written with exp(-tau q) only:
    rr4 * rr5 = 1 / (qq6 - qq7 rr2 rr3^2), algebraically the same as the Fortran."""
    exact = taustr.dtype == torch.float64
    cap = 9.9999999e-01 if exact else 1.0 - 1.0e-6
    om = torch.where(omegastr >= cap, torch.full_like(omegastr, cap), omegastr)
    tau = torch.where(taustr >= 1.0e02, torch.full_like(taustr, 1.0e02), taustr)
    g = gstr
    mu = cosang
    qq1 = 3.0 * (1.0 - om)
    qq2 = 1.0 - om * g
    qq3 = qq1 / qq2
    qq4 = torch.sqrt(qq1 * qq2)
    qq5 = torch.sqrt(qq3)
    qq6 = 1.0 + TWODI3 * qq5
    qq7 = 1.0 - TWODI3 * qq5
    rr1 = 1. / qq6
    rr2 = qq7 * rr1
    rr3 = torch.exp(-tau * qq4)
    if exact:
        rr4 = 1.0 / rr3
        rr5 = 1.0 / (qq6 * rr4 - qq7 * rr3 * rr2)
    else:
        rr45 = 1.0 / (qq6 - qq7 * rr2 * rr3 * rr3)          # rr4 * rr5

    def fluxes(m):
        s1 = 0.75 * om / (1.0 - (qq4 * m) ** 2)
        s2 = s1 * m * (1.0 + g * qq1 * ONEDI3)
        s3 = s1 * (1.0 + g * qq1 * m ** 2)
        s4 = s2 - TWODI3 * s3
        s5 = s2 + TWODI3 * s3
        s6 = torch.exp(-tau / m)
        if exact:
            s7 = (s4 * s6 - s5 * rr3 * rr2) * rr5
            q7r4s7 = qq7 * rr4 * s7
        else:
            num = s4 * s6 - s5 * rr3 * rr2
            s7 = num * rr3 * rr45
            q7r4s7 = qq7 * (num * rr45)
        s8 = (s5 - qq7 * s7) * rr1
        r = qq7 * s8 + qq6 * s7 - s4
        t = ((rr3 * qq6 * s8 + q7r4s7 - s5 * s6) + s6)
        return r, t, s6

    rdir, tdir, tde = fluxes(mu)
    if not want_dif:
        return rdir, tdir, tde, None, None
    rsum = torch.zeros_like(rdir)
    tsum = torch.zeros_like(rdir)
    for ns in range(len(cosangstr)):
        r, t, _ = fluxes(float(cosangstr[ns]))
        if nstr4:
            rsum = rsum + r * float(wtstr[ns]) * float(cosangstr[ns])
            tsum = tsum + t * float(wtstr[ns]) * float(cosangstr[ns])
        else:
            rsum = rsum + r
            tsum = tsum + t
    return rdir, tdir, tde, rsum, tsum


def adding(rdir, tdir, rdif, tdif, tde, alb_dir, alb_dif):
    """Adding method: layer properties (ncol, kx) -> reflectance, transmittance, tr_dir
    at the kx+1 interfaces (ncol, kx+1)."""
    ncol, kx = rdir.shape
    rup_dif = [None] * kx
    rup_dir = [None] * kx
    pdif = alb_dif
    pdir = alb_dir
    for k in range(kx - 1, -1, -1):
        dm2tl2 = tdif[:, k] / (1.0 - rdif[:, k] * pdif)
        rdm2tl2 = dm2tl2 * pdif
        rup_dif[k] = rdif[:, k] + tdif[:, k] * rdm2tl2
        rup_dir[k] = rdir[:, k] + tde[:, k] * pdir * dm2tl2 + (tdir[:, k] - tde[:, k]) * rdm2tl2
        pdir = rup_dir[k]
        pdif = rup_dif[k]
    one = torch.ones_like(alb_dir)
    trans = [one]
    refl = [pdir]
    trd = [one]
    tlevel = tde[:, 0]
    rdn = rdif[:, 0]
    tdn = tdir[:, 0]
    for k in range(1, kx):
        dm1tl2 = tdif[:, k] / (1.0 - rdif[:, k] * rdn)
        rdn2 = rdif[:, k] + rdn * tdif[:, k] * dm1tl2
        tdn2 = tlevel * (tdir[:, k] + rdir[:, k] * rdn * dm1tl2) + (tdn - tlevel) * dm1tl2
        dm32 = 1.0 / (1.0 - rup_dif[k] * rdn)
        dm3r2 = dm32 * rdn
        dm3r1p2 = 1.0 + rup_dif[k] * dm3r2
        alpp2 = (tdn - tlevel) * dm32
        trans.append(tlevel * (1.0 + rup_dir[k] * dm3r2) + alpp2)
        trd.append(tlevel)
        refl.append(tlevel * rup_dir[k] * dm3r1p2 + alpp2 * rup_dif[k])
        tlevel = tlevel * tde[:, k]
        rdn = rdn2
        tdn = tdn2
    dm32 = 1.0 / (1.0 - alb_dif * rdn)
    dm3r2 = dm32 * rdn
    dm3r1p2 = 1.0 + alb_dif * dm3r2
    alpp2 = (tdn - tlevel) * dm32
    trans.append(tlevel * (1.0 + alb_dir * dm3r2) + alpp2)
    trd.append(tlevel)
    refl.append(tlevel * alb_dir * dm3r1p2 + alpp2 * alb_dif)
    return torch.stack(refl, 1), torch.stack(trans, 1), torch.stack(trd, 1)


class ShortwaveESF:
    def __init__(self, spec: SolarSpectrum, solar_constant: float = 1365.0, do_totcld_forcing: bool = True,
                 lean=None):
        """lean: (solar_constant, solflxband) from esfsw_data.lean_solar, or None."""
        self.s = spec
        self.solar_constant = solar_constant
        self.do_totcld_forcing = do_totcld_forcing
        self.solflxband = spec.solflxband
        self.solflxtotal = spec.solflxtotal
        if lean is not None:
            self.solar_constant, self.solflxband = lean
            tot = 0.0
            for v in self.solflxband:
                tot = tot + v
            self.solflxtotal = tot

    # ---------------------------------------------------------------------------------
    def gas_opdep(self, press_in, pflux_in, rh2o, qo3, rrvco2, cosz):
        """compute_gas_props -> gasopdep (ncol, kx, nfrqpts)."""
        s = self.s
        ncol, kx = rh2o.shape
        press = 0.1 * (10.0 * press_in)
        pflux_mks = (10.0 * pflux_in) * 1.0e-1
        deltap = pflux_mks[:, 1:] - pflux_mks[:, :-1]
        delpdig = deltap / GRAV
        scalestr = pflux_mks                          # index k (use 1..kx)
        scale = scalestr * pflux_mks / PSTD_MKS
        wh2ostr = rh2o * delpdig
        wo3 = qo3 * delpdig
        cosang = torch.where(cosz == 0.0, torch.ones_like(cosz), cosz)[:, None]
        denom = 1.0 / (GRAV * RHOAIR * cosang * 2.0)
        totco2 = 1.0e02 * rrvco2 * scale * denom
        totco2str = 2.0e02 * rrvco2 * scalestr * denom
        toto2 = 1.0e02 * O2MIXRAT * scale * denom
        toto2str = 2.0e02 * O2MIXRAT * scalestr * denom
        zero = torch.zeros_like(rh2o[:, 0])
        # carried state (index 0 = the Fortran's k = 1, fixed at 0)
        st = {n: [zero] * (kx + 1) for n in ("aco2", "aco2s", "ao2", "ao2s")}
        eff = {n: [zero] * kx for n in ("co2", "o2")}
        wh2o = torch.zeros_like(rh2o)
        cos1 = cosang[:, 0]

        def fitted(name, cnt, tot, tots, co, k):
            """One level of the CO2/O2 absorptivity fit."""
            ok = (tot[:, k] < co["totmax"][b]) & (tots[:, k] < co["totstrmax"][b])
            a = co["c1"][b] * torch.exp(co["c3"][b] * torch.log(torch.where(ok, tot[:, k] + co["c2"][b],
                                                                                torch.ones_like(tot[:, k])))) - co["c4"][b]
            a_s = co["c1str"][b] * torch.exp(co["c3str"][b] * torch.log(torch.where(ok, tots[:, k] + co["c2str"][b],
                                                                                    torch.ones_like(tot[:, k])))) \
                - co["c4str"][b]
            anew = torch.where(ok, a, st[cnt][k])
            asnew = torch.where(ok, a_s, st[cnt + "s"][k])
            t = (1.0 - anew) * (1.0 - asnew) / ((1.0 - st[cnt][k - 1]) * (1.0 - st[cnt + "s"][k - 1]))
            e = -cos1 * torch.log(torch.where(ok, t, torch.ones_like(t)))
            prev = eff[name][k - 2] if k > 1 else zero
            st[cnt][k] = anew
            st[cnt + "s"][k] = asnew
            eff[name][k - 1] = torch.where(ok, e, prev)

        out = []
        np_ = 0
        for b in range(s.nbands):
            if b < s.nh2obands:
                wh2o = rh2o * delpdig * torch.exp(s.powph2o[b] * torch.log(press[:, :kx] * s.p0h2o[b]))
                if s.co2["c1"][b] != FLAG:
                    for k in range(1, kx + 1):
                        fitted("co2", "aco2", totco2, totco2str, s.co2, k)
                else:
                    eff["co2"] = [zero] * kx
                if s.o2["c1"][b] != FLAG:
                    for k in range(1, kx + 1):
                        fitted("o2", "ao2", toto2, toto2str, s.o2, k)
                else:
                    eff["o2"] = [zero] * kx
            if b == s.nbands - 1:
                sr = s.o2schrun
                for k in range(1, kx + 1):
                    ok = toto2str[:, k] < sr["totmax"]
                    a_s = sr["c1"] * torch.exp(sr["c3"] * torch.log(torch.where(ok, toto2str[:, k] + sr["c2"],
                                                                                torch.ones_like(cos1)))) - sr["c4"]
                    asnew = torch.where(ok, a_s, st["ao2s"][k])
                    t = (1.0 - asnew) / (1.0 - st["ao2s"][k - 1])
                    e = -cos1 * torch.log(torch.where(ok, t, torch.ones_like(t)))
                    prev = eff["o2"][k - 2] if k > 1 else zero
                    st["ao2s"][k] = asnew
                    eff["o2"][k - 1] = torch.where(ok, e, prev)
            eco2 = torch.stack(eff["co2"], 1)
            eo2 = torch.stack(eff["o2"], 1)
            for nf in range(int(s.nfreqpts[b])):
                if s.strterm[np_]:
                    opdep = s.kh2o[np_] * wh2ostr + s.ko3[np_] * wo3
                else:
                    opdep = s.kh2o[np_] * wh2o + s.ko3[np_] * wo3
                out.append(opdep + 1.0 * eco2 + eo2 + 0.0 + 0.0)
                np_ += 1
        return torch.stack(out, -1)

    # ---------------------------------------------------------------------------------
    def __call__(self, atm: dict, qo3, rrvco2, aero, cld: dict, surf: dict, astro: dict):
        """swresf.

        atm: press, pflux (ncol, kx+1), temp (ncol, kx+1), rh2o, deltaz, clouddeltaz (ncol, kx);
        aero: (extopdep, sctopdep, asymfac) (ncol, kx, nbands);
        cld: camtsw (ncol, kx), cldext, cldsct, cldasymm (ncol, kx, nbands);
        surf: asfc_vis_dir, asfc_nir_dir, asfc_vis_dif, asfc_nir_dif (ncol,);
        astro: cosz, fracday (ncol,), rrsun."""
        s = self.s
        press_in, pflux_in = atm["press"], atm["pflux"]
        ncol, kx = atm["rh2o"].shape
        daylight = astro["fracday"] != 0.0
        dl = daylight[:, None]
        cldfrac = cld["camtsw"]
        cloud = dl & (cldfrac > 0.0)
        cloud_in_column = cloud.any(dim=1)
        cloudfrac = torch.where(cloud, cldfrac, torch.zeros_like(cldfrac))
        press = 0.1 * (10.0 * press_in)
        pflux = 10.0 * pflux_in
        temp = atm["temp"]
        deltaz = atm["deltaz"]
        cdz = atm["clouddeltaz"]
        pflux_mks = pflux * 1.0e-1
        deltap = pflux_mks[:, 1:] - pflux_mks[:, :-1]
        gocpdp = RADCON_MKS / deltap
        cosz = astro["cosz"]
        gasopdep = self.gas_opdep(press_in, pflux_in, atm["rh2o"], qo3, rrvco2, cosz)
        densitymol = s.refquanray * press[:, :kx] / temp[:, :kx]
        aeroext, aerosct, aeroasy = aero
        ssolar = self.solar_constant * astro["rrsun"]
        cosang = torch.where(cosz == 0.0, torch.ones_like(cosz), cosz)[:, None]
        zero_l = torch.zeros(ncol, kx + 1, dtype=temp.dtype, device=temp.device)
        o = {k: zero_l for k in ("dfsw", "ufsw", "fsw", "dfswcf", "ufswcf", "fswcf")}
        o["hsw"] = torch.zeros(ncol, kx, dtype=temp.dtype, device=temp.device)
        o["hswcf"] = o["hsw"]
        z1 = torch.zeros(ncol, dtype=temp.dtype, device=temp.device)
        for k in ("dfsw_dir_sfc", "ufsw_dir_sfc", "ufsw_dif_sfc", "dfsw_vis_sfc", "ufsw_vis_sfc",
                  "dfsw_vis_sfc_dir", "ufsw_vis_sfc_dir", "ufsw_vis_sfc_dif", "dfsw_dir_sfc_clr",
                  "dfsw_vis_sfc_clr"):
            o[k] = z1
        cstr, wstr, nstr4 = s.cosangstr, s.wtstr, s.nstreams == 4
        np_ = 0
        for b in range(s.nbands):
            rayop = s.betaddensitymol[b] * densitymol * deltaz
            cext = torch.where(cloud, 1.0e-03 * cld["cldext"][..., b] * cdz, torch.zeros_like(cdz))
            csct = torch.where(cloud, 1.0e-03 * cld["cldsct"][..., b] * cdz, torch.zeros_like(cdz))
            casy = torch.where(cloud, cld["cldasymm"][..., b], torch.zeros_like(cdz))
            ae, as_, ag = aeroext[..., b], aerosct[..., b], aeroasy[..., b]
            sctclr = rayop + as_
            gclr = ag * as_ / sctclr
            fclr = ag * gclr
            gstrclr = (gclr - fclr) / (1.0 - fclr)
            sctovc = rayop + as_ + csct
            govc = ((casy * csct) + (ag * as_)) / sctovc
            fovc = ((casy ** 2 * csct) + (ag ** 2 * as_)) / sctovc
            gstrovc = (govc - fovc) / (1.0 - fovc)
            sumtr = sumtr_dir = sumre = zero_l
            sumtr_dir_up = z1
            sumtrclr = sumreclr = sumtr_dir_clr = zero_l
            if b < s.nirbands:
                alb_dir, alb_dif = surf["asfc_nir_dir"], surf["asfc_nir_dif"]
            else:
                alb_dir, alb_dif = surf["asfc_vis_dir"], surf["asfc_vis_dif"]
            for nf in range(int(s.nfreqpts[b])):
                gop = gasopdep[..., np_]
                extclr = gop + rayop + ae
                ssaclr = sctclr / extclr
                taustrclr = extclr * (1.0 - ssaclr * fclr)
                omstrclr = ssaclr * ((1.0 - fclr) / (1.0 - ssaclr * fclr))
                rdc, tdc, tec, rfc, tfc = deledd(taustrclr, omstrclr, gstrclr, cosang, cstr, wstr, nstr4)
                extovc = gop + rayop + ae + cext
                ssaovc = sctovc / extovc
                taustrovc = extovc * (1.0 - ssaovc * fovc)
                omstrovc = ssaovc * ((1.0 - fovc) / (1.0 - ssaovc * fovc))
                rdo, tdo, teo, rfo, tfo = deledd(taustrovc, omstrovc, gstrovc, cosang, cstr, wstr, nstr4)
                cf = cloudfrac
                mix = lambda ovc, clr: torch.where(cloud, cf * ovc + (1.0 - cf) * clr, clr)
                rdir, rdif = mix(rdo, rdc), mix(rfo, rfc)
                tdir, tdif, tde = mix(tdo, tdc), mix(tfo, tfc), mix(teo, tec)
                refl, trans, trd = adding(rdir, tdir, rdif, tdif, tde, alb_dir, alb_dif)
                wtfac = s.wtfreq[np_] * 1.0 * cosang[:, 0]
                w = wtfac[:, None]
                sumtr = torch.where(dl, sumtr + trans * w, sumtr)
                sumtr_dir = torch.where(dl, sumtr_dir + trd * w, sumtr_dir)
                sumre = torch.where(dl, sumre + refl * w, sumre)
                sumtr_dir_up = torch.where(daylight, sumtr_dir_up + trd[:, kx] * alb_dir * wtfac, sumtr_dir_up)
                if self.do_totcld_forcing:
                    reflc, transc, trdc = adding(rdc, tdc, rfc, tfc, tec, alb_dir, alb_dif)
                    cic = cloud_in_column[:, None]
                    sumtrclr = torch.where(cic, sumtrclr + transc * w,
                                           torch.where(dl, sumtrclr + trans * w, sumtrclr))
                    sumtr_dir_clr = torch.where(cic, sumtr_dir_clr + trdc * w,
                                                torch.where(dl, sumtr_dir_clr + trd * w, sumtr_dir_clr))
                    sumreclr = torch.where(cic, sumreclr + reflc * w,
                                           torch.where(dl, sumreclr + refl * w, sumreclr))
                np_ += 1
            solarflux = astro["fracday"] * float(self.solflxband[b]) * ssolar / self.solflxtotal
            sf = solarflux[:, None]
            o["dfsw"] = torch.where(dl, o["dfsw"] + sumtr * sf, o["dfsw"])
            o["ufsw"] = torch.where(dl, o["ufsw"] + sumre * sf, o["ufsw"])
            fswband = (sumre * sf) - (sumtr * sf)
            o["fsw"] = torch.where(dl, o["fsw"] + fswband, o["fsw"])
            dls = daylight
            o["dfsw_dir_sfc"] = torch.where(dls, o["dfsw_dir_sfc"] + sumtr_dir[:, kx] * solarflux, o["dfsw_dir_sfc"])
            o["ufsw_dir_sfc"] = torch.where(dls, o["ufsw_dir_sfc"] + sumtr_dir_up * solarflux, o["ufsw_dir_sfc"])
            o["ufsw_dif_sfc"] = torch.where(dls, o["ufsw_dif_sfc"] + sumre[:, kx] * solarflux, o["ufsw_dif_sfc"])
            if b >= s.nirbands:
                for key, v in (("dfsw_vis_sfc", sumtr[:, kx]), ("ufsw_vis_sfc", sumre[:, kx]),
                               ("dfsw_vis_sfc_dir", sumtr_dir[:, kx]), ("ufsw_vis_sfc_dir", sumtr_dir_up),
                               ("ufsw_vis_sfc_dif", sumre[:, kx])):
                    o[key] = torch.where(dls, o[key] + v * solarflux, o[key])
            hswband = (fswband[:, 1:] - fswband[:, :-1]) * gocpdp
            o["hsw"] = torch.where(dl, o["hsw"] + hswband, o["hsw"])
            if self.do_totcld_forcing:
                o["dfsw_dir_sfc_clr"] = torch.where(dls, o["dfsw_dir_sfc_clr"] + sumtr_dir_clr[:, kx] * solarflux,
                                                    o["dfsw_dir_sfc_clr"])
                if b >= s.nirbands:
                    o["dfsw_vis_sfc_clr"] = torch.where(dls, o["dfsw_vis_sfc_clr"] + sumtrclr[:, kx] * solarflux,
                                                        o["dfsw_vis_sfc_clr"])
                o["dfswcf"] = o["dfswcf"] + sumtrclr * sf
                o["ufswcf"] = o["ufswcf"] + sumreclr * sf
                fbc = (sumreclr * sf) - (sumtrclr * sf)
                o["fswcf"] = o["fswcf"] + fbc
                o["hswcf"] = o["hswcf"] + (fbc[:, 1:] - fbc[:, :-1]) * gocpdp
        o["dfsw_dif_sfc"] = torch.where(daylight, o["dfsw"][:, kx] - o["dfsw_dir_sfc"], z1)
        o["dfsw_dif_sfc_clr"] = torch.where(daylight, o["dfswcf"][:, kx] - o["dfsw_dir_sfc_clr"], z1)
        o["dfsw_vis_sfc_dif"] = torch.where(daylight, o["dfsw_vis_sfc"] - o["dfsw_vis_sfc_dir"], z1)
        for key in ("fsw", "dfsw", "ufsw", "fswcf", "dfswcf", "ufswcf"):
            o[key] = 1.0e-03 * (1.0e03 * o[key])
        return o
