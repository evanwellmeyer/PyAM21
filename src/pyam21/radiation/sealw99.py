"""The sealw99 longwave radiation package (sea_esf_rad, siena_201207) in torch.

Translation of sealw99.F90 (sealw99, cool_to_space_exact/approx, e1e290, e290, esfc, enear,
co2_source_calc), optical_path.F90, the run-time part of gas_tf.F90 (co2coef, transfn,
transcolrow, trans_nearby, trans_sfc), longwave_clouds.F90 (cldtau, cloud),
longwave_fluxes.F90 and the cfc_* routines of lw_gases_stdtf.F90, for the configuration
of the SCM cases: continuum_form 'ckd2.1', hitran_2000, one 1200-1400 cm-1 H2O band,
do_lwcldemiss (7 cloud bands), co2/ch4/n2o/cfc active, LW aerosols, do_totcld_forcing,
no NLTE, do_thick false.

Tensors are (ncol, level[, band]) with level 0 at the top; K layers, K+1 interfaces.
Fortran level k (1-based) is index k-1 here. The Fortran keeps several work arrays
between the per-level calls (overod, to3cnt, co21c, co21r, tch4n2oe, tcfc8, cldtf,
trans_band1/2) and later steps read values an earlier call left behind, so they are
carried the same way here.
"""
from __future__ import annotations

import math

import torch

from ..constants import AVOGNO, DIFFAC, GRAV, PSTD, RADCON_MKS, RDGAS, RVGAS, TFREEZE, WTMAIR
from .lw_tables import (BAND_NO_END, BAND_NO_START, CLD_INDX_TABLE, MASS_INC, MASS_MIN, NBLY, OFFSET,
                        TEMP_INC, TEMP_MIN, LongwaveData)

D622 = RDGAS / RVGAS
NBTRGE = 1
NBANDS = 6 + NBTRGE
B0, B1, B2, B3 = -0.51926410e-04, -0.18113332e-03, -0.10680132e-05, -0.67303519e-07
WTMF = dict(f11=137.36855, f12=120.91395, f113=187.3765, f22=86.46892)
STRF11 = (0.000000E+00, 0.000000E+00, 0.527655E+02, 0.297523E+04, 0.134488E+03, 0.247279E+03, 0.710717E+03, 0.0)
STRF12 = (0.552499E+01, 0.136436E+03, 0.243867E+02, 0.612532E+03, 0.252378E+04, 0.438226E+02, 0.274950E+04, 0.0)
STRF113 = (0.627223E+01, 0.690936E+02, 0.506764E+02, 0.122039E+04, 0.808762E+03, 0.742843E+03, 0.109485E+04,
           0.194768E+03)
STRF22 = (0.301881E+02, 0.550826E+01, 0.397496E+03, 0.124802E+04, 0.190285E+02, 0.460065E+02, 0.367359E+04,
          0.508838E+03)
SF1115, SF1215, SF11315, SF2215 = 0.219856E+02, 0.515665E+02, 0.430969E+02, 0.176035E+03


def _set(x, idx, v):
    """Functional x[idx] = v (keeps autograd happy when x is reused)."""
    x = x.clone()
    x[idx] = v
    return x


def _loc(x, vmin, inc, nmax):
    """locate_in_table: 0-based index and offset (index clamped to the table)."""
    fx = torch.trunc((x - vmin) / inc)
    return fx.long().clamp(0, nmax - 1), x - fx * inc - vmin


class _Tab:
    def __init__(self, t, dt, dev):
        T = lambda a: None if a is None else torch.as_tensor(a, dtype=dt, device=dev)
        self.vae, self.td, self.md, self.cd = T(t.vae), T(t.td), T(t.md), T(t.cd)

    def __call__(self, ix, iy, dx, dy, m=None):
        """looktab_type1/2 (iy 0-based)."""
        iy = iy.clamp(0, self.vae.shape[1] - 1)
        if m is None:
            v, td, md, cd = self.vae[ix, iy], self.td[ix, iy], self.md[ix, iy], self.cd[ix, iy]
        else:
            v, td, md, cd = (a[..., m][ix, iy] for a in (self.vae, self.td, self.md, self.cd))
        return v + dx * td + dy * md + dx * dy * cd

    def col(self, ix, dx, n):
        """looktab_type3 in column n (0-based)."""
        return self.vae[:, n][ix] + dx * self.td[:, n][ix]


class SeaLW99:
    """sealw99. ``gas_tf`` holds the standard-profile transmission tables of gas_tf_mod
    (co251 ... n2o98, co2m51 ..., co215nbps1 ..., stemp, gtemp, ixprkminh2o)."""

    def __init__(self, data: LongwaveData, gas_tf: dict, do_totcld_forcing: bool = True):
        self.d = data
        self.g = gas_tf
        self.do_totcld_forcing = do_totcld_forcing
        self._cache = {}

    # ------------------------------------------------------------------ tables on device
    def _t(self, dt, dev):
        key = (dt, dev)
        if key not in self._cache:
            d = self.d
            T = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
            c = {n: _Tab(getattr(d, n), dt, dev) for n in ("tab1", "tab2", "tab3", "tab1w", "tab1a", "tab2a",
                                                          "tab3a", "tabsr", "radfunc")}
            c.update({k: T(v) for k, v in self.g.items() if k != "ixprkminh2o"})
            c.update(apcm=T(d.apcm), bpcm=T(d.bpcm), atpcm=T(d.atpcm), btpcm=T(d.btpcm), acomb=T(d.acomb),
                     bcomb=T(d.bcomb))
            self._cache[key] = c
        return self._cache[key]

    # ------------------------------------------------------------------ optical_path
    def _optical_path(self, atm, qo3, gases, aer_tot, aer_ke):
        d = self.d
        press, pflux = 10.0 * atm["press"], 10.0 * atm["pflux"]
        temp, tflux, rh2o = atm["temp"], atm["tflux"], atm["rh2o"]
        K = rh2o.shape[1]
        atmden = (pflux[:, 1:] - pflux[:, :-1]) / (1.0e02 * GRAV)
        vv = 0.5 * (pflux[:, 1:] + pflux[:, :-1]) / PSTD
        o = {}
        # optical_h2o
        tpl1 = torch.cat([temp[:, K - 1:K], tflux[:, 1:K], 0.5 * (tflux[:, K:] + temp[:, K - 1:K])], dim=1)
        tpl2 = torch.cat([tflux[:, :1] * 0, tflux[:, 1:K], 0.5 * (tflux[:, K - 1:K] + temp[:, K - 1:K])], dim=1)
        qh2o = rh2o * DIFFAC
        var1 = atmden * qh2o
        var2 = var1 * (vv + 3.0e-04)
        z = torch.zeros_like(var2[:, :1])
        totphi = torch.cat([z, torch.cumsum(var2, dim=1)], dim=1)
        den = 1.0e02 * GRAV * PSTD
        emx1 = qh2o[:, K - 1] * press[:, K - 1] * (press[:, K - 1] - pflux[:, K - 1]) / den
        emx2 = qh2o[:, K - 1] * press[:, K - 1] * (pflux[:, K] - press[:, K - 1]) / den
        empl1 = torch.cat([var2[:, K - 1:K], qh2o * pflux[:, 1:] * (pflux[:, 1:] - press[:, :K]) / den], dim=1)
        e2 = qh2o[:, 1:] * pflux[:, 1:K] * (press[:, 1:K] - pflux[:, 1:K]) / den
        empl2 = torch.cat([z, e2, e2[:, -1:]], dim=1)
        c1, c2 = float(d.csfah2o[0, 0]), float(d.csfah2o[1, 0])
        tdif = temp[:, :K] - 2.5e02
        vrpf = var2 * torch.exp(c1 * tdif + c2 * (tdif * tdif))
        tphf = torch.cat([z, torch.cumsum(vrpf, dim=1)], dim=1)
        tdif2 = tpl2 - 2.5e02
        tdif = tpl1 - 2.5e02
        f = lambda t: torch.exp(c1 * t + c2 * (t * t))
        emx1f = emx1 * f(tdif2[:, K])
        emx2f = emx2 * f(tdif[:, K])
        empl1f = torch.cat([vrpf[:, K - 1:K], empl1[:, 1:] * f(tdif[:, 1:])], dim=1)
        e2f = empl2[:, 1:K] * f(tdif2[:, 1:K])
        empl2f = torch.cat([z, e2f, e2f[:, -1:]], dim=1)
        o.update(var1=var1, var2=var2, totphi=totphi, emx1=emx1, emx2=emx2, empl1=empl1, empl2=empl2,
                 vrpfh2o=vrpf, tphfh2o=tphf, emx1f=emx1f, emx2f=emx2f, empl1f=empl1f, empl2f=empl2f)
        # optical_path_ckd
        rvh2o = rh2o / D622
        wk = rvh2o * AVOGNO / WTMAIR * atmden / (1.0 + rvh2o)
        rhoave = (press[:, :K] / PSTD) * (TFREEZE / temp[:, :K])
        rfrgn = rhoave * (296.0 / TFREEZE) / (1.0 + rvh2o)
        rh2os = rfrgn * rvh2o
        tfac = temp[:, :K] - 296.0
        tmpexp = torch.exp(-.020 * tfac)
        xch2obd = torch.stack([wk * 1.0e-20 * (float(d.svj[nu]) * rh2os * tmpexp + float(d.fvj[nu]) * rfrgn)
                               * float(d.radfnbd[nu]) for nu in range(7)], dim=-1)
        xwd = wk * 1.0e-20 * (d.svjwd * rh2os * tmpexp + d.fvjwd * rfrgn) * d.radfnbdwd
        o.update(wk=wk, rfrgn=rfrgn, rh2os=rh2os, tfac=tfac, xch2obd=xch2obd, xch2obdwd=xwd,
                 totch2obdwd=torch.cat([z, torch.cumsum(xwd, dim=1)], dim=1))
        # optical_o3
        var3 = atmden * qo3 * DIFFAC
        var4 = var3 * (vv + 3.0e-03)
        o.update(var3=var3, var4=var4, toto3=torch.cat([z, torch.cumsum(var3, dim=1)], dim=1),
                 tphio3=torch.cat([z, torch.cumsum(var4, dim=1)], dim=1))
        # cfc_optical_depth
        for g in ("f11", "f12", "f113", "f22"):
            rr = gases["rrv" + g] * (WTMF[g] / WTMAIR)
            o["tot" + g] = torch.cat([z, torch.cumsum(atmden * rr * 2.0, dim=1)], dim=1)
        o["totaerooptdep"], o["aerooptdep_KE_15"] = aer_tot, aer_ke
        return o

    def _totch2obd(self, o, n):
        """get_totch2obd (n 1-based)."""
        x = o["xch2obd"][..., n - 1]
        return torch.cat([torch.zeros_like(x[:, :1]), torch.cumsum(x, dim=1)], dim=1)

    def _totch2o(self, o, t, n, ixoe1, dte1):
        """get_totch2o (n 1-based band of the 5-1195 cm-1 region)."""
        d = self.d
        K = o["tfac"].shape[1]
        nu = n + d.ioffh2o - 1
        tmpexp = torch.exp(-.013 * o["tfac"])
        radf = t["radfunc"].col(ixoe1[:, :K], dte1[:, :K], nu)
        sh2o = float(d.ssh2o_296[nu] * d.sfac[nu]) * tmpexp
        fh2o0 = float(d.sfh2o[nu] * d.fscal[nu])
        inc = o["wk"] * 1.0e-20 * (sh2o * o["rh2os"] + fh2o0 * o["rfrgn"]) * float(d.vvj[n - 1]) * radf
        return torch.cat([torch.zeros_like(inc[:, :1]), torch.cumsum(inc, dim=1)], dim=1)

    @staticmethod
    def _cfc_exact(o, i):
        """cfc_exact (index i 1-based) -> (ncol, K)."""
        i -= 1
        return (1.0 - STRF113[i] * o["totf113"][:, 1:] - STRF22[i] * o["totf22"][:, 1:]
                - STRF11[i] * o["totf11"][:, 1:] - STRF12[i] * o["totf12"][:, 1:])

    @staticmethod
    def _cfc_exact_part(o, i, k):
        """cfc_exact_part (index i 1-based, level k 0-based) -> values for layers k..K-1."""
        i -= 1
        dd = lambda g: o["tot" + g][:, k + 1:] - o["tot" + g][:, k:k + 1]
        return 1.0 - STRF113[i] * dd("f113") - STRF22[i] * dd("f22") - STRF11[i] * dd("f11") - STRF12[i] * dd("f12")

    @staticmethod
    def _cfc_overod(o, k=0):
        dd = lambda g: o["tot" + g][:, k + 1:] - (o["tot" + g][:, k:k + 1] if k > 0 else 0.0)
        if k == 0:
            dd = lambda g: o["tot" + g][:, 1:]
        return 1.0 - SF11315 * dd("f113") - SF2215 * dd("f22") - SF1115 * dd("f11") - SF1215 * dd("f12")

    @staticmethod
    def _cfc_indx8_part(o, k):
        """cfc_indx8_part(8, ...): tcfc8 values for levels k+1..K (k 0-based)."""
        dd = lambda g: o["tot" + g][:, k + 1:] - o["tot" + g][:, k:k + 1]
        return 1.0 - STRF113[7] * dd("f113") - STRF22[7] * dd("f22")

    # ------------------------------------------------------------------ gas_tf
    def _co2coef(self, atm, t):
        g = self.g
        press, pflux = 10.0 * atm["press"], 10.0 * atm["pflux"]
        temp, tflux = atm["temp"], atm["tflux"]
        K = press.shape[1] - 1
        tmpdiff = temp - t["stemp"]
        pdflux = pflux[:, 1:] - pflux[:, :-1]
        z = torch.zeros_like(pdflux[:, :1])
        tstdav = torch.cat([z, torch.cumsum(t["gtemp"][:K] * pdflux, dim=1)], dim=1)
        tdav = torch.cat([z, torch.cumsum(t["gtemp"][:K] * pdflux * tmpdiff[:, :K], dim=1)], dim=1)
        alogps8 = math.log(PSTD * 0.8)
        palog8 = -math.log(0.8)
        a1 = (torch.log(press[:, K]) - alogps8) / palog8
        a2 = 1.0 - a1
        tdif = tflux - 2.5e02
        tlsqu = torch.where(tflux <= 2.5e02, B0 + tdif * (B1 + tdif * (B2 + B3 * tdif)), torch.full_like(tflux, B0))
        A1, A2 = a1[:, None], a2[:, None]
        # transfn
        co215nb = A1[..., None] * t["co215nbps1"] + A2[..., None] * t["co215nbps8"]
        co2dt15nb = 1.0e-2 * (A1[..., None] * t["co2dt15nbps1"] + A2[..., None] * t["co2dt15nbps8"])
        co2d2t15nb = 1.0e-3 * (A1[..., None] * t["co2d2t15nbps1"] + A2[..., None] * t["co2d2t15nbps8"])
        co2mr = A1 * t["co2m51"] + A2 * t["co2m58"]
        co2md = 1.0e-02 * (A1 * t["cdtm51"] + A2 * t["cdtm58"])
        co2m2d = 1.0e-03 * (A1 * t["c2dm51"] + A2 * t["c2dm58"])
        dift = (tdav[:, 1:] / tstdav[:, 1:])[..., None]
        sp = co215nb[:, 1:] + dift * (co2dt15nb[:, 1:] + 0.5 * dift * co2d2t15nb[:, 1:])
        co2spnb = torch.cat([torch.ones_like(sp[:, :1]), sp], dim=1)
        tl0 = tlsqu[:, :1, None]
        co2spnb = co2spnb * (1.0 - tl0) + tl0
        td = tmpdiff[:, :K]
        co2nbl = co2mr + td * (co2md + 0.5 * td * co2m2d)
        co2nbl = co2nbl * (1.0 - tlsqu[:, :K]) + tlsqu[:, :K]
        return dict(a1=a1, a2=a2, tdav=tdav, tstdav=tstdav, tmpdiff=tmpdiff, tlsqu=tlsqu, co2spnb=co2spnb,
                    co2nbl=co2nbl, K=K)

    def _transcolrow(self, gt, t, k, st):
        """transcolrow(Gas_tf, k, k, k, KE+1, k+1, KE+1, co21c, co21r, tch4n2oe), k 0-based;
        updates st (co21c, co21r, tch4n2oe, n2o9c, tn2o17)."""
        K = gt["K"]
        A1, A2 = gt["a1"][:, None], gt["a2"][:, None]
        tdav, tstdav, tmpdiff = gt["tdav"], gt["tstdav"], gt["tmpdiff"]
        kp = slice(k, K + 1)
        dift = (tdav[:, k + 1:] - tdav[:, k:k + 1]) / (tstdav[:, k + 1:] - tstdav[:, k:k + 1])
        own = 0.5 * (tmpdiff[:, k] + tmpdiff[:, k - 1]) if k > 0 else torch.zeros_like(tmpdiff[:, k])
        dift = torch.cat([own[:, None], dift], dim=1)
        col = lambda n: t[n][k:, k]
        row = lambda n: t[n][k, k:]
        co2p = A1 * col("co251") + A2 * col("co258")
        dco2dt = 1.0e-02 * (A1 * col("cdt51") + A2 * col("cdt58"))
        d2cdt2 = 1.0e-03 * (A1 * col("c2d51") + A2 * col("c2d58"))
        ch41c = A1 * col("ch451") + A2 * col("ch458")
        n2o1c = A1 * col("n2o51") + A2 * col("n2o58")
        n2o17c = A1 * col("n2o71") + A2 * col("n2o78")
        n2o9c = A1 * col("n2o91") + A2 * col("n2o98")
        c = co2p + dift * (dco2dt + 0.5 * dift * d2cdt2)
        tl = gt["tlsqu"][:, kp]
        c = c * (1.0 - tl) + tl
        st["co21c"] = _set(st["co21c"], (slice(None), kp), c)
        co2p = A1 * row("co251") + A2 * row("co258")
        dco2dt = 1.0e-02 * (A1 * row("cdt51") + A2 * row("cdt58"))
        d2cdt2 = 1.0e-03 * (A1 * row("c2d51") + A2 * row("c2d58"))
        r = co2p[:, 1:] + dift[:, 1:] * (dco2dt[:, 1:] + 0.5 * dift[:, 1:] * d2cdt2[:, 1:])
        tlk = gt["tlsqu"][:, k:k + 1]
        r = r * (1.0 - tlk) + tlk
        one = torch.ones_like(gt["tdav"])
        st["co21r"] = torch.cat([one[:, :k + 1], r], dim=1)
        e = ch41c * n2o1c
        lo = k if k == 0 else k + 1
        st["tch4n2oe"] = _set(st["tch4n2oe"], (slice(None), slice(lo, K + 1)), e[:, lo - k:])
        st["n2o9c"] = torch.cat([one[:, :k], n2o9c], dim=1)
        st["tn2o17"] = torch.cat([one[:, :k], n2o17c], dim=1)

    def _trans_near(self, atm, gt, overod):
        """Common part of trans_nearby / trans_sfc: ca, cb (0-based, full length)."""
        press, pflux = 10.0 * atm["press"], 10.0 * atm["pflux"]
        K = gt["K"]
        ix = self.g["ixprkminh2o"]
        km = max(ix - 1, 1) - 1
        kmp1 = max(ix - 1, 2) - 1
        pdfinv = 1.0 / (pflux[:, 1:] - pflux[:, :-1])
        N = press.shape[0]
        rlog = torch.zeros_like(press)
        rlog = _set(rlog, (slice(None), slice(km, K)), torch.log(gt["co2nbl"][:, km:K] * overod[:, km + 1:K + 1]))
        alpa = torch.zeros_like(press)
        alpb = torch.zeros_like(press)
        delpr1 = pdfinv[:, kmp1:K] * (press[:, kmp1:K] - pflux[:, kmp1:K])
        alpb = _set(alpb, (slice(None), slice(kmp1, K)), -torch.sqrt(delpr1) * rlog[:, kmp1:K])
        delpr2 = torch.zeros_like(press)
        delpr2 = _set(delpr2, (slice(None), slice(kmp1, K + 1)),
                      pdfinv[:, kmp1 - 1:K] * (pflux[:, kmp1:K + 1] - press[:, kmp1 - 1:K]))
        alpa = _set(alpa, (slice(None), slice(km, K)), -torch.sqrt(delpr2[:, km + 1:K + 1]) * rlog[:, km:K])
        alpa = _set(alpa, (slice(None), K), -rlog[:, K - 1])
        alpb = _set(alpb, (slice(None), K),
                    -rlog[:, K - 1] * torch.sqrt(pdfinv[:, K - 1] * (pflux[:, K] - press[:, K - 2])))
        poly = lambda a: a * (-0.66667e+00 + a * (0.25e+00 + a * (-0.066667e+00)))
        ca = torch.zeros_like(press)
        cb = torch.zeros_like(press)
        ca = _set(ca, (slice(None), slice(km, K + 1)), poly(alpa[:, km:K + 1]))
        cb = _set(cb, (slice(None), slice(kmp1, K + 1)), poly(alpb[:, kmp1:K + 1]))
        return ca, cb, press, pflux

    def _trans_nearby(self, atm, gt, overod, co21c):
        ca, cb, _, _ = self._trans_near(atm, gt, overod)
        K = gt["K"]
        ix = self.g["ixprkminh2o"] - 1
        v = 1.0 + 0.5 * (cb[:, ix:K] + ca[:, ix - 1:K - 1])
        co21c = _set(co21c, (slice(None), slice(ix, K)), v)
        return _set(co21c, (slice(None), K), 1.0 + ca[:, K - 1])

    def _trans_sfc(self, atm, gt, overod):
        ca, cb, press, pflux = self._trans_near(atm, gt, overod)
        K = gt["K"]
        pdfl = pflux[:, K] - pflux[:, K - 1]
        c = 1.0 + (pdfl * ca[:, K] - (press[:, K - 1] - pflux[:, K - 1]) * cb[:, K - 1]) / (pflux[:, K] - press[:, K - 1])
        r = 1.0 + ((pflux[:, K] - press[:, K - 2]) * cb[:, K] - (pflux[:, K] - press[:, K - 1]) * ca[:, K - 1]) / \
            (press[:, K - 1] - press[:, K - 2])
        return c, r

    # ------------------------------------------------------------------ optical transmissions
    def _ot_from_ks(self, o, st):
        d = self.d
        K = o["var2"].shape[1]
        ao3, bo3 = float(d.ao3rnd[1]), float(d.bo3rnd[1])
        aer = o["totaerooptdep"]
        tmp1 = bo3 * o["tphio3"][:, 1:] / o["toto3"][:, 1:]
        tmp2 = 0.5 * (tmp1 * (torch.sqrt(1.0 + (4.0 * ao3 * o["toto3"][:, 1:]) / tmp1) - 1.0))
        tmp2 = tmp2 + DIFFAC * self._totch2obd(o, 6)[:, 1:]
        tmp2 = tmp2 + aer[:, 1:, 5]
        one = torch.ones_like(tmp2[:, :1])
        st["to3cnt"] = torch.cat([one, torch.exp(-1.0 * tmp2) * self._cfc_exact(o, 6)], dim=1)
        tmp1 = torch.sqrt(d.ab15wd * o["totphi"][:, 1:])
        tmp1 = tmp1 + DIFFAC * o["totch2obdwd"][:, 1:]
        tmp1 = tmp1 + aer[:, 1:, 8]
        ov = torch.exp(-1.0 * tmp1)
        ov = ov * (170. / 240. + 70. / 240. * st["tn2o17"][:, 1:])
        ov = ov * self._cfc_overod(o)
        st["overod"] = torch.cat([one, ov], dim=1)
        cnt = []
        for nb, na, ic in ((4, 3, 4), (5, 4, 5), (7, 6, 7)):
            tt = DIFFAC * self._totch2obd(o, nb)[:, 1:]
            tt = tt + aer[:, 1:, na]
            cnt.append(torch.cat([one, torch.exp(-1.0 * tt) * self._cfc_exact(o, ic)], dim=1))
        st["cnttaub"] = cnt
        av = o["totphi"][:, 1:]
        st["avephi"] = torch.cat([av, av[:, K - 2:K - 1] + o["emx1"][:, None]], dim=1)
        av = o["tphfh2o"][:, 1:]
        st["avephif"] = torch.cat([av, av[:, K - 2:K - 1] + o["emx1f"][:, None]], dim=1)

    def _ot_k_down(self, o, st, k):
        """optical_trans_funct_k_down, k 0-based (Fortran k+1)."""
        d = self.d
        K = o["var2"].shape[1]
        ao3, bo3 = float(d.ao3rnd[1]), float(d.bo3rnd[1])
        aer = o["totaerooptdep"]
        dk = lambda x: x[:, k + 1:] - x[:, k:k + 1]
        avmo3 = torch.clamp(dk(o["toto3"]), min=1.0e-10)
        avpho3 = torch.clamp(dk(o["tphio3"]), min=1.0e-12)
        avckdwd = dk(o["totch2obdwd"])
        avckdo3 = dk(self._totch2obd(o, 6))
        avaero3 = dk(aer[..., 5])
        av = dk(o["totphi"])
        avephi = st["avephi"]
        avephi = _set(avephi, (slice(None), slice(k, K)), av)
        st["avephi"] = _set(avephi, (slice(None), K), avephi[:, K - 2] + o["emx1"])
        av = dk(o["tphfh2o"])
        avephif = _set(st["avephif"], (slice(None), slice(k, K)), av)
        st["avephif"] = _set(avephif, (slice(None), K), avephif[:, K - 2] + o["emx1f"])
        tmp1 = torch.sqrt(d.ab15wd * st["avephi"][:, k:K])
        tmp1 = tmp1 + DIFFAC * avckdwd
        tmp1 = tmp1 + dk(aer[..., 8])
        ov = torch.exp(-1.0 * tmp1)
        ov = ov * (170. / 240. + 70. / 240. * st["tn2o17"][:, k + 1:])
        ov = ov * self._cfc_overod(o, k) if k > 0 else ov * self._cfc_overod(o)
        st["overod"] = _set(st["overod"], (slice(None), slice(k + 1, K + 1)), ov)
        tmp1 = bo3 * avpho3 / avmo3
        tmp2 = 0.5 * (tmp1 * (torch.sqrt(1.0 + (4.0 * ao3 * avmo3) / tmp1) - 1.0))
        tmp2 = tmp2 + DIFFAC * avckdo3
        tmp2 = tmp2 + avaero3
        to3 = torch.exp(-1.0 * tmp2) * self._cfc_exact_part(o, 6, k)
        st["to3cnt"] = _set(st["to3cnt"], (slice(None), slice(k + 1, K + 1)), to3)

    def _ot_ke(self, o, st):
        d = self.d
        K = o["var2"].shape[1]
        ao3, bo3 = float(d.ao3rnd[1]), float(d.bo3rnd[1])
        tmp1 = torch.sqrt(d.ab15wd * o["var2"][:, K - 1])
        tmp1 = tmp1 + DIFFAC * o["xch2obdwd"][:, K - 1]
        tmp1 = tmp1 + o["aerooptdep_KE_15"]
        ov = torch.exp(-1.0 * tmp1)
        ov = ov * (170. / 240. + 70. / 240. * st["tn2o17"][:, K])
        ov = ov * self._cfc_overod(o, K - 1)[:, 0]
        st["overod"] = _set(st["overod"], (slice(None), K), ov)
        tmp1 = bo3 * o["var4"][:, K - 1] / o["var3"][:, K - 1]
        tmp2 = 0.5 * (tmp1 * (torch.sqrt(1.0 + (4.0 * ao3 * o["var3"][:, K - 1]) / tmp1) - 1.0))
        tmp2 = tmp2 + DIFFAC * o["xch2obd"][:, K - 1, 5]
        to3 = torch.exp(-1.0 * tmp2) * self._cfc_exact_part(o, 6, K - 1)[:, 0]
        st["to3cnt"] = _set(st["to3cnt"], (slice(None), K), to3)

    def _ot_diag(self, atm, o):
        press, pflux = 10.0 * atm["press"], 10.0 * atm["pflux"]
        K = o["var2"].shape[1]
        pdfinv = 1.0 / (pflux[:, 1:] - pflux[:, :-1])
        delpr1 = pdfinv[:, 1:K] * (press[:, 1:K] - pflux[:, 1:K])
        delpr2 = pdfinv[:, 0:K] * (pflux[:, 1:K + 1] - press[:, 0:K])
        poly = lambda c: c * (-0.5e+00 + c * (0.166666e+00 - c * 0.416666e-01))

        def one(nb):
            x = o["xch2obd"][..., nb - 1]
            csuba = DIFFAC * x[:, 1:K] * delpr1
            csubb = DIFFAC * x[:, 0:K - 1] * delpr2[:, 0:K - 1]
            ca, cb = poly(csuba), poly(csubb)
            return torch.cat([1.0 + 0.5 * (ca + cb), 1.0 + cb[:, -1:]], dim=1)     # levels 1..K

        return [one(4), one(5), one(7)], one(6)

    # ------------------------------------------------------------------ clouds
    @staticmethod
    def _cldtau(cld):
        cmx, crn = cld["cmxolw"], cld["crndlw"]
        taum = 1.0 - cld["emmxolw"]
        r = (crn / (1.0 - cmx))[..., None]
        taur = torch.where((crn > 0.0)[..., None], r * (1.0 - cld["emrndlw"]) + 1.0 - r,
                           torch.ones_like(cld["emrndlw"]))
        return taum, taur

    @staticmethod
    def _cloud(kl, cmx, taum, taur, cldtf):
        """cloud(kl) with kl 0-based; taunbl_mxolw = 0. Returns the updated cldtf."""
        K = cmx.shape[1]
        nb = taum.shape[-1]
        zero = torch.zeros_like(taum[:, 0])
        mo = zero
        rd = torch.ones_like(zero)
        if 0 < kl < K:
            m = ((cmx[:, kl - 1] != 0.0) & (cmx[:, kl] == cmx[:, kl - 1]))[:, None]
            mo = torch.where(m, cmx[:, kl, None] * zero, mo)
            rd = torch.where(m, (1.0 - cmx[:, kl])[:, None].expand_as(rd), rd)
        cols = {kl: mo + rd}
        if kl < K:
            mo = cmx[:, kl, None] * taum[:, kl]
            rd = (1.0 - cmx[:, kl])[:, None] * taur[:, kl]
            cols[kl + 1] = mo + rd
            for kp in range(kl + 2, K + 1):
                m = ((cmx[:, kp - 2] == 0.0) | (cmx[:, kp - 2] != cmx[:, kp - 1]))[:, None]
                prev = cols[kp - 1]
                mo_a = prev * cmx[:, kp - 1, None] * taum[:, kp - 1]
                rd_a = prev * (1.0 - cmx[:, kp - 1])[:, None] * taur[:, kp - 1]
                mo_b = mo * taum[:, kp - 1]
                rd_b = rd * taur[:, kp - 1]
                mo = torch.where(m, mo_a, mo_b)
                rd = torch.where(m, rd_a, rd_b)
                cols[kp] = mo + rd
        lo = kl
        new = torch.stack([cols[j] for j in range(lo, K + 1)], dim=1)
        return torch.cat([cldtf[:, :lo], new], dim=1)

    # ------------------------------------------------------------------ main
    def __call__(self, atm: dict, qo3, gases: dict, aer_tot, aer_ke, cld: dict):
        """atm: press, pflux, temp, tflux (ncol, K+1) [Pa, K], rh2o (ncol, K); qo3 (ncol, K);
        gases: rrvf11, rrvf12, rrvf113, rrvf22; aer_tot (ncol, K+1, 9), aer_ke (ncol,) from
        AerosolOptics.lw_optical_depths; cld: cmxolw, crndlw (ncol, K), emmxolw, emrndlw
        (ncol, K, 7). Returns heatra (K/day), flxnet (W/m2) and the clear-sky heatracf, flxnetcf."""
        d = self.d
        press_mks, pflux_mks = atm["press"], atm["pflux"]
        temp, tflux = atm["temp"], atm["tflux"]
        N, K = atm["rh2o"].shape
        dt, dev = temp.dtype, temp.device
        t = self._t(dt, dev)
        tcf = self.do_totcld_forcing

        o = self._optical_path(atm, qo3, gases, aer_tot, aer_ke)
        gt = self._co2coef(atm, t)
        # co2_source_calc
        ixoe1, dte1 = _loc(temp, TEMP_MIN, TEMP_INC, 28)
        sorc = torch.stack([t["tabsr"].col(ixoe1, dte1, n - 1) for n in range(9 + OFFSET, NBLY)], dim=-1)
        t2 = temp * temp
        t4 = t2 * t2
        source_band = torch.stack([t4, sorc[..., 0] + sorc[..., 1] + sorc[..., 2], sorc[..., 3], sorc[..., 4],
                                   sorc[..., 5], sorc[..., 6], t4], dim=-1)
        dsrcdp = torch.cat([torch.zeros_like(source_band[:, :1]), source_band[:, 1:] - source_band[:, :-1]], dim=1)

        taum, taur = self._cldtau(cld)
        cmx = cld["cmxolw"]
        ones = torch.ones(N, K + 1, dtype=dt, device=dev)
        st = dict(co21c=ones, co21r=ones, tch4n2oe=ones, n2o9c=ones, tn2o17=ones, overod=ones, to3cnt=ones,
                  avephi=ones, avephif=ones, tcfc8=ones)
        cldtf = torch.ones(N, K + 1, NBANDS, dtype=dt, device=dev)
        fluxn = torch.zeros(N, K + 1, NBANDS, dtype=dt, device=dev)
        fluxncf = torch.zeros_like(fluxn)
        ci = list(range(NBANDS))                               # cld_indx - 1

        self._transcolrow(gt, t, 0, st)
        self._ot_from_ks(o, st)
        cnt1, cnt2, cnt3 = st["cnttaub"]
        cldtf = self._cloud(0, cmx, taum, taur, cldtf)
        cts_sum, cts_sumcf, gxcts, gxctscf = self._cool_to_space_exact(atm, o, gt, t, st, sorc, cldtf,
                                                                        ixoe1, dte1)
        tb1, tb2, e1ctw1, e1ctw2, flx1e1, flx1e1cf = self._e1e290(atm, o, st, t, t4, cldtf)
        tb1 = tb1.clone(); tb2 = tb2.clone()
        tb1[:, :, 1] = st["co21r"] * st["overod"]
        tb1[:, :, 2], tb1[:, :, 3], tb1[:, :, 4], tb1[:, :, 5] = cnt1, cnt2, st["to3cnt"], cnt3
        tb2[:, 1:, 1] = st["co21c"][:, 1:] * st["overod"][:, 1:]
        tb2[:, 1:, 2:6] = tb1[:, 1:, 2:6]

        # longwave_fluxes_ks
        sk = source_band[:, 0]
        first = torch.stack([sk[:, m] * tb1[:, 0, m] if (m == 0 or m >= 6) else sk[:, m]
                             for m in range(NBANDS)], dim=-1)
        ft = sk[:, None, :] * tb1[:, 1:]
        ft2 = dsrcdp[:, 1:] * tb2[:, 1:]
        cl = cldtf[:, 1:][..., ci]
        fluxn = fluxn + torch.cat([(first + (ft2 * cl).sum(dim=1))[:, None], ft * cl], dim=1)
        if tcf:
            fluxncf = fluxncf + torch.cat([(first + ft2.sum(dim=1))[:, None], ft], dim=1)

        # cool_to_space_approx (trans_band1(:,:,:,1) = e1ctw2)
        tb1 = tb1.clone()
        tb1[:, :K, 0] = e1ctw2
        cts_out, cts_outcf = self._cool_to_space_approx(pflux_mks, source_band, tb1, cldtf, e1ctw1)

        for k in range(1, K - 1):
            self._transcolrow(gt, t, k, st)
            self._ot_k_down(o, st, k)
            cldtf = self._cloud(k, cmx, taum, taur, cldtf)
            tb1, tb2 = self._e290(atm, o, st, k, tb1, tb2)
            tb1 = tb1.clone(); tb2 = tb2.clone()
            tb1[:, k + 1:, 2] = cnt1[:, k + 1:] / cnt1[:, k:k + 1]
            tb1[:, k + 1:, 3] = cnt2[:, k + 1:] / cnt2[:, k:k + 1]
            tb1[:, k + 1:, 5] = cnt3[:, k + 1:] / cnt3[:, k:k + 1]
            tb1[:, k + 1:, 1] = st["co21r"][:, k + 1:] * st["overod"][:, k + 1:]
            tb1[:, k + 1:, 4] = st["to3cnt"][:, k + 1:]
            tb2[:, k + 1:, 1] = st["co21c"][:, k + 1:] * st["overod"][:, k + 1:]
            tb2[:, k + 1:, 2:6] = tb1[:, k + 1:, 2:6]
            # longwave_fluxes_k_down (source = dsrcdp_band)
            cl = cldtf[:, k + 1:][..., ci]
            ft = dsrcdp[:, k, None, :] * tb1[:, k + 1:]
            f4 = (dsrcdp[:, k + 1:] * tb2[:, k + 1:])
            add = torch.cat([(f4 * cl).sum(dim=1)[:, None], ft * cl], dim=1)
            fluxn = torch.cat([fluxn[:, :k], fluxn[:, k:] + add], dim=1)
            if tcf:
                addcf = torch.cat([f4.sum(dim=1)[:, None], ft], dim=1)
                fluxncf = torch.cat([fluxncf[:, :k], fluxncf[:, k:] + addcf], dim=1)

        # level KE
        self._transcolrow(gt, t, K - 1, st)
        self._ot_ke(o, st)
        b1 = [None] * NBANDS
        b1[2] = cnt1[:, K] / cnt1[:, K - 1]
        b1[3] = cnt2[:, K] / cnt2[:, K - 1]
        b1[5] = cnt3[:, K] / cnt3[:, K - 1]
        cldtf = self._cloud(K - 1, cmx, taum, taur, cldtf)
        emspec, emspecf = self._esfc(atm, o, st, t)
        co21c_ke, co21r_ke = self._trans_sfc(atm, gt, st["overod"])
        b1[0], b1[1], b1[4], b1[6] = emspec[:, 1], co21c_ke, st["to3cnt"][:, K], emspecf[:, 1]
        b2 = [emspec[:, 0], co21r_ke, b1[2], b1[3], b1[4], b1[5], emspecf[:, 0]]
        b1, b2 = torch.stack(b1, dim=-1), torch.stack(b2, dim=-1)
        cl = cldtf[:, K][..., ci]
        f1 = dsrcdp[:, K] * b1
        f2 = dsrcdp[:, K - 1] * b2
        fluxn = torch.cat([fluxn[:, :K - 1], (fluxn[:, K - 1] + f1 * cl)[:, None], (fluxn[:, K] + f2 * cl)[:, None]],
                          dim=1)
        if tcf:
            fluxncf = torch.cat([fluxncf[:, :K - 1], (fluxncf[:, K - 1] + f1)[:, None],
                                 (fluxncf[:, K] + f2)[:, None]], dim=1)

        emisdg, emisdgf = self._enear(atm, o, st, t)
        contdg, to3dg = self._ot_diag(atm, o)
        cldtf = self._cloud(K, cmx, taum, taur, cldtf)
        co21c = self._trans_nearby(atm, gt, st["overod"], st["co21c"])
        tbd = torch.stack([emisdg, co21c[:, 1:], contdg[0], contdg[1], to3dg, contdg[2], emisdgf], dim=-1)
        ft = dsrcdp[:, 1:] * tbd
        fluxn = torch.cat([fluxn[:, :1], fluxn[:, 1:] + ft * cldtf[:, 1:][..., ci]], dim=1)
        if tcf:
            fluxncf = torch.cat([fluxncf[:, :1], fluxncf[:, 1:] + ft], dim=1)

        flx = fluxn[..., 0]
        for m in range(1, NBANDS):
            flx = flx + fluxn[..., m]
        pdfinv = 1.0 / (pflux_mks[:, 1:] - pflux_mks[:, :-1])
        heatem = (RADCON_MKS * (flx[:, 1:] - flx[:, :-1]) * pdfinv) * 1.0e-03
        cs = cts_sum
        for m in (1, 4, 0, 2, 3, 5):
            cs = cs - cts_out[..., m]
        heatra = heatem + cs
        pdflux = pflux_mks[:, 1:] - pflux_mks[:, :-1]
        tmp1 = 1.0e03 * heatra * pdflux / RADCON_MKS
        flxnet = torch.cumsum(torch.cat([(flx1e1 + gxcts)[:, None], tmp1], dim=1), dim=1)
        out = dict(heatra=heatra, flxnet=1.0e-03 * flxnet, flx=flx, fluxn=fluxn, cts_sum=cs, heatem=heatem)
        if tcf:
            flxcf = fluxncf[..., 0]
            for m in range(1, NBANDS):
                flxcf = flxcf + fluxncf[..., m]
            heatemcf = RADCON_MKS * (flxcf[:, 1:] - flxcf[:, :-1]) * pdfinv * 1.0e-03
            cscf = cts_sumcf
            for m in (1, 4, 0, 2, 3, 5):
                cscf = cscf - cts_outcf[..., m]
            heatracf = heatemcf + cscf
            tmp1 = 1.0e03 * heatracf * pdflux / RADCON_MKS
            flxnetcf = torch.cumsum(torch.cat([(flx1e1cf + gxctscf)[:, None], tmp1], dim=1), dim=1)
            out.update(heatracf=heatracf, flxnetcf=1.0e-03 * flxnetcf, flxcf=flxcf, fluxncf=fluxncf)
        return out

    # ------------------------------------------------------------------ cool-to-space
    def _cool_to_space_exact(self, atm, o, gt, t, st, sorc, cldtf, ixoe1, dte1):
        press, pflux = 10.0 * atm["press"], 10.0 * atm["pflux"]
        temp = atm["temp"]
        K = gt["K"]
        pdfinv2 = 1.0 / (pflux[:, 1:] - pflux[:, :-1])
        pdfinv = 1.0 / (atm["pflux"][:, 1:] - atm["pflux"][:, :-1])
        x = temp[:, :K] - 2.5e02
        y = x * x
        aer = o["totaerooptdep"]
        excts = torch.zeros_like(x)
        exctscf = torch.zeros_like(x)
        gxcts = torch.zeros_like(x[:, 0])
        gxctscf = torch.zeros_like(x[:, 0])
        exctsn = []
        for n in range(1, NBLY):
            f = 0.44194e-01 * (t["apcm"][n - 1] * x + t["bpcm"][n - 1] * y)
            ff = 0.44194e-01 * (t["atpcm"][n - 1] * x + t["btpcm"][n - 1] * y)
            ag = (1.418191e+00 + f) * f + 1.0
            agg = (1.418191e+00 + ff) * ff + 1.0
            p = ag * ag; p = p * p; p = p * p; p = p * p
            q = agg * agg; q = q * q; q = q * q; q = q * q
            topm = torch.cumsum(o["var1"] * p, dim=1)
            topphi = torch.cumsum(o["var2"] * q, dim=1)
            fac1 = t["acomb"][n - 1] * topm
            fac2 = fac1 * topm / (t["bcomb"][n - 1] * topphi)
            tmp1 = fac1 / torch.sqrt(1.0 + fac2)
            if n <= BAND_NO_END[1]:
                tot = self._totch2o(o, t, n, ixoe1, dte1)
                tt = torch.exp(-1.0 * (tmp1 + DIFFAC * tot[:, 1:]))
            elif n == BAND_NO_START[6]:
                tt = torch.exp(-1.0 * tmp1) * st["to3cnt"][:, 1:]
            else:
                tmp2 = tmp1 + DIFFAC * self._totch2obd(o, n - 40)[:, 1:]
                na = {41: 0, 42: 1, 43: 2, 44: 3, 45: 4, 47: 6}[n]
                tmp2 = tmp2 + aer[:, 1:, na]
                tt = torch.exp(-1.0 * tmp2)
                if n in (41, 42, 43):
                    sp = gt["co2spnb"][:, 1:, n - 41]
                    tt = tt * (sp * st["tn2o17"][:, 1:]) if n == 41 else tt * sp
                elif n == 47:
                    tt = tt * st["n2o9c"][:, 1:]
            if n <= 8 + OFFSET:
                so = t["tabsr"].col(ixoe1, dte1, n - 1)
            else:
                so = sorc[..., n - 9 - OFFSET]
                tt = tt * self._cfc_exact(o, n - 8 - OFFSET)
            pfac1 = 0.5 * pdfinv2[:, K - 1] * (pflux[:, K] - press[:, K - 1]) * tt[:, K - 2]
            pfac2 = 0.5 * pdfinv2[:, K - 1] * (pflux[:, K] + press[:, K - 1] - 2.0 * pflux[:, K - 1]) * tt[:, K - 1]
            fcf = tt[:, K - 1] * so[:, K - 1] + (pfac1 + pfac2) * (so[:, K] - so[:, K - 1])
            c = CLD_INDX_TABLE[n - 1] - 1
            fctsg = cldtf[:, K, c] * fcf
            gxctscf = gxctscf + fcf
            gxcts = gxcts + fctsg
            ctmp = torch.cat([torch.ones_like(tt[:, :1]), tt * cldtf[:, 1:, c]], dim=1)
            en = so[:, :K] * (ctmp[:, 1:] - ctmp[:, :-1])
            encf = so[:, :K] * (tt - torch.cat([torch.ones_like(tt[:, :1]), tt[:, :-1]], dim=1))
            excts = excts + en
            exctscf = exctscf + encf
        for k in range(K):
            gxcts = gxcts - excts[:, k]
            gxctscf = gxctscf - exctscf[:, k]
        excts = (excts * RADCON_MKS * pdfinv) * 1.0e-03
        exctscf = (exctscf * RADCON_MKS * pdfinv) * 1.0e-03
        return excts, exctscf, gxcts, gxctscf

    def _cool_to_space_approx(self, pflux, source, trans, cldtf, trans2):
        K = pflux.shape[1] - 1
        pdfinv = 1.0 / (pflux[:, 1:] - pflux[:, :-1])
        out, outcf = [], []
        for m in range(NBANDS - NBTRGE):
            if m == 0:
                a, b = trans[:, :K, 0], trans2[:, :K]
            else:
                a, b = trans[:, 1:, m], trans[:, :K, m]
            out.append(RADCON_MKS * pdfinv * source[:, :K, m] * (a * cldtf[:, 1:, m] - b * cldtf[:, :K, m]) * 1.0e-03)
            outcf.append(RADCON_MKS * pdfinv * source[:, :K, m] * (a - b) * 1.0e-03)
        return torch.stack(out, dim=-1), torch.stack(outcf, dim=-1)

    # ------------------------------------------------------------------ H2O emissivity lookups
    def _temp_index(self, atm):
        temp, tflux = atm["temp"], atm["tflux"]
        K = temp.shape[1] - 1
        ixoe1, dte1 = _loc(temp, TEMP_MIN, TEMP_INC, 28)
        ixoe2, dte2 = _loc(tflux, TEMP_MIN, TEMP_INC, 28)
        ixoe2 = torch.cat([ixoe2[:, 1:], ixoe1[:, K - 1:K]], dim=1)
        dte2 = torch.cat([dte2[:, 1:], dte1[:, K - 1:K]], dim=1)
        return ixoe1, dte1, ixoe2, dte2

    @staticmethod
    def _mass_index(x):
        iy, dy = _loc(torch.log10(x), MASS_MIN, MASS_INC, 10 ** 6)
        return iy + 1, dy

    def _e1e290(self, atm, o, st, t, t4, cldtf):
        K = o["var2"].shape[1]
        ixoe1, dte1, ixoe2, dte2 = self._temp_index(atm)
        iyo, du = self._mass_index(st["avephi"])
        tb2_1 = t["tab2"](ixoe2, iyo, dte2, du)
        tb2_1 = torch.cat([tb2_1[:, :1], tb2_1[:, :K - 1], 0.5 * (tb2_1[:, K - 1:K] + tb2_1[:, K:])], dim=1)
        iyop = torch.cat([torch.ones_like(iyo[:, :1]), iyo[:, :K]], dim=1)
        dup = torch.cat([torch.zeros_like(du[:, :1]), du[:, :K]], dim=1)
        ixo1 = ixoe1[:, :1].expand_as(ixoe1)
        dt1 = dte1[:, :1].expand_as(dte1)
        e1cts1 = t["tab1"](ixoe1, iyop, dte1, dup)
        e1cts2 = t["tab1"](ixoe1[:, :K], iyo[:, :K], dte1[:, :K], du[:, :K])
        tb1_1 = t["tab1"](ixo1, iyop, dt1, dup)
        e1ctw1 = t["tab1w"](ixoe1, iyop, dte1, dup)
        e1ctw2 = t["tab1w"](ixoe1[:, :K], iyo[:, :K], dte1[:, :K], du[:, :K])
        iyo, du = self._mass_index(st["avephif"])
        iyop = torch.cat([torch.ones_like(iyo[:, :1]), iyo[:, :K]], dim=1)
        dup = torch.cat([torch.zeros_like(du[:, :1]), du[:, :K]], dim=1)
        tb2_7 = t["tab2a"](ixoe2, iyo, dte2, du, 0)
        e1cts1f = t["tab1a"](ixoe1, iyop, dte1, dup, 0)
        e1cts2f = t["tab1a"](ixoe1[:, :K], iyo[:, :K], dte1[:, :K], du[:, :K], 0)
        tb1_7 = t["tab1a"](ixo1, iyop, dt1, dup, 0)
        tb2_7 = torch.cat([tb2_7[:, :1], tb2_7[:, :K - 1], 0.5 * (tb2_7[:, K - 1:K] + tb2_7[:, K:])], dim=1)
        e = st["tch4n2oe"]
        tb1_7 = torch.cat([tb1_7[:, :1], tb1_7[:, 1:] * e[:, 1:]], dim=1)
        e1cts1f = e1cts1f * e
        e1cts2f = e1cts2f * e[:, 1:]
        tb2_7 = torch.cat([tb2_7[:, :1], tb2_7[:, 1:] * e[:, 1:]], dim=1)
        tcfc8 = 1.0 - STRF113[7] * o["totf113"] - STRF22[7] * o["totf22"]
        st["tcfc8"] = tcfc8
        tb1_7 = torch.cat([tb1_7[:, :1], tb1_7[:, 1:] * tcfc8[:, 1:]], dim=1)
        e1cts1f = e1cts1f * tcfc8
        e1cts2f = e1cts2f * tcfc8[:, 1:]
        tb2_7 = torch.cat([tb2_7[:, :1], tb2_7[:, 1:] * tcfc8[:, 1:]], dim=1)
        taero8 = torch.exp(-1.0 * o["totaerooptdep"][..., 7])
        tb1_7 = torch.cat([tb1_7[:, :1], tb1_7[:, 1:] * taero8[:, 1:]], dim=1)
        e1cts1f = e1cts1f * taero8
        e1cts2f = e1cts2f * taero8[:, 1:]
        tb2_7 = torch.cat([tb2_7[:, :1], tb2_7[:, 1:] * taero8[:, 1:]], dim=1)

        s1a = t4[:, K] * (e1cts1[:, K] - e1ctw1[:, K])
        flx1e1 = s1a * cldtf[:, K, 0]
        tmp1 = t4[:, :K] * (e1cts1[:, :K] - e1ctw1[:, :K])
        tmp2 = t4[:, :K] * (e1cts2 - e1ctw2)
        for k in range(K):
            flx1e1 = flx1e1 + tmp1[:, k] * cldtf[:, k, 0] - tmp2[:, k] * cldtf[:, k + 1, 0]
        flx1e1cf = s1a
        for k in range(K):
            flx1e1cf = flx1e1cf + tmp1[:, k] - tmp2[:, k]
        s1a = t4[:, K] * e1cts1f[:, K]
        flx1e1f = s1a * cldtf[:, K, 6]
        tmp1 = t4[:, :K] * e1cts1f[:, :K]
        tmp2 = t4[:, :K] * e1cts2f
        for k in range(K):
            flx1e1f = flx1e1f + tmp1[:, k] * cldtf[:, k, 6] - tmp2[:, k] * cldtf[:, k + 1, 6]
        flx1e1 = flx1e1 + flx1e1f
        f = s1a
        for k in range(K):
            f = f + tmp1[:, k] - tmp2[:, k]
        flx1e1cf = flx1e1cf + f
        N = t4.shape[0]
        z = torch.zeros(N, K + 1, dtype=t4.dtype, device=t4.device)
        tb1 = torch.stack([tb1_1, z, z, z, z, z, tb1_7], dim=-1)
        tb2 = torch.stack([tb2_1, z, z, z, z, z, tb2_7], dim=-1)
        return tb1, tb2, e1ctw1, e1ctw2, flx1e1, flx1e1cf

    def _e290(self, atm, o, st, k, tb1, tb2):
        K = o["var2"].shape[1]
        ixoe1, dte1, ixoe2, dte2 = self._temp_index(atm)
        ixok = ixoe2[:, k - 1:k].expand(-1, K - k)
        dtk = dte2[:, k - 1:k].expand(-1, K - k)
        out1, out2 = [], []
        for m, (tab, av) in enumerate(((self._t(atm["temp"].dtype, atm["temp"].device)["tab2"], st["avephi"]),
                                       (self._t(atm["temp"].dtype, atm["temp"].device)["tab2a"], st["avephif"]))):
            iyo, du = self._mass_index(av[:, k:])
            mm = None if m == 0 else 0
            b2 = tab(ixoe2[:, k:], iyo, dte2[:, k:], du, mm)                       # levels k..K
            b1 = tab(ixok, iyo[:, :K - k], dtk, du[:, :K - k], mm)               # levels k..K-1
            b2 = torch.cat([b2[:, :-1], 0.5 * (b2[:, -2:-1] + b2[:, -1:])], dim=1)
            b2 = torch.cat([b2[:, :1], b2[:, :K - k - 1], b2[:, -1:]], dim=1)    # shift k+1..K-1
            out1.append(b1); out2.append(b2)
        # band 1: trans_band1(k+1..KE+1) = lookups at k..KE; trans_band2(k..KE+1)
        t1_1 = torch.cat([tb1[:, :k + 1, 0], out1[0]], dim=1)
        t2_1 = torch.cat([tb2[:, :k, 0], out2[0]], dim=1)
        t1_7 = out1[1]
        t2_7 = out2[1][:, 1:]
        e = st["tch4n2oe"][:, k + 1:]
        tc = self._cfc_indx8_part(o, k)
        st["tcfc8"] = torch.cat([st["tcfc8"][:, :k + 1], tc], dim=1)
        taero8 = torch.exp(-1.0 * o["totaerooptdep"][..., 7])
        ta = taero8[:, k + 1:] / taero8[:, k:k + 1]
        t1_7 = t1_7 * e * tc * ta
        t2_7 = t2_7 * e * tc * ta
        t1_7 = torch.cat([tb1[:, :k + 1, 6], t1_7], dim=1)
        t2_7 = torch.cat([tb2[:, :k, 6], out2[1][:, :1], t2_7], dim=1)
        tb1 = torch.cat([t1_1[..., None], tb1[..., 1:6], t1_7[..., None]], dim=-1)
        tb2 = torch.cat([t2_1[..., None], tb2[..., 1:6], t2_7[..., None]], dim=-1)
        return tb1, tb2

    def _esfc(self, atm, o, st, t):
        K = o["var2"].shape[1]
        temp, tflux = atm["temp"], atm["tflux"]
        ixoe1, dte1, ixoe2, dte2 = self._temp_index(atm)
        ixsp = torch.stack([ixoe2[:, K - 2], ixoe1[:, K - 2]], dim=1)
        dxsp = torch.stack([dte2[:, K - 2], dte1[:, K - 2]], dim=1)
        v2 = o["var2"][:, K - 1]
        iy, dy = self._mass_index(torch.stack([v2, v2 + o["empl1"][:, K - 1]], dim=1))
        emiss = t["tab2"](ixsp, iy, dxsp, dy)
        vf = o["vrpfh2o"][:, K - 1]
        iy, dy = self._mass_index(torch.stack([vf, vf + o["empl1f"][:, K - 1]], dim=1))
        emissf = t["tab2a"](ixsp, iy, dxsp, dy, 0)
        emd1, emd1f, emd2, emd2f = self._emd(atm, o, t)
        e1, e1f, e2, e2f = o["empl1"], o["empl1f"], o["empl2"], o["empl2f"]
        emspec = torch.stack([(emd1[:, 0] * e1[:, 0] - emd1[:, K] * e1[:, K]) / o["emx1"]
                              + 0.25 * (emiss[:, 0] + emiss[:, 1]),
                              2.0 * (emd1[:, 0] * e1[:, 0] - emd2[:, K - 1] * e2[:, K]) / o["emx2"]], dim=1)
        emspecf = torch.stack([(emd1f[:, 0] * e1f[:, 0] - emd1f[:, K] * e1f[:, K]) / o["emx1f"]
                               + 0.25 * (emissf[:, 0] + emissf[:, 1]),
                               2.0 * (emd1f[:, 0] * e1f[:, 0] - emd2f[:, K - 1] * e2f[:, K]) / o["emx2f"]], dim=1)
        emspecf = emspecf * st["tch4n2oe"][:, K:K + 1]
        tc = self._cfc_indx8_part(o, K - 1)
        st["tcfc8"] = torch.cat([st["tcfc8"][:, :K], tc], dim=1)
        emspecf = emspecf * tc
        return emspec, emspecf

    def _emd(self, atm, o, t):
        """emd1, emd1f (levels 0..K) and emd2, emd2f (levels 1..K) of esfc / enear."""
        temp, tflux = atm["temp"], atm["tflux"]
        K = o["var2"].shape[1]
        tpl1 = torch.cat([temp[:, K - 1:K], tflux[:, 1:K], 0.5 * (tflux[:, K:] + temp[:, K - 1:K])], dim=1)
        tpl2 = torch.cat([tflux[:, 1:K], 0.5 * (tflux[:, K - 1:K] + temp[:, K - 1:K])], dim=1)
        ix1, dx1 = _loc(tpl1, TEMP_MIN, TEMP_INC, 28)
        ix2, dx2 = _loc(tpl2, TEMP_MIN, TEMP_INC, 28)
        iy, dy = self._mass_index(o["empl1"])
        emd1 = t["tab3"](ix1, iy, dx1, dy)
        iy, dy = self._mass_index(o["empl1f"])
        emd1f = t["tab3a"](ix1, iy, dx1, dy, 0)
        iy, dy = self._mass_index(o["empl2"][:, 1:])
        emd2 = t["tab3"](ix2, iy, dx2, dy)
        iy, dy = self._mass_index(o["empl2f"][:, 1:])
        emd2f = t["tab3a"](ix2, iy, dx2, dy, 0)
        return emd1, emd1f, emd2, emd2f

    def _enear(self, atm, o, st, t):
        K = o["var2"].shape[1]
        emd1, emd1f, emd2, emd2f = self._emd(atm, o, t)
        emisdg = torch.cat([emd2[:, :K - 1] + emd1[:, 1:K], 2.0 * emd1[:, K:]], dim=1)          # levels 1..K
        emisdgf = torch.cat([emd2f[:, :K - 1] + emd1f[:, 1:K], 2.0 * emd1f[:, K:]], dim=1)
        emisdgf = emisdgf * st["tch4n2oe"][:, 1:]
        tc = self._cfc_indx8_part(o, K - 1)
        st["tcfc8"] = torch.cat([st["tcfc8"][:, :K], tc], dim=1)
        emisdgf = emisdgf * st["tcfc8"][:, 1:]
        return emisdg, emisdgf
