"""CO2, CH4 and N2O transmission-function tables for sealw99 on the model's reference
pressures (lw_gases_stdtf.F90, siena_201207; gas_tf.F90 put_*_for_gas_tf), numpy float64.

The line-by-line transmissions on 496 standard levels (INPUT/cns_<vmr>_<band>[.nc]) are
fitted (coeint), interpolated in mixing ratio between the two bracketing standard amounts
(rctrns), and interpolated to the model's reference levels (lvl), layers (lyr, 7-point
Simpson in the layer) and the nearby-layer means (51 points) for the two reference surface
pressures 1013.25 hPa ("51"/"10") and 810.6 hPa ("58"/"8"). Only the configuration used
by the SCM cases is supported: NSTDCO2LVLS = 496, interp_form 'log' does not enter here,
do_ch4lbltmpint = do_n2olbltmpint = false. Loops run over vectors of (hi, lo) pressure pairs.
"""
from __future__ import annotations

import os

import numpy as np

NSTD = 496
DOP_CORE0 = 25.0
CO2_STD = (0., 165.0, 300.0, 330.0, 348.0, 356.0, 360.0, 600.0, 660.0, 1320.0, 1600.0)
CH4_STD = (0., 300., 700., 1250., 1750., 2250., 2800., 4000.)
N2O_STD = (0., 180., 275., 310., 340., 375., 500.)
CO2_FILES = [tuple(f"cns_{v}_{b}" for b in ("490850", "490630", "630700", "700850", "43um"))
             for v in (0, 165, 300, 330, 348, 356, 360, 600, 660, 1320, 1600)]
CH4_FILES = [(f"cns_{v}_12001400",) for v in (0, 300, 700, 1250, 1750, 2250, 2800, 4000)]
N2O_FILES = [(f"cns_{v}_12001400", f"cns_{v}_10701200", f"cns_{v}_560630")
             for v in (0, 180, 275, 310, 340, 375, 500)]
NTBND = dict(co2=(3, 3, 3, 3, 1), ch4=(3,), n2o=(3, 3, 3))
DOP_NUM = dict(co2=(None, 560., 665., 775., 2325.), ch4=(1300.,), n2o=(1300., 1135., 595.))
C12 = dict(co2=((1.5, 0.5), (2.0, 1.0), (1.0, 0.0), (1.5, 0.5), (1.0, 0.0)), ch4=((1.0, 0.0),),
           n2o=((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))


def _wgt_lyr():
    w = np.zeros(7)
    w[0] = w[6] = 1. / 18.
    for n in range(1, 4):
        w[2 * n - 1] = 4. / 18.
    for n in range(1, 3):
        w[2 * n] = 2. / 18.
    return w


def _wgt_near():
    w = np.zeros(51)
    w[0] = w[50] = 1. / 150.
    for n in range(1, 26):
        w[2 * n - 1] = 4. / 150.
    for n in range(1, 25):
        w[2 * n] = 2. / 150.
    return w


class LblTransmission:
    """lw_gases_stdtf for one set of reference pressures pref (nlev+1, 2) [Pa]."""

    def __init__(self, data_dir: str, pref: np.ndarray):
        self.dir = data_dir
        pa = []
        for line in open(os.path.join(data_dir, "stdlvls")).read().splitlines()[:NSTD]:
            pa.append(float(line.split()[0]))
        self.pa = np.array(pa)
        pref = np.asarray(pref, float)
        kmax = pref.shape[0] - 1
        self.K = kmax
        pd, pd8 = pref[:, 0].copy(), pref[:, 1].copy()
        plm, plm8 = np.zeros(kmax + 1), np.zeros(kmax + 1)
        for k in range(1, kmax):
            plm[k] = 0.5 * (pd[k - 1] + pd[k])
            plm8[k] = 0.5 * (pd8[k - 1] + pd8[k])
        plm[kmax], plm8[kmax] = pd[kmax], pd8[kmax]
        self.pd, self.pd8 = pd * 1.0e-02, pd8 * 1.0e-02
        self.plm, self.plm8 = plm * 1.0e-02, plm8 * 1.0e-02
        self.kp_std, self.k_std = np.tril_indices(NSTD, -1)      # kp > k
        self.wlyr, self.wnear = _wgt_lyr(), _wgt_near()

    # ------------------------------------------------------------------ input
    def _read(self, name, nt):
        p = os.path.join(self.dir, name)
        if os.path.exists(p + ".nc"):
            import netCDF4
            with netCDF4.Dataset(p + ".nc") as d:
                v = np.asarray(d["trns_std_nf"][:nt], float)
            return [v[i].T.copy() for i in range(nt)]
        raw = np.fromfile(p, dtype="<f8", count=nt * NSTD * NSTD)
        return [raw[i * NSTD * NSTD:(i + 1) * NSTD * NSTD].reshape(NSTD, NSTD, order="F") for i in range(nt)]

    # ------------------------------------------------------------------ fits
    def _coeint(self, trns, dop):
        pa = self.pa
        pa2 = pa * pa
        sexp = .505 + 2.0e-5 * pa + .035 * (pa2 - .25) / (pa2 + .25)
        uexp = sexp * (1.0 + 0.33 * pa2 / (pa2 + 40000.))
        k = np.arange(2, NSTD)
        s, u = sexp[k], uexp[k]
        upathb = np.exp((1. / s) * np.log(pa[k] - pa[k - 1])) * (pa[k] + pa[k - 1] + dop)
        upatha = np.exp((1. / s) * np.log(pa[k] - pa[k - 2])) * (pa[k] + pa[k - 2] + dop)
        t2, t1 = trns[k, k - 2], trns[k, k - 1]
        r = (1.0 - t2) / (1.0 - t1)
        rexp = np.exp((u / s) * np.log(r))
        upatha = np.exp(u * np.log(upatha))
        upathb = np.exp(u * np.log(upathb))
        xx = 2.0 * (upathb * rexp - upatha) / (upathb * upathb * rexp - upatha * upatha)
        ca = np.zeros(NSTD); xa = np.zeros(NSTD)
        cak, xak = np.zeros(len(k)), np.zeros(len(k))
        with np.errstate(all="ignore"):
            for _ in range(20):
                ft1 = xx * upatha
                ft2 = xx * upathb
                m1 = ft1 <= 1.0e-10
                m2 = ~m1 & (ft2 >= 1.0e+8)
                m3 = ~m1 & ~m2
                xak = np.where(m1, 1.0, xak)
                cak = np.where(m1, np.exp((u / s) * np.log(1.0 - t2)) / upatha, cak)
                xxlog = (np.log(upatha) - rexp * np.log(upathb)) / (rexp - 1.0)
                xak = np.where(m2, np.exp(xxlog), xak)
                cak = np.where(m2, np.exp((u / s) * np.log(1.0 - t2)) / (xxlog + np.log(upatha)), cak)
                f1 = np.log(1.0 + xx * upatha)
                f2 = np.log(1.0 + xx * upathb)
                f = f1 / f2 - rexp
                fp = (f2 * upatha / (1.0 + xx * upatha) - f1 * upathb / (1.0 + xx * upathb)) / (f2 * f2)
                xx = np.where(m3, xx - f / fp, xx)
            m = (ft1 > 1.0e-10) & (ft2 < 1.0e+8)
            cak = np.where(m, (1.0 - t2) ** (u / s) / (np.log(1.0 + xx * upatha) + 1.0e-20), cak)
            xak = np.where(m, xx, xak)
        ca[2:], xa[2:] = cak, xak
        ca[:2], xa[:2] = ca[2], xa[2]
        return ca, sexp, xa, uexp

    def _index(self, p):
        """1-based bracketing index: pa(i) <= p < pa(i+1), limited to 1..NSTD-1."""
        return np.clip(np.searchsorted(self.pa, p, side="right"), 1, NSTD - 1)

    def _intcoef(self, p, co):
        """intcoef_2d / intcoef_2d_std at pressures p -> ca, sexp, xa, uexp at p."""
        ca, sexp, xa, uexp = co
        pa = self.pa
        I = self._index(p)
        i = I - 1
        i1, i2 = i + 1, np.minimum(i + 2, NSTD - 1)
        lin = lambda a: a[i] + (a[i1] - a[i]) / (pa[i1] - pa[i]) * (p - pa[i])
        s, u = lin(sexp), lin(uexp)
        caxa = ca * xa
        quad = (I > 2) & (I < NSTD - 1)
        delp = p - pa[i1]

        def q(a):
            d1 = (a[i2] - a[i1]) / (pa[i2] - pa[i1])
            d2 = (a[i1] - a[i]) / (pa[i1] - pa[i])
            b = (d1 - d2) / (pa[i2] - pa[i])
            aa = d1 - b * (pa[i2] - pa[i1])
            return a[i1] + delp * (aa + delp * b)

        with np.errstate(all="ignore"):
            xq, xl = q(xa), lin(xa)
            xh = np.where(quad, np.where(xq <= 0.0, xl, xq), xl)
            prod = np.where(quad, q(caxa), lin(caxa))
        return prod / xh, s, xh, u

    def _interp_error(self, err, phi, plo):
        pa = self.pa
        Ih, Il = self._index(phi), self._index(plo)
        h, l = Ih - 1, Il - 1
        c1 = ((Ih - Il) >= 3) & (Ih < NSTD - 1)
        c2 = ~c1 & (Ih > Il)
        cl = lambda a: np.minimum(a, NSTD - 1)
        E = lambda dr, dc: err[cl(h + dr), cl(l + dc)]
        P = lambda a: pa[cl(a)]
        with np.errstate(all="ignore"):
            dlo = plo - P(l + 1)

            def f(dr):
                d1 = (E(dr, 2) - E(dr, 1)) / (P(l + 2) - P(l + 1))
                d2 = (E(dr, 1) - E(dr, 0)) / (P(l + 1) - P(l))
                b = (d1 - d2) / (P(l + 2) - P(l))
                a = d1 - b * (P(l + 2) - P(l + 1))
                return E(dr, 1) + dlo * (a + dlo * b)

            fk, fk1, fk2 = f(0), f(1), f(2)
            dhi = phi - P(h + 1)
            d1 = (fk2 - fk1) / (P(h + 2) - P(h + 1))
            d2 = (fk1 - fk) / (P(h + 1) - P(h + 0))
            b = (d1 - d2) / (P(h + 2) - P(h))
            a = d1 - b * (P(h + 2) - P(h + 1))
            r1 = fk1 + dhi * (a + dhi * b)
            dlo2 = plo - P(l)
            fk = E(0, 0) + dlo2 * ((E(0, 1) - E(0, 0)) / (P(l + 1) - P(l)))
            fk1 = E(1, 0) + dlo2 * ((E(1, 1) - E(1, 0)) / (P(l + 1) - P(l)))
            r2 = (fk * (P(h + 1) - phi) + fk1 * (phi - P(h))) / (P(h + 1) - P(h))
        return np.where(c1, r1, np.where(c2, r2, 0.0))

    @staticmethod
    def _approx(phi, plo, co, dop):
        ca, sexp, xa, uexp = co
        up = (phi - plo) ** (1. / sexp) * (phi + plo + dop)
        up = up ** uexp
        return (ca * np.log(1.0 + xa * up)) ** (sexp / uexp)

    @staticmethod
    def _approx_std(phi, plo, co, dop):
        ca, sexp, xa, uexp = co
        up = np.exp((1.0 / sexp) * np.log(phi - plo)) * (phi + plo + dop)
        up = np.exp(uexp * np.log(up))
        return np.exp((sexp / uexp) * np.log(ca * np.log(1.0 + xa * up)))

    def _std_error(self, trns, co, dop):
        """error_guess1 on the standard grid (lower triangle; diagonal and upper zero)."""
        kp, k = self.kp_std, self.k_std
        pa = self.pa
        c = tuple(a[kp] for a in co)
        ap = self._approx_std(pa[kp], pa[k], c, dop)
        err = np.zeros((NSTD, NSTD))
        err[kp, k] = 1.0 - trns[kp, k] - ap
        return err

    def _interp_pairs(self, err, co, phi, plo, dop):
        c = self._intcoef(phi, co)
        return 1.0 - (self._interp_error(err, phi, plo) + self._approx(phi, plo, c, dop))

    # ------------------------------------------------------------------ rctrns / gasint
    def _rctrns(self, t_hi, t_lo, vmr, lo, hi, dop):
        co = self._coeint(t_hi, dop)
        err = self._std_error(t_hi, co, dop)
        kp, k = self.kp_std, self.k_std
        pa = self.pa
        out = []
        for v in (vmr, lo):
            ph = ((v + hi) * pa[kp] + (hi - v) * pa[k]) / (2. * hi)
            pl = ((hi - v) * pa[kp] + (v + hi) * pa[k]) / (2. * hi)
            c = self._intcoef(ph, co)
            out.append(1.0 - (self._interp_error(err, ph, pl) + self._approx_std(ph, pl, c, dop)))
        tv = np.eye(NSTD)
        val = out[0] + (hi - vmr) / (hi - lo) * (t_lo[kp, k] - out[1])
        tv[kp, k] = val
        tv[k, kp] = val
        return tv

    def _gasint(self, trns_vmr, dop, do_lyr):
        trns_vmr = trns_vmr.copy()
        np.fill_diagonal(trns_vmr, 1.0)
        co = self._coeint(trns_vmr, dop)
        err = self._std_error(trns_vmr, co, dop)
        K = self.K
        kp, k = np.tril_indices(K + 1, -1)
        res = {}
        for tag, plm, pd in (("10", self.plm, self.pd), ("8", self.plm8, self.pd8)):
            lvl = np.eye(K + 1)
            v = self._interp_pairs(err, co, plm[kp], plm[k], dop)
            lvl[kp, k] = v
            lvl[k, kp] = v
            res["lvl" + tag] = lvl
            if not do_lyr:
                continue
            lyr = np.zeros((K + 1, K + 1))
            lyr[0, 0] = 1.0
            # layers below the top level k (k = 0..K-1), kp = k+1..K
            for nq in range(7):
                ph = pd[kp - 1] + (nq * (pd[kp] - pd[kp - 1])) / 6
                v = self._interp_pairs(err, co, ph, plm[k], dop)
                lyr[kp, k] = lyr[kp, k] + self.wlyr[nq] * v
                if nq == 3:
                    top = k == 0
                    lyr[0, kp[top]] = v[top]
                if nq == 6:
                    lyr[0, K] = v[(kp == K) & (k == 0)][0]
            # layers above, k = 1..K-1, kp = k+1..K
            m = k >= 1
            kp2, k2 = kp[m], k[m]
            for nq in range(7):
                pl = pd[k2 - 1] + (nq * (pd[k2] - pd[k2 - 1])) / 6
                v = self._interp_pairs(err, co, plm[kp2], pl, dop)
                lyr[k2, kp2] = lyr[k2, kp2] + self.wlyr[nq] * v
            # nearby layers, k = 1..K
            phi = np.zeros((51, K + 1)); plo = np.zeros((51, K + 1))
            for kk in range(1, K):
                for n in range(25):
                    plo[n, kk] = pd[kk - 1] + (n * (pd[kk] - pd[kk - 1])) / 50
                    phi[n, kk] = plm[kk]
                plo[25, kk] = plm[kk]
                phi[25, kk] = plm[kk] + 1.0E-13 * plm[kk]
                for n in range(26, 51):
                    phi[n, kk] = pd[kk - 1] + (n * (pd[kk] - pd[kk - 1])) / 50
                    plo[n, kk] = plm[kk]
            for n in range(50):
                plo[n, K] = pd[K - 1] + (n * (pd[K] - pd[K - 1])) / 50
                phi[n, K] = plm[K]
            plo[50, K] = plm[K]
            phi[50, K] = plm[K] + 1.0E-13 * plm[K]
            v = self._interp_pairs(err, co, phi[:, 1:].ravel(), plo[:, 1:].ravel(), dop).reshape(51, K)
            for kk in range(1, K + 1):
                s = lyr[kk, kk]
                for n in range(51):
                    s = s + self.wnear[n] * v[n, kk - 1]
                lyr[kk, kk] = s
            res["lyr" + tag] = lyr
        return res

    # ------------------------------------------------------------------ gases
    def _gas(self, gas, vmr, std, files, tol, lyr_nf):
        if vmr <= std[0] or vmr > std[-1]:
            raise ValueError(f"{gas} volume mixing ratio {vmr} out of range (or zero, unsupported)")
        n = next(i for i in range(len(std) - 1) if std[i] < vmr <= std[i + 1])
        lo, hi = std[n], std[n + 1]
        call = abs(vmr - hi) > tol
        out = []
        for nf, ntb in enumerate(NTBND[gas]):
            t_hi = self._read(files[n + 1][nf], ntb)
            t_lo = self._read(files[n][nf], ntb) if call else None
            num = DOP_NUM[gas][nf]
            dop = DOP_CORE0 if num is None else DOP_CORE0 * num / 670.
            per_nt = []
            for nt in range(ntb):
                tv = self._rctrns(t_hi[nt], t_lo[nt], vmr, lo, hi, dop) if call else t_hi[nt]
                per_nt.append(self._gasint(tv, dop, lyr_nf[nf]))
            out.append(self._gasins(gas, nf, per_nt))
        return out

    @staticmethod
    def _gasins(gas, nf, r):
        c1, c2 = C12[gas][nf]
        o = {}
        for kind in ("lvl", "lyr"):
            for tag in ("10", "8"):
                key = kind + tag
                if key not in r[0]:
                    continue
                o["gasp" + tag + "_" + kind] = c1 * r[0][key] - c2
                if len(r) == 3:
                    o["dgasdt" + tag + "_" + kind] = .02 * (r[1][key] - r[2][key]) * 100.
                    # (sic) the Fortran's 8-profile lvl second derivative uses the 10 profile's nt=1
                    t1 = r[0]["lvl10"] if (kind == "lvl") else r[0][key]
                    o["d2gast" + tag + "_" + kind] = .0016 * (r[1][key] + r[2][key] - 2.0 * t1) * 1000.
        return o

    def tables(self, rrvco2: float, rrvch4: float, rrvn2o: float) -> dict:
        """The gas_tf_mod tables (names as in gas_tf.F90) and stemp/gtemp are not included."""
        K = self.K
        t = {}
        co2 = self._gas("co2", rrvco2 * 1.0e+06, CO2_STD, CO2_FILES, 1.0e-4, (True, False, False, False, True))
        g = co2[0]
        t.update(co251=g["gasp10_lyr"], co258=g["gasp8_lyr"], cdt51=g["dgasdt10_lyr"], cdt58=g["dgasdt8_lyr"],
                 c2d51=g["d2gast10_lyr"], c2d58=g["d2gast8_lyr"])
        k = np.arange(K)
        t.update(co2m51=g["gasp10_lvl"][k, k + 1], co2m58=g["gasp8_lvl"][k, k + 1],
                 cdtm51=g["dgasdt10_lvl"][k, k + 1], cdtm58=g["dgasdt8_lvl"][k, k + 1],
                 c2dm51=g["d2gast10_lvl"][k, k + 1], c2dm58=g["d2gast8_lvl"][k, k + 1])
        for name, tag, src in (("co215nbps1", "10", "gasp"), ("co215nbps8", "8", "gasp"),
                               ("co2dt15nbps1", "10", "dgasdt"), ("co2dt15nbps8", "8", "dgasdt"),
                               ("co2d2t15nbps1", "10", "d2gast"), ("co2d2t15nbps8", "8", "d2gast")):
            t[name] = np.stack([co2[nf][src + tag + "_lvl"][:, 0] for nf in (1, 2, 3)], axis=1)
        t.update(co211=co2[4]["gasp10_lyr"][:, 0], co218=co2[4]["gasp8_lyr"][:, 0])
        ch4 = self._gas("ch4", rrvch4 * 1.0e+09, CH4_STD, CH4_FILES, 1.0e-1, (True,))
        t.update(ch451=ch4[0]["gasp10_lyr"], ch458=ch4[0]["gasp8_lyr"])
        n2o = self._gas("n2o", rrvn2o * 1.0e+09, N2O_STD, N2O_FILES, 1.0e-1, (True, True, True))
        t.update(n2o51=n2o[0]["gasp10_lyr"], n2o58=n2o[0]["gasp8_lyr"], n2o91=n2o[1]["gasp10_lyr"],
                 n2o98=n2o[1]["gasp8_lyr"], n2o71=n2o[2]["gasp10_lyr"], n2o78=n2o[2]["gasp8_lyr"])
        return t
