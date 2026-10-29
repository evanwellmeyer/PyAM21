"""Set-up data of the sealw99 longwave package (siena_201207), numpy float64:

- longwave_tables.F90 (table): H2O emissivity tables tab1, tab2, tab3, tab1w, tab1a, tab2a,
  tab3a on (temperature, log10 path) axes, and the band Planck sources tabsr;
- sealw99_init / optical_path_init / optical_ckd_init: the band coefficients, the CKD2.1
  continuum data, the O3 and 1200-1400 cm-1 H2O coefficients;
- gas_tf.F90 (ptz, antemp): the standard temperature profile stemp and weights gtemp on
  the model's reference pressures, and ixprkminh2o.

Only the configuration of the SCM cases is supported: continuum_form 'ckd2.1',
linecatalog_form 'hitran_2000', no_h2o_bands_1200_1400 = 1, do_lwcldemiss.
Integer powers are written as products, as gfortran evaluates them.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from ..constants import GRAV, RDGAS
from .esfsw_data import _Records

NBLW, NBLX, NBLY, NBCO215 = 300, 48, 48, 3
NTTAB, NUTAB = 28, 181
TEMP_MIN, TEMP_INC = 100.0, 10.0          # rad_utilities temp_1
MASS_MIN, MASS_INC = -16.0, 0.1           # rad_utilities mass_1
OFFSET = 32                               # Lw_parameters%offset for ckd
BAND_NO_START = (1, 25, 41, 42, 43, 44, 46, 47)
BAND_NO_END = (24, 40, 41, 42, 43, 45, 46, 47)
CLD_INDX_TABLE = [1] * 40 + [2, 2, 2, 3, 4, 5, 6]      # do_lwcldemiss
NO_H2O_12001400 = (1, 2, 4, 10, 20)
_vexp = np.vectorize(math.exp, otypes=[float])


@dataclass
class Table:
    vae: np.ndarray
    td: np.ndarray
    md: np.ndarray = None
    cd: np.ndarray = None


def _read_12001400(path, nbtrge):
    r = _Records(path)
    for nb in NO_H2O_12001400:
        vals = [r.read(nb) for _ in range(8)]
        if nb == nbtrge:
            return [np.array(v) for v in vals]
    raise ValueError("NBTRGE is inconsistent with available data")


def _tables(arndm, brndm, bandlo, bandhi, bdlocm, bdhicm, fbdlo, fbdhi, nbtrge):
    """longwave_tables_mod table (ckd: iband = identity, betad unused)."""
    centnb = 0.5 * (bandlo + bandhi)
    delnb = bandhi - bandlo
    alfanb = brndm * arndm
    arotnb = np.sqrt(alfanb)
    freq_cutoff = 1400.0 if nbtrge > 0 else 1200.0
    zmass = np.zeros(NUTAB)
    zmass[1] = math.pow(10.0, MASS_MIN)
    incr = math.pow(10.0, MASS_INC)
    for j in range(2, NUTAB):
        zmass[j] = zmass[j - 1] * incr
    zroot = np.sqrt(zmass)
    xtemv = TEMP_MIN + TEMP_INC * np.arange(NTTAB)
    t2 = xtemv * xtemv
    tfour = t2 * t2
    fortcu = 4.0 * (xtemv * xtemv * xtemv)

    # band sources tabsr
    srcwd = np.zeros((NTTAB, NBLX))
    for n in range(1, NBLX + 1):
        if n <= 40:
            bdlo, bdhi = bandlo[n + 15], bandhi[n + 15]
            dl = delnb[n + 15]
        else:
            bdlo, bdhi = bdlocm[n - 33 + OFFSET], bdhicm[n - 33 + OFFSET]
            dl = bdhi - bdlo
        nsubds = int((dl - 1.0e-03) / 10 + 1)
        for nsb in range(1, nsubds + 1):
            if nsb != nsubds:
                cnu = 10.0 * (nsb - 1) + bdlo + 5.0
                dnu = 10.0
            else:
                cnu = 0.5 * (10.0 * (nsb - 1) + bdlo + bdhi)
                dnu = bdhi - (10.0 * (nsb - 1) + bdlo)
            c1 = 3.7412e-05 * (cnu * cnu * cnu)
            x1 = _vexp(1.4387 * cnu / xtemv)
            srcwd[:, n - 1] = srcwd[:, n - 1] + c1 / (x1 - 1.0) * dnu
    tabsr_vae = np.zeros((NTTAB, NBLY))
    tabsr_vae[:, :40] = tabsr_vae[:, :40] + srcwd[:, :40]
    tabsr_vae[:, 8 + OFFSET:] = srcwd[:, 8 + OFFSET:NBLY]
    tabsr_td = np.zeros_like(tabsr_vae)
    tabsr_td[:-1] = (tabsr_vae[1:] - tabsr_vae[:-1]) * 0.1
    tabsr = Table(tabsr_vae, tabsr_td)

    src1nb = np.zeros((NTTAB, NBLW))
    dbdtnb = np.zeros((NTTAB, NBLW))
    for n in range(NBLW):
        anu = centnb[n]
        c1 = 3.7412e-05 * (anu * anu * anu)
        x = 1.4387 * anu / xtemv
        x1 = _vexp(x)
        sc = c1 / (x1 - 1.0)
        ddsc = sc / (x1 - 1.0) * x1 * x / xtemv
        src1nb[:, n] = delnb[n] * sc
        dbdtnb[:, n] = delnb[n] * ddsc

    def in_main(c):
        return c < 5.6e2 or (c > freq_cutoff and c <= 2.2e3)

    def in_sub(c, m):
        return 1.2e3 < c <= 1.4e3 and fbdlo[m] < c <= fbdhi[m]

    sum4 = np.zeros(NTTAB); sum6 = np.zeros(NTTAB); sum7 = np.zeros(NTTAB); sum8 = np.zeros(NTTAB)
    sum4wd = np.zeros(NTTAB)
    sum4a = np.zeros((NTTAB, nbtrge)); sum6a = np.zeros_like(sum4a)
    sum7a = np.zeros_like(sum4a); sum8a = np.zeros_like(sum4a)
    for n in range(NBLW):
        c = centnb[n]
        if in_main(c):
            sum4 = sum4 + src1nb[:, n]
            sum6 = sum6 + dbdtnb[:, n]
            sum7 = sum7 + dbdtnb[:, n] * arotnb[n]
            sum8 = sum8 + dbdtnb[:, n] * alfanb[n]
        for m in range(nbtrge):
            if in_sub(c, m):
                sum4a[:, m] = sum4a[:, m] + src1nb[:, n]
                sum6a[:, m] = sum6a[:, m] + dbdtnb[:, n]
                sum7a[:, m] = sum7a[:, m] + dbdtnb[:, n] * arotnb[n]
                sum8a[:, m] = sum8a[:, m] + dbdtnb[:, n] * alfanb[n]
        if 1.6e2 < c < 5.6e2:
            sum4wd = sum4wd + src1nb[:, n]
    r1, r2, s2, t3 = sum4 / tfour, sum6 / fortcu, sum7 / fortcu, sum8 / fortcu
    r1wd = sum4wd / tfour
    r1a, r2a = sum4a / tfour[:, None], sum6a / fortcu[:, None]
    s2a, t3a = sum7a / fortcu[:, None], sum8a / fortcu[:, None]

    ssum = np.zeros((NTTAB, NUTAB)); sumdbe = np.zeros_like(ssum); sum3 = np.zeros_like(ssum)
    sumwde = np.zeros_like(ssum)
    suma = np.zeros((NTTAB, NUTAB, nbtrge)); sumdbea = np.zeros_like(suma); sum3a = np.zeros_like(suma)
    j122 = slice(121, NUTAB)
    for n in range(NBLW):
        c = centnb[n]
        expo = None
        if in_main(c):
            x2 = arotnb[n] * zroot
            expo = _vexp(-x2)
            fac = (1.0 - (1.0 + x2[j122]) * expo[j122]) / (alfanb[n] * zmass[j122])
            ssum = ssum + src1nb[:, n:n + 1] * expo[None, :]
            sumdbe = sumdbe + dbdtnb[:, n:n + 1] * expo[None, :]
            sum3[:, j122] = sum3[:, j122] + dbdtnb[:, n:n + 1] * fac[None, :]
        for m in range(nbtrge):
            if in_sub(c, m):
                x2 = arotnb[n] * zroot
                expo = _vexp(-x2)
                fac = (1.0 - (1.0 + x2[j122]) * expo[j122]) / (alfanb[n] * zmass[j122])
                suma[:, :, m] = suma[:, :, m] + src1nb[:, n:n + 1] * expo[None, :]
                sumdbea[:, :, m] = sumdbea[:, :, m] + dbdtnb[:, n:n + 1] * expo[None, :]
                sum3a[:, j122, m] = sum3a[:, j122, m] + dbdtnb[:, n:n + 1] * fac[None, :]
        if 1.6e2 < c < 5.6e2:
            sumwde = sumwde + src1nb[:, n:n + 1] * expo[None, :]

    def fill(vae1, vae3, r1_, r2_, s2_, t3_):
        vae1[:, :3] = r1_[:, None]
        vae3[:, :121] = (r2_[:, None] / 2.0 - s2_[:, None] * zroot[None, :121] / 3.0
                         + t3_[:, None] * zmass[None, :121] / 8.0)

    tab1v = ssum / tfour[:, None]
    tab2v = sumdbe / fortcu[:, None]
    tab3v = sum3 / fortcu[:, None]
    fill(tab1v, tab3v, r1, r2, s2, t3)
    tab1wv = sumwde / tfour[:, None]
    tab1wv[:, :3] = r1wd[:, None]

    def derivs(v, cross=True):
        td = np.zeros_like(v); md = np.zeros_like(v); cd = np.zeros_like(v)
        td[:-1] = (v[1:] - v[:-1]) / TEMP_INC
        md[:, 1:-1] = (v[:, 2:] - v[:, 1:-1]) / MASS_INC
        if cross:
            cd[:-1, 1:-1] = (v[1:, 2:] - v[1:, 1:-1] - v[:-1, 2:] + v[:-1, 1:-1]) / (TEMP_INC * MASS_INC)
        return Table(v, td, md, cd)

    tabs = dict(tab1=derivs(tab1v), tab2=derivs(tab2v, cross=False), tab3=derivs(tab3v), tab1w=derivs(tab1wv),
                tabsr=tabsr)
    t1a, t2a, t3a_ = [], [], []
    for m in range(nbtrge):
        v1 = suma[:, :, m] / tfour[:, None]
        v2 = sumdbea[:, :, m] / fortcu[:, None]
        v3 = sum3a[:, :, m] / fortcu[:, None]
        fill(v1, v3, r1a[:, m], r2a[:, m], s2a[:, m], t3a[:, m])
        t1a.append(derivs(v1)); t2a.append(derivs(v2, cross=False)); t3a_.append(derivs(v3))
    stack = lambda ts: Table(*[np.stack([getattr(t, f) for t in ts], axis=-1) for f in ("vae", "td", "md", "cd")])
    tabs.update(tab1a=stack(t1a), tab2a=stack(t2a), tab3a=stack(t3a_))
    return tabs


def _antemp(z):
    zb = (11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.8520, 90.0, 91.0, 92.0)
    c = (-6.5, 0.0, 1.0, 2.80, 0.0, -2.80, -2.00, 0.0, 0.0, 0.0, 0.0)
    delta = (0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    temp = 288.15 + c[0] * z
    for n in range(10):
        expo = (z - zb[n]) / delta[n]
        if abs(expo) <= 60.0:
            x = math.exp(expo)
            zlog = math.log(x + 1.0 / x)
        else:
            zlog = abs(expo)
        expp = zb[n] / delta[n]
        if abs(expp) <= 60.0:
            x = math.exp(expp)
            faclog = math.log(x + 1.0 / x)
        else:
            faclog = abs(expp)
        temp = temp + (c[n + 1] - c[n]) * 0.5 * (z + delta[n] * (zlog - faclog))
    return temp


def ptz(pref1: np.ndarray):
    """gas_tf ptz on the first reference profile pref(:,1) [Pa] (nlev+1 values):
    -> stemp, gtemp (nlev+1), ixprkminh2o (1-based, as the Fortran)."""
    kmax = len(pref1) - 1
    pd = np.asarray(pref1, float).copy()
    plm = np.zeros(kmax + 1)
    for k in range(1, kmax):
        plm[k] = 0.5 * (pd[k - 1] + pd[k])
    plm[kmax] = pd[kmax]
    plm = plm * 1.0e-02
    pd = pd * 1.0e-02
    nlay = kmax
    plmcgs = pd * 1.0e03
    gtemp = np.zeros(kmax + 1)
    for k in range(kmax):
        gtemp[k] = math.pow(plmcgs[k], 0.2) * math.pow(1. + plmcgs[k] / 30000., 0.8) / 1013250.
    gtemp[kmax] = 1.0
    press = np.zeros(nlay + 1)
    press[0] = plm[nlay]
    for k in range(2, nlay + 2):
        press[k - 1] = pd[nlay + 1 - k]
    alt = [0.0]
    tq = [_antemp(0.0)]
    for k in range(nlay):
        dlogp = 7.0 * math.log(press[k] / press[k + 1])
        nint = int(dlogp / 0.5) + 1
        dz = 1.0e-05 * (1.0e02 * RDGAS) * dlogp / (7.0 * GRAV * float(nint))
        ht = alt[k]
        for _ in range(nint):
            rk1 = _antemp(ht) * dz
            rk2 = _antemp(ht + 0.5 * rk1) * dz
            rk3 = _antemp(ht + 0.5 * rk2) * dz
            rk4 = _antemp(ht + rk3) * dz
            ht = ht + 0.16666667 * (rk1 + rk2 + rk2 + rk3 + rk3 + rk4)
        alt.append(ht)
        tq.append(_antemp(ht))
    stemp = np.array(tq[::-1])
    prkminh2o = 28.0
    if plm[0] >= prkminh2o:
        ix = 1
    elif plm[kmax - 1] < prkminh2o:
        ix = kmax + 1
    else:
        ix = None
        for k in range(2, kmax + 1):
            if plm[k - 1] - prkminh2o >= 0.0:
                ix = k
                break
    return stemp, gtemp, ix


class LongwaveData:
    """All time-independent sealw99 set-up data read from INPUT/ (data_dir = sea_esf)."""

    def __init__(self, data_dir: str, nbtrge: int = 1):
        if nbtrge != 1:
            raise NotImplementedError("no_h2o_bands_1200_1400 /= 1")
        self.nbtrge = nbtrge
        r = _Records(os.path.join(data_dir, "h2ocoeff_ckd_speccombwidebds_hi00"))
        awide, bwide = r.read(1)[0], r.read(1)[0]
        self.apwd, self.bpwd, self.atpwd, self.btpwd = [r.read(1)[0] for _ in range(4)]
        r.read(1); r.read(1)
        self.acomb, self.bcomb, self.apcm, self.bpcm, self.atpcm, self.btpcm, self.bdlocm, self.bdhicm = \
            [np.array(r.read(NBLY)) for _ in range(8)]
        self.ab15wd = awide * bwide
        r = _Records(os.path.join(data_dir, "o39001200_hi00_data"))
        self.ao3rnd = np.array(r.read(3))
        self.bo3rnd = np.array(r.read(3))
        v = _read_12001400(os.path.join(data_dir, "h2o12001400_hi00_data"), nbtrge)
        self.csfah2o = np.stack([v[4], v[5]])                         # (2, nbtrge)
        fbdlo, fbdhi = v[6], v[7]
        r = _Records(os.path.join(data_dir, "h2ocoeff_ckd_0_3000_10cm_hi00"))
        arndm, brndm = np.array(r.read(NBLW)), np.array(r.read(NBLW))
        for _ in range(4):
            r.read(NBLW)
        bandlo, bandhi = np.array(r.read(NBLW)), np.array(r.read(NBLW))
        self.bandlo, self.bandhi = bandlo, bandhi
        for k, t in _tables(arndm, brndm, bandlo, bandhi, self.bdlocm, self.bdhicm, fbdlo, fbdhi, nbtrge).items():
            setattr(self, k, t)
        self._ckd_init(data_dir)

    def _ckd_init(self, data_dir):
        r = _Records(os.path.join(data_dir, "h2ockd2.1_data"))
        v1s, _, dvs, _ = r.read(4)
        ssh2o_296 = np.array(r.read(2000))
        r.read(4); r.read(2000)
        r.read(4)
        sfh2o = np.array(r.read(2000))
        r = _Records(os.path.join(data_dir, "h2ockd2.1_corrdata"))
        sfac, fscal = np.array(r.read(2000)), np.array(r.read(2000))
        r = _Records(os.path.join(data_dir, "radfn_5-2995_100-490k"))
        vals = np.array(r.read(2 * 40 * 300)).reshape(300, 40, 2)
        self.radfunc = Table(vals[:, :, 0].T.copy(), vals[:, :, 1].T.copy())      # (40, 300)
        vjtab = 5. + 10. * np.arange(300)
        self.ioffh2o = 16
        nptch2o = 56 - self.ioffh2o
        self.vvj = np.array([v1s + dvs * float(j + self.ioffh2o - 1) for j in range(1, nptch2o + 1)])
        self.ssh2o_296, self.sfac, self.sfh2o, self.fscal = ssh2o_296, sfac, sfh2o, fscal

        def avg(j1, j2, div, dfv=None):
            s = f = 0.0
            for j in range(j1 - 1, j2):
                s = s + vjtab[j] * ssh2o_296[j] * sfac[j] / div
                f = f + vjtab[j] * sfh2o[j] * fscal[j] / (div if dfv is None else dfv)
            return s, f

        rng = [(57, 63, 7.), (64, 70, 7.), (71, 80, 10.), (81, 90, 10.), (91, 99, 9.), (100, 107, 8.),
               (108, 120, 13.)]
        sf = [avg(*x) for x in rng]
        self.svj = np.array([x[0] for x in sf]); self.fvj = np.array([x[1] for x in sf])
        self.radfnbd = np.array([0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.992])
        self.svjwd, self.fvjwd = avg(57, 80, 24.)
        self.radfnbdwd = 0.92
        s, f = avg(81, 99, 22., 32.)           # (sic: 22 and 32)
        for j in range(107, 120):              # the same sums continue over 108-120
            s = s + vjtab[j] * ssh2o_296[j] * sfac[j] / 32.
            f = f + vjtab[j] * sfh2o[j] * fscal[j] / 32.
        self.svjinw, self.fvjinw = s, f
        self.radfnbdinw = 0.98
