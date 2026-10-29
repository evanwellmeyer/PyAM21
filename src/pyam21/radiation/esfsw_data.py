"""Set-up data of the ESF shortwave package (sea_esf_rad/esfsw_parameters.F90,
esfsw_driver.F90 esfsw_driver_init, siena_201207): band structure, the high-resolution
solar spectrum, gas absorption coefficients and the Rayleigh coefficients, read from
esf_sw_input_data_n38b18 (sw_resolution 'low') or _n72b25 ('high'). numpy, float64.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ..constants import PI

TOT_WVNUMS = 57600
INPUT_FLAG = 1.0e-99


class _Records:
    """Sequential Fortran formatted reads: each read starts a new record (line) and
    consumes as many lines as it needs values (whitespace-separated fields)."""

    def __init__(self, path):
        self.lines = open(path).read().splitlines()
        self.pos = 0

    def read(self, n, conv=float):
        vals = []
        while len(vals) < n:
            vals.extend(self.lines[self.pos].replace(",", " ").split())
            self.pos += 1
        return [conv(v) for v in vals[:n]]


def _logical(s):
    return s.strip().upper().startswith("T") or s.strip().upper().startswith(".T")


@dataclass
class SolarSpectrum:
    nbands: int
    nfrqpts: int
    nh2obands: int
    nstreams: int
    solflxbandref: np.ndarray
    solflxband: np.ndarray
    nfreqpts: np.ndarray
    endwvnbands: np.ndarray          # (0:nbands)
    firstrayband: int
    nirbands: int
    solarfluxtoa: np.ndarray         # (TOT_WVNUMS,)
    powph2o: np.ndarray
    p0h2o: np.ndarray
    co2: dict = field(default_factory=dict)   # c1..c4, c1str..c4str, totmax, totstrmax
    o2: dict = field(default_factory=dict)
    o2schrun: dict = field(default_factory=dict)
    kh2o: np.ndarray = None
    ko3: np.ndarray = None
    wtfreq: np.ndarray = None
    strterm: np.ndarray = None
    cosangstr: np.ndarray = None
    wtstr: np.ndarray = None
    betaddensitymol: np.ndarray = None
    refquanray: float = 0.0
    solflxtotal: float = 0.0
    visible_band_indx: int = 0
    onepsix_band_indx: int = 0


def _abs_coeffs(c1, c2, c3, c1s, c2s, c3s, nbands, schrun_band4: bool):
    n = len(c1)
    c4 = np.zeros(n); c4s = np.zeros(n); tmax = np.zeros(n); tsmax = np.zeros(n)
    for i in range(n):
        if c1[i] != INPUT_FLAG and c2[i] != INPUT_FLAG and c3[i] != INPUT_FLAG:
            c4[i] = c1[i] * c2[i] ** c3[i]
            c4s[i] = c1s[i] * c2s[i] ** c3s[i]
            tmax[i] = ((1.0 / c1[i]) + c2[i] ** c3[i]) ** (1.0 / c3[i]) - c2[i]
            if nbands == 18 and i == 3:
                tsmax[i] = np.finfo(np.float64).max
            else:
                tsmax[i] = ((1.0 / c1s[i]) + c2s[i] ** c3s[i]) ** (1.0 / c3s[i]) - c2s[i]
    return dict(c1=np.array(c1), c2=np.array(c2), c3=np.array(c3), c4=c4, c1str=np.array(c1s),
                c2str=np.array(c2s), c3str=np.array(c3s), c4str=c4s, totmax=tmax, totstrmax=tsmax)


def read_solar_spectrum(data_dir: str, nml: dict) -> SolarSpectrum:
    p = nml.get("esfsw_parameters_nml", {})
    res = str(p.get("sw_resolution", "high")).strip()
    nstreams = int(p.get("sw_diff_streams", 4))
    if res == "low":
        nbands, nfrqpts, nh2o, fname = 18, 38, 9, "esf_sw_input_data_n38b18"
    else:
        nbands, nfrqpts, nh2o, fname = 25, 72, 14, "esf_sw_input_data_n72b25"
    drv = nml.get("esfsw_driver_nml", {})
    for f in ("do_herzberg", "do_quench", "do_ch4_sw_effects", "do_n2o_sw_effects", "do_ica_calcs"):
        if bool(drv.get(f, False)):
            raise NotImplementedError(f"esfsw_driver_nml {f}")
    do_rayleigh_all_bands = bool(drv.get("do_rayleigh_all_bands", True))
    R = _Records(os.path.join(data_dir, fname))
    solflxbandref = np.array(R.read(nbands))
    nfreqpts = np.array(R.read(nbands, int))
    endwvn = np.array([0] + R.read(nbands, int))
    firstrayband, nirbands = R.read(2, int)
    powph2o = np.array(R.read(nh2o))
    p0h2o = np.array(R.read(nh2o))
    rd = lambda: R.read(nh2o)
    c1co2, c1co2s, c2co2, c2co2s, c3co2, c3co2s = rd(), rd(), rd(), rd(), rd(), rd()
    c1o2 = rd()
    v = R.read(nh2o + 1); c1o2s, c1o2schrun = v[:nh2o], v[nh2o]
    c2o2 = rd()
    v = R.read(nh2o + 1); c2o2s, c2o2schrun = v[:nh2o], v[nh2o]
    c3o2 = rd()
    v = R.read(nh2o + 1); c3o2s, c3o2schrun = v[:nh2o], v[nh2o]
    for _ in range(12):                                   # ch4, n2o coefficients (unused)
        rd()
    wtfreq = np.zeros(nfrqpts); kh2o = np.zeros(nfrqpts); ko3 = np.zeros(nfrqpts)
    strterm = np.zeros(nfrqpts, dtype=bool)
    for nf in range(nfrqpts):
        f = R.lines[R.pos].split(); R.pos += 1
        wtfreq[nf], kh2o[nf], ko3[nf], strterm[nf] = float(f[0]), float(f[1]), float(f[2]), _logical(f[3])
    nint = int(R.lines[R.pos].split()[0]); R.pos += 1
    sol = np.zeros(TOT_WVNUMS)
    nw1 = 0
    for ni in range(nint):
        f = R.lines[R.pos].split(); R.pos += 1
        n, val = int(f[0]), float(f[1])
        sol[nw1:nw1 + n] = val
        nw1 += n
    if nstreams == 4:
        ptstr = np.array([-0.861136312, -0.339981044, 0.861136312, 0.339981044])
        wtstr = np.array([0.347854845, 0.652145155, 0.347854845, 0.652145155])
    else:
        ptstr, wtstr = np.array([0.2]), np.array([1.0])
    if do_rayleigh_all_bands:
        firstrayband = 1
    p0h2o = 1.0e-2 / p0h2o
    kh2o = kh2o * 1.0e-01
    ko3 = ko3 * 1.0e-01
    co2 = _abs_coeffs(c1co2, c2co2, c3co2, c1co2s, c2co2s, c3co2s, nbands, True)
    o2 = _abs_coeffs(c1o2, c2o2, c3o2, c1o2s, c2o2s, c3o2s, nbands, True)
    c4s = c1o2schrun * c2o2schrun ** c3o2schrun
    tmaxs = ((1.0 / c1o2schrun) + c2o2schrun ** c3o2schrun) ** (1.0 / c3o2schrun) - c2o2schrun
    o2schrun = dict(c1=c1o2schrun, c2=c2o2schrun, c3=c3o2schrun, c4=c4s, totmax=tmaxs)
    solflxtotal = 0.0
    for b in range(nbands):
        solflxtotal = solflxtotal + solflxbandref[b]
    freqnu = np.array([0.5 * (endwvn[b] + endwvn[b + 1]) for b in range(nbands)])
    temprefray, pressrefray, densmolref, convfac = 288.15, 101325.0, 2.54743e19, 1.0e18
    refquanray = densmolref * temprefray / pressrefray
    corrfac = (6.0 + 3.0 * 1.39e-02) / (6.0 - 7.0 * 1.39e-02)
    gamma = 1.39e-02 / (2.0 - 1.39e-02)
    f1 = 7.5e-01 / (gamma * 2.0 + 1.0)
    f2 = gamma * 3.0 + 1.0
    f3 = 1.0 - gamma
    pinteg = 2.0 * PI * (2.0 * f1 * f2 * (1.0 + f3 / f2 / 3.0))
    twopiesq = 2.0 * PI ** 2
    densmolrefsqt3 = 3.0 * densmolref ** 2
    beta = np.zeros(nbands)
    for b in range(nbands):
        wl = 1.0e4 / freqnu[b]
        fsq = 1.0 / wl ** 2
        ristdm1 = (6.4328e3 + 2.94981e6 / (1.46e2 - fsq) + 2.554e4 / (4.1e1 - fsq)) * 1.0e-8
        ri = ristdm1 + 1
        beta[b] = twopiesq * (ri ** 2 - 1.0) ** 2 * corrfac / (densmolrefsqt3 * wl ** 4) * pinteg * convfac
    vis = 1.0e4 / 0.55
    visible = next(b + 1 for b in range(nbands) if endwvn[b + 1] > vis)
    onepsix = next(b + 1 for b in range(nbands) if endwvn[b + 1] > 1.0e4 / 1.61)
    return SolarSpectrum(nbands, nfrqpts, nh2o, nstreams, solflxbandref, solflxbandref.copy(), nfreqpts, endwvn,
                         firstrayband, nirbands, sol, powph2o, p0h2o, co2, o2, o2schrun, kh2o, ko3, wtfreq, strterm,
                         (ptstr + 1.0) * 5.0e-01, wtstr, beta, refquanray, solflxtotal, visible, onepsix)


def band_intervals(spec: SolarSpectrum, endivl):
    """The microphys_rad / aerosolrad mapping of parameterisation intervals (ending at
    wavenumbers endivl, 1-based) onto the solar bands: (nivl1, nivl2) 0-based per band
    and solivl (nbands, nivl), the TOA solar flux of each interval portion in each band."""
    nb = spec.nbands
    nivl = len(endivl)
    solivl = np.zeros((nb, nivl))
    n1 = np.zeros(nb, dtype=int); n2 = np.zeros(nb, dtype=int)
    iv = 0
    s = 0.0
    band = 0
    for nw in range(1, spec.endwvnbands[nb] + 1):
        s = s + spec.solarfluxtoa[nw - 1]
        if nw == endivl[iv]:
            solivl[band, iv] = s
            s = 0.0
        if nw == spec.endwvnbands[band + 1]:
            if nw != endivl[iv]:
                solivl[band, iv] = s
                s = 0.0
            n2[band] = iv
            band += 1
            if band < nb:
                n1[band] = iv + 1 if nw == endivl[iv] else iv
        if nw == endivl[iv]:
            iv += 1
    return n1, n2, solivl


def lean_solar(data_dir: str, nml: dict, nbands: int):
    """radiation_driver's solar input: with solar_dataset_entry set (not 1,1,1,0,0,0) and
    no time variation, the Lean monthly spectrum of that date gives the solar constant
    and the band fluxes (using_solar_timeseries_data). -> (solar_constant, solflxband) or
    None for the fixed nml solar constant."""
    g = nml.get("radiation_driver_nml", {})
    entry = [int(x) for x in g.get("solar_dataset_entry", [1, 1, 1, 0, 0, 0])]
    if bool(g.get("time_varying_solar_constant", False)):
        raise NotImplementedError("time_varying_solar_constant")
    if entry == [1, 1, 1, 0, 0, 0]:
        return None
    scale = float(g.get("solar_scale_factor", -1.0))
    lines = open(os.path.join(data_dir, "lean_solar_spectral_data.dat")).read().splitlines()
    f = lines[0].split()
    y0, y1, nv, nb = (int(x) for x in f[:4])
    if nb != nbands:
        raise ValueError("lean data band count")
    pos = 1

    def record():
        nonlocal pos
        tot = float(lines[pos][12:29]); pos += 1
        vals = []
        while len(vals) < nb:
            vals.extend(float(x) for x in lines[pos].split()); pos += 1
        return tot, np.array(vals[:nb])

    t1882, b1882 = record()
    table = {}
    for yr in range(y0, y1 + 1):
        for m in range(1, nv + 1):
            table[(yr, m)] = record()
    year, month = entry[0], entry[1]
    if year < y0:
        tot, band = t1882, b1882
    elif year > y1:
        tot, band = record()                     # the 2000 annual mean record that follows
    else:
        tot, band = table[(year, month)]
    if scale >= 0.0:
        tot, band = tot * scale, band * scale
    return tot, band
