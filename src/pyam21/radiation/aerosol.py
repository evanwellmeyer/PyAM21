"""Aerosol concentrations (sea_esf_rad/aerosol.F90, siena_201207).

use_aerosol_timeseries = .false. with a dataset entry: the monthly
aerosol.climatology.nc (sigma levels, kg/m2 per layer) for the entry year, interpolated
onto the radiation interfaces (Atmos_input%pflux) as layer column amounts (kg/m2).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..constants import DIFFAC

from ..time_manager import ModelTime, days_in_month, get_date, set_date
from .interpolator import ClimatologyInterpolator


class AerosolClimatology:
    def __init__(self, nml: dict, data_dir: str, boxes):
        g = nml.get("aerosol_nml", {})
        if bool(g.get("use_aerosol_timeseries", False)):
            raise NotImplementedError("use_aerosol_timeseries")
        self.names = [str(x).strip() for x in g["data_names"]]
        entry = [int(x) for x in g.get("aerosol_dataset_entry", [1] * 6)]
        if entry[:6] == [1, 1, 1, 1, 1, 1]:
            raise NotImplementedError("aerosol climatology without a fixed year")
        self.year = entry[0]
        self.family_names = [str(x).strip() for x in g.get("family_names", [])]
        self.families = []
        for n in range(1, len(self.family_names) + 1):
            f = g.get(f"in_family{n}", [])
            self.families.append([bool(x) for x in f][:len(self.names)])
        fname = str(g.get("filename", "aerosol.climatology.nc")).strip()
        self.interp = ClimatologyInterpolator(os.path.join(data_dir, fname), self.names, boxes)

    def aerosol_time(self, time: ModelTime) -> ModelTime:
        y, mo, dy, hr, mn, sc = get_date(time)
        entry = set_date(self.year, 2, 1, 0, 0, 0)
        if mo == 2 and dy == 29 and days_in_month(entry) != 29:
            dy -= 1
        return set_date(self.year, mo, dy, hr, mn, sc)

    def __call__(self, time: ModelTime, pflux: torch.Tensor) -> torch.Tensor:
        """-> (ncol, nlev, nfields) kg/m2."""
        t = self.aerosol_time(time)
        return torch.stack([self.interp(n, t, pflux) for n in self.names], dim=-1)


# ---- optical properties (aerosolrad_package.F90, 'shettle_fenn') ---------------------------------

NBLW = 300
SULFATE_FLAG, OMPHILIC_FLAG, BCPHILIC_FLAG = 0, -1, -2
SULFATE_INDICES = ([30] * 33 + [35] * 5 + [40] * 5 + [45] * 5 + [50] * 5 + [55] * 5 + [60] * 5 + [65] * 5
                   + [70] * 5 + [75] * 5 + [80] * 4 + [82, 82, 84, 84, 86, 86, 88, 88, 90, 91, 92, 93, 94, 95, 96,
                                                     97, 98, 99, 100])
_TARGET = {**{f"{p}_dust_{s}": f"dust_{s}" for p in ("anthro", "natural")
              for s in ("0.1", "0.2", "0.4", "0.8", "1.0", "2.0", "4.0", "8.0")},
           "black_carbon": "soot", "organic_carbon": "organic_carbon", "sea_salt": "sea_salt"}


def read_optical_file(path, names):
    """read_optical_input_file (list-directed): -> endwvn, ext/ssalb/asym (nwvn, nmodels)."""
    toks_lines = open(path).read().splitlines()
    pos = 0

    def line():
        nonlocal pos
        l = toks_lines[pos]; pos += 1
        return l

    def values(n):
        vals = []
        while len(vals) < n:
            vals.extend(x for x in line().replace(",", " ").split() if x)
        return [float(v) for v in vals[:n]]

    nw = int(line().split(";")[0].replace(",", " ").split()[0])
    ncat = int(line().split(";")[0].replace(",", " ").split()[0])
    line()
    endw = [int(v) for v in values(nw)]
    ext = np.zeros((nw, len(names))); ssa = np.zeros_like(ext); asy = np.zeros_like(ext)
    found = [False] * len(names)
    for _ in range(ncat):
        l = line()
        while not l.strip():                      # list-directed read skips blank records
            l = line()
        name = l.strip().split()[0]
        line(); e = values(nw); line(); s = values(nw); line(); a = values(nw)
        if name in names:
            i = names.index(name)
            ext[:, i], ssa[:, i], asy[:, i] = e, s, a
            found[i] = True
    if not all(found):
        raise ValueError("aerosol optical models missing: " + str([n for n, f in zip(names, found) if not f]))
    return endw, ext, ssa, asy


class AerosolOptics:
    """Band-averaged Shettle-Fenn optical properties and the species -> model mapping."""

    def __init__(self, nml: dict, data_dir: str, spec, data_names):
        from .esfsw_data import band_intervals
        g = nml.get("aerosolrad_package_nml", {})
        if str(g.get("aerosol_data_set", "")).strip() != "shettle_fenn":
            raise NotImplementedError("aerosol_data_set other than shettle_fenn")
        self.names = [str(x).strip() for x in g["aerosol_optical_names"] if str(x).strip()]
        self.do_sw = bool(g.get("do_swaerosol", False))
        self.do_lw = bool(g.get("do_lwaerosol", False))
        fname = str(g.get("optical_filename", "aerosol.optical.dat")).strip()
        self.endw, ext, ssa, asy = read_optical_file(os.path.join(data_dir, fname), self.names)
        self.ext_ivl, self.ssa_ivl, self.asy_ivl = ext, ssa, asy
        # species -> optical model (1-based model indices as the Fortran, or a flag)
        self.optical_index = []
        for n in data_names:
            if n in ("so4", "so4_anthro", "so4_natural"):
                self.optical_index.append(SULFATE_FLAG)
            else:
                self.optical_index.append(self.names.index(_TARGET.get(n, n)) + 1)
        self.sulfate_index = np.array([self.names.index(f"sulfate_{v}%") + 1 for v in SULFATE_INDICES])
        # sw_aerosol_interaction: thickavg_0d onto the solar bands
        n1, n2, sol = band_intervals(spec, self.endw)
        nb, nm = spec.nbands, len(self.names)
        self.extband = np.zeros((nb, nm)); self.ssaband = np.zeros((nb, nm)); self.asyband = np.zeros((nb, nm))
        for m in range(nm):
            # thickavg clips ssalbivl in place (the Fortran passes the module array), and
            # lw_aerosol_interaction, called after it, sees the clipped values
            e, s, a = ext[:, m], ssa[:, m], asy[:, m]
            for b in range(nb):
                sk = so = sog = sr = 0.0
                for ni in range(n1[b], n2[b] + 1):
                    if e[ni] != 0.0:
                        s[ni] = min(s[ni], 1.0)
                        sp = np.sqrt((1.0 - s[ni]) / (1.0 - s[ni] * a[ni]))
                        sr = sr + (1.0 - sp) / (1.0 + sp) * sol[b, ni]
                        sk = sk + e[ni] * sol[b, ni]
                        so = so + s[ni] * e[ni] * sol[b, ni]
                        sog = sog + s[ni] * e[ni] * a[ni] * sol[b, ni]
                self.extband[b, m] = sk / spec.solflxbandref[b]
                self.asyband[b, m] = sog / (so + 1.0e-100)
                r = sr / spec.solflxbandref[b]
                self.ssaband[b, m] = 4.0 * r / ((1.0 + r) ** 2 - self.asyband[b, m] * (1.0 - r) ** 2)

        self._lw_aerosol_interaction()

    def _lw_aerosol_interaction(self):
        """lw_aerosol_interaction + the band sums of aerosol_radiative_properties: Planck-
        weighted (283.15 K) extinction and single-scattering albedo in the 9 LW aerosol bands
        (8 non-continuum bands + the 560-800 cm-1 band)."""
        endw = np.asarray(self.endw, float)
        nw = len(endw)
        iend = [int((e + 0.01) / 10.0) for e in endw]
        src = np.zeros(NBLW)
        for n in range(NBLW):
            cent = 5.0 + n * 10.0
            c1 = 3.7412e-05 * (cent * cent * cent)
            src[n] = 10.0 * (c1 / (math.exp(1.4387 * cent / 283.15) - 1.0))

        def weights(starts, ends):
            planck = []
            for a, b in zip(starts, ends):
                p = 0.0
                for ib in range(a - 1, b):
                    p = p + src[ib]
                planck.append(p)
            pivl = np.zeros((len(ends), nw))
            nivl1, nivl2 = [0] * len(ends), [0] * len(ends)
            nivl, sump, nband, band1 = 1, 0.0, 1, True
            nivl1[0] = 1
            for w in range(1, NBLW + 1):
                sump = sump + src[w - 1]
                if w == iend[nivl - 1]:
                    pivl[nband - 1, nivl - 1] = sump
                    sump = 0.0
                if w == ends[nband - 1]:
                    if w != iend[nivl - 1]:
                        pivl[nband - 1, nivl - 1] = sump
                        sump = 0.0
                    nivl2[nband - 1] = nivl
                    nband += 1
                    if nband <= len(ends):
                        nivl1[nband - 1] = nivl + 1 if w == iend[nivl - 1] else nivl
                if w == iend[nivl - 1]:
                    nivl += 1
                    if band1 and nband == 1 and starts[0] <= iend[nivl - 2] < ends[0]:
                        nivl1[0] = nivl - 1
                        band1 = False
                if w >= ends[-1]:
                    break
            wts = np.zeros((len(ends), nw))
            for n in range(len(ends)):
                for ni in range(nivl1[n], nivl2[n] + 1):
                    wts[n, ni - 1] = pivl[n, ni - 1] / planck[n]
            return wts

        wfr = weights([57, 64, 71, 81, 91, 100, 108, 121], [63, 70, 80, 90, 99, 107, 120, 140])
        wco = weights([57], [80])
        wts = np.concatenate([wfr, wco])
        nm = self.ext_ivl.shape[1]
        self.extbandlw = np.zeros((wts.shape[0], nm))
        self.ssabandlw = np.zeros((wts.shape[0], nm))
        for m in range(nm):
            for na in range(wts.shape[0]):
                e = s = 0.0
                for ni in range(nw):
                    e = e + self.ext_ivl[ni, m] * wts[na, ni] * 1.0e03
                    s = s + self.ssa_ivl[ni, m] * wts[na, ni]
                self.extbandlw[na, m], self.ssabandlw[na, m] = e, s

    def lw_optical_depths(self, aerosol, relhum):
        """optical_depth_aerosol for every LW aerosol band: absorption optical depth
        summed from the top, totaerooptdep (ncol, k+1, nbands), and the lowest layer's
        depth in the last band, aerooptdep_KE_15 (ncol,)."""
        dt, dev = aerosol.dtype, aerosol.device
        ext = torch.as_tensor(self.extbandlw, dtype=dt, device=dev)
        ssa = torch.as_tensor(self.ssabandlw, dtype=dt, device=dev)
        irh = torch.clamp(torch.floor(100.0 * relhum + 0.5), 0, 100).long()
        sul = torch.as_tensor(self.sulfate_index - 1, device=dev)[irh]
        om = 1.0 - ssa                                                  # (nb, nmodel)
        dep = torch.zeros(aerosol.shape[:2] + (ext.shape[0],), dtype=dt, device=dev)
        for nsc, oi in enumerate(self.optical_index):
            if oi > 0:
                o, e = om[:, oi - 1], ext[:, oi - 1]
            elif oi == SULFATE_FLAG:
                o, e = om.T[sul], ext.T[sul]
            else:
                raise NotImplementedError("aerosol optical flag")
            dep = dep + DIFFAC * aerosol[..., nsc:nsc + 1] * o * e
        tot = torch.cat([torch.zeros_like(dep[:, :1]), torch.cumsum(dep, dim=1)], dim=1)
        return tot, dep[:, -1, -1]

    def sw_optical_depths(self, aerosol, relhum):
        """compute_aerosol_optical_props: aerosol (ncol, k, nsp) kg/m2, relhum (ncol, k)
        -> aeroextopdep, aerosctopdep, aeroasymfac (ncol, k, nbands)."""
        dt, dev = aerosol.dtype, aerosol.device
        T = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
        ext, ssa, asy = T(self.extband), T(self.ssaband), T(self.asyband)
        # NINT rounds half away from zero; relhum >= 0 so round-half-up
        irh = torch.clamp(torch.floor(100.0 * relhum + 0.5), 0, 100).long()
        sul = torch.as_tensor(self.sulfate_index - 1, device=dev)[irh]      # 0-based model
        nb = ext.shape[0]
        se = torch.zeros(aerosol.shape[:2] + (nb,), dtype=dt, device=dev)
        ss = torch.zeros_like(se); sg = torch.zeros_like(se)
        for nsc, oi in enumerate(self.optical_index):
            amt = (1.e3 * aerosol[..., nsc])[..., None]
            if oi > 0:
                e_i, s_i, a_i = ext[:, oi - 1], ssa[:, oi - 1], asy[:, oi - 1]
            elif oi == SULFATE_FLAG:
                e_i, s_i, a_i = ext.T[sul], ssa.T[sul], asy.T[sul]          # (ncol, k, nb)
            else:
                raise NotImplementedError("aerosol optical flag")
            arprod = e_i * amt
            se = se + arprod
            ss = ss + s_i * arprod
            sg = sg + a_i * (s_i * arprod)
        return se, ss, sg / (ss + 1.0e-30)
