"""Climatology interpolation (atmos_shared/interpolator/interpolator.F90, siena_201207).

For the monthly time-series files the radiation reads (o3.climatology.nc,
aerosol.climatology.nc): horizontal conservative averaging onto each column's grid box
(overlap area in longitude x sin(latitude), as horiz_interp_conserve), linear time
interpolation between the bracketing records (time_interp, TIME_FLAG = LINEAR), and
the mass-weighted vertical interpolation onto the model half levels
(interp_weighted_scalar, INTERP_WEIGHTED_P) with CONSTANT extension beyond the data's
top and bottom. Fields in kg/m2 are converted to per-Pa before and back after the
vertical step. Sigma-level files are scaled by each column's maximum pressure.

The file is read once per record needed; the horizontally interpolated records are
cached (numpy, float64); the per-step vertical interpolation is torch.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..time_manager import ModelTime, set_date, set_time

NO_CONV, KG_M2 = 0, 1


def _chomp(s):
    return str(s).split("\0")[0].strip()


def _units_mr(units: str) -> int:
    return KG_M2 if _chomp(units) in ("kg/m2", "kg/m^2", "kg/m**2", "kg m^-2", "kg m**-2", "molecules/cm2/s",
                                      "molecule/cm2/s", "molec/cm2/s", "kg/m2/s") else NO_CONV


def _tsec(t: ModelTime) -> int:
    return t.days * 86400 + t.seconds


def conservative_weights(src_lonb, src_latb, box_lon, box_lat):
    """Overlap weights (nlat, nlon) of the source cells with one box (radians)."""
    lo0, lo1 = box_lon
    wx = np.zeros(len(src_lonb) - 1)
    for shift in (-2 * math.pi, 0.0, 2 * math.pi):
        a, b = lo0 + shift, lo1 + shift
        wx += np.clip(np.minimum(src_lonb[1:], b) - np.maximum(src_lonb[:-1], a), 0.0, None)
    s = np.sin(src_latb)
    wy = np.clip(np.minimum(s[1:], math.sin(box_lat[1])) - np.maximum(s[:-1], math.sin(box_lat[0])), 0.0, None)
    return wy[:, None] * wx[None, :]


def interp_weighted_scalar(grdin, grdout, datin):
    """interp_weighted_scalar_1D, vectorised over columns: grdin (ncol, nin+1),
    grdout (ncol, nout+1), datin (ncol, nin) -> (ncol, nout). The Fortran's four
    overlap cases, evaluated in its order (first match wins)."""
    gi0, gi1 = grdin[:, None, :-1], grdin[:, None, 1:]            # (ncol, 1, nin)
    go0, go1 = grdout[:, :-1, None], grdout[:, 1:, None]         # (ncol, nout, 1)
    c1 = (gi0 <= go0) & (gi1 >= go0) & (gi1 <= go1)
    c2 = ~c1 & (gi0 >= go0) & (gi0 <= go1) & (gi1 >= go1)
    c3 = ~c1 & ~c2 & (gi0 >= go0) & (gi1 <= go1)
    c4 = ~c1 & ~c2 & ~c3 & (gi0 <= go0) & (gi1 >= go1)
    zero = torch.zeros((), dtype=grdin.dtype, device=grdin.device)
    w = torch.where(c1, gi1 - go0, zero)
    w = torch.where(c2, go1 - gi0, w)
    w = torch.where(c3, gi1 - gi0, w)
    w = torch.where(c4, go1 - go0, w)
    # sequential accumulation over the input levels, as the Fortran
    contrib = datin[:, None, :] * w
    acc = torch.zeros_like(contrib[..., 0])
    for j in range(contrib.shape[-1]):
        acc = acc + contrib[..., j]
    return acc / (grdout[:, 1:] - grdout[:, :-1])


class ClimatologyInterpolator:
    def __init__(self, path: str, field_names, boxes, out_of_bounds="constant"):
        """boxes: list of ((lon0, lon1), (lat0, lat1)) in radians, one per column."""
        import netCDF4
        self.path = path
        self.names = list(field_names)
        self.out_of_bounds = out_of_bounds
        with netCDF4.Dataset(path) as ds:
            dtr = math.pi / 180.0
            lonb = np.asarray(ds["lonb"][:], np.float64) * dtr
            latb = np.asarray(ds["latb"][:], np.float64) * dtr
            if "pfull" in ds.variables:
                zf, zh, self.sigma = "pfull", "phalf", False
            else:
                zf, zh, self.sigma = "sigma_full", "sigma_half", True
            lev = np.asarray(ds[zf][:], np.float64)
            half = np.asarray(ds[zh][:], np.float64)
            if not self.sigma and _chomp(getattr(ds[zf], "units", "")) in ("mb", "hPa"):
                lev, half = lev * 100.0, half * 100.0
            if _chomp(getattr(ds[zf], "positive", "down")) == "up":
                lev, half = lev[::-1].copy(), half[::-1].copy()
                self.flip = True
            else:
                self.flip = False
            self.halflevs = half
            units = _chomp(ds["time"].units)
            if not units.startswith("day"):
                raise NotImplementedError(f"time units {units}")
            base = units[11:]
            y, mo, d = int(base[0:4]), int(base[5:7]), int(base[8:10])
            hh, mi, ss = int(base[11:13]), int(base[14:16]), int(base[17:19])
            if y == 0:
                raise NotImplementedError("climatological-year files")
            base_time = set_date(y, mo, d, hh, mi, ss)
            tin = np.asarray(ds["time"][:], np.float64)
            self.times = [set_time(int((t - int(t)) * 86400), int(t)) + base_time for t in tin]
            self.mr = {n: _units_mr(getattr(ds[n], "units", "")) for n in self.names}
        self.weights = [conservative_weights(lonb, latb, lo, la) for lo, la in boxes]
        self._cache = {}

    # ---- data ---------------------------------------------------------------------------
    def _record(self, name, n):
        """Horizontally interpolated record n (0-based) of a field: (ncol, nlev)."""
        key = (name, n)
        if key not in self._cache:
            import netCDF4
            with netCDF4.Dataset(self.path) as ds:
                a = np.asarray(ds[name][n], np.float64)          # (lev, lat, lon)
            if self.flip:
                a = a[::-1]
            prof = []
            for w in self.weights:
                ws = w.sum()
                prof.append(np.array([(w * a[k]).sum() / ws for k in range(a.shape[0])]))
            self._cache[key] = np.stack(prof)
        return self._cache[key]

    def time_weights(self, time: ModelTime):
        """time_interp_list (modtime = NONE): -> (index1, index2, weight), 0-based."""
        T = _tsec(time)
        ts = [_tsec(t) for t in self.times]
        n = len(ts)
        if ts[0] <= T < ts[-1]:
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ts[mid] <= T:
                    lo = mid
                else:
                    hi = mid
            return lo, hi, float(T - ts[lo]) / float(ts[hi] - ts[lo])
        if T == ts[-1]:
            return n - 1, n - 1, 0.0
        raise ValueError(f"{self.path}: time outside the dataset")

    def __call__(self, name, time: ModelTime, phalf: torch.Tensor):
        """interpolator_3D for one field: phalf (ncol, k+1) -> (ncol, k)."""
        i1, i2, w = self.time_weights(time)
        h = (1 - w) * self._record(name, i1) + w * self._record(name, i2)
        ncol = phalf.shape[0]
        if h.shape[0] == 1 and ncol > 1:
            h = np.repeat(h, ncol, axis=0)
        hd = torch.as_tensor(h, dtype=phalf.dtype, device=phalf.device)
        half = torch.as_tensor(self.halflevs, dtype=phalf.dtype, device=phalf.device)
        p_fact = phalf.max(dim=1).values[:, None] if self.sigma else torch.ones_like(phalf[:, :1])
        if self.mr[name] == KG_M2:
            hd = hd / ((half[1:] - half[:-1])[None, :] * p_fact)
        pclim = p_fact * half[None, :]
        if self.out_of_bounds == "constant":
            pmax_mod = phalf.max(dim=1).values
            pmin_mod = phalf.min(dim=1).values
            last = torch.where(pmax_mod > pclim[:, -1], pmax_mod, pclim[:, -1])
            first = torch.where(pmin_mod < pclim[:, 0], pmin_mod, pclim[:, 0])
            pclim = torch.cat([first[:, None], pclim[:, 1:-1], last[:, None]], dim=1)
        out = interp_weighted_scalar(pclim, phalf, hd)
        if self.mr[name] == KG_M2:
            out = out * (phalf[:, 1:] - phalf[:, :-1])
        return out
