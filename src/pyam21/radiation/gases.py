"""Radiative gases (sea_esf_rad/radiative_gases.F90, ozone.F90, siena_201207).

Well-mixed gases from the *_gblannualdata time series at the fixed dataset entry
(time_varying_* = .false., *_data_source = 'input', specification 'time_series');
ozone from the monthly o3.climatology.nc (basic_ozone_type 'fixed_year'), interpolated
horizontally, in time (the model date moved to the dataset year) and onto the
radiation interfaces (Atmos_input%pflux).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ..time_manager import ModelTime, days_in_month, get_date, set_date, set_time
from .interpolator import ClimatologyInterpolator

GASES = ("co2", "ch4", "n2o", "f11", "f12", "f113", "f22")
SCALE = dict(co2=1.0e-6, ch4=1.0e-9, n2o=1.0e-9, f11=1.0e-12, f12=1.0e-12, f113=1.0e-12, f22=1.0e-12)


def _tsec(t):
    return t.days * 86400 + t.seconds


def read_gas_timeseries(path, gas, entry):
    """-> the value (mixing ratio) at the dataset entry date (read_gas_timeseries)."""
    lines = open(path).read().splitlines()
    n = int(lines[0][:12])
    times, vals = [], []
    for i in range(n):
        l = lines[1 + i]
        t, v = float(l[0:12]), float(l[12:24])
        vals.append(v * SCALE[gas])
        year = int(t)
        y0 = set_date(year, 1, 1, 0, 0, 0)
        diy = 366 if (year % 4 == 0) else 365             # Julian calendar
        extra = (t - year) * diy * 86400.0
        times.append(y0 + set_time(int(np.rint(extra)), 0))
    T = _tsec(set_date(*entry))
    ts = [_tsec(x) for x in times]
    if not (ts[0] <= T <= ts[-1]):
        raise ValueError(f"{gas}: dataset entry outside the time series")
    if T == ts[-1]:
        return vals[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= T:
            lo = mid
        else:
            hi = mid
    w = float(T - ts[lo]) / float(ts[hi] - ts[lo])
    return vals[lo] + w * (vals[hi] - vals[lo])


class RadiativeGases:
    def __init__(self, nml: dict, data_dir: str, boxes):
        g = nml.get("radiative_gases_nml", {})
        self.rrv = {}
        for gas in GASES:
            if bool(g.get(f"time_varying_{gas}", False)):
                raise NotImplementedError(f"time-varying {gas}")
            src = str(g.get(f"{gas}_data_source", "")).strip()
            spec = str(g.get(f"{gas}_specification_type", "")).strip()
            if src != "input" or spec != "time_series":
                raise NotImplementedError(f"{gas}: data source {src!r} / {spec!r}")
            entry = [int(x) for x in g.get(f"{gas}_dataset_entry", [1, 1, 1, 0, 0, 0])]
            self.rrv[gas] = read_gas_timeseries(os.path.join(data_dir, f"{gas}_gblannualdata"), gas, entry)
        o = nml.get("ozone_nml", {})
        if str(o.get("basic_ozone_type", "")).strip() != "fixed_year":
            raise NotImplementedError("basic_ozone_type other than fixed_year")
        if str(o.get("ozone_data_source", "fortuin_kelder")).strip() != "fortuin_kelder":
            raise NotImplementedError("ozone_data_source")
        self.ozone_year = int(o.get("ozone_dataset_entry", [1])[0])
        self.o3_name = str(o.get("data_name", ["ozone"])[0] if isinstance(o.get("data_name"), list)
                           else o.get("data_name", "ozone")).strip()
        fname = str(o.get("filename", "o3.trend.nc")).strip()
        self.o3 = ClimatologyInterpolator(os.path.join(data_dir, fname), [self.o3_name], boxes)

    def ozone_time(self, time: ModelTime) -> ModelTime:
        y, mo, dy, hr, mn, sc = get_date(time)
        entry = set_date(self.ozone_year, 2, 1, 0, 0, 0)
        if mo == 2 and dy == 29 and days_in_month(entry) != 29:
            dy -= 1
        return set_date(self.ozone_year, mo, dy, hr, mn, sc)

    def qo3(self, time: ModelTime, pflux: torch.Tensor) -> torch.Tensor:
        return self.o3(self.o3_name, self.ozone_time(time), pflux)
