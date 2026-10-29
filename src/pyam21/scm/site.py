"""Site surface properties derived from the Navy 1/6-degree topography.

mg_drag (hprime) and the LM2 land model (topographic roughness scale) both use the
subgrid standard deviation of the land height over the SCM box, computed with
topography_mod's get_topog_stdev: a conservative (area-weighted in lon x sin(lat))
average of z and z**2 over the box, stdev = sqrt(max(<z^2> - <z>^2, 0)).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .case import DATA_ROOT, CaseConfig, read_grid_spec

NAVY_TOPOG = os.path.join(DATA_ROOT, "navy_topography.data", "navy_topography.data.nc")

_cache: dict = {}


def _navy():
    if "navy" not in _cache:
        import netCDF4
        with netCDF4.Dataset(NAVY_TOPOG) as d:
            _cache["navy"] = (np.asarray(d["xdat"][:], np.float64), np.asarray(d["ydat"][:], np.float64),
                              np.asarray(d["zdat"][:], np.float64))
    return _cache["navy"]


def box_edges(case: CaseConfig):
    """Longitude/latitude edges (radians) of the SCM column's grid box."""
    xba, yba, _, _ = read_grid_spec(os.path.join(case.path, "grid_spec.nc"))
    lon = np.sort(np.radians(xba[-2:]))
    lat = np.sort(np.radians(yba[-2:]))
    return lon, lat


def topography_stdev(lon_edges, lat_edges) -> float:
    """get_topog_stdev for a single box (edges in radians)."""
    xd, yd, z = _navy()
    lo = np.asarray(lon_edges, np.float64)
    while lo[0] < 0.0:
        lo = lo + 2.0 * math.pi
    la = np.asarray(lat_edges, np.float64)
    wx = np.clip(np.minimum(xd[1:], lo[1]) - np.maximum(xd[:-1], lo[0]), 0.0, None)
    if lo[1] > 2.0 * math.pi:                        # box straddling the 0/360 meridian
        wx = wx + np.clip(np.minimum(xd[1:], lo[1] - 2 * math.pi) - np.maximum(xd[:-1], lo[0] - 2 * math.pi),
                          0.0, None)
    sy = np.sin(yd)
    wy = np.clip(np.minimum(sy[1:], math.sin(la[1])) - np.maximum(sy[:-1], math.sin(la[0])), 0.0, None)
    iy = np.nonzero(wy)[0]
    ix = np.nonzero(wx)[0]
    w = wy[iy, None] * wx[None, ix]
    zz = z[np.ix_(iy, ix)]
    m = (w * zz).sum() / w.sum()
    m2 = (w * zz * zz).sum() / w.sum()
    v = m2 - m * m
    return math.sqrt(v) if v > 0.0 else 0.0


def site_hprime(case: CaseConfig) -> float:
    """mg_drag's hprime for the case's box (0 when the topography data is unavailable)."""
    if not os.path.exists(NAVY_TOPOG):
        return 0.0
    return topography_stdev(*box_edges(case))


def land_rough_scale(case: CaseConfig, rough_mom: float, hprime: float | None = None) -> float:
    """LM2 topographic roughness scale over land (land_properties_nml use_topo_rough)."""
    g = case.nml.get("land_properties_nml", {})
    if not bool(g.get("use_topo_rough", False)):
        return rough_mom
    stdev = site_hprime(case) if hprime is None else hprime
    stdev = min(stdev * float(g.get("topo_rough_factor", 1.0)), float(g.get("max_topo_rough", 100.0)))
    # (LM2 takes the max with the cover-type roughness; the specified roughness stands in for it)
    return max(stdev, rough_mom)
