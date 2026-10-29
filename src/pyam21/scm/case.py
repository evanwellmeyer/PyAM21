"""Loading an SCM case from its reference input directory (reference/input/<case>).

The same files drive the Fortran reference and the port: input.nml (all
namelists), grid_spec.nc (site location and box size) and the case data files.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from ..constants import PI, RADIUS
from ..namelist import read_namelist
from ..time_manager import ModelTime, set_date, set_time
from .vert_grid import VertGrid, vert_grid_init

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUT_ROOT = os.path.join(REPO, "reference", "input")
DATA_ROOT = os.path.join(REPO, "reference", "input_data")


@dataclass
class CaseConfig:
    name: str
    path: str
    nml: dict
    vgrid: VertGrid
    lat_deg: float
    lon_deg: float
    dx: float
    dy: float
    area: float
    start: ModelTime
    dt: int
    nsteps: int
    pdamp: float
    sst_override: float | None = None

    @property
    def lat(self) -> float:
        return self.lat_deg * PI / 180.0

    @property
    def lon(self) -> float:
        return self.lon_deg * PI / 180.0


def read_grid_spec(path: str):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        xba = np.asarray(ds.variables["xba"][:], dtype=np.float64)
        yba = np.asarray(ds.variables["yba"][:], dtype=np.float64)
        xta = np.asarray(ds.variables["xta"][:], dtype=np.float64)
        yta = np.asarray(ds.variables["yta"][:], dtype=np.float64)
    return xba, yba, xta, yta


def load_case(name: str, root: str = INPUT_ROOT) -> CaseConfig:
    path = os.path.join(root, name)
    nml = read_namelist(os.path.join(path, "input.nml"))
    atm = nml["scm_atmosphere_nml"]
    kdim = int(atm["kdim"])
    As = np.asarray(atm["as"], dtype=np.float64)
    Bs = np.asarray(atm["bs"], dtype=np.float64)
    vgrid = vert_grid_init(Bs[:kdim + 1], As[:kdim + 1])
    xba, yba, xta, yta = read_grid_spec(os.path.join(path, "grid_spec.nc"))
    clon, clat = float(xta[0]), float(yta[0])
    while clon < 0.0:
        clon += 360.0
    while clon > 360.0:
        clon -= 360.0
    deg2rad = PI / 180.0
    bdx = abs(xba[-1] - xba[-2])
    bdy = abs(yba[-1] - yba[-2])
    dx = bdx * deg2rad * RADIUS
    dy = bdy * deg2rad * RADIUS
    area = bdy * bdx * (deg2rad * RADIUS) ** 2
    cpl = nml["coupler_nml"]
    start = set_date(*[int(x) for x in cpl["current_date"]])
    dt = int(cpl.get("dt_atmos", 1800))
    months = int(cpl.get("months", 0))
    if months:
        raise NotImplementedError("run length in months")
    length = set_time(int(cpl.get("hours", 0)) * 3600 + int(cpl.get("minutes", 0)) * 60
                      + int(cpl.get("seconds", 0)), int(cpl.get("days", 0)))
    nsteps = length.total_seconds // dt
    return CaseConfig(name, path, nml, vgrid, clat, clon, dx, dy, area, start, dt, nsteps,
                      float(atm.get("pdamp", 5000.0)), read_sst_override(os.path.join(path, "data_table")))


def read_sst_override(path: str):
    """Constant ICE t_surf from a data_table line  "ICE", "t_surf", "", "", .false., value."""
    if not os.path.exists(path):
        return None
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 6 and parts[0] == "ICE" and parts[1] == "t_surf" and parts[3] == "":
            return float(parts[5])
    return None


def make_forcing(case: CaseConfig, device="cpu", dtype=torch.float64):
    exp = case.nml["scm_forc_nml"]["experiment"]
    if exp == "bomex":
        from .forcing.bomex import BomexForcing
        return BomexForcing(case.nml, case.vgrid, case.lat, case.lon, device, dtype)
    if exp == "rf01":
        from .forcing.rf01 import RF01Forcing
        return RF01Forcing(case.nml, case.vgrid, case.lat, case.lon, case.path, device, dtype)
    if exp == "rf02":
        from .forcing.rf02 import RF02Forcing
        return RF02Forcing(case.nml, case.vgrid, case.lat, case.lon, case.path, device, dtype)
    if exp == "atex":
        from .forcing.atex import AtexForcing
        return AtexForcing(case.nml, case.vgrid, case.lat, case.lon, device, dtype)
    if exp == "gcss_astex":
        from .forcing.gcss_astex import GcssAstexForcing
        return GcssAstexForcing(case.nml, case.vgrid, case.lat, case.lon, case.path, device, dtype)
    if exp == "gcss_arm":
        from .forcing.gcss_arm import GcssArmForcing
        return GcssArmForcing(case.nml, case.vgrid, case.lat, case.lon, device, dtype)
    if exp == "rico":
        from .forcing.rico import RicoForcing
        return RicoForcing(case.nml, case.vgrid, case.lat, case.lon, device, dtype)
    if exp == "mpaceb":
        from .forcing.mpaceb import MpacebForcing
        return MpacebForcing(case.nml, case.vgrid, case.lat, case.lon, case.path, case.pdamp, device, dtype)
    if exp == "astex":
        from .forcing.astex import AstexForcing
        return AstexForcing(case.nml, case.vgrid, case.lat, case.lon, case.path, case.pdamp, device, dtype)
    if exp in ("arm", "mpacea"):
        from .forcing.iop import ArmForcing, MpaceaForcing
        cls = ArmForcing if exp == "arm" else MpaceaForcing
        return cls(case.nml, case.vgrid, case.lat, case.lon, case.path, case.pdamp, device, dtype)
    raise NotImplementedError(f"forcing '{exp}' not ported yet")
