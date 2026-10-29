"""Per-scheme replay of the Fortran reference (float64, CPU)."""
import numpy as np
import pytest
import torch

from pyam21.namelist import read_namelist
from pyam21.testing import dump_files, dump_pairs, col, maxrel, read_dump
from pyam21.scm.case import load_case, make_forcing
from pyam21.physics.monin_obukhov import MoninObukhov, MoninObukhovConfig
from pyam21.physics.entrain import EntrainConfig
from pyam21.physics.stable_bl_turb import StableBLConfig
from pyam21.physics.vert_turb import VertTurbDriver, VertTurbConfig
from pyam21.physics.damping import DampingDriver, DampingConfig
from pyam21.physics.surface import SurfaceFlux, SurfaceFluxConfig
from pyam21.physics.vert_diff import VertDiff, VertDiffConfig
from pyam21.physics.ras import RAS, RASConfig
from pyam21.physics.strat_cloud import StratCloud, StratCloudConfig
from pyam21.physics.moist import MoistProcesses
from conftest import requires_dumps

CASES = [("gcss_bomex", "bomex"), ("arm", "arm"), ("gcss_rico_short", "rico"), ("astex", "astex"),
         ("gcss_dycoms2_rf01_a", "rf01_a"), ("gcss_dycoms2_rf01_b", "rf01_b"), ("mpace_a", "mpace_a"),
         ("mpace_b", "mpace_b")]


def nml(case):
    return read_namelist(f"reference/input/{case}/input.nml")


def check(worst, tol):
    bad = {k: v for k, v in worst.items() if v > tol}
    assert not bad, bad


@pytest.mark.reference
@pytest.mark.parametrize("case,tag", CASES)
def test_vert_turb(case, tag):
    requires_dumps(tag)
    n = nml(case)
    mo = MoninObukhov(MoninObukhovConfig.from_nml(n))
    drv = VertTurbDriver(VertTurbConfig.from_nml(n), EntrainConfig.from_nml(n), StableBLConfig.from_nml(n), mo)
    worst = {}
    for s, c, i, o in dump_pairs(tag, "vturb"):
        R, Rm, Rdt = col(i["r"]), col(i["rm"]), col(i["rdt"])
        out = drv(float(i["dt"]), col(i["tdtlw"]), col(i["u_star"]), col(i["b_star"]), col(i["lat"]),
                  col(i["convect"]) > 0, col(i["u"]), col(i["v"]), col(i["t"]), col(i["q"]), R[..., 1], R[..., 2],
                  R[..., 3], col(i["um"]), col(i["vm"]), col(i["tm"]), col(i["qm"]), Rm[..., 1], Rm[..., 2],
                  Rm[..., 3], col(i["udt"]), col(i["vdt"]), col(i["tdt"]), col(i["qdt"]), Rdt[..., 1], Rdt[..., 2],
                  Rdt[..., 3], col(i["p_full"]), col(i["p_half"]), col(i["z_full"]), col(i["z_half"]))
        for k, ref in (("diff_t", "diff_t_vert"), ("diff_m", "diff_m_vert"), ("gust", "gust"), ("z_pbl", "z_pbl")):
            worst[k] = max(worst.get(k, 0), maxrel(out[k], col(o[ref])))
    check(worst, 1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("case,tag", CASES)
def test_moist_processes(case, tag):
    requires_dumps(tag)
    mp = MoistProcesses.from_nml(nml(case))
    worst = {}
    for s, c, i, o in dump_pairs(tag, "moist"):
        r = mp(float(i["dt"]), col(i["frac_land"]), col(i["p_half"]), col(i["p_full"]), col(i["z_half"]),
               col(i["z_full"]), col(i["omega"]), col(i["diff_t"]), col(i["radturbten"]), col(i["t"]), col(i["q"]),
               col(i["r"]), col(i["u"]), col(i["v"]), col(i["tm"]), col(i["qm"]), col(i["rm"]), col(i["um"]),
               col(i["vm"]), col(i["tdt"]), col(i["qdt"]), col(i["rdt"]), col(i["udt"]), col(i["vdt"]))
        for k, (a, b) in dict(tdt=(r["tdt"], o["tdt"]), qdt=(r["qdt"], o["qdt"]), rdt=(r["rdt"], o["rdt"]),
                              lprec=(r["lprec"], o["lprec"]), fprec=(r["fprec"], o["fprec"]),
                              diff_cu_mo=(r["diff_cu_mo"], o["diff_cu_mo"])).items():
            worst[k] = max(worst.get(k, 0), maxrel(a, col(b)))
    check(worst, 1e-10)


@pytest.mark.reference
@pytest.mark.parametrize("case,tag", CASES)
def test_vert_diff(case, tag):
    requires_dumps(tag)
    vd = VertDiff(VertDiffConfig.from_nml(nml(case)))
    worst = {}
    for s, c, i, o in dump_pairs(tag, "vdiff_down"):
        R, Rdt = col(i["rm"]), col(i["rdt"])
        out = vd.down(float(i["dt"]), col(i["um"]), col(i["vm"]), col(i["tm"]), col(i["qm"]), R,
                      col(i["diff_m"]), col(i["diff_t"]), col(i["p_half"]), col(i["p_full"]), col(i["z_full"]),
                      col(i["tau_x"]), col(i["tau_y"]), col(i["dtau_du"]), col(i["dtau_dv"]),
                      col(i["udt"]), col(i["vdt"]), col(i["tdt"]), Rdt[..., 0], Rdt)
        for k, (a, b) in dict(dt_u=(out["dt_u"], o["udt"]), dt_v=(out["dt_v"], o["vdt"]), dt_t=(out["dt_t"], o["tdt"]),
                              delta_t=(out["delta_t"], o["sd_delta_t"])).items():
            worst[k] = max(worst.get(k, 0), maxrel(a, col(b)))
    check(worst, 1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("case,tag", CASES)
def test_damping(case, tag):
    requires_dumps(tag)
    import netCDF4
    from pyam21.testing import DUMPS
    d = DampingDriver(DampingConfig.from_nml(nml(case)))
    hp = torch.tensor(np.asarray(netCDF4.Dataset(f"{DUMPS}/{tag}/mg_drag.res.nc").variables["ghprime"][:],
                                 dtype=float).reshape(-1))
    worst = {}
    for s, c, i, o in dump_pairs(tag, "damping"):
        u, v, t = d(float(i["dt"]), col(i["p_full"]), col(i["p_half"]), col(i["z_full"]), col(i["z_half"]),
                    col(i["um"]), col(i["vm"]), col(i["tm"]), hp, col(i["udt"]), col(i["vdt"]), col(i["tdt"]))
        for k, (a, b) in dict(udt=(u, o["udt"]), vdt=(v, o["vdt"]), tdt=(t, o["tdt"])).items():
            worst[k] = max(worst.get(k, 0), maxrel(a, col(b)))
    check(worst, 1e-12)
