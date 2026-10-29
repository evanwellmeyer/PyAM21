"""Radiation components against the Fortran's radiation-step dumps (rad_<case>):
set-up tables, aerosol LW optics, the sealw99 longwave, the whole sea_esf package."""
import os

import numpy as np
import pytest
import torch

from pyam21.radiation.aerosol import AerosolOptics
from pyam21.radiation.esfsw_data import read_solar_spectrum
from pyam21.radiation.lw_gases_stdtf import LblTransmission
from pyam21.radiation.lw_tables import LongwaveData, ptz
from pyam21.radiation.sea_esf import SeaEsfRad
from pyam21.radiation.sealw99 import SeaLW99
from pyam21.scm.case import DATA_ROOT, load_case
from pyam21.scm.site import box_edges
from pyam21.scm.vert_grid import compute_pres_full
from pyam21.testing import col, dump_files, maxrel, read_dump
from pyam21.time_manager import set_time
from conftest import requires_dumps

CASES = ["mpace_b", "astex", "mpace_a", "arm"]
SEA = os.path.join(DATA_ROOT, "sea_esf")


def _pref(cfg):
    pf = compute_pres_full(cfg.vgrid, torch.tensor([101325., 81060.], dtype=torch.float64)).numpy()
    return np.stack([np.append(pf[0], 101325.), np.append(pf[1], 81060.)], axis=1)


@pytest.fixture(scope="module")
def lwdata():
    return LongwaveData(SEA)


@pytest.mark.reference
def test_lw_tables(lwdata):
    requires_dumps("rad_arm")
    d = read_dump(dump_files("rad_arm", "lw_a")[0][2])
    for t in ("tab1", "tab2", "tab3", "tab1w", "tab1a", "tab2a", "tab3a"):
        for c in ("vae", "td", "md", "cd"):
            assert maxrel(getattr(getattr(lwdata, t), c), d[f"{t}_{c}"]) < 1e-14, (t, c)
    assert maxrel(lwdata.tabsr.vae, d["tabsr_vae"]) < 1e-14
    for n in ("apcm", "bpcm", "atpcm", "btpcm", "acomb", "bcomb"):
        assert maxrel(getattr(lwdata, n), d[n]) == 0.0


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_gas_transmission_tables(case):
    requires_dumps("rad_" + case)
    cfg = load_case(case)
    pref = _pref(cfg)
    g = read_dump(dump_files("rad_" + case, "gt_tab")[0][2])
    stemp, gtemp, ix = ptz(pref[:, 0])
    assert maxrel(stemp, g["stemp"]) < 1e-14 and maxrel(gtemp, g["gtemp"]) < 1e-14
    assert ix == int(g["ixprkminh2o"])
    d = read_dump(dump_files("rad_" + case, "rd_in")[0][2])
    T = LblTransmission(SEA, pref).tables(float(d["rrvco2"]), float(d["rrvch4"]), float(d["rrvn2o"]))
    for k, v in T.items():
        assert maxrel(v, g[k]) < 1e-11, k


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_sealw99(case, lwdata):
    """sealw99 fed the dumped inputs and the dumped gas tables."""
    tag = "rad_" + case
    requires_dumps(tag)
    cfg = load_case(case)
    spec = read_solar_spectrum(SEA, cfg.nml)
    names = [str(x).strip() for x in cfg.nml["aerosol_nml"]["data_names"]]
    A = AerosolOptics(cfg.nml, SEA, spec, names)
    g = read_dump(dump_files(tag, "gt_tab")[0][2])
    gas_tf = {k: np.asarray(v, float) for k, v in g.items() if k != "ixprkminh2o"}
    gas_tf["ixprkminh2o"] = int(g["ixprkminh2o"])
    LW = SeaLW99(lwdata, gas_tf)
    la = {s: read_dump(p) for s, c, p in dump_files(tag, "lw_a")}
    li = {s: read_dump(p) for s, c, p in dump_files(tag, "lw_i")}
    for s, c, p in dump_files(tag, "rd_in"):
        d = read_dump(p)
        atm = {k: col(d["ai_" + k]) for k in ("press", "pflux", "temp", "tflux", "rh2o")}
        tot, ke = A.lw_optical_depths(col(d["aerosol"]), col(d["ai_aerosolrelhum"]))
        assert maxrel(tot, col(la[s]["op_totaerooptdep"])) < 1e-14
        gases = {k: float(d[k]) for k in ("rrvf11", "rrvf12", "rrvf113", "rrvf22")}
        cld = dict(cmxolw=col(d["cs_cmxolw"]), crndlw=col(d["cs_crndlw"]), emmxolw=col(d["cr_emmxolw"]),
                   emrndlw=col(d["cr_emrndlw"]))
        o = LW(atm, col(d["qo3"]), gases, tot, ke, cld)
        for k in ("heatra", "flxnet", "heatracf", "flxnetcf"):
            assert maxrel(o[k], col(li[s][k])) < 1e-12, (s, k)


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_sea_esf_package(case):
    """The whole package from the radiation inputs (T, q, cloud tracers, albedos, astronomy)."""
    tag = "rad_" + case
    requires_dumps(tag)
    cfg = load_case(case)
    lo, la = box_edges(cfg)
    P = SeaEsfRad(cfg.nml, SEA, [(tuple(lo), tuple(la))], _pref(cfg))
    astro = {s: read_dump(p) for s, c, p in dump_files(tag, "rd_astro")}
    calc = {s: read_dump(p) for s, c, p in dump_files(tag, "rd_calc")}
    v = lambda x: torch.as_tensor(np.asarray(x).ravel(), dtype=torch.float64)
    for s, c, p in dump_files(tag, "rd_in"):
        d, a, r = read_dump(p), astro[s], calc[s]
        atm = {k: col(d["ai_" + k]) for k in ("press", "pflux", "temp", "tflux", "rh2o", "deltaz", "clouddeltaz",
                                              "cloudtemp", "aerosolrelhum")}
        surf = {k: v(d["sf_" + k]) for k in ("asfc_vis_dir", "asfc_nir_dir", "asfc_vis_dif", "asfc_nir_dif")}
        surf["land"] = v(d["sf_land"]) > 0.5
        ast = dict(cosz=v(a["cosz"]), fracday=v(a["fracday"]), rrsun=float(a["rrsun"]))
        extra = dict(ql=col(d["cs_cloud_water"]), qi=col(d["cs_cloud_ice"]), qa=col(d["cs_cloud_area"]))
        o = P(atm, surf, ast, cfg.start + set_time(int((s - 1) * cfg.dt)), True, True, extra)
        for k in ("dfsw", "ufsw", "hsw", "dfswcf", "ufswcf", "hswcf"):
            ref = col(r["sw_" + k])
            assert maxrel(o[k], ref[..., 0] if ref.dim() == 3 else ref) < 1e-12, (s, k)
        for k in ("heatra", "flxnet", "heatracf", "flxnetcf"):
            assert maxrel(o[k], col(r["lw_" + k])) < 1e-12, (s, k)
