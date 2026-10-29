"""Whole SCM runs of the port against the Fortran reference state, step by step."""
import pytest
import torch

from pyam21.scm.model import SCM
from pyam21.scm.case import load_case
from pyam21.testing import DumpRadiation, dump_files, read_dump, col, maxrel
from conftest import requires_dumps

RUNS = [("gcss_bomex", "bomex", 1e-11),
        ("gcss_dycoms2_rf01_a", "rf01_a", 1e-10),
        ("gcss_dycoms2_rf01_b", "rf01_b", 1e-10),
        ("gcss_dycoms2_rf02", "rf02", 1e-10),
        ("gcss_atex", "atex", 1e-10),
        ("gcss_astex", "gcss_astex", 1e-10),
        ("gcss_arm", "gcss_arm", 1e-10),
        ("gcss_rico_short", "rico", 1e-10),
        # radiation cases, with the port's sea_esf radiation
        ("mpace_b", "mpace_b", 1e-10),
        ("astex", "astex", 1e-10),
        ("mpace_a", "mpace_a", 1e-10),
        ("arm", "arm", 1e-10)]
RAD_CASES = ["mpace_b", "astex", "mpace_a", "arm"]


@pytest.mark.reference
@pytest.mark.parametrize("case,tag,tol", RUNS)
def test_scm_run(case, tag, tol):
    requires_dumps(tag)
    ref = {s: read_dump(p) for s, c, p in dump_files(tag, "scm_forc_in")}
    m = SCM(load_case(case))
    for n in range(1, max(ref)):
        m.step()
        r = ref[n + 1]
        for k, k2 in (("T", "T"), ("u", "u"), ("v", "v"), ("R", "tracers")):
            assert maxrel(m.state[k2], col(r[k])) < tol, (n, k)


@pytest.mark.reference
@pytest.mark.parametrize("case", RAD_CASES)
def test_scm_run_replayed_radiation(case):
    """The rest of the model with the Fortran's radiative tendencies replayed (isolates
    radiation errors from everything else)."""
    requires_dumps(case)
    ref = {s: read_dump(p) for s, c, p in dump_files(case, "scm_forc_in")}
    m = SCM(load_case(case), radiation=DumpRadiation(case))
    for n in range(1, min(max(ref), 25)):
        m.step()
        assert maxrel(m.state["T"], col(ref[n + 1]["T"])) < 1e-10, n
        assert maxrel(m.state["tracers"], col(ref[n + 1]["R"])) < 1e-10, n
