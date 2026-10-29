"""Site topography statistics against the Fortran's mg_drag restart."""
import os

import netCDF4
import numpy as np
import pytest

from pyam21.scm.case import load_case
from pyam21.scm.site import NAVY_TOPOG, site_hprime
from pyam21.testing import DUMPS

CASES = [("gcss_arm", "gcss_arm"), ("mpace_a", "mpace_a"), ("gcss_rico_short", "rico"), ("gcss_bomex", "bomex")]


@pytest.mark.parametrize("case,tag", CASES)
def test_hprime(case, tag):
    f = os.path.join(DUMPS, tag, "mg_drag.res.nc")
    if not (os.path.exists(f) and os.path.exists(NAVY_TOPOG)):
        pytest.skip("reference restart or Navy topography not present")
    ref = float(np.asarray(netCDF4.Dataset(f)["ghprime"][:]).ravel()[0])
    assert abs(site_hprime(load_case(case)) - ref) <= 1e-11 * max(ref, 1.0)
