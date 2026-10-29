"""float32 (and MPS when available): the runs stay finite and close to float64."""
import pytest
import torch

from pyam21.scm.case import load_case
from pyam21.scm.model import SCM

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["gcss_bomex", "gcss_dycoms2_rf01_a", "mpace_b", "arm"])
def test_float32_tracks_float64(case, device):
    cfg = load_case(case)
    a = SCM(cfg, dtype=torch.float64)
    b = SCM(cfg, device=device, dtype=torch.float32)
    for _ in range(12):
        a.step()
        b.step()
    T32 = b.state["T"].cpu().double()
    q32 = b.state["tracers"][..., 0].cpu().double()
    assert torch.isfinite(T32).all() and torch.isfinite(q32).all()
    assert float((T32 - a.state["T"]).abs().max()) < 1e-2              # K
    assert float((q32 - a.state["tracers"][..., 0]).abs().max()) < 1e-5  # kg/kg
