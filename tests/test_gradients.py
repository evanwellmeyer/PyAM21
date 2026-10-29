"""Differentiability: finite gradients through several steps, and autodiff against finite
differences (the physics is piecewise smooth, so the check uses a small random direction)."""
import pytest
import torch

from pyam21.scm.case import load_case
from pyam21.scm.model import SCM

CASES = ["gcss_bomex", "gcss_dycoms2_rf01_a", "gcss_arm", "mpace_b"]


def _loss(cfg, T, q, nstep):
    m = SCM(cfg)
    m.state["T"] = T
    m.state["tracers"] = q
    for _ in range(nstep):
        m.step()
    return (m.state["T"] ** 2).mean() + 1e6 * (m.state["tracers"][..., 0] ** 2).mean()


@pytest.mark.parametrize("case", CASES)
def test_gradients_finite_and_match_fd(case):
    cfg = load_case(case)
    m0 = SCM(cfg)
    T0, q0 = m0.state["T"].clone(), m0.state["tracers"].clone()
    T = T0.clone().requires_grad_(True)
    q = q0.clone().requires_grad_(True)
    gT, gq = torch.autograd.grad(_loss(cfg, T, q, 3), (T, q))
    assert torch.isfinite(gT).all() and torch.isfinite(gq).all()
    torch.manual_seed(0)
    v = 0.1 * torch.randn_like(T0)
    eps = 1e-5
    fd = (float(_loss(cfg, T0 + eps * v, q0, 3)) - float(_loss(cfg, T0 - eps * v, q0, 3))) / (2 * eps)
    ad = float((gT * v).sum())
    assert abs(fd - ad) <= 1e-5 * abs(ad), (fd, ad)
