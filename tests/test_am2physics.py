"""The host-model interface: AM2Physics.forward on a batch of columns."""
import torch

from pyam21.constants import GRAV
from pyam21.physics.am2 import AM2Physics
from pyam21.scm.case import load_case
from pyam21.scm.model import SCM
from pyam21.scm.vert_grid import compute_height, compute_pres_full, compute_pres_half
from pyam21.time_manager import set_time


def test_forward_batch_columns_independent():
    cfg = load_case("gcss_bomex")
    m = SCM(cfg)
    n = 3
    st = m.state
    rep = lambda x: x.expand(n, *x.shape[1:]).clone()
    T, u, v, R, ps = (rep(st[k]) for k in ("T", "u", "v", "tracers", "ps"))
    T = T + torch.tensor([0.0, 0.5, -0.5], dtype=T.dtype)[:, None]     # three different columns
    vg = cfg.vgrid
    pfull, phalf = compute_pres_full(vg, ps), compute_pres_half(vg, ps)
    zfull, zhalf = compute_height(vg, GRAV * rep(st["elev"]), T, R[..., 0], pfull, phalf)
    lat = torch.full((n,), cfg.lat, dtype=T.dtype)
    phys = AM2Physics(cfg.nml, lat, lat * 0 + cfg.lon, hprime=0.0)
    surf = phys.init_ocean_surface(n, T.dtype)
    surf["t_surf"] = torch.full((n,), 300.4, dtype=T.dtype)
    mem = AM2Physics.init_memory(n, vg.nlev, T.dtype)
    z = torch.zeros_like(T)
    geom = dict(pfull=pfull, phalf=phalf, zfull=zfull, zhalf=zhalf, omega=z)
    out = phys(float(cfg.dt), cfg.start, dict(T=T, u=u, v=v, tracers=R, ps=ps), geom,
               dict(udt=z, vdt=z, tdt=z, rdt=torch.zeros_like(R)), surf, mem)
    for k in ("udt", "vdt", "tdt", "rdt"):
        assert out[k].shape[0] == n and torch.isfinite(out[k]).all()
    # each column on its own gives the same answer as in the batch
    for i in range(n):
        s1 = phys.init_ocean_surface(1, T.dtype)
        s1["t_surf"] = surf["t_surf"][i:i + 1]
        p1 = AM2Physics(cfg.nml, lat[i:i + 1], lat[i:i + 1] * 0 + cfg.lon, hprime=0.0)
        g1 = {k: x[i:i + 1] for k, x in geom.items()}
        o1 = p1(float(cfg.dt), cfg.start, dict(T=T[i:i + 1], u=u[i:i + 1], v=v[i:i + 1], tracers=R[i:i + 1],
                                              ps=ps[i:i + 1]), g1,
                dict(udt=z[i:i + 1], vdt=z[i:i + 1], tdt=z[i:i + 1], rdt=torch.zeros_like(R[i:i + 1])), s1,
                AM2Physics.init_memory(1, vg.nlev, T.dtype))
        assert torch.allclose(o1["tdt"], out["tdt"][i:i + 1], rtol=0, atol=1e-15)
        assert torch.allclose(o1["rdt"], out["rdt"][i:i + 1], rtol=0, atol=1e-18)
