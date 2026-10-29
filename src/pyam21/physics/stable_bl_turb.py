"""Stable boundary-layer diffusivities (atmos_param/stable_bl_turb/stable_bl_turb.F90).

Local Richardson-number closure: Monin-Obukhov stable mixing function for Ri > 0
blended with a Louis (1979) form for Ri < 0, mixing lengths alsm/alsh, capped by an
Ekman-like depth alpha*u*/f and (with entrainment) by vspblcap. kbot = nlev.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, GRAV, HLS, HLV, OMEGA, RDGAS, RVGAS, TFREEZE, VONKARM
from .monin_obukhov import MoninObukhov
from ..core import spow, ssqrt

D608 = (RVGAS - RDGAS) / RDGAS


@dataclass(frozen=True)
class StableBLConfig:
    akmax: float = 1.0e4
    alpha: float = 0.5
    alsm: float = 150.0
    alsh: float = 150.0
    fmin: float = 5.0e-5
    hpbl_cap: float = 1000.0
    diff_min: float = 0.001
    ri_crit: float = 0.2
    winddifmin: float = 0.01
    small: float = 1.0e-5
    b_louis: float = 9.4
    cmstar_louis: float = 7.4
    chstar_louis: float = 5.3

    @classmethod
    def from_nml(cls, nml: dict) -> "StableBLConfig":
        g = nml.get("stable_bl_turb_nml", {})
        kw = {f: float(g[f.lower()]) for f in ("akmax", "alpha", "alsm", "alsh", "fmin", "hpbl_cap",
                                             "diff_min", "ri_crit") if f.lower() in g}
        return cls(**kw)


class StableBLTurb:
    def __init__(self, config: StableBLConfig, mo: MoninObukhov):
        self.c = config
        self.mo = mo
        self.oalsm = 1.0 / config.alsm
        self.oalsh = 1.0 / config.alsh

    def __call__(self, temp, qv, ql, qi, um, vm, zhalf, zfull, u_star, b_star, lat, vspblcap=None):
        """Returns akm, akh (ncol, nlev); level k is the interface above full level k."""
        c = self.c
        ncol, kx = temp.shape
        zsurf = zhalf[:, kx]
        zfull_ag = zfull - zsurf[:, None]
        zhalf_ag = zhalf - zsurf[:, None]
        fcor = torch.clamp(torch.abs(2.0 * OMEGA * torch.sin(lat)), min=c.fmin)
        hpbl = c.alpha * u_star / fcor
        hpbl = torch.clamp(hpbl, max=c.hpbl_cap)
        if vspblcap is not None:
            hpbl = torch.minimum(hpbl, vspblcap)
        zk1 = zhalf_ag[:, 1:kx]                                   # zhalf_ag(k+1), k = 1..kx-1
        zfunc = torch.where((b_star < 0.0)[:, None],
                            torch.clamp(zk1 / torch.clamp(hpbl, min=0.1)[:, None], 0.0, 1.0),
                            torch.ones_like(zk1))
        hleff = (torch.clamp(0.05 * (temp - TFREEZE + 20.0), 0.0, 1.0) * HLV
                 + torch.clamp(0.05 * (TFREEZE - temp), 0.0, 1.0) * HLS)
        slv = CP_AIR * temp + GRAV * zfull_ag - hleff * (ql + qi)
        slv = slv * (1 + D608 * (qv + ql + qi))
        dsdzh = 1.0 / (zfull_ag[:, :-1] - zfull_ag[:, 1:])
        xxm1 = dsdzh * (um[:, :-1] - um[:, 1:])
        xxm2 = dsdzh * (vm[:, :-1] - vm[:, 1:])
        shear = xxm1 * xxm1 + xxm2 * xxm2
        smin = dsdzh * c.winddifmin * dsdzh * c.winddifmin
        shear = torch.where(shear < smin, smin, shear)
        xxm1 = slv[:, :-1] - slv[:, 1:]
        xxm2 = 0.5 * (slv[:, :-1] + slv[:, 1:])
        buoync = GRAV * dsdzh * xxm1 / xxm2
        Ri = buoync / shear
        xl = 1.0 / (VONKARM * zk1)
        lm = 1.0 / (xl + self.oalsm)
        lh = 1.0 / (xl + self.oalsh)
        Ritmp = torch.where(Ri < c.small, torch.full_like(Ri, c.small), Ri)
        phi = self.mo.stable_mix(Ritmp)
        phi = (1 - zfunc) * phi + zfunc * ((1 - torch.clamp(Ritmp / c.ri_crit, max=1.0)) ** 2.0)
        fm = phi
        fh = fm
        Ritmp = torch.where(Ri > 0.0, torch.zeros_like(Ri), Ri)
        zf = 1.0 + (1.0 / (dsdzh * zk1))
        zf = spow(zf, 1.0 / 3.0) - 1.0
        zf = spow(zf, 1.5)
        zf = zf / torch.sqrt(zk1)
        zf = zf * spow(dsdzh, 1.5)
        cmtmp = c.cmstar_louis * lm * lm * c.b_louis * zf
        chtmp = c.chstar_louis * lh * lh * c.b_louis * zf
        sq = ssqrt(-1.0 * Ritmp)
        fmtmp = 1.0 - (c.b_louis * Ritmp / (1.0 + cmtmp * sq))
        fhtmp = 1.0 - (c.b_louis * Ritmp / (1.0 + chtmp * sq))
        unst = Ri < c.small
        fm = torch.where(unst, fmtmp, fm)
        fh = torch.where(unst, fhtmp, fh)
        shear = ssqrt(shear)
        zero_top = torch.zeros_like(temp[:, :1])
        akm = torch.cat([zero_top, lm * lm * fm * shear], dim=1)
        akm = torch.where(akm < c.diff_min, torch.zeros_like(akm), akm)
        akm = torch.clamp(akm, max=c.akmax)
        akh = torch.cat([zero_top, lh * lh * fh * shear], dim=1)
        akh = torch.where(akh < c.diff_min, torch.zeros_like(akh), akh)
        akh = torch.clamp(akh, max=c.akmax)
        return akm, akh
