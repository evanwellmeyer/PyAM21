"""Boundary-layer turbulence driver (atmos_param/vert_turb_driver/vert_turb_driver.F90).

The AM2.1 configuration: do_entrain and do_stable_bl, no Mellor-Yamada/EDT/
diffusivity/shallow convection, use_tau = .false., gust_scheme = 'beljaars'.
Other options of the Fortran raise NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .entrain import Entrain, EntrainConfig
from .monin_obukhov import MoninObukhov
from .stable_bl_turb import StableBLConfig, StableBLTurb


@dataclass(frozen=True)
class VertTurbConfig:
    do_shallow_conv: bool = False
    do_mellor_yamada: bool = True
    do_diffusivity: bool = False
    do_molecular_diffusion: bool = False
    do_edt: bool = False
    do_stable_bl: bool = False
    use_tau: bool = True
    do_entrain: bool = False
    do_simple: bool = False
    gust_scheme: str = "constant"
    constant_gust: float = 1.0
    gust_factor: float = 1.0
    gust_zi: float = 1000.0

    @classmethod
    def from_nml(cls, nml: dict) -> "VertTurbConfig":
        g = nml.get("vert_turb_driver_nml", {})
        kw = {}
        for f, t in (("do_shallow_conv", bool), ("do_mellor_yamada", bool), ("do_diffusivity", bool),
                     ("do_molecular_diffusion", bool), ("do_edt", bool), ("do_stable_bl", bool),
                     ("use_tau", bool), ("do_entrain", bool), ("do_simple", bool),
                     ("gust_scheme", str), ("constant_gust", float), ("gust_factor", float)):
            if f in g:
                kw[f] = t(g[f])
        return cls(**kw)


class VertTurbDriver:
    def __init__(self, config: VertTurbConfig, entrain_cfg: EntrainConfig,
                 stable_cfg: StableBLConfig, mo: MoninObukhov):
        c = config
        if c.do_mellor_yamada or c.do_diffusivity or c.do_edt or c.do_shallow_conv:
            raise NotImplementedError("only the AM2.1 turbulence options (entrain + stable_bl) are ported")
        self.c = c
        self.entrain = Entrain(entrain_cfg, mo) if c.do_entrain else None
        self.stable = StableBLTurb(stable_cfg, mo) if c.do_stable_bl else None

    def __call__(self, dt, tdtlw, u_star, b_star, lat, convect,
                 u, v, t, q, ql, qi, qa, um, vm, tm, qm, qlm, qim, qam,
                 udt, vdt, tdt, qdt, qldt, qidt, qadt,
                 pfull, phalf, zfull, zhalf):
        """Returns diff_t, diff_m (ncol, nlev) at the interface above each level, gust
        and z_pbl (ncol,).

        ql/qi/qa are the stratiform cloud tracers (r), *m their previous values (rm),
        *dt the accumulated tendencies (rdt).
        """
        c = self.c
        if c.use_tau:
            uu, vv, tt, qq = u, v, t, q
            qlin, qiin, qain = ql, qi, qa
        else:
            uu = um + dt * udt
            vv = vm + dt * vdt
            tt = tm + dt * tdt
            qq = qm + dt * qdt
            qlin = qlm + qldt * dt
            qiin = qim + qidt * dt
            qain = qam + qadt * dt
        diff_t = torch.zeros_like(t)
        diff_m = torch.zeros_like(t)
        z_pbl = torch.full_like(u_star, -999.0)
        vspblcap = torch.zeros_like(u_star)
        if self.entrain is not None:
            e = self.entrain(tdtlw, convect, u_star, b_star, tt, qq, qlin, qiin, qain, uu, vv,
                             zfull, pfull, zhalf, phalf)
            use_entr = e["use_entr"]
            z_pbl = e["zsml"]
            vspblcap = e["vspblcap"]
            if self.entrain.c.apply_entrain:
                # entrain sets diff = k_entr where use_entr = 1 (levels 2..nlev)
                lev2 = torch.zeros_like(use_entr, dtype=torch.bool)
                lev2[:, 1:] = True
                sel = lev2 & (use_entr == 1.0)
                diff_t = torch.where(sel, e["k_t_entr"], diff_t)
                diff_m = torch.where(sel, e["k_m_entr"], diff_m)
        if self.stable is not None:
            if self.entrain is not None:
                akm, akh = self.stable(tt, qq, qlin, qiin, uu, vv, zhalf, zfull, u_star, b_star, lat,
                                       vspblcap=vspblcap)
                diff_m = use_entr * e["k_m_entr"] + (1 - use_entr) * akm
                diff_t = use_entr * e["k_t_entr"] + (1 - use_entr) * akh
            else:
                akm, akh = self.stable(tt, qq, qlin, qiin, uu, vv, zhalf, zfull, u_star, b_star, lat)
                diff_m = diff_m + torch.clamp(akm - diff_m, min=0.0)
                diff_t = diff_t + torch.clamp(akh - diff_t, min=0.0)
        if c.gust_scheme == "constant":
            gust = torch.full_like(u_star, c.constant_gust)
        elif c.gust_scheme == "beljaars":
            bpos = b_star > 0.0
            prod = torch.where(bpos, u_star * b_star * c.gust_zi, torch.ones_like(u_star))
            gust = torch.where(bpos, c.gust_factor * prod ** (1.0 / 3.0), torch.zeros_like(u_star))
        else:
            raise ValueError("invalid gust_scheme")
        return dict(diff_t=diff_t, diff_m=diff_m, gust=gust, z_pbl=z_pbl)
