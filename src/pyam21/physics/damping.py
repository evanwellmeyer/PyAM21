"""Damping: Rayleigh friction and orographic gravity-wave drag.

Translation of atmos_param/damping_driver/damping_driver.F90 (rayleigh) and
atmos_param/mg_drag/mg_drag.F90 (Stern & Pierrehumbert gravity-wave drag,
the non-MCM path), siena_201207. cg_drag and topo_drag are off in AM2.1.

hprime (sub-grid topography standard deviation, m) is a per-column input, computed
by the SCM site set-up from the Navy 10' topography (topography_mod
get_topog_stdev) or given by the host model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, GRAV, KAPPA, RDGAS
from .colops import at, first_true, levels
from ..core import spow, ssqrt

DAYPSEC = 1.0 / 86400.0
_EPS64 = 2.220446049250313e-16          # epsilon(real(8)), as in the Fortran


@dataclass(frozen=True)
class DampingConfig:
    trayfric: float = 0.0
    nlev_rayfric: int = 1
    do_mg_drag: bool = False
    do_cg_drag: bool = False
    do_topo_drag: bool = False
    do_conserve_energy: bool = False
    # mg_drag_nml
    xl_mtn: float = 1.0e5
    gmax: float = 2.0
    acoef: float = 1.0
    rho: float = 1.13
    low_lev_frac: float = 0.23
    flux_cut_level: float = 0.0
    mg_conserve_energy: bool = False
    do_mcm_mg_drag: bool = False

    @classmethod
    def from_nml(cls, nml: dict) -> "DampingConfig":
        d = nml.get("damping_driver_nml", {})
        m = nml.get("mg_drag_nml", {})
        kw = dict(trayfric=float(d.get("trayfric", 0.0)), nlev_rayfric=int(d.get("nlev_rayfric", 1)),
                  do_mg_drag=bool(d.get("do_mg_drag", False)), do_cg_drag=bool(d.get("do_cg_drag", False)),
                  do_topo_drag=bool(d.get("do_topo_drag", False)),
                  do_conserve_energy=bool(d.get("do_conserve_energy", False)),
                  xl_mtn=float(m.get("xl_mtn", 1.0e5)), gmax=float(m.get("gmax", 2.0)),
                  acoef=float(m.get("acoef", 1.0)), rho=float(m.get("rho", 1.13)),
                  low_lev_frac=float(m.get("low_lev_frac", 0.23)),
                  flux_cut_level=float(m.get("flux_cut_level", 0.0)),
                  mg_conserve_energy=bool(m.get("do_conserve_energy", False)),
                  do_mcm_mg_drag=bool(m.get("do_mcm_mg_drag", False)))
        return cls(**kw)


class DampingDriver:
    def __init__(self, config: DampingConfig):
        c = config
        if c.do_cg_drag or c.do_topo_drag or c.do_mcm_mg_drag:
            raise NotImplementedError("cg_drag, topo_drag and the MCM mg_drag are not ported")
        if c.flux_cut_level > 0.0:
            raise NotImplementedError("mg_drag flux_cut_level")
        self.c = c
        if abs(c.trayfric) > 0.0001 and c.nlev_rayfric > 0:
            self.rfactr = (1.0 / c.trayfric) if c.trayfric > 0.0 else (1.0 / abs(c.trayfric)) * DAYPSEC
            self.do_rayleigh = True
        else:
            self.rfactr = 0.0
            self.do_rayleigh = False

    def rayleigh(self, dt, pfull, u, v):
        c = self.c
        p2 = pfull * pfull
        n = c.nlev_rayfric
        fact = self.rfactr * (1.0 + (p2[:, :1] - p2[:, :n]) / (p2[:, :1] + p2[:, :n]))
        pad = torch.zeros_like(u[:, n:])
        udt = torch.cat([-u[:, :n] * fact, pad], dim=1)
        vdt = torch.cat([-v[:, :n] * fact, pad], dim=1)
        if c.do_conserve_energy:
            tdt = -((u + 0.5 * dt * udt) * udt + (v + 0.5 * dt * vdt) * vdt) / CP_AIR
            tdt = torch.cat([tdt[:, :n], pad], dim=1)
        else:
            tdt = torch.zeros_like(u)
        return udt, vdt, tdt

    def mg_drag(self, dt, u, v, t, pfull, phalf, zfull, zhalf, hprime):
        c = self.c
        ncol, kdim = u.shape
        kbtm = kdim
        psurf = phalf[:, kdim]
        ptop = (1.0 - c.low_lev_frac) * psurf
        ktop, found = first_true(pfull >= ptop[:, None])
        ktop = torch.where(found, ktop, torch.full_like(ktop, kdim))
        ktop = torch.clamp(ktop, max=kbtm - 1)
        lev = levels(u)
        # --- mgwd_base_flux
        xli = 1.0 / c.xl_mtn
        grav2 = GRAV * GRAV
        layer = lev >= ktop[:, None]
        delp = phalf[:, 1:] - phalf[:, :-1]
        dl = torch.where(layer, delp, torch.zeros_like(delp))
        sumw = 1.0 / dl.sum(dim=1)
        ubar = (u * dl).sum(dim=1) * sumw
        vbar = (v * dl).sum(dim=1) * sumw
        ulow = ssqrt(ubar * ubar + vbar * vbar)
        xn = ubar / (ulow + 1.0e-20)
        yn = vbar / (ulow + 1.0e-20)
        theta = t * (pfull / 1.0e5) ** (-KAPPA)
        pkt, pkb = at(pfull, ktop), pfull[:, kbtm - 1]
        tkt, tkb = at(theta, ktop), theta[:, kbtm - 1]
        Tkt, Tkb = at(t, ktop), t[:, kbtm - 1]
        bnv2 = grav2 * (pkt + pkb) * (tkt - tkb) / (RDGAS * (tkt + tkb) * (pkb - pkt) * 0.5 * (Tkt + Tkb))
        pos = bnv2 > 0.0
        bnv = torch.sqrt(torch.where(pos, bnv2, torch.ones_like(bnv2)))
        fr = bnv * hprime / (ulow + _EPS64)
        g = c.gmax * fr * fr / (fr * fr + c.acoef * c.acoef)
        taub = torch.where(pos, -c.rho * xli * ulow * ulow * ulow / bnv * g, torch.zeros_like(bnv2))
        # --- mgwd_satur_flux (interfaces 1..kdim+1)
        um_mid = torch.abs(0.50 * (u[:, :-1] + u[:, 1:]) * xn[:, None] + 0.50 * (v[:, :-1] + v[:, 1:]) * yn[:, None])
        um1 = torch.abs(u[:, :1] * xn[:, None] + v[:, :1] * yn[:, None])
        umag = torch.cat([um1, um_mid, torch.zeros_like(um1)], dim=1)        # umag(kdim+1) = 0
        umag = torch.where(umag < _EPS64, torch.zeros_like(umag), umag)
        dudz1 = (umag[:, 0] - umag[:, 1]) / (zfull[:, 0] - zhalf[:, 1])
        dudzk = (umag[:, 1:kdim] - umag[:, 2:kdim + 1]) / (zhalf[:, 1:kdim] - zhalf[:, 2:kdim + 1])
        dudz = torch.cat([dudz1[:, None], dudzk], dim=1)                     # (ncol, kdim)
        d2k = (dudz[:, 1:] - dudz[:, :-1]) / (zfull[:, 1:] - zfull[:, :-1])  # k = 2..kdim
        d2b = dudz[:, kbtm - 1] / (zfull[:, kbtm - 1] - zhalf[:, kbtm])
        d2udz2 = torch.cat([d2k[:, :1], d2k, d2b[:, None]], dim=1)
        upos = umag > 0.0
        usafe = torch.where(upos, umag, torch.ones_like(umag))
        extend = torch.where(upos, d2udz2 / usafe, torch.zeros_like(umag))    # vsamp = 1
        bk = (grav2 * (pfull[:, :-1] + pfull[:, 1:]) * (theta[:, :-1] - theta[:, 1:])
              / (RDGAS * (theta[:, :-1] + theta[:, 1:]) * (pfull[:, 1:] - pfull[:, :-1])
                 * 0.5 * (t[:, :-1] + t[:, 1:])))
        bnvk2 = torch.cat([bk[:, :1], bk, bk[:, -1:]], dim=1)                # (kb+1)=(kdim+1) <- kdim
        d2i = torch.where(upos, bnvk2 / (usafe * usafe) - extend, torch.full_like(umag, 1.0e30))
        small_d2i = d2i < 1.0e-30
        d = torch.where(small_d2i, torch.full_like(d2i, 1.0e30),
                        torch.sqrt(1.0 / torch.where(small_d2i, torch.ones_like(d2i), d2i)))
        d = torch.where(umag == 0.0, torch.zeros_like(d), d)
        ilev = torch.arange(1, kdim + 2, device=u.device)[None, :]
        tmid = torch.cat([t[:, :1], 0.50 * (t[:, :-1] + t[:, 1:]), t[:, -1:]], dim=1)
        taus_sat = -phalf * umag * umag * d * xli * c.gmax / (tmid * RDGAS)
        taus = torch.where(ilev >= ktop[:, None], taub[:, None], taus_sat)
        taus = torch.cat([taus[:, 1:2], taus[:, 1:]], dim=1)                  # taus(1) = taus(2)
        taus = torch.where(bnvk2 < 0.0, taub[:, None], taus)
        # --- mgwd_tend: taup(k) = max(taus(k), taup(k+1)), taup(kdim+1) = taub
        tp = [None] * (kdim + 1)
        tp[kdim] = taub
        for k in range(kdim - 1, -1, -1):
            tp[k] = torch.maximum(taus[:, k], tp[k + 1])
        taup = torch.stack(tp, dim=1)
        dterm = GRAV * (taup[:, 1:] - taup[:, :-1]) / (phalf[:, 1:] - phalf[:, :-1])
        dtaux = xn[:, None] * dterm
        dtauy = yn[:, None] * dterm
        if c.mg_conserve_energy:
            dtemp = -((u + 0.5 * dt * dtaux) * dtaux + (v + 0.5 * dt * dtauy) * dtauy) / CP_AIR
        else:
            dtemp = torch.zeros_like(u)
        return dtaux, dtauy, dtemp, taub * xn, taub * yn

    def __call__(self, dt, pfull, phalf, zfull, zhalf, u, v, t, hprime, udt, vdt, tdt):
        """Add the damping tendencies to udt, vdt, tdt (in the Fortran order) and return
        them; u, v, t are the previous-time-level fields."""
        if self.do_rayleigh:
            ru, rv, rt = self.rayleigh(dt, pfull, u, v)
            udt, vdt, tdt = udt + ru, vdt + rv, tdt + rt
        if self.c.do_mg_drag:
            gu, gv, gt, _, _ = self.mg_drag(dt, u, v, t, pfull, phalf, zfull, zhalf, hprime)
            udt, vdt, tdt = udt + gu, vdt + gv, tdt + gt
        return udt, vdt, tdt
