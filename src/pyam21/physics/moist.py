"""Moist processes driver for AM2.1 (atmos_param/moist_processes, siena_201207).

do_ras, do_strat and do_cmt (diffusive cumulus momentum transport fed by the RAS mass
flux); use_tau = .false.; no dry adjustment, Donner, UW, MCA, Betts-Miller or LSC.
Sequence of moist_processes:
  tin = tm + tdt*dt (etc.);  coldT from the 150 hPa near-surface mean temperature;
  RAS updates tin/qin/tracers and the tendencies;  CMT diffusivity from the RAS mass flux;
  convect = (convective precip > 0);  strat_cloud on the updated state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import GRAV, RDGAS, TFREEZE
from .ras import RAS, RASConfig
from .strat_cloud import StratCloud, StratCloudConfig
from ..core import spow, ssqrt


@dataclass(frozen=True)
class MoistConfig:
    do_ras: bool = False
    do_strat: bool = False
    do_cmt: bool = False
    do_lsc: bool = True
    do_mca: bool = True
    do_donner_deep: bool = False
    do_uw_conv: bool = False
    do_gust_cv: bool = False
    use_tau: bool = False
    pdepth: float = 150.0e2
    gustmax: float = 3.0
    gustconst: float = 10.0 / 86400.0
    cmt_diff_norm: float = 1.0
    cmt_transport_scheme: str = "diffusive"
    cmt_mass_flux_source: str = "ras"

    @classmethod
    def from_nml(cls, nml: dict) -> "MoistConfig":
        g = nml.get("moist_processes_nml", {})
        kw = {}
        for f in ("do_ras", "do_strat", "do_cmt", "do_lsc", "do_mca", "do_donner_deep", "do_uw_conv",
                  "do_gust_cv", "use_tau"):
            if f in g:
                kw[f] = bool(g[f])
        for f in ("pdepth", "gustmax", "gustconst"):
            if f in g:
                kw[f] = float(g[f])
        if "cmt_mass_flux_source" in g:
            kw["cmt_mass_flux_source"] = str(g["cmt_mass_flux_source"])
        m = nml.get("cu_mo_trans_nml", {})
        if "diff_norm" in m:
            kw["cmt_diff_norm"] = float(m["diff_norm"])
        if "transport_scheme" in m:
            kw["cmt_transport_scheme"] = str(m["transport_scheme"])
        return cls(**kw)


def tempavg(pdepth, phalf, temp):
    """Pressure-weighted mean temperature of the lowest pdepth (Pa) of the column."""
    ncol, kx = temp.shape
    tsnow = torch.zeros_like(temp[:, 0])
    prsum = torch.zeros_like(tsnow)
    done = torch.ones_like(tsnow)
    pdep = torch.full_like(tsnow, pdepth)
    for k in range(kx - 1, -1, -1):
        pdel = (phalf[:, k + 1] - phalf[:, k]) * done
        m = (prsum + pdel) > pdep
        pdel = torch.where(m, pdepth - prsum, pdel)
        done = torch.where(m, torch.zeros_like(done), done)
        pdep = torch.where(m, torch.zeros_like(pdep), pdep)
        tsnow = tsnow + pdel * temp[:, k]
        prsum = prsum + pdel
    return tsnow / prsum


def diffusive_cu_mo_trans(diff_norm, mass_flux, t, p_full, z_half):
    """Momentum diffusivity from the convective mass flux (at the interface above each level)."""
    ncol, nlev = t.shape
    zbot = z_half[:, nlev]
    ztop = z_half[:, nlev]
    for k in range(1, nlev):                       # Fortran k = 2..nlev
        m1 = (mass_flux[:, k] != 0.0) & (mass_flux[:, k + 1] == 0.0)
        zbot = torch.where(m1, z_half[:, k], zbot)
        m2 = (mass_flux[:, k - 1] == 0.0) & (mass_flux[:, k] != 0.0)
        ztop = torch.where(m2, z_half[:, k], ztop)
    rho = p_full / (RDGAS * t)
    diff = diff_norm * mass_flux[:, 1:nlev] * (ztop - zbot)[:, None] / (rho[:, 1:] * GRAV)
    return torch.cat([torch.zeros_like(diff[:, :1]), diff], dim=1)


class MoistProcesses:
    TRACERS = ("sphum", "liq_wat", "ice_wat", "cld_amt")

    def __init__(self, config: MoistConfig, ras: RAS | None, strat: StratCloud | None):
        c = config
        if c.do_lsc or c.do_mca or c.do_donner_deep or c.do_uw_conv:
            raise NotImplementedError("only the AM2.1 moist configuration (RAS + strat_cloud) is ported")
        if c.do_cmt and (c.cmt_transport_scheme != "diffusive" or c.cmt_mass_flux_source != "ras"):
            raise NotImplementedError("only diffusive CMT fed by RAS is ported")
        self.c = c
        self.ras = ras
        self.strat = strat

    @classmethod
    def from_nml(cls, nml):
        c = MoistConfig.from_nml(nml)
        ras = RAS(RASConfig.from_nml(nml)) if c.do_ras else None
        strat = StratCloud(StratCloudConfig.from_nml(nml)) if c.do_strat else None
        return cls(c, ras, strat)

    def __call__(self, dt, land, phalf, pfull, zhalf, zfull, omega, diff_t, radturbten,
                 t, q, r, u, v, tm, qm, rm, um, vm, tdt, qdt, rdt, udt, vdt):
        """One call of moist_processes. r/rm/rdt are (ncol, nlev, 4) tracers
        (sphum, liq_wat, ice_wat, cld_amt); qdt is the sphum tendency (the same
        memory as rdt[..., 0] in the Fortran; here it is kept consistent).

        Returns updated tdt, qdt, rdt, udt, vdt, lprec, fprec, diff_cu_mo, convect and
        the stratiform cloud fields after the call.
        """
        c = self.c
        if c.use_tau:
            tin, qin, uin, vin, tracer = t, q, u, v, r
        else:
            tin = tm + tdt * dt
            qin = qm + qdt * dt
            uin = um + udt * dt
            vin = vm + vdt * dt
            tracer = rm + rdt * dt
        tsnow = tempavg(c.pdepth, phalf, t)
        coldT = tsnow <= TFREEZE
        z = torch.zeros_like(tsnow)
        lprec = z
        fprec = z
        ql, qi, qa = tracer[..., 1], tracer[..., 2], tracer[..., 3]
        rdt_l, rdt_i, rdt_a = rdt[..., 1], rdt[..., 2], rdt[..., 3]
        mc = torch.zeros_like(phalf)
        if self.ras is not None:
            rr = self.ras(dt, tin, qin, uin, vin, pfull, phalf, zhalf, coldT, ql, qi, qa)
            tin, qin, uin, vin = rr["t"], rr["q"], rr["u"], rr["v"]
            ql, qi, qa = rr["ql"], rr["qi"], rr["qa"]
            tdt = tdt + rr["ttnd"]
            qdt = qdt + rr["qtnd"]
            udt = udt + rr["utnd"]
            vdt = vdt + rr["vtnd"]
            rdt_l = rdt_l + rr["qltnd"]
            rdt_i = rdt_i + rr["qitnd"]
            rdt_a = rdt_a + rr["qatnd"]
            lprec = lprec + rr["rain"]
            fprec = fprec + rr["snow"]
            mc = rr["mc"]
        if c.do_cmt:
            mc_cmt = torch.zeros_like(mc) + mc
            diff_cu_mo = diffusive_cu_mo_trans(c.cmt_diff_norm, mc_cmt, tin, pfull, zhalf)
            # the diffusive scheme returns zero tendencies; the Fortran still adds them
            zero = torch.zeros_like(tin)
            tdt, udt, vdt = tdt + zero, udt + zero, vdt + zero
        else:
            diff_cu_mo = torch.zeros_like(tin)
        kx = tin.shape[1]
        mc_full = torch.cat([torch.zeros_like(tin[:, :1]),
                             0.5 * (mc[:, 1:kx] + mc[:, 2:kx + 1]) + 0.5 * (0.0 + 0.0) + 0.0], dim=1)
        precip = lprec + fprec
        convect = precip > 0.0
        gust_cv = torch.zeros_like(precip)
        if c.do_gust_cv:
            pp = precip > 0.0
            gust_cv = torch.where(pp, c.gustmax * ssqrt(precip / (c.gustconst + precip)), gust_cv)
        if self.strat is not None:
            sc = self.strat(dt, pfull, phalf, radturbten, tin, qin, ql, qi, qa, omega, mc_full, diff_t, land)
            ql = ql + sc["SL"]
            qi = qi + sc["SI"]
            qa = qa + sc["SA"]
            tin = tin + sc["ST"]
            qin = qin + sc["SQ"]
            dtinv = 1.0 / dt
            ttnd = sc["ST"] * dtinv
            qtnd = sc["SQ"] * dtinv
            rain = sc["surfrain"] * dtinv
            snow = sc["surfsnow"] * dtinv
            tdt = tdt + ttnd
            qdt = qdt + qtnd
            rdt_l = rdt_l + sc["SL"] * dtinv
            rdt_i = rdt_i + sc["SI"] * dtinv
            rdt_a = rdt_a + sc["SA"] * dtinv
            lprec = lprec + rain
            fprec = fprec + snow
        rdt = torch.stack([qdt, rdt_l, rdt_i, rdt_a], dim=-1)
        return dict(tdt=tdt, qdt=qdt, rdt=rdt, udt=udt, vdt=vdt, lprec=lprec, fprec=fprec,
                    diff_cu_mo=diff_cu_mo, convect=convect, gust_cv=gust_cv,
                    lsc_cloud_area=qa, lsc_liquid=ql, lsc_ice=qi, tin=tin, qin=qin)
