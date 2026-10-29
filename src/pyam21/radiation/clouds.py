"""Cloud specification for the radiation (sea_esf_rad/cloud_spec.F90 + strat_clouds_W.F90,
atmos_param/cloud_rad/cloud_rad.F90 cloud_summary3 / rnd_overlap, siena_201207).

cloud_type_form 'strat' with predicted microphysics and random overlap (cloud_rad_nml
overlap = 2): the prognostic liquid, ice and cloud fraction give the in-cloud water
paths, effective radii (droplet number from N_land / N_ocean, no prognostic number),
ice sizes from the temperature tables, and the microphysical concentrations and sizes
used by microphys_rad. No stochastic clouds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import DENS_H2O, GRAV, RDGAS, TFREEZE

# (upper temperature offset below freezing, size) tables of rnd_overlap
_ICE_SIZE_MICRO = (100.6, 80.8, 93.5, 63.9, 42.5, 39.9, 21.6, 20.2)
_ICE_REFF = (92.46298, 72.35392, 85.19071, 55.65818, 35.29989, 32.89967, 16.60895, 15.41627)


@dataclass(frozen=True)
class CloudRadConfig:
    overlap: int = 1
    scale_factor: float = 0.85
    qamin: float = 1.0e-2
    do_brenguier: bool = True
    qmin: float = 1.0e-10
    N_land: float = 3.0e8
    N_ocean: float = 1.0e8
    k_land: float = 1.143
    k_ocean: float = 1.077
    use_fu2007: bool = False

    @classmethod
    def from_nml(cls, nml: dict) -> "CloudRadConfig":
        from ..physics.strat_cloud import StratCloudConfig
        g = nml.get("cloud_rad_nml", {})
        sc = StratCloudConfig.from_nml(nml)
        kw = dict(qmin=sc.qmin, N_land=sc.N_land, N_ocean=sc.N_ocean)
        for f in ("overlap",):
            if f in g:
                kw[f] = int(g[f])
        for f in ("scale_factor", "qamin"):
            if f in g:
                kw[f] = float(g[f])
        if "do_brenguier" in g:
            kw["do_brenguier"] = bool(g["do_brenguier"])
        cs = nml.get("cloud_spec_nml", {})
        if str(cs.get("cloud_type_form", "")).strip() != "strat":
            raise NotImplementedError("cloud_type_form other than 'strat'")
        kw["use_fu2007"] = bool(cs.get("do_fu2007", False))
        c = cls(**kw)
        if c.overlap != 2:
            raise NotImplementedError("max-random overlap with predicted microphysics")
        return c


def _ice_table(t, table):
    d = t - TFREEZE
    out = torch.full_like(t, table[-1])
    # evaluated from the coldest band upward so the warmest applicable band wins
    for i, lim in reversed(list(enumerate((-25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0)))):
        out = torch.where(d > lim, torch.full_like(t, table[i]), out)
    return out


def cloud_spec_strat(c: CloudRadConfig, ql, qi, qa, press, pflux, temp, land):
    """strat_clouds_amt -> cloud_summary3 -> rnd_overlap (predicted microphysics).

    press (ncol, k+1) (level k+1 = surface), pflux (ncol, k+1), temp (ncol, k) the
    cloud temperature, land (ncol,). Returns the Cld_spec and Lsc_microphys fields."""
    ncol, kdim = ql.shape
    qmin, qamin = c.qmin, c.qamin
    pfull = press[:, :kdim]
    phalf = pflux
    zero = torch.zeros_like(ql)
    liq = (qa > qamin) & (ql > qmin)
    ice = (qa > qamin) & (qi > qmin)
    ql_l = torch.where(liq, ql, zero)
    qi_l = torch.where(ice, qi, zero)
    qa_l = torch.where(liq | ice, qa, zero)
    land = land.to(ql.dtype)
    N2 = c.N_land * land + c.N_ocean * (1.0 - land)
    kr = c.k_land * land + c.k_ocean * (1.0 - land)
    droplet_number = N2[:, None] / (pfull / (RDGAS * temp))
    # rnd_overlap
    cld = (ql_l > qmin) | (qi_l > qmin)
    lq = ql_l > qmin
    iq = qi_l > qmin
    qa_s = torch.where(cld, qa_l, torch.ones_like(qa_l))
    dp = phalf[:, 1:] - phalf[:, :-1]
    lwp = torch.where(lq, ql_l * dp / GRAV / qa_s, zero)
    iwp = torch.where(iq, qi_l * dp / GRAV / qa_s, zero)
    logp = torch.log(phalf[:, 1:] / torch.maximum(phalf[:, :-1], pfull[:, :1]))
    conc_drop = torch.where(lq, 1000.0 * ql_l * dp / RDGAS / temp / logp / qa_s, zero)
    conc_ice = torch.where(iq, 1000.0 * qi_l * dp / RDGAS / temp / logp / qa_s, zero)
    arg = pfull * ql_l / qa_s / RDGAS / temp / DENS_H2O / N2[:, None]
    reff = kr[:, None] * 620350.49 * torch.where(lq, arg, torch.ones_like(arg)) ** (1.0 / 3.0)
    if c.do_brenguier:
        lowa = qa < qamin            # the neighbours' (unfiltered) cloud fraction
        nb_top = torch.cat([lowa[:, 1:2], lowa[:, :-2] & lowa[:, 2:], lowa[:, kdim - 2:kdim - 1]], dim=1)
        reff = torch.where(nb_top, 1.134 * reff, reff)
    reff_liq = torch.where(lq, reff, torch.full_like(ql, 10.0))
    size_drop = torch.where(lq, 2.0 * reff, torch.full_like(ql, 20.0))
    if c.use_fu2007:
        d = temp - TFREEZE
        s_ice = 47.05 + 0.6624 * d + 0.001741 * d ** 2
    else:
        s_ice = _ice_table(temp, _ICE_SIZE_MICRO)
    size_ice = torch.where(iq, s_ice, torch.full_like(ql, 60.0))
    reff_ice = torch.where(iq, _ice_table(temp, _ICE_REFF), torch.full_like(ql, 30.0))
    cldamt = torch.where(cld, qa_l, zero)
    ncld = cld.sum(dim=1)
    conc_drop = c.scale_factor * conc_drop
    conc_ice = c.scale_factor * conc_ice
    cldamt = torch.clamp(cldamt, max=1.0)
    return dict(camtsw=cldamt, crndlw=cldamt, cmxolw=zero, ncldsw=ncld, nrndlw=ncld,
                nmxolw=torch.zeros_like(ncld), lwp=lwp, iwp=iwp, reff_liq=reff_liq, reff_ice=reff_ice,
                cloud_water=ql, cloud_ice=qi, cloud_area=qa, conc_drop=conc_drop, conc_ice=conc_ice,
                size_drop=size_drop, size_ice=size_ice, cldamt=cldamt, droplet_number=droplet_number)
