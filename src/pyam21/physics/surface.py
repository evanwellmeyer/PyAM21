"""Surface fluxes and the implicit atmosphere-surface coupling of the FMS coupler.

Translations (siena_201207):
  coupler/surface_flux.F90        surface_flux_1d (bulk fluxes via Monin-Obukhov drag)
  ice_param/ocean_rough.F90       compute_ocean_roughness ('beljaars', 'charnock', 'fixed')
  ice_param/ocean_albedo.F90      compute_ocean_albedo_new (options 3, 4, 5)
  coupler/flux_exchange.F90       the arithmetic of flux_down_from_atmos / flux_up_to_atmos
                                  for one surface tile whose skin temperature does not
                                  change within the step (prescribed SST, or a specified
                                  land skin temperature)

The exchange grid of the Fortran is not reproduced: every SCM case has one surface
type in its column (the 4-6 ocean exchange cells of the ocean cases hold identical
values), so one tile per column reproduces it up to the rounding of the area
average.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import CP_AIR, GRAV, HLV, RDGAS, RVGAS, STEFAN
from ..sat_vapor_pres import get_sat_vapor_pres
from .monin_obukhov import MoninObukhov
from ..core import spow, ssqrt

D622 = RDGAS / RVGAS
D378 = 1.0 - D622
KAPPA_SF = RDGAS / CP_AIR


def powi(x: torch.Tensor, n: int) -> torch.Tensor:
    """x**n for a small integer n as gfortran evaluates it (binary powering)."""
    result = None
    base = x
    while n > 0:
        if n & 1:
            result = base if result is None else result * base
        n >>= 1
        if n:
            base = base * base
    return result


@dataclass(frozen=True)
class SurfaceFluxConfig:
    """surface_flux_nml (not set in the SCM input.nml files, so Fortran defaults)."""
    no_neg_q: bool = False
    use_virtual_temp: bool = True
    alt_gustiness: bool = False
    old_dtaudv: bool = False
    use_mixing_ratio: bool = False
    gust_const: float = 1.0
    gust_min: float = 0.0
    ncar_ocean_flux: bool = False
    raoult_sat_vap: bool = False
    do_simple: bool = False

    @classmethod
    def from_nml(cls, nml: dict) -> "SurfaceFluxConfig":
        g = nml.get("surface_flux_nml", {})
        kw = {f: type(getattr(cls, f))(g[f]) for f in cls.__dataclass_fields__ if f in g}
        return cls(**kw)


class SurfaceFlux:
    DEL_TEMP = 0.1

    def __init__(self, config: SurfaceFluxConfig, mo: MoninObukhov):
        if config.ncar_ocean_flux or config.alt_gustiness:
            raise NotImplementedError("ncar_ocean_flux / alt_gustiness")
        self.c = config
        self.mo = mo
        self.d608 = D378 / D622 if config.use_virtual_temp else 0.0

    def __call__(self, t_atm, q_atm_in, u_atm, v_atm, p_atm, z_atm, p_surf, t_surf, t_ca, q_surf,
                 u_surf, v_surf, rough_mom, rough_heat, rough_moist, rough_scale, gust, land, seawater):
        """surface_flux_1d for available points. All inputs (ncol,); land/seawater bool.

        Returns a dict with the Fortran output names (q_surf is updated).
        """
        c = self.c
        svp = get_sat_vapor_pres()
        t_surf0 = torch.where(land, t_ca, t_surf)
        t_surf1 = t_surf0 + self.DEL_TEMP
        e_sat = svp.lookup_es(t_surf0)
        e_sat1 = svp.lookup_es(t_surf1)
        if c.use_mixing_ratio:
            q_sat = D622 * e_sat / (p_surf - e_sat)
            q_sat1 = D622 * e_sat1 / (p_surf - e_sat1)
        elif c.do_simple:
            q_sat = D622 * e_sat / p_surf
            q_sat1 = D622 * e_sat1 / p_surf
        else:
            q_sat = D622 * e_sat / (p_surf - D378 * e_sat)
            q_sat1 = D622 * e_sat1 / (p_surf - D378 * e_sat1)
        q_surf0 = torch.where(land, q_surf, q_sat)
        if c.raoult_sat_vap:
            q_surf0 = torch.where(seawater, 0.98 * q_surf0, q_surf0)
        q_atm = torch.clamp(q_atm_in, min=0.0) if c.no_neg_q else q_atm_in
        p_ratio = (p_surf / p_atm) ** KAPPA_SF
        tv_atm = t_atm * (1.0 + self.d608 * q_atm)
        th_atm = t_atm * p_ratio
        thv_atm = tv_atm * p_ratio
        thv_surf = t_surf0 * (1.0 + self.d608 * q_surf0)
        u_dif = u_surf - u_atm
        v_dif = v_surf - v_atm
        w_gust = torch.clamp(gust, min=c.gust_min) if c.gust_min > 0.0 else gust
        w_atm = ssqrt(u_dif * u_dif + v_dif * v_dif + w_gust * w_gust)
        dw_atmdu = u_dif / w_atm
        dw_atmdv = v_dif / w_atm
        cd_m, cd_t, cd_q, u_star, b_star = self.mo.drag(thv_atm, thv_surf, z_atm, rough_mom, rough_heat,
                                                        rough_moist, w_atm)
        cd_m = cd_m * (torch.log(z_atm / rough_mom + 1) / torch.log(z_atm / rough_scale + 1)) ** 2
        drag_t = cd_t * w_atm
        drag_q = cd_q * w_atm
        drag_m = cd_m * w_atm
        rho = p_atm / (RDGAS * tv_atm)
        rho_drag = CP_AIR * drag_t * rho
        flux_t = rho_drag * (t_surf0 - th_atm)
        dhdt_surf = rho_drag
        dhdt_atm = -rho_drag * p_ratio
        rho_drag = drag_q * rho
        flux_q = rho_drag * (q_surf0 - q_atm)
        zero = torch.zeros_like(t_atm)
        dedq_surf = torch.where(land, rho_drag, zero)
        dedt_surf = torch.where(land, zero, rho_drag * (q_sat1 - q_sat) * (1.0 / self.DEL_TEMP))
        dedq_atm = -rho_drag
        q_star = flux_q / (u_star * rho)
        q_surf_new = q_atm + flux_q / (rho * cd_q * w_atm)
        flux_r = STEFAN * powi(t_surf, 4)
        drdt_surf = 4 * STEFAN * powi(t_surf, 3)
        rho_drag = drag_m * rho
        flux_u = rho_drag * u_dif
        flux_v = rho_drag * v_dif
        if c.old_dtaudv:
            dtaudu_atm = -rho_drag
            dtaudv_atm = -rho_drag
        else:
            dtaudu_atm = -cd_m * rho * (dw_atmdu * u_dif + w_atm)
            dtaudv_atm = -cd_m * rho * (dw_atmdv * v_dif + w_atm)
        return dict(flux_t=flux_t, flux_q=flux_q, flux_r=flux_r, flux_u=flux_u, flux_v=flux_v,
                    cd_m=cd_m, cd_t=cd_t, cd_q=cd_q, w_atm=w_atm, u_star=u_star, b_star=b_star,
                    q_star=q_star, dhdt_surf=dhdt_surf, dedt_surf=dedt_surf, dedq_surf=dedq_surf,
                    drdt_surf=drdt_surf, dhdt_atm=dhdt_atm, dedq_atm=dedq_atm,
                    dtaudu_atm=dtaudu_atm, dtaudv_atm=dtaudv_atm, q_surf=q_surf_new)


# ---------------------------------------------------------------------------------------------
# ocean roughness and albedo
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OceanRoughConfig:
    roughness_min: float = 1.0e-6
    charnock: float = 0.032
    roughness_mom: float = 5.8e-5
    roughness_heat: float = 5.8e-5
    roughness_moist: float = 5.8e-5
    zcoh1: float = 0.0
    zcoq1: float = 0.0
    do_highwind: bool = False
    do_cap40: bool = False
    rough_scheme: str = "fixed"

    @classmethod
    def from_nml(cls, nml: dict) -> "OceanRoughConfig":
        g = nml.get("ocean_rough_nml", {})
        kw = {f: type(getattr(cls, f))(g[f]) for f in cls.__dataclass_fields__ if f in g}
        cfg = cls(**kw)
        # ocean_rough_init
        return cls(**{**cfg.__dict__,
                      "roughness_moist": max(cfg.roughness_moist, cfg.roughness_min),
                      "roughness_heat": max(cfg.roughness_heat, cfg.roughness_min),
                      "roughness_mom": max(cfg.roughness_mom, cfg.roughness_min)})


ZCOM1, GNU, ZCOM2, ZCOH2, ZCOQ2 = 1.8e-2, 1.5e-5, 0.11, 0.40, 0.62


def fixed_ocean_roughness(c: OceanRoughConfig, like: torch.Tensor):
    return (torch.full_like(like, c.roughness_mom), torch.full_like(like, c.roughness_heat),
            torch.full_like(like, c.roughness_moist))


def compute_ocean_roughness(c: OceanRoughConfig, u_star: torch.Tensor):
    """Roughness lengths (mom, heat, moist) over open ocean from u*."""
    if c.rough_scheme == "fixed":
        return fixed_ocean_roughness(c, u_star)
    ustar2 = torch.clamp(u_star * u_star, min=GNU * GNU)
    xx1 = GNU / torch.sqrt(ustar2)
    xx2 = ustar2 / GRAV
    if c.rough_scheme == "charnock":
        rm = torch.clamp(c.charnock * xx2, min=c.roughness_min)
        return rm, rm, rm
    if c.rough_scheme != "beljaars":
        raise ValueError(c.rough_scheme)
    if c.do_highwind:
        w10 = 2.458 + u_star * (20.255 - 0.56 * u_star)
        rm = torch.where(w10 > 12.5, 0.001 * (0.085 * w10 - 0.58), 0.0185 / GRAV * u_star ** 2)
        if c.do_cap40:
            rm = torch.where(w10 > 12.5, torch.clamp(rm, max=2.82e-3), rm)
    else:
        rm = ZCOM1 * xx2 + ZCOM2 * xx1
    rh = c.zcoh1 * xx2 + ZCOH2 * xx1
    rq = c.zcoq1 * xx2 + ZCOQ2 * xx1
    return (torch.clamp(rm, min=c.roughness_min), torch.clamp(rh, min=c.roughness_min),
            torch.clamp(rq, min=c.roughness_min))


def compute_ocean_albedo(coszen: torch.Tensor, option: int = 1, const_alb: float = 0.10):
    """compute_ocean_albedo_new -> (vis_dir, vis_dif, nir_dir, nir_dif)."""
    if option == 3:
        nz = coszen != 0.0
        cos14 = torch.where(nz, torch.where(nz, coszen, torch.ones_like(coszen)) ** 1.4, torch.zeros_like(coszen))
        a = 0.037 / (1.1 * cos14 + 0.15)
        return a, a, a, a
    if option == 4:
        a = torch.full_like(coszen, const_alb)
        return a, a, a, a
    if option == 5:
        pos = coszen >= 0.0
        cz = torch.clamp(coszen, min=0.0)
        vd = torch.where(pos, 0.026 / (cz ** 1.7 + 0.065) + 0.15 * (cz - 0.10) * (cz - 0.5) * (cz - 1.0),
                         torch.full_like(coszen, 0.4075))
        d = torch.full_like(coszen, 0.06)
        return vd, d, vd, d
    raise NotImplementedError(f"ocean_albedo_option {option}")


# ---------------------------------------------------------------------------------------------
# implicit coupling (flux_down_from_atmos / flux_up_to_atmos arithmetic)
# ---------------------------------------------------------------------------------------------
def implicit_surface_coupling(dtmass, delta_t, dflux_t, delta_q, dflux_q, delta_u, delta_v,
                              flux_t, flux_q, flux_u, flux_v, dhdt_surf, dedt_surf, dedq_surf,
                              dhdt_atm, dedq_atm, dtaudu_atm, dtaudv_atm):
    """Complete the implicit surface flux step for a tile with fixed skin temperature.

    Inputs are the lowest-level quantities from gcm_vert_diff_down (Surf_diff) and the
    surface-flux results. With dt_t_surf = dt_t_ca = 0 (fixed SST, or a specified land
    skin temperature) and a sphum surface value that is not changed by a surface
    model, flux_up_to_atmos gives the atmosphere delta_t = f_t_delt_n and
    delta_q = f_q_delt_n. Returns the new Surf_diff deltas and the updated fluxes.
    """
    cp_inv = 1.0 / CP_AIR
    flux_u = flux_u + delta_u * dtaudu_atm
    flux_v = flux_v + delta_v * dtaudv_atm
    gamma = 1.0 / (1.0 - dtmass * (dflux_t + dhdt_atm * cp_inv))
    e_t_n = dtmass * dhdt_surf * cp_inv * gamma
    f_t_delt_n = (delta_t + dtmass * flux_t * cp_inv) * gamma
    flux_t = flux_t + dhdt_atm * f_t_delt_n
    dhdt_surf = dhdt_surf + dhdt_atm * e_t_n
    gamma_q = 1.0 / (1.0 - dtmass * (dflux_q + dedq_atm))
    e_q_n = dtmass * dedt_surf * gamma_q
    dedt_surf = dedt_surf + dedq_atm * e_q_n
    e_tr_n = dtmass * dedq_surf * gamma_q
    f_q_delt_n = (delta_q + dtmass * flux_q) * gamma_q
    flux_q = flux_q + dedq_atm * f_q_delt_n
    dedq_surf = dedq_surf + dedq_atm * e_tr_n
    return dict(delta_t=f_t_delt_n, delta_q=f_q_delt_n, flux_t=flux_t, flux_q=flux_q,
                flux_u=flux_u, flux_v=flux_v, e_t_n=e_t_n, e_q_n=e_q_n, e_tr_n=e_tr_n,
                dhdt_surf=dhdt_surf, dedt_surf=dedt_surf, dedq_surf=dedq_surf)
