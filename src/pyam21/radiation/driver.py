"""Radiation driver (atmos_param/radiation_driver/radiation_driver.F90, siena_201207).

The sea_esf path as the AM2.1 SCM uses it: radiation every rad_time_step (alarms), the
solar geometry averaged over the radiation step, the atmospheric input fields
(define_atmos_input_fields + calculate_auxiliary_variables), the package call, the
output assembly of radiation_calc, and renormalize_sw_fluxes: on every step the saved
shortwave fluxes and heating are rescaled by the ratio of the current to the saved
insolation (update_rad_fields). coszen for the surface is the zenith angle averaged
over the next radiation interval.

The radiation package is a callable
    package(atmos: dict, surface: dict, astro: dict, time, do_lw, do_sw) -> dict
returning the sea_esf outputs (heatra, flxnet[, heatracf, flxnetcf] in K/day and W/m2;
hsw, dfsw, ufsw, fsw[, hswcf, ...]; dfsw_dir_sfc, dfsw_dif_sfc, ufsw_dir_sfc,
ufsw_dif_sfc, dfsw_vis_sfc, ufsw_vis_sfc, dfsw_vis_sfc_dir, dfsw_vis_sfc_dif,
ufsw_vis_sfc_dir, ufsw_vis_sfc_dif), each (ncol, ...) with the nzens axis dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import GRAV, RDGAS, RVGAS, STEFAN
from ..sat_vapor_pres import get_sat_vapor_pres
from ..time_manager import ModelTime, set_time
from .astronomy import Astronomy, AstronomyConfig

SECONDS_PER_DAY = 86400.0
D608 = (RVGAS - RDGAS) / RDGAS


@dataclass(frozen=True)
class RadiationDriverConfig:
    rad_time_step: int = 14400
    lw_rad_time_step: int = 0
    sw_rad_time_step: int = 0
    renormalize_sw_fluxes: bool = False
    do_clear_sky_pass: bool = False
    zenith_spec: str = "diurnally_varying"
    rad_package: str = "sea_esf"
    use_mixing_ratio: bool = False
    rh2o_lower_limit: float = 2.0e-07      # rh2o_lower_limit_seaesf
    temp_lower_limit: float = 100.0
    temp_upper_limit: float = 370.0
    log_p_at_top: float = 2.0

    @classmethod
    def from_nml(cls, nml: dict) -> "RadiationDriverConfig":
        g = nml.get("radiation_driver_nml", {})
        kw = {}
        for f in ("rad_time_step", "lw_rad_time_step", "sw_rad_time_step"):
            if f in g:
                kw[f] = int(g[f])
        for f in ("renormalize_sw_fluxes", "do_clear_sky_pass", "use_mixing_ratio"):
            if f in g:
                kw[f] = bool(g[f])
        for f in ("zenith_spec", "rad_package"):
            if f in g:
                kw[f] = str(g[f]).strip()
        c = cls(**kw)
        if c.rad_package != "sea_esf":
            raise NotImplementedError(f"rad_package {c.rad_package}")
        if c.zenith_spec != "diurnally_varying":
            raise NotImplementedError(f"zenith_spec {c.zenith_spec}")
        return c

    @property
    def lw_step(self):
        return self.lw_rad_time_step or self.rad_time_step

    @property
    def sw_step(self):
        return self.sw_rad_time_step or self.rad_time_step


def atmos_input(c: RadiationDriverConfig, pfull, phalf, t, q, t_surf, do_rad=True):
    """define_atmos_input_fields (+ calculate_auxiliary_variables): the radiation
    column with the surface appended as level kmax+1."""
    kmax = t.shape[1]
    press = torch.cat([pfull, phalf[:, -1:]], dim=1)
    temp = torch.cat([t, t_surf[:, None]], dim=1)
    rh2o = q if c.use_mixing_ratio else q / (1.0 - q)
    cloudvapor = rh2o
    aerosolvapor = rh2o
    cloudtemp, aerosoltemp = t, t
    if do_rad:
        lo, hi = c.temp_lower_limit, c.temp_upper_limit
        rh2o = torch.clamp(rh2o, min=c.rh2o_lower_limit)
        cloudvapor = torch.clamp(cloudvapor, min=c.rh2o_lower_limit)
        aerosolvapor = torch.clamp(aerosolvapor, min=c.rh2o_lower_limit)
        temp = torch.cat([torch.clamp(torch.clamp(temp[:, :kmax], min=lo), max=hi), temp[:, kmax:]], dim=1)
        cloudtemp = torch.clamp(torch.clamp(cloudtemp, min=lo), max=hi)
        aerosoltemp = torch.clamp(torch.clamp(aerosoltemp, min=lo), max=hi)
    aerosolpress = torch.cat([pfull, phalf[:, -1:]], dim=1)
    out = dict(press=press, phalf=phalf, temp=temp, rh2o=rh2o, cloudtemp=cloudtemp, cloudvapor=cloudvapor,
               aerosoltemp=aerosoltemp, aerosolvapor=aerosolvapor, aerosolpress=aerosolpress,
               psfc=phalf[:, -1], tsfc=t_surf)
    if not do_rad:
        return out
    # calculate_auxiliary_variables
    pflux = torch.cat([torch.zeros_like(press[:, :1]), 0.5 * (press[:, :kmax - 1] + press[:, 1:kmax]),
                       press[:, kmax:kmax + 1]], dim=1)
    tflux = torch.cat([temp[:, :1], 0.5 * (temp[:, :kmax - 1] + temp[:, 1:kmax]), temp[:, kmax:kmax + 1]], dim=1)
    tv = temp[:, :kmax] * (1.0 + D608 * rh2o)
    dz_top = c.log_p_at_top * RDGAS * tv[:, :1] / GRAV
    deltaz = torch.cat([dz_top, torch.log(pflux[:, 2:] / pflux[:, 1:kmax]) * RDGAS * tv[:, 1:] / GRAV], dim=1)
    tv = cloudtemp * (1.0 + D608 * cloudvapor)
    cdz = torch.cat([c.log_p_at_top * RDGAS * tv[:, :1] / GRAV,
                     torch.log(pflux[:, 2:] / pflux[:, 1:kmax]) * RDGAS * tv[:, 1:] / GRAV], dim=1)
    svp = get_sat_vapor_pres()
    qv = rh2o / (1.0 + rh2o)
    qsat = svp.compute_qs(temp[:, :kmax], press[:, :kmax], q=qv)
    rel_hum = torch.clamp(qv / qsat, max=1.0)
    qv = aerosolvapor / (1.0 + aerosolvapor)
    qsat = svp.compute_qs(aerosoltemp, aerosolpress[:, :kmax], q=qv)
    aerosolrelhum = torch.clamp(qv / qsat, max=1.0)
    out.update(pflux=pflux, tflux=tflux, deltaz=deltaz, clouddeltaz=cdz, rel_hum=rel_hum,
               aerosolrelhum=aerosolrelhum)
    return out


class RadiationDriver:
    """Callable as the SCM's radiation: driver(model, ctx) -> dict(tdt_rad, tdtlw, coszen, ...)."""

    def __init__(self, config: RadiationDriverConfig, package, astronomy: Astronomy | None = None):
        c = self.c = config
        if c.lw_step % 1 or c.sw_step != c.lw_step:
            raise NotImplementedError("different lw and sw radiation time steps")
        self.package = package
        self.astro = astronomy or Astronomy()
        self.lw_alarm = 1
        self.sw_alarm = 1
        self.saved = None               # fields kept between radiation steps
        self.coszen = None

    @classmethod
    def from_nml(cls, nml, package):
        return cls(RadiationDriverConfig.from_nml(nml), package, Astronomy(AstronomyConfig.from_nml(nml)))

    def for_physics(self):
        """The callable AM2Physics expects: radiation(time, dt, ctx) -> dict."""
        return lambda time, dt, ctx: self.step(time, int(dt), ctx["lat"], ctx["lon"], ctx["pfull"], ctx["phalf"],
                                               ctx["T"], ctx["q"], ctx["t_surf"], ctx["surface"], extra=ctx)

    def __call__(self, model, ctx):
        return self.step(model.time, int(model.dt), ctx["lat"], ctx["lon"], ctx["pfull"], ctx["phalf"],
                         ctx["T"], ctx["q"], ctx["t_surf"], ctx["surface"], extra=ctx)

    def step(self, time: ModelTime, dt: int, lat, lon, pfull, phalf, t, q, t_surf, surface, extra=None):
        c = self.c
        # define_rad_times
        self.lw_alarm -= dt
        self.sw_alarm -= dt
        do_lw = self.lw_alarm <= 0
        do_sw = self.sw_alarm <= 0
        do_rad = do_lw or do_sw
        # obtain_astronomy_variables (diurnally varying, renormalize_sw_fluxes)
        clamp1 = lambda f: torch.clamp(f, max=1.0)
        astro_r = astro_p = None
        if do_sw:
            cz, fr, rr = self.astro.diurnal_solar(lat, lon, time, c.sw_step)
            fr = clamp1(fr)
            astro_r = dict(cosz=cz, fracday=fr, rrsun=rr, solar=cz * fr * rr)
        if c.renormalize_sw_fluxes:
            cz, fr, rr = self.astro.diurnal_solar(lat, lon, time, dt)
            fr = clamp1(fr)
            astro_p = dict(cosz=cz, fracday=fr, rrsun=rr, solar=cz * fr * rr)
        astro = astro_r if (do_sw or not c.renormalize_sw_fluxes) else astro_p
        if do_sw:
            cza, _, _ = self.astro.diurnal_solar(lat, lon, time + set_time(c.sw_step), c.sw_step)
            self.coszen = cza
        atm = atmos_input(c, pfull, phalf, t, q, t_surf, do_rad=do_rad)
        out = None
        if do_rad:
            out = self.package(atm, surface, astro, time, do_lw, do_sw, extra)
            self._radiation_calc(out, atm, do_lw, do_sw)
        flux_ratio = self._update_rad_fields(astro, astro_p, do_sw)
        s = self.saved
        # radiation_driver_endts
        if do_lw:
            self.lw_alarm += c.lw_step
        if do_sw:
            self.sw_alarm += c.sw_step
        return dict(tdt_rad=s["tdt_rad"], tdtlw=s["tdtlw"], coszen=self.coszen, flux_ratio=flux_ratio,
                    flux_sw_surf=s["flux_sw_surf"], flux_lw_surf=s["flux_lw_surf"], dfsw=s["dfsw"],
                    ufsw=s["ufsw"], tdtsw=s["tdtsw"], flxnet=s["flxnet"], do_rad=do_rad, package_out=out,
                    atmos=atm)

    def _radiation_calc(self, o, atm, do_lw, do_sw):
        s = self.saved = self.saved or {}
        kmax = atm["temp"].shape[1] - 1
        if do_sw:
            s["tdtsw"] = o["hsw"] / SECONDS_PER_DAY
            s["ufsw"], s["dfsw"] = o["ufsw"], o["dfsw"]
        if do_lw:
            s["tdtlw"] = o["heatra"] / SECONDS_PER_DAY
            s["flxnet"] = o["flxnet"]
        s["tdt_rad"] = s["tdtsw"] + s["tdtlw"]
        if do_sw:
            s["flux_sw_surf"] = o["dfsw"][:, kmax] - o["ufsw"][:, kmax]
            for k in ("dfsw_dir_sfc", "ufsw_dir_sfc", "dfsw_dif_sfc", "ufsw_dif_sfc", "dfsw_vis_sfc",
                      "ufsw_vis_sfc", "dfsw_vis_sfc_dir", "dfsw_vis_sfc_dif", "ufsw_vis_sfc_dir",
                      "ufsw_vis_sfc_dif"):
                if k in o:
                    s["sfc_" + k] = o[k]
        if do_lw:
            s["flux_lw_surf"] = STEFAN * atm["temp"][:, kmax] ** 4 - o["flxnet"][:, kmax]

    def _update_rad_fields(self, astro, astro_p, do_sw):
        c = self.c
        s = self.saved
        if not c.renormalize_sw_fluxes:
            return None
        if do_sw:
            s["solar_save"] = astro["solar"]
            s["sw_heating_save"] = s["tdtsw"]
            s["tot_heating_save"] = s["tdt_rad"]
            s["ufsw_save"], s["dfsw_save"] = s["ufsw"], s["dfsw"]
            s["flux_sw_surf_save"] = s["flux_sw_surf"]
            cur = astro_p["solar"]
        else:
            cur = astro["solar"]
        sv = s["solar_save"]
        ratio = torch.where(sv != 0.0, cur / torch.where(sv != 0.0, sv, torch.ones_like(sv)), torch.zeros_like(sv))
        tdtlw = s["tot_heating_save"] - s["sw_heating_save"]
        s["tdtsw"] = s["sw_heating_save"] * ratio[:, None]
        s["ufsw"] = s["ufsw_save"] * ratio[:, None]
        s["dfsw"] = s["dfsw_save"] * ratio[:, None]
        s["tdt_rad"] = tdtlw + s["tdtsw"]
        s["flux_sw_surf"] = ratio * s["flux_sw_surf_save"]
        return ratio
