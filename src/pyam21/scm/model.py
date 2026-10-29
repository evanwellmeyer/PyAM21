"""The single column model: one coupled time step of the GFDL SCM with AM2.1 physics.

Order of a step (coupler_main -> atmos_model -> atmosphere / physics_driver):
  1. sfc_boundary_layer   surface fluxes from the lowest-level state (surface_flux,
                          SCM specified-surface overrides)
  2. atmosphere_down      SCM forcing (scm_core_driver), pressures and heights, then
                          physics_driver_down: radiation, damping, turbulence,
                          vertical diffusion (down sweep)
  3. flux_down/up         implicit surface coupling; ocean roughness and albedo update
  4. atmosphere_up        physics_driver_up: vertical diffusion (up sweep), moist
                          processes; SCM damping above pdamp, vapour filler, forward step

The model state and the physics memory are plain tensors held in `self.state` and
`self.mem`; `step()` advances them by dt. Everything is (ncol, nlev) with index 0 at
the top, so an ensemble of columns runs in one call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..constants import GRAV, STEFAN
from ..physics.am2 import ALBEDO_BANDS, AM2Physics
from ..time_manager import ModelTime, set_time
from .site import land_rough_scale, site_hprime
from .case import CaseConfig, load_case, make_forcing
from .vert_grid import compute_height, compute_height_bottom, compute_pres_full, compute_pres_half

SPHUM, LIQ, ICE, CLD = 0, 1, 2, 3


def build_radiation(cfg: CaseConfig):
    """The sea_esf radiation of the case: radiation_driver with the full package, set up on
    the reference pressures physics_driver_init receives (scm_physics.F90)."""
    import os
    from ..radiation.driver import RadiationDriver
    from ..radiation.sea_esf import SeaEsfRad
    from .case import DATA_ROOT
    from .site import box_edges
    ps = torch.tensor([101325., 81060.], dtype=torch.float64)
    pf = compute_pres_full(cfg.vgrid, ps).numpy()
    pref = np.stack([np.append(pf[0], 101325.), np.append(pf[1], 81060.)], axis=1)
    lo, la = box_edges(cfg)
    drv_nml = cfg.nml.get("radiation_driver_nml", {})
    package = SeaEsfRad(cfg.nml, os.path.join(DATA_ROOT, "sea_esf"), [(tuple(lo), tuple(la))], pref,
                        do_totcld_forcing=bool(drv_nml.get("do_clear_sky_pass", False)))
    return RadiationDriver.from_nml(cfg.nml, package)


class SCM:
    def __init__(self, case: str | CaseConfig, ncol: int = 1, device="cpu", dtype=torch.float64,
                 hprime: float | None = None, radiation=None):
        self.case = load_case(case) if isinstance(case, str) else case
        cfg = self.case
        nml = cfg.nml
        self.device, self.dtype = torch.device(device), dtype
        self.ncol = ncol
        self.dt = float(cfg.dt)
        self.forcing = make_forcing(cfg, device, dtype)
        fnml = nml["scm_forc_nml"]
        self.use_scm_rad = bool(fnml.get("use_scm_rad", False))
        self.do_specified_flux = bool(fnml.get("do_specified_flux", False))
        self.do_specified_land = bool(fnml.get("do_specified_land", False))
        pd = nml.get("physics_driver_nml", {})
        self.do_radiation = bool(pd.get("do_radiation", True))
        self.diffusion_smooth = bool(pd.get("diffusion_smooth", True))
        self.tau_diff = float(pd.get("tau_diff", 3600.0))
        self.diff_min = float(pd.get("diff_min", 1.0e-3))
        # radiation(model, ctx) -> dict(tdt_rad, tdtlw, coszen): by default the sea_esf
        # package (build_radiation), or a stand-in such as pyam21.testing.DumpRadiation
        if self.do_radiation and radiation is None:
            radiation = build_radiation(cfg)
        self.radiation = radiation
        n = ncol
        lat = torch.full((n,), cfg.lat, dtype=dtype, device=self.device)
        lon = torch.full((n,), cfg.lon, dtype=dtype, device=self.device)
        rad = None if radiation is None else (lambda time, dt, ctx: self.radiation(self, ctx))
        self.physics = AM2Physics(nml, lat, lon, radiation=rad)
        self.time = cfg.start
        self.step_count = 0
        self._init_state(hprime)

    # ------------------------------------------------------------------------------------------
    def T(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def _init_state(self, hprime):
        n = self.ncol
        st = self.forcing.initial_state(self.time)
        rep = lambda x: x.expand(n, *x.shape[1:]).clone()
        self.state = dict(T=rep(st["T"]), u=rep(st["u"]), v=rep(st["v"]), tracers=rep(st["tracers"]),
                          ps=rep(st["ps"]), elev=rep(st["elev"]))
        nlev = self.case.vgrid.nlev
        self.mem = AM2Physics.init_memory(n, nlev, self.dtype, self.device)
        z1 = torch.zeros(n, dtype=self.dtype, device=self.device)
        # surface (ice_amip initial state for the open ocean)
        ocean = not self.do_specified_land
        self.surf = self.physics.init_ocean_surface(n, self.dtype, self.device,
                                                    land=torch.full((n,), not ocean, dtype=torch.bool,
                                                                    device=self.device))
        if hprime is None:
            hprime = site_hprime(self.case)
        self.hprime = z1 + hprime
        self.physics.hprime = self.hprime
        spec = self.forcing.surface
        if self.do_specified_land:
            if spec.rough_mom is None:
                raise NotImplementedError("specified land without specified roughness (LM2 land model)")
            rs = land_rough_scale(self.case, spec.rough_mom, hprime)
            self.surf.update(rough_mom=z1 + spec.rough_mom, rough_heat=z1 + spec.rough_heat,
                             rough_moist=z1 + spec.rough_heat, rough_scale=z1 + rs)
            if spec.albedo_obs is not None:
                self.surf["albedo"] = z1 + spec.albedo_obs

    # ------------------------------------------------------------------------------------------
    def surface_temperature(self, t_bot=None):
        """Skin temperature of the tile: the data_table SST override or the forcing's TSKIN.

        Over an open ocean with specified fluxes the (AMIP climatological) SST never reaches
        the atmosphere: every flux, u*, b* and q* is replaced and the skin temperature does
        not change within the step, so the lowest-level temperature stands in for it."""
        f = self.forcing
        if getattr(f.surface, "tskin", None) is not None:
            return torch.full((self.ncol,), float(f.surface.tskin), dtype=self.dtype, device=self.device)
        sst = getattr(self.case, "sst_override", None)
        if sst is None and self.do_specified_flux and not self.do_specified_land and t_bot is not None:
            return t_bot
        if sst is None:
            raise RuntimeError("no surface temperature for this case")
        return torch.full((self.ncol,), sst, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------------------------------
    def step(self):
        cfg, vg, dt = self.case, self.case.vgrid, self.dt
        s, m, sf, ph = self.state, self.mem, self.surf, self.physics
        T0, u0, v0, R0, ps = s["T"], s["u"], s["v"], s["tracers"], s["ps"]
        nlev = vg.nlev
        # ---------------- 1. surface boundary layer ------------------------------------------
        t_bot, q_bot = T0[:, -1], R0[:, -1, SPHUM]
        z_bot, p_bot = compute_height_bottom(vg, ps, t_bot, q_bot, kbot=nlev)
        t_surf = self.surface_temperature(t_bot)
        q_surf_in = None if not self.do_specified_land else torch.full_like(q_bot, 15.0e-3)
        override = None
        if self.do_specified_flux:
            override = lambda out, ctx: self.forcing.specified_surface_flux(ctx)
        out = ph.surface_layer(T0, u0, v0, R0, ps, z_bot, p_bot, t_surf, sf, m, override=override,
                               q_surf_in=q_surf_in)
        tr = self.trace = dict(t_bot=t_bot, q_bot=q_bot, z_bot=z_bot, p_bot=p_bot, sfc=dict(out))
        # ---------------- 2. atmosphere_down: SCM forcing, physics_driver_down ----------------
        ft = self.forcing.update(self.time, self.time + set_time(int(dt)), dt,
                                 dict(T=T0, u=u0, v=v0, tracers=R0, elev=s["elev"]))
        ps = ft.ps
        if ft.state is not None:                           # forcing changed the state in place
            T0, R0 = ft.state.get("T", T0), ft.state.get("tracers", R0)
        pfull = compute_pres_full(vg, ps)
        phalf = compute_pres_half(vg, ps)
        zfull, zhalf = compute_height(vg, GRAV * s["elev"], T0, R0[..., SPHUM], pfull, phalf)
        lw_extra = None
        if self.use_scm_rad:
            f = self.forcing
            lw_extra = lambda lw, rt: (f.add_tdtlw(lw), f.add_tdtsw(f.add_tdtlw(rt)))
        dn = ph.down(dt, self.time, T0, u0, v0, R0, pfull, phalf, zfull, zhalf, ft.u, ft.v, ft.T, ft.tracers,
                     out, sf, m, t_surf, lw_extra=lw_extra)
        tr.update(forc_tdt=ft.T, pfull=pfull, zfull=zfull, zhalf=zhalf, **dn["diag"])
        # ---------------- 3. implicit surface coupling ----------------------------------------
        cpl = ph.couple(dn["vd"], out, sf, update_ocean=not self.do_specified_land)
        if self.do_specified_land and self.forcing.surface.albedo_obs is not None:
            # specified land: the observed albedo of this step reaches the radiation next step
            sf["albedo"] = torch.full_like(sf["albedo"], self.forcing.surface.albedo_obs)
        # ---------------- 4. atmosphere_up -----------------------------------------------------
        up = ph.up(dt, T0, u0, v0, R0, pfull, phalf, zfull, zhalf, ft.omega, dn, cpl, sf, m)
        tdt, rdt, udt, vdt = up["tdt"], up["rdt"], up["udt"], up["vdt"]
        mo = up["moist"]
        tr.update(vd=dn["vd"], cpl=cpl, up_tdt=up["up_tdt"], up_rdt=up["up_rdt"], moist=mo)
        # SCM: remove T and q tendencies above pdamp, vapour filler, forward step
        above = pfull < cfg.pdamp
        tdt = tdt + torch.where(above, -tdt, torch.zeros_like(tdt))
        qdt = rdt[..., SPHUM]
        qdt = qdt + torch.where(above, -qdt, torch.zeros_like(qdt))
        qn = R0[..., SPHUM] + qdt * dt
        fill_q = torch.where(qn < 0.5e-15, (0.5e-15 - R0[..., SPHUM] - qdt * dt) / dt, torch.zeros_like(qdt))
        qdt = qdt + fill_q
        rdt = torch.cat([qdt[..., None], rdt[..., 1:]], dim=-1)
        self.state = dict(T=T0 + dt * tdt, u=u0 + dt * udt, v=v0 + dt * vdt, tracers=R0 + dt * rdt,
                          ps=ps, elev=s["elev"])
        self.diag = dict(lprec=mo["lprec"], fprec=mo["fprec"], u_star=out["u_star"], b_star=out["b_star"],
                         flux_t=cpl["flux_t"], flux_q=cpl["flux_q"], pfull=pfull, zfull=zfull,
                         z_pbl=dn["vt"]["z_pbl"], omega=ft.omega)
        self.time = self.time + set_time(int(dt))
        self.step_count += 1
        return self.state
