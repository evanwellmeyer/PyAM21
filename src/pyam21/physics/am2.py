"""AM2.1 column physics as a torch module, for the SCM and for a host dynamical core.

The sequence of one physics step follows the FMS coupler / physics_driver:

  surface_layer  sfc_boundary_layer: surface fluxes and their derivatives from the
                 lowest model level and the surface state
  down           physics_driver_down: radiation, gravity-wave/Rayleigh damping, vertical
                 turbulence (entrain + stable_bl), the down sweep of the implicit vertical
                 diffusion
  couple         flux_down/up with a prescribed surface temperature (open ocean or a
                 specified land skin temperature), then the ocean roughness and albedo
                 update of ice_amip
  up             physics_driver_up: the up sweep of the vertical diffusion and the moist
                 processes (RAS, cumulus momentum transport, strat_cloud)

``forward`` runs the four in order for a host model that supplies the state, the
geometry and the dynamical tendencies; the SCM calls them separately because its
forcing acts between the surface layer and physics_driver_down.

Shapes: (ncol, nlev) with index 0 at the model top, tracers (ncol, nlev, ntracer) in the
order sphum, liq_wat, ice_wat, cld_amt. The physics memory (``mem``) and surface state
(``surf``) are plain dicts of tensors that the caller keeps between steps; float64 on CPU
reproduces the Fortran, float32 runs on MPS.
"""
from __future__ import annotations

import torch

from ..core import ssqrt
from .damping import DampingConfig, DampingDriver
from .entrain import EntrainConfig
from .moist import MoistProcesses
from .monin_obukhov import MoninObukhov, MoninObukhovConfig
from .stable_bl_turb import StableBLConfig
from .surface import (OceanRoughConfig, SurfaceFlux, SurfaceFluxConfig, compute_ocean_albedo,
                      compute_ocean_roughness, fixed_ocean_roughness, implicit_surface_coupling)
from .vert_diff import VertDiff, VertDiffConfig
from .vert_turb import VertTurbConfig, VertTurbDriver

SPHUM, LIQ, ICE, CLD = 0, 1, 2, 3
ALBEDO_BANDS = ("asfc_vis_dir", "asfc_nir_dir", "asfc_vis_dif", "asfc_nir_dif")


class AM2Physics(torch.nn.Module):
    def __init__(self, nml: dict, lat, lon, hprime=0.0, radiation=None):
        """nml: the parsed input.nml; lat, lon (ncol,) [radians]; hprime (ncol,) or scalar
        [m], the subgrid orography standard deviation for the gravity-wave drag;
        radiation: None (no radiation), or a callable radiation(time, dt, ctx) -> dict with
        tdt_rad, tdtlw, coszen (e.g. radiation.driver.RadiationDriver.for_physics)."""
        super().__init__()
        pd = nml.get("physics_driver_nml", {})
        self.diffusion_smooth = bool(pd.get("diffusion_smooth", True))
        self.tau_diff = float(pd.get("tau_diff", 3600.0))
        self.diff_min = float(pd.get("diff_min", 1.0e-3))
        self.mo = MoninObukhov(MoninObukhovConfig.from_nml(nml))
        self.sflux = SurfaceFlux(SurfaceFluxConfig.from_nml(nml), self.mo)
        self.rough_cfg = OceanRoughConfig.from_nml(nml)
        self.ocean_albedo_option = int(nml.get("ocean_albedo_nml", {}).get("ocean_albedo_option", 1))
        self.damping = DampingDriver(DampingConfig.from_nml(nml))
        self.vturb = VertTurbDriver(VertTurbConfig.from_nml(nml), EntrainConfig.from_nml(nml),
                                    StableBLConfig.from_nml(nml), self.mo)
        self.vdiff = VertDiff(VertDiffConfig.from_nml(nml))
        self.moist = MoistProcesses.from_nml(nml)
        self.radiation = radiation
        self.lat, self.lon = lat, lon
        self.hprime = hprime

    # ------------------------------------------------------------------------------ state
    @staticmethod
    def init_memory(ncol, nlev, dtype=torch.float64, device="cpu") -> dict:
        z3 = torch.zeros(ncol, nlev, dtype=dtype, device=device)
        z1 = torch.zeros(ncol, dtype=dtype, device=device)
        return dict(diff_cu_mo=z3.clone(), convect=torch.zeros(ncol, dtype=torch.bool, device=device),
                    radturbten=z3.clone(), pbltop=z1 - 999.0, gust=z1 + 1.0, diff_t=z3.clone(), diff_m=z3.clone())

    def init_ocean_surface(self, ncol, dtype=torch.float64, device="cpu", land=None) -> dict:
        """ice_amip's initial open-ocean state: albedo 0.14, roughness from the fixed values."""
        z1 = torch.zeros(ncol, dtype=dtype, device=device)
        land = torch.zeros(ncol, dtype=torch.bool, device=device) if land is None else land
        s = dict(land=land, albedo=z1 + 0.14, coszen=z1.clone())
        rm, rh, rq = fixed_ocean_roughness(self.rough_cfg, z1)
        s.update(rough_mom=rm, rough_heat=rh, rough_moist=rq, rough_scale=rm)
        return s

    # ------------------------------------------------------------------------------ 1
    def surface_layer(self, T, u, v, R, ps, z_bot, p_bot, t_surf, surf, mem, override=None, q_surf_in=None):
        """sfc_boundary_layer. override(out, ctx) -> dict of replaced fields (the SCM's
        specified surface fluxes); ctx carries the inputs and a surface_flux callable."""
        t_bot, q_bot = T[:, -1], R[:, -1, SPHUM]
        land = surf["land"]
        seawater = ~land
        q_surf_in = q_bot if q_surf_in is None else q_surf_in
        z = torch.zeros_like(t_bot)
        out = self.sflux(t_bot, q_bot, u[:, -1], v[:, -1], p_bot, z_bot, ps, t_surf, t_surf, q_surf_in,
                         z, z, surf["rough_mom"], surf["rough_heat"], surf["rough_moist"], surf["rough_scale"],
                         mem["gust"], land, seawater)
        if override is not None:
            def surface_flux(t_s, t_c, q_a, q_s):
                return self.sflux(t_bot, q_a, u[:, -1], v[:, -1], p_bot, z_bot, ps, t_s, t_c, q_s,
                                  torch.zeros_like(t_bot), torch.zeros_like(t_bot), surf["rough_mom"],
                                  surf["rough_heat"], surf["rough_moist"], surf["rough_scale"], mem["gust"],
                                  land, seawater)
            out.update(override(out, dict(t_atm=t_bot, q_atm=q_bot, u_atm=u[:, -1], v_atm=v[:, -1], p_atm=p_bot,
                                          z_atm=z_bot, t_surf=t_surf, t_ca=t_surf, q_surf=out["q_surf"],
                                          surface_flux=surface_flux)))
        return out

    # ------------------------------------------------------------------------------ 2
    def down(self, dt, time, T, u, v, R, pfull, phalf, zfull, zhalf, udt, vdt, tdt, rdt, sfc, surf, mem,
             t_surf, lw_extra=None):
        """physics_driver_down. udt ... rdt: the tendencies accumulated so far (dynamics /
        forcing). lw_extra(lw_tendency, radturbten) -> (lw_tendency, radturbten) adds a
        host's own radiative tendencies (the SCM's use_scm_rad). Returns the tendencies,
        the vertical-diffusion fields for couple/up and diagnostics; updates mem."""
        m = mem
        land = surf["land"]
        radturbten = m["radturbten"]
        lw_tendency = torch.zeros_like(T)
        diag = {}
        if self.radiation is not None:
            bands = surf.get("albedo_bands") or {k: surf["albedo"] for k in ALBEDO_BANDS}
            rad = self.radiation(time, dt, dict(pfull=pfull, phalf=phalf, zfull=zfull, zhalf=zhalf, T=T, R=R,
                                                q=R[..., SPHUM], ql=R[..., LIQ], qi=R[..., ICE], qa=R[..., CLD],
                                                t_surf=t_surf, albedo=surf["albedo"], land=land,
                                                lat=self.lat, lon=self.lon, surface=dict(land=land, **bands)))
            tdt = tdt + rad["tdt_rad"]
            lw_tendency = rad["tdtlw"]
            radturbten = radturbten + rad["tdt_rad"]
            surf["coszen"] = rad["coszen"]
            diag["radiation"] = rad
        if lw_extra is not None:
            lw_tendency, radturbten = lw_extra(lw_tendency, radturbten)
        udt, vdt, tdt = self.damping(dt, pfull, phalf, zfull, zhalf, u, v, T, self.hprime, udt, vdt, tdt)
        diag.update(damp_tdt=tdt, damp_udt=udt)
        vt = self.vturb(dt, lw_tendency, sfc["u_star"], sfc["b_star"], self.lat, m["convect"],
                        u, v, T, R[..., SPHUM], R[..., LIQ], R[..., ICE], R[..., CLD],
                        u, v, T, R[..., SPHUM], R[..., LIQ], R[..., ICE], R[..., CLD],
                        udt, vdt, tdt, rdt[..., SPHUM], rdt[..., LIQ], rdt[..., ICE], rdt[..., CLD],
                        pfull, phalf, zfull, zhalf)
        if self.diffusion_smooth:
            alpha = dt / self.tau_diff
            diff_m = (m["diff_m"] + alpha * (vt["diff_m"] + m["diff_cu_mo"])) / (1.0 + alpha)
            diff_m = torch.where(diff_m < self.diff_min, torch.zeros_like(diff_m), diff_m)
            diff_t = (m["diff_t"] + alpha * vt["diff_t"]) / (1.0 + alpha)
            diff_t = torch.where(diff_t < self.diff_min, torch.zeros_like(diff_t), diff_t)
        else:
            diff_t = vt["diff_t"]
            diff_m = vt["diff_m"] + m["diff_cu_mo"]
        m["diff_t"], m["diff_m"] = diff_t, diff_m
        radturbten = radturbten - tdt
        vd = self.vdiff.down(dt, u, v, T, R[..., SPHUM], R, diff_m, diff_t, phalf, pfull, zfull,
                             sfc["flux_u"], sfc["flux_v"], sfc["dtaudu_atm"], sfc["dtaudv_atm"],
                             udt, vdt, tdt, rdt[..., SPHUM], rdt)
        diag.update(diff_t=diff_t, diff_m=diff_m, gust=vt["gust"], z_pbl=vt["z_pbl"])
        return dict(udt=vd["dt_u"], vdt=vd["dt_v"], tdt=vd["dt_t"], rdt=vd["dt_tr"], vd=vd, vt=vt,
                    radturbten=radturbten, diag=diag)

    # ------------------------------------------------------------------------------ 3
    def couple(self, vd, sfc, surf, update_ocean=True):
        """Implicit surface coupling with the surface temperature held fixed, then the
        ice_amip fast update of the open-ocean roughness (from u*) and band albedos (from
        this step's coszen). update_ocean=False for a specified land surface."""
        cpl = implicit_surface_coupling(vd["dtmass"], vd["delta_t"], vd["dflux_t"], vd["delta_tr"][:, SPHUM],
                                        vd["dflux_tr"][:, SPHUM], vd["delta_u"], vd["delta_v"],
                                        sfc["flux_t"], sfc["flux_q"], sfc["flux_u"], sfc["flux_v"],
                                        sfc["dhdt_surf"], sfc["dedt_surf"], sfc["dedq_surf"],
                                        sfc["dhdt_atm"], sfc["dedq_atm"], sfc["dtaudu_atm"], sfc["dtaudv_atm"])
        if update_ocean:
            rm, rh, rq = compute_ocean_roughness(self.rough_cfg, sfc["u_star"])
            surf.update(rough_mom=rm, rough_heat=rh, rough_moist=rq, rough_scale=rm)
            a_vd, a_vf, a_nd, a_nf = compute_ocean_albedo(surf["coszen"], self.ocean_albedo_option)
            surf["albedo_bands"] = dict(asfc_vis_dir=a_vd, asfc_vis_dif=a_vf, asfc_nir_dir=a_nd, asfc_nir_dif=a_nf)
        return cpl

    # ------------------------------------------------------------------------------ 4
    def up(self, dt, T, u, v, R, pfull, phalf, zfull, zhalf, omega, dn, cpl, surf, mem):
        """physics_driver_up. dn: the output of down. Returns udt, vdt, tdt, rdt and the
        moist diagnostics; updates mem."""
        m = mem
        vd = dn["vd"]
        udt, vdt = dn["udt"], dn["vdt"]
        rdt = dn["rdt"]
        dt_t, dt_q = self.vdiff.up(dt, vd["e_global"], vd["f_t_global"], vd["f_q_global"],
                                   cpl["delta_t"], cpl["delta_q"])
        tdt = dt_t
        rdt = torch.cat([dt_q[..., None], rdt[..., 1:]], dim=-1)
        radturbten = dn["radturbten"] + tdt
        up = dict(up_tdt=tdt, up_rdt=rdt)
        mo = self.moist(dt, surf["land"].to(T.dtype), phalf, pfull, zhalf, zfull, omega, m["diff_t"], radturbten,
                        T, R[..., SPHUM], R, u, v, T, R[..., SPHUM], R, u, v,
                        tdt, rdt[..., SPHUM], rdt, udt, vdt)
        m["diff_cu_mo"] = mo["diff_cu_mo"]
        m["convect"] = mo["convect"]
        m["radturbten"] = torch.zeros_like(T)
        gust = dn["vt"]["gust"]
        m["gust"] = ssqrt(gust * gust + mo["gust_cv"] * mo["gust_cv"])
        m["pbltop"] = dn["vt"]["z_pbl"]
        return dict(udt=mo["udt"], vdt=mo["vdt"], tdt=mo["tdt"], rdt=mo["rdt"], moist=mo, **up)

    # ------------------------------------------------------------------------------ all
    def forward(self, dt, time, state: dict, geom: dict, tend: dict, surf: dict, mem: dict):
        """One physics step for a host model.

        state: T, u, v (ncol, nlev), tracers (ncol, nlev, 4), ps (ncol,);
        geom: pfull, phalf, zfull, zhalf [m above the surface], omega [Pa/s];
        tend: udt, vdt, tdt, rdt from the dynamics (zeros if the host adds them itself);
        surf: t_surf (ncol,) and the surface state (init_ocean_surface); mem: init_memory.
        Returns the total tendencies (udt, vdt, tdt, rdt) including tend, surface fluxes
        and precipitation. surf and mem are updated in place (new tensors)."""
        T, u, v, R, ps = state["T"], state["u"], state["v"], state["tracers"], state["ps"]
        pfull, phalf, zfull, zhalf = geom["pfull"], geom["phalf"], geom["zfull"], geom["zhalf"]
        t_surf = surf["t_surf"]
        sfc = self.surface_layer(T, u, v, R, ps, zfull[:, -1] - zhalf[:, -1], pfull[:, -1], t_surf, surf, mem)
        dn = self.down(dt, time, T, u, v, R, pfull, phalf, zfull, zhalf, tend["udt"], tend["vdt"], tend["tdt"],
                       tend["rdt"], sfc, surf, mem, t_surf)
        cpl = self.couple(dn["vd"], sfc, surf, update_ocean=not bool(surf["land"].any()))
        upo = self.up(dt, T, u, v, R, pfull, phalf, zfull, zhalf, geom["omega"], dn, cpl, surf, mem)
        return dict(udt=upo["udt"], vdt=upo["vdt"], tdt=upo["tdt"], rdt=upo["rdt"], flux_t=cpl["flux_t"],
                    flux_q=cpl["flux_q"], u_star=sfc["u_star"], lprec=upo["moist"]["lprec"],
                    fprec=upo["moist"]["fprec"], z_pbl=dn["vt"]["z_pbl"])
