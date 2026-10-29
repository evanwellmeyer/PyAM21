"""Common interface of the SCM forcing cases (atmos_scm/forc/scm_forc.F90).

A forcing object provides, for its case:
  initial_state(time)                   the initial profiles (scm_*_forc_init)
  update(time, time_diag, dt, state)    large-scale tendencies and omega (update_*_forc)
  specified_surface_flux(...)           surface fluxes when do_specified_flux (scm_surface_flux)
  add_tdtlw / add_tdtsw                 prescribed radiation when use_scm_rad

States are dicts of tensors: T, u, v (ncol, nlev), tracers (ncol, nlev, ntracer) in
the order of TRACERS, ps (ncol,). Tendencies are returned in the same layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

TRACERS = ("sphum", "liq_wat", "ice_wat", "cld_amt")
SPHUM, LIQ_WAT, ICE_WAT, CLD_AMT = range(4)


@dataclass
class SurfaceSpec:
    """Surface properties a case prescribes (scm_forc_mod public variables)."""
    tskin: float | None = None
    qskin: float | None = None
    albedo_obs: float | None = None
    rough_mom: float | None = None
    rough_heat: float | None = None
    senflux: float | None = None
    evapflux: float | None = None


@dataclass
class ForcingTendencies:
    ps: torch.Tensor                 # (ncol,) surface pressure set by the forcing
    T: torch.Tensor                  # (ncol, nlev) K/s
    u: torch.Tensor
    v: torch.Tensor
    tracers: torch.Tensor            # (ncol, nlev, ntracer)
    omega: torch.Tensor              # (ncol, nlev) Pa/s
    diag: dict = field(default_factory=dict)
    state: dict | None = None        # prognostic fields the forcing changed in place (T, tracers)


class ScmForcing:
    experiment: str = ""

    def __init__(self, nml: dict, vgrid, lat_rad: float, lon_rad: float,
                 device="cpu", dtype=torch.float64):
        self.nml = nml
        self.vgrid = vgrid
        self.lat = lat_rad
        self.lon = lon_rad
        self.device = torch.device(device)
        self.dtype = dtype
        self.surface = SurfaceSpec()

    def tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    # interfaces -------------------------------------------------------------------------
    def initial_state(self, time) -> dict:
        raise NotImplementedError

    def update(self, time, time_diag, dt: float, state: dict) -> ForcingTendencies:
        raise NotImplementedError

    def specified_surface_flux(self, sfc: dict) -> dict:
        raise NotImplementedError(f"{self.experiment}: no specified surface fluxes")

    def add_tdtlw(self, tdtlw: torch.Tensor) -> torch.Tensor:
        return tdtlw

    def add_tdtsw(self, tdtsw: torch.Tensor) -> torch.Tensor:
        return tdtsw


# ---- helpers shared by the forcing modules ------------------------------------------------
from ... import vert_advection as _va

VADV_SCHEMES = {1: _va.SECOND_CENTERED, 2: _va.FOURTH_CENTERED, 3: _va.FINITE_VOLUME_LINEAR,
                4: _va.FINITE_VOLUME_PARABOLIC, 5: _va.SECOND_CENTERED_WTS, 6: _va.FOURTH_CENTERED_WTS}


def vadv(dt, omega_h, dp, x, scheme: int):
    """vert_advection in advective form with the SCM scheme numbering (1..6)."""
    return _va.vert_advection(dt, omega_h, dp, x, scheme=VADV_SCHEMES[scheme], form=_va.ADVECTIVE_FORM)


def vadv_all(dt, omega_h, dp, T, tracers, u, v, temp_scheme, tracer_scheme, momentum_scheme):
    """The advection block the GCSS forcing modules share. Returns dT_vadv (without the
    adiabatic term), d(tracers), du, dv. As in the Fortran, a momentum scheme other
    than 1 advects T again (overwriting dT_vadv) and leaves u and v alone."""
    dT = vadv(dt, omega_h, dp, T, temp_scheme)
    dtr = torch.stack([vadv(dt, omega_h, dp, tracers[..., n], tracer_scheme)
                       for n in range(tracers.shape[-1])], dim=-1)
    du = torch.zeros_like(u)
    dv = torch.zeros_like(v)
    if momentum_scheme == 1:
        du = vadv(dt, omega_h, dp, u, 1)
        dv = vadv(dt, omega_h, dp, v, 1)
    elif momentum_scheme in VADV_SCHEMES:
        dT = vadv(dt, omega_h, dp, T, momentum_scheme)
    return dT, dtr, du, dv


def fixed_width_table(path: str, widths, nrows: int | None = None):
    """Rows of a Fortran fixed-format numeric file (e.g. '(f9.2,f8.2,...)')."""
    rows = []
    with open(path) as f:
        for line in f:
            if nrows is not None and len(rows) >= nrows:
                break
            vals, pos = [], 0
            for w in widths:
                field = line[pos:pos + w].strip()
                vals.append(float(field) if field else 0.0)
                pos += w
            rows.append(vals)
    return rows


def surface_flux_loop(surface_flux, sfc: dict, senflux, evapflux, land: bool = False, niter_max: int = 20,
                      max_flux_error: float = 0.1) -> dict:
    """The specified-flux iteration of the ARM / M-PACE cases (*_surface_flux_loop).

    surface_flux(t_surf, t_ca, q_atm, q_surf) -> dict (the surface_flux outputs, with the
    updated q_surf). Ocean (mpaceb): the skin temperature is adjusted (inner loop) until
    the sensible heat flux matches senflux, then the humidity of the lowest level seen by
    the surface layer (outer loop) until the evaporation matches evapflux. Land (arm,
    mpacea): the canopy temperature (from t_surf) and the surface humidity (from
    1.2 q_atm, through dedq_surf) instead. Each to max_flux_error W/m2, exactly as the
    Fortran: a shared iteration count, only unconverged points updated, q_surf carried
    from call to call. Non-converged points get the prescribed fluxes with u*, b*, q*
    diagnosed from them.
    """
    from ...constants import CP_AIR, GRAV, HLV, RDGAS, RVGAS
    d608 = (RVGAS - RDGAS) / RDGAS
    t_atm, q_atm, p_atm, t_surf = sfc["t_atm"], sfc["q_atm"], sfc["p_atm"], sfc["t_surf"]
    senflux = torch.as_tensor(senflux, dtype=t_atm.dtype, device=t_atm.device).expand_as(t_atm)
    evapflux = torch.as_tensor(evapflux, dtype=t_atm.dtype, device=t_atm.device).expand_as(t_atm)
    out = surface_flux(t_surf, sfc.get("t_ca", t_surf), q_atm, sfc["q_surf"])
    flux_lw_tmp = out["flux_r"]
    if land:
        t_var, q_var, q_s = t_surf, 1.2 * q_atm, None      # t_ca, q_surf
        call = lambda: surface_flux(t_surf, t_var, q_atm, q_var)
    else:
        t_var, q_var, q_s = t_surf, q_atm, out["q_surf"]   # t_surf, q_atm
        call = lambda: surface_flux(t_var, t_var, q_var, q_s)
    nouter = 1
    while True:
        ninner = 1
        while True:
            out = call()
            if land:
                q_var = out["q_surf"]
            else:
                q_s = out["q_surf"]
            upd = (torch.abs(out["flux_t"] - senflux) > max_flux_error) & (ninner < niter_max)
            if bool(upd.any()):
                t_new = torch.clamp(t_var - (out["flux_t"] - senflux) / torch.clamp(out["dhdt_surf"], min=1.0e-10),
                                    min=200.0, max=350.0)
                t_var = torch.where(upd, t_new, t_var)
            ninner += 1
            if not bool(upd.any()):
                break
        out = call()
        if land:
            q_var = out["q_surf"]
        else:
            q_s = out["q_surf"]
        upd = (HLV * torch.abs(out["flux_q"] - evapflux) > max_flux_error) & (nouter < niter_max)
        if bool(upd.any()):
            deriv = out["dedq_surf"] if land else out["dedq_atm"]
            q_new = torch.clamp(q_var - (out["flux_q"] - evapflux) / deriv, min=1.0e-8, max=0.05)
            q_var = torch.where(upd, q_new, q_var)
        nouter += 1
        if not bool(upd.any()):
            break
    out = dict(out)
    rho = p_atm / (RDGAS * t_atm * (1.0 + d608 * q_atm))
    if nouter > niter_max or ninner > niter_max:
        bad = (HLV * torch.abs(out["flux_q"] - evapflux) > max_flux_error) | \
              (torch.abs(out["flux_t"] - senflux) > max_flux_error)
        u_star = torch.sqrt(torch.sqrt(out["flux_u"] ** 2.0 + out["flux_v"] ** 2.0) / rho)
        bfact = GRAV / (t_atm * (1.0 + d608 * q_atm))
        chu = (1.0 + d608 * q_atm) * bfact / CP_AIR
        cmu = d608 * bfact * t_atm
        b_star = (chu * senflux + cmu * evapflux) / rho / u_star
        out["flux_t"] = torch.where(bad, senflux, out["flux_t"])
        out["flux_q"] = torch.where(bad, evapflux, out["flux_q"])
        out["u_star"] = torch.where(bad, u_star, out["u_star"])
        out["q_star"] = torch.where(bad, evapflux / rho / u_star, out["q_star"])
        out["b_star"] = torch.where(bad, b_star, out["b_star"])
    out["flux_r"] = flux_lw_tmp
    out["q_surf"] = q_atm + out["flux_q"] / (rho * out["cd_q"] * out["w_atm"])
    return out
