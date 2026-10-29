# Plan: PyTorch port of the GFDL single column model (AM2.1 physics)

## Goal

A PyTorch version of the GFDL SCM (public release `siena_201207`, from
https://www.gfdl.noaa.gov/scm-quickstart/) that:

1. runs the 12 SCM cases standalone and reproduces the Fortran model, and
2. provides the AM2.1 physics as batched, differentiable `torch.nn.Module`s that a
   PyTorch dynamical core can call (tensors `(ncol, nlev)`, index 0 at the model top,
   pressures from the host), on CPU (float64) and Apple MPS (float32).

## Reference

- Source: `ftp://ftp.gfdl.noaa.gov/perm/GFDL_pubrelease/SCM/SCM_public_release.tar.gz`
  (5.8 MB, FMS tag `siena_201207`), input data `input_data.tar.gz` (3.1 GB) and GFDL's
  `sample_output.tar.gz` (Intel build, 2012).
- Built here with gfortran 15 (conda env `eamxx-ref`), `-fdefault-real-8`,
  `-ffp-contract=off`, `-O1`: `reference/scripts/build_reference.sh`.
  Three portability overrides, none touching physics: `reference/overrides/README.md`.
- `reference/scripts/run_case.sh CASE` runs a case. All 12 run (0–5 s each).
- `reference/scripts/compare_sample.py`: our gfortran build reproduces GFDL's
  published Intel output to 1e-8–1e-16 relative in the 1-day cases. The exceptions are
  fields of roundoff size (~1e-21) and the long ARM (29 d) and MPACE-A (6 d) runs,
  which drift apart chaotically after days.

## What the 12 cases run

One physics configuration for all cases (the physics namelists are identical):

| Process (call order in a step) | Fortran | Notes |
|---|---|---|
| SCM forcing: large-scale advection, subsidence, geostrophic wind, nudging, prescribed radiation | `atmos_scm/forc/*`, `vert_advection` | 11 forcing modules for 12 cases |
| Surface layer (fluxes, u*, b*, q*) | `coupler/flux_exchange` `sfc_boundary_layer`, `surface_flux`, `monin_obukhov`, `ocean_rough` | Overridden by the SCM forcing when `do_specified_flux` |
| Radiation (every 3 h) | `radiation_driver`, `sea_esf_rad`: `esfsw_driver` (18 bands), `sealw99` + `gas_tf` + `lw_gases_stdtf` + `optical_path` + `longwave_*`, clouds `strat_clouds_W` + `microphys_rad` + `cloudrad_package`, aerosols `aerosol` + `aerosolrad_package` (Shettle–Fenn), `ozone`, `radiative_gases`, `astronomy` | Only in `arm`, `astex`, `mpace_a`, `mpace_b` (`do_radiation`) |
| Damping | `damping_driver`: Rayleigh friction (top level), `mg_drag` | |
| Boundary-layer turbulence | `vert_turb_driver`: `entrain` (Lock et al. 2000), `stable_bl_turb`, `monin_obukhov` | |
| Vertical diffusion (implicit, with surface coupling) | `vert_diff_driver`, `vert_diff` | down/up split around the coupler |
| Surface update | `flux_down_from_atmos`, `flux_up_to_atmos`; ocean with prescribed SST; land "specified" in land cases | |
| Moist processes | `moist_processes`: RAS (`ras`), cloud momentum transport (`cu_mo_trans`, diffusive), `strat_cloud_legacy` (Tiedtke prognostic cloud + Rotstayn–Klein microphysics) | `do_ras`, `do_strat`, `do_cmt` |
| Time step | `atmosphere_up`: damping of T/q tendencies above `pdamp`, q filler, forward step | |

Per case:

| Case | Forcing | Radiation | Surface fluxes | Length |
|---|---|---|---|---|
| `gcss_bomex` | `scm_bomex` | prescribed | specified | 1 d |
| `gcss_dycoms2_rf01_a` | `scm_rf01` | SCM LW/SW (`use_scm_rad`) | specified | 1 d |
| `gcss_dycoms2_rf01_b` | `scm_rf01` | SCM LW/SW | interactive (ocean) | 4 d |
| `gcss_dycoms2_rf02` | `scm_rf02` | SCM LW/SW | specified | 1 d |
| `gcss_rico_short` | `scm_rico` | prescribed | RICO bulk formula | 4 d |
| `gcss_atex` | `scm_atex` | SCM LW/SW | ATEX bulk formula | 8 h |
| `gcss_astex` | `gcss_astex` | SCM LW/SW | specified | 8 h |
| `gcss_arm` | `scm_gcss_arm` | none | specified (land) | 15 h |
| `astex` | `scm_astex` | full sea_esf | interactive (ocean) | 42 h |
| `arm` | `scm_arm` | full sea_esf | specified (land) | 29 d |
| `mpace_a` | `scm_mpacea` | full sea_esf | specified (land) | 6 d |
| `mpace_b` | `scm_mpaceb` | full sea_esf | specified | 1 d |

Not used by any case (not ported): Donner deep convection, UW shallow convection,
MCA, large-scale condensation `lscale_cond`, Morrison–Gettelman and the non-legacy
`strat_cloud`, COSP/ISCCP diagnostics, MY2.5/EDT turbulence, LM3, tracer chemistry.

## Validation method

As in Py-E3SM-EAMxx: per-scheme replay against the running reference.
A switchable dump hook (`reference/overrides/.../ref_dump.F90`, on only when
`REF_DUMP_DIR` is set) writes every input and output of each scheme call. Tests replay
each call through the port on CPU in float64 and compare. Then whole-step and
whole-run comparisons per case. Tolerances are stated per scheme in `docs/status.md`.
MPS (float32) runs are compared with the float64 CPU port at stated tolerances.

## Design of the port

- `src/pyam21/` package. Schemes are `torch.nn.Module`s whose namelist values are
  constructor arguments (plain floats by default; optionally `nn.Parameter` for tuning).
- State and physics memory are explicit tensors passed in and returned (no hidden module
  state), so steps can be batched, differentiated through, and embedded in a dycore.
- Everything is vectorised over columns with `torch.where`. Level loops that are running
  sums or first-order recurrences (most of RAS, the level-local part of strat_cloud) are
  cumulative sums over level tensors; true vertical recurrences (RAS re-evaporation,
  strat_cloud sedimentation, the longwave level loop) remain Python loops. A few
  data-dependent early exits (RAS cloud types active in no column, converged
  Monin–Obukhov iterations) sync with the host once per test; on MPS they cost less
  than the work they skip.
- Lookup tables (saturation vapour pressure, radiation tables) are built in float64
  and cached per device/dtype.

## License

The 2012 FMS release is GPL-2.0-or-later, so this port (a translation) is too.
