# Status

All numbers are max |port − Fortran| / max |Fortran| over every dumped call or step,
in float64 on CPU, against the gfortran reference (`reference/scripts/build_reference.sh`,
`-fdefault-real-8 -ffp-contract=off -O1`). The tests in `tests/` assert them with some
margin.

## Whole runs (state after every step: T, u, v, tracers)

| case | steps compared | max rel. error | radiation |
|---|---|---|---|
| gcss_bomex | all dumped | ~1e-14 | prescribed |
| gcss_dycoms2_rf01_a | all dumped | ~1e-13 | SCM LW/SW |
| gcss_dycoms2_rf01_b | all dumped | ~1e-12 | SCM LW/SW |
| gcss_dycoms2_rf02 | all dumped | ~1e-12 | SCM LW/SW |
| gcss_atex, gcss_astex, gcss_arm, gcss_rico_short | all dumped | ≤ 3e-12 | SCM / none |
| mpace_b | 47 | T 2e-16, tracers 6e-15, tdt_rad 8e-15 | port's sea_esf |
| astex | 83 | T 1e-14, tracers 2e-11, tdt_rad 1e-12 | port's sea_esf |
| mpace_a | 95 | T 2e-14, tracers 1e-11, tdt_rad 7e-12 | port's sea_esf |
| arm | 95 | T 9e-16, tracers 2e-13, tdt_rad 3e-13 | port's sea_esf |

The four radiation cases are also run with the Fortran's radiative tendencies replayed
(`test_scm_run_replayed_radiation`), which separates radiation errors from the rest.

## Physics schemes (replayed call by call, 8 cases, every dumped call)

| scheme | max rel. error | test tolerance |
|---|---|---|
| vert_turb (entrain + stable_bl + Monin–Obukhov) | 2.3e-15 | 1e-12 |
| vert_diff down (tendencies, surface deltas) | 1.4e-14 | 1e-12 |
| damping (Rayleigh + mg_drag) | 2.2e-15 | 1e-12 |
| moist processes (RAS, cu_mo_trans, strat_cloud) | 9.2e-12 | 1e-10 |

The surface layer, implicit coupling and SCM forcings are covered by the whole-run
comparisons.

## Radiation components (radiation steps 1, 7, 13, 19 of the four cases)

| component | max rel. error |
|---|---|
| astronomy (orbit, zenith angles, fracday) | exact |
| radiation_driver (alarms, renormalisation) | roundoff |
| atmos_input, ozone, radiative gases, aerosol climatology | ~1e-15 |
| cloud specification (strat, predicted microphysics, random overlap) | ~1e-15 |
| SW and LW cloud optics | ~1e-15 |
| aerosol SW and LW optical depths | exact |
| ESF shortwave (18 bands, fluxes, heating, clear sky) | ≤ 6e-15 |
| LW set-up tables (H₂O emissivities, band sources, CKD, coefficients) | exact (bit for bit) |
| CO₂/CH₄/N₂O transmission tables from the line-by-line data (lw_gases_stdtf) | ≤ 3e-13 (second-derivative tables), ≤ 1e-14 otherwise |
| sealw99 longwave (heating, fluxes, clear sky) | ≤ 2e-14 |
| whole sea_esf package from T, q, clouds, albedos | ≤ 3e-14 |

## float32 and MPS

float32 on CPU and MPS runs all cases without NaN and stays within
~1e-4–3e-3 K of float64 over 24 steps. That is the level set by float32's resolution of
T (~3e-5 K at 300 K) amplified by the model's thresholds. `tests/test_precision.py`
asserts < 1e-2 K and < 1e-5 kg/kg after 12 steps. Four formulations have float32-stable
variants, used only below float64:

- `scm/vert_grid.py`: Simmons–Burridge full-level pressure and interpolation weights via
  log1p of pressure ratios instead of differences of p·ln p. This was the largest
  error: 7 Pa, then 0.4 m at the lowest level.
- `physics/vert_diff.py`: dry static energy differences as ΔT + gΔz/cp.
- `physics/entrain.py`: the parcel/level virtual static energies as anomalies from the
  lowest level.
- `radiation/esfsw.py` (delta-Eddington): single-scattering albedo cap 1 − 1e-6 (the
  Fortran's 1 − 1e-8 rounds to 1), and a solution written without 1/exp(−τq).

## Differentiability

Reverse-mode gradients through 12 steps are finite in all 12 cases, including the
radiation. Directional derivatives agree with central finite differences to 1e-7–1e-9
relative (`tests/test_gradients.py`). Fixes that made this possible, none of which
changes forward values:

- fractional powers and square roots of arguments that can be 0 use
  `core.spow` / `core.ssqrt`;
- divisions inside masked branches use safe denominators (vertical advection, RF01
  inversion height, cloud thickavg, Monin–Obukhov Newton iteration);
- no in-place modification of tensors saved for backward.

## Not ported

These are not used by any SCM case:
- Donner deep and UW shallow convection, MCA, `lscale_cond`;
- Morrison–Gettelman and the non-legacy `strat_cloud`;
- max-random cloud overlap with predicted microphysics;
- stochastic clouds, `do_thick`, NLTE;
- the rsb continuum and hitran_1992;
- COSP/ISCCP diagnostics, MY2.5/EDT turbulence, the LM2/LM3 land models;
- tracer chemistry, time-varying gases, solar variability.

The port raises `NotImplementedError` if a namelist asks for one of these.

## Known limitations

- A step is still dominated by the count of small tensor ops. The main costs are the
  vertical recurrences in RAS re-evaporation, strat_cloud sedimentation and the
  longwave level loop. MPS therefore needs thousands of columns to beat the CPU; see
  the README table.
- The whole-run dumps cover the first 48–96 steps of each case. The long ARM (29 d) and
  MPACE-A (6 d) runs diverge chaotically from any other build after days, including
  from GFDL's own Intel output.
