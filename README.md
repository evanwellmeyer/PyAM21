# PyAM21

A PyTorch port of the GFDL single column model (SCM, public release `siena_201207`)
with the AM2.1 atmospheric physics. It does two things:

- Runs the 12 SCM cases standalone and reproduces the Fortran model.
- Provides the physics as a batched, differentiable `torch.nn.Module` (`AM2Physics`)
  that a PyTorch dynamical core can call.

It runs on CPU in float64 and on Apple MPS in float32. Validation is in float64 on
CPU against a gfortran build of the reference, scheme by scheme and case by case. See
[docs/status.md](docs/status.md) for the numbers and [docs/plan.md](docs/plan.md) for
the design.

## What is ported

Everything the 12 SCM cases run:
- the SCM forcings;
- surface layer (Monin–Obukhov, ocean roughness and albedo);
- the full `sea_esf` radiation: ESF shortwave, `sealw99` longwave with the CKD2.1
  continuum and CO₂/CH₄/N₂O transmission tables computed from the line-by-line data,
  strat clouds with predicted microphysics, Shettle–Fenn aerosols, ozone and gas
  climatologies, astronomy;
- gravity-wave and Rayleigh damping;
- Lock entrainment and stable-BL turbulence;
- implicit vertical diffusion with surface coupling;
- RAS convection, cumulus momentum transport, and Tiedtke/Rotstayn–Klein stratiform
  clouds.

## Install

```bash
conda env create -f environment.yml
```

```bash
conda activate pyam21
```

```bash
pip install -e .
```

The cases need GFDL's input data (`reference/input_data`, 5.6 GB, from
`input_data.tar.gz` of the SCM release); the case namelists and forcing files are in
`reference/input/`.

## Running a case

```python
import torch
from pyam21.scm.case import load_case
from pyam21.scm.model import SCM

m = SCM(load_case("mpace_b"))               # float64 on CPU (reproduces the Fortran)
for _ in range(48):
    m.step()
T, q = m.state["T"], m.state["tracers"][..., 0]  # (ncol, nlev), index 0 = model top

m32 = SCM(load_case("mpace_b"), device="mps", dtype=torch.float32, ncol=4096)
```

`ncol > 1` replicates the case column. It is useful for ensembles, for example with
perturbed initial states, and for throughput.

## Using the physics in a dynamical core

```python
from pyam21.physics.am2 import AM2Physics
from pyam21.radiation.driver import RadiationDriver
from pyam21.radiation.sea_esf import SeaEsfRad

rad = RadiationDriver.from_nml(nml, SeaEsfRad(nml, data_dir, boxes, pref))  # optional
phys = AM2Physics(nml, lat, lon, hprime=hprime, radiation=rad.for_physics())
surf = phys.init_ocean_surface(ncol); surf["t_surf"] = sst
mem = AM2Physics.init_memory(ncol, nlev)

out = phys(dt, time, state, geom, dyn_tend, surf, mem)
# state: T, u, v (ncol, nlev), tracers (ncol, nlev, 4) = sphum, liq, ice, cloud fraction, ps
# geom:  pfull, phalf, zfull, zhalf (height above the surface), omega
# out:   udt, vdt, tdt, rdt (total tendencies including dyn_tend), surface fluxes, precip
```

`nml` is the parsed `input.nml`, from `pyam21.namelist.read_namelist(path)` or `load_case(...).nml`.

- `boxes` gives the grid box of each column for the climatologies.
- `pref` is the two reference pressure profiles that `physics_driver_init` receives.
- `mem` and `surf` carry the physics memory between steps: smoothed diffusivities,
  convection flag, gustiness, roughness and albedos.

The four stages (`surface_layer`, `down`, `couple`, `up`) are also callable separately,
the way the SCM uses them.

## Precision, devices, gradients

- **float64 / CPU** reproduces the Fortran to roundoff, typically 1e-14 to 1e-11 after
  a day. The float64 code paths keep the Fortran's operation order.
- **float32 / MPS**: a few formulations lose most of their digits in float32. These
  are the Simmons–Burridge full-level pressures, dry static energy differenced across
  mixed layers, and the delta-Eddington solution at single-scattering albedo near 1.
  Below float64 they use algebraically identical but stable forms. float32 then stays
  within ~1e-4–3e-3 K of float64 over 24 steps, which is the level set by float32's
  resolution of T itself.
- **Gradients**: backpropagation through many steps gives finite gradients in all 12
  cases, including through the radiation. They agree with central finite differences
  to 1e-7–1e-9. Singular points of derivatives in masked branches are guarded
  (`pyam21.core.spow` / `ssqrt`, safe denominators). The physics contains thresholds,
  so gradients are those of the piecewise-smooth model.

## Performance

These are wall-clock times per physics step on an Apple-silicon laptop, bomex, 24 levels.

| device | 1 column | 1024 columns | 16384 columns | 65536 columns |
|---|---|---|---|---|
| CPU float64 | 59 ms | 180 ms | — | — |
| CPU float32 | 70 ms | 158 ms | 1.6 s | — |
| MPS float32 | 370 ms | 363 ms | 382 ms | 673 ms |

The cost is dominated by the number of small tensor ops: vertical recurrences in RAS
re-evaporation, strat_cloud sedimentation, and the longwave level loop. MPS pays
off from a few thousand columns: about 100k column-steps/s at 65k columns, 9× the CPU.
Radiation steps (every 3 h) add about 180 ms per call for one column on CPU.

## Tests

```bash
pytest -q
```

This runs 78 tests in about 3.5 minutes:
- replay of every physics scheme against Fortran dumps;
- the whole-run comparison of all 12 cases;
- the radiation components and the full `sea_esf` package;
- gradients against finite differences;
- float32 CPU/MPS against float64;
- the `AM2Physics` batch interface.

Tests that need the reference dumps (`reference/dumps`, made by
`reference/scripts/make_dumps.sh`) skip when they are absent.

## Reference build

GFDL's Fortran source is not in the repository. `reference/scripts/get_source.sh`
unpacks it into `reference/src` from the release tarball, downloading it from GFDL if
needed. `reference/scripts/build_reference.sh` then builds the Fortran SCM with gfortran and
inserts switchable dump hooks (`reference/scripts/make_hooks.py`).
`reference/scripts/make_dumps.sh CASE [STEPS] [TAG]` writes the per-scheme
input/output dumps that the tests replay.

## License

GPL-2.0-or-later, like the FMS release it translates.
