# Reference source overrides

Files here replace the file at the same path under `src/` when the reference is built
(`scripts/build_reference.sh`). None changes physics results.

| File | Why |
|---|---|
| `shared/mpp/affinity.c` | Linux `sched_getaffinity` does not exist on macOS. Stub returning 0. |
| `shared/field_manager/field_manager.F90` | In `fm_intersection` a local variable is named `field_type`, the same as a derived type. gfortran 15 rejects it. Renamed to `fld_type_str` (3 lines). |
| `shared/memutils/memutils.F90` | `mem_dump` reads `/proc/self/status` (Linux only) for a memory report at shutdown. Returns 0. |
