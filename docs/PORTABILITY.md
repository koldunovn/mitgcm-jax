# Portability record: what mitjax needs from our machine (docs plan 20261006, S1 Task 1)

Goal of S1: mitjax runs any configuration on a user's machine with only an MITgcm checkout at 63cdc0b (`MJX_UPSTREAM`):
no machine paths, no Fortran oracle (`MJX_REFERENCE`, `MJX_REFERENCE_RUNS`), the experiment anywhere on disk. This
page is the measured list of every dependency, with the fix chosen per item and its state. It is re-measured by
`scripts/noref_probe.py` after every fix.

## How it was measured

- **Probe** `scripts/noref_probe.py` (+ `scripts/noref_probe.sbatch`, compute node, frozen snapshot): `MJX_REFERENCE` and
  `MJX_REFERENCE_RUNS` empty directories, `MJX_WORK` a new directory (so `MJX_RUNS`, `MJX_CACHE` inside it), every
  experiment COPIED outside the MITgcm tree (code*, input*, results; symlinks dereferenced; plus the sibling experiment
  a `prepare_run` reaches). Cases: forward whole runs of tutorial_barotropic_gyre/input and 1D_ocean_ice_column/input
  (`python -m mitjax run`'s driver), lab_sea/input and global_ocean.cs32x15/input 2 steps, the gradient of
  1D_ocean_ice_column/input_ad (GenarrAdjoint, xx_theta), a sharded forward P=2 of tutorial_baroclinic_gyre (4 tiles,
  2 steps, compared with P=1). Each case runs at levels that add stand-ins for the fixes not made yet, so the failures
  behind the first become visible: L0 nothing; L1 the in-tree experiment; L2 + the oracle toolchain; L3 + the probe
  exchange maps. An audit hook lists every file opened, listed or executed outside the probe directory, the clone, the
  repository and the Python installation ("foreign"), with the mitjax frame that did it.
- **Tier 1 without the oracle** (`noref_probe.sbatch <repo> tier1`): smoke + tier1 with the same environment.
- **Static inventory**: every `paths.*` use and every load of `reference/` or `tools/` from `mitjax/` (grep at 54ada69).

Jobs (commit, job, result directory `$MJX_RUNS/port/...`):

| what | commit | job | result |
|---|---|---|---|
| probe L0-L3, 6 cases | b86adab | 27924629 | `noref-probe-b86adab-27924629/probe/SUMMARY.txt` |
| tier 1 without the oracle | b86adab | 27924630 | `noref-tier1-b86adab-27924630/` (65 passed, 32 failed, 1 error) |
| probe L0-L2 after Tasks 2-3 | 701b081 | 27925169 | `noref-probe-701b081-27925169/probe/SUMMARY.txt` |

## Result at 54ada69 (job 27924629): three run-time dependencies, nothing else

Every case failed at L0 on the experiment location, at L1 on the toolchain, at L2 on the exchange maps (except the
cube, whose maps were already replayed), and every case ran at L3: the gradient finite (fc = 691935.7199964371), the
P=2 run bitwise equal to P=1. The only foreign accesses at L3 were the three items below plus the system `cpp` at an
absolute Levante path.

| # | dependency | site (file:line at 54ada69) | class | fix | state |
|---|---|---|---|---|---|
| R1 | experiment must be `<MITgcm tree>/verification/<exp>` | `drivers/run.py:78-82` (load_experiment); `config/params.py:210-213`, `config/namelists.py:236-237`; the kernel sites `model/src/cg2d_h.py:120`, `ini_fields.py:20`, `ini_parms.py:161`, `pkg/ctrl/ctrl_readparms.py:149`, `pkg/ptracers/ptracers_forcing_surf.py:18`, `ptracers_readparms.py:22`; `eesupp/exch_maps.py:244`; `reference/make_rundir.py:336` (prepare_run mirror of `upstream/verification`) | run-time read | Task 2: `ExperimentConfig.exp_dir`; `params.load(exp_dir=)`; one helper `config.params.code_path(cfg)` at the six kernel sites; `make_rundir --exp-dir` (the experiment's parent directory plays `verification/` for prepare_run's relative paths, as testreport) | fixed 007930d (gate 27925167 PASS incl. the renamed copy) |
| R2 | the C preprocessor and DEFINES/INCLUDES read from the oracle's build records `$MJX_REFERENCE/bin/*/{Makefile,provenance.txt}`; the cpp program is the absolute path recorded there (`/sw/spack-levante/gcc-11.2.0-bcn7mb/bin/cpp`) | `config/cpp_options.py:108-119` (default_toolchain), `:90-105` (toolchain_from_record), run at `:226, 238, 398, 422` | run-time read | Task 4: system cpp detection as genmake2, declared default DEFINES, `MJX_CPP` / `MJX_CPP_DEFINES`, explicit `have_netcdf`; the oracle records stay an optional source and the equality test | fixed 483f3d3 (gate 27929160 PASS: 28 builds identical) |
| R3 | exch1 and single-facet exch2 halo maps read from `$MJX_REFERENCE/exch_maps/<exp>-<layout>.npz`, looked up by experiment name | `eesupp/exch_maps.py:202-228` (map_dir, load_maps), `drivers/model.py:144-153` | run-time read | Task 3: `eesupp/exch1_tables.py` replays EXCH1_RX / EXCH0_RX from the templates; `exch_maps.exchange_maps(exp)` (Model default) replays every family, caches the copy maps in `$MJX_CACHE/exch_maps` under a layout + topology key | fixed f45a67a + cafa0b5 (gate 27925167 PASS) |
| R4 | `reference/make_rundir.py` loaded from the repository at import | `config/namelists.py:36-38` | run-time import (packaging, plan-review (f)) | Task 5: move the linkdata / prepare_run logic into `mitjax/` | fixed 4544977, 3179551, 3a6ecd5 (`mitjax/make_rundir.py`; shim at the old path) |
| R5 | the link farm of the preprocessor lives under `$MJX_RUNS/config_cpp` | `config/cpp_options.py:194` | cache | Task 5: move to `$MJX_CACHE` (the neutral defaults) | fixed 483f3d3 |
| R6 | build identity by name `<experiment>-<code dir>-<commit>` for build-dependent MAX/MIN winners | `ops/fortran_minmax.py:129-133` | not a read; a renamed or edited build is "unknown" or, worse, "known" | Task 4: identity = hash of the preprocessed options; decision 7 default + warning | fixed 0ba0a9e (`config/build_identity.py`) |
| R7 | `tools/testreport_jax.py` (the digit comparison) is not in the package | `drivers/run.py:570` (docstring); used by tests and README | packaging (f) | Task 5: move into `mitjax/` | fixed 4544977 (`mitjax/testreport_jax.py`; shim at the old path) |
| R8 | `MJX_UPSTREAM` itself: pinned-commit check, cited lines, W2 headers | `params_io.py:46-56, 213-220`, `pkg/exch2/w2_eeboot.py:42` | run-time read of the one required input | keep (the user's MITgcm checkout) | by design |
| R9 | prepare_run scripts that reach sibling experiments by relative paths (`global_ocean.cs32x15/input/prepare_run`: `../../tutorial_held_suarez_cs/input`) | `reference/make_rundir.py` | user-side requirement | documented: copy the sibling experiment next to the copied one (the probe does) | by design (configurations.md, Task 10) |

`$MJX_REFERENCE` mentions in kernel docstrings (`pkg/monitor/mon_*.py`, `pkg/kpp/kpp_routines.py:21`,
`pkg/generic_advdiff/gad_flux_limiter_h.py:17`, `pkg/seaice/seaice_growth.py:15`, `model/src/calc_surf_dr.py:40`,
`ops/fortran_minmax*.py`, `ops/libm.py:5`, `config/cpp_options.py:17`, `eesupp/exch_maps.py:39`) are citations only:
the MAX/MIN winners are literal `p=` arguments in the code. No module of `mitjax/` imports `tools/`; the only load of
`reference/` is R4.

## Decision 11 (an experiment's own Fortran) -- measured with Task 2

`config/own_code.py` classifies every `*.F`, `*.F90`, `*.c` of the build's code directory: equal to a ported
experiment version (`PORTED`, 14 files: advect_xy, advect_cs, solid-body.cs-32x32x1, tutorial_global_oce_latlon,
tutorial_tracer_adjsens, tutorial_global_oce_optim, global_ocean.cs32x15/code_ad cost_test.F, and
adjustment.cs-32x32x1/code_min/main.F as set-up only), or unchanged from the MITgcm file it shadows; anything else is
refused at `params.load` naming its routines. Over all 63cdc0b verification code directories: every registered
experiment passes; 18 code directories of unported experiments are refused (hs94.*, aim.5l_*, fizhi-*, dome,
internal_wave, isomip, bottom_ctrl_5x5, halfpipe_streamice, tutorial_global_oce_biogeo, tutorial_held_suarez_cs,
tutorial_rotating_tank). Before Task 2 these files were ignored silently (only ini_fields and the ptracers/cost
dispatch looked for them).

## Tier 1 without the oracle (job 27924630 on b86adab)

65 passed, 32 failed, 1 error. Every failure is one of:

- **R2 toolchain** (`config/cpp_options.py:114`), 16 tests: test_config (10: namelist cases, unknown variable, cpp
  options vs records, cpp flags, all M1 variants, use flags, params pytree, fortran_default x2, crosscheck negative
  controls), test_grid_tier1, test_r1_barotropic_gyre_tier1, test_r2_baroclinic_gyre_tier1, test_r3_advection_quick,
  test_r5_optim_ad_tier1, scripts/tests/test_cpp_live::test_viewer_differs_between_experiments. After Task 4 most of
  them will reach their oracle comparison and belong to the next class.
- **Oracle data, test-only by design** (to skip cleanly without oracle data, plan Task 11), 17 tests: build records
  (test_config::test_packages_config_equals_build_records), oracle runs (test_config::test_link_sources_equal_make_rundir,
  ::test_stdout_crosscheck_where_oracle_runs_exist, test_monitor_tier1, test_mds), probe maps (test_exchange), replay
  outputs (test_gad_som_quick, test_gad_pqm_compact [error]), global_sum_ref (test_global_sum), oracle binaries'
  libm (test_libm x5), `tools/63cdc0b/tr_cmpnum` (test_testreport_jax x2), an oracle build directory
  (test_cpp_live::test_compiled_line_map_known_lines).

Tests reference the oracle widely (static count over `mitjax/tests` and `scripts/tests`: `paths.REFERENCE` 49,
`paths.REFERENCE_RUNS` 40, `paths.UPSTREAM` 82, `paths.REPO` 28, `paths.RUNS` 28): all test-only.

## After Tasks 2-3

Re-probe job 27925169 on 701b081 (code of cafa0b5), L0-L2: every case fails at L0 and L1 only at
`config/cpp_options.py:114` (R2, the toolchain); at L2 every case runs (gradient fc 691935.7199964371 finite, P=2
bitwise P=1) and the only foreign accesses are the toolchain stand-in's and the system cpp at its Levante path. R1 and
R3 are gone. Gate 27925167 PASS (63 passed, 2 xfailed, 2 known skips), tier 1 27925168 PASS 98/98. Remaining for
session 2: R2, R4, R5, R6, R7 (Tasks 4-5), then the no-reference test (Task 5) built from this probe.

## Session 2: Tasks 4-5 (branch port-s1b, 2026-10-06)

Measured state after session 2 (commits 483f3d3 .. 3a6ecd5; jobs below, sacct-checked):

- **R2 toolchain.** `cpp_options.default_toolchain()` = `system_toolchain()`: `cpp -traditional -P` found as genmake2
  finds it (`$CPP`, then `/lib/$CPP`, tested with `#define A a`, genmake2:1942-1965; absolute path recorded), the
  declared DEFINES of the verification builds (`linux_amd64_gfortran:43` + genmake2's gfortran test results
  HAVE_SYSTEM, FDATE, ETIME_SBR, CLOC, SETRLSTK, SIGREG, STAT, NETCDF, FLUSH in genmake2's order), INCLUDES empty, an
  explicit `have_netcdf` (default true, as the verification builds); overrides `MJX_CPP`, `MJX_CPP_DEFINES`,
  `MJX_CPP_INCLUDES`, `MJX_HAVE_NETCDF`; no cpp is a `CppNotFound` naming MJX_CPP. The oracle records stay an optional
  source (`oracle_toolchain()`: the record tests, `tools/cpp_live.tool_toolchain`).
  Gate `test_toolchain_system.py`, job 27929160 (483f3d3): with Levante's system cpp (`/usr/bin/cpp`, GCC 8.5.0; the
  oracle used GCC 11.2.0's) every one of the 28 registry builds has the same ExperimentConfig, NamelistParams and
  packages_boot macros as with the oracle's toolchain, every `.F`/`.F90`/`.h` of every build preprocesses byte for byte
  equal and every conditional arm is taken alike; the only difference allowed and seen is the empty INCLUDES: the
  `pkg/mnc` sources with `#include "netcdf.inc"` fail to preprocess (never preprocessed by mitjax). Controls: a
  planted `-DALLOW_AUTODIFF` changes the configuration and the_model_main.F's text; no-cpp and override errors.
  PASS, 50 tests (with test_config.py and test_cpp_live.py), 10 min.
- **Decision 10, clang (best effort).** No macOS machine here. Measured instead: LLVM 18.1.6's `clang-cpp
  -traditional -P` on Levante (`MJX_CPP=/sw/spack-levante/llvm-18.1.6-a6h6jc/bin/clang-cpp -traditional -P`, the
  mode macOS's `cpp` runs), job 27929161: the configuration (ExperimentConfig, NamelistParams, packages_boot macros)
  of all 28 builds equal to the oracle's; the preprocessed text differs in blank lines and in the exponent letter
  of `_d` constants (`3.14...D0` for GCC's `3.14...d0`, equal in Fortran), so the byte-for-byte text check fails by
  design (29 failed, the planted control included because its difference list is longer). Not measured: Apple's
  own clang, `fortran_default` values read from the clang-preprocessed text beyond the configuration.
- **R6 + decision 7.** `config/build_identity.py`: identity = sha256 of the MITgcm commit, the package list, the
  macros of every *OPTIONS.h and the model/src prologue (the toolchain's DEFINES included) and the SIZE.h parameters;
  `KNOWN` names the 28 registry builds (no collision). `fortran_minmax.build_winner` finds the build by identity: a
  known build keeps its measured winner (or is refused when the table has none), any other build takes the module's
  `MINMAX_DEFAULT` (the measured majority: seaice_growth.F:1847 MAX "a" 6 of 8 builds, :1848 MIN "b" 6 of 8) with one
  `MinMaxDefaultWarning` per site. Gate `test_build_identity.py` (registry identities; renamed copy = known build, no
  warning; an edited CPP option or tiling = new build, the default + one warning per site; the old name lookup
  planted back fails it, measured on the login node), audit `test_minmax_sites.py::test_audit_default_winners`.
- **R4, R7 packaging.** `mitjax/make_rundir.py`, `mitjax/testreport_jax.py` (git mv; stdlib only, loadable by path);
  shims at `reference/make_rundir.py`, `tools/testreport_jax.py` re-export them (scripts unchanged); the oracle
  scripts' snapshots archive the moved modules where the commit has them. Nothing in `mitjax/` outside the tests
  loads `reference/` or `tools/`. `environment.yml` (hand-written) + `test_env_pins.py`; `pip install -e .`:
  `scripts/pip_install_check.sbatch` (a new venv on env mitjax, editable install of a snapshot, imports from outside
  the tree, `python -m mitjax run` + `python -m mitjax.testreport_jax`).
- **R5.** The cpp link farms are in `$MJX_CACHE/config_cpp` (the old `$MJX_RUNS/config_cpp`, 203 MB, 73 farms, is
  unused, a candidate for removal).
- **Neutral defaults.** `mitjax/paths.py` names no machine path: `MJX_UPSTREAM` required (or `$MJX_WORK/upstream/
  MITgcm`), `MJX_RUNS` ./runs, `MJX_CACHE` the user cache, `MJX_PYTHON` the running interpreter, `MJX_REFERENCE`
  optional; an unset location raises `MissingPath` naming the variable and levante.env. `levante.env` (MJX_WORK,
  MJX_PYTHON) is sourced by all 54 batch scripts and tier runners (resolved values diffed: identical to before).
  Consequence: without levante.env (or MJX_REFERENCE) the oracle-dependent test modules fail to COLLECT with
  MissingPath (plan Task 11 must turn that into clean skips); the smoke command needs `. ./levante.env`.
- **The no-reference test** `test_no_reference.py` (tier1x): fwd_gyre, fwd_cs32, grad_col, shard_p2 as
  `noref_probe.py` L0 children with only MJX_UPSTREAM, MJX_RUNS, MJX_CACHE set (MJX_REFERENCE unset), experiments
  copied outside the tree; planted control (probe level PLANT, toolchain read re-pointed at the real oracle).

| job | what | commit | result |
|---|---|---|---|
| 27929160 | gate R2 + test_config + test_cpp_live | 483f3d3 | PASS 50/50 |
| 27929161 | clang measurement (test_toolchain_system with MJX_CPP=clang-cpp) | 483f3d3 | measurement: 3 passed, 29 failed (text only) |
| 27929314 | pip install -e check | HEAD at start | PASS (`PIP INSTALL CHECK PASS`) |
| 27929417 | session-2 gate (21 files) | d2dea7a | 257/261: 3 test-helper failures (test_paths, test_no_reference's grad_col, test_coverage's HAVE_NETCDF control), fixed in 6f12cfd; the three files 7/7 in job 27936400 |
| 27929418 | tier 1 | HEAD at start | 97/98 (test_paths, fixed in 6f12cfd as above) |

