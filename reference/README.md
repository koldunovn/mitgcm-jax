# Fortran oracle: builds and runs of MITgcm master (plan Task 3a)

The oracle is our own gfortran build of MITgcm master at the pinned commit (`63cdc0b`, branch `pinned` of
`$MJX_UPSTREAM`), built and run the way `verification/testreport` builds and runs a forward test. Everything heavy
lives under `$MJX_REFERENCE` (`mitjax/paths.py`; on Levante `$MJX_WORK/reference`, `levante.env`):

| path | what |
|---|---|
| `build/<name>/` | one build: `scripts/` (this directory at the mitjax commit, `git archive`), `MITgcm/` (upstream snapshot, the genmake2 rootdir), `bld/` (Makefile, links, preprocessed `*.f`, objects), `options/`, logs, checks |
| `bin/<name>/` | the frozen result: `mitgcmuv` (read-only), `sha256`, `build.log`, `logs/`, `genmake2_command.txt`, `genmake_state`, `Makefile`, `PACKAGES_CONFIG.h`, `AD_CONFIG.h`, `options/`, check reports, `ldd.txt`, `provenance.txt`, the optfile |
| `runs/<experiment>/<input dir>/<run id>/` | one run directory (`make_rundir.py`) |

`<name>` = `<experiment>-<code|code_ad>-<upstream short hash>-<mitjax short hash>[-tag]`, e.g.
`tutorial_barotropic_gyre-code-63cdc0b-f48c062`. Nothing under `$MJX_REFERENCE` is deleted or overwritten: every
script refuses an existing target; a new commit gives a new name. Job logs: `$MJX_WORK/logs/<job name>-<job id>.out` (`SBATCH_OUTPUT` in levante.env).

## Building

```bash
sbatch -J mjx_build_m1 reference/jobs/build.sbatch $(git rev-parse --short HEAD) \
    tutorial_barotropic_gyre:code tutorial_baroclinic_gyre:code advect_xy:code advect_xz:code \
    global_ocean.90x40x15:code tutorial_global_oce_optim:code_ad
```

Arguments go after the script name, never through `sbatch --export` (it splits values at commas, [E§9]). The job
(compute partition, 30 min) builds all listed codes in parallel, each from a frozen snapshot: it unpacks
`git archive <commit> reference mitjax/paths.py` into `build/<name>/scripts` and runs `build.sh` from there, so
uncommitted edits never reach a binary. Verdict: one `BUILD OK <name>` / `BUILD FAIL <name>` line per build.

`build.sh` follows testreport's forward sequence (`verification/testreport:1765-1771`, functions at 381-417,
519-540, 542-600) with testreport's defaults:

| testreport default | here |
|---|---|
| `OptLev=1` → genmake2 `-ieee` (`testreport:1100, 408-409`) | `-ieee`: the optfile's IEEE branch, `FOPTIM='-O0'` (`linux_amd64_gfortran:74-99`) |
| `MPI=0` → no `-mpi` (`testreport:1133, 414-416`) | serial build, no MPI (no `SIZE.h_mpi`) |
| `MULTI_THREAD=f` (`testreport:1135`) | no `-omp` |
| `genmake2 -ds -m make -mods=../code -optfile=...` (`testreport:394-406`) | the same, with absolute `-rootdir` (snapshot) and `-mods` |
| `make Clean` (`testreport:465-490`) | not needed: the build directory is new |
| `make depend`, `make` (`testreport:519-540, 542-600`) | the same (`make -j 16`) |

Upstream source: genmake2 writes generated sources into its rootdir (exch templates in `eesupp/src`,
`pkg/exch2`, `pkg/regrid`, the mnc templates; `tools/genmake2:2376-2405, 2571`), so the read-only clone is never
the rootdir: each build gets `git archive pinned tools eesupp model pkg doc/tag-index verification/<exp>/<code>`.

Checks (`check_build.py`; each fails the build):
- **Dropped packages** [E§9]: the requested set is derived independently of genmake2's messages — the first
  `packages.conf` in the build directory or a `-mods` directory, else `default_pkg_list`
  (`tools/genmake2:2444-2459`), groups of `pkg/pkg_groups` expanded, `-name` entries removed
  (`tools/genmake2:2483-2500`) — and every requested package must appear in the Makefile's `ENABLED_PACKAGES` and
  as `#define ALLOW_<NAME>` in `PACKAGES_CONFIG.h`, before and after `make`. Packages added by `pkg/pkg_depend`
  are listed, not failed. genmake2 itself drops mnc/profiles/obsfit with only a warning when its NetCDF test fails
  (`tools/genmake2:2527-2568`); `build.sh` also fails on that warning.
- **Flags**: `FFLAGS`/`F90FLAGS` contain `-ffp-contract=off` and `-fconvert=big-endian`; no `-ffast-math`, `-Ofast`,
  `-funsafe-math-optimizations`, `-fassociative-math`, `-freciprocal-math`, `-ffp-contract=fast|on`, `-march=`,
  `-mfma`, `-mavx`, `-fno-signed-zeros`, `-ffinite-math-only`, `-fno-trapping-math` in any flag variable;
  `DEFINES` has `-DHAVE_NETCDF`; `FOPTIM` is exactly `-O0`.
- **Forward only**: `AD_CONFIG.h` (written by plain `make`, `tools/genmake2:3337-3339`) undefines
  `ALLOW_ADJOINT_RUN` and `ALLOW_TANGENTLINEAR_RUN`.
- **Options**: every `*OPTIONS.h` the build used (the `-mods` copy where there is one) is kept with its source path
  and its effective macros (`cpp -traditional -P -dM` with the build's `DEFINES`/`INCLUDES`) in `options/`.

The M1 builds (one per code directory; testreport runs every `input.<v>` of an experiment with its one build):
`tutorial_barotropic_gyre/code` (no `packages.conf`: `default_pkg_list` = gfd), `tutorial_baroclinic_gyre/code`
(gfd, diagnostics, mnc), `advect_xy/code` (custom `ini_{salt,theta,vel}.F`; `CPP_OPTIONS.h` defines
`ALLOW_ADAMSBASHFORTH_3`, so `input.ab3_c4` shares the build), `advect_xz/code` (defines `NONLIN_FRSURF`: `input`,
`input.nlfs`, `input.pqm` share it), `global_ocean.90x40x15/code` (exch2, oceanic minus kpp, cd_code, down_slope,
ggl90, ptracers, sbo, diagnostics), `tutorial_global_oce_optim/code_ad` forward only (gfd, cd_code, gmredi, autodiff,
cost, ctrl, grdchk; plain `make` → `mitgcmuv` with `ALLOW_AUTODIFF` but neither adjoint nor tangent run).

**`code_ad` forward-only call path (verified at 63cdc0b):** with `ALLOW_AUTODIFF` and neither run mode defined,
`the_model_main.F` takes the branch at line 630 (`#elif ( defined ALLOW_AUTODIFF )`) and calls `THE_MAIN_LOOP` at
**line 711** (forward run within the AD setting, 704-714); `COST_FINAL` is called in `the_main_loop.F:774`. Line 747
(cited in the plan) is the `#else /* ALL AD-related undef */` branch, which a `code_ad` build does not compile. After
the loop, `the_model_main.F:733-737` calls `GRDCHK_MAIN` when `useGrdchk` — and
`tutorial_global_oce_optim/input_ad/data.pkg` sets `useGrdchk=.TRUE.`, so a forward-only run of `input_ad` also runs
the gradient check's perturbed forward runs (Task 3b: look at what it prints; its "adjoint" gradient is not a TAF
gradient).

## Optfile audit (`optfile_levante_gfortran` vs master's `tools/build_options/linux_amd64_gfortran`)

The file is master's optfile **byte for byte** between the `MJX-BEGIN-MASTER` / `MJX-END-MASTER` lines
(sha256 `60dc7148…` at 63cdc0b; `test_optfile_is_master_plus_levante_block` compares it with `git show pinned:...`),
followed by the `MJX-LEVANTE` block. Every difference:

| difference | reason | numerics |
|---|---|---|
| `FFLAGS`/`F90FLAGS += -ffp-contract=off` | the JAX bitwise gates run without FMA [E§5]; x86-64 gfortran without `-march` emits no FMA anyway | none expected (no FMA either way); explicit guarantee |
| NetCDF: `INCLUDES=-I$MJX_NF_PREFIX/include`, `LIBS=-L<netcdf-fortran lib> -L<netcdf-c lib> -Wl,-rpath,...` (genmake2 appends `-lnetcdff -lnetcdf`, `tools/genmake2:1184-1215`) | Levante's netcdf-fortran prefix has no libnetcdf, and without an rpath the executable cannot find `libnetcdff.so` at run time (the modules set no `LD_LIBRARY_PATH`); master's `nf-config` branch would link but not run. A conda `nc-config` may be first on `PATH`, so `build.sh` takes the C library directory from `ldd libnetcdff.so` and removes conda directories from `PATH` | none (I/O library location) |

Verified in master's optfile (no change needed): `-fconvert=big-endian -fimplicit-none` (line 64), `-mcmodel=medium`
(69-70), `-fallow-argument-mismatch` for gfortran ≥ 10 (61), `FOPTIM='-O0'` under `-ieee` (95), `-O3 -funroll-loops`
only without it (76), no fast-math anywhere.

Differences from the ECCO port's optfile and build (R = the ECCO port's private handoff tree,
`reference/optfile_levante_gfortran`, `reference/build.sh`): R's master block came from ECCO's Docker optfile, which
is identical to master's; R added `seaice_growth.F` to `NOOPTFILES` (copying the V4r4 ifort optfile) — dropped here:
master's optfile does not do it and no M1 experiment compiles seaice (M4 decides with its own build); R built
**without** `-ieee` (`-O3 -funroll-loops`) — here testreport's default `-ieee` (`-O0`). With SSE2 arithmetic, no FMA
and no fast-math the two should round identically, but that is not measured; the yardstick (Task 3b) is defined
against testreport's own build, so the oracle uses testreport's options. R used the OpenMPI netcdf-fortran module
for MPI builds; the plain builds here are serial (`netcdf-fortran/4.5.3-gcc-11.2.0`).

Toolchain (recorded per build in `provenance.txt`): modules `gcc/11.2.0-gcc-11.2.0`
(GNU Fortran (Spack GCC) 11.2.0, `cpp -traditional -P` from the same GCC), `netcdf-fortran/4.5.3-gcc-11.2.0`
(netCDF-Fortran 4.5.3 on netCDF-C 4.8.1, HDF5 1.12.1). `ldd.txt` records the run-time libraries (libgfortran, libm)
the binary resolves — needed for the libm transcription audit (Task 7c).

## Run directories (`make_rundir.py`)

```bash
python reference/make_rundir.py EXPERIMENT INPUT_DIR [--run-id ID] [--binary <bin dir name>]
# e.g. make_rundir.py tutorial_global_oce_optim input_ad --binary tutorial_global_oce_optim-code_ad-63cdc0b-f48c062
```

Exactly testreport's run directory (verification/testreport at 63cdc0b):
- run directory `run` for the main input directory, `tr_run.<v>` for a variant (`testreport:1606-1607, 1793`),
  inside `<run id>/verification/<experiment>/`, a mirror of upstream's `verification/` (every other experiment a
  symlink into the clone), because `prepare_run` scripts reach other experiments by relative paths
  (`tutorial_global_oce_optim/input_ad/prepare_run` links `../../tutorial_global_oce_latlon/input/*.bin` and
  `../../isomip/input_ad/ones_64b.bin`; `global_ocean.90x40x15/input/prepare_run` the same `*.bin`).
  `<run id>/rundir` points to the run directory.
- `linkdata` (`testreport:771-834`) with `MPI=0`, `MULTI_THREAD=f`: `linkdata run input` (`:1771`) or
  `linkdata tr_run.<v> input.<v> input` (`:1800`), the same with `input_ad` (`:1323-1332`): every visible,
  non-directory entry of each directory (`ls -1 | grep -v CVS`, `:816`) not yet readable in the run directory is
  linked `ln -sf ../<dir>/<name> <name>` (`:821`), so the variant's files shadow `input/`'s; the `*.mpi` and
  `eedata.mth` branches only remove links of an earlier MPI/threaded run (none in a new directory), and those files
  are linked under their own names, as testreport does; then `./prepare_run` if executable (`:829-831`, output in
  `<run id>/prepare_run.log`).
- `--binary`: `mitgcmuv` → `bin/<name>/mitgcmuv`, checked against `bin/<name>/sha256` (testreport links
  `../build/mitgcmuv`, `:861-864`).
- Required inputs (a check testreport does not make): every non-blank quoted value of a `*File` key in `data`, of a
  `*_weight*` key in `data.ctrl`, and the pickup `pickup.<nIter0 as I10.10>` when `nIter0 > 0`, must exist as
  `<name>` or `<name>.data` (the names `MDS_READ_FIELD` tries, `pkg/mdsio/mdsio_read_field.F:229-255`). A
  `prepare_run` whose source directory is missing prints an error and exits 0, which this check turns into a refusal.
- Result: `MANIFEST.json` (every entry with its link target and origin), then `READY`; on a refusal `REFUSED.txt`
  (the directory stays; nothing is deleted) and exit 1.

`test_make_rundir_matches_testreport` compares the result with a literal hand-derived list and with **testreport's
own `linkdata` function**, cut from `verification/testreport` at `pinned` and run in a second mirror
(`advect_xz/input.nlfs`: overlay; `tutorial_global_oce_optim/input_ad`: `prepare_run`).

## Running (`run.sh`, `jobs/run.sbatch`)

```bash
sbatch -J mjx_run_x reference/jobs/run.sbatch RUN_TOP [RUN_TOP ...]          # existing run directories
sbatch -J mjx_run_x --dependency=afterany:<build job> reference/jobs/run.sbatch --make <commit> \
    tutorial_barotropic_gyre input tutorial_barotropic_gyre-code-63cdc0b-<commit>   # make + run (run id job<id>)
```

As testreport runs a serial test (`testreport:1414-1417`, `896-899`): `( ./mitgcmuv > output.txt ) >> run.tr_log 2>&1`
in the run directory. **The model's standard output — the STDOUT with `%MON` and `cg2d_init_res` that testreport
compares with `results/output.txt` — is `<run id>/rundir/output.txt`**; stderr goes to `run.tr_log`. Success =
exit 0 and `PROGRAM MAIN: Execution ended Normally` in the last lines of `output.txt` (`testreport:899`); both are
checked. `run_provenance.txt` records host, binary, sha256 and `ldd`. `run.sh` refuses a directory without `READY`,
with `REFUSED.txt`, or one that already ran. Chained runs use `afterany` (a failed build ends the run job at once with
a "no frozen binary" failure instead of leaving it pending forever).

## Instrumented oracle (`jaxdump/`, plan Task 4)
- `reference/jaxdump/SUBSTEPS.md` is the reference: stage table (generated from `jaxdump/instrument.py`), record
  format, iteration rule, the changes from the ECCO port and the c66g -> master audit of the instrumented routines.
- Build: `sbatch -J mjx_build_jd reference/jobs/build.sbatch <commit> <exp>:<code>:jaxdump ...` -> binaries
  `$MJX_REFERENCE/bin/<exp>-<code>-63cdc0b-<commit>-jaxdump/` with `jaxdump_stages.txt`, `instrument_report.txt`,
  `jaxdump_mods/`. Current: commit `5f16129`, build job 27826872 (all six M1 codes).
- Invisibility: `sbatch -J mjx_jdrun reference/jobs/jaxdump_runs.sbatch <commit> EXP INPUT PLAIN_BIN JD_BIN ...`;
  verdict files `$MJX_REFERENCE_RUNS/<exp>/<input>/invisibility-job<ID>.txt`; runs `job<ID>-{plain,jdoff,jdon}`
  (the jdon run's dumps in `job<ID>-jdon/dumps`). All nine M1 variants INVISIBLE (jobs 27826873, 27826998).
- Dumps: `mitjax/io/dump.py` (DumpSet, probe_decode); comparisons `tools/diffdump.py`, `tools/step_vs_dump.py`.

## Tests (`scripts/tests/test_reference_build.py`, tier 1)

`test_optfile_is_master_plus_levante_block`, `test_m1_binaries` (oracle-dependent: fails while a binary is missing),
`test_bin_problems_negative_controls`, `test_make_rundir_matches_testreport` (two variants),
`test_make_rundir_refuses_missing_input`, `test_requested_package_sets` (hand-derived package sets of all six
builds), `test_build_checks_negative_controls`. Each negative control was also shown to bite by mutating the code
(variant order reversed, `prepare_run` skipped, input check disabled, `-name` disables ignored, package check
disabled, forbidden-flag check disabled, optfile edited, sha256 comparison disabled): each mutation fails its test.

## Task 3b: oracle runs, yardstick, invariance checks, FD oracle
- Registry and checks: `reference/reference_runs.py` (`--check`, `--lock` -> `reference_runs.lock.json` with the sha256 of
  every reference file, `--docs` -> `docs/YARDSTICK.md`, `docs/REFERENCE_RUNS.md` via `reference/reference_docs.py`).
- Runs: `jobs/oracle_runs.sbatch MJX_COMMIT EXP@INPUT@BIN@LABEL[@make_rundir option ...]` (run ids `job<ID>-<LABEL>`).
- `make_rundir.py` options beyond testreport: `--mpi` (linkdata's MPI branch for a serial build with the SIZE.h_mpi
  tile shape), `--set FILE:GROUP:KEY=VALUE` (namelist overlay written as a file; recorded in OVERLAY.txt and MANIFEST),
  `--copy SOURCE:NAME`; `linkdata_plan()` is the pure file list (for mitjax/config).
- Second tiling: build tag `retile` with `reference/retile/<exp>-<code>/SIZE.h` (build job 27827362).
- Comparisons: `reference/compare_runs.py BASE TEST --kind tiling|output|debug [--exp E --variant V]`.
- Call order (debugMode runs): `reference/call_order.py` -> `reference/call_order/<exp>-<input>.txt`.
- FD oracle of `tutorial_global_oce_optim/input_ad` (option (i), Nikolay 2026-10-01): `reference/grdchk_adxx.py` writes
  the zero `adxx_qnet` files (layout of an adjoint build) into `$MJX_REFERENCE/grdchk_adxx/`, copied into the run
  directory with `make_rundir.py --copy` (job 27827478: zero and random file, identical cost and FD lines).

## Lane A session 4 (2026-10-01): jaxdump3, run registration, the global_ocean pickup-halo ±0
- Build tag `jaxdump3` = commit `275a02b` (build job 27829310; advect_xy, advect_xz, global_ocean.90x40x15):
  initialisation stages `I00_pickup_read` (READ_PICKUP before its exchanges), `I01_read_pickup`, `I02_ini_fields`;
  `rStarDhCDt` in group `r` (so `S12_calc_rstar` holds the value the step's MONITOR reads); mixed signed-zero exchange
  probes `zu*` (u=-0, v=+0) and `zv*` (u=+0, v=-0). SUBSTEPS.md lists the stages.
- Runs (`reference/jobs/jaxdump_runs.sbatch`, invisibility triple each): JAXDUMP_STEPS=0:1:2:9 for advect_xz
  (input, input.nlfs, input.pqm) and advect_xy/input.ab3_c4 (job 27829313), 0:1:2:15 for advect_xy/input (27829314),
  default steps for global_ocean (27829315). Registered as kinds `jdplain3/jdoff3/jdon3`; the core lane's
  job27828737 triple as `jdplain2/jdoff2/jdon2` (build tag `jaxdump2` = b73c05c). `jdon` (job27826873/27826998)
  stays the one oracle per variant that the substep gates address; `check_run` also compares the dumped iterations
  (dumps/jaxdump_info.txt) with the registered steps.
- Pickup-halo ±0 (global_ocean): exch2 forms each buffer value of the C-grid vector exchange as
  `sa1*array1 + sa2*array2` at the source index (`pkg/exch2/exch2_put_rx2.template:229-230`, `:317-318`), i.e.
  `1*u + 0*v` and `0*u + 1*v` without rotation; a -0 therefore arrives as +0 where the other component's value at the
  same index is +0 or positive. `scripts/tests/test_jaxdump.py::test_pickup_halo_signed_zeros_from_exch2_buffer`
  reproduces every halo point of uVel, vVel, guNm1, gvNm1 after READ_PICKUP from the I00 interior with that formula.

## Lane A session 5 (2026-10-01): the Fortran oracle's own restart
- `jobs/restart_runs.sbatch MJX_COMMIT EXP INPUT_DIR JD_BIN PCHKPT K NB STEPS`: run A (overlay
  `data:PARM03:pChkptFreq=PCHKPT`: permanent pickups, do_write_pickup.F:60-61) and its restart B (same overlay plus
  `nIter0=K`, `nTimeSteps=NB`; A's `pickup.<K>.*` copied with `make_rundir.py --copy`), both dumps on with the same
  JAXDUMP_STEPS, then `restart_compare.py A B` (bit patterns of every dumped record incl. halos, MONITOR blocks,
  solver lines, %CHECKPOINT lines, files both runs write) into
  `$MJX_REFERENCE_RUNS/<EXP>/<INPUT_DIR>/restart-job<ID>.{txt,json}`.
- `make_rundir.py` now accepts a per-tile pickup (`pickup.<n>.<iG>.<jG>.data`, MDS_READ_FIELD's globalFile=F
  branch, mdsio_read_field.F:446-454) for a nIter0 > 0 start.
- Job 27831497 (tutorial_barotropic_gyre, jaxdump2 binary, pChkptFreq=6000.0, K=5, NB=5, steps 5-9): registered as
  kinds `restart_a`, `restart_b` (gate fixtures with overlays, never yardsticks; their pickups are in the lock);
  pattern and cause in `scripts/tests/test_restart_oracle.py`.

## Lane A session 6 (2026-10-01): the M2 oracle
- Builds (all checks passed, `test_m2_binaries`): plain `<exp>-<code>-63cdc0b-06b4417` (job 27832154), instrumented
  `...-8316e95-jaxdump` (job 27832225; jaxdump group `P` = ptracers, stage `T04_ptracers_integrate`, SUBSTEPS.md), gcov
  `...-8316e95-gcov` (job 27832234, lane E's `reference/coverage/build_gcov.sbatch`) of adjustment.cs-32x32x1/code,
  solid-body.cs-32x32x1/code, advect_cs/code, global_ocean.cs32x15/code, global_ocean.90x40x15/code_ad,
  tutorial_global_oce_latlon/code, tutorial_advection_in_gyre/code, tutorial_tracer_adjsens/code_ad (forward only).
  Preprocessed sources: `$MJX_REFERENCE/build/<name>/bld/*.f`; `tools/cpp_live.py <exp> <routine> --code <code>`
  works for every M2 code (it reads the frozen build records).
- `make_rundir.py`: a prepare_run never removes anything — `rm`/`mv`/`unlink`/`rmdir` refused before it runs; `gunzip`
  runs as `gunzip --keep` through a shim first on PATH (`global_ocean.cs32x15/input.viscA4/prepare_run` gunzips three
  linked `.gz` files: same decompressed files, the `.gz` links stay; MANIFEST `prepare_run.shims`); a linkdata entry
  missing after prepare_run refuses the directory. Required inputs: `horizGridFile` is read as
  `<name>.face<nnn>.bin` (ini_curvilinear_grid.F:282-283); the genarr weight `wunit[.data]` is written by the model
  itself (ctrl_init_fixed.F:92-110).
- `run_verdict.py`: the grdchk stop of the six M2 code_ad variants declared (the optim signature with their control
  name; `adxx_<name>.0000000000.001.001.data`).
- `probe_map.py DUMPS [--npz OUT] [--json OUT]`: decodes the X00 exchange probe of any layout (exch1, exch2 single
  facet, exch2 cube): per probe field halo points written / unwritten (corner blocks vs edge strips), copies from a
  halo point, cross-face copies, component swaps, sign flips; signed-zero records as -0 counts in interior and halo.
  `--npz`: per field int arrays [tile, j, i] src_tile, src_ja, src_ia, comp, sign, written + the tile layout.
- Runs (registered in `reference/reference_runs.py`, kinds `yardstick`/`jdoff`/`jdon` per variant, `fdzero`/`fdrandom`
  for the code_ad variants): invisibility triples jobs 27832226 (cube: adjustment input/nlfs, solid-body, advect_cs),
  27832228 (cs32x15 input, viscA4), 27832348 (cs32x15 in_p, after the make_rundir fix), 27832229 (code_ad: 90x40x15
  input_ad/kapgm/kapredi/bottomdrag, tracer_adjsens input_ad/som81), 27832231 (latlon, advection_in_gyre); FD oracle
  job 27832347 (zero / random adxx from `$MJX_REFERENCE/grdchk_adxx/30b10ae-*`). All invisible (cs32x15/input with the
  new exemption DIAGSTATS_UNSET_REGIONS, verdict file `invisibility-job27832228-r30b10ae.txt`).
- Probe maps for lane B: `$MJX_REFERENCE/probe_maps/<exp>-<input>-<run id>.{json,npz}` (the JSON summaries also in
  `reference/probe_maps/`, frozen by `scripts/tests/test_m2_oracle.py`).
- Coverage: `reference/coverage/CURRENT_M2` -> `docs/coverage/<experiment>.md` (90x40x15 code_ad:
  `global_ocean.90x40x15-code_ad.md`).

## Lane A session 7 (2026-10-01): the remaining M2 oracle items
- `build.sbatch` takes any `code*` directory that exists at `pinned`; `build.sh` knows the genmake2 options per code
  directory: code/code_ad as testreport, `code_min` with `-standarddirs eesupp` (README of adjustment.cs-32x32x1,
  "to build"; no jaxdump build of it: model/src is not compiled), anything else refused. testreport runs genmake2 in
  `verification/<exp>/build`, where genmake2 sources `./genmake_local` (tools/genmake2:1447, 1768-1770): `build.sh`
  now copies the experiment's `build/genmake_local` into the build directory (10 experiments have one; of the M1/M2
  ones only tutorial_advection_in_gyre: `ALWAYS_USE_F90=1`, read only by the pgf77 optfiles and
  tools/adjoint_options/adjoint_default, so its session-6 binaries are unaffected). gcov scripts take code_min too.
- `make_rundir.py`: `input_min` (linked like input_ad: its own directory, run directory `run`); the check after
  prepare_run is the pure `removed_by_prepare_run(before, after)` (a missing or replaced linkdata entry), planted in
  `test_make_rundir_removal_check_pure` on fabricated listings.
- Builds (job 27832600, `78ca390`): `global_ocean.cs32x15-code_ad-63cdc0b-78ca390` (+`-jaxdump`; seaice, thsice, exf
  compiled), `adjustment.cs-32x32x1-code_min-63cdc0b-78ca390`; gcov `...-78ca390-gcov` (job 27832601).
- `global_ocean.cs32x15/input_ad` is M2: its data.pkg leaves useSEAICE at the default .FALSE. and sets useTHSICE,
  useEXF .FALSE. (STDOUT: "compiled but not used"). But in an ALLOW_AUTODIFF build SEAICE_INIT_VARIA runs anyway
  (packages_init_variables.F:428-440, the IF (useSEAICE) is compiled out), as do the seaice/thsice readparms and cost
  init routines (coverage). Its pickup.0000072000 (from ../input) has no GuNm2/GvNm2 while code_ad defines
  ALLOW_ADAMSBASHFORTH_3: "approximated Restart", mom_StartAB = 1 (stderr; declared in run_verdict.py).
  Triple job 27832624 (INVISIBLE; FAILED in sacct only because the grdchk stop was declared afterwards from these
  runs), FD oracle job 27832649 (adxx_theta, 12 tiled files from `$MJX_REFERENCE/grdchk_adxx/d117407-*`).
- `adjustment.cs-32x32x1/input_min` (code_min): run job27832625-plain, kind `minimal`: EEBOOT + W2 topology only. Its
  W2 log equals the code build's except the CUMSUM tile matrix (code/W2_OPTIONS.h defines W2_CUMSUM_USE_MATRIX).
- Coverage: `tools/coverage.py` titles each report by the milestone of its rungs (`RUNG_MILESTONE`); the M2 reports
  were re-rendered (-r27832652, -r27832653: only title and commit line changed); new reports
  `docs/coverage/global_ocean.cs32x15-code_ad.md`, `docs/coverage/adjustment.cs-32x32x1-code_min.md` (job 27832650).
- Registry: the FD runs' copied adxx files are now locked from MANIFEST.json (before, only `adxx_qnet.*` was globbed,
  so the M2 FD runs' adxx files were not in the lock). `docs/YARDSTICK.md`: TAF adjoint monitor table
  (`reference_runs.adm_monitor_blocks`) and the minimal-case section.

## Lane A session 8 (2026-10-01): output requests off, planted controls, M2 MAX/MIN tables
- Output-request overlay runs (`job27833806-{mncoff,diagoff}`, `job27833807-*` for the code_ad variants): every M2
  variant whose plain STDOUT says `pkg/mnc compiled and used` (global_ocean.cs32x15/input, tutorial_advection_in_gyre,
  global_ocean.90x40x15/input_ad.bottomdrag) or `pkg/diagnostics compiled and used` (adjustment.cs/input,
  solid-body, advect_cs, cs32x15 input/viscA4/in_p/input_ad, advection_in_gyre). Not mnc (measured): 90x40x15/input
  (mnc not compiled), input_ad/.kapgm/.kapredi, cs32x15/input.viscA4, latlon (compiled, useMNC=F). Registry
  `OUTPUT_OFF_M2`; `reference_runs.output_request_problems`: model lines (%MON incl. trcstat, %SBO, cg2d, cost, ADM)
  identical and every MDS pickup both runs wrote byte-identical, in all 11 M2 and 4 M1 overlay runs.
- `run_verdict.py`: a stop can be declared per overlay (`stop_key`: (exp, input, sorted KEY=VALUE of the overlays),
  falling back to (exp, input)); the code_ad overlay runs print the switched-off package's two warnings on stderr.
- `reference/jobs/oracle_runs.sbatch`: `@--dumps@STEPS` runs a jaxdump binary with dumps on.
- Planted first-guess controls of the 90x40x15 code_ad family (`job27833840-ctrlxx`, kind ctrlxx, as optim's
  job27829159-ctrlxx): `reference/grdchk_adxx.py --prefix '' --random 20261001 --amp A` writes the control file
  itself; overlays doInitXX=F, doMainUnpack=F; jaxdump_m2 binary, dumps at steps 0-3; the planted fields reach the
  model (xx_*.effective nonzero; about half the model lines differ from the zero-control run).
- MAX/MIN site tables of the M2 oracle builds: `reference/jobs/minmax_sites.sbatch` (jobs 27833815, 27833918) ->
  `$MJX_REFERENCE/minmax_sites/<build>.json` (+ .sha256). `tools/fortran_minmax_sites.py` compiles each file in a
  scratch directory: builds with Fortran modules (tutorial_advection_in_gyre's ptracers_dyn_state_mod) need the
  build's .mod files on the module path (`-I<bld>`, recorded as meta.module_search).

## Lane A session 9 (2026-10-02): the M3 oracle (plan Task 28)
- Builds (job 27840380, `463504e`, ALL BUILDS OK): plain + jaxdump `vermix`, `front_relax`, `ideal_2D_oce`,
  `tutorial_reentrant_channel`, `MLAdjust` (`<exp>-code-63cdc0b-463504e[-jaxdump]`) and the jaxdump build
  `global_ocean.90x40x15-code-63cdc0b-463504e-jaxdump` (input.dwnslp/input.idemix run the M1 plain build f48c062 and
  the M1 gcov build 7fd68e0); gcov builds `...-463504e-gcov` (job 27840381).
- jaxdump (reference/jaxdump/SUBSTEPS.md "M3"): groups `K` KPP, `Q` PP81, `Y` MY82, IDEMIX in `k`; stages P07_kpp,
  P08_pp81, P09_my82, P10_kpp_exch, P11_ggl90_exch, T05_opps, T06_convective_adjustment (tracers_correction_step.F
  now instrumented); anchors re-validated on pinned. Every mixing stage that runs holds nonzero fields
  (`scripts/tests/test_m3_oracle.py`).
- Invisibility triples of all 24 M3 variants (jobs 27840385 vermix + front_relax, 27840386 ideal_2D_oce + reentrant
  channel + dwnslp/idemix, 27840387 MLAdjust): ALL INVISIBLE with the existing exemptions only. front_relax/input.in_p
  is M3 (its code compiles gfd, gmredi, diagnostics only; no M4 package).
- Yardstick (docs/YARDSTICK.md "M3"): deciding variable 16 digits in all 24; below 10 only in the vermix column (Ssd 1:
  uniform salt; Vmx 8/9: v ~ 1e-90), proposals there. Coverage: `reference/coverage/CURRENT_M3` -> docs/coverage/
  (job 27840435, 27840436 for global_ocean.90x40x15-dwnslp-idemix.md); pkg/layers' unbalanced `#ifdef ALLOW_LAYERS`
  (docs/ISSUES_UPSTREAM.md) leaves two layers routines unmapped in the reentrant-channel report.
- MAX/MIN tables of the five M3 plain builds (job 27840408): 0 conflicts over 21 tables (766 sites); in
  `mitjax/tests/test_minmax_sites.py` as M3_BUILDS with exact counts.

## Lane A session 11 (2026-10-02): the M4 oracle
- Builds (job 27855975, `704fd6b`, ALL BUILDS OK): plain + jaxdump of 1D_ocean_ice_column, offline_exf_seaice and lab_sea
  (code, code_ad), seaice_itd (code); jaxdump of global_ocean.cs32x15 code/code_ad (its sea-ice variants run the M2 plain
  builds 06b4417 / 78ca390). 1D_ocean_ice_column jaxdump rebuilt at `1ac79cb` (job 27856045) with the B-grid stages
  I00b_seaice_begin / I01b_dynsolver (reference/jaxdump/SUBSTEPS.md, session-11 note). gcov `dad681f-gcov` (27856009).
- Invisibility triples of the 31 M4 variants: ALL INVISIBLE (27855986-27855989, 1D: 27856057); offline_exf_seaice
  input_ad.obcs is M5. The grdchk stops of the 9 forward-only code_ad variants are declared in run_verdict.py (M4_STOPS).
- FD oracle runs (27856073, 27856074): reference/grdchk_adxx.py `--subdir` and make_rundir `--copy <dir>/<file>` for
  lab_sea's ctrlDir=./ctrl_variables.
- Registry: reference_runs.py M4_VARIANTS (plain_m4, jaxdump_m4, jaxdump_m4b); docs/YARDSTICK.md "M4" (below 10:
  lab_sea/input.hb87, seaice_itd/input.thermo). MAX/MIN tables of the 7 M4 plain builds (27856188): two build-dependent
  winners (seaice_growth.F:1847-1848), test_minmax_sites.py KNOWN_DIFFER.
- Coverage: one report per experiment and build (reference/coverage/run_gcov.sbatch, report_gcov.sbatch; `-code_ad`
  suffix); M4 rung checks in tools/coverage.py; pages listed in reference/coverage/CURRENT_M4.
