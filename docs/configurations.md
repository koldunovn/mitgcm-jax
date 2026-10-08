# Configurations: your own experiment

mitjax runs any MITgcm experiment directory laid out as the verification experiments are, inside or outside the
MITgcm tree, as long as it uses ported packages, options and branches. Anything that is not ported stops the run with
an error that names the Fortran routine (and usually the lines) it would need; it never runs silently with a
different model. The list of what is ported and what is refused is generated from the code and is the second half of
this page.

## What mitjax reads

An experiment directory holds, as for `testreport`:

- `code/` (or `code_ad/` for an adjoint build): `SIZE.h`, the `*_OPTIONS.h` files you change, `packages.conf`, and
  any `.F` files of your own (below). Files you do not provide come from `$MJX_UPSTREAM`, as genmake2 takes them.
- `input/` and variants `input.<v>`, `input_ad`, `input_ad.<v>`: the namelist files (`data`, `data.pkg`, `eedata`,
  `data.<package>`) and the binary input files; optionally a `prepare_run` script.
- `results/`: the reference output for the comparison (optional for your own experiment).

How each piece is read:

- **CPP options.** The experiment's `*_OPTIONS.h` (and the packages' defaults) are preprocessed with the system's C
  preprocessor as `genmake2` would preprocess them (`cpp -traditional -P` with genmake2's default macro set for
  gfortran; [install.md](install.md) lists the overrides). `packages.conf` is expanded as `genmake2` expands it
  (package groups, dependencies). In the code, `cfg.cpp.NAME` is `#ifdef NAME`.
- **`SIZE.h`.** Any tiling: `sNx`, `sNy`, `OLx`, `OLy`, `nSx`, `nSy`, `Nr`. The exchange maps for the tiles (the plain
  exchange and the cubed-sphere `pkg/exch2`) are computed from `SIZE.h` and the `data.exch2` topology as MITgcm's
  exchange routines compute them, and cached in `MJX_CACHE`. `nPx = nPy = 1` is required (mitjax splits tiles over
  devices instead of MPI processes, [parallel.md](parallel.md)).
- **Namelists.** Each file is read with the `NAMELIST` statements of the routines your build compiles, as the Fortran
  `READ` reads it: a variable that is not in its group stops with `UnknownNamelistVariable`, a value of the wrong
  type is an error, a repeated key keeps its last value. A value you do not set takes MITgcm's default, read from the
  `set_defaults.F` / `*_readparms.F` line the code cites.
- **The run directory** is made as testreport makes it ([running.md](running.md#run-directories)).

## Changing an experiment

Copy a verification experiment anywhere and change it: namelist values, input files, run length, `SIZE.h`. Notebook
04 does all of these (a 2 x 2 tiling of `tutorial_barotropic_gyre`, a longer run with another viscosity and a new
wind file) and builds a box from scratch (`code/` with `SIZE.h`, `CPP_OPTIONS.h`, `packages.conf`; `input/` written
with numpy).

**A new tiling changes the result where MITgcm's own result depends on the tiling.** The comparison with `results/`
then tells you how much, and it is a property of MITgcm, not of the port:

- `tutorial_barotropic_gyre` on 2 x 2 tiles matches the one-tile `results/` to 13 digits in the surface pressure; the
  mean velocities `Uav`/`Vav` match to 0 digits, because in the first records they are round-off noise around zero
  (about 1e-21) and any change of summation order changes them.
- `lab_sea/input` on one 20 x 16 tile instead of `code/SIZE.h`'s 2 x 2 tiles of 10 x 8 matches `results/` to 2 digits
  in the surface pressure and 6-7 digits in sea ice. MITgcm itself does the same: a gfortran build of MITgcm at
  63cdc0b with the one-tile `SIZE.h` gives exactly these numbers, and mitjax with one tile reproduces that Fortran run
  bit for bit (every MONITOR line and the pickups), as it reproduces the four-tile run. The cause is `SEAICE_LSR`: it
  solves its line systems within a tile, sees the neighbouring tile's previous iterate across a tile edge
  (`pkg/seaice/seaice_lsr.F:2001-2002`, one exchange per sweep at `:987`), and stops at `LSR_ERROR = 1.E-4`
  (`lab_sea/input/data.seaice`), so another tiling stops at another, equally converged, iterate (sea-ice velocity
  differs by up to 4e-4 m/s after 9 steps). MITgcm's tile-independent line solver (`SEAICEuseMultiTileSolver =
  .TRUE.` with `SEAICE_GLOBAL_3DIAG_SOLVER`) is not ported: `SEAICEuseMultiTileSolver = .TRUE.` stops in
  `SEAICE_READPARMS`. A tighter `LSR_ERROR` makes the tilings agree more closely. MITgcm's monitor also prints the
  `dynstat_sst`/`dynstat_sss` lines only with one tile (`pkg/monitor/monitor.F:122-131`); mitjax prints them in the
  same cases.

## MAX/MIN at build-dependent statements

Where a Fortran `MAX` or `MIN` has two equal arguments (`+0` and `-0`) or a NaN, which argument it returns depends on
how gfortran compiled that statement. mitjax reproduces gfortran's choice per statement (`MAX(a, b, p="a")`,
[fortran_map.md](fortran_map.md#how-to-read-a-mitjax-file-next-to-its-f)), measured from the project's gfortran
builds. At a few statements gfortran's choice differs between builds (the first: `pkg/seaice/seaice_growth.F:1847-1848`,
`SEAICE_areaLossFormula = 3`); there the code holds the measured choice per build, and a build is identified by a hash
of its preprocessed options and `SIZE.h` (`mitjax/config/build_identity.py`), not by its name.

- **Your build** (your own options, an edited `code_ad`, another tiling, or a verification build with a namelist
  switch that makes such a statement run where it was never measured) takes the documented default choice of that
  statement and issues one `mitjax.MinMaxDefaultWarning` naming the statement, once per build and statement in a
  Python process. The run is correct; bit-for-bit agreement with your own gfortran build is not guaranteed at that
  statement, and the two can only differ where its arguments tie (`+0` vs `-0`) or one is NaN.
- **Strict mode:** with `MJX_MINMAX_STRICT=1` (set by the project's own test suite) a verification build without a
  measured choice is refused instead, which keeps the measured tables complete.
- A statement without a documented default is refused in both modes.

To see or silence the warning, use Python's `warnings` module with the category `mitjax.MinMaxDefaultWarning`.

## Your own `.F` files

genmake2 compiles a `.F` in your `code/` directory instead of the MITgcm file of the same name. mitjax cannot compile
Fortran, so it checks every `code/*.F` (and `.F90`, `.c`) when the experiment is loaded (plan decision 11,
`mitjax/config/own_code.py`):

- a file equal byte for byte to the MITgcm file it replaces is fine (it builds the same model);
- a file equal to one of the verification experiments' own routines that mitjax has a port of is fine (the table
  "Experiment code files" below);
- anything else stops with `UnportedRoutine` naming the file and its routines. A comment-only change counts as a
  change.

To run a routine of your own, change its Python port instead (the file of the same name under `mitjax/`, see
[fortran_map.md](fortran_map.md)) in your checkout of mitjax, and leave the `.F` out of `code/`. The table of ported
experiment routines compares with the experiments' files in the MITgcm checkout, so a routine of a new experiment
cannot be registered there today.

## Adding a port

When a run stops with `NotImplementedError` (or `UnsupportedOption`, `UnportedPackage`, `UnportedRoutine`), the
message names the Fortran routine and, usually, the lines of the `.F` that are not ported. To port them:

1. **Find the place.** [fortran_map.md](fortran_map.md) gives the Python file of every ported `.F`, and lists the
   files of `model/src` and the ported packages that nothing in mitjax names ("not cited"). One `.py` per `.F`, same
   path under `mitjax/`, the routine as a function of the same name in lower case.
2. **Translate literally.** Same statements in the same order with the same parentheses; a docstring that starts with
   the Fortran call and `@63cdc0b <path>:<first>-<last>`; a `# :NN` comment per statement with its `.F` line.
   [READING_GUIDE.md](READING_GUIDE.md) shows how each Fortran construct is written (Fortran-index arrays,
   `jnp.where` with guarded divisions, `scan_levels`/`scan_k` for loops over `k`, `MAX(a, b, p=...)`, `real4()` for
   REAL*4 literals); `docs/KERNEL_GUIDE.md` has the rules.
3. **Remove the refusal** at the call site, and keep every other refusal of that routine: a branch you did not port
   must still stop the run.
4. **Check it** against the Fortran: testreport digits of a whole run against `results/`, and where you have a
   gfortran build of the experiment, against its STDOUT ([checking_a_change.md](checking_a_change.md)).
5. **Regenerate the reference pages** (`python tools/gen_docs.py`): the map and the options list below are generated
   from the code's citations and refusals.

<!-- BEGIN GENERATED: supported options (python tools/gen_docs.py) -->

## Supported options (generated)

*Generated from the code by `python tools/gen_docs.py`; do not edit between the GENERATED markers.*

### Packages

A package whose `use<Package>` switch is `.TRUE.` in your `data.pkg` must be one of these, or the set-up
stops with `UnportedPackage` naming the switch (`require_ported` in `mitjax/config/params.py`, called by
`check_packages` in `mitjax/drivers/model.py`). Compiled but switched-off packages are fine.

| package | how |
|---|---|
| `pkg/autodiff` | ported |
| `pkg/cal` | ported |
| `pkg/ctrl` | ported |
| `pkg/diagnostics` | accepted, output only: not ported, its output is not written |
| `pkg/down_slope` | ported |
| `pkg/ecco` | ported |
| `pkg/exf` | ported |
| `pkg/generic_advdiff` | ported |
| `pkg/ggl90` | ported |
| `pkg/gmredi` | ported |
| `pkg/grdchk` | ported |
| `pkg/kpp` | ported |
| `pkg/layers` | accepted, output only: not ported, its output is not written |
| `pkg/mnc` | accepted, output only: not ported, its output is not written |
| `pkg/my82` | ported |
| `pkg/opps` | ported |
| `pkg/pp81` | ported |
| `pkg/ptracers` | ported |
| `pkg/rbcs` | ported |
| `pkg/sbo` | ported |
| `pkg/seaice` | ported |

Reading input through `pkg/mnc` is refused (`MNC_READ_SWITCHES` in `mitjax/drivers/model.py`): with
`useMNC`, each of these switches set to `.TRUE.` stops the set-up with `UnsupportedOption`.

| file | namelist group | switch | read at (MITgcm) |
|---|---|---|---|
| `data.mnc` | `MNC_01` | `pickup_read_mnc` | model/src/read_pickup.F:488, pkg/cd_code/cd_code_read_pickup.F:50, pkg/generic_advdiff/gad_read_pickup.F:51 |
| `data.mnc` | `MNC_01` | `readgrid_mnc` | model/src/ini_curvilinear_grid.F:209 |
| `data.mnc` | `MNC_01` | `mnc_read_bathy` | model/src/ini_depths.F:106 |
| `data.mnc` | `MNC_01` | `mnc_read_salt` | model/src/ini_salt.F:67 |
| `data.mnc` | `MNC_01` | `mnc_read_theta` | model/src/ini_theta.F:68 |
| `data.diagnostics` | `DIAGNOSTICS_LIST` | `diag_pickup_read_mnc` | pkg/diagnostics/diagnostics_read_pickup.F:58 (diagnostics_readparms.F:269: .AND. diag_mnc) |

### Experiment code files

Every `.F` in an experiment's `code*/` directory must be the upstream routine or one of these ported
experiment routines (compared byte for byte); any other stops the set-up with `UnportedRoutine` naming it
(`mitjax/config/own_code.py`, plan decision 11).

| experiment | code dir | file | ported as |
|---|---|---|---|
| `adjustment.cs-32x32x1` | `code_min` | `main.F` | `mitjax.pkg.exch2.w2_eeboot:w2_eeboot` |
| `advect_cs` | `code` | `ini_vel.F` | `mitjax.verification.advect_cs.code.ini_vel` |
| `advect_xy` | `code` | `ini_salt.F` | `mitjax.verification.advect_xy.code.ini_salt` |
| `advect_xy` | `code` | `ini_theta.F` | `mitjax.verification.advect_xy.code.ini_theta` |
| `advect_xy` | `code` | `ini_vel.F` | `mitjax.verification.advect_xy.code.ini_vel` |
| `global_ocean.cs32x15` | `code_ad` | `cost_test.F` | `mitjax.pkg.cost.cost_test` |
| `solid-body.cs-32x32x1` | `code` | `ini_psurf.F` | `mitjax.verification.solid_body_cs_32x32x1.code.ini_psurf` |
| `solid-body.cs-32x32x1` | `code` | `ini_vel.F` | `mitjax.verification.solid_body_cs_32x32x1.code.ini_vel` |
| `tutorial_global_oce_latlon` | `code` | `ptracers_apply_forcing.F` | `mitjax.verification.tutorial_global_oce_latlon.code.ptracers_apply_forcing` |
| `tutorial_global_oce_latlon` | `code` | `ptracers_forcing_surf.F` | `mitjax.verification.tutorial_global_oce_latlon.code.ptracers_forcing_surf` |
| `tutorial_global_oce_optim` | `code_ad` | `cost_hflux.F` | `mitjax.pkg.cost.cost_hflux` |
| `tutorial_global_oce_optim` | `code_ad` | `cost_temp.F` | `mitjax.pkg.cost.cost_temp` |
| `tutorial_global_oce_optim` | `code_ad` | `cost_weights.F` | `mitjax.pkg.cost.cost_weights` |
| `tutorial_tracer_adjsens` | `code_ad` | `ptracers_forcing_surf.F` | `mitjax.verification.tutorial_tracer_adjsens.code_ad.ptracers_forcing_surf` |

### CPP options with an explicit refusal

309 checks in the code stop a run with an error when the condition holds; the options are the
macros of your preprocessed `*_OPTIONS.h` (`cfg.cpp.NAME` is `#ifdef NAME`). A check sits in the routine
that reaches the unported code, so it fires when your run gets there (at set-up or when the step is
traced). An option that appears in no row is either ported or not read by any ported routine.

| option | where (mitjax module, function) | stops when |
|---|---|---|
| `ALLOW_3D_DIFFKR` | [`pkg/my82/my82_calc.py`](../mitjax/pkg/my82/my82_calc.py) `my82_calc` | `cfg.cpp.flag("ALLOW_3D_DIFFKR", "MY82_OPTIONS.h")` |
| `ALLOW_3D_DIFFKR` | [`pkg/my82/my82_calc_diff.py`](../mitjax/pkg/my82/my82_calc_diff.py) `my82_calc_diff` | `cfg.cpp.flag("ALLOW_3D_DIFFKR", "MY82_OPTIONS.h")` |
| `ALLOW_3D_DIFFKR` | [`pkg/pp81/pp81_calc.py`](../mitjax/pkg/pp81/pp81_calc.py) `pp81_calc` | `cfg.cpp.flag("ALLOW_3D_DIFFKR", "PP81_OPTIONS.h")` |
| `ALLOW_3D_DIFFKR` | [`pkg/pp81/pp81_calc_diff.py`](../mitjax/pkg/pp81/pp81_calc_diff.py) `pp81_calc_diff` | `cfg.cpp.flag("ALLOW_3D_DIFFKR", "PP81_OPTIONS.h")` |
| `ALLOW_ADAMSBASHFORTH_3`, `ALLOW_NONHYDROSTATIC` | [`model/state.py`](../mitjax/model/state.py) `fields_of` | `cfg.cpp.ALLOW_NONHYDROSTATIC and cfg.cpp.ALLOW_ADAMSBASHFORTH_3` |
| `ALLOW_ADDFLUID` | [`model/src/calc_div_ghat.py`](../mitjax/model/src/calc_div_ghat.py) `calc_div_ghat` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `ALLOW_ADDFLUID` | [`model/src/ini_forcing.py`](../mitjax/model/src/ini_forcing.py) `ini_forcing` | `cfg.cpp.ALLOW_ADDFLUID and fp.addMassFile.strip()` |
| `ALLOW_ADDFLUID` | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `ALLOW_ADDFLUID` | [`model/src/integrate_for_w.py`](../mitjax/model/src/integrate_for_w.py) `integrate_for_w` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `ALLOW_ADDFLUID` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_ADDFLUID and fp.selectAddFluid != 0` |
| `ALLOW_ADDFLUID` | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_ADDFLUID") and params.selectAddFluid != 0` |
| `ALLOW_AIM` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_AIM and _use(cfg, "useAIM")` |
| `ALLOW_AIM` | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `cfg.cpp.flag("ALLOW_AIM", _OPT)` |
| `ALLOW_AIM` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("ALLOW_AIM", _OPT)` |
| `ALLOW_AIM` | [`pkg/generic_advdiff/gad_implicit_r.py`](../mitjax/pkg/generic_advdiff/gad_implicit_r.py) `gad_implicit_r.level_k` | `cfg.cpp.flag("ALLOW_AIM", _OPT)` |
| `ALLOW_AIM` | [`pkg/monitor/mon_surfcor.py`](../mitjax/pkg/monitor/mon_surfcor.py) `mon_surfcor` | `cfg.ALLOW_AIM and cfg.useAIM` |
| `ALLOW_ATM_TEMP`, `ALLOW_BULKFORMULAE` | [`pkg/exf/exf_wind.py`](../mitjax/pkg/exf/exf_wind.py) `exf_wind` | `not (cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h"))` |
| `ALLOW_ATM_TEMP`, `ALLOW_DOWNWARD_RADIATION` | [`pkg/seaice/seaice_solve4temp.py`](../mitjax/pkg/seaice/seaice_solve4temp.py) `seaice_solve4temp` | `not (cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_DOWNWARD_RADIATION", "EXF_OPTION…` |
| `ALLOW_AUTODIFF` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `cfg.cpp.ALLOW_AUTODIFF and not params.fluidIsWater` |
| `ALLOW_AUTODIFF` | [`model/src/ini_pressure.py`](../mitjax/model/src/ini_pressure.py) `ini_pressure` | `cfg.cpp.ALLOW_AUTODIFF` |
| `ALLOW_AUTODIFF` | [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `cfg.cpp.flag("ALLOW_AUTODIFF", opt) and ggl.adMxlMaxFlag != ggl.mxlMaxFlag` |
| `ALLOW_AUTODIFF` | [`pkg/seaice/dynsolver.py`](../mitjax/pkg/seaice/dynsolver.py) `dynsolver` | `cfg.cpp.flag("ALLOW_AUTODIFF")` |
| `ALLOW_AUTODIFF` | [`pkg/seaice/seaice_advection.py`](../mitjax/pkg/seaice/seaice_advection.py) `seaice_advection` | `ppm and cfg.cpp.flag("ALLOW_AUTODIFF")` |
| `ALLOW_AUTODIFF` | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cfg.cpp.flag("ALLOW_AUTODIFF") and not cfg.use_flag("useSEAICE")` |
| `ALLOW_AUTODIFF`, `ALLOW_OFFLINE` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `cfg.cpp.ALLOW_AUTODIFF and not cfg.cpp.ALLOW_OFFLINE` |
| `ALLOW_AUTODIFF`, `ALLOW_SITRACER`, `SEAICE_BGRID_DYNAMICS` | [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `cfg.cpp.flag("ALLOW_AUTODIFF") and (cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h") or cfg.cpp.flag…` |
| `ALLOW_AUTODIFF`, `ALLOW_ZENITHANGLE` | [`pkg/exf/exf_radiation.py`](../mitjax/pkg/exf/exf_radiation.py) `exf_radiation` | `cfg.cpp.flag("ALLOW_ZENITHANGLE", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_AUTODIFF")` |
| `ALLOW_AUTODIFF`, `SEAICE_DYN_STABLE_ADJOINT` | [`pkg/seaice/seaice_calc_strainrates.py`](../mitjax/pkg/seaice/seaice_calc_strainrates.py) `seaice_calc_strainrates` | `cfg.cpp.flag("ALLOW_AUTODIFF") and cfg.cpp.flag("SEAICE_DYN_STABLE_ADJOINT", "SEAICE_OPTIONS.h")` |
| `ALLOW_AUTODIFF`, `TARGET_NEC_SX` | [`pkg/generic_advdiff/gad_u3c4_impl_r.py`](../mitjax/pkg/generic_advdiff/gad_u3c4_impl_r.py) `gad_u3c4_impl_r` | `cfg.ALLOW_AUTODIFF and cfg.TARGET_NEC_SX` |
| `ALLOW_AUTODIFF_TAMC` | [`ad/modes.py`](../mitjax/ad/modes.py) `cg2d_operator_adjoint` | `not cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", _AD_HEADER)` |
| `ALLOW_AUTODIFF_TAMC` | [`model/src/cg2d_nsa.py`](../mitjax/model/src/cg2d_nsa.py) `cg2d_nsa` | `not cfg.cpp.ALLOW_AUTODIFF_TAMC` |
| `ALLOW_AUTODIFF_TAMC` | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_fields_load` | `cfg.cpp.ALLOW_AUTODIFF_TAMC` |
| `ALLOW_BALANCE_FLUXES` | [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag("ALLOW_BALANCE_FLUXES") and (op.selectBalanceEmPmR == 1 or op.balanceQnet)` |
| `ALLOW_BALANCE_RELAX` | [`model/src/forcing_surf_relax.py`](../mitjax/model/src/forcing_surf_relax.py) `forcing_surf_relax` | `cfg.cpp.ALLOW_BALANCE_RELAX and (fp.balanceThetaClimRelax or fp.balanceSaltClimRelax)` |
| `ALLOW_BL79_LAT_VARY` | [`model/src/ini_mixing.py`](../mitjax/model/src/ini_mixing.py) `ini_mixing` | `cfg.cpp.ALLOW_BL79_LAT_VARY` |
| `ALLOW_BL79_LAT_VARY` | [`model/src/ini_parms_tracer.py`](../mitjax/model/src/ini_parms_tracer.py) `ini_parms_tracer` | `cfg.cpp.ALLOW_BL79_LAT_VARY` |
| `ALLOW_BL79_LAT_VARY`, `ALLOW_SMAG_3D_DIFFUSIVITY`, `INCLUDE_SOUNDSPEED_CALC_CODE` | [`model/state.py`](../mitjax/model/state.py) `fields_of` | `getattr(cfg.cpp, opt)` |
| `ALLOW_BLING` | [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `exf_cfg_flag(cfg, "ALLOW_BLING")` |
| `ALLOW_BOTTOMDRAG_ROUGHNESS` | [`pkg/mom_common/mom_u_botdrag_coeff.py`](../mitjax/pkg/mom_common/mom_u_botdrag_coeff.py) `mom_u_botdrag_coeff` | `cfg.cpp.flag("ALLOW_BOTTOMDRAG_ROUGHNESS", _OPT)` |
| `ALLOW_BOTTOMDRAG_ROUGHNESS` | [`pkg/mom_common/mom_v_botdrag_coeff.py`](../mitjax/pkg/mom_common/mom_v_botdrag_coeff.py) `mom_v_botdrag_coeff` | `cfg.cpp.flag("ALLOW_BOTTOMDRAG_ROUGHNESS", _OPT)` |
| `ALLOW_BULKFORMULAE` | [`pkg/exf/exf_wind.py`](../mitjax/pkg/exf/exf_wind.py) `exf_wind` | `not cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h")` |
| `ALLOW_BULK_FORCE` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_BULK_FORCE and _use(cfg, "useBulkForce")` |
| `ALLOW_CD_CODE` | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `params.implicitViscosity and params.useCDscheme and not cfg.cpp.ALLOW_CD_CODE` |
| `ALLOW_CD_CODE`, `ALLOW_MNC` | [`pkg/cd_code/cd_code_init_fixed.py`](../mitjax/pkg/cd_code/cd_code_init_fixed.py) `cd_code_init_fixed` | `cfg.cpp.ALLOW_CD_CODE and cfg.cpp.ALLOW_MNC and ip.useMNC` |
| `ALLOW_CD_CODE`, `CD_CODE_NO_AB_MOMENTUM` | [`model/src/timestep.py`](../mitjax/model/src/timestep.py) `timestep` | `cfg.cpp.ALLOW_CD_CODE and cfg.cpp.flag("CD_CODE_NO_AB_MOMENTUM", "CD_CODE_OPTIONS.h")` |
| `ALLOW_CG2D_NSA` | [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `cg2d_solve` | `cfg.cpp.ALLOW_CG2D_NSA and params.useNSACGSolver` |
| `ALLOW_CG2D_NSA` | [`model/src/cg2d_nsa.py`](../mitjax/model/src/cg2d_nsa.py) `cg2d_nsa` | `not cfg.cpp.ALLOW_CG2D_NSA` |
| `ALLOW_CHEAPAML` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_CHEAPAML and _use(cfg, "useCheapAML")` |
| `ALLOW_COST` | [`drivers/model.py`](../mitjax/drivers/model.py) `Model._ecco_init` | `not cfg.cpp.ALLOW_COST or getattr(self, "cal", None) is None` |
| `ALLOW_COST_ATLANTIC_HEAT` | [`pkg/cost/cost_atlantic_heat.py`](../mitjax/pkg/cost/cost_atlantic_heat.py) `cost_atlantic_heat` | `not cfg.cpp.flag("ALLOW_COST_ATLANTIC_HEAT", "COST_OPTIONS.h")` |
| `ALLOW_COST_ATLANTIC_HEAT_DOMASS` | [`pkg/cost/cost_atlantic_heat.py`](../mitjax/pkg/cost/cost_atlantic_heat.py) `cost_atlantic_heat` | `cfg.cpp.flag("ALLOW_COST_ATLANTIC_HEAT_DOMASS", "COST_OPTIONS.h")` |
| `ALLOW_COST_STATE_FINAL`, `ALLOW_COST_VECTOR` | [`pkg/cost/cost_init_varia.py`](../mitjax/pkg/cost/cost_init_varia.py) `cost_init_varia` | `cfg.cpp.ALLOW_COST_VECTOR or cfg.cpp.ALLOW_COST_STATE_FINAL` |
| `ALLOW_COST_TEST`, `ALLOW_COST_TSQUARED` | [`pkg/cost/cost_test.py`](../mitjax/pkg/cost/cost_test.py) `cost_test` | `not (cfg.cpp.flag("ALLOW_COST_TEST", "COST_OPTIONS.h") and cfg.cpp.flag("ALLOW_COST_TSQUARED", "COST_OPTIONS.…` |
| `ALLOW_CTRL`, `ALLOW_GENTIM2D_CONTROL` | [`pkg/exf/exf_set_fld.py`](../mitjax/pkg/exf/exf_set_fld.py) `exf_set_fld` | `pre is None and cfg.cpp.flag("ALLOW_CTRL") and cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h") and f…` |
| `ALLOW_DEBUG` | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_DEBUG", _OPT) and p.debugLevel >= debLevC and k == 4 \ and p.useCubedSphereExchange` |
| `ALLOW_DEPTH_CONTROL` | [`model/src/calc_adv_flow.py`](../mitjax/model/src/calc_adv_flow.py) `calc_adv_flow` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`model/src/grad_sigma.py`](../mitjax/model/src/grad_sigma.py) `grad_sigma` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`model/src/initialise_varia.py`](../mitjax/model/src/initialise_varia.py) `initialise_varia` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`model/src/integrate_for_w.py`](../mitjax/model/src/integrate_for_w.py) `integrate_for_w` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.ALLOW_DEPTH_CONTROL` |
| `ALLOW_DEPTH_CONTROL` | [`pkg/mom_common/mom_calc_hfacz.py`](../mitjax/pkg/mom_common/mom_calc_hfacz.py) `mom_calc_hfacz` | `cfg.cpp.flag("ALLOW_DEPTH_CONTROL", _OPT)` |
| `ALLOW_DIAGNOSTICS` | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/generic_advdiff/gad_implicit_r.py`](../mitjax/pkg/generic_advdiff/gad_implicit_r.py) `gad_implicit_r` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/generic_advdiff/gad_som_advect.py`](../mitjax/pkg/generic_advdiff/gad_som_advect.py) `gad_som_advect` | `cfg.ALLOW_DIAGNOSTICS and cfg.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `mom_calc_visc` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/mom_common/mom_init_fixed.py`](../mitjax/pkg/mom_common/mom_init_fixed.py) `mom_init_fixed` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/mom_common/mom_u_sidedrag.py`](../mitjax/pkg/mom_common/mom_u_sidedrag.py) `mom_u_sidedrag` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics` |
| `ALLOW_DIAGNOSTICS` | [`pkg/mom_common/mom_v_sidedrag.py`](../mitjax/pkg/mom_common/mom_v_sidedrag.py) `mom_v_sidedrag` | `cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics` |
| `ALLOW_DIAGNOSTICS`, `ALLOW_MOM_TEND_EXTRA_DIAGS` | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_MOM_TEND_EXTRA_DIAGS", _COPT) and cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useD…` |
| `ALLOW_DOWN_SLOPE` | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_DOWN_SLOPE", _OPT) and ptr.PTRACERS_useDWNSLP[n]` |
| `ALLOW_ECCO` | [`pkg/cost/cost_final.py`](../mitjax/pkg/cost/cost_final.py) `cost_final` | `cfg.cpp.ALLOW_ECCO and pkg_final is None` |
| `ALLOW_EDDYPSI` | [`pkg/gmredi/gmredi_do_exch.py`](../mitjax/pkg/gmredi/gmredi_do_exch.py) `gmredi_do_exch` | `cfg.cpp.ALLOW_EDDYPSI` |
| `ALLOW_EDDYPSI` | [`pkg/gmredi/gmredi_residual_flow.py`](../mitjax/pkg/gmredi/gmredi_residual_flow.py) `gmredi_residual_flow` | `cfg.cpp.ALLOW_EDDYPSI` |
| `ALLOW_EDDYPSI`, `ALLOW_FRICTION_HEATING`, `ALLOW_GEOTHERMAL_FLUX`, `EXCLUDE_FFIELDS_LOAD` | [`model/src/ini_ffields.py`](../mitjax/model/src/ini_ffields.py) `fields_of` | `getattr(cfg.cpp, opt)` |
| `ALLOW_EDDYPSI`, `ALLOW_GMREDI` | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_EDDYPSI") and _cpp(cfg, "ALLOW_GMREDI")` |
| `ALLOW_EXCH2` | [`model/src/ini_curvilinear_grid.py`](../mitjax/model/src/ini_curvilinear_grid.py) `ini_curvilinear_grid` | `not cfg.cpp.ALLOW_EXCH2` |
| `ALLOW_EXCH2` | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `not cfg.cpp.ALLOW_EXCH2` |
| `ALLOW_EXCH2` | [`pkg/generic_advdiff/gad_som_advect.py`](../mitjax/pkg/generic_advdiff/gad_som_advect.py) `gad_som_advect` | `not cfg.ALLOW_EXCH2` |
| `ALLOW_EXCH2` | [`pkg/mom_common/mom_calc_relvort3.py`](../mitjax/pkg/mom_common/mom_calc_relvort3.py) `mom_calc_relvort3` | `not cfg.cpp.flag("ALLOW_EXCH2", _OPT)` |
| `ALLOW_EXCH2` | [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `_fill` | `not cfg.cpp.flag("ALLOW_EXCH2", _OPT)` |
| `ALLOW_EXCH2` | [`pkg/mom_vecinv/mom_vi_del2uv.py`](../mitjax/pkg/mom_vecinv/mom_vi_del2uv.py) `_fill` | `not cfg.cpp.flag("ALLOW_EXCH2", _OPT)` |
| `ALLOW_EXCH2` | [`pkg/seaice/seaice_advection.py`](../mitjax/pkg/seaice/seaice_advection.py) `seaice_advection` | `useCubedSphereExchange and not cfg.cpp.ALLOW_EXCH2` |
| `ALLOW_EXCH2`, `W2_FILL_NULL_REGIONS` | [`model/src/calc_r_star.py`](../mitjax/model/src/calc_r_star.py) `calc_r_star` | `cfg.cpp.ALLOW_EXCH2 and cfg.cpp.flag("W2_FILL_NULL_REGIONS", "W2_OPTIONS.h")` |
| `ALLOW_EXF` | [`drivers/model.py`](../mitjax/drivers/model.py) `Model._ctrl_init` | `cfg.cpp.flag("ALLOW_EXF") and fstr_prefix(g.xx_gentim2d_file, n, lit)` |
| `ALLOW_EXF`, `SEAICE_EXTERNAL_FLUXES` | [`pkg/seaice/seaice_get_dynforcing.py`](../mitjax/pkg/seaice/seaice_get_dynforcing.py) `seaice_get_dynforcing` | `not cfg.cpp.flag("ALLOW_EXF") or not cfg.cpp.flag("SEAICE_EXTERNAL_FLUXES", "SEAICE_OPTIONS.h")` |
| `ALLOW_FRICTION_HEATING` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_FRICTION_HEATING` |
| `ALLOW_FRICTION_HEATING` | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `cfg.cpp.ALLOW_FRICTION_HEATING` |
| `ALLOW_GCHEM` | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_GCHEM and _use(cfg, "useGCHEM")` |
| `ALLOW_GCHEM` | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `check_unported` | `cfg.cpp.flag("ALLOW_GCHEM", "PTRACERS_OPTIONS.h") and cfg.use_flag("useGCHEM")` |
| `ALLOW_GENERIC_ADVDIFF` | [`pkg/seaice/seaice_advdiff.py`](../mitjax/pkg/seaice/seaice_advdiff.py) `seaice_advdiff` | `not (sp.SEAICEmultiDimAdvection and cfg.cpp.flag("ALLOW_GENERIC_ADVDIFF"))` |
| `ALLOW_GENTIM2D_CONTROL` | [`pkg/exf/exf_getsurfacefluxes.py`](../mitjax/pkg/exf/exf_getsurfacefluxes.py) `exf_getsurfacefluxes` | `not cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h")` |
| `ALLOW_GEOTHERMAL_FLUX` | [`pkg/ctrl/ctrl_map_ini_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_ini_genarr.py) `ctrl_map_ini_genarr` | `cpp.flag("ALLOW_GEOTHERMAL_FLUX") and igen_geoth > 0` |
| `ALLOW_GGL90`, `ALLOW_GGL90_LANGMUIR` | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_GGL90", _COPT) and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") \ and p.useLAN…` |
| `ALLOW_GGL90_HORIZDIFF` | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("ALLOW_GGL90_HORIZDIFF", opt) and ggl.static_float("GGL90diffTKEh") > 0.` |
| `ALLOW_GGL90_LANGMUIR` | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `ggl is not None and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") and ggl.useLANGMUIR \ and not par…` |
| `ALLOW_GGL90_SMOOTH` | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("ALLOW_GGL90_SMOOTH", opt)` |
| `ALLOW_GGL90_SMOOTH` | [`pkg/ggl90/ggl90_init_fixed.py`](../mitjax/pkg/ggl90/ggl90_init_fixed.py) `ggl90_init_fixed` | `cfg.cpp.flag("ALLOW_GGL90_SMOOTH", "GGL90_OPTIONS.h")` |
| `ALLOW_GMREDI` | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `cfg.cpp.ALLOW_GMREDI and params.useGMRedi and gm is None` |
| `ALLOW_KL10` | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `cfg.cpp.ALLOW_KL10 and _use(cfg, "useKL10")` |
| `ALLOW_KL10` | [`model/src/calc_viscosity.py`](../mitjax/model/src/calc_viscosity.py) `calc_viscosity` | `cfg.cpp.flag("ALLOW_KL10") and cfg.use_flag("useKL10")` |
| `ALLOW_LONGSTEP` | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `cfg.cpp.ALLOW_LONGSTEP and trIdentity >= 3` |
| `ALLOW_LONGSTEP` | [`pkg/gmredi/gmredi_calc_diff.py`](../mitjax/pkg/gmredi/gmredi_calc_diff.py) `gmredi_calc_diff` | `cfg.cpp.ALLOW_LONGSTEP and tracerIdentity >= GAD_TR1` |
| `ALLOW_LONGSTEP` | [`pkg/gmredi/gmredi_rtransport.py`](../mitjax/pkg/gmredi/gmredi_rtransport.py) `gmredi_rtransport` | `cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1` |
| `ALLOW_LONGSTEP` | [`pkg/gmredi/gmredi_xtransport.py`](../mitjax/pkg/gmredi/gmredi_xtransport.py) `gmredi_xtransport` | `cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1` |
| `ALLOW_LONGSTEP` | [`pkg/gmredi/gmredi_ytransport.py`](../mitjax/pkg/gmredi/gmredi_ytransport.py) `gmredi_ytransport` | `cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1` |
| `ALLOW_LONGSTEP` | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_LONGSTEP", _OPT)` |
| `ALLOW_LONGSTEP` | [`pkg/ptracers/ptracers_readparms.py`](../mitjax/pkg/ptracers/ptracers_readparms.py) `ptracers_readparms` | `cfg.cpp.flag("ALLOW_LONGSTEP", _OPT)` |
| `ALLOW_MATRIX` | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_MATRIX", _OPT)` |
| `ALLOW_MNC` | [`drivers/adjoint_run.py`](../mitjax/drivers/adjoint_run.py) `ad_exf_records` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `ALLOW_MNC` | [`model/src/ini_depths.py`](../mitjax/model/src/ini_depths.py) `ini_depths` | `cfg.cpp.ALLOW_MNC and params.useMNC and params.mnc_read_bathy` |
| `ALLOW_MNC` | [`model/src/ini_salt.py`](../mitjax/model/src/ini_salt.py) `ini_salt` | `cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_salt` |
| `ALLOW_MNC` | [`model/src/ini_theta.py`](../mitjax/model/src/ini_theta.py) `ini_theta` | `cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_theta` |
| `ALLOW_MNC` | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_MNC") and ip.useMNC and io.pickup_write_mnc` |
| `ALLOW_MNC` | [`pkg/autodiff/adjoint_monitor.py`](../mitjax/pkg/autodiff/adjoint_monitor.py) `admonitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `ALLOW_MNC` | [`pkg/cd_code/cd_code_write_pickup.py`](../mitjax/pkg/cd_code/cd_code_write_pickup.py) `cd_code_write_pickup` | `cfg.cpp.ALLOW_MNC and dict(cfg.use).get("useMNC", False) and getattr(io, "pickup_write_mnc", False)` |
| `ALLOW_MNC` | [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `ALLOW_MNC` | [`pkg/generic_advdiff/gad_write_pickup.py`](../mitjax/pkg/generic_advdiff/gad_write_pickup.py) `gad_write_pickup` | `cfg.cpp.flag("ALLOW_MNC", _OPT) and dict(cfg.use).get("useMNC", False)` |
| `ALLOW_MNC` | [`pkg/monitor/mon_out.py`](../mitjax/pkg/monitor/mon_out.py) `mon_out_all` | `cfg.ALLOW_MNC and cfg.useMNC and mon.mon_write_mnc` |
| `ALLOW_MNC` | [`pkg/monitor/monitor.py`](../mitjax/pkg/monitor/monitor.py) `monitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `ALLOW_MNC` | [`pkg/ptracers/ptracers_monitor.py`](../mitjax/pkg/ptracers/ptracers_monitor.py) `ptracers_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc` |
| `ALLOW_MNC` | [`pkg/ptracers/ptracers_monitor_ad.py`](../mitjax/pkg/ptracers/ptracers_monitor_ad.py) `adptracers_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc` |
| `ALLOW_MNC` | [`pkg/seaice/seaice_monitor.py`](../mitjax/pkg/seaice/seaice_monitor.py) `seaice_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and mon_mnc` |
| `ALLOW_NONDIMENSIONAL_CONTROL_IO` | [`pkg/cost/cost_weights.py`](../mitjax/pkg/cost/cost_weights.py) `cost_weights` | `cfg.cpp.ALLOW_NONDIMENSIONAL_CONTROL_IO` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/calc_div_ghat.py`](../mitjax/model/src/calc_div_ghat.py) `calc_div_ghat` | `cfg.cpp.ALLOW_NONHYDROSTATIC and params.use3Dsolver` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `ini_fields` | `cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/pressure_for_eos.py`](../mitjax/model/src/pressure_for_eos.py) `pressure_for_eos` | `cfg.cpp.ALLOW_NONHYDROSTATIC and sel == 3` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/solve_for_pressure.py`](../mitjax/model/src/solve_for_pressure.py) `solve_for_pressure` | `cfg.cpp.ALLOW_NONHYDROSTATIC and params.use3Dsolver` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_NONHYDROSTATIC") and params.use3Dsolver` |
| `ALLOW_NONHYDROSTATIC` | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_NONHYDROSTATIC") and params.selectNHfreeSurf >= 1` |
| `ALLOW_NONHYDROSTATIC` | [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `mom_calc_visc` | `cfg.cpp.flag("ALLOW_NONHYDROSTATIC", _OPT) and p.nonHydrostatic` |
| `ALLOW_NONHYDROSTATIC` | [`pkg/monitor/mon_ke.py`](../mitjax/pkg/monitor/mon_ke.py) `mon_ke` | `cfg.ALLOW_NONHYDROSTATIC and cfg.nonHydrostatic` |
| `ALLOW_NONHYDROSTATIC` | [`pkg/opps/opps_calc.py`](../mitjax/pkg/opps/opps_calc.py) `state1_pressure` | `cfg.cpp.ALLOW_NONHYDROSTATIC and sel == 3` |
| `ALLOW_OBCS` | [`model/src/calc_r_star.py`](../mitjax/model/src/calc_r_star.py) `calc_r_star` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`model/src/calc_surf_dr.py`](../mitjax/model/src/calc_surf_dr.py) `calc_surf_dr` | `cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS")` |
| `ALLOW_OBCS` | [`model/src/ini_cg2d.py`](../mitjax/model/src/ini_cg2d.py) `ini_cg2d` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`model/src/ini_depths.py`](../mitjax/model/src/ini_depths.py) `ini_depths` | `cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS")` |
| `ALLOW_OBCS` | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`model/src/solve_for_pressure.py`](../mitjax/model/src/solve_for_pressure.py) `solve_for_pressure` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `cfg.cpp.ALLOW_OBCS and params.useOBCS` |
| `ALLOW_OBCS` | [`model/src/update_cg2d.py`](../mitjax/model/src/update_cg2d.py) `update_cg2d` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`model/src/update_etah.py`](../mitjax/model/src/update_etah.py) `update_etah` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`pkg/cd_code/cd_code_scheme.py`](../mitjax/pkg/cd_code/cd_code_scheme.py) `cd_code_scheme` | `cfg.cpp.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`pkg/exf/exf_init_fixed.py`](../mitjax/pkg/exf/exf_init_fixed.py) `exf_init_fixed` | `cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS")` |
| `ALLOW_OBCS` | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS` |
| `ALLOW_OBCS` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS` |
| `ALLOW_OBCS` | [`pkg/generic_advdiff/gad_som_adv_r.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_r.py) `gad_som_adv_r` | `cfg.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`pkg/generic_advdiff/gad_som_adv_x.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_x.py) `gad_som_adv_x` | `cfg.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`pkg/generic_advdiff/gad_som_adv_y.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_y.py) `gad_som_adv_y` | `cfg.ALLOW_OBCS` |
| `ALLOW_OBCS` | [`pkg/gmredi/gmredi_calc_qgleith.py`](../mitjax/pkg/gmredi/gmredi_calc_qgleith.py) `gmredi_calc_qgleith` | `cfg.cpp.flag("ALLOW_OBCS", _OPT)` |
| `ALLOW_OBCS` | [`pkg/mom_common/mom_calc_hdiv.py`](../mitjax/pkg/mom_common/mom_calc_hdiv.py) `mom_calc_hdiv` | `cfg.cpp.flag("ALLOW_OBCS", _OPT)` |
| `ALLOW_OBCS` | [`pkg/mom_common/mom_calc_tension.py`](../mitjax/pkg/mom_common/mom_calc_tension.py) `mom_calc_tension` | `cfg.cpp.flag("ALLOW_OBCS", _OPT)` |
| `ALLOW_OBCS` | [`pkg/mom_fluxform/mom_u_del2u.py`](../mitjax/pkg/mom_fluxform/mom_u_del2u.py) `mom_u_del2u` | `cfg.cpp.flag("ALLOW_OBCS", opt)` |
| `ALLOW_OBCS` | [`pkg/mom_fluxform/mom_v_del2v.py`](../mitjax/pkg/mom_fluxform/mom_v_del2v.py) `mom_v_del2v` | `cfg.cpp.flag("ALLOW_OBCS", opt)` |
| `ALLOW_OBCS` | [`pkg/mom_vecinv/mom_vi_del2uv.py`](../mitjax/pkg/mom_vecinv/mom_vi_del2uv.py) `mom_vi_del2uv` | `cfg.cpp.flag("ALLOW_OBCS", _OPT)` |
| `ALLOW_OBCS` | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and cfg.use_flag("useOBCS")` |
| `ALLOW_OBCS` | [`pkg/seaice/seaice_advection.py`](../mitjax/pkg/seaice/seaice_advection.py) `seaice_advection` | `cfg.cpp.flag("ALLOW_OBCS")` |
| `ALLOW_OBCS` | [`pkg/seaice/seaice_calc_strainrates.py`](../mitjax/pkg/seaice/seaice_calc_strainrates.py) `seaice_calc_strainrates` | `cfg.cpp.flag("ALLOW_OBCS")` |
| `ALLOW_OBCS` | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS")` |
| `ALLOW_OBCS` | [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `seaice_lsr` | `cfg.cpp.flag("ALLOW_OBCS", "SEAICE_OPTIONS.h")` |
| `ALLOW_OBCS` | [`pkg/seaice/seaice_oceandrag_coeffs.py`](../mitjax/pkg/seaice/seaice_oceandrag_coeffs.py) `seaice_oceandrag_coeffs` | `cfg.cpp.flag("ALLOW_OBCS")` |
| `ALLOW_OBCS`, `ALLOW_THSICE` | [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `(cfg.cpp.flag("ALLOW_THSICE") and cfg.use_flag("useThSIce")) \ or (cfg.cpp.flag("ALLOW_OBCS") and cfg.use_fla…` |
| `ALLOW_OFFLINE` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_autodiff_resets` | `cfg.cpp.ALLOW_OFFLINE` |
| `ALLOW_OFFLINE` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `cfg.cpp.ALLOW_OFFLINE and params.useOffLine` |
| `ALLOW_OFFLINE` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `calcKPP and cfg.cpp.ALLOW_OFFLINE` |
| `ALLOW_OPENAD` | [`pkg/cost/cost_hflux.py`](../mitjax/pkg/cost/cost_hflux.py) `cost_hflux` | `cfg.cpp.ALLOW_OPENAD` |
| `ALLOW_OPENAD` | [`pkg/cost/cost_temp.py`](../mitjax/pkg/cost/cost_temp.py) `cost_temp` | `cfg.cpp.ALLOW_OPENAD` |
| `ALLOW_OPENAD`, `ALLOW_SMOOTH` | [`pkg/ctrl/ctrl_map_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_genarr.py) `ctrl_map_genarr2d` | `cfg.cpp.flag("ALLOW_OPENAD", "CTRL_OPTIONS.h") or cfg.cpp.flag("ALLOW_SMOOTH", "CTRL_OPTIONS.h")` |
| `ALLOW_OPENAD`, `ALLOW_SMOOTH` | [`pkg/ctrl/ctrl_map_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_genarr.py) `ctrl_map_genarr3d` | `cfg.cpp.flag("ALLOW_OPENAD", "CTRL_OPTIONS.h") or cfg.cpp.flag("ALLOW_SMOOTH", "CTRL_OPTIONS.h")` |
| `ALLOW_OPENAD`, `ALLOW_SMOOTH` | [`pkg/ctrl/ctrl_map_ini_gentim2d.py`](../mitjax/pkg/ctrl/ctrl_map_ini_gentim2d.py) `ctrl_map_ini_gentim2d` | `cfg.cpp.ALLOW_OPENAD or cfg.cpp.ALLOW_SMOOTH` |
| `ALLOW_OPPS_SNAPSHOT` | [`pkg/opps/opps_interface.py`](../mitjax/pkg/opps/opps_interface.py) `opps_interface` | `cfg.cpp.flag("ALLOW_OPPS_SNAPSHOT", "OPPS_OPTIONS.h")` |
| `ALLOW_PP81_LOWERBOUND` | [`pkg/pp81/pp81_calc.py`](../mitjax/pkg/pp81/pp81_calc.py) `pp81_calc` | `cfg.cpp.flag("ALLOW_PP81_LOWERBOUND", "PP81_OPTIONS.h")` |
| `ALLOW_PSBAR_STERIC` | [`pkg/ecco/ecco_write_pickup.py`](../mitjax/pkg/ecco/ecco_write_pickup.py) `ecco_write_pickup` | `cfg.cpp.flag("ALLOW_PSBAR_STERIC", "ECCO_OPTIONS.h")` |
| `ALLOW_PTRACERS` | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `cfg.cpp.ALLOW_PTRACERS and trIdentity >= GAD_TR1` |
| `ALLOW_PTRACERS` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("ALLOW_PTRACERS", _OPT) and trIdentity >= GAD_TR1` |
| `ALLOW_PTRACERS` | [`pkg/opps/opps_interface.py`](../mitjax/pkg/opps/opps_interface.py) `opps_interface` | `cfg.cpp.ALLOW_PTRACERS` |
| `ALLOW_PTRACERS` | [`pkg/rbcs/rbcs_add_tendency.py`](../mitjax/pkg/rbcs/rbcs_add_tendency.py) `rbcs_add_tendency` | `cfg.cpp.ALLOW_PTRACERS and tracerNum > 2` |
| `ALLOW_PTRACERS` | [`pkg/rbcs/rbcs_init_varia.py`](../mitjax/pkg/rbcs/rbcs_init_varia.py) `rbcs_init_varia` | `cfg.cpp.ALLOW_PTRACERS` |
| `ALLOW_PTRACERS` | [`pkg/rbcs/rbcs_readparms.py`](../mitjax/pkg/rbcs/rbcs_readparms.py) `rbcs_readparms` | `cfg.cpp.ALLOW_PTRACERS` |
| `ALLOW_QHYD_STAGGER_TS` | [`model/src/ini_nh_vars.py`](../mitjax/model/src/ini_nh_vars.py) `ini_nh_vars` | `cfg.cpp.ALLOW_QHYD_STAGGER_TS` |
| `ALLOW_QHYD_STAGGER_TS` | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep` |
| `ALLOW_QHYD_STAGGER_TS` | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep` |
| `ALLOW_QHYD_STAGGER_TS` | [`pkg/mom_common/mom_quasihydrostatic.py`](../mitjax/pkg/mom_common/mom_quasihydrostatic.py) `mom_quasihydrostatic` | `cfg.cpp.flag("ALLOW_QHYD_STAGGER_TS", _OPT) and params.staggerTimeStep` |
| `ALLOW_RBCS` | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `check_unported` | `cfg.cpp.flag("ALLOW_RBCS", "PTRACERS_OPTIONS.h") and cfg.use_flag("useRBCS")` |
| `ALLOW_ROTATE_UV_CONTROLS` | [`pkg/ctrl/ctrl_get_mask.py`](../mitjax/pkg/ctrl/ctrl_get_mask.py) `ctrl_get_mask2d` | `cfg.cpp.ALLOW_ROTATE_UV_CONTROLS` |
| `ALLOW_ROTATE_UV_CONTROLS` | [`pkg/exf/exf_getffields.py`](../mitjax/pkg/exf/exf_getffields.py) `exf_getffields` | `use_ctrl and cfg.cpp.flag("ALLOW_ROTATE_UV_CONTROLS", "CTRL_OPTIONS.h")` |
| `ALLOW_ROTATE_UV_CONTROLS` | [`pkg/exf/exf_getsurfacefluxes.py`](../mitjax/pkg/exf/exf_getsurfacefluxes.py) `exf_getsurfacefluxes` | `cfg.cpp.flag("ALLOW_ROTATE_UV_CONTROLS", "CTRL_OPTIONS.h")` |
| `ALLOW_SALT_PLUME` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `cfg.cpp.ALLOW_SALT_PLUME and params.useSALT_PLUME` |
| `ALLOW_SALT_PLUME` | [`model/src/external_forcing_surf.py`](../mitjax/model/src/external_forcing_surf.py) `external_forcing_surf` | `cfg.cpp.ALLOW_SALT_PLUME and _use(cfg, "useSALT_PLUME")` |
| `ALLOW_SALT_PLUME` | [`pkg/kpp/kpp_calc_dummy.py`](../mitjax/pkg/kpp/kpp_calc_dummy.py) `kpp_calc_dummy` | `cfg.cpp.flag("ALLOW_SALT_PLUME")` |
| `ALLOW_SALT_PLUME`, `SALT_PLUME_VOLUME` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_autodiff_resets` | `cfg.cpp.ALLOW_SALT_PLUME and cfg.cpp.flag("SALT_PLUME_VOLUME")` |
| `ALLOW_SEAICE` | [`pkg/sbo/sbo_calc.py`](../mitjax/pkg/sbo/sbo_calc.py) `sbo_calc` | `cfg.cpp.ALLOW_SEAICE` |
| `ALLOW_SEAICE_COST_EXPORT` | [`pkg/seaice/seaice_cost_final.py`](../mitjax/pkg/seaice/seaice_cost_final.py) `seaice_cost_final` | `cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h")` |
| `ALLOW_SEAICE_COST_EXPORT` | [`pkg/seaice/seaice_cost_init_varia.py`](../mitjax/pkg/seaice/seaice_cost_init_varia.py) `seaice_cost_init_varia` | `cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h")` |
| `ALLOW_SEAICE_COST_EXPORT` | [`pkg/seaice/seaice_cost_sensi.py`](../mitjax/pkg/seaice/seaice_cost_sensi.py) `seaice_cost_sensi` | `cfg.cpp.flag("ALLOW_SEAICE_COST_EXPORT", "SEAICE_OPTIONS.h")` |
| `ALLOW_SHELFICE` | [`model/src/calc_phi_hyd.py`](../mitjax/model/src/calc_phi_hyd.py) `calc_phi_hyd` | `cfg.cpp.ALLOW_SHELFICE` |
| `ALLOW_SHELFICE` | [`model/src/external_forcing_surf.py`](../mitjax/model/src/external_forcing_surf.py) `external_forcing_surf` | `cfg.cpp.ALLOW_SHELFICE and _use(cfg, "useSHELFICE")` |
| `ALLOW_SHELFICE` | [`model/src/ini_masks_etc.py`](../mitjax/model/src/ini_masks_etc.py) `ini_masks_etc` | `cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce")` |
| `ALLOW_SHELFICE` | [`model/src/ini_psurf.py`](../mitjax/model/src/ini_psurf.py) `ini_psurf` | `cfg.cpp.ALLOW_SHELFICE` |
| `ALLOW_SHELFICE` | [`pkg/ctrl/ctrl_get_mask.py`](../mitjax/pkg/ctrl/ctrl_get_mask.py) `ctrl_get_mask2d` | `cfg.cpp.ALLOW_SHELFICE and any(fstr_prefix(xx_filename, 11, n) for n in ("xx_shicoeff", "xx_shicdrag", "xx_sh…` |
| `ALLOW_SHELFICE` | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("ALLOW_SHELFICE", opt) and cfg.use_flag("useShelfIce")` |
| `ALLOW_SHELFICE` | [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce")` |
| `ALLOW_SHELFICE` | [`pkg/kpp/kpp_routines.py`](../mitjax/pkg/kpp/kpp_routines.py) `kppmix` | `cfg.cpp.flag("ALLOW_SHELFICE", "KPP_OPTIONS.h")` |
| `ALLOW_SHELFICE` | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_SHELFICE", _OPT) and p.useShelfIce and p.selectImplicitDrag == 0` |
| `ALLOW_SHELFICE` | [`pkg/seaice/seaice_init_fixed.py`](../mitjax/pkg/seaice/seaice_init_fixed.py) `seaice_init_fixed` | `cfg.cpp.flag("ALLOW_SHELFICE") and cfg.use_flag("useShelfIce")` |
| `ALLOW_SHELFICE` | [`verification/solid_body_cs_32x32x1/code/ini_psurf.py`](../mitjax/verification/solid_body_cs_32x32x1/code/ini_psurf.py) `ini_psurf` | `cfg.cpp.ALLOW_SHELFICE` |
| `ALLOW_SITRACER_DEBUG_DIAG` | [`pkg/seaice/seaice_tracer_phys.py`](../mitjax/pkg/seaice/seaice_tracer_phys.py) `seaice_tracer_phys` | `cfg.cpp.flag("ALLOW_SITRACER_DEBUG_DIAG", "SEAICE_OPTIONS.h")` |
| `ALLOW_SMAG_3D_DIFFUSIVITY` | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `cfg.cpp.ALLOW_SMAG_3D_DIFFUSIVITY` |
| `ALLOW_SMAG_3D_DIFFUSIVITY` | [`pkg/generic_advdiff/gad_diff_x.py`](../mitjax/pkg/generic_advdiff/gad_diff_x.py) `gad_diff_x` | `cfg.ALLOW_SMAG_3D_DIFFUSIVITY` |
| `ALLOW_SMAG_3D_DIFFUSIVITY` | [`pkg/generic_advdiff/gad_diff_y.py`](../mitjax/pkg/generic_advdiff/gad_diff_y.py) `gad_diff_y` | `cfg.ALLOW_SMAG_3D_DIFFUSIVITY` |
| `ALLOW_SMOOTH`, `CTRL_SKIP_FIRST_TWO_ATM_REC_ALL` | [`pkg/ctrl/ctrl_get_gen.py`](../mitjax/pkg/ctrl/ctrl_get_gen.py) `ctrl_get_gen` | `cfg.cpp.ALLOW_SMOOTH or cfg.cpp.CTRL_SKIP_FIRST_TWO_ATM_REC_ALL` |
| `ALLOW_SOLVE4_PS_AND_DRAG` | [`model/src/update_cg2d.py`](../mitjax/model/src/update_cg2d.py) `update_cg2d` | `cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG and params.selectImplicitDrag == 2` |
| `ALLOW_SOLVE4_PS_AND_DRAG` | [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `cfg.cpp.flag("ALLOW_SOLVE4_PS_AND_DRAG") and p.selectImplicitDrag == 2` |
| `ALLOW_STEEP_ICECAVITY` | [`model/src/ini_masks_etc.py`](../mitjax/model/src/ini_masks_etc.py) `ini_masks_etc` | `cfg.cpp.ALLOW_STEEP_ICECAVITY` |
| `ALLOW_STREAMICE` | [`pkg/ctrl/ctrl_map_forcing.py`](../mitjax/pkg/ctrl/ctrl_map_forcing.py) `ctrl_map_forcing` | `cfg.cpp.ALLOW_STREAMICE` |
| `ALLOW_UVEL0_CONTROL`, `ALLOW_VVEL0_CONTROL` | [`pkg/ctrl/ctrl_get_mask.py`](../mitjax/pkg/ctrl/ctrl_get_mask.py) `ctrl_get_mask3d` | `cfg.cpp.flag("ALLOW_UVEL0_CONTROL", "CTRL_OPTIONS.h") and cfg.cpp.flag("ALLOW_VVEL0_CONTROL", "CTRL_OPTIONS.h…` |
| `ALLOW_ZENITHANGLE` | [`pkg/exf/exf_radiation.py`](../mitjax/pkg/exf/exf_radiation.py) `exf_radiation` | `cfg.cpp.flag("ALLOW_ZENITHANGLE", "EXF_OPTIONS.h") and (exf.useExfZenAlbedo or exf.useExfZenIncoming)` |
| `ATMOSPHERIC_LOADING` | [`pkg/seaice/dynsolver.py`](../mitjax/pkg/seaice/dynsolver.py) `dynsolver` | `not cfg.cpp.flag("ATMOSPHERIC_LOADING")` |
| `ATMOSPHERIC_LOADING` | [`pkg/seaice/seaice_dynsolver.py`](../mitjax/pkg/seaice/seaice_dynsolver.py) `_dynamics_forcing` | `not cfg.cpp.flag("ATMOSPHERIC_LOADING")` |
| `CD_CODE_NO_AB_CORIOLIS` | [`pkg/cd_code/cd_code_scheme.py`](../mitjax/pkg/cd_code/cd_code_scheme.py) `cd_code_scheme` | `cfg.cpp.flag("CD_CODE_NO_AB_CORIOLIS", "CD_CODE_OPTIONS.h")` |
| `CHECK_SALINITY_FOR_NEGATIVE_VALUES` | [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `find_rho_2d` | `cfg.cpp.CHECK_SALINITY_FOR_NEGATIVE_VALUES` |
| `COSINEMETH_III`, `ISOTROPIC_COS_SCALING` | [`pkg/mom_fluxform/mom_u_del2u.py`](../mitjax/pkg/mom_fluxform/mom_u_del2u.py) `mom_u_del2u` | `cfg.cpp.flag("ISOTROPIC_COS_SCALING", opt) and cfg.cpp.flag("COSINEMETH_III", opt)` |
| `COSINEMETH_III`, `ISOTROPIC_COS_SCALING` | [`pkg/mom_fluxform/mom_v_del2v.py`](../mitjax/pkg/mom_fluxform/mom_v_del2v.py) `mom_v_del2v` | `cfg.cpp.flag("ISOTROPIC_COS_SCALING", opt) and cfg.cpp.flag("COSINEMETH_III", opt)` |
| `DISABLE_RBCS_MOM` | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_fields_load` | `not cfg.cpp.DISABLE_RBCS_MOM and (params.useRBCuVel or params.useRBCvVel)` |
| `DISABLE_RSTAR_CODE` | [`model/src/update_r_star.py`](../mitjax/model/src/update_r_star.py) `update_r_star` | `cfg.cpp.DISABLE_RSTAR_CODE` |
| `DISABLE_SIGMA_CODE` | [`model/src/forcing_surf_relax.py`](../mitjax/model/src/forcing_surf_relax.py) `forcing_surf_relax` | `not cfg.cpp.DISABLE_SIGMA_CODE` |
| `DISABLE_SIGMA_CODE` | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `nlfs and not (params.select_rStar > 0) and params.selectSigmaCoord_ne_0 and not cfg.cpp.DISABLE_SIGMA_CODE` |
| `DISABLE_SIGMA_CODE`, `NONLIN_FRSURF` | [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `ini_fields` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and ip.selectSigmaCoord != 0` |
| `DISABLE_SIGMA_CODE`, `NONLIN_FRSURF` | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.nonlinFreeSurf > 0 \ and params.selectSig…` |
| `DISABLE_SIGMA_CODE`, `NONLIN_FRSURF` | [`model/src/integrate_for_w.py`](../mitjax/model/src/integrate_for_w.py) `integrate_for_w` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.selectSigmaCoord != 0` |
| `EXCLUDE_FFIELDS_LOAD` | [`model/src/external_fields_load.py`](../mitjax/model/src/external_fields_load.py) `external_fields_load` | `cfg.cpp.EXCLUDE_FFIELDS_LOAD` |
| `EXCLUDE_KPP_DOUBLEDIFF` | [`pkg/kpp/kpp_calc.py`](../mitjax/pkg/kpp/kpp_calc.py) `kpp_calc` | `cfg.cpp.flag("EXCLUDE_KPP_DOUBLEDIFF", "KPP_OPTIONS.h")` |
| `EXF_READ_EVAP` | [`pkg/exf/exf_getffields.py`](../mitjax/pkg/exf/exf_getffields.py) `_gentim2d_controls` | `opt("EXF_READ_EVAP")` |
| `EXF_SEAICE_FRACTION` | [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `exf_cfg_flag(cfg, "EXF_SEAICE_FRACTION") and exf.areamaskfile.strip()` |
| `GAD_ALLOW_TS_SOM_ADV`, `PTRACERS_ALLOW_DYN_STATE` | [`pkg/generic_advdiff/gad_som_advect.py`](../mitjax/pkg/generic_advdiff/gad_som_advect.py) `gad_som_advect` | `not (cfg.GAD_ALLOW_TS_SOM_ADV or cfg.PTRACERS_ALLOW_DYN_STATE)` |
| `GAD_SMOLARKIEWICZ_HACK` | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("GAD_SMOLARKIEWICZ_HACK", _OPT) and trUseSmolHack` |
| `GGL90_IDEMIX_CVMIX_VERSION` | [`pkg/ggl90/ggl90_idemix.py`](../mitjax/pkg/ggl90/ggl90_idemix.py) `ggl90_idemix` | `cfg.cpp.flag("GGL90_IDEMIX_CVMIX_VERSION", opt)` |
| `GGL90_REGULARIZE_MIXINGLENGTH` | [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `cfg.cpp.flag("GGL90_REGULARIZE_MIXINGLENGTH", opt)` |
| `GMREDI_WITH_STABLE_ADJOINT` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `cfg.cpp.flag("GMREDI_WITH_STABLE_ADJOINT")` |
| `GM_BATES_PASSIVE` | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `not cpp.GM_BATES_PASSIVE and gm.GM_useBatesK3d` |
| `GM_BOLUS_ADVEC` | [`pkg/gmredi/gmredi_rtransport.py`](../mitjax/pkg/gmredi/gmredi_rtransport.py) `gmredi_rtransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_BOLUS_ADVEC` | [`pkg/gmredi/gmredi_xtransport.py`](../mitjax/pkg/gmredi/gmredi_xtransport.py) `gmredi_xtransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_BOLUS_ADVEC` | [`pkg/gmredi/gmredi_ytransport.py`](../mitjax/pkg/gmredi/gmredi_ytransport.py) `gmredi_ytransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_BOLUS_BVP` | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `cpp.GM_BOLUS_BVP and gm.GM_useBVP` |
| `GM_EXCLUDE_SUBMESO` | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `not cpp.GM_EXCLUDE_SUBMESO and gm.GM_useSubMeso and gm.GM_AdvForm` |
| `GM_EXTRA_DIAGONAL`, `GM_NON_UNITY_DIAGONAL` | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `diagLoops and not (cpp.GM_NON_UNITY_DIAGONAL and cpp.GM_EXTRA_DIAGONAL)` |
| `GM_GEOM_VARIABLE_K` | [`pkg/gmredi/gmredi_do_exch.py`](../mitjax/pkg/gmredi/gmredi_do_exch.py) `gmredi_do_exch` | `cfg.cpp.GM_GEOM_VARIABLE_K` |
| `GM_GEOM_VARIABLE_K` | [`pkg/gmredi/gmredi_init_fixed.py`](../mitjax/pkg/gmredi/gmredi_init_fixed.py) `gmredi_init_fixed` | `cfg.cpp.GM_GEOM_VARIABLE_K` |
| `GM_VISBECK_VARIABLE_K` | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_autodiff_resets` | `cfg.cpp.flag("GM_VISBECK_VARIABLE_K", "GMREDI_OPTIONS.h")` |
| `HACK_FOR_GMAO_CPL` | [`pkg/seaice/seaice_get_dynforcing.py`](../mitjax/pkg/seaice/seaice_get_dynforcing.py) `_stress_from_fu_fv` | `cfg.cpp.flag("HACK_FOR_GMAO_CPL")` |
| `HAVE_NETCDF` | [`config/params.py`](../mitjax/config/params.py) `use_flags` | `"HAVE_NETCDF" not in macros` |
| `INCLUDE_CONVECT_CALL` | [`model/src/convective_adjustment.py`](../mitjax/model/src/convective_adjustment.py) `convective_adjustment` | `not cfg.cpp.INCLUDE_CONVECT_CALL` |
| `INCLUDE_CONVECT_INI_CALL` | [`model/src/convective_adjustment_ini.py`](../mitjax/model/src/convective_adjustment_ini.py) `convective_adjustment_ini` | `not cfg.cpp.INCLUDE_CONVECT_INI_CALL` |
| `INCLUDE_EP_FORCING_CODE` | [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `ini_fields` | `cfg.cpp.INCLUDE_EP_FORCING_CODE` |
| `INCLUDE_PHIHYD_CALCULATION_CODE` | [`model/src/calc_grad_phi_hyd.py`](../mitjax/model/src/calc_grad_phi_hyd.py) `calc_grad_phi_hyd` | `not cfg.cpp.INCLUDE_PHIHYD_CALCULATION_CODE` |
| `KPP_SMOOTH_DVSQ` | [`pkg/kpp/kpp_forcing_surf.py`](../mitjax/pkg/kpp/kpp_forcing_surf.py) `kpp_forcing_surf` | `cfg.cpp.flag("KPP_SMOOTH_DVSQ", "KPP_OPTIONS.h")` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_u_adv_uu.py`](../mitjax/pkg/mom_fluxform/mom_u_adv_uu.py) `mom_u_adv_uu` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_u_adv_vu.py`](../mitjax/pkg/mom_fluxform/mom_u_adv_vu.py) `mom_u_adv_vu` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_u_adv_wu.py`](../mitjax/pkg/mom_fluxform/mom_u_adv_wu.py) `mom_u_adv_wu` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_v_adv_uv.py`](../mitjax/pkg/mom_fluxform/mom_v_adv_uv.py) `mom_v_adv_uv` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_v_adv_vv.py`](../mitjax/pkg/mom_fluxform/mom_v_adv_vv.py) `mom_v_adv_vv` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_BOUNDARY_CONSERVE` | [`pkg/mom_fluxform/mom_v_adv_wv.py`](../mitjax/pkg/mom_fluxform/mom_v_adv_wv.py) `mom_v_adv_wv` | `cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT)` |
| `MOM_USE_OLD_DEEP_VERT_ADV` | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("MOM_USE_OLD_DEEP_VERT_ADV", _COPT)` |
| `MOM_VI_ORIGINAL_VISCA4` | [`pkg/mom_vecinv/mom_vi_hdissip.py`](../mitjax/pkg/mom_vecinv/mom_vi_hdissip.py) `mom_vi_hdissip` | `cfg.cpp.flag("MOM_VI_ORIGINAL_VISCA4", _OPT)` |
| `MONITOR_TEST_HFACZ` | [`pkg/monitor/mon_vort3.py`](../mitjax/pkg/monitor/mon_vort3.py) `mon_vort3` | `cfg.MONITOR_TEST_HFACZ` |
| `MY82_SMOOTH_RI` | [`pkg/my82/my82_ri_number.py`](../mitjax/pkg/my82/my82_ri_number.py) `my82_ri_number` | `cfg.cpp.flag("MY82_SMOOTH_RI", "MY82_OPTIONS.h")` |
| `NONLIN_FRSURF` | [`model/src/forward_step.py`](../mitjax/model/src/forward_step.py) `nlfs_update_cg2d` | `not cfg.cpp.NONLIN_FRSURF` |
| `NONLIN_FRSURF` | [`pkg/monitor/mon_surfcor.py`](../mitjax/pkg/monitor/mon_surfcor.py) `mon_surfcor` | `cfg.fluidIsAir and cfg.NONLIN_FRSURF and cfg.select_rStar != 0` |
| `OLD_ADV_BCS` | [`pkg/mom_fluxform/mom_u_adv_vu.py`](../mitjax/pkg/mom_fluxform/mom_u_adv_vu.py) `mom_u_adv_vu` | `cfg.cpp.flag("OLD_ADV_BCS", _OPT)` |
| `OLD_ADV_BCS` | [`pkg/mom_fluxform/mom_v_adv_uv.py`](../mitjax/pkg/mom_fluxform/mom_v_adv_uv.py) `mom_v_adv_uv` | `cfg.cpp.flag("OLD_ADV_BCS", _OPT)` |
| `OLD_DST3_FORMULATION` | [`pkg/generic_advdiff/gad_dst3_adv_r.py`](../mitjax/pkg/generic_advdiff/gad_dst3_adv_r.py) `gad_dst3_adv_r` | `cfg.OLD_DST3_FORMULATION` |
| `OLD_DST3_FORMULATION` | [`pkg/generic_advdiff/gad_dst3_adv_x.py`](../mitjax/pkg/generic_advdiff/gad_dst3_adv_x.py) `gad_dst3_adv_x` | `cfg.OLD_DST3_FORMULATION` |
| `OLD_DST3_FORMULATION` | [`pkg/generic_advdiff/gad_dst3_adv_y.py`](../mitjax/pkg/generic_advdiff/gad_dst3_adv_y.py) `gad_dst3_adv_y` | `cfg.OLD_DST3_FORMULATION` |
| `PP81_SMOOTH_RI` | [`pkg/pp81/pp81_ri_number.py`](../mitjax/pkg/pp81/pp81_ri_number.py) `pp81_ri_number` | `cfg.cpp.flag("PP81_SMOOTH_RI", "PP81_OPTIONS.h")` |
| `REAL4_IS_SLOW` | [`eesupp/exch_rs.py`](../mitjax/eesupp/exch_rs.py) `check_rs_is_real8` | `not cfg.cpp.flag("REAL4_IS_SLOW", "CPP_EEOPTIONS.h")` |
| `SALT_PLUME_SPLIT_BASIN` | [`pkg/salt_plume/salt_plume_readparms.py`](../mitjax/pkg/salt_plume/salt_plume_readparms.py) `salt_plume_readparms` | `cfg.cpp.flag("SALT_PLUME_SPLIT_BASIN", "SALT_PLUME_OPTIONS.h")` |
| `SALT_PLUME_VOLUME` | [`pkg/kpp/kpp_forcing_surf.py`](../mitjax/pkg/kpp/kpp_forcing_surf.py) `kpp_forcing_surf` | `salt_plume and cfg.cpp.flag("SALT_PLUME_VOLUME", "KPP_OPTIONS.h")` |
| `SEAICE_ALLOW_BOTTOMDRAG` | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `cfg.cpp.flag("SEAICE_ALLOW_BOTTOMDRAG", "SEAICE_OPTIONS.h") and v["SEAICEbasalDragK2"] > 0.0` |
| `SEAICE_ALLOW_EVP`, `SEAICE_CGRID` | [`pkg/seaice/seaice_read_pickup.py`](../mitjax/pkg/seaice/seaice_read_pickup.py) `seaice_read_pickup` | `cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h") and cfg.cpp.flag("SEAICE_ALLOW_EVP", "SEAICE_OPTIONS.h") \ a…` |
| `SEAICE_ALLOW_FREEDRIFT`, `SEAICE_CGRID` | [`pkg/seaice/seaice_freedrift.py`](../mitjax/pkg/seaice/seaice_freedrift.py) `seaice_freedrift` | `not (cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h") and cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTI…` |
| `SEAICE_CGRID` | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["SEAICEuseDYNAMICS"] and not cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")` |
| `SEAICE_EXTERNAL_FLUXES` | [`pkg/seaice/seaice_budget_ocean.py`](../mitjax/pkg/seaice/seaice_budget_ocean.py) `seaice_budget_ocean` | `not cfg.cpp.flag("SEAICE_EXTERNAL_FLUXES", "SEAICE_OPTIONS.h")` |
| `SEAICE_GREASE` | [`pkg/seaice/seaice_init_fixed.py`](../mitjax/pkg/seaice/seaice_init_fixed.py) `_sitracer_specs` | `cfg.cpp.flag("SEAICE_GREASE", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_bottomdrag_coeffs.py`](../mitjax/pkg/seaice/seaice_bottomdrag_coeffs.py) `seaice_bottomdrag_coeffs` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_calc_ice_strength.py`](../mitjax/pkg/seaice/seaice_calc_ice_strength.py) `seaice_calc_ice_strength` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_check_pickup.py`](../mitjax/pkg/seaice/seaice_check_pickup.py) `seaice_check_pickup` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_h.py`](../mitjax/pkg/seaice/seaice_h.py) `seaice_fields` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_read_pickup.py`](../mitjax/pkg/seaice/seaice_read_pickup.py) `seaice_read_pickup` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SEAICE_ITD` | [`pkg/seaice/seaice_write_pickup.py`](../mitjax/pkg/seaice/seaice_write_pickup.py) `seaice_write_pickup` | `cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h")` |
| `SHORTWAVE_HEATING` | [`model/src/external_fields_load.py`](../mitjax/model/src/external_fields_load.py) `external_fields_load` | `cfg.cpp.SHORTWAVE_HEATING and fp.surfQswFile.strip()` |
| `SOLVE_DIAGONAL_KINNER` | [`model/src/solve_pentadiagonal.py`](../mitjax/model/src/solve_pentadiagonal.py) `solve_pentadiagonal` | `cfg.cpp.SOLVE_DIAGONAL_KINNER` |
| `SOLVE_DIAGONAL_KINNER` | [`model/src/solve_tridiagonal.py`](../mitjax/model/src/solve_tridiagonal.py) `solve_tridiagonal` | `cfg.cpp.SOLVE_DIAGONAL_KINNER` |
| `SOLVE_DIAGONAL_LOWMEMORY` | [`model/src/solve_pentadiagonal.py`](../mitjax/model/src/solve_pentadiagonal.py) `solve_pentadiagonal` | `cfg.cpp.SOLVE_DIAGONAL_LOWMEMORY` |
| `SOLVE_DIAGONAL_LOWMEMORY` | [`model/src/solve_tridiagonal.py`](../mitjax/model/src/solve_tridiagonal.py) `solve_tridiagonal` | `cfg.cpp.SOLVE_DIAGONAL_LOWMEMORY` |
| `SOLVE_DIAGONAL_LOWMEMORY` | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("SOLVE_DIAGONAL_LOWMEMORY", opt)` |
| `TARGET_NEC_SX` | [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `_no_factorized_eos` | `cfg.cpp.TARGET_NEC_SX` |
| `TARGET_NEC_SX` | [`model/src/impldiff.py`](../mitjax/model/src/impldiff.py) `impldiff` | `cfg.cpp.TARGET_NEC_SX` |
| `USE_EXF_INTERPOLATION` | [`pkg/exf/exf_init_fixed.py`](../mitjax/pkg/exf/exf_init_fixed.py) `exf_init_fixed` | `cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h")` |
| `USE_EXF_INTERPOLATION` | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h")` |
| `USE_EXF_INTERPOLATION` | [`pkg/exf/exf_set_uv.py`](../mitjax/pkg/exf/exf_set_uv.py) `exf_set_uv` | `cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h")` |
| `USE_MASK_AND_NO_IF` | [`model/src/ini_nlfs_vars.py`](../mitjax/model/src/ini_nlfs_vars.py) `ini_nlfs_vars` | `cfg.cpp.USE_MASK_AND_NO_IF` |
| `USE_MASK_AND_NO_IF` | [`model/src/update_r_star.py`](../mitjax/model/src/update_r_star.py) `update_r_star` | `cfg.cpp.USE_MASK_AND_NO_IF` |
| `USE_OLD_EXTERNAL_FORCING` | [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_s` | `cfg.cpp.USE_OLD_EXTERNAL_FORCING` |
| `USE_OLD_EXTERNAL_FORCING` | [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_t` | `cfg.cpp.USE_OLD_EXTERNAL_FORCING` |

### Namelist parameters with an explicit refusal

243 checks name a parameter that MITgcm reads from a namelist (any `NAMELIST` statement
upstream); the group is given for orientation. Values that a check refuses stop the run with an error
naming the routine.

| parameter (group) | where (mitjax module, function) | stops when |
|---|---|---|
| `addMassFile` (PARM05) | [`model/src/ini_forcing.py`](../mitjax/model/src/ini_forcing.py) `ini_forcing` | `cfg.cpp.ALLOW_ADDFLUID and fp.addMassFile.strip()` |
| `addSwallFile` (PARM05), `addWwallFile` (PARM05) | [`model/src/add_walls2masks.py`](../mitjax/model/src/add_walls2masks.py) `add_walls2masks` | `not _blank(params.addWwallFile) or not _blank(params.addSwallFile)` |
| `adMxlMaxFlag` (GGL90_PARM01), `mxlMaxFlag` (GGL90_PARM01) | [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `cfg.cpp.flag("ALLOW_AUTODIFF", opt) and ggl.adMxlMaxFlag != ggl.mxlMaxFlag` |
| `areamaskfile` (EXF_NML_02) | [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `exf_cfg_flag(cfg, "EXF_SEAICE_FRACTION") and exf.areamaskfile.strip()` |
| `areamaskfile` (EXF_NML_02) | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `seaice_fraction and v["areamaskfile"].strip()` |
| `balanceQnet` (PARM01), `selectBalanceEmPmR` (PARM01) | [`model/src/external_forcing_surf.py`](../mitjax/model/src/external_forcing_surf.py) `external_forcing_surf` | `(fp.selectBalanceEmPmR >= 1 and noSeaice) or (fp.balanceQnet and noSeaice)` |
| `balanceQnet` (PARM01), `selectBalanceEmPmR` (PARM01) | [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag("ALLOW_BALANCE_FLUXES") and (op.selectBalanceEmPmR == 1 or op.balanceQnet)` |
| `balanceSaltClimRelax` (PARM01), `balanceThetaClimRelax` (PARM01) | [`model/src/forcing_surf_relax.py`](../mitjax/model/src/forcing_surf_relax.py) `forcing_surf_relax` | `cfg.cpp.ALLOW_BALANCE_RELAX and (fp.balanceThetaClimRelax or fp.balanceSaltClimRelax)` |
| `bathyFile` (PARM05) | [`model/src/ini_depths.py`](../mitjax/model/src/ini_depths.py) `ini_depths` | `params.usingPCoords and not _blank(params.bathyFile)` |
| `blankList` (W2_EXCH2_PARM01) | [`pkg/exch2/w2_e2setup.py`](../mitjax/pkg/exch2/w2_e2setup.py) `w2_e2setup` | `w2.blankList[i] != 0` |
| `buoyancyRelation` (PARM01) | [`model/src/calc_phi_hyd.py`](../mitjax/model/src/calc_phi_hyd.py) `calc_phi_hyd` | `params.buoyancyRelation not in ("OCEANIC", "ATMOSPHERIC")` |
| `buoyancyRelation` (PARM01) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `gp.buoyancyRelation not in ("OCEANIC", "ATMOSPHERIC")` |
| `calendarDumps` (CAL_NML/PARM03), `chkPtFreq` (PARM03), `pChkPtFreq` (PARM03) | [`model/src/do_write_pickup.py`](../mitjax/model/src/do_write_pickup.py) `do_write_pickup` | `cal.calendarDumps and (io.pChkPtFreq != 0. or io.chkPtFreq != 0.)` |
| `cosPower` (PARM01) | [`model/src/ini_spherical_polar_grid.py`](../mitjax/model/src/ini_spherical_polar_grid.py) `ini_spherical_polar_grid` | `params.cosPower != 0.` |
| `debugLevel` (PARM01) | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `params.debugLevel >= 4` |
| `debugLevel` (PARM01) | [`model/src/solve_for_pressure.py`](../mitjax/model/src/solve_for_pressure.py) `solve_for_pressure` | `params.debugLevel >= 4` |
| `debugLevel` (PARM01) | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `params.debugLevel >= 4` |
| `debugLevel` (PARM01) | [`pkg/rbcs/rbcs_init_fixed.py`](../mitjax/pkg/rbcs/rbcs_init_fixed.py) `rbcs_init_fixed` | `params.debugLevel >= 3` |
| `debugLevel` (PARM01) | [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `seaice_lsr` | `op.debugLevel >= debLevD` |
| `debugLevel` (PARM01), `printResidualFreq` (PARM02) | [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `cg2d` | `params.debugLevel >= debLevZero and params.printResidualFreq >= 1` |
| `debugLevel` (PARM01), `printResidualFreq` (PARM02) | [`model/src/cg2d_ex0.py`](../mitjax/model/src/cg2d_ex0.py) `cg2d_ex0` | `params.debugLevel >= debLevZero and params.printResidualFreq >= 1` |
| `debugLevel` (PARM01), `printResidualFreq` (PARM02) | [`model/src/cg2d_nsa.py`](../mitjax/model/src/cg2d_nsa.py) `cg2d_nsa` | `params.debugLevel >= debLevZero and params.printResidualFreq >= 1` |
| `debugLevel` (PARM01), `useCubedSphereExchange` (EEPARMS) | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_DEBUG", _OPT) and p.debugLevel >= debLevC and k == 4 \ and p.useCubedSphereExchange` |
| `deepAtmosphere` (PARM04) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `gp.deepAtmosphere` |
| `deepAtmosphere` (PARM04) | [`model/src/set_grid_factors.py`](../mitjax/model/src/set_grid_factors.py) `set_grid_factors` | `params.deepAtmosphere` |
| `deepAtmosphere` (PARM04) | [`model/src/update_cg2d.py`](../mitjax/model/src/update_cg2d.py) `update_cg2d` | `params.deepAtmosphere` |
| `delRFile` (PARM04), `delRcFile` (PARM04) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_grid` | `not _blank(files["delRcFile"]) or not _blank(files["delRFile"])` |
| `diag_pickup_write` (DIAGNOSTICS_LIST) | [`model/src/packages_write_pickup.py`](../mitjax/model/src/packages_write_pickup.py) `packages_write_pickup` | `rp.has("data.diagnostics", "DIAGNOSTICS_LIST", "diag_pickup_write") and \ rp.get("data.diagnostics", "DIAGNOS…` |
| `diagFreq` (PARM03) | [`model/src/solve_for_pressure.py`](../mitjax/model/src/solve_for_pressure.py) `solve_for_pressure` | `params.diagFreq != 0.` |
| `diagFreq` (PARM03) | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `p.diagFreq != 0.` |
| `DIFF1` (SEAICE_PARM01), `SEAICEadvArea` (SEAICE_PARM01), `SEAICEadvHeff` (SEAICE_PARM01), `SEAICEadvSalt` (SEAICE_PARM01), `SEAICEadvSchArea` (SEAICE_PARM01), `SEAICEadvSchHeff` (SEAICE_PARM01), `SEAICEadvSchSnow` (SEAICE_PARM01), `SEAICEadvSnow` (SEAICE_PARM01), `SEAICEdiffKhArea` (SEAICE_PARM01), `SEAICEdiffKhHeff` (SEAICE_PARM01), `SEAICEdiffKhSnow` (SEAICE_PARM01), `SEAICEuseFluxForm` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `(v["SEAICEadvHeff"] or v["SEAICEadvArea"] or v["SEAICEadvSnow"] or v["SEAICEadvSalt"]) and not ( v["SEAICEuse…` |
| `diffKpS` (PARM01) | [`model/src/ini_parms_tracer.py`](../mitjax/model/src/ini_parms_tracer.py) `ini_parms_tracer` | `rp.has("data", "PARM01", "diffKpS")` |
| `dTtracerLev` (PARM03) | [`model/src/cg2d_h.py`](../mitjax/model/src/cg2d_h.py) `ini_parms_cg2d` | `rp.nml.var("data", "PARM03", "dTtracerLev") is not None and rp.nml.var("data", "PARM03", "dTtracerLev").value` |
| `eosType` (PARM01) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `fp.fluidIsWater and fp.eosType == "TEOS10"` |
| `freeSurfFac` (PARM01) | [`model/src/cg2d_h.py`](../mitjax/model/src/cg2d_h.py) `ini_parms_cg2d` | `rp.has("data", "PARM01", "freeSurfFac") and not (implicitFreeSurface or rigidLid)` |
| `gencost_datafile` (ECCO_GENCOST_NML) | [`pkg/ecco/cost_gencost_all.py`](../mitjax/pkg/ecco/cost_gencost_all.py) `gencost_setup` | `_eq(g.gencost_datafile, " ")` |
| `gencost_is3d` (ECCO_GENCOST_NML) | [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `check_customize` | `g.gencost_is3d != (arm in ("m_theta", "m_salt"))` |
| `gencost_mask` (ECCO_GENCOST_NML) | [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `not _eq(g.gencost_mask, " ") and g.gencost_flag in (-3, -4, -5)` |
| `gencost_msk_is3d` (ECCO_GENCOST_NML), `gencost_useDensityMask` (ECCO_GENCOST_NML) | [`pkg/ecco/ecco_phys.py`](../mitjax/pkg/ecco/ecco_phys.py) `check_ecco_phys` | `g.gencost_useDensityMask or g.gencost_msk_is3d or any(_eq(_sub(bf, n), p) for n, p in ((11, "m_freeboard"), (…` |
| `gencost_name` (ECCO_GENCOST_NML) | [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `customize_arm` | `g.gencost_name.rstrip() in ("siv4-conc", "siv4-deconc", "siv4-exconc")` |
| `gencost_name` (ECCO_GENCOST_NML), `using_gencost` (ECCO_GENCOST_NML) | [`pkg/ecco/cost_gencost_all.py`](../mitjax/pkg/ecco/cost_gencost_all.py) `gencost_setup` | `g.gencost_flag == -1 and g.using_gencost and g.gencost_name.rstrip() == "bpv4-grace"` |
| `GGL90diffTKEh` (GGL90_PARM01) | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("ALLOW_GGL90_HORIZDIFF", opt) and ggl.static_float("GGL90diffTKEh") > 0.` |
| `GGL90TKEFile` (GGL90_PARM01) | [`pkg/ggl90/ggl90_init_varia.py`](../mitjax/pkg/ggl90/ggl90_init_varia.py) `ggl90_init_varia` | `not _blank(ggl.GGL90TKEFile)` |
| `GM_AdvForm` (GM_PARM01), `GM_AdvSeparate` (GM_PARM01), `GM_InMomAsStress` (GM_PARM01) | [`pkg/gmredi/gmredi_rtransport.py`](../mitjax/pkg/gmredi/gmredi_rtransport.py) `gmredi_rtransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_AdvForm` (GM_PARM01), `GM_AdvSeparate` (GM_PARM01), `GM_InMomAsStress` (GM_PARM01) | [`pkg/gmredi/gmredi_xtransport.py`](../mitjax/pkg/gmredi/gmredi_xtransport.py) `gmredi_xtransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_AdvForm` (GM_PARM01), `GM_AdvSeparate` (GM_PARM01), `GM_InMomAsStress` (GM_PARM01) | [`pkg/gmredi/gmredi_ytransport.py`](../mitjax/pkg/gmredi/gmredi_ytransport.py) `gmredi_ytransport` | `cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate and not gm.GM_InMomAsStress)` |
| `GM_AdvForm` (GM_PARM01), `GM_useSubMeso` (GM_PARM01) | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `not cpp.GM_EXCLUDE_SUBMESO and gm.GM_useSubMeso and gm.GM_AdvForm` |
| `GM_K3dGMFile` (GM_PARM01) | [`pkg/gmredi/gmredi_init_fixed.py`](../mitjax/pkg/gmredi/gmredi_init_fixed.py) `gmredi_init_fixed` | `gm.GM_K3dGMFile.strip() != ""` |
| `GM_K3dGMFile` (GM_PARM01) | [`pkg/gmredi/gmredi_init_varia.py`](../mitjax/pkg/gmredi/gmredi_init_varia.py) `gmredi_init_varia` | `kapgm and gm.GM_K3dGMFile.strip() != ""` |
| `GM_K3dRediFile` (GM_PARM01) | [`pkg/gmredi/gmredi_init_fixed.py`](../mitjax/pkg/gmredi/gmredi_init_fixed.py) `gmredi_init_fixed` | `gm.GM_K3dRediFile.strip() != ""` |
| `GM_K3dRediFile` (GM_PARM01) | [`pkg/gmredi/gmredi_init_varia.py`](../mitjax/pkg/gmredi/gmredi_init_varia.py) `gmredi_init_varia` | `kapredi and gm.GM_K3dRediFile.strip() != ""` |
| `GM_useBatesK3d` (GM_PARM01) | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `not cpp.GM_BATES_PASSIVE and gm.GM_useBatesK3d` |
| `GM_useBVP` (GM_PARM01) | [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `cpp.GM_BOLUS_BVP and gm.GM_useBVP` |
| `GM_UseBVP` (GM_PARM01) | [`pkg/gmredi/gmredi_readparms.py`](../mitjax/pkg/gmredi/gmredi_readparms.py) `gmredi_readparms` | `f["GM_UseBVP"]` |
| `gravityFile` (PARM01) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `set_ref_state_eos` | `rp.has("data", "PARM05", "gravityFile")` |
| `grdchkvarindex` (GRDCHK_NML) | [`drivers/grdchk.py`](../mitjax/drivers/grdchk.py) `grdchk_settings` | `rp.has("data.grdchk", "GRDCHK_NML", "grdchkvarindex")` |
| `highOrderVorticity` (PARM01), `upwindVorticity` (PARM01) | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `p.highOrderVorticity or p.upwindVorticity` |
| `hMixSmooth` (PARM01) | [`model/src/calc_oce_mxlayer.py`](../mitjax/model/src/calc_oce_mxlayer.py) `calc_oce_mxlayer` | `params.static_float("hMixSmooth") > 0.` |
| `HsaltFile` (SEAICE_PARM01) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `sp.HsaltFile.strip()` |
| `IDEMIX_include_GM` (GGL90_PARM02), `IDEMIX_include_GM_bottom` (GGL90_PARM02) | [`pkg/ggl90/ggl90_idemix.py`](../mitjax/pkg/ggl90/ggl90_idemix.py) `ggl90_idemix` | `useGmredi and (ggl.IDEMIX_include_GM or ggl.IDEMIX_include_GM_bottom)` |
| `implicitIntGravWave` (PARM01) | [`model/src/do_stagger_fields_exchanges.py`](../mitjax/model/src/do_stagger_fields_exchanges.py) `do_stagger_fields_exchanges` | `params.implicitIntGravWave` |
| `implicitIntGravWave` (PARM01) | [`model/src/forward_step.py`](../mitjax/model/src/forward_step.py) `forward_step` | `params.implicitIntGravWave` |
| `implicitIntGravWave` (PARM01), `nIter0` (PARM03) | [`model/src/calc_phi_hyd.py`](../mitjax/model/src/calc_phi_hyd.py) `calc_phi_hyd` | `params.implicitIntGravWave or (params.nIter0 < 0 and not rho_from_eos)` |
| `implicitViscosity` (PARM01), `useCDscheme` (PARM01) | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `params.implicitViscosity and params.useCDscheme and not cfg.cpp.ALLOW_CD_CODE` |
| `interDiffKr_pCell` (PARM04), `pCellMix_select` (PARM04) | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `params.interDiffKr_pCell or params.pCellMix_select > 0` |
| `interViscAr_pCell` (PARM04) | [`model/src/calc_viscosity.py`](../mitjax/model/src/calc_viscosity.py) `calc_viscosity` | `params.interViscAr_pCell` |
| `linFSConserveTr` (PARM01) | [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_s` | `fp.linFSConserveTr` |
| `linFSConserveTr` (PARM01) | [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_t` | `fp.linFSConserveTr` |
| `linFSConserveTr` (PARM01) | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `params.linFSConserveTr` |
| `LSR_mixIniGuess` (SEAICE_PARM01) | [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `seaice_lsr` | `sp.LSR_mixIniGuess >= 2` |
| `LSR_mixIniGuess` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["LSR_mixIniGuess"] >= 2` |
| `mnc_read_bathy` (MNC_01) | [`model/src/ini_depths.py`](../mitjax/model/src/ini_depths.py) `ini_depths` | `cfg.cpp.ALLOW_MNC and params.useMNC and params.mnc_read_bathy` |
| `mnc_read_salt` (MNC_01) | [`model/src/ini_salt.py`](../mitjax/model/src/ini_salt.py) `ini_salt` | `cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_salt` |
| `mnc_read_theta` (MNC_01) | [`model/src/ini_theta.py`](../mitjax/model/src/ini_theta.py) `ini_theta` | `cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_theta` |
| `momImplVertAdv` (PARM01) | [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `p.momImplVertAdv and Nr > 1` |
| `momImplVertAdv` (PARM01), `selectImplicitDrag` (PARM01) | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `(params.momImplVertAdv or params.selectImplicitDrag >= 1) and not implvert` |
| `monitor_mnc` (MNC_01) | [`drivers/adjoint_run.py`](../mitjax/drivers/adjoint_run.py) `ad_exf_records` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `monitor_mnc` (MNC_01) | [`pkg/autodiff/adjoint_monitor.py`](../mitjax/pkg/autodiff/adjoint_monitor.py) `admonitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `monitor_mnc` (MNC_01) | [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `monitor_mnc` (MNC_01) | [`pkg/monitor/monitor.py`](../mitjax/pkg/monitor/monitor.py) `monitor` | `cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc` |
| `monitorFreq` (PARM03) | [`pkg/sbo/sbo_readparms.py`](../mitjax/pkg/sbo/sbo_readparms.py) `sbo_readparms` | `fp.monitorFreq < 0.` |
| `nonHydrostatic` (PARM01) | [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `ini_fields` | `cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic` |
| `nonHydrostatic` (PARM01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic` |
| `nonHydrostatic` (PARM01) | [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `mom_calc_visc` | `cfg.cpp.flag("ALLOW_NONHYDROSTATIC", _OPT) and p.nonHydrostatic` |
| `nonHydrostatic` (PARM01) | [`pkg/monitor/mon_ke.py`](../mitjax/pkg/monitor/mon_ke.py) `mon_ke` | `cfg.ALLOW_NONHYDROSTATIC and cfg.nonHydrostatic` |
| `nonHydrostatic` (PARM01), `quasiHydrostatic` (PARM01), `staggerTimeStep` (PARM01) | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, opt) and (params.nonHydrostatic or (params.quasiHydrostatic and params.staggerTimeStep))` |
| `nonlinFreeSurf` (PARM01), `select_rStar` (PARM01) | [`model/src/calc_grad_phi_hyd.py`](../mitjax/model/src/calc_grad_phi_hyd.py) `calc_grad_phi_hyd` | `params.select_rStar < 2 and params.nonlinFreeSurf >= 4` |
| `nonlinFreeSurf` (PARM01), `selectSigmaCoord` (PARM04) | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.nonlinFreeSurf > 0 \ and params.selectSig…` |
| `num_v_smooth_Ri` (KPP_PARM01) | [`pkg/kpp/kpp_readparms.py`](../mitjax/pkg/kpp/kpp_readparms.py) `kpp_readparms` | `v["num_v_smooth_Ri"] > 0` |
| `pCellMix_select` (PARM04) | [`model/src/calc_viscosity.py`](../mitjax/model/src/calc_viscosity.py) `calc_viscosity` | `params.pCellMix_select > 0` |
| `periodicExternalForcing` (PARM03) | [`model/src/forward_step.py`](../mitjax/model/src/forward_step.py) `forward_step` | `fp.periodicExternalForcing and "forcing" not in pkc` |
| `pickup_read_mnc` (MNC_01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `dict(cfg.static).get(("data.mnc", "mnc_01", "pickup_read_mnc"), False)` |
| `pickup_read_mnc` (MNC_01) | [`pkg/cd_code/cd_code_read_pickup.py`](../mitjax/pkg/cd_code/cd_code_read_pickup.py) `cd_code_read_pickup` | `dict(cfg.static).get(("data.mnc", "mnc_01", "pickup_read_mnc"), False)` |
| `pickup_write_mnc` (MNC_01) | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_MNC") and ip.useMNC and io.pickup_write_mnc` |
| `pickup_write_mnc` (MNC_01) | [`pkg/cd_code/cd_code_write_pickup.py`](../mitjax/pkg/cd_code/cd_code_write_pickup.py) `cd_code_write_pickup` | `cfg.cpp.ALLOW_MNC and dict(cfg.use).get("useMNC", False) and getattr(io, "pickup_write_mnc", False)` |
| `postSolvTempIter` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["postSolvTempIter"] not in (0, 2)` |
| `postSolvTempIter` (SEAICE_PARM01) | [`pkg/seaice/seaice_solve4temp.py`](../mitjax/pkg/seaice/seaice_solve4temp.py) `seaice_solve4temp` | `sp.postSolvTempIter not in (0, 2)` |
| `preDefTopol` (W2_EXCH2_PARM01) | [`pkg/exch2/w2_e2setup.py`](../mitjax/pkg/exch2/w2_e2setup.py) `w2_e2setup` | `w2.preDefTopol == 0` |
| `preDefTopol` (W2_EXCH2_PARM01) | [`pkg/exch2/w2_e2setup.py`](../mitjax/pkg/exch2/w2_e2setup.py) `w2_e2setup` | `w2.preDefTopol == 2` |
| `PTRACERS_addSrelax2EmP` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_forcing_surf.py`](../mitjax/pkg/ptracers/ptracers_forcing_surf.py) `check_unported` | `ptr.PTRACERS_addSrelax2EmP` |
| `PTRACERS_Iter0` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_init_varia.py`](../mitjax/pkg/ptracers/ptracers_init_varia.py) `ptracers_init_varia` | `nIter0 > ptr.PTRACERS_Iter0 or (nIter0 == ptr.PTRACERS_Iter0 and str(pickupSuff).strip() != "")` |
| `PTRACERS_linFSConserve` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `check_unported` | `ptr.PTRACERS_linFSConserve[iTracer-1]` |
| `PTRACERS_monitor_mnc` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_monitor.py`](../mitjax/pkg/ptracers/ptracers_monitor.py) `ptracers_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc` |
| `PTRACERS_monitor_mnc` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_monitor_ad.py`](../mitjax/pkg/ptracers/ptracers_monitor_ad.py) `adptracers_monitor` | `cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc` |
| `PTRACERS_monitorFreq` (PTRACERS_PARM01), `monitorFreq` (PARM03) | [`drivers/run.py`](../mitjax/drivers/run.py) `ptracers_monitor_records` | `p.PTRACERS_monitorFreq != p.monitorFreq` |
| `PTRACERS_pickup_write_mnc` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_write_pickup.py`](../mitjax/pkg/ptracers/ptracers_write_pickup.py) `ptracers_write_pickup` | `ptr.PTRACERS_pickup_write_mnc` |
| `PTRACERS_useDWNSLP` (PTRACERS_PARM01) | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_DOWN_SLOPE", _OPT) and ptr.PTRACERS_useDWNSLP[n]` |
| `quasiHydrostatic` (PARM01), `staggerTimeStep` (PARM01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep` |
| `quasiHydrostatic` (PARM01), `staggerTimeStep` (PARM01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep` |
| `rbcsForcingPeriod` (RBCS_PARM01) | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `_intervals` | `params.rbcsForcingPeriod > 0.0` |
| `rbcsSingleTimeFiles` (RBCS_PARM01) | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_preload` | `params.rbcsSingleTimeFiles` |
| `rbcsVanishingTime` (RBCS_PARM01) | [`pkg/rbcs/rbcs_add_tendency.py`](../mitjax/pkg/rbcs/rbcs_add_tendency.py) `rbcs_add_tendency` | `params.rbcsVanishingTime > 0.0` |
| `readgrid_mnc` (MNC_01) | [`model/src/ini_curvilinear_grid.py`](../mitjax/model/src/ini_curvilinear_grid.py) `ini_curvilinear_grid` | `params.useMNC and params.readgrid_mnc` |
| `relaxSFile` (RBCS_PARM01), `useRBCsalt` (RBCS_PARM01) | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_preload` | `params.useRBCsalt and str(params.relaxSFile).strip()` |
| `relaxUFile` (RBCS_PARM01), `relaxVFile` (RBCS_PARM01), `useRBCuVel` (RBCS_PARM01), `useRBCvVel` (RBCS_PARM01) | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_preload` | `(params.useRBCuVel and str(params.relaxUFile).strip()) or (params.useRBCvVel and str(params.relaxVFile).strip…` |
| `rhoRefFile` (PARM01) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `gp.usingZCoords and not _blank(_get(exp, rp, "PARM01", "rhoRefFile", f"{SD}:59"))` |
| `rigidLid` (PARM01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `check_pickup` | `f in ("dPhiNH ", "Phi_NHyd", "SmagDiff", "FricHeat") or (f == "EtaN " and getattr( ip, "rigidLid", False))` |
| `rotateStressOnAgrid` (EXF_NML_01) | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `v["rotateStressOnAgrid"]` |
| `rotateStressOnAgrid` (EXF_NML_01) | [`pkg/exf/exf_set_uv.py`](../mitjax/pkg/exf/exf_set_uv.py) `exf_set_uv` | `exf.rotateStressOnAgrid` |
| `salt_useDWNSLP` (DWNSLP_PARM01), `temp_useDWNSLP` (DWNSLP_PARM01) | [`pkg/down_slope/dwnslp_readparms.py`](../mitjax/pkg/down_slope/dwnslp_readparms.py) `dwnslp_readparms` | `bool(v["temp_useDWNSLP"]) != bool(tempStepping) or bool(v["salt_useDWNSLP"]) != bool(saltStepping)` |
| `SEAICE_2ndOrderBC` (SEAICE_PARM01), `SEAICE_no_slip` (SEAICE_PARM01) | [`pkg/seaice/seaice_calc_strainrates.py`](../mitjax/pkg/seaice/seaice_calc_strainrates.py) `seaice_calc_strainrates` | `sp.SEAICE_no_slip and sp.SEAICE_2ndOrderBC` |
| `SEAICE_areaGainFormula` (SEAICE_PARM01), `SEAICE_areaLossFormula` (SEAICE_PARM01) | [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `sp.SEAICE_areaGainFormula not in (1, 2) or sp.SEAICE_areaLossFormula not in (1, 2, 3)` |
| `SEAICE_areaGainFormula` (SEAICE_PARM01), `SEAICE_areaLossFormula` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["SEAICE_areaLossFormula"] not in (1, 2, 3) or v["SEAICE_areaGainFormula"] not in (1, 2)` |
| `SEAICE_maskRHS` (SEAICE_PARM01) | [`pkg/seaice/seaice_dynsolver.py`](../mitjax/pkg/seaice/seaice_dynsolver.py) `_dynamics_forcing` | `sp.SEAICE_maskRHS` |
| `SEAICE_tensilFac` (SEAICE_PARM01) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cgrid and sp.SEAICE_tensilFac != 0.` |
| `SEAICEadvArea` (SEAICE_PARM01), `SEAICEadvHeff` (SEAICE_PARM01), `SEAICEadvSalt` (SEAICE_PARM01), `SEAICEadvSnow` (SEAICE_PARM01) | [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `sp.SEAICEadvHeff or sp.SEAICEadvArea or sp.SEAICEadvSnow or sp.SEAICEadvSalt` |
| `SEAICEadvSalt` (SEAICE_PARM01) | [`pkg/seaice/seaice_advdiff.py`](../mitjax/pkg/seaice/seaice_advdiff.py) `_advdiff_single_dim` | `sp.SEAICEadvSalt` |
| `SEAICEbasalDragK2` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `cfg.cpp.flag("SEAICE_ALLOW_BOTTOMDRAG", "SEAICE_OPTIONS.h") and v["SEAICEbasalDragK2"] > 0.0` |
| `SEAICEetaZmethod` (SEAICE_PARM01), `SEAICEsideDrag` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["SEAICEetaZmethod"] not in (0, 3) or v["SEAICEsideDrag"] != 0.0` |
| `SEAICEpresPow0` (SEAICE_PARM01), `SEAICEpresPow1` (SEAICE_PARM01) | [`pkg/seaice/seaice_calc_ice_strength.py`](../mitjax/pkg/seaice/seaice_calc_ice_strength.py) `seaice_calc_ice_strength` | `sp.SEAICEpresPow0 != 1 or sp.SEAICEpresPow1 != 1` |
| `SEAICEuseBDF2` (SEAICE_PARM01) | [`pkg/seaice/seaice_read_pickup.py`](../mitjax/pkg/seaice/seaice_read_pickup.py) `seaice_read_pickup` | `getattr(sp, "SEAICEuseBDF2", False)` |
| `SEAICEuseBDF2` (SEAICE_PARM01) | [`pkg/seaice/seaice_write_pickup.py`](../mitjax/pkg/seaice/seaice_write_pickup.py) `seaice_write_pickup` | `getattr(sp, "SEAICEuseBDF2", False)` |
| `SEAICEuseDYNAMICS` (SEAICE_PARM01) | [`pkg/seaice/dynsolver.py`](../mitjax/pkg/seaice/dynsolver.py) `dynsolver` | `sp.SEAICEuseDYNAMICS` |
| `SEAICEuseDYNAMICS` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["SEAICEuseDYNAMICS"] and not cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")` |
| `SEAICEuseFluxForm` (SEAICE_PARM01) | [`pkg/seaice/advect.py`](../mitjax/pkg/seaice/advect.py) `advect` | `not sp.SEAICEuseFluxForm` |
| `SEAICEuseFREEDRIFT` (SEAICE_PARM01) | [`pkg/seaice/seaice_dynsolver.py`](../mitjax/pkg/seaice/seaice_dynsolver.py) `seaice_dynsolver` | `sp.SEAICEuseFREEDRIFT` |
| `SEAICEuseFREEDRIFT` (SEAICE_PARM01) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v["SEAICEuseFREEDRIFT"]` |
| `SEAICEuseMCE` (SEAICE_PARM01), `SEAICEuseMCS` (SEAICE_PARM01), `SEAICEusePL` (SEAICE_PARM01), `SEAICEuseTD` (SEAICE_PARM01), `SEAICEuseTEM` (SEAICE_PARM01) | [`pkg/seaice/seaice_calc_viscosities.py`](../mitjax/pkg/seaice/seaice_calc_viscosities.py) `seaice_calc_viscosities` | `sp.SEAICEuseTD or sp.SEAICEusePL or sp.SEAICEuseMCS or sp.SEAICEuseMCE or sp.SEAICEuseTEM` |
| `select_rStar` (PARM01) | [`model/src/reset_nlfs_vars.py`](../mitjax/model/src/reset_nlfs_vars.py) `reset_nlfs_vars` | `params.fluidIsAir and params.select_rStar >= 1` |
| `select_rStar` (PARM01) | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `nlfs and not (params.select_rStar > 0) and params.selectSigmaCoord_ne_0 and not cfg.cpp.DISABLE_SIGMA_CODE` |
| `select_rStar` (PARM01) | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `params.select_rStar < 0` |
| `select_rStar` (PARM01) | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `params.select_rStar < 0` |
| `select_rStar` (PARM01) | [`pkg/monitor/mon_surfcor.py`](../mitjax/pkg/monitor/mon_surfcor.py) `mon_surfcor` | `cfg.fluidIsAir and cfg.NONLIN_FRSURF and cfg.select_rStar != 0` |
| `selectAddFluid` (PARM01) | [`model/src/calc_div_ghat.py`](../mitjax/model/src/calc_div_ghat.py) `calc_div_ghat` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `selectAddFluid` (PARM01) | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `selectAddFluid` (PARM01) | [`model/src/integrate_for_w.py`](../mitjax/model/src/integrate_for_w.py) `integrate_for_w` | `cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1` |
| `selectAddFluid` (PARM01) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_ADDFLUID and fp.selectAddFluid != 0` |
| `selectAddFluid` (PARM01) | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_ADDFLUID") and params.selectAddFluid != 0` |
| `selectImplicitDrag` (PARM01) | [`model/src/update_cg2d.py`](../mitjax/model/src/update_cg2d.py) `update_cg2d` | `cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG and params.selectImplicitDrag == 2` |
| `selectImplicitDrag` (PARM01) | [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `p.selectImplicitDrag >= 1` |
| `selectImplicitDrag` (PARM01) | [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `cfg.cpp.flag("ALLOW_SOLVE4_PS_AND_DRAG") and p.selectImplicitDrag == 2` |
| `selectImplicitDrag` (PARM01), `useShelfIce` (PACKAGES) | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_SHELFICE", _OPT) and p.useShelfIce and p.selectImplicitDrag == 0` |
| `selectNHfreeSurf` (PARM01) | [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `check_pickup` | `getattr(ip, "selectNHfreeSurf", 0) >= 1` |
| `selectNHfreeSurf` (PARM01) | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, "ALLOW_NONHYDROSTATIC") and params.selectNHfreeSurf >= 1` |
| `selectSigmaCoord` (PARM04) | [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `ini_fields` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and ip.selectSigmaCoord != 0` |
| `selectSigmaCoord` (PARM04) | [`model/src/ini_masks_etc.py`](../mitjax/model/src/ini_masks_etc.py) `ini_masks_etc` | `params.selectSigmaCoord != 0` |
| `selectSigmaCoord` (PARM04) | [`model/src/initialise_varia.py`](../mitjax/model/src/initialise_varia.py) `executed_pending` | `ip.selectSigmaCoord != 0` |
| `selectSigmaCoord` (PARM04) | [`model/src/integrate_for_w.py`](../mitjax/model/src/integrate_for_w.py) `integrate_for_w` | `cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.selectSigmaCoord != 0` |
| `SItrFile` (SEAICE_PARM03) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `sitracer and any(f.strip() for f in sp.SItrFile)` |
| `snowprecipfile` (EXF_NML_02) | [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `exfp.snowprecipfile.strip()` |
| `staggerTimeStep` (PARM01) | [`pkg/mom_common/mom_quasihydrostatic.py`](../mitjax/pkg/mom_common/mom_quasihydrostatic.py) `mom_quasihydrostatic` | `cfg.cpp.flag("ALLOW_QHYD_STAGGER_TS", _OPT) and params.staggerTimeStep` |
| `startFromPickupAB2` (PARM03) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `rp.has("data", "PARM03", "startFromPickupAB2")` |
| `surfQswFile` (PARM05) | [`model/src/external_fields_load.py`](../mitjax/model/src/external_fields_load.py) `external_fields_load` | `cfg.cpp.SHORTWAVE_HEATING and fp.surfQswFile.strip()` |
| `surfQswFile` (PARM05) | [`model/src/ini_forcing.py`](../mitjax/model/src/ini_forcing.py) `ini_forcing` | `fp.surfQswFile.strip()` |
| `tidePotfile` (EXF_NML_02) | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `v["tidePotfile"].strip()` |
| `uIceFile` (SEAICE_PARM01), `vIceFile` (SEAICE_PARM01) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `sp.uIceFile.strip() or sp.vIceFile.strip()` |
| `useAIM` (PACKAGES) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_AIM and _use(cfg, "useAIM")` |
| `useAIM` (PACKAGES) | [`pkg/monitor/mon_surfcor.py`](../mitjax/pkg/monitor/mon_surfcor.py) `mon_surfcor` | `cfg.ALLOW_AIM and cfg.useAIM` |
| `useAIM` (PACKAGES), `useAtm_Phys` (PACKAGES), `useFIZHI` (PACKAGES) | [`model/src/do_atmospheric_phys.py`](../mitjax/model/src/do_atmospheric_phys.py) `do_atmospheric_phys` | `any(dict(cfg.use).get(n, False) for n in ("useFIZHI", "useAtm_Phys", "useAIM"))` |
| `useAtmWind` (EXF_NML_01), `useRelativeWind` (CHEAPAML_PARM02/EXF_NML_01) | [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `exfp.useRelativeWind and exfp.useAtmWind` |
| `useBulkForce` (PACKAGES) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_BULK_FORCE and _use(cfg, "useBulkForce")` |
| `useCentralDiff` (GRDCHK_NML) | [`api.py`](../mitjax/api.py) `Experiment.grdchk` | `not s["useCentralDiff"]` |
| `useCheapAML` (PACKAGES) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_CHEAPAML and _use(cfg, "useCheapAML")` |
| `useCubedSphereExchange` (EEPARMS) | [`pkg/gmredi/gmredi_calc_qgleith.py`](../mitjax/pkg/gmredi/gmredi_calc_qgleith.py) `gmredi_calc_qgleith` | `params.useCubedSphereExchange` |
| `useExfYearlyFields` (EXF_NML_01) | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `v["useExfYearlyFields"]` |
| `useExfZenIncoming` (EXF_NML_01) | [`pkg/exf/exf_radiation.py`](../mitjax/pkg/exf/exf_radiation.py) `exf_radiation` | `cfg.cpp.flag("ALLOW_ZENITHANGLE", "EXF_OPTIONS.h") and (exf.useExfZenAlbedo or exf.useExfZenIncoming)` |
| `useExfZenIncoming` (EXF_NML_01) | [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `v["useExfZenIncoming"]` |
| `useGCHEM` (PACKAGES) | [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `cfg.cpp.ALLOW_GCHEM and _use(cfg, "useGCHEM")` |
| `useGCHEM` (PACKAGES) | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `check_unported` | `cfg.cpp.flag("ALLOW_GCHEM", "PTRACERS_OPTIONS.h") and cfg.use_flag("useGCHEM")` |
| `useGMRedi` (PACKAGES) | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `params.useGMRedi and gm is None` |
| `useGMRedi` (PACKAGES) | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `params.useGMRedi` |
| `useGMRedi` (PACKAGES) | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `cfg.cpp.ALLOW_GMREDI and params.useGMRedi and gm is None` |
| `useHB87stressCoupling` (SEAICE_PARM01) | [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `seaice_lsr` | `sp.useHB87stressCoupling` |
| `useHB87stressCoupling` (SEAICE_PARM01) | [`pkg/seaice/seaice_ocean_stress.py`](../mitjax/pkg/seaice/seaice_ocean_stress.py) `seaice_ocean_stress` | `sp.useHB87stressCoupling` |
| `useIDEMIX` (GGL90_PARM01) | [`pkg/ggl90/ggl90_readparms.py`](../mitjax/pkg/ggl90/ggl90_readparms.py) `ggl90_readparms` | `v["useIDEMIX"]` |
| `useKL10` (PACKAGES) | [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `cfg.cpp.ALLOW_KL10 and _use(cfg, "useKL10")` |
| `useKL10` (PACKAGES) | [`model/src/calc_viscosity.py`](../mitjax/model/src/calc_viscosity.py) `calc_viscosity` | `cfg.cpp.flag("ALLOW_KL10") and cfg.use_flag("useKL10")` |
| `useLANGMUIR` (GGL90_PARM01) | [`pkg/ggl90/ggl90_readparms.py`](../mitjax/pkg/ggl90/ggl90_readparms.py) `ggl90_readparms` | `v["useLANGMUIR"]` |
| `useLANGMUIR` (GGL90_PARM01) | [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `cfg.cpp.flag("ALLOW_GGL90", _COPT) and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") \ and p.useLAN…` |
| `useLANGMUIR` (GGL90_PARM01), `vectorInvariantMomentum` (PARM01) | [`model/src/dynamics.py`](../mitjax/model/src/dynamics.py) `dynamics` | `ggl is not None and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") and ggl.useLANGMUIR \ and not par…` |
| `useMin4hFacEdges` (PARM04) | [`model/src/ini_masks_etc.py`](../mitjax/model/src/ini_masks_etc.py) `ini_masks_etc` | `params.useMin4hFacEdges` |
| `useNSACGSolver` (PARM02) | [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `cg2d_solve` | `cfg.cpp.ALLOW_CG2D_NSA and params.useNSACGSolver` |
| `useOBCS` (PACKAGES) | [`model/src/calc_surf_dr.py`](../mitjax/model/src/calc_surf_dr.py) `calc_surf_dr` | `cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS")` |
| `useOBCS` (PACKAGES) | [`model/src/ini_depths.py`](../mitjax/model/src/ini_depths.py) `ini_depths` | `cfg.cpp.ALLOW_OBCS and cfg.use_flag("useOBCS")` |
| `useOBCS` (PACKAGES) | [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `cfg.cpp.ALLOW_OBCS and params.useOBCS` |
| `useOBCS` (PACKAGES) | [`pkg/exf/exf_init_fixed.py`](../mitjax/pkg/exf/exf_init_fixed.py) `exf_init_fixed` | `cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS")` |
| `useOBCS` (PACKAGES) | [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS` |
| `useOBCS` (PACKAGES) | [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS` |
| `useOBCS` (PACKAGES) | [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `cfg.cpp.flag("ALLOW_OBCS", _OPT) and cfg.use_flag("useOBCS")` |
| `useOBCS` (PACKAGES) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS")` |
| `useOBCS` (PACKAGES), `useThSIce` (PACKAGES) | [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `(cfg.cpp.flag("ALLOW_THSICE") and cfg.use_flag("useThSIce")) \ or (cfg.cpp.flag("ALLOW_OBCS") and cfg.use_fla…` |
| `useOffLine` (PACKAGES) | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `cfg.cpp.ALLOW_OFFLINE and params.useOffLine` |
| `useOffLine` (PACKAGES) | [`model/src/do_stagger_fields_exchanges.py`](../mitjax/model/src/do_stagger_fields_exchanges.py) `do_stagger_fields_exchanges` | `params.useOffLine` |
| `usePickupBeforeC54` (PARM01) | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `_exact_conserv` | `realFW and params.usePickupBeforeC54` |
| `useRBCS` (PACKAGES) | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `check_unported` | `cfg.cpp.flag("ALLOW_RBCS", "PTRACERS_OPTIONS.h") and cfg.use_flag("useRBCS")` |
| `useRBCsalt` (RBCS_PARM01) | [`pkg/rbcs/rbcs_add_tendency.py`](../mitjax/pkg/rbcs/rbcs_add_tendency.py) `rbcs_add_tendency` | `tracerNum == 2 and params.useRBCsalt` |
| `useRBCuVel` (RBCS_PARM01), `useRBCvVel` (RBCS_PARM01) | [`pkg/rbcs/rbcs_add_tendency.py`](../mitjax/pkg/rbcs/rbcs_add_tendency.py) `rbcs_add_tendency` | `(tracerNum == -1 and params.useRBCuVel) or (tracerNum == -2 and params.useRBCvVel)` |
| `useRBCuVel` (RBCS_PARM01), `useRBCvVel` (RBCS_PARM01) | [`pkg/rbcs/rbcs_fields_load.py`](../mitjax/pkg/rbcs/rbcs_fields_load.py) `rbcs_fields_load` | `not cfg.cpp.DISABLE_RBCS_MOM and (params.useRBCuVel or params.useRBCvVel)` |
| `useRBCuVel` (RBCS_PARM01), `useRBCvVel` (RBCS_PARM01) | [`pkg/rbcs/rbcs_init_fixed.py`](../mitjax/pkg/rbcs/rbcs_init_fixed.py) `rbcs_init_fixed` | `params.useRBCuVel or params.useRBCvVel` |
| `useRealFreshWaterFlux` (PARM01) | [`model/src/integr_continuity.py`](../mitjax/model/src/integr_continuity.py) `integr_continuity.level_k` | `k == sz.Nr and params.usingPCoords and params.fluidIsWater and params.useRealFreshWaterFlux` |
| `useRealFreshWaterFlux` (PARM01) | [`pkg/mom_fluxform/mom_calc_rtrans.py`](../mitjax/pkg/mom_fluxform/mom_calc_rtrans.py) `mom_calc_rtrans` | `nonlin and (k == Nr+1 and params.useRealFreshWaterFlux and params.usingPCoords)` |
| `useRelativeWind` (CHEAPAML_PARM02/EXF_NML_01) | [`pkg/seaice/seaice_get_dynforcing.py`](../mitjax/pkg/seaice/seaice_get_dynforcing.py) `seaice_get_dynforcing` | `exfp.useRelativeWind` |
| `useSALT_PLUME` (PACKAGES) | [`drivers/model.py`](../mitjax/drivers/model.py) `Model._packages` | `cfg.use_flag("useSALT_PLUME")` |
| `useSALT_PLUME` (PACKAGES) | [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `cfg.cpp.ALLOW_SALT_PLUME and params.useSALT_PLUME` |
| `useSALT_PLUME` (PACKAGES) | [`model/src/external_forcing_surf.py`](../mitjax/model/src/external_forcing_surf.py) `external_forcing_surf` | `cfg.cpp.ALLOW_SALT_PLUME and _use(cfg, "useSALT_PLUME")` |
| `useSALT_PLUME` (PACKAGES) | [`pkg/kpp/kpp_calc.py`](../mitjax/pkg/kpp/kpp_calc.py) `kpp_calc` | `salt_plume_on and cfg.use_flag("useSALT_PLUME")` |
| `useSALT_PLUME` (PACKAGES) | [`pkg/kpp/kpp_forcing_surf.py`](../mitjax/pkg/kpp/kpp_forcing_surf.py) `kpp_forcing_surf` | `salt_plume and cfg.use_flag("useSALT_PLUME")` |
| `useSALT_PLUME` (PACKAGES) | [`pkg/kpp/kpp_routines.py`](../mitjax/pkg/kpp/kpp_routines.py) `_salt_plume_args` | `cfg.use_flag("useSALT_PLUME")` |
| `useSALT_PLUME` (PACKAGES) | [`pkg/salt_plume/salt_plume_readparms.py`](../mitjax/pkg/salt_plume/salt_plume_readparms.py) `salt_plume_readparms` | `cfg.use_flag("useSALT_PLUME")` |
| `useSEAICE` (PACKAGES) | [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `params.usingPCoords and dict(cfg.use).get("useSEAICE", False)` |
| `useSEAICE` (PACKAGES) | [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cfg.cpp.flag("ALLOW_AUTODIFF") and not cfg.use_flag("useSEAICE")` |
| `useSHELFICE` (PACKAGES) | [`model/src/external_forcing_surf.py`](../mitjax/model/src/external_forcing_surf.py) `external_forcing_surf` | `cfg.cpp.ALLOW_SHELFICE and _use(cfg, "useSHELFICE")` |
| `useShelfIce` (PACKAGES) | [`model/src/ini_masks_etc.py`](../mitjax/model/src/ini_masks_etc.py) `ini_masks_etc` | `cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce")` |
| `useShelfIce` (PACKAGES) | [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `cfg.cpp.flag("ALLOW_SHELFICE", opt) and cfg.use_flag("useShelfIce")` |
| `useShelfIce` (PACKAGES) | [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce")` |
| `useShelfIce` (PACKAGES) | [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `k_surface` | `params.usingZCoords and params.useShelfIce` |
| `useShelfIce` (PACKAGES) | [`pkg/seaice/seaice_init_fixed.py`](../mitjax/pkg/seaice/seaice_init_fixed.py) `seaice_init_fixed` | `cfg.cpp.flag("ALLOW_SHELFICE") and cfg.use_flag("useShelfIce")` |
| `useSRCGSolver` (PARM02) | [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `cg2d_solve` | `params.useSRCGSolver` |
| `useThSIce` (PACKAGES) | [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `cfg.use_flag("useThSIce")` |
| `usingCurvilinearGrid` (PARM04) | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `params.usingCurvilinearGrid or params.rotateGrid` |
| `usingCylindricalGrid` (PARM04) | [`model/src/ini_grid.py`](../mitjax/model/src/ini_grid.py) `ini_grid` | `params.usingCylindricalGrid` |
| `usingCylindricalGrid` (PARM04) | [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `gp.usingCylindricalGrid` |
| `usingCylindricalGrid` (PARM04) | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `params.usingCylindricalGrid` |
| `usingCylindricalGrid` (PARM04) | [`pkg/mom_fluxform/mom_fluxform.py`](../mitjax/pkg/mom_fluxform/mom_fluxform.py) `mom_fluxform` | `params.usingCylindricalGrid` |
| `viscFacInFw` (AUTODIFF_PARM01) | [`pkg/autodiff/autodiff_readparms.py`](../mitjax/pkg/autodiff/autodiff_readparms.py) `adjoint_mode_check` | `v["viscFacInFw"] != 1.0` |
| `W2_mapIO` (W2_EXCH2_PARM01) | [`pkg/exch2/w2_set_map_tiles.py`](../mitjax/pkg/exch2/w2_set_map_tiles.py) `w2_set_map_tiles` | `w2.W2_mapIO == 0` |
| `W2_mapIO` (W2_EXCH2_PARM01) | [`pkg/exch2/w2_set_map_tiles.py`](../mitjax/pkg/exch2/w2_set_map_tiles.py) `w2_set_map_tiles` | `w2.W2_mapIO == 0` |
| `wghtBalanceFile` (PARM05) | [`model/src/ini_forcing.py`](../mitjax/model/src/ini_forcing.py) `ini_forcing` | `fp.wghtBalanceFile.strip()` |
| `wspeedfile` (BULKF_PARM01/EXF_NML_02) | [`pkg/exf/exf_wind.py`](../mitjax/pkg/exf/exf_wind.py) `exf_wind` | `exf.wspeedfile.strip() == ""` |
| `xx_gentim2d_file` (CTRL_NML_GENARR) | [`drivers/model.py`](../mitjax/drivers/model.py) `Model._ctrl_init` | `cfg.cpp.flag("ALLOW_EXF") and fstr_prefix(g.xx_gentim2d_file, n, lit)` |
| `xx_gentim2d_glosum` (CTRL_NML_GENARR) | [`pkg/ctrl/ctrl_map_gentim2d.py`](../mitjax/pkg/ctrl/ctrl_map_gentim2d.py) `ctrl_map_gentim2d` | `g.xx_gentim2d_glosum` |
| `xx_gentim2d_preproc` (CTRL_NML_GENARR) | [`drivers/grdchk.py`](../mitjax/drivers/grdchk.py) `controls` | `startrec != 1 or any(p.strip() == "docycle" for p in g.xx_gentim2d_preproc)` |
| `xx_gentim2d_preproc` (CTRL_NML_GENARR) | [`pkg/ctrl/ctrl_cost_driver.py`](../mitjax/pkg/ctrl/ctrl_cost_driver.py) `ctrl_cost_driver` | `any(p.strip() == "replicate" for p in g.xx_gentim2d_preproc)` |

### Other checks

9 checks on the output-only switches (`useDiagnostics`, `useMNC`, `useLayers`) are not
reached by a run: the Model runs the kernels with these switches off. The remaining
290 checks guard grid, tiling, input-file and internal conditions; they are listed so that
every refusal in the code is on this page.

| where (mitjax module, function) | stops when |
|---|---|
| [`api.py`](../mitjax/api.py) `Experiment.grdchk` | `len(pts) > gp.MAXGRDCHECKS` |
| [`api.py`](../mitjax/api.py) `TileMap.of` | `len(dims) != 1` |
| [`api.py`](../mitjax/api.py) `TileMap.of` | `sz.nPx * sz.nPy != 1` |
| [`config/namelists.py`](../mitjax/config/namelists.py) `_coerce` | `len(digits) > REAL4_READ_DIGITS` |
| [`config/namelists.py`](../mitjax/config/namelists.py) `read_run_namelists` | `re.search(r"(?<![\w.$])(eedata\|data)(\.[\w.]+)?(?![\w.])", body)` |
| [`config/own_code.py`](../mitjax/config/own_code.py) `check_own_routines` | `bad` |
| [`config/packages.py`](../mitjax/config/packages.py) `ad_config_h` | `version != "Forward version"` |
| [`config/params.py`](../mitjax/config/params.py) `require_ported` | `missing` |
| [`config/params.py`](../mitjax/config/params.py) `require_supported` | `bad` |
| [`drivers/ad_switches.py`](../mitjax/drivers/ad_switches.py) `with_run_switches` | `bad` |
| [`drivers/adjoint_run.py`](../mitjax/drivers/adjoint_run.py) `GenarrAdjoint.__init__` | `len(m.genarr_live) != 1` |
| [`drivers/adjoint_run.py`](../mitjax/drivers/adjoint_run.py) `GenarrAdjoint.sharded_value_and_grad_fn` | `self.monitor` |
| [`drivers/grdchk.py`](../mitjax/drivers/grdchk.py) `grdchk_settings` | `sz.nPx * sz.nPy != 1` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model.__init__` | `any(own.get(n, ("",))[0] == "ported" for n in own_code.MAIN_ONLY)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model.__init__` | `rp_has_pickupSuff(exp)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model.__init__` | `unported` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model._cal_exf_seaice` | `not useEXF` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model._cal_exf_seaice` | `rp_has_pickupSuff(self.exp)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model._cost_init_generic` | `not (atl or trc or tst or ice or ecco)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model._ctrl_first_guess_zero` | `not ((doInitXX and optimcycle == 0) or doAdmTlm)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `Model._ecco_init` | `mo is None` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `check_packages` | `on` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `ptracers_init_state` | `rp_has_pickupSuff(m.exp)` |
| [`drivers/model.py`](../mitjax/drivers/model.py) `with_namelist` | `old.shape` |
| [`eesupp/exch1_tables.py`](../mitjax/eesupp/exch1_tables.py) `exch1_exchange_maps` | `sz.nPx * sz.nPy != 1` |
| [`eesupp/exch1_tables.py`](../mitjax/eesupp/exch1_tables.py) `exch1_exchange_maps` | `useCubedSphereExchange` |
| [`eesupp/exchange.py`](../mitjax/eesupp/exchange.py) `_check_signs` | `withSigns not in probed` |
| [`eesupp/global_sum_singlecpu.py`](../mitjax/eesupp/global_sum_singlecpu.py) `global_sum_singlecpu_rl` | `"exch2" in cfg.packages` |
| [`eesupp/print.py`](../mitjax/eesupp/print.py) `print_message` | `io.myProcId != 0` |
| [`eesupp/print.py`](../mitjax/eesupp/print.py) `print_message` | `io.numberOfProcs == 0` |
| [`eesupp/print.py`](../mitjax/eesupp/print.py) `print_message` | `unit == ERROR_MESSAGE_UNIT and message.strip(" ") != ""` |
| [`model/grid.py`](../mitjax/model/grid.py) `n_tiles` | `size.nPx != 1 or size.nPy != 1` |
| [`model/src/adams_bashforth2.py`](../mitjax/model/src/adams_bashforth2.py) `adams_bashforth2` | `kArg == 0` |
| [`model/src/adams_bashforth3.py`](../mitjax/model/src/adams_bashforth3.py) `adams_bashforth3` | `kArg == 0` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `_hooks` | `getattr(cfg.cpp, opt) and on` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_s` | `kSurface == -1` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_t` | `fp.fluidIsAir` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_t` | `kSurface == -1` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_u` | `kSurface == -1` |
| [`model/src/apply_forcing.py`](../mitjax/model/src/apply_forcing.py) `apply_forcing_v` | `kSurface == -1` |
| [`model/src/calc_3d_diffusivity.py`](../mitjax/model/src/calc_3d_diffusivity.py) `calc_3d_diffusivity` | `getattr(cfg.cpp, opt)` |
| [`model/src/calc_grad_phi_hyd.py`](../mitjax/model/src/calc_grad_phi_hyd.py) `calc_grad_phi_hyd` | `params.fluidIsAir` |
| [`model/src/calc_grad_phi_hyd.py`](../mitjax/model/src/calc_grad_phi_hyd.py) `calc_grad_phi_hyd` | `params.fluidIsWater` |
| [`model/src/calc_oce_mxlayer.py`](../mitjax/model/src/calc_oce_mxlayer.py) `calc_oce_mxlayer` | `method == 2` |
| [`model/src/calc_oce_mxlayer.py`](../mitjax/model/src/calc_oce_mxlayer.py) `calc_oce_mxlayer` | `params.usingPCoords` |
| [`model/src/calc_phi_hyd.py`](../mitjax/model/src/calc_phi_hyd.py) `calc_phi_hyd` | `params.selectSigmaCoord_ne_0` |
| [`model/src/calc_phi_hyd.py`](../mitjax/model/src/calc_phi_hyd.py) `calc_phi_hyd` | `params.selectSigmaCoord_ne_0` |
| [`model/src/calc_r_star.py`](../mitjax/model/src/calc_r_star.py) `calc_r_star` | `params.fluidIsAir` |
| [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `EXCH_S3D_RL` | `myNz != 1` |
| [`model/src/cg2d.py`](../mitjax/model/src/cg2d.py) `cg2d` | `nIterMin >= 0` |
| [`model/src/convective_adjustment.py`](../mitjax/model/src/convective_adjustment.py) `convective_adjustment` | `not params.usingZCoords` |
| [`model/src/convective_adjustment_ini.py`](../mitjax/model/src/convective_adjustment_ini.py) `convective_adjustment_ini` | `not params.usingZCoords` |
| [`model/src/correction_step.py`](../mitjax/model/src/correction_step.py) `correction_step` | `getattr(cfg.cpp, opt)` |
| [`model/src/correction_step.py`](../mitjax/model/src/correction_step.py) `correction_step` | `params.use3Dsolver` |
| [`model/src/diags_phi_hyd.py`](../mitjax/model/src/diags_phi_hyd.py) `diags_phi_hyd` | `nlfs_rstar and (params.fluidIsAir or params.usingPCoords)` |
| [`model/src/diags_phi_rlow.py`](../mitjax/model/src/diags_phi_rlow.py) `diags_phi_rlow` | `nlfs_rstar and params.fluidIsAir` |
| [`model/src/diags_phi_rlow.py`](../mitjax/model/src/diags_phi_rlow.py) `diags_phi_rlow` | `nlfs_rstar and params.usingPCoords` |
| [`model/src/diags_phi_rlow.py`](../mitjax/model/src/diags_phi_rlow.py) `diags_phi_rlow` | `not (params.usingZCoords or params.usingPCoords)` |
| [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_autodiff_resets` | `getattr(cfg.cpp, opt)` |
| [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `_oceanic_phys_k_loop` | `doDiagsRho >= 4` |
| [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `getattr(params, pkg)` |
| [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `kpp is None` |
| [`model/src/do_oceanic_phys.py`](../mitjax/model/src/do_oceanic_phys.py) `do_oceanic_phys` | `useDWNSLP` |
| [`model/src/do_write_pickup.py`](../mitjax/model/src/do_write_pickup.py) `do_write_pickup` | `exp is None` |
| [`model/src/do_write_pickup.py`](../mitjax/model/src/do_write_pickup.py) `do_write_pickup` | `permPickup` |
| [`model/src/external_fields_load.py`](../mitjax/model/src/external_fields_load.py) `get_periodic_interval` | `cycleLength == 0.` |
| [`model/src/external_forcing.py`](../mitjax/model/src/external_forcing.py) `external_forcing_u` | `kSurface == -1` |
| [`model/src/external_forcing.py`](../mitjax/model/src/external_forcing.py) `external_forcing_v` | `kSurface == -1` |
| [`model/src/find_alpha.py`](../mitjax/model/src/find_alpha.py) `find_alpha` | `eqs in ("POLY3", "UNESCO", "TEOS10")` |
| [`model/src/find_alpha.py`](../mitjax/model/src/find_alpha.py) `find_beta` | `eqs in ("POLY3", "UNESCO", "TEOS10")` |
| [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `find_rho_2d` | `eqs.rstrip() == "POLY3"` |
| [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `find_rho_2d` | `eqs.rstrip() in ("TEOS10", "IDEALG")` |
| [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `find_rho_scalar` | `eqs.rstrip() == "POLY3"` |
| [`model/src/find_rho.py`](../mitjax/model/src/find_rho.py) `find_rho_scalar` | `eqs.rstrip() in ("TEOS10", "IDEALG")` |
| [`model/src/forward_step.py`](../mitjax/model/src/forward_step.py) `forward_step` | `getattr(cfg.cpp, opt)` |
| [`model/src/forward_step.py`](../mitjax/model/src/forward_step.py) `nlfs_update_hfac` | `params.selectSigmaCoord_ne_0` |
| [`model/src/freesurf_rescale_g.py`](../mitjax/model/src/freesurf_rescale_g.py) `freesurf_rescale_g` | `params.selectSigmaCoord_ne_0` |
| [`model/src/ini_cori.py`](../mitjax/model/src/ini_cori.py) `ini_cori` | `sel == 3` |
| [`model/src/ini_curvilinear_grid.py`](../mitjax/model/src/ini_curvilinear_grid.py) `ini_curvilinear_grid` | `old_io` |
| [`model/src/ini_eos.py`](../mitjax/model/src/ini_eos.py) `ini_eos` | `eqs == "POLY3"` |
| [`model/src/ini_eos.py`](../mitjax/model/src/ini_eos.py) `ini_eos` | `eqs in ("TEOS10", "IDEALG")` |
| [`model/src/ini_fields.py`](../mitjax/model/src/ini_fields.py) `_routines` | `importlib.util.find_spec(f"mitjax.verification.{pkg}") is None or \ importlib.util.find_spec(modname) is None` |
| [`model/src/ini_forcing.py`](../mitjax/model/src/ini_forcing.py) `_swfrac3d` | `not fp.usingZCoords` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `grid_max` | `cfg.experiment and _code_override(cfg, "W2_EXCH2_SIZE.h")` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `momStepping` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_dyn` | `rp.has("data", grp, key)` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_grid` | `buoyancyRelation == "OCEANICP"` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_grid` | `rp.has("data", "PARM04", key)` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `ini_parms_time` | `abs(endTime - tmpVar) > deltaTClock * 1.e-6` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `load_ref_files_tref_sref` | `not _blank(tRefFile) or not _blank(sRefFile)` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `set_ref_state_eos` | `buoyancyRelation.strip() != "OCEANIC"` |
| [`model/src/ini_parms.py`](../mitjax/model/src/ini_parms.py) `set_ref_state_hyd_press_1d` | `selectP_inEOS_Zc == 1` |
| [`model/src/ini_parms_forcing.py`](../mitjax/model/src/ini_parms_forcing.py) `ini_parms_forcing` | `str(buoyancyRelation).strip() != "OCEANIC"` |
| [`model/src/ini_parms_tracer.py`](../mitjax/model/src/ini_parms_tracer.py) `ini_parms_tracer` | `rp.has("data", grp, key)` |
| [`model/src/ini_spherical_polar_grid.py`](../mitjax/model/src/ini_spherical_polar_grid.py) `ini_spherical_polar_grid` | `params.rotateGrid` |
| [`model/src/ini_vertical_grid.py`](../mitjax/model/src/ini_vertical_grid.py) `ini_vertical_grid` | `params.setCenterDr` |
| [`model/src/initialise_varia.py`](../mitjax/model/src/initialise_varia.py) `initialise_varia` | `convect_ini and not convect_ini_by_caller` |
| [`model/src/initialise_varia.py`](../mitjax/model/src/initialise_varia.py) `initialise_varia` | `getattr(cfg.cpp, opt)` |
| [`model/src/load_fields_driver.py`](../mitjax/model/src/load_fields_driver.py) `load_fields_driver` | `ctrl is None` |
| [`model/src/load_grid_spacing.py`](../mitjax/model/src/load_grid_spacing.py) `load_grid_spacing` | `not _blank(getattr(params, key))` |
| [`model/src/momentum_correction_step.py`](../mitjax/model/src/momentum_correction_step.py) `momentum_correction_step` | `getattr(cfg.cpp, opt)` |
| [`model/src/packages_write_pickup.py`](../mitjax/model/src/packages_write_pickup.py) `packages_write_pickup` | `others` |
| [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `check_pickup` | `f in ("GwNm1 ", "GwNm2 ", "QH_GwNm1", "QH_GwNm2")` |
| [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `check_pickup` | `nbFields <= 0` |
| [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `nbFields <= 0` |
| [`model/src/read_pickup.py`](../mitjax/model/src/read_pickup.py) `read_pickup` | `not ip.pickup_read_mdsio` |
| [`model/src/rotate_uv2en.py`](../mitjax/model/src/rotate_uv2en.py) `rotate_uv2en_rl` | `not xy2en or not switchGrid or kSize != sz.Nr` |
| [`model/src/salt_integrate.py`](../mitjax/model/src/salt_integrate.py) `salt_integrate` | `params.AdamsBashforth_S` |
| [`model/src/solve_pentadiagonal.py`](../mitjax/model/src/solve_pentadiagonal.py) `solve_pentadiagonal` | `Nr < 3` |
| [`model/src/temp_integrate.py`](../mitjax/model/src/temp_integrate.py) `temp_integrate` | `params.AdamsBashforth_T` |
| [`model/src/thermodynamics.py`](../mitjax/model/src/thermodynamics.py) `thermodynamics` | `ptr_on and params.PTRACERS_calcSurfCor` |
| [`model/src/timestep.py`](../mitjax/model/src/timestep.py) `timestep` | `params.selectSigmaCoord_ne_0` |
| [`model/src/tracers_correction_step.py`](../mitjax/model/src/tracers_correction_step.py) `tracers_correction_step` | `getattr(cfg.cpp, opt, False) and cfg.use_flag(use)` |
| [`model/src/update_cg2d.py`](../mitjax/model/src/update_cg2d.py) `update_cg2d` | `not isinstance(myIter, int)` |
| [`model/src/write_pickup.py`](../mitjax/model/src/write_pickup.py) `write_pickup` | `_cpp(cfg, opt)` |
| [`ops/fortran_minmax.py`](../mitjax/ops/fortran_minmax.py) `build_winner` | `defaults is None or site not in defaults` |
| [`ops/fortran_minmax.py`](../mitjax/ops/fortran_minmax.py) `build_winner` | `len(hits) > 1 or strict()` |
| [`pkg/autodiff/adjoint_monitor.py`](../mitjax/pkg/autodiff/adjoint_monitor.py) `admonitor` | `mon_AdVarExch != 2` |
| [`pkg/autodiff/autodiff_readparms.py`](../mitjax/pkg/autodiff/autodiff_readparms.py) `adjoint_mode_check` | `diff` |
| [`pkg/cal/cal_timeinterval.py`](../mitjax/pkg/cal/cal_timeinterval.py) `cal_TimeInterval` | `timeunit == "model"` |
| [`pkg/cd_code/cd_code_read_pickup.py`](../mitjax/pkg/cd_code/cd_code_read_pickup.py) `cd_code_read_pickup` | `not ip.pickup_read_mdsio` |
| [`pkg/cost/cost_atlantic_heat.py`](../mitjax/pkg/cost/cost_atlantic_heat.py) `cost_atlantic_heat` | `sz.nPx != 1 or sz.nPy != 1` |
| [`pkg/cost/cost_final.py`](../mitjax/pkg/cost/cost_final.py) `cost_final` | `getattr(cfg.cpp, flag)` |
| [`pkg/ctrl/ctrl_get_gen.py`](../mitjax/pkg/ctrl/ctrl_get_gen.py) `ctrl_get_gen` | `(tauu or tauv) and not isinstance(gencount0, (int, np.integer))` |
| [`pkg/ctrl/ctrl_get_gen_rec.py`](../mitjax/pkg/ctrl/ctrl_get_gen_rec.py) `ctrl_get_gen_rec_cal` | `xx_genperiod == -12.0` |
| [`pkg/ctrl/ctrl_get_mask.py`](../mitjax/pkg/ctrl/ctrl_get_mask.py) `ctrl_get_mask2d` | `fstr_prefix(xx_filename, 7, "xx_tauu") or fstr_prefix(xx_filename, 7, "xx_tauv")` |
| [`pkg/ctrl/ctrl_init_ctrlvar.py`](../mitjax/pkg/ctrl/ctrl_init_ctrlvar.py) `ctrl_init_ctrlvar` | `varType in ("SecXZ", "SecYZ")` |
| [`pkg/ctrl/ctrl_map_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_genarr.py) `_flags` | `dolog10ctrl` |
| [`pkg/ctrl/ctrl_map_ini_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_ini_genarr.py) `ctrl_map_ini_genarr` | `igen_etan > 0` |
| [`pkg/ctrl/ctrl_map_ini_genarr.py`](../mitjax/pkg/ctrl/ctrl_map_ini_genarr.py) `ctrl_map_ini_genarr` | `not fstr_blank(g.weight) and any(fstr_prefix(g.file, len(n), n) for n in ( "xx_shicoeff", "xx_shicdrag", "xx_…` |
| [`pkg/ctrl/ctrl_map_ini_gentim2d.py`](../mitjax/pkg/ctrl/ctrl_map_ini_gentim2d.py) `ctrl_map_ini_gentim2d` | `p` |
| [`pkg/ctrl/ctrl_map_ini_gentim2d.py`](../mitjax/pkg/ctrl/ctrl_map_ini_gentim2d.py) `ctrl_map_ini_gentim2d` | `p in ("WC01", "smooth", "noscaling", "docycle", "variaweight", "rmcycle")` |
| [`pkg/ctrl/ctrl_map_ini_gentim2d.py`](../mitjax/pkg/ctrl/ctrl_map_ini_gentim2d.py) `ctrl_map_ini_gentim2d` | `replicated_ntimes > 0` |
| [`pkg/ctrl/ctrl_readparms.py`](../mitjax/pkg/ctrl/ctrl_readparms.py) `ctrl_readparms_genarr` | `k not in allowed` |
| [`pkg/ctrl/ctrl_readparms.py`](../mitjax/pkg/ctrl/ctrl_readparms.py) `ctrl_readparms_gentim2d` | `k not in {s.lower() for s in _SET}` |
| [`pkg/ctrl/ctrl_toolbox.py`](../mitjax/pkg/ctrl/ctrl_toolbox.py) `ctrl_cprsrs` | `nzOut != 1` |
| [`pkg/ctrl/rotate_uv2en_standin.py`](../mitjax/pkg/ctrl/rotate_uv2en_standin.py) `rotate_uv2en_rl` | `usingPCoords` |
| [`pkg/ctrl/rotate_uv2en_standin.py`](../mitjax/pkg/ctrl/rotate_uv2en_standin.py) `rotate_uv2en_rl` | `xy2en or not switchGrid or kSize != 1` |
| [`pkg/down_slope/dwnslp_apply.py`](../mitjax/pkg/down_slope/dwnslp_apply.py) `dwnslp_apply` | `not usingZCoords` |
| [`pkg/down_slope/dwnslp_calc_flow.py`](../mitjax/pkg/down_slope/dwnslp_calc_flow.py) `dwnslp_calc_flow` | `usingPCoords` |
| [`pkg/down_slope/dwnslp_init_fixed.py`](../mitjax/pkg/down_slope/dwnslp_init_fixed.py) `dwnslp_init_fixed` | `usingPCoords` |
| [`pkg/down_slope/dwnslp_readparms.py`](../mitjax/pkg/down_slope/dwnslp_readparms.py) `dwnslp_readparms` | `debugLevel >= debLevD` |
| [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `averages_table` | `not 1 <= genrec <= g.gencost_nrec` |
| [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `averages_table` | `sorted({rr for _, rr in t["writes"]}) != list(range(1, t["nrec"] + 1))` |
| [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `cost_gencost_assignperiod` | `ap in ("const", "CONST")` |
| [`pkg/ecco/cost_averagesfields.py`](../mitjax/pkg/ecco/cost_averagesfields.py) `customize_arm` | `_eq(_sub(bf, 9), "m_boxmean") or _eq(_sub(bf, 9), "m_horflux")` |
| [`pkg/ecco/cost_gencost_all.py`](../mitjax/pkg/ecco/cost_gencost_all.py) `_preproc_flags` | `pre not in ("", "mean", "offset", "mindepth") or pos != ""` |
| [`pkg/ecco/cost_gencost_all.py`](../mitjax/pkg/ecco/cost_gencost_all.py) `cost_gencal` | `localperiod == 86400.` |
| [`pkg/ecco/cost_gencost_all.py`](../mitjax/pkg/ecco/cost_gencost_all.py) `cost_gencost_all` | `pp["dooffset"]` |
| [`pkg/ecco/ecco_cost_init_fixed.py`](../mitjax/pkg/ecco/ecco_cost_init_fixed.py) `ecco_cost_init_fixed` | `ap in ("day", "DAY")` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `check_ecco_options` | `bad` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `_eq(_sub(bf, 9), "m_boxmean") or _eq(_sub(bf, 9), "m_horflux")` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `_eq(_sub(nm, 3), "moc")` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `_eq(_sub(nm, 6), "transp")` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `cost_iprec not in (32, 64)` |
| [`pkg/ecco/ecco_readparms.py`](../mitjax/pkg/ecco/ecco_readparms.py) `ecco_readparms` | `cost_nml` |
| [`pkg/exch2/exch2_cube_tables.py`](../mitjax/pkg/exch2/exch2_cube_tables.py) `_check_w2_options` | `on` |
| [`pkg/exch2/exch2_cube_tables.py`](../mitjax/pkg/exch2/exch2_cube_tables.py) `rx1_cube` | `w2.W2_myCommFlag[N, bi, bj] != "P"` |
| [`pkg/exch2/exch2_rx2_cube.py`](../mitjax/pkg/exch2/exch2_rx2_cube.py) `exch2_rx2_cube_tables` | `w2.W2_myCommFlag[N, bi, bj] != "P"` |
| [`pkg/exch2/exch2_rx2_cube.py`](../mitjax/pkg/exch2/exch2_rx2_cube.py) `exch2_uv_3d_rx_tables` | `useCubedSphereExchange` |
| [`pkg/exch2/exch2_rx2_cube.py`](../mitjax/pkg/exch2/exch2_rx2_cube.py) `exch2_uv_dgrid_3d_rx_tables` | `useCubedSphereExchange` |
| [`pkg/exch2/w2_eeboot.py`](../mitjax/pkg/exch2/w2_eeboot.py) `_check_build` | `links.get(name) != want` |
| [`pkg/exch2/w2_eeboot.py`](../mitjax/pkg/exch2/w2_eeboot.py) `w2_eeboot` | `io.myProcId != 0 or sz.nPx * sz.nPy != 1` |
| [`pkg/exch2/w2_map_procs.py`](../mitjax/pkg/exch2/w2_map_procs.py) `w2_map_procs` | `commFlag == "M"` |
| [`pkg/exch2/w2_readparms.py`](../mitjax/pkg/exch2/w2_readparms.py) `open_copy_data_file_echo` | `len(line) > MAX_LEN_PREC or "\t" in line` |
| [`pkg/exch2/w2_readparms.py`](../mitjax/pkg/exch2/w2_readparms.py) `w2_readparms` | `rp.has("data.exch2", "W2_EXCH2_PARM01", key)` |
| [`pkg/exch2/w2_set_f2f_index.py`](../mitjax/pkg/exch2/w2_set_f2f_index.py) `w2_set_f2f_index` | `w2.facet_link[i, j] == r4(0.0)` |
| [`pkg/exch2/w2_set_map_cumsum.py`](../mitjax/pkg/exch2/w2_set_map_cumsum.py) `w2_set_map_cumsum` | `fCnt < nActiveFacets` |
| [`pkg/exf/exf_bulkformulae.py`](../mitjax/pkg/exf/exf_bulkformulae.py) `exf_bulkformulae` | `cfg.cpp.flag(o, "EXF_OPTIONS.h")` |
| [`pkg/exf/exf_getclim.py`](../mitjax/pkg/exf/exf_getclim.py) `exf_getclim` | `cfg.cpp.flag(o, "EXF_OPTIONS.h")` |
| [`pkg/exf/exf_getffieldrec.py`](../mitjax/pkg/exf/exf_getffieldrec.py) `exf_GetFFieldRec` | `usefldyearlyfields` |
| [`pkg/exf/exf_getffields.py`](../mitjax/pkg/exf/exf_getffields.py) `exf_getffields` | `opt(o)` |
| [`pkg/exf/exf_init_varia.py`](../mitjax/pkg/exf/exf_init_varia.py) `exf_init_varia` | `cfg.cpp.flag(opt, "EXF_OPTIONS.h")` |
| [`pkg/exf/exf_mapfields.py`](../mitjax/pkg/exf/exf_mapfields.py) `exf_mapfields` | `fp.temp_EvPrRn_set` |
| [`pkg/exf/exf_monitor.py`](../mitjax/pkg/exf/exf_monitor.py) `exf_monitor` | `exf_cfg_flag(cfg, o)` |
| [`pkg/exf/exf_readparms.py`](../mitjax/pkg/exf/exf_readparms.py) `exf_readparms` | `v["useExfZenAlbedo"]` |
| [`pkg/exf/exf_set_fld.py`](../mitjax/pkg/exf/exf_set_fld.py) `exf_set_fld` | `useCAL and float(fldPeriod) == -1.` |
| [`pkg/exf/exf_set_fld.py`](../mitjax/pkg/exf/exf_set_fld.py) `exf_set_fld` | `useCAL and float(fldPeriod) == -12.` |
| [`pkg/exf/exf_wind.py`](../mitjax/pkg/exf/exf_wind.py) `exf_wind` | `gentim and ctrl is None` |
| [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `_scheme_x` | `advectionScheme in (ENUM_UPWIND_1RST, ENUM_DST2)` |
| [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `_scheme_y` | `advectionScheme in (ENUM_UPWIND_1RST, ENUM_DST2)` |
| [`pkg/generic_advdiff/gad_advection.py`](../mitjax/pkg/generic_advdiff/gad_advection.py) `gad_advection.level_r` | `vertAdvecScheme in (ENUM_UPWIND_1RST, ENUM_DST2, ENUM_FLUX_LIMIT)` |
| [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `diffK4_ne_0` |
| [`pkg/generic_advdiff/gad_calc_rhs.py`](../mitjax/pkg/generic_advdiff/gad_calc_rhs.py) `gad_calc_rhs` | `trUseDiffKr4` |
| [`pkg/generic_advdiff/gad_implicit_r.py`](../mitjax/pkg/generic_advdiff/gad_implicit_r.py) `gad_implicit_r.level_k` | `advectionScheme == ENUM_CENTERED_2ND` |
| [`pkg/generic_advdiff/gad_implicit_r.py`](../mitjax/pkg/generic_advdiff/gad_implicit_r.py) `gad_implicit_r.level_k` | `advectionScheme == ENUM_DST3_FLUX_LIMIT` |
| [`pkg/generic_advdiff/gad_init_varia.py`](../mitjax/pkg/generic_advdiff/gad_init_varia.py) `gad_init_varia` | `not (startTime == baseTime and nIter0 == 0 and str(pickupSuff).strip() == "")` |
| [`pkg/generic_advdiff/gad_som_adv_r.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_r.py) `gad_som_adv_r` | `not cfg.uniformFreeSurfLev and k != 1 and not noFlowAcrossSurf` |
| [`pkg/generic_advdiff/gad_som_adv_x.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_x.py) `gad_som_adv_x` | `overlapOnly or interiorOnly` |
| [`pkg/generic_advdiff/gad_som_adv_y.py`](../mitjax/pkg/generic_advdiff/gad_som_adv_y.py) `gad_som_adv_y` | `overlapOnly or interiorOnly` |
| [`pkg/generic_advdiff/gad_som_advect.py`](../mitjax/pkg/generic_advdiff/gad_som_advect.py) `gad_som_advect` | `advectionScheme % 10 != 0` |
| [`pkg/generic_advdiff/gad_u3c4_impl_r.py`](../mitjax/pkg/generic_advdiff/gad_u3c4_impl_r.py) `gad_u3c4_impl_r` | `flagC4` |
| [`pkg/ggl90/ggl90_calc.py`](../mitjax/pkg/ggl90/ggl90_calc.py) `ggl90_calc` | `params.usingPCoords` |
| [`pkg/ggl90/ggl90_h.py`](../mitjax/pkg/ggl90/ggl90_h.py) `declare` | `size.nPx != 1 or size.nPy != 1` |
| [`pkg/ggl90/ggl90_idemix.py`](../mitjax/pkg/ggl90/ggl90_idemix.py) `ggl90_idemix` | `params.usingPCoords` |
| [`pkg/ggl90/ggl90_init_varia.py`](../mitjax/pkg/ggl90/ggl90_init_varia.py) `ggl90_init_varia` | `nIter0 != 0 or not _blank(pickupSuff)` |
| [`pkg/ggl90/ggl90_mixinglength.py`](../mitjax/pkg/ggl90/ggl90_mixinglength.py) `ggl90_mixinglength` | `params.usingPCoords` |
| [`pkg/gmredi/gmredi_calc_psi_bolus.py`](../mitjax/pkg/gmredi/gmredi_calc_psi_bolus.py) `gmredi_calc_psi_bolus` | `getattr(cfg.cpp, opt)` |
| [`pkg/gmredi/gmredi_calc_tensor.py`](../mitjax/pkg/gmredi/gmredi_calc_tensor.py) `gmredi_calc_tensor` | `getattr(cpp, opt)` |
| [`pkg/gmredi/gmredi_h.py`](../mitjax/pkg/gmredi/gmredi_h.py) `declare` | `size.nPx != 1 or size.nPy != 1` |
| [`pkg/gmredi/gmredi_init_fixed.py`](../mitjax/pkg/gmredi/gmredi_init_fixed.py) `gmredi_init_fixed` | `getattr(gm, name).strip() != ""` |
| [`pkg/gmredi/gmredi_init_fixed.py`](../mitjax/pkg/gmredi/gmredi_init_fixed.py) `gmredi_init_fixed` | `getattr(gm, name).strip() != ""` |
| [`pkg/gmredi/gmredi_init_varia.py`](../mitjax/pkg/gmredi/gmredi_init_varia.py) `gmredi_init_varia` | `getattr(cfg.cpp, opt)` |
| [`pkg/gmredi/gmredi_slope_limit.py`](../mitjax/pkg/gmredi/gmredi_slope_limit.py) `gmredi_slope_limit` | `scheme == "ac02"` |
| [`pkg/gmredi/gmredi_slope_limit.py`](../mitjax/pkg/gmredi/gmredi_slope_limit.py) `gmredi_slope_limit` | `scheme == "fm07"` |
| [`pkg/gmredi/gmredi_slope_limit.py`](../mitjax/pkg/gmredi/gmredi_slope_limit.py) `gmredi_slope_limit` | `scheme == "stableGmAdjTap"` |
| [`pkg/gmredi/gmredi_slope_limit.py`](../mitjax/pkg/gmredi/gmredi_slope_limit.py) `gmredi_slope_limit` | `scheme in ("orig", "clipping")` |
| [`pkg/gmredi/gmredi_slope_psi.py`](../mitjax/pkg/gmredi/gmredi_slope_psi.py) `gmredi_slope_psi` | `scheme == "ldd97"` |
| [`pkg/gmredi/gmredi_slope_psi.py`](../mitjax/pkg/gmredi/gmredi_slope_psi.py) `gmredi_slope_psi` | `scheme == "stableGmAdjTap"` |
| [`pkg/gmredi/gmredi_slope_psi.py`](../mitjax/pkg/gmredi/gmredi_slope_psi.py) `gmredi_slope_psi` | `scheme in ("orig", "clipping")` |
| [`pkg/gmredi/gmredi_write_pickup.py`](../mitjax/pkg/gmredi/gmredi_write_pickup.py) `gmredi_write_pickup` | `on` |
| [`pkg/grdchk/grdchk.py`](../mitjax/pkg/grdchk/grdchk.py) `grdchk_get_mask` | `ncvargrd != "c"` |
| [`pkg/grdchk/grdchk.py`](../mitjax/pkg/grdchk/grdchk.py) `grdchk_loc` | `ncvargrd != "c"` |
| [`pkg/grdchk/grdchk_get_position.py`](../mitjax/pkg/grdchk/grdchk_get_position.py) `grdchk_get_position` | `ncvargrd != "c"` |
| [`pkg/kpp/kpp_calc.py`](../mitjax/pkg/kpp/kpp_calc.py) `kpp_calc` | `cfg.cpp.flag(opt, "KPP_OPTIONS.h")` |
| [`pkg/kpp/kpp_calc.py`](../mitjax/pkg/kpp/kpp_calc.py) `kpp_calc` | `not kpp.kpp_freq_eq_deltaTClock` |
| [`pkg/kpp/kpp_routines.py`](../mitjax/pkg/kpp/kpp_routines.py) `_salt_plume_args` | `cfg.cpp.flag(opt, "KPP_OPTIONS.h")` |
| [`pkg/kpp/kpp_routines.py`](../mitjax/pkg/kpp/kpp_routines.py) `blmix` | `cfg.cpp.flag(opt, "KPP_OPTIONS.h")` |
| [`pkg/kpp/kpp_routines.py`](../mitjax/pkg/kpp/kpp_routines.py) `ri_iwmix` | `cfg.cpp.flag(opt, "KPP_OPTIONS.h")` |
| [`pkg/mdsio/mdsio_write_field.py`](../mitjax/pkg/mdsio/mdsio_write_field.py) `mds_wr_metafiles` | `mds.w2 is None` |
| [`pkg/mdsio/mdsio_write_field.py`](../mitjax/pkg/mdsio/mdsio_write_field.py) `mds_write_field` | `arrType != "RL"` |
| [`pkg/mdsio/mdsio_write_field.py`](../mitjax/pkg/mdsio/mdsio_write_field.py) `mds_write_field` | `mds.w2 is None` |
| [`pkg/mom_common/mom_calc_hfacz.py`](../mitjax/pkg/mom_common/mom_calc_hfacz.py) `mom_calc_hfacz` | `hZoption == 1` |
| [`pkg/mom_common/mom_calc_hfacz.py`](../mitjax/pkg/mom_common/mom_calc_hfacz.py) `mom_calc_hfacz` | `hZoption == 2` |
| [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `mom_calc_visc` | `cfg.cpp.flag(name, _AD_OPT)` |
| [`pkg/mom_common/mom_calc_visc.py`](../mitjax/pkg/mom_common/mom_calc_visc.py) `mom_calc_visc` | `cfg.cpp.flag(name, _OPT)` |
| [`pkg/mom_common/mom_implicit_r.py`](../mitjax/pkg/mom_common/mom_implicit_r.py) `_implicit_r` | `diagonalNumber == 1` |
| [`pkg/mom_common/mom_init_fixed.py`](../mitjax/pkg/mom_common/mom_init_fixed.py) `mom_init_fixed` | `cfg.cpp.flag(name, _OPT)` |
| [`pkg/mom_common/mom_quasihydrostatic.py`](../mitjax/pkg/mom_common/mom_quasihydrostatic.py) `mom_quasihydrostatic` | `params.fluidIsWater` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `writeDiag` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `writeDiag` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `writeDiag` |
| [`pkg/mom_vecinv/mom_vecinv.py`](../mitjax/pkg/mom_vecinv/mom_vecinv.py) `mom_vecinv` | `writeDiag` |
| [`pkg/mom_vecinv/mom_vi_hdissip.py`](../mitjax/pkg/mom_vecinv/mom_vi_hdissip.py) `mom_vi_hdissip` | `diagnostics` |
| [`pkg/mom_vecinv/mom_vi_hdissip.py`](../mitjax/pkg/mom_vecinv/mom_vi_hdissip.py) `mom_vi_hdissip` | `diagnostics` |
| [`pkg/monitor/mon_vort3.py`](../mitjax/pkg/monitor/mon_vort3.py) `mon_vort3` | `cube and getattr(cfg, "cs_vort3_corners", None) is None` |
| [`pkg/monitor/mon_vort3.py`](../mitjax/pkg/monitor/mon_vort3.py) `mon_vort3` | `north.any() or south.any()` |
| [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `k_surface` | `params.fluidIsAir` |
| [`pkg/ptracers/ptracers_apply_forcing.py`](../mitjax/pkg/ptracers/ptracers_apply_forcing.py) `k_surface` | `params.usingPCoords` |
| [`pkg/ptracers/ptracers_forcing_surf.py`](../mitjax/pkg/ptracers/ptracers_forcing_surf.py) `check_unported` | `ptr.PTRACERS_StepFwd[iTrc-1] and ptr.PTRACERS_EvPrRn_ne_UNSET[iTrc-1]` |
| [`pkg/ptracers/ptracers_forcing_surf.py`](../mitjax/pkg/ptracers/ptracers_forcing_surf.py) `routine_of_build` | `mod is None` |
| [`pkg/ptracers/ptracers_integrate.py`](../mitjax/pkg/ptracers/ptracers_integrate.py) `ptracers_integrate` | `ptr.PTRACERS_AdamsBash_Tr[n]` |
| [`pkg/ptracers/ptracers_readparms.py`](../mitjax/pkg/ptracers/ptracers_readparms.py) `ptracers_readparms` | `_elems(rp, name) or rp.has(_F, _G, name)` |
| [`pkg/ptracers/ptracers_readparms.py`](../mitjax/pkg/ptracers/ptracers_readparms.py) `ptracers_readparms` | `num > 99` |
| [`pkg/ptracers/ptracers_reset.py`](../mitjax/pkg/ptracers/ptracers_reset.py) `ptracers_reset` | `any(getattr(ptr, "PTRACERS_resetFreq_gt_0", ()))` |
| [`pkg/ptracers/ptracers_switch_onoff.py`](../mitjax/pkg/ptracers/ptracers_switch_onoff.py) `ptracers_switch_onoff` | `not ptr.PTRACERS_startAllTrc` |
| [`pkg/seaice/dynsolver.py`](../mitjax/pkg/seaice/dynsolver.py) `dynsolver` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h") != want` |
| [`pkg/seaice/dynsolver.py`](../mitjax/pkg/seaice/dynsolver.py) `kgeo_level` | `k.size != 1` |
| [`pkg/seaice/ostres.py`](../mitjax/pkg/seaice/ostres.py) `ostres` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h") != want` |
| [`pkg/seaice/seaice_advdiff.py`](../mitjax/pkg/seaice/seaice_advdiff.py) `_advdiff_single_dim` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_advdiff.py`](../mitjax/pkg/seaice/seaice_advdiff.py) `seaice_advdiff` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_advection.py`](../mitjax/pkg/seaice/seaice_advection.py) `seaice_advection` | `advectionScheme not in (ENUM_FLUX_LIMIT, ENUM_DST3, ENUM_DST3_FLUX_LIMIT, ENUM_OS7MP) and not ppm` |
| [`pkg/seaice/seaice_advection.py`](../mitjax/pkg/seaice/seaice_advection.py) `seaice_advection` | `tamc and getattr(sp, "mjx_tamc_maxpass", None) is None` |
| [`pkg/seaice/seaice_calc_viscosities.py`](../mitjax/pkg/seaice/seaice_calc_viscosities.py) `seaice_calc_viscosities` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h") or cfg.cpp.flag(o)` |
| [`pkg/seaice/seaice_cost_test.py`](../mitjax/pkg/seaice/seaice_cost_test.py) `seaice_cost_test` | `flag in (3, 4, 5, 6, 7)` |
| [`pkg/seaice/seaice_dynsolver.py`](../mitjax/pkg/seaice/seaice_dynsolver.py) `_dynamics_forcing` | `op.usingPCoords` |
| [`pkg/seaice/seaice_get_dynforcing.py`](../mitjax/pkg/seaice/seaice_get_dynforcing.py) `seaice_get_dynforcing` | `op.usingPCoords` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag(o) != want` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag(o, "EXF_OPTIONS.h") != want` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag(o, "SALT_PLUME_OPTIONS.h")` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `op.usingPCoords` |
| [`pkg/seaice/seaice_growth.py`](../mitjax/pkg/seaice/seaice_growth.py) `seaice_growth` | `w != ("a", "b")` |
| [`pkg/seaice/seaice_init_fixed.py`](../mitjax/pkg/seaice/seaice_init_fixed.py) `_sitracer_specs` | `name in ("salinity", "ridge")` |
| [`pkg/seaice/seaice_init_fixed.py`](../mitjax/pkg/seaice/seaice_init_fixed.py) `seaice_init_fixed` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_init_varia.py`](../mitjax/pkg/seaice/seaice_init_varia.py) `seaice_init_varia` | `changed` |
| [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `_check_back_loop` | `iMs != list(range(hi-1, lo-1, -1))` |
| [`pkg/seaice/seaice_lsr.py`](../mitjax/pkg/seaice/seaice_lsr.py) `seaice_lsr` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `op.usingPCoords` |
| [`pkg/seaice/seaice_model.py`](../mitjax/pkg/seaice/seaice_model.py) `seaice_model` | `salt_plume is not None` |
| [`pkg/seaice/seaice_read_pickup.py`](../mitjax/pkg/seaice/seaice_read_pickup.py) `seaice_read_pickup` | `nbFields <= 0` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `_cost_parm02` | `fname == _F and group == g2.lower() and exp.run.is_set(fname, group, key) and key not in known` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `_sitracer_parm03` | `fname == _F and group == g3.lower() and exp.run.is_set(fname, group, key) and key not in known` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `fname == _F and group == _G.lower() and exp.run.is_set(fname, group, key) and key not in known` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `not v["SEAICEuseLSR"]` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `tau > 0.0` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v[f].strip()` |
| [`pkg/seaice/seaice_readparms.py`](../mitjax/pkg/seaice/seaice_readparms.py) `seaice_readparms` | `v[n]` |
| [`pkg/seaice/seaice_reg_ridge.py`](../mitjax/pkg/seaice/seaice_reg_ridge.py) `seaice_reg_ridge` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_solve4temp.py`](../mitjax/pkg/seaice/seaice_solve4temp.py) `seaice_solve4temp` | `cfg.cpp.flag(o, "SEAICE_OPTIONS.h")` |
| [`pkg/seaice/seaice_tracer_phys.py`](../mitjax/pkg/seaice/seaice_tracer_phys.py) `seaice_tracer_phys` | `name in ("salinity", "ridge")` |
| [`pkg/seaice/seaice_write_pickup.py`](../mitjax/pkg/seaice/seaice_write_pickup.py) `seaice_write_pickup` | `sp.SEAICEuseEVP` |
| [`verification/tutorial_global_oce_latlon/code/ptracers_forcing_surf.py`](../mitjax/verification/tutorial_global_oce_latlon/code/ptracers_forcing_surf.py) `ptracers_forcing_surf` | `params.usingPCoords` |

<!-- END GENERATED: supported options -->
