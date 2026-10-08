# SUBSTEPS — dump points of the Fortran oracle on MITgcm master 63cdc0b (plan Task 4)

The stage table at the end is written by `python3 reference/jaxdump/instrument.py --git $MJX_UPSTREAM --markdown`
from the `STAGES` list in `reference/jaxdump/instrument.py` (the same list inserts the calls; edit there, then
regenerate; `scripts/tests/test_jaxdump.py::test_substeps_md_in_sync` checks it). Copied from the ECCO port's
`reference/jaxdump/` (R = `mit_jax_private`, branch `handoff-2026-09-private`) and adapted to master and to the M1
experiments; every change is listed below.

## Use
- Build: `sbatch -J mjx_build_<tag> reference/jobs/build.sbatch <commit> <exp>:<code>:jaxdump ...` (reference/build.sh
  step 1b: instrumented copies of `model/src` files + `jaxdump.F` + `JAXDUMP.h` in `BUILD/jaxdump_mods`, given to
  genmake2 in front of the experiment's code directory; the first directory holding a file wins,
  `tools/genmake2:2945-2961`). The frozen binary directory holds `instrument_report.txt`, the instrumented sources
  (`jaxdump_mods/`) and `jaxdump_stages.txt` (the stages compiled into this binary, read from the preprocessed
  `bld/*.f`).
- Run with `JAXDUMP_DIR=<dir>` and optionally `JAXDUMP_STEPS=a:b:c` (`:` or `,`; `,` does not survive
  `sbatch --export`; default nIter0, nIter0+1, nIter0+2). Without `JAXDUMP_DIR` every entry point returns at once.
  Nothing is written to STDOUT even with dumps on (settings go to `<dir>/jaxdump_info.txt`).
- Read with `mitjax.io.dump.DumpSet`; compare with `tools/diffdump.py` (two dump sets) or `tools/step_vs_dump.py`
  (a candidate step against the oracle, first differing stage).
- Invisibility: `sbatch -J mjx_jdrun reference/jobs/jaxdump_runs.sbatch <commit> EXP INPUT PLAIN_BIN JD_BIN ...`
  (plain binary, instrumented with dumps off, instrumented with dumps on; `reference/jaxdump/invisibility.py`).

## Record format (version 2) and the iteration of a record
Big-endian stream files `jd_<iter %010d>_t<tile %04d>.bin`; record = header (magic 1245990224, version 2, iter, seq,
stage char*32, field char*32, kind char*4, nz, sNx, sNy, OLx, OLy, tile, face, tBasex, tBasey) + float64 values with
halos. Kinds: `C W S Z` (point type), `N` scalar (constant field, `DumpSet.scalar`), `V` vertical 1-D array (one
record of n levels on the first tile). exch2 builds write the W2 tile, facet and offsets; exch1 builds write face 0,
`tBasex = myXGlobalLo-1+(bi-1)*sNx`, `tBasey = myYGlobalLo-1+(bj-1)*sNy`, tile `1+tBasex/sNx+(tBasey/sNy)*nSx*nPx`.

Every record of a step carries the step's START iteration. Master advances the counter right after DYNAMICS
(`myIter = nIter0 + iLoop`, `model/src/forward_step.F:807`; c66g `:823`). THERMODYNAMICS runs before DYNAMICS when
`staggerTimeStep=.FALSE.` (`forward_step.F:733`) and after the update when `.TRUE.` (`:1005`), so the ECCO port's
static `myIter-1` list cannot serve both. The shim keeps a runtime shift instead: `JAXDUMP_ITERSHIFT(0)` before stage
S00 and `JAXDUMP_ITERSHIFT(1)` right after the update line; records carry `myIter - shift`. `instrument.py` keeps the
ECCO assert: the update line exists once, after `CALL DYNAMICS(` and before `CALL UPDATE_R_STAR( .TRUE.`, and both
shift calls are outside any CPP conditional (planted moves fail: `test_planted_iteration_update_moved_fails`).
`E01_cost_final` (after the time loop) therefore carries the start iteration of the LAST step.

Repeated keys: a key written twice into one tile file (a stage in a loop without a pass suffix, or the forward runs
of grdchk) is kept per occurrence (`DumpSet.tiles(..., occ=n)`, `repeated_keys()`); the ECCO reader kept only the
last record of a repeated key.

## Changes from R (ECCO port, c66g / V4r4)
- **Anchors re-validated on master** (`test_anchor_counts_on_master`): every anchor has its expected count at
  `63cdc0b`. R's sea-ice and EXF stages (five sea-ice anchors drifted, Task 7a) are dropped with their dump groups
  `i u y v h I b e`, and the M1-unused groups `p` (salt plume) and `x` (EXF); re-add from R with a re-audit at M2/M4.
- **S00/G00 anchor**: R inserted after `CALL AUTODIFF_INADMODE_UNSET(`, which plain builds do not compile (inside
  `#ifdef ALLOW_AUTODIFF`, `forward_step.F:433-435`); new anchor mode `CPPEND:` inserts after that block's `#endif`.
  Each insertion point's CPP context is listed in the table ("compiled only if").
- **Stage lists per package** (`PACKAGES` in instrument.py): init, core, gad, gmredi, ggl90, monitor, sbo, ctrl, cost,
  exch. All instrumented files are in `model/src` (always compiled), so no copy pulls an unused package into a build;
  package fields are written by `jaxdump.F` under `#ifdef ALLOW_<PKG>`. New stages: `S05_thermodynamics_sync`
  (non-staggered THERMODYNAMICS), `P02` after the k loop (R's anchor `CALC_OCE_MXLAYER` runs only with GM or
  `doDiagsRho`), `P03_mxlayer`, `D00b_mom_fluxform`, `S16_blocking_exchanges`, `S17_monitor`, `S18_cost_tile`,
  `S19_sbo_calc`, `E01_cost_final`, `X00_exch_probe` (split from G00); `T13/T23` moved after the implicit block
  (GAD_IMPLICIT_R exists only with `INCLUDE_IMPLVERTADV_CODE`; IMPLDIFF otherwise); `C02` adds the cg2d iteration
  count and residuals as `N:` scalars. Groups new in jaxdump.F: `o` (SBO scalars), `q` (cost: fc, glofc, tile_fc,
  objf_hflux_tut, objf_temp_tut).
- **V group**: R packed the 1-D vertical grid into the rows of one tile-shaped record, assuming `sNx+OLx > Nr` and
  `sNy+OLy >= 15` (LLC90 tiles). On advect_xz (sNx=10, sNy=1, OLx=OLy=4, Nr=20) it wrote past its automatic buffer
  and the dumps-on run crashed (job 27826794, SIGSEGV); on advect_xy and global_ocean it silently wrote rows into
  the wrong tile slot. Now one kind-V record per array (`JAXDUMP_VEC`).
- **Exchange probe** (`JAXDUMP_EXCH_PROBE`, group X): code `comp*cbase + ((tile-1)*(sNy+2OLy) + ja)*(sNx+2OLx) +
  ia + 1` with `cbase` the smallest power of 10 above `nTiles*(sNy+2OLy)*(sNx+2OLx)` (dumped as `xCbase`,
  `xNtiles`); R's `1e6 + tile*1e4 + (j+OLy)*100 + (i+OLx)` needed `sNx+2OLx < 100` and <= 99 tiles. Halos are coded
  too (halo-sourced copies are visible). New: signed-zero probes `zp*`/`zm*` (fields of +0 and of -0 through every
  exchange that takes `withSigns=.TRUE.`). exch1 tile numbers as in the record header. Decoder:
  `mitjax.io.dump.probe_decode`.
- **Mixed signed-zero pairs** (new): the vector exchanges also run on u=-0, v=+0 (records `zu*`) and u=+0, v=-0
  (`zv*`). exch2 forms every buffer value as `sa1*array1 + sa2*array2` at the source index
  (`pkg/exch2/exch2_put_rx2.template:229-230` u buffer with `sa1=pi(1)`, `sa2=pj(1)`; `:317-318` v buffer with
  `sa1=pi(2)`, `sa2=pj(2)`; no rotation: `1*u + 0*v` and `0*u + 1*v`), so a -0 of one component arrives as +0 where
  the other component's value at the same index is +0 or positive (`-0 + +0 = +0`), and as NaN where it is Inf/NaN.
  Equal pairs (`zp*`, `zm*`) cannot show this; it is the cause of the global_ocean pickup-halo ±0 (lane A handoff).
- **Initialisation stages** (new, stage list `init`): `I00_pickup_read` (READ_PICKUP after the reads, before its
  exchanges), `I01_read_pickup` (after READ_PICKUP), `I02_ini_fields` (after INI_FIELDS), groups `dta`; records
  carry nIter0 (option `iter`: INITIALISE_VARIA and INI_FIELDS have no myIter). I00/I01 exist only in restarts.
- **rStarDhCDt in group `r`** (new, NONLIN_FRSURF): at `S12_calc_rstar` it is the value the step's MONITOR call
  reads (lane MON's request; before, only group `R` of G00 held it).
- **No STDOUT output** with dumps on (R printed a `JAXDUMP:` line via PRINT_MESSAGE).
- **exch1 coordinates** in the header (R wrote face 0, tBase 0 without exch2); header version 1 -> 2.

## c66g -> master audit of the instrumented routines and the probed exchanges
`python3 scripts/audit_upstream.py --routines FORWARD_STEP DO_OCEANIC_PHYS DYNAMICS SOLVE_FOR_PRESSURE THERMODYNAMICS
TEMP_INTEGRATE SALT_INTEGRATE THE_MAIN_LOOP DO_THE_MODEL_IO EXCH_XY_RX EXCH_UV_XY_RX EXCH_3D_RX EXCH_UV_3D_RX
EXCH_Z_3D_RX EXCH_UV_AGRID_3D_RX EXCH_UV_BGRID_3D_RX EXCH_UV_DGRID_3D_RX EXCH_SM_3D_RX EXCH_S3D_RX EXCH1_RX`
(2026-10-01; no hunk undecided). The shim copies none of the c66g text of these routines: it instruments master's
files, so what matters is anchor drift (none, see above) and that the dumped variables exist in master's headers
(all do except R's `kapGM`/`kapRedi` control fields, dropped). Forward-value hunks change the physics the JAX port
reads from master directly (M1+); listed per file:

| file | status | hunks | classes | forward-value hunks |
|---|---|---|---|---|
| `model/src/forward_step.F` | modified | 31 | forward value 8, AD-only 18, diagnostics 2, cosmetic 3 | #2 #3 #12 #16 #17 #19 #20 #30 |
| `model/src/do_oceanic_phys.F` | modified | 37 | forward value 12, AD-only 17, cosmetic 8 | #1 #12 #17 #18 #20 #23 #24 #30 #31 #34 #35 #37 |
| `model/src/dynamics.F` | modified | 21 | forward value 6, AD-only 10, cosmetic 5 | #1 #8 #11 #15 #20 #21 |
| `model/src/solve_for_pressure.F` | modified | 9 | forward value 4, AD-only 1, diagnostics 1, cosmetic 3 | #5 #7 #8 #9 |
| `model/src/thermodynamics.F` | modified | 9 | forward value 2, AD-only 6, cosmetic 1 | #2 #7 |
| `model/src/temp_integrate.F` | modified | 17 | forward value 3, AD-only 13, cosmetic 1 | #9 #10 #17 |
| `model/src/salt_integrate.F` | modified | 16 | forward value 2, AD-only 13, cosmetic 1 | #9 #10 |
| `model/src/the_main_loop.F` | modified | 20 | forward value 6, AD-only 12, cosmetic 2 | #1 #2 #14 #18 #19 #20 |
| `model/src/do_the_model_io.F` | modified | 9 | forward value 5, cosmetic 4 | #5 #6 #7 #8 #9 |
| `eesupp/src/exch_xy_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_uv_xy_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_uv_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_z_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_uv_agrid_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_uv_bgrid_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_uv_dgrid_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_sm_3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch_s3d_rx.template` | modified | 1 | cosmetic 1 | - |
| `eesupp/src/exch1_rx.template` | modified | 1 | cosmetic 1 | - |

The exchange templates changed only cosmetically (the CVS header lines); the probe measures master's behaviour
directly. The headers `SIZE.h`, `EEPARAMS.h`, `PARAMS.h`, `GRID.h`, `DYNVARS.h`, `SURFACE.h`, `FFIELDS.h`, `CG2D.h`
are audited in `docs/AUDIT_C66G_MASTER.md` (module `jaxdump`); `jaxdump.F` compiled against them in all six M1
builds (build jobs 27826745, 27826872).

## Stages (generated)
| stage | package list | anchor (master `63cdc0b`) | compiled only if | scope | dumps | what |
|---|---|---|---|---|---|---|
| `(iteration shift 1)` | core | `model/src/forward_step.F:807` | always | - |  | myIter = nIter0 + iLoop: later records carry myIter-1 |
| `I00_pickup_read` | init | before `model/src/read_pickup.F:538` | always | all | S:dta | pickup fields as read (interior from the file; halos still those of INI_DYNVARS), before READ_PICKUP's exchanges (read_pickup.F:538-567) |
| `I01_read_pickup` | init | `model/src/ini_fields.F:38` | always | all, iteration `nIter0` | S:dta | state after READ_PICKUP (its exchanges included) |
| `I02_ini_fields` | init | `model/src/initialise_varia.F:225` | always | all, iteration `nIter0` | S:dta | state after INI_FIELDS (cold start: INI_VEL ... INI_PRESSURE; restart: READ_PICKUP) |
| `S00_begin` | core | after the `#endif` around `model/src/forward_step.F:434` | always | all | S:dtarfmkgcP | state at the start of the step (after the AD-only iteration reset and AUTODIFF_INADMODE_UNSET, forward_step.F:427-435, which plain builds do not compile) |
| `G00_geometry` | core | after the `#endif` around `model/src/forward_step.F:434` | always | all | S:GVR | grid, masks, 3-D mixing parameters, packed vertical grid, r* fields |
| `X00_exch_probe` | exch | after the `#endif` around `model/src/forward_step.F:434` | always | all | S:X | exchange probe (jaxdump.F JAXDUMP_EXCH_PROBE): every exchange routine on index-coded and +0/-0 fields |
| `S01_update_rstar_F` | core | `model/src/forward_step.F:475` | `#ifdef NONLIN_FRSURF`; `# ifndef DISABLE_RSTAR_CODE` | all | S:rd | RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) (select_rStar > 0) |
| `X01_exf_getffields` | exf | `pkg/exf/exf_getforcing.F:199` | always | all | S:x | EXF_GETFFIELDS: forcing records read and time-interpolated (A-grid stress not yet exchanged) |
| `X02_exf_radiation` | exf | `pkg/exf/exf_getforcing.F:262` | `#ifdef ALLOW_DOWNWARD_RADIATION` | all | S:x | EXF_RADIATION: lwflux (emissivity, surface temperature), swflux |
| `X03_exf_wind` | exf | `pkg/exf/exf_getforcing.F:266` | always | all | S:x | EXF_WIND: wind speed and stress (wStress, cw, sw, sh; uwind/vwind from the stress when useAtmWind=F) |
| `X04_exf_bulkformulae` | exf | `pkg/exf/exf_getforcing.F:281` | `#ifdef ALLOW_ATM_TEMP`; `# ifdef ALLOW_BULKFORMULAE` | all | S:x | EXF_BULKFORMULAE: hs, hl, evap, the stress over open ocean |
| `X05_exf_hflux_sflux` | exf | before `pkg/exf/exf_getforcing.F:344` | always | all | S:x | hflux, sflux (net heat and fresh-water fluxes, runoff, masks) and the stress exchange |
| `X06_exf_mapfields` | exf | `pkg/exf/exf_getforcing.F:383` | always | all | S:xf | EXF_MAPFIELDS: fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad (end of EXF_GETFORCING) |
| `S02_load_fields` | core | `model/src/forward_step.F:540` | always | all | S:f | external forcing fields read and time-interpolated |
| `S03_ctrl_map_forcing` | ctrl | `model/src/forward_step.F:573` | `#ifdef ALLOW_CTRL` | all | S:f | time-varying controls added to the forcing (useCTRL) |
| `P12_thsice_main` | thsice | `model/src/do_oceanic_phys.F:402` | `#if (defined ALLOW_THSICE) && !(defined ALLOW_ATM2D)` | all | S:hf | THSICE_MAIN: thermodynamic sea ice (pkg/thsice) and the ocean forcing it modifies |
| `I00_seaice_begin` | seaice | before `pkg/seaice/seaice_model.F:186` | `#ifdef SEAICE_BGRID_DYNAMICS [#else]` | all | S:Iix | SEAICE_MODEL inputs: masks and metric terms, ice state, EXF fields (uwind/vwind after EXCH_UV_AGRID_3D_RL) |
| `I00b_seaice_begin` | seaice | before `pkg/seaice/seaice_model.F:182` | `#ifdef SEAICE_BGRID_DYNAMICS` | all | S:IixB | SEAICE_MODEL inputs, B-grid build (SEAICE_BGRID_DYNAMICS): masks, ice state, B-grid fields, EXF fields |
| `I01b_dynsolver` | seaice | `pkg/seaice/seaice_model.F:182` | `#ifdef SEAICE_BGRID_DYNAMICS` | all | S:iyBf | DYNSOLVER (B grid, pkg/seaice/dynsolver.F): forcing, PRESS0/zMax/zMin, LSR if SEAICEuseDYNAMICS, then OSTRES (fu, fv under ice) in every case |
| `Y01_get_dynforcing` | seaice | `pkg/seaice/seaice_dynsolver.F:130` | always | all | G:TAUX:W:1, G:TAUY:S:1 | SEAICE_GET_DYNFORCING: wind stress on the ice TAUX, TAUY (routine-local all-tile arrays) |
| `Y02_ice_strength` | seaice | `pkg/seaice/seaice_dynsolver.F:299` | `#ifdef SEAICE_CGRID` | tile | S:y | ice mass, FORCEX0/Y0 (stress + tilt) and SEAICE_CALC_ICE_STRENGTH: PRESS0, SEAICE_zMax, SEAICE_zMin |
| `Y03_freedrift` | seaice | `pkg/seaice/seaice_dynsolver.F:307` | `#ifdef SEAICE_CGRID`; `# ifdef SEAICE_ALLOW_FREEDRIFT` | all | S:y | SEAICE_FREEDRIFT: uice_fd, vice_fd (SEAICEuseFREEDRIFT, EVP or LSR_mixIniGuess) |
| `Y04_solver_inputs` | seaice | after the `#endif` around `pkg/seaice/seaice_dynsolver.F:327` | `#ifdef SEAICE_CGRID` | all | S:iy | all inputs of the momentum solver (EVP, LSR, Krylov or JFNK) |
| `Y05_evp` | seaice | `pkg/seaice/seaice_dynsolver.F:340` | `#ifdef SEAICE_CGRID`; `# ifdef SEAICE_ALLOW_EVP` | all | S:iy | SEAICE_EVP result (EVP, mEVP, aEVP) |
| `Y06_lsr` | seaice | `pkg/seaice/seaice_dynsolver.F:346` | `#ifdef SEAICE_CGRID` | all | S:iy | SEAICE_LSR result |
| `Y07_krylov` | seaice | `pkg/seaice/seaice_dynsolver.F:355` | `#ifdef SEAICE_CGRID`; `# ifdef SEAICE_ALLOW_KRYLOV`; `#  ifdef ALLOW_AUTODIFF [#  else]` | all | S:iy | SEAICE_KRYLOV result |
| `Y08_jfnk` | seaice | `pkg/seaice/seaice_dynsolver.F:366` | `#ifdef SEAICE_CGRID`; `# ifdef SEAICE_ALLOW_JFNK`; `#  ifdef ALLOW_AUTODIFF [#  else]` | all | S:iy | SEAICE_JFNK result |
| `Y09_ocean_stress` | seaice | `pkg/seaice/seaice_dynsolver.F:384` | always | all | S:f | SEAICE_OCEAN_STRESS: fu, fv under ice (SEAICEupdateOceanStress) |
| `I01_dynsolver` | seaice | `pkg/seaice/seaice_model.F:186` | `#ifdef SEAICE_BGRID_DYNAMICS [#else]` | all | S:iyf | SEAICE_DYNSOLVER incl. velocity clipping (SEAICE_clipVelocities) |
| `I02a_thsice_advect` | thsice | `pkg/seaice/seaice_model.F:213` | `#ifdef ALLOW_THSICE` | all | S:h | THSICE_DO_ADVECT: pkg/thsice fields advected by the sea-ice velocity (useThSice) |
| `I02_advdiff` | seaice | `pkg/seaice/seaice_model.F:231` | `#ifdef SEAICE_BGRID_DYNAMICS [#else]` | all | S:i | SEAICE_ADVDIFF (C grid): HEFF, AREA, HSNOW (and ITD categories) advected and diffused |
| `I03_reg_ridge` | seaice | `pkg/seaice/seaice_model.F:249` | always | all | S:in | SEAICE_REG_RIDGE: negative-value and area regularisation (d_HEFFbyNEG, d_HSNWbyNEG) |
| `I04a_growth_adx` | seaice | `pkg/seaice/seaice_model.F:272` | `#ifdef DISABLE_SEAICE_GROWTH [#else]`; `# ifdef SEAICE_USE_GROWTH_ADX` | all | S:infp | SEAICE_GROWTH_ADX (SEAICE_USE_GROWTH_ADX): thermodynamics and the ocean forcing |
| `I04_growth` | seaice | `pkg/seaice/seaice_model.F:277` | `#ifdef DISABLE_SEAICE_GROWTH [#else]`; `# ifdef SEAICE_USE_GROWTH_ADX [# else]` | all | S:infp | SEAICE_GROWTH: thermodynamics, ocean forcing Qnet/Qsw/EmPmR/saltFlux, sIceLoad, salt-plume flux |
| `P13_seaice_model` | seaice | `model/src/do_oceanic_phys.F:453` | `#ifdef ALLOW_SEAICE` | all | S:iyfh | all of SEAICE_MODEL (after its HEFF/AREA/HSNOW and forcing exchanges) |
| `P14_salt_plume_exch` | salt_plume | `model/src/do_oceanic_phys.F:548` | `#ifdef ALLOW_SALT_PLUME` | all | S:p | salt-plume depth and flux after SALT_PLUME_DO_EXCH (useSALT_PLUME) |
| `P01_external_forcing_surf` | core | `model/src/do_oceanic_phys.F:579` | always | all | S:f | surface forcing arrays (before the tile loop) |
| `P02_rho_sigma_ivdc` | core | after the ENDDO of the loop around `model/src/do_oceanic_phys.F:841` | always | tile | S:m, T:sigmaX:W:Nr, T:sigmaY:S:Nr, T:sigmaR:C:Nr | FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC: after the k loop (do_oceanic_phys.F:803-887) |
| `P03_mxlayer` | core | `model/src/do_oceanic_phys.F:897` | always | tile | S:m | CALC_OCE_MXLAYER (calcGMRedi or doDiagsRho odd) |
| `P04_ggl90` | ggl90 | `model/src/do_oceanic_phys.F:1010` | `#ifdef  ALLOW_GGL90` | tile | S:k | GGL90 TKE, viscosity, diffusivity |
| `P07_kpp` | kpp | `model/src/do_oceanic_phys.F:956` | `#ifdef  ALLOW_KPP` | tile | S:K | KPP_CALC: viscosity, diffusivities (double diffusion with KPPuseDoubleDiff), non-local term, boundary-layer depth (calcKPP; KPP_CALC_DUMMY in AD builds is not this stage) |
| `P08_pp81` | pp81 | `model/src/do_oceanic_phys.F:973` | `#ifdef  ALLOW_PP81` | tile | S:Q | PP81_CALC: Richardson-number viscosity, diffusivity |
| `P09_my82` | my82 | `model/src/do_oceanic_phys.F:995` | `#ifdef  ALLOW_MY82` | tile | S:Y | MY82_CALC: Mellor-Yamada viscosity, diffusivity, boundary-layer depth |
| `P10_kpp_exch` | kpp | `model/src/do_oceanic_phys.F:1103` | `#ifdef ALLOW_KPP` | all | S:K | KPP fields after their halo exchange |
| `P11_ggl90_exch` | ggl90 | `model/src/do_oceanic_phys.F:1109` | `#ifdef ALLOW_GGL90` | all | S:k | GGL90 fields after GGL90_EXCHANGES (useGGL90; the dump runs after the one-line IF) |
| `P05_gmredi_tensor` | gmredi | `model/src/do_oceanic_phys.F:1034` | `#ifdef ALLOW_GMREDI` | tile | S:g | GM/Redi slopes, taper, tensor |
| `P06_gmredi_exch` | gmredi | `model/src/do_oceanic_phys.F:1097` | `#ifdef ALLOW_GMREDI` | all | S:g | GM/Redi tensor halo exchange |
| `S04_oceanic_phys` | core | `model/src/forward_step.F:657` | always | all | S:fmkgrt | all of DO_OCEANIC_PHYS |
| `S05_thermodynamics_sync` | core | `model/src/forward_step.F:733` | always | all | S:ta | THERMODYNAMICS before DYNAMICS (staggerTimeStep=.FALSE.) |
| `D00a_phi_hyd` | core | `model/src/dynamics.F:482` | always | tile | K:dPhiHydX:W, K:dPhiHydY:S, K:phiHydC:C, K:phiHydF:C | hydrostatic pressure (per level k): gradient terms dPhiHydX/Y, phiHydC, phiHydF (next interface) |
| `D00b_mom_fluxform` | core | `model/src/dynamics.F:517` | `#ifdef ALLOW_MOM_FLUXFORM` | tile | K:gU:W:gU(1-OLx,1-OLy,k,bi,bj), K:gV:S:gV(1-OLx,1-OLy,k,bi,bj), K:guDissip:W, K:gvDissip:S | flux-form momentum tendency of level k (gU, gV) and dissipation kept out of AB |
| `D00c_mom_vecinv` | core | `model/src/dynamics.F:527` | `#ifdef ALLOW_MOM_VECINV` | tile | K:gU:W:gU(1-OLx,1-OLy,k,bi,bj), K:gV:S:gV(1-OLx,1-OLy,k,bi,bj), K:guDissip:W, K:gvDissip:S | vector-invariant momentum tendency of level k (gU, gV) and dissipation kept out of AB |
| `D01_before_impl_visc` | core | before `model/src/dynamics.F:587` | always | tile | S:a, T:kappaRU:W:Nr+1, T:kappaRV:S:Nr+1 | explicit gU, gV (after TIMESTEP) and vertical viscosities, input of IMPLDIFF (implicitViscosity) |
| `D02_after_impl_visc` | core | `model/src/dynamics.F:595` | always | tile | S:a | gU, gV after implicit viscosity |
| `S06_dynamics` | core | `model/src/forward_step.F:792` | `#ifdef ALLOW_MOM_STEPPING` | all | S:adm | phi_hyd, momentum tendencies, AB, implicit viscosity -> gU, gV |
| `S07_update_rstar_T` | core | `model/src/forward_step.F:839` | `#ifdef NONLIN_FRSURF`; `# ifndef DISABLE_RSTAR_CODE` | all | S:r | r* at the new time |
| `S08_update_cg2d` | core | `model/src/forward_step.F:869` | `#if ( defined NONLIN_FRSURF || defined ALLOW_SOLVE4_PS_AND_DRAG || \` | all | S:c | cg2d operator + preconditioner |
| `C01_cg2d_inputs` | core | before `model/src/solve_for_pressure.F:308` | `#ifdef DISCONNECTED_TILES [#else]` | all | G:cg2d_b:C:1, G:cg2d_x:C:1, S:c | cg2d right-hand side, first guess and operator |
| `C02_cg2d_solution` | core | `model/src/solve_for_pressure.F:308` | `#ifdef DISCONNECTED_TILES [#else]` | all | G:cg2d_x:C:1, N:numIters, N:nIterMin, N:firstResidual, N:minResidualSq, N:lastResidual | cg2d solution (before its exchange), iteration count and residuals |
| `S09_solve_for_pressure` | core | `model/src/forward_step.F:904` | always | all | S:d | cg2d solve -> etaN |
| `S10_momentum_correction` | core | `model/src/forward_step.F:914` | `#ifdef ALLOW_MOM_STEPPING` | all | S:d | u, v corrected |
| `S11_integr_continuity` | core | `model/src/forward_step.F:928` | always | all | S:d | w, etaH |
| `S12_calc_rstar` | core | `model/src/forward_step.F:949` | `#ifdef NONLIN_FRSURF`; `# ifndef DISABLE_RSTAR_CODE` | all | S:r | rStarFac from etaH |
| `S13_stagger_exchanges` | core | `model/src/forward_step.F:983` | always | all | S:d | exchanges before the staggered tracer step (staggerTimeStep=.TRUE.) |
| `T01_residual_flow` | gmredi | `model/src/thermodynamics.F:272` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifdef ALLOW_GMREDI` | tile | T:uFld:W:Nr, T:vFld:S:Nr, T:wFld:C:Nr | Eulerian + bolus velocity used by tracer advection |
| `T10_temp_adv` | gad | `model/src/temp_integrate.F:277` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifndef DISABLE_MULTIDIM_ADVECTION` | tile | T:gT_loc:C:Nr | theta: multi-dimensional advective tendency (multiDimAdvection schemes) |
| `T11_temp_gT` | gad | before `model/src/temp_integrate.F:469` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gT_loc:C:Nr | theta: total explicit tendency after forcing, diffusion, AB and r* rescale |
| `T12_temp_step` | gad | `model/src/temp_integrate.F:469` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gT_loc:C:Nr | theta: T + dt*gT |
| `T13_temp_impl` | gad | before `model/src/temp_integrate.F:506` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gT_loc:C:Nr, T:kappaRk:C:Nr | theta after the implicit vertical step (GAD_IMPLICIT_R or IMPLDIFF, temp_integrate.F:480-504, when one runs) and its input kappaRk |
| `T02_temp_integrate` | core | `model/src/thermodynamics.F:321` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | S:ta | theta advanced |
| `T20_salt_adv` | gad | `model/src/salt_integrate.F:275` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifndef DISABLE_MULTIDIM_ADVECTION` | tile | T:gS_loc:C:Nr | salt: multi-dimensional advective tendency |
| `T21_salt_gS` | gad | before `model/src/salt_integrate.F:467` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gS_loc:C:Nr | salt: total explicit tendency after forcing, diffusion, AB and r* rescale |
| `T22_salt_step` | gad | `model/src/salt_integrate.F:467` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gS_loc:C:Nr | salt: S + dt*gS |
| `T23_salt_impl` | gad | before `model/src/salt_integrate.F:504` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | T:gS_loc:C:Nr, T:kappaRk:C:Nr | salt after the implicit vertical step (GAD_IMPLICIT_R or IMPLDIFF) and its input kappaRk |
| `T03_salt_integrate` | core | `model/src/thermodynamics.F:332` | `#ifdef ALLOW_GENERIC_ADVDIFF` | tile | S:ta | salt advanced |
| `T04_ptracers_integrate` | ptracers | `model/src/thermodynamics.F:348` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifdef DO_PTRACERS_HERE` | tile | S:P | passive tracers advanced (PTRACERS_INTEGRATE, thermodynamics.F:348; DO_PTRACERS_HERE = ALLOW_PTRACERS without ALLOW_LONGSTEP) |
| `S14_thermodynamics_stagger` | core | `model/src/forward_step.F:1005` | always | all | S:ta | THERMODYNAMICS after the momentum step (staggerTimeStep=.TRUE.) |
| `T05_opps` | opps | `model/src/tracers_correction_step.F:109` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifdef ALLOW_OPPS` | tile | S:t | theta, salt after the OPPS convective adjustment (useOPPS, tracers_correction_step.F:104-113) |
| `T06_convective_adjustment` | core | `model/src/tracers_correction_step.F:116` | `#ifdef ALLOW_GENERIC_ADVDIFF`; `#ifdef INCLUDE_CONVECT_CALL` | tile | S:t | theta, salt after CONVECTIVE_ADJUSTMENT (INCLUDE_CONVECT_CALL, .NOT.useOPPS and cAdjFreq /= 0; M3: front_relax input.bvp/.mxl/.top, cAdjFreq=-1) |
| `S15_tracers_correction` | core | `model/src/forward_step.F:1025` | always | all | S:tP | end of the physics |
| `S16_blocking_exchanges` | core | `model/src/forward_step.F:1093` | always | all | S:dtP | state after the end-of-step exchanges (input of MONITOR, cost and output) |
| `S17_monitor` | monitor | `model/src/forward_step.F:1154` | `#ifdef ALLOW_MONITOR` | all | S:dtP | state at MONITOR (its statistics: STDOUT) |
| `S18_cost_tile` | cost | `model/src/forward_step.F:1163` | `#ifdef ALLOW_COST` | all | S:q | cost terms after COST_TILE |
| `S19_sbo_calc` | sbo | `model/src/do_the_model_io.F:181` | `#ifdef ALLOW_SBO` | all | S:o | SBO_CALC: OAM, mass, centre of mass scalars |
| `E01_cost_final` | cost | `model/src/the_main_loop.F:774` | `#ifdef ALLOW_COST` | all | S:q | COST_FINAL after the time loop (records carry the start iteration of the LAST step) |

## M2 (lane A session 6)
- Group `P` (ALLOW_PTRACERS): `pTracer_nn`, `gpTrNm1_nn`, `surfaceForcingPTr_nn` for nn = 1..PTRACERS_num (the
  compiled count, `pkg/ptracers/PTRACERS_FIELDS.h`), added to `S00_begin`, `S15_tracers_correction`,
  `S16_blocking_exchanges`, `S17_monitor`; new tile stage `T04_ptracers_integrate` after PTRACERS_INTEGRATE
  (`thermodynamics.F:348`, inside `DO_PTRACERS_HERE` = ALLOW_PTRACERS without ALLOW_LONGSTEP). Builds without
  ptracers compile none of it.
- No KPP stage: `tutorial_tracer_adjsens` compiles pkg/kpp but sets `useKPP=.FALSE.` (input_ad/data.pkg:8;
  results/output_adm.txt:233 "compiled but not used"); no other M2 variant compiles kpp.
- `mom_vecinv`: `D00c_mom_vecinv` (per level after MOM_VECINV) already existed; cube layouts use the exch2 probe
  unchanged (W2 tile numbers, faces and tile bases from the W2 topology; any tile count decodes).

## M3 (lane A session 9)
Column-mixing stages for the M3 experiments (plan Task 28); new groups in `jaxdump.F` JAXDUMP_STATE: `K` KPP
(KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat, KPPhbl, KPPfrac; pkg/kpp/KPP.h), `Q` PP81 (PPviscAr, PPdiffKr), `Y` MY82
(MYviscAr, MYdiffKr, MYhbl), and IDEMIX_E, IDEMIX_F_B, IDEMIX_F_S in group `k` under ALLOW_GGL90_IDEMIX. Stages:
`P07_kpp` (after KPP_CALC; double diffusion with KPPuseDoubleDiff is inside it), `P08_pp81`, `P09_my82`,
`P10_kpp_exch` (after KPP_DO_EXCH), `P11_ggl90_exch` (after the one-line `IF (useGGL90) CALL GGL90_EXCHANGES`, so it
dumps in every GGL90 build), and in `model/src/tracers_correction_step.F` (now instrumented): `T05_opps` (after
OPPS_INTERFACE) and `T06_convective_adjustment` (after CONVECTIVE_ADJUSTMENT, front_relax's cAdjFreq=-1). Anchors
re-validated on `pinned` (`instrument.py --check`: ANCHORS OK); existing binaries are untouched (new builds only).

## M4 (lane A session 10)
Forcing and sea-ice stages for the M4 experiments. **Package files are instrumented now** (`pkg/seaice/seaice_model.F`,
`pkg/seaice/seaice_dynsolver.F`, `pkg/exf/exf_getforcing.F`): genmake2 compiles every file of a `-mods` directory
whether or not its package is enabled, so `instrument.py --root --mods` copies a package file only when the build's
`packages.conf` requests that package (`FILE_PKG`; `reference/check_build.py` `requested_packages`, the same set
`check_build.py packages` verifies against ENABLED_PACKAGES) and prints each skipped file in `instrument_report.txt`.
New groups in `jaxdump.F` (re-added from the ECCO port and re-audited on `63cdc0b`): `I` sea-ice static fields
(HEFFM, SIMaskU/V, seaiceMaskU/V, k1/k2 metric terms), `i` sea-ice state (AREA, HEFF, HSNOW, UICE, VICE, TICES, HSALT,
ITD categories), `y` sea-ice dynamics (ice mass, FORCEX0/Y0, PRESS0, SEAICE_zMax/zMin (were ZMAX/ZMIN), viscosities,
strain rates, PRESS, DWATN, FORCEX/Y, uIceNm1/vIceNm1, stress divergence, EVP stresses, free drift, bottom/side drag),
`n` sea-ice budget terms (d_HEFFbyNEG, d_HSNWbyNEG, saltWtrIce, frWtrIce), `x` EXF fields (EXF_FIELDS.h, each under
the flag that declares it; ustress/vstress kind W/S with stressIsOnCgrid, else C), `h` pkg/thsice (THSICE_VARS.h), `p`
salt plume (SaltPlumeDepth, saltPlumeFlux). TAUX/TAUY are locals of SEAICE_DYNSOLVER on master (SEAICE.h fields in
c66g): dumped there as G records (`Y01_get_dynforcing`). Not re-added: the EXF bracketing records (`e`), the
bulk-formulae locals and the LSR/growth internals of the ECCO port (its anchors were c66g/V4r4 code); the solver
stages dump the solver result (`Y05_evp` ... `Y08_jfnk`) and all solver inputs (`Y04_solver_inputs`).

Lane A session 11: `1D_ocean_ice_column/code` is the only M4 code with `SEAICE_BGRID_DYNAMICS` (its
SEAICE_OPTIONS.h:91; every other M4 jaxdump build defines SEAICE_CGRID, and 1D_ocean_ice_column/code_ad neither). There
SEAICE_MODEL calls the B-grid `DYNSOLVER` (pkg/seaice/dynsolver.F; seaice_model.F:180-183), which runs in every step
even with SEAICEuseDYNAMICS=.FALSE.: forcing FORCEX/Y(0), PRESS0, SEAICE_zMax/zMin, then OSTRES (fu, fv under ice).
The C-grid stages I00/I01 sit in the `#else` branch and did not exist there, so two B-grid stages were added:
`I00b_seaice_begin` (before CALL DYNSOLVER) and `I01b_dynsolver` (after it), with a new group `B` (SEAICE.h:185-211:
AMASS, DAIRN, uIceB, vIceB, WINDX, WINDY, GWATX, GWATY). Kinds in B-grid builds: UICE/VICE, FORCEX/Y, FORCEX0/Y0 and the
B-group corner fields are dumped as kind Z (South-West corner, SEAICE.h:11-18), WINDX/Y as C. The B-grid
SEAICE_ADVDIFF call (seaice_model.F:229, first occurrence) is not instrumented (1D_ocean_ice_column advects nothing:
data.seaice SEAICEadv* = .FALSE.).
