# R5: forward branches the code_ad build changes (`tutorial_global_oce_optim/input_ad`)

Plan Task 16, first item (lane ctrl, 2026-10-01). Which `if cfg.cpp.ALLOW_AUTODIFF:` (or `ALLOW_CTRL`, `ALLOW_COST`,
`ALLOW_AUTODIFF_TAMC`, `ALLOW_GRDCHK`) branches the forward of R5 needs from each lane.

## How the list was made (reproducible)
`python mitjax/tests/code_ad_forward.py [--kinds code --text]` (seconds, login node). Two link farms of
`mitjax/config/cpp_options` (the project's single CPP module, the oracle build's own `cpp -traditional`):
* **code_ad**: the experiment's build, as the oracle compiles it (packages `gfd cd_code gmredi autodiff cost ctrl
  grdchk` + dependencies);
* **plain**: the same `code_ad` directory with `autodiff cost ctrl grdchk` removed from `packages.conf` (genmake2 then
  writes `#undef ALLOW_AUTODIFF/ALLOW_COST/ALLOW_CTRL/ALLOW_GRDCHK` into PACKAGES_CONFIG.h; AUTODIFF_OPTIONS.h,
  COST_OPTIONS.h and CTRL_OPTIONS.h define nothing without them).

For each of the 337 executed routines of `docs/coverage/tutorial_global_oce_optim.md` (gcov of the code_ad build, run
job 27827383) the lines live in one build only (cpp arm markers, `cpp_options.live_lines`) are grouped into blocks and
classified: `taf` (only comment lines: CADJ directives), `include`, `decl` (declarations only), `tafkey` (TAF tape keys
`tkey`/`kkey` and their declarations), `code` (anything else). Result: **208 blocks in 42 files: 72 taf, 25 include,
13 decl, 17 tafkey, 81 code (in 38 files)** (+ = live only in code_ad, - = live only in the plain build), plus **80 executed routines
that only the code_ad build compiles** (pkg/autodiff, cost, ctrl, grdchk; `code_ad/cost_*.F`, which include `cost.h`).
The template-generated eesupp exchange routines (`exch1_rl.F`, `exch_xy_rl.F`, ...) and `eeintro_msg.F` are not in
the link farms (genmake2 makes them in the build directory) and appear in the script's "only in code_ad" list only for
that reason; their `.f` in the code_ad build directory has no `ALLOW_AUTODIFF` branch that the forward runs (the
exchanges are gated bitwise through the probe maps of `mitjax/eesupp`, which were measured on this very build).

Runtime switches that exist only under `ALLOW_AUTODIFF` and are FALSE in this forward run (STDOUT of the oracle):
`inAdMode` (AUTODIFF_INADMODE_SET/UNSET each step, forward_step.F:434, 1208: `inAdMode = inAdTrue/inAdFalse`, both
.FALSE. in a forward-only build), `useApproxAdvectionInAdMode`, `useGMRediInAdMode` (T but only read with inAdMode),
`cg2dFullAdjoint` (F). A port keeps the switch and its branch (R5 reads its value from the run, not a constant).

## Code blocks that matter for R5 (by owning lane)
Effect: **value** = can change a forward value or halo (port the code_ad arm under `if cfg.cpp.<MACRO>:` and gate it);
**control flow** = which routines run (driver level); **print** = STDOUT only; **none** = a runtime switch that is
.FALSE./.TRUE. in R5 so both arms do the same, or a check that only stops (still port: it is the live code).

| file:lines (sign) | routine | arm | what changes in R5 | effect | lane |
|---|---|---|---|---|---|
| calc_adv_flow.F:90-94 (+), 96-97 (-) | CALC_ADV_FLOW | `#ifdef ALLOW_AUTODIFF` :89 | `rTransKp = wFld(k+1)*rA*maskC(k)*maskC(k+1)*deepFac2F(k+1)*rhoFacF(k+1)` recomputed instead of copying `rTrans` of the previous call (k+1); different operand order and an extra `maskC(k)`: equal values only where the masks are 1 and the factors 1 (signed zeros may differ at land) | value | core (Task 13) |
| do_oceanic_phys.F:352-364 (+) | DO_OCEANIC_PHYS | :351 | `adjustColdSST_diag = 0` every step (all points) | value (diag field) | COL / core |
| do_oceanic_phys.F:640-713 (+) | DO_OCEANIC_PHYS | :637 | every step, all points incl. halos: `rhoInSitu = 0.`, `IVDConvCount = 0.` (REAL*4 `0.`), `Kwx Kwy Kwz Kux Kvy (Kuz Kvz) = 0` before FIND_RHO_2D/CALC_IVDC/GMREDI_CALC_TENSOR; in the plain build points those routines do not write keep the previous step's values (halos, k ranges) | value (halos) | COL / GMREDI |
| do_oceanic_phys.F:572-574 (-), 731, 769-790 (+), 1068-1069 (-) | DO_OCEANIC_PHYS | :571, :730, :768, :1067 | the `IF (fluidIsWater)` moves from around the whole tile loop (plain) to around FIND_RHO_2D/... with an atmosphere ELSE arm (code_ad); fluidIsWater = T in R5 | none | COL / core |
| do_oceanic_phys.F:1039-1043 (+) | DO_OCEANIC_PHYS | :1038 | `ELSE CALL GMREDI_CALC_TENSOR_DUMMY` (useGMRedi = T: not taken) | none | GMREDI |
| dynamics.F:298-306, 325, 492-497 (+) | DYNAMICS | :297, :324, :491 | `gU = gV = 0` (all k, all points), `phiHydLow = 0`, `guDissip = gvDissip = 0` (per k) before use | value (halos) | core (Task 12) |
| dynamics.F:370, 381 (-) | DYNAMICS | :369, :380 | `kappaRU/kappaRV = 0` (k=1..Nr+1, all points) unconditionally in code_ad; plain only `IF (.NOT.momViscosity)` (momViscosity = T in R5: plain skips it) | value (halos, unwritten k) | core (Task 12) |
| external_fields_load.F:87-92 (+), 97-100 (-) | EXTERNAL_FIELDS_LOAD | :86 / :96 | new records read when `intime0.NE.intimeP .OR. myIter.EQ.nIter0` (code_ad) instead of `intime1.NE.loadedRec(bi,bj)` (plain); same records for a monotone forward run, but the condition and `loadedRec` bookkeeping differ | control flow (reads) | FORCING |
| load_fields_driver.F:170 (+), 172 (-) | LOAD_FIELDS_DRIVER | :169 | `ELSE` (code_ad) vs `ELSEIF ( fluidIsWater )` (plain): T in R5 | none | FORCING |
| load_fields_driver.F:187-192 (+) | LOAD_FIELDS_DRIVER | `#ifdef ALLOW_CTRL` :186 | `IF (useCTRL) CALL CTRL_MAP_GENTIM2D` | value (xx_gentim2d) | FORCING calls `pkg/ctrl` (lane ctrl: `ctrl_map_gentim2d.py`) |
| forward_step.F:570-575 (+) | FORWARD_STEP | `#ifdef ALLOW_CTRL` :569 | `IF (useCTRL) CALL CTRL_MAP_FORCING` after LOAD_FIELDS_DRIVER (dump stage S03) | value (Qnet + xx, exchanges of 7 fields and fu/fv) | core calls `pkg/ctrl` (lane ctrl) |
| forward_step.F:1162-1164 (+) | FORWARD_STEP | `#ifdef ALLOW_COST` :1159 | `CALL COST_TILE` after MONITOR (stage S18) | value (cost) | core calls `pkg/cost` (lane ctrl) |
| forward_step.F:429-434, 1208 (+) | FORWARD_STEP | :427, :1207 | `myIter = nIter0 + (iloop-1)`, `myTime = startTime + deltaTClock*(iLoop-1)` recomputed at the start of each step (plain: carried); AUTODIFF_INADMODE_UNSET/SET | value (myTime rounding: the recomputed time can differ by an ulp from an accumulated one) | core (Task 11) |
| forward_step.F:786, 795, 911, 917 (-) | FORWARD_STEP | `#ifndef ALLOW_AUTODIFF` | `IF (momStepping)` around DYNAMICS and MOMENTUM_CORRECTION_STEP only in the plain build; momStepping = T | none | core |
| gad_calc_rhs.F:174-186 (+) | GAD_CALC_RHS | :165 | `IF (inAdMode .AND. useApproxAdvectionInAdMode)` replaces flux-limited DST3 by DST3 (inAdMode = F) | none (forward) | core (Task 13) |
| gmredi_calc_tensor.F:264-278, 387-393 (+) | GMREDI_CALC_TENSOR | :263, :386 | `Kwx..Kvz = 0` (all k, all points) at entry; `SlopeX, SlopeY, dSigmaDx/Dy/Dr, SlopeSqr, taperFct = 0` per k | value (halos) | GMREDI |
| gmredi_init_varia.F:75, 221 (+); packages_init_variables.F:267, 274 (-) | GMREDI_INIT_VARIA | :70 | the `IF (useGMRedi)` moves into GMREDI_INIT_VARIA | none | GMREDI / core |
| find_rho.F:76-82 (+) | FIND_RHO_2D | :75 | `rhoLoc = rhoP0 = bulkMod = 0` (all points) at entry | value (halos of locals) | COL |
| mom_calc_ke.F:54-58 (+) | MOM_CALC_KE | :53 | `KE = 0.` (all points) at entry | value (halos) | MOM |
| mom_u_botdrag_coeff.F:94-98, mom_v_botdrag_coeff.F:94-98 (+) | MOM_U/V_BOTDRAG_COEFF | :92 | `cDrag = 0` (all points) at entry | value (halos) | MOM |
| temp_integrate.F:213-228, salt_integrate.F:211-226 (+) | TEMP/SALT_INTEGRATE | :212, :210 | `kappaRk = 0` (all k, all points) at entry | value (halos) | core (Task 13) |
| thermodynamics.F:150-151 (+) | THERMODYNAMICS | :149 | `TsurfCor = 0`, `SsurfCor = 0` before `IF (linFSConserveTr)` (F in R5: they stay 0 and are read only under linFSConserveTr) | none | core (Task 13) |
| ini_linear_phisurf.F:53-62 (+) | INI_LINEAR_PHISURF | :52 | `Bo_surf = recip_Bo = 0` (all points) first | value (halos) | core (Task 11) |
| ini_linear_phisurf.F:218 (+), 221-235 (-) | INI_LINEAR_PHISURF | :217 | inside `IF (fluidIsAir .AND. topoFile...)`: STOP in code_ad; F in R5 | none | core |
| ini_pressure.F:77-181 (-) | INI_PRESSURE | `#ifndef ALLOW_AUTODIFF` :76 | inside `IF (storePhiHyd4Phys)`: iterative initial hydrostatic pressure only in plain; F in R5 ("Pressure is predetermined for buoyancyRelation OCEANIC") | none | core |
| initialise_varia.F:184 (+); the_main_loop.F:296-345, 650-656 (+) | INITIALISE_VARIA, THE_MAIN_LOOP | :183, :295, :649 | `nIter0 = NINT((startTime-baseTime)/deltaTClock)` recomputed (0 in R5) | none in R5 | core |
| set_defaults.F:245 (+) | SET_DEFAULTS | :244 | `debugLevel = debLevA` (plain: debLevB): fewer STDOUT messages | print | core |
| set_parms.F:178 (+) | SET_PARMS | :177 | `doResetHFactors = .TRUE.`, then `.FALSE.` again without NONLIN_FRSURF (:180-182, undefined here) | none | core |
| solve_for_pressure.F:318-319 (+) | SOLVE_FOR_PRESSURE | :316 | `IF (.NOT.useNSACGSolver .AND. cg2dFullAdjoint) CALL CG2D_STORE` (cg2dFullAdjoint = F) | none | CG2D |
| initialise_varia.F:246, 268; packages_*.F (+/-) | init drivers | ALLOW_AUTODIFF/CTRL/COST/GRDCHK | AUTODIFF_INIT_VARIA, COST_INIT_VARIA, CTRL_INIT_FIXED/VARIABLES, COST_INIT_FIXED, *_READPARMS, *_CHECK, package summary lines | control flow, print | core calls lane ctrl's init routines |
| the_main_loop.F:417-631, 714-720, 767-775, 644, 730; the_model_main.F:633-737, 743-748 | drivers | ALLOW_AUTODIFF / TAMC_CHECKPOINTING / COST | checkpoint-level loops around the time loop (`nchklev_1/2/3`), COST_DRIVER + COST_FINAL after the loop, CTRL_UNPACK/PACK and GRDCHK_MAIN in THE_MODEL_MAIN | control flow | drivers (Task 16/17) |
| config_check.F:264-281, gad_check.F:134-173 (+) | checks | :263, :133 | stop on momImplVertAdv, selectImplicitDrag, PPM/PQM schemes in an AD build (none set in R5) | none (checks) | core / GAD |
| gmredi_rtransport/xtransport/ytransport.F (+) | GMREDI_*TRANSPORT | ALLOW_AUTODIFF_TAMC | TAF keys + STOP if `trIdentity > maxpass` (2 tracers, maxpass = 2: never) | none | GMREDI |
| gmredi_slope_limit.F:506-511 (+) | GMREDI_SLOPE_LIMIT | :505 | loop split for TAF (same statements) | none (check order) | GMREDI |
| do_atmospheric_phys.F:102-113 (+) | DO_ATMOSPHERIC_PHYS | :101 | `ELSE rhoInSitu = 0.` when not atmospheric; executed 10 times in R5 (gcov) | value (rhoInSitu all points each step) | core |

All other blocks are TAF directives, includes, declarations or tape keys (no forward effect). The full block list with
line ranges and texts is the script's output.

## What this means for the lanes
* Every "value (halos)" entry is an initialisation the code_ad build executes and the plain build does not. Forward
  interiors are the same, but **halo and unwritten points differ**, and the oracle dumps of R5 come from the code_ad
  build: a kernel gated against the optim dumps must take the code_ad arm (`if cfg.cpp.ALLOW_AUTODIFF:`) to be bitwise
  on all points. The plain-build experiments of M1 (barotropic/baroclinic gyre, advect, global_ocean `input`) take the
  other arm.
* CALC_ADV_FLOW is the one place where the arms compute an interior value differently (the recomputed `rTransKp`).
* EXTERNAL_FIELDS_LOAD's read condition differs (FORCING lane).
* FORWARD_STEP's per-step `myTime = startTime + deltaTClock*(iLoop-1)` (code_ad) vs the plain build's accumulation:
  the core lane's time loop needs the code_ad form for R5.
