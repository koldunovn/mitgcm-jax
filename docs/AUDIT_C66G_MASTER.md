# c66g → master audit of the Fortran behind the copied infrastructure (plan Task 7a)

The ECCO port (R, MITgcm checkpoint66g) encodes MITgcm Fortran in the modules this project copies (lessons [E§11]).
This file records, per module, which Fortran each one encodes, every hunk of `git diff checkpoint66g pinned` for those
files (`pinned` = `63cdc0b`), its class, and what the copied module must change. The tables between the BEGIN/END
markers are written by `scripts/audit_upstream.py --update` and compared byte for byte by `--check`; everything else is
hand-written. `--check` also fails when a forward-value hunk has no reference (`file.F#n` or `file.F#n-m`) in the
hand-written part below, when a hunk is undecided, or when an override is invalid.

```bash
python3 scripts/audit_upstream.py --check docs/AUDIT_C66G_MASTER.md   # gate (scripts/tests/test_audit_upstream.py)
python3 scripts/audit_upstream.py --update docs/AUDIT_C66G_MASTER.md  # regenerate the tables
python3 scripts/audit_upstream.py --routines FORWARD_STEP DYNAMICS    # Task 4: classify any routine list
python3 scripts/audit_upstream.py --show model/src/cg2d.F#14          # one hunk, its verdict and content sha
```

## Classes (rules in the script docstring; the script is authoritative)
- **forward value**: changes what a forward run computes. The default for every executable statement, `#define`,
  `PARAMETER`, and for `#include` of an `*_OPTIONS.h` header. A hunk in this class is not necessarily active in an
  M1 run; the module notes below say whether it is.
- **I/O**: file reading or writing (READ/OPEN/CLOSE/NAMELIST, MDS/READ_FLD/MNC calls, all of `pkg/mdsio`, `pkg/rw`,
  `eesupp/src/mds_*.F`).
- **AD-only**: compiled only into adjoint builds (`#ifdef ALLOW_AUTODIFF*`/`ALLOW_ADJOINT*`/`ALLOW_TAPENADE`/...,
  `CADJ` and `$TAF` directives, `*.flow`, `*_ad.F`, `*_mad.F`, `pkg/autodiff`). Such code still runs in the forward
  sweep of a `code_ad` build, so for R5 (`tutorial_global_oce_optim/code_ad`) "AD-only" is not "backward-only".
- **diagnostics**: messages, error checks and counters, `DEBUG_*`, `TIMER_*`, `DIAGNOSTICS_FILL`, time averages
  (`*_TAVE`), blocks under a debug-only condition.
- **cosmetic**: comments, blank lines, layout and case (token-equal code), declarations, `#include` of plain headers,
  CPP comment text, include guards, NEC/OpenMP directives. Most of them are the removal of the CVS `$Header$`/`$Name$`
  lines (2017) and re-indentation.

A hunk takes the most severe class among its lines (forward value > I/O > AD-only > diagnostics > cosmetic). Ten
hunks carry a manual override (`OVERRIDES` in the script, reason in the generated table): five set debug, message or
log-name code to diagnostics, two decide byte-swap rewrites with changed declarations as I/O, three demote CPP
restructurings around TAF directives (the rule conservatively gives a
moved `#ifdef`/`#endif` the class of every statement in its block) to AD-only.

## Line-number corrections (checked at both tags on 2026-10-01)
| citation (source) | c66g | master `63cdc0b` | note |
|---|---|---|---|
| `cg2d.F:148-192` "master's min-residual solution" (plan Task 7a) | `cg2d.F:112, 152-158, 199-202, 347-360, 367-378` | `cg2d.F:101, 148-155, 190-193, 338-351, 358-369` | **not new in master**: c66g has the same `nIterMin`/`cg2d_min` logic, and the same default (`cg2dUseMinResSol`=1 only for a Cartesian grid with no `topoFile`/`bathyFile`, c66g `ini_parms.F:1450-1455`, master `ini_parms.F:1582-1589`) |
| `set_defaults.F:297` `useSRCGSolver=.FALSE.` (plan) | `set_defaults.F:290` | `set_defaults.F:297` (correct) | `ALLOW_SRCG` is `#define`d in the default `CPP_OPTIONS.h` at both tags (c66g :88, master :126) |
| `set_defaults.F:280-286` cg2d defaults [E§11] | `:280-290` | `:286-297` | |
| `ini_parms.F:1451-1455` [E§11] | `:1450-1455` | `:1582-1589` | |
| `doc/tag-index:1339-1348` `cg2dFullAdjoint` (plan) | — | `:1339-1348` (correct; checkpoint68c entry) | |
| `cg2d.F:181-183` per-tile partial loops [E§11] | `:181-183` (`errTile`), `:184` `sumRHStile` inside `#else` | `:173-175`, `sumRHStile` `:176` now after `#endif` | `cg2d.F#10-11` |
| `cg2d.F:122, 130` `_GLOBAL_MAX_RL` [E§11] | `:130` only | `:119` | the second site is not in c66g's `cg2d.F` |
| `global_sum_tile.F:164-194` [E§11] | `:164-199` | `:161-196`; non-MPI path `:198-207` | single-process builds sum the tiles in the bi-inner loop `:199-204` |
| `CPP_EEOPTIONS.h:132` `GLOBAL_SUM_ORDER_TILES` [E§11] | `:132` | `:127` | |
| `ini_cg2d.F:65-78, 90-135` [E§11] | `:53-84, 90-135` | `:51-70, 76-121` | |
| `solve_for_pressure.F:273-311` [E§11] | `:272-312` | `:277-315` (+ `CG2D_STORE` `:316-321`) | `solve_for_pressure.F#8` |
| `cg2d.flow:7-12` [E§11] | `:7-12` | `:4-9` (+ `COMMON` lines `:10-`) | ADNAME `cg2d` → `cg2d_mad`, DEPEND `8` → `6,7,8` |
| `adams_bashforth3.F:87-92` [E§11] | `:88-97` | `:85-94` | |
| `autodiff_readparms.F:66-125` [E§11] | `:36-95` (namelist to CLOSE) | `:49-119` | |
| `autodiff_inadmode_set_ad.F:33-53` [E§11] | `:33-53` | `:53-82` | signature now `(myTime, myIter, myThid)` |
| `temp_integrate.F:488/505`, `salt_integrate.F:480/497` [E§11] | not in c66g (files have 552/533 lines; STOREs at `:234-314`/`:226-`) | `:222-305` / `:220-` | the [E§11] numbers come from R's V4r4 override copies |
| `dynamics.F:399-402` kappaRU/V STOREs [E§11] | `:398-401` | `:402-403` | |
| `diagnostics_is_on.F:47-72` [E§11] | function `:12-67` | `:9-64` | |
| `w2_readparms.F:130-134` default single facet (plan) | `:67-72`, message `:131-134` | `:64-69` (`preDefTopol=1` unless `useCubedSphereExchange`), message `:132-134` | |

## Files that moved, appeared or disappeared
The generated table lists them with the routines they define. In words: `global_vec_sum.F` (`GLOBAL_VEC_SUM_INT/R4/R8`)
was replaced by `global_sum_vector.F` (`GLB_SUM_VEC`, `GLOBAL_SUM_VECTOR_RL/RS`, `GLOBAL_SUM_VEC_ALT_RL/RS`; master
commit `5237154b9`, new `GSVec_size` in `EEPARAMS.h`); `gsum.F` is gone;
`SOLVE_FOR_PRESSURE.h` is gone (cg2d work arrays became locals of each cg2d routine, tag-index checkpoint68c);
`cg2d_mad.F` (renamed from `cg2d_sad.F`) and `g_zero_adj.F` are new; `tamc_keys.h` is gone (TAF keys became local
variables, which is why many AD-only hunks touch `ikey`/`tkey` lines). Every other routine defined at c66g in
one of the listed files is defined in the same file at master (checked 2026-10-01 for all listed files; `--routines`
resolves any name at both tags and reports a move).

## Modules: what the copy must change
M1 runs referred to below: R1 `tutorial_barotropic_gyre`, R2 `tutorial_baroclinic_gyre`, R3 `advect_xy`/`advect_xz`
(+ `ab3_c4`, `nlfs`, `pqm`), R4 `global_ocean.90x40x15/input`, R5 `tutorial_global_oce_optim/input_ad`. Only R4
compiles `pkg/exch2` (single facet, OLx=OLy=3); R1, R2, R3 and R5 use the eesupp (exch1) exchange. All use `mdsio`/`rw`
(package group `gfd`). Only R5 is an AD build (`code_ad`, own `CPP_OPTIONS.h`, `tamc.h`, `AUTODIFF_OPTIONS.h`).

### exch2 maps and the single-device exchanger (`exch2`)
- The maps are re-probed on the master oracle (Task 4), so master's exchange behaviour enters through the probe, not
  through R's code; the Exchanger's gather logic does not depend on any hunk below.
- `exch2_uv_3d_rx.template#5-8`: master wraps the corner updates of the vector exchange in
  `IF ( OLx.GE.2 .AND. OLy.GE.2 )` (e.g. `exch2_uv_3d_rx.template:138-141`). Same values whenever OLx, OLy ≥ 2, which
  holds for every M1 run (R4: 3); differs only for OLx or OLy = 1. Nothing to change in the copy; the probe covers it.
- Twelve `exch_*_rx.template` and all other exch2 files: cosmetic or diagnostics only (log formats, the
  `w2_tile_topology` log name, `w2_eeboot.F#3`).

### eesupp (exch1) exchange (`exch1`)
- New work (no R module): Task 4's probe must cover `EXCH1_RX` (R1, R2, R3, R5).
- `exch1_uv_rx_cube.template#4-5`: the same `OLx.GE.2 .AND. OLy.GE.2` guard, cubed-sphere exch1 only
  (`useCubedSphereExchange`), not used in M1.
- `CPP_EEOPTIONS.h#3-5, #7`: options removed (`USE_OLD_MACROS_R4R8toRSRL`, `FAST_BYTESWAP`,
  `ALLOW_ASYNC_COMMUNICATION`) or added as `#undef` (`USE_FORTRAN_SCRATCH_FILES`, `HACK_FOR_GMAO_CPL`). None changes
  a single-process exchange; `GLOBAL_SUM_ORDER_TILES` stays defined (`:127`). `CPP_EEMACROS.h#2-4`: the old
  `_GLOBAL_SUM_R4/R8`, `_GLOBAL_MAX_R4/R8`, `_EXCH_*_R4/R8` aliases (`USE_OLD_MACROS_R4R8toRSRL`) are gone and
  the file-name/open macros `FMT_PROC_ID`, `FMT_TSK_ID`, `_READONLY_ACTION` are new; no value change. `EEPARAMS.h#2`: new `GSVec_size=1024` for `GLOBAL_SUM_VECTOR` (not used by the copied modules).

### Fixed-order global sums (`global_sum`)
- `global_sum_tile.F`: only cosmetic hunks (the two condition rewrites `#2-3` are token-equal). The copy's tile
  order stands; for a single-process oracle the order is the bi-inner loop at `global_sum_tile.F:199-204`.
- `global_sum_vector.F#1` (was `global_vec_sum.F`): rewritten with an ordered-tiles variant; the copied
  `global_sum.py` does not encode it. If a ported routine calls `GLOBAL_SUM_VECTOR_RL`, port it from master then.
- The caller-side partial sums changed in `cg2d.F#10-11` (see cg2d).

### MDS readers (`mds`)
- I/O hunks only, no forward-value hunk in the readers: `OPEN(..., _READONLY_ACTION)` (`mdsio_read_field.F#2-4`,
  `mdsio_facef_read.F#2-3`, `mdsio_read_meta.F#2`), and the meta-file record count read as `I10` when the line is
  long enough (`mdsio_read_meta.F#3`): `io/mds.py`'s meta parser must accept both widths. `read_rec.F#17,19,22,25`:
  argument lists of the MDS calls. `mds_byteswapr4/r8.F#1-2`: the `FAST_BYTESWAP` variant is gone; the plain byte
  swap remains. `readBinaryPrec` handling (`set_defaults.F`, `ini_parms.F`) is unchanged.
- `MDSIO_OPTIONS.h#2` is I/O (options of the I/O package): `ALLOW_BROKEN_MDSIO_GL` and `ALLOW_WHIO` replaced by
  `EXCLUDE_WHIO_GLOBUFF_2D`/`INCLUDE_WHIO_GLOBUFF_3D` (tape I/O under `ALLOW_AUTODIFF`); no effect on reading inputs.

### cg2d solver and the derivative rule (`cg2d`)
- `cg2d.F#7`, `ini_cg2d.F#3, #5`: `cg2dTolerance_sq` moved from a local of `cg2d.F` into `CG2D.h` and is set in
  `ini_cg2d.F:163` (`cg2dTolerance*cg2dTolerance`, same value). The copy must read it as a constant of the solver
  setup (a traced argument for the rule, [E§11]), not recompute it.
- `ini_cg2d.F#5`: with `cg2dTargetResWunit > 0` master scales the tolerance by `implicDiv2DFlow/deltaTMom *
  globalArea/SQRT(n2dWetPts)` (was `globalArea/deltaTmom`), with a separate `implicDiv2DFlow=0` branch. No M1 run sets
  `cg2dTargetResWunit` (all use `cg2dTargetResidual`, so `cg2dNormaliseRHS=T`); the copy must raise for
  `cg2dTargetResWunit > 0` until a run needs it.
- `ini_cg2d.F#7`: the preconditioner loop runs over `0..sNx`, `0..sNy` (was `0..sNx+1`), followed at both tags by
  `EXCH_XY_RS(pC)` and `EXCH_UV_XY_RS(pW,pS)` (`ini_cg2d.F:240-241`). Same values wherever the exchange fills the
  halo; at an exch2 facet edge without a neighbour the point `sNx+1` keeps its initial 0 (`ini_cg2d.F:58-60`). Task 7b's
  R4 probe shows whether such edges exist. **Answered by the probe (Task 7b, 2026-10-01):** none in M1. R4's single
  facet is periodic in x and y; `EXCH_XY` writes all 5616 halo points from interior sources and the vector exchanges
  5610 of them: the 6 unwritten points are tile 36's north-east corner halo beyond i = sNx+1 / j = sNy+1 (u: (j, i) =
  (11..13, 12..13); v: (12..13, 11..13); Fortran indices, sNx = sNy = 10), so `pW(sNx+1, j)` and `pS(i, sNy+1)` are
  filled by `EXCH_UV_XY_RS` on every tile and the loop-bound change has no effect on R4 (`ini_cg2d.F#4, #8`: `0.` →
  `zeroRS` in comparisons, same value).
- `cg2d.F#9`: `cg2d_r` and `cg2d_s` (now locals) are zeroed on `0..sNx+1` before the first residual (c66g zeroed
  `cg2d_s` only; `cg2d_r` was a common block zeroed in `ini_cg2d.F`). The halo row is then filled by `EXCH_S3D_RL`, so
  values are the same for connected tiles; the copy initialises both, literally.
- `cg2d.F#10-11`: `sumRHS` is now the tile sum for both `CG2D_SINGLECPU_SUM` settings. It is only printed
  (`cg2d: Sum(rhs),rhsMax`, a STDOUT line testreport does not compare), so no value change.
- `cg2d.F#14` (override → diagnostics): a debug-only residual check.
- Min-residual solution: present at both tags (see corrections). R1, R2, R4, R5 read a bathymetry file
  (`cg2dUseMinResSol`=0); R3 is Cartesian without one (=1 by default) but has `momStepping=.FALSE.`, so
  `SOLVE_FOR_PRESSURE` never runs (`forward_step.F:897-905`). No M1 run exercises it: the copy raises when
  `cg2dUseMinResSol=1` and the solver is called.
- `solve_for_pressure.F#8`: solver choice by run-time flags (`useSRCGSolver` → `CG2D_SR`, `useNSACGSolver` →
  `CG2D_NSA`, else `CG2D`; c66g chose `CG2D_NSA` at compile time) plus `CG2D_STORE` for `cg2dFullAdjoint`. No M1 run
  sets either flag or `cg2dFullAdjoint`: `CG2D` is the solver everywhere; the copy raises on the others.
- `solve_for_pressure.F#5, #7, #9`: non-hydrostatic bookkeeping (`zeroPsNH`, `cg3d_b` zeroing; not M1) and the
  `maskInC` factor on the fresh-water term removed in `#7` because `cg2d_b` is multiplied by `maskInC` at
  `solve_for_pressure.F:252` at both tags (m·m = m for a 0/1 mask: same value; R4 uses real fresh water).
- `cg2d_sr.F#6-7, #9`, `cg2d_ex0.F#6-7`, `cg2d_nsa.F#1, #7-8, #10-11, #13`: the same tolerance and work-array changes
  in the solver variants no M1 run selects. Not copied.
- `cg2d.flow#1` (AD-only): TAF's adjoint of `cg2d` is now `cg2d_mad` (`cg2d_mad.F`, which calls `cg2d` again unless
  `cg2dFullAdjoint`), DEPEND gained arguments 6, 7 and the `CG2D_I_RS` common block. The JAX rule ([E§6]) differentiates
  the converged solve and does not reproduce TAF's re-solve; with `cg2dFullAdjoint=F` (R5) TAF's adjoint is again a
  cg2d solve of the transposed (= same) operator, so the rule's semantics match. The rule's transpose tolerance
  (1e-13 in R vs TAF's `cg2dTargetResidual`) remains a documented difference ([E§11]).

### Checkpoint stack and gradient drivers (`driver`)
- The drivers depend on `FORWARD_STEP`'s call order. Forward-value hunks there (`forward_step.F#2-3, #12, #16-17,
  #19-20, #30`): `#2-3` (new options
  headers, no M1 package among them), `#12` (`SHELFICE_REMESHING`, not M1), `#16-17` (`GCHEM_ADD2TR_TENDENCY`, not
  M1), `#19` (the `UPDATE_CG2D` call now also compiled for `ALLOW_CG2D_NSA`/`ALLOW_DEPTH_CONTROL`; R4 has
  `NONLIN_FRSURF`, so it is called at both tags), `#20` (the Shapiro/zonal filters on `uVel,vVel` are no longer applied
  before the solve when `implicDiv2DFlow<1`; neither filter is in M1), `#30` (`ECCO_PHYS`, not M1). Task 4 re-derives the
  step order and where `myIter` advances from master's `forward_step.F` itself.
- `the_main_loop.F#1-2, #14, #18-20`: options headers (`#1-2`), `USE_PDAF` (`#14, #20`), `#18-19` (`ALLOW_PROFILES`/`ALLOW_OBSFIT`
  cost calls): none in M1. The TAF tape layout changed (AD-only hunks, `tamc.h`, `tamc_keys.h` removed); TAF
  comparisons use the experiment's own `code_ad/tamc.h` (R5 has one).
- `adams_bashforth3.F#5`: history index `kl = kArg` and current-tendency index `k = MIN(kArg, kSize)`, so a 2-D
  tendency (kSize=1) can be stepped; same values for kSize = Nr. The coefficients (`:85-94`) are unchanged. R3 `ab3_c4`
  uses it: port from master.

### jaxdump shim: instrumented routines and headers (`jaxdump`)
- The shim instruments master's own routines; nothing of their c66g text is copied. What matters is anchor drift.
  Measured with R's `instrument.py` STAGES against both tags (non-comment lines): all anchors in `forward_step.F`,
  `do_oceanic_phys.F`, `dynamics.F`, `thermodynamics.F`, `solve_for_pressure.F`, `temp_integrate.F`,
  `salt_integrate.F` and the EXF files have the same count on master as R expects; five sea-ice anchors differ
  (`seaice_lsr.F` `DO m = 1, SOLV_MAX_TMP` 1 → 0; `seaice_advdiff.F` `SEAICE_ADVECTION` 6 → 9 and
  `SEAICE_DIFFUSION` 11 → 13; `seaice_model.F` `SEAICE_ADVDIFF` 1 → 2; `seaice_growth.F` `QSW ... *` 1 → 0) — M4, and
  R's sea-ice anchors were written for the V4r4 override of `seaice_growth.F`. Task 4 keeps the count assert.
- Forward-value hunks in the instrumented routines (anchor drift only for the shim; the physics is ported from master in
  M1+): `do_oceanic_phys.F#1, #12, #17-18, #20, #23-24, #30-31, #34-35, #37` (THSICE call sequence, OBCS, a
  `FIND_RHO_2D` call, GGL90, time averages; `#24`: the zeroing of the locals `rhoKm1`, `rhoKp1` left the
  `#ifdef ALLOW_AUTODIFF` block, so plain builds now run it too), `dynamics.F#1, #8, #11, #15, #20-21` (hydrostatic-pressure diagnostic
  switch, `DIAGS_SOUND_SPEED`, implicit bottom drag), `thermodynamics.F#2, #7`, `temp_integrate.F#9-10, #17`,
  `salt_integrate.F#9-10` (argument lists, `useVariableK`), `exf_getforcing.F#2-5`, `exf_radiation.F#1-2`,
  `exf_bulkformulae.F#2, #6-9, #12, #14`, `EXF_PARAM.h#44`, `seaice_model.F#1, #3-6`, `seaice_dynsolver.F#4-6`,
  `seaice_lsr.F#3, #7-11, #13-16, #18, #20-21, #24, #26-27, #32-35, #37, #44, #46, #52`, `seaice_advdiff.F#6-10, #14-17`,
  `seaice_growth.F#3, #11-12, #16, #19, #27-28, #31, #33, #43, #47, #51, #58, #60, #63, #65, #67-71, #75`,
  `SEAICE.h#2, #4` (EXF and sea ice are M4; re-audit then).
- `SIZE.h#3`: the default tile sizes changed; every M1 run has its own `SIZE.h`. `PARAMS.h`, `GRID.h`, `DYNVARS.h`,
  `FFIELDS.h`, `SURFACE.h`, `CG2D.h`: declarations only (cosmetic); `jaxdump.F` must compile against master's
  common blocks (Task 4), e.g. `CG2D.h` now holds `cg2dTolerance_sq` and no work arrays.

### AD helpers (`ad_helpers`; copy deferred, plan Task 7c)
- No M1 run uses a `data.autodiff` approximation (R5's `data.autodiff` sets nothing: `inAdExact=T`, `viscFacInAd=1`,
  `viscFacInFw=1` by default, `autodiff_readparms.F:88-92`). AD-only changes to note for later: `viscFacInFw` is new
  and sets the forward value of `viscFacAdj` (`autodiff_readparms.F:152`, `autodiff_inadmode_unset_ad.F:76`; c66g reset
  it to 1); `ADAUTODIFF_INADMODE_SET` takes `myTime, myIter`; `inAdTrue/inAdFalse` are gone.
- `mom_calc_visc.F#2, #8-11, #13-20` (now four `viscFacAdj` sites, `:511, 525, 658, 672`, c66g two),
  `GMREDI_OPTIONS.h#2-3`, `gmredi_slope_limit.F#2, #6-10, #12-19`, `gmredi_slope_psi.F#1, #3-16`: physics that M1
  ports from master directly (Tasks 13, 15b); `GMREDI_WITH_STABLE_ADJOINT` and `ZERO_ADJ_LOC` are re-derived from
  master when `ad/modes.py` is copied.

### Params pytree and namelist reader (`params`)
- Task 6 reads master's namelists and cites master's defaults; R's `params_io.py` holds no defaults table and nothing of
  `set_defaults.F`/`ini_parms.F` is copied. The forward-value hunks are therefore inputs to Task 6, not changes to a
  copied module: `set_defaults.F#2-15` (new parameters with defaults, removed `use3dCoriolis`,
  `useEnergyConservingCoriolis`, `balanceEmPmR` defaults, `taveFreq`), `ini_parms.F#11-12, #14-16, #18-24` (new
  `select*` switches derived from old logicals for backward compatibility, e.g. `select3dCoriScheme` from
  `use3dCoriolis`, `selectMetricTerms` from `metricTerms`, `selectBalanceEmPmR` from `balanceEmPmR`, `ini_parms.F:705-736`;
  `writeStatePrec`; `selectPenetratingSW`; `selectBotDragQuadr` with `zRoughBot`), `packages_boot.F#4-5, #7, #11`
  (`useOBSFIT`, `useSTIC`, time-average flag), `packages_readparms.F#5-9` (call order of the package readers),
  `eeset_parms.F#1, #4-5, #7-8` (`eedata`: `useNest2W_*`, scratch files, the read loop). The derived `select*` values
  must be computed exactly as `ini_parms.F` does, and the STDOUT cross-check (Task 6) checks them.
- `nml_set_terminator.F`, `nml_change_syntax.F`, `diagnostics_is_on.F`: cosmetic only; the three namelist terminators
  ([E§3]) are unchanged. `ini_parms.F#17` (override → diagnostics): `printResidualFreq` default 0 → -1.

### Default `CPP_OPTIONS.h` (`cg2d` module; compiled by R1 and R2, which have no `code/CPP_OPTIONS.h`)
- `CPP_OPTIONS.h#2-4`: reorganised; `EXACT_CONSERV` is no longer a CPP option (removed in master commit `c3be04357`,
  the code is always compiled and `exactConserv` decides at run time), new `#undef` options
  (`ALLOW_FRICTION_HEATING`, `CHECK_SALINITY_FOR_NEGATIVE_VALUES`, `STORE_LOADEDREC_TEST`,
  `INCLUDE_SOUNDSPEED_CALC_CODE`, `EXCLUDE_PCELL_MIX_CODE`, `ALLOW_SMAG_3D_DIFFUSIVITY`, `ALLOW_QHYD_STAGGER_TS`,
  `USE_MASK_AND_NO_IF`, ...), `ALLOW_SRCG` stays defined, `ALLOW_CG2D_NSA` undefined. Nothing is copied: Task 6 reads
  each experiment's options through the preprocessor (one CPP module), so a copied module never assumes a c66g default.

### Layout and Fortran-index helpers (`layout`)
- `SIZE.h#3` (see jaxdump) and the exch2 topology files: no forward-value change. OLx stays a per-experiment setting.

<!-- BEGIN AUDIT (written by scripts/audit_upstream.py --update; do not edit) -->

Diff `git diff checkpoint66g pinned` (c66g = `4207dc8`, pinned = `63cdc0b`), 3 lines of context, myers. Hunk IDs `path#n` number the hunks of one file in diff order.

### Files

| file | modules | status | hunks | fwd | I/O | AD | diag | cosm |
|---|---|---|---|---|---|---|---|---|
| `eesupp/src/exch_xy_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_xy_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_z_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_agrid_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_bgrid_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_sm_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_dgrid_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_3d_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_s3d_rx.template` | exch2, exch1, cg2d | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_xyz_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_uv_xyz_rx.template` | exch2, exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_z_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_sm_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_s3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_uv_3d_rx.template` | exch2 | modified | 8 | 4 | 0 | 0 | 0 | 4 |
| `pkg/exch2/exch2_uv_agrid_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_uv_bgrid_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_uv_cgrid_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_uv_dgrid_3d_rx.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_rx1_cube.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_rx2_cube.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_get_rx1.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_get_rx2.template` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_put_rx1.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_put_rx2.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_send_rx1.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_send_rx2.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_recv_rx1.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_recv_rx2.template` | exch2 | modified | 2 | 0 | 0 | 0 | 1 | 1 |
| `pkg/exch2/exch2_get_scal_bounds.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/exch2_get_uv_bounds.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/W2_EXCH2_SIZE.h` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/W2_EXCH2_TOPOLOGY.h` | exch2, layout | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/W2_EXCH2_PARAMS.h` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/W2_EXCH2_BUFFER.h` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/W2_OPTIONS.h` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/w2_readparms.F` | exch2, params | modified | 2 | 0 | 1 | 0 | 0 | 1 |
| `pkg/exch2/w2_map_procs.F` | exch2 | modified | 4 | 0 | 0 | 0 | 3 | 1 |
| `pkg/exch2/w2_e2setup.F` | exch2, layout | modified | 4 | 0 | 0 | 0 | 3 | 1 |
| `pkg/exch2/w2_eeboot.F` | exch2 | modified | 3 | 0 | 0 | 0 | 1 | 2 |
| `pkg/exch2/w2_set_single_facet.F` | exch2, layout | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/w2_set_gen_facets.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/w2_set_cs6_facets.F` | exch2 | modified | 3 | 0 | 0 | 0 | 1 | 2 |
| `pkg/exch2/w2_set_myown_facets.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/exch2/w2_set_map_tiles.F` | exch2 | modified | 4 | 0 | 0 | 0 | 3 | 1 |
| `pkg/exch2/w2_set_map_cumsum.F` | exch2 | modified | 3 | 0 | 0 | 0 | 2 | 1 |
| `pkg/exch2/w2_set_tile2tiles.F` | exch2 | modified | 5 | 0 | 0 | 0 | 4 | 1 |
| `pkg/exch2/w2_set_f2f_index.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/fill_cs_corner_ag_rl.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/fill_cs_corner_tr_rl.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/fill_cs_corner_uv_rl.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/fill_cs_corner_uv_rs.F` | exch2 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch0_rx.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch1_rx.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch1_rx_cube.template` | exch1 | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `eesupp/src/exch1_uv_rx_cube.template` | exch1 | modified | 6 | 2 | 0 | 0 | 0 | 4 |
| `eesupp/src/exch1_z_rx_cube.template` | exch1 | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `eesupp/src/exch1_bg_rx_cube.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_rx_send_put_x.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_rx_send_put_y.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_rx_recv_get_x.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_rx_recv_get_y.template` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_init.F` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/ini_communication_patterns.F` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/exch_cycle_ebl.F` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/inc/EXCH.h` | exch1 | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/inc/EESUPPORT.h` | exch1 | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `eesupp/inc/EEPARAMS.h` | exch1, jaxdump | modified | 6 | 1 | 0 | 0 | 0 | 5 |
| `eesupp/inc/CPP_EEOPTIONS.h` | exch1, global_sum | modified | 7 | 4 | 0 | 0 | 0 | 3 |
| `eesupp/inc/CPP_EEMACROS.h` | exch1 | modified | 4 | 3 | 0 | 0 | 0 | 1 |
| `eesupp/src/global_sum_tile.F` | global_sum | modified | 3 | 0 | 0 | 0 | 0 | 3 |
| `eesupp/src/global_sum.F` | global_sum | modified | 8 | 0 | 0 | 0 | 0 | 8 |
| `eesupp/src/global_max.F` | global_sum | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/global_sum_singlecpu.F` | global_sum | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/inc/GLOBAL_SUM.h` | global_sum | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `eesupp/inc/GLOBAL_MAX.h` | global_sum | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/global_vec_sum.F -> eesupp/src/global_sum_vector.F` | global_sum | moved | 1 | 1 | 0 | 0 | 0 | 0 |
| `pkg/mdsio/mdsio_read_field.F` | mds | modified | 4 | 0 | 3 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_facef_read.F` | mds | modified | 3 | 0 | 2 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_rd_rec_rl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_rd_rec_rs.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_seg4torl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_seg8torl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_seg4tors.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_seg8tors.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_pass_r4torl.F` | mds | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `pkg/mdsio/mdsio_pass_r8torl.F` | mds | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `pkg/mdsio/mdsio_pass_r4tors.F` | mds | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `pkg/mdsio/mdsio_pass_r8tors.F` | mds | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `pkg/mdsio/mdsio_buffertorl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_buffertors.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_read_meta.F` | mds | modified | 3 | 0 | 2 | 0 | 0 | 1 |
| `pkg/mdsio/mdsio_check4file.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/mdsio/MDSIO_OPTIONS.h` | mds | modified | 2 | 0 | 1 | 0 | 0 | 1 |
| `pkg/mdsio/MDSIO_BUFF_3D.h` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/rw/read_fld_xy_rl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/rw/read_fld_xy_rs.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/rw/read_fld_xyz_rl.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/rw/read_fld_xyz_rs.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/rw/read_rec.F` | mds | modified | 25 | 0 | 4 | 0 | 0 | 21 |
| `pkg/rw/RW_OPTIONS.h` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/mds_byteswapr4.F` | mds | modified | 1 | 0 | 1 | 0 | 0 | 0 |
| `eesupp/src/mds_byteswapr8.F` | mds | modified | 2 | 0 | 2 | 0 | 0 | 0 |
| `eesupp/src/mds_reclen.F` | mds | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/mdsfindunit.F` | mds, jaxdump | modified | 2 | 0 | 1 | 0 | 0 | 1 |
| `model/src/cg2d.F` | cg2d | modified | 14 | 4 | 0 | 0 | 1 | 9 |
| `model/src/cg2d_sr.F` | cg2d | modified | 9 | 3 | 0 | 0 | 0 | 6 |
| `model/src/cg2d_nsa.F` | cg2d | modified | 15 | 6 | 0 | 4 | 2 | 3 |
| `model/src/cg2d_ex0.F` | cg2d | modified | 8 | 2 | 0 | 0 | 0 | 6 |
| `model/src/ini_cg2d.F` | cg2d | modified | 8 | 5 | 0 | 0 | 1 | 2 |
| `model/src/update_cg2d.F` | cg2d | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `model/src/solve_for_pressure.F` | cg2d, jaxdump | modified | 9 | 4 | 0 | 1 | 1 | 3 |
| `model/inc/CG2D.h` | cg2d, jaxdump | modified | 3 | 0 | 0 | 0 | 0 | 3 |
| `model/inc/SOLVE_FOR_PRESSURE.h` | cg2d | removed | 1 | 0 | 0 | 0 | 0 | 1 |
| `model/inc/CPP_OPTIONS.h` | cg2d | modified | 5 | 3 | 0 | 0 | 0 | 2 |
| `pkg/autodiff/cg2d.flow` | cg2d, ad_helpers | modified | 1 | 0 | 0 | 1 | 0 | 0 |
| `pkg/autodiff/cg2d_mad.F` | cg2d | added | 1 | 0 | 0 | 1 | 0 | 0 |
| `pkg/autodiff/AUTODIFF_PARAMS.h` | cg2d, ad_helpers | modified | 6 | 0 | 0 | 4 | 0 | 2 |
| `model/src/forward_step.F` | driver, jaxdump | modified | 31 | 8 | 0 | 18 | 2 | 3 |
| `model/src/the_main_loop.F` | driver | modified | 20 | 6 | 0 | 12 | 0 | 2 |
| `model/src/adams_bashforth2.F` | driver | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `model/src/adams_bashforth3.F` | driver | modified | 5 | 1 | 0 | 0 | 0 | 4 |
| `pkg/autodiff/tamc.h` | driver | modified | 4 | 0 | 0 | 3 | 0 | 1 |
| `pkg/autodiff/tamc_keys.h` | driver | removed | 1 | 0 | 0 | 1 | 0 | 0 |
| `model/src/do_oceanic_phys.F` | jaxdump | modified | 37 | 12 | 0 | 17 | 0 | 8 |
| `model/src/dynamics.F` | jaxdump, ad_helpers | modified | 21 | 6 | 0 | 10 | 0 | 5 |
| `model/src/thermodynamics.F` | jaxdump | modified | 9 | 2 | 0 | 6 | 0 | 1 |
| `model/src/temp_integrate.F` | jaxdump, ad_helpers | modified | 17 | 3 | 0 | 13 | 0 | 1 |
| `model/src/salt_integrate.F` | jaxdump, ad_helpers | modified | 16 | 2 | 0 | 13 | 0 | 1 |
| `pkg/exf/exf_getforcing.F` | jaxdump | modified | 8 | 4 | 0 | 0 | 0 | 4 |
| `pkg/exf/exf_radiation.F` | jaxdump | modified | 4 | 2 | 0 | 1 | 0 | 1 |
| `pkg/exf/exf_bulkformulae.F` | jaxdump | modified | 14 | 7 | 0 | 4 | 0 | 3 |
| `pkg/seaice/seaice_model.F` | jaxdump | modified | 7 | 5 | 0 | 1 | 0 | 1 |
| `pkg/seaice/seaice_dynsolver.F` | jaxdump | modified | 6 | 3 | 0 | 0 | 0 | 3 |
| `pkg/seaice/seaice_lsr.F` | jaxdump | modified | 52 | 24 | 0 | 5 | 2 | 21 |
| `pkg/seaice/seaice_advdiff.F` | jaxdump | modified | 17 | 9 | 0 | 1 | 0 | 7 |
| `pkg/seaice/seaice_growth.F` | jaxdump | modified | 75 | 22 | 0 | 20 | 2 | 31 |
| `model/inc/SIZE.h` | jaxdump, layout | modified | 3 | 1 | 0 | 0 | 0 | 2 |
| `model/inc/PARAMS.h` | jaxdump, params | modified | 42 | 0 | 0 | 0 | 0 | 42 |
| `model/inc/SURFACE.h` | jaxdump | modified | 2 | 0 | 0 | 0 | 0 | 2 |
| `model/inc/GRID.h` | jaxdump | modified | 10 | 0 | 0 | 0 | 0 | 10 |
| `model/inc/DYNVARS.h` | jaxdump | modified | 5 | 0 | 0 | 0 | 0 | 5 |
| `model/inc/FFIELDS.h` | jaxdump | modified | 10 | 0 | 0 | 0 | 0 | 10 |
| `pkg/exf/EXF_PARAM.h` | jaxdump | modified | 44 | 1 | 0 | 0 | 0 | 43 |
| `pkg/exf/EXF_FIELDS.h` | jaxdump | modified | 7 | 0 | 0 | 0 | 0 | 7 |
| `pkg/seaice/SEAICE_SIZE.h` | jaxdump | modified | 2 | 0 | 0 | 1 | 0 | 1 |
| `pkg/seaice/SEAICE_PARAMS.h` | jaxdump | modified | 28 | 0 | 0 | 0 | 0 | 28 |
| `pkg/seaice/SEAICE.h` | jaxdump | modified | 4 | 2 | 0 | 0 | 0 | 2 |
| `eesupp/src/print.F` | jaxdump | modified | 6 | 0 | 0 | 0 | 5 | 1 |
| `pkg/autodiff/autodiff_readparms.F` | ad_helpers, params | modified | 10 | 0 | 0 | 10 | 0 | 0 |
| `pkg/autodiff/autodiff_inadmode_set_ad.F` | ad_helpers | modified | 5 | 0 | 0 | 4 | 0 | 1 |
| `pkg/autodiff/autodiff_inadmode_unset_ad.F` | ad_helpers | modified | 4 | 0 | 0 | 4 | 0 | 0 |
| `pkg/autodiff/autodiff_inadmode_set.F` | ad_helpers | modified | 1 | 0 | 0 | 1 | 0 | 0 |
| `pkg/autodiff/autodiff_inadmode_unset.F` | ad_helpers | modified | 1 | 0 | 0 | 1 | 0 | 0 |
| `pkg/autodiff/autodiff_inadmode.flow` | ad_helpers | modified | 2 | 0 | 0 | 2 | 0 | 0 |
| `pkg/autodiff/AUTODIFF_OPTIONS.h` | ad_helpers | modified | 3 | 0 | 0 | 3 | 0 | 0 |
| `pkg/autodiff/zero_adj.F` | ad_helpers | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `pkg/autodiff/g_zero_adj.F` | ad_helpers | added | 1 | 0 | 0 | 1 | 0 | 0 |
| `pkg/mom_common/mom_calc_visc.F` | ad_helpers | modified | 21 | 13 | 0 | 4 | 1 | 3 |
| `pkg/gmredi/GMREDI_OPTIONS.h` | ad_helpers | modified | 3 | 2 | 0 | 0 | 0 | 1 |
| `pkg/gmredi/gmredi_slope_limit.F` | ad_helpers | modified | 19 | 14 | 0 | 1 | 0 | 4 |
| `pkg/gmredi/gmredi_slope_psi.F` | ad_helpers | modified | 16 | 15 | 0 | 1 | 0 | 0 |
| `model/src/set_defaults.F` | params | modified | 15 | 14 | 0 | 0 | 0 | 1 |
| `model/src/ini_parms.F` | params | modified | 26 | 12 | 5 | 0 | 1 | 8 |
| `model/src/packages_boot.F` | params | modified | 11 | 4 | 3 | 0 | 3 | 1 |
| `model/src/packages_readparms.F` | params | modified | 9 | 5 | 0 | 0 | 0 | 4 |
| `pkg/diagnostics/diagnostics_is_on.F` | params | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/nml_set_terminator.F` | params | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/nml_change_syntax.F` | params | modified | 1 | 0 | 0 | 0 | 0 | 1 |
| `eesupp/src/eeset_parms.F` | params | modified | 10 | 5 | 2 | 0 | 0 | 3 |
| `eesupp/src/open_copy_data_file.F` | params | modified | 7 | 0 | 4 | 0 | 0 | 3 |
| **total** | | | 967 | 246 | 31 | 169 | 44 | 477 |

### Files added, removed or renamed between the tags

| file | status | its routines at the other tag |
|---|---|---|
| `eesupp/src/global_vec_sum.F -> eesupp/src/global_sum_vector.F` | moved | GLOBAL_VEC_SUM_INT: absent at pinned; GLOBAL_VEC_SUM_R4: absent at pinned; GLOBAL_VEC_SUM_R8: absent at pinned |
| `model/inc/SOLVE_FOR_PRESSURE.h` | removed | no routines (header or flow file) |
| `pkg/autodiff/cg2d_mad.F` | added | CG2D_MAD: absent at checkpoint66g; CG2D_STORE: absent at checkpoint66g |
| `pkg/autodiff/tamc_keys.h` | removed | no routines (header or flow file) |
| `pkg/autodiff/g_zero_adj.F` | added | G_ZERO_ADJ: absent at checkpoint66g; G_ZERO_ADJ_1D: absent at checkpoint66g; G_ZERO_ADJ_LOC: absent at checkpoint66g |

### Hunks that are not cosmetic

| hunk | lines c66g → master | class | decided by | CPP context | first changed line |
|---|---|---|---|---|---|
| `pkg/exch2/exch2_uv_3d_rx.template#5` | -138,8 +135,10 | forward value | rule: block closer; executable statement |  | `+ IF ( OLx.GE.2 .AND. OLy.GE.2 ) THEN` |
| `pkg/exch2/exch2_uv_3d_rx.template#6` | -159,12 +158,14 | forward value | rule: block closer; executable statement |  | `- IF ( withSigns ) THEN` |
| `pkg/exch2/exch2_uv_3d_rx.template#7` | -185,12 +186,14 | forward value | rule: block closer; executable statement |  | `- IF ( withSigns ) THEN` |
| `pkg/exch2/exch2_uv_3d_rx.template#8` | -211,8 +214,10 | forward value | rule: block closer; executable statement |  | `+ IF ( OLx.GE.2 .AND. OLy.GE.2 ) THEN` |
| `pkg/exch2/exch2_put_rx1.template#2` | -127,7 +124,7 | diagnostics | rule: diagnostics statement | W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(2A,I5,I3,A,I5)') 'EXCH2_PUT_RX1',` |
| `pkg/exch2/exch2_put_rx2.template#2` | -159,7 +156,7 | diagnostics | rule: diagnostics statement | W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(2A,I5,I3,A,I5)') 'EXCH2_PUT_RX2',` |
| `pkg/exch2/exch2_send_rx1.template#2` | -87,18 +84,18 | diagnostics | rule: diagnostics statement | ALLOW_USE_MPI & W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(A,I5,A,I5,A)')` |
| `pkg/exch2/exch2_send_rx2.template#2` | -91,24 +88,24 | diagnostics | rule: diagnostics statement | ALLOW_USE_MPI & W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(A,I5,A,I5,A)')` |
| `pkg/exch2/exch2_recv_rx1.template#2` | -83,18 +80,18 | diagnostics | rule: diagnostics statement | ALLOW_USE_MPI & W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(A,I4,A,I4,A)')` |
| `pkg/exch2/exch2_recv_rx2.template#2` | -88,24 +85,24 | diagnostics | rule: diagnostics statement | ALLOW_USE_MPI & W2_E2_DEBUG_ON | `- WRITE(msgBuf,'(A,I4,A,I4,A)')` |
| `pkg/exch2/w2_readparms.F#2` | -124,7 +121,11 | I/O | rule: CPP structure change (class of the wrapped statements); I/O statement | !(SINGLE_DISK_IO) | `+ #ifdef SINGLE_DISK_IO` |
| `pkg/exch2/w2_map_procs.F#2` | -112,7 +109,7 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(3(A,I5))')` |
| `pkg/exch2/w2_map_procs.F#3` | -120,7 +117,7 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(3(A,I5))')` |
| `pkg/exch2/w2_map_procs.F#4` | -151,14 +148,14 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(A,I3,A,I5,A,I3,2A,I5,A)')` |
| `pkg/exch2/w2_e2setup.F#2` | -68,7 +65,7 | diagnostics | rule: diagnostics statement | ALLOW_EXCH2 | `- WRITE(msgBuf,'(A,I5,A,2I3,A)')` |
| `pkg/exch2/w2_e2setup.F#3` | -95,7 +92,7 | diagnostics | rule: diagnostics statement | ALLOW_EXCH2 | `- WRITE(msgBuf,'(3(A,I7))') 'W2_E2SETUP: Number of Tiles=',` |
| `pkg/exch2/w2_e2setup.F#4` | -107,10 +104,10 | diagnostics | rule: diagnostics statement | ALLOW_EXCH2 | `- WRITE(msgBuf,'(A,I5,A,I8)')` |
| `pkg/exch2/w2_eeboot.F#3` | -84,8 +83,9 | diagnostics | override: file name of the w2_tile_topology.<proc>.log listing (digits of the processor number) |  | `+ iTmp = MAX(4,1 + INT(LOG10(DFLOAT(nPx*nPy))))` |
| `pkg/exch2/w2_set_cs6_facets.F#2` | -112,7 +109,7 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(3(A,I4),A,I10,A,I6,A)')` |
| `pkg/exch2/w2_set_map_tiles.F#2` | -93,10 +90,10 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(A,I6,A)')` |
| `pkg/exch2/w2_set_map_tiles.F#3` | -163,7 +160,7 | diagnostics | rule: diagnostics statement |  | `- WRITE(W2_oUnit,'(A,I3,2(A,I6),A,I5,2(A,I4),A)')` |
| `pkg/exch2/w2_set_map_tiles.F#4` | -205,7 +202,7 | diagnostics | rule: diagnostics statement (logical IF) |  | `- &    WRITE(W2_oUnit,'(A,I5,3(A,I3),2A,2I5,2A,2I8)') '  tile',tId,` |
| `pkg/exch2/w2_set_map_cumsum.F#2` | -256,7 +253,7 | diagnostics | rule: diagnostics statement (logical IF) |  | `- IF ( myProcId.EQ.0 ) WRITE(W2_oUnit,'(3(A,I6))')` |
| `pkg/exch2/w2_set_map_cumsum.F#3` | -341,16 +338,16 | diagnostics | rule: diagnostics statement | W2_CUMSUM_USE_MATRIX | `- WRITE(W2_oUnit,'(A,I6,A)')` |
| `pkg/exch2/w2_set_tile2tiles.F#2` | -222,12 +219,12 | diagnostics | rule: diagnostics statement |  | `- WRITE(W2_oUnit,'(A,I5,A,I3,A,4(A,I2))')` |
| `pkg/exch2/w2_set_tile2tiles.F#3` | -283,10 +280,10 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(A,I5,2(A,I3),A,I5)') 'Tile',is,' neighb:',` |
| `pkg/exch2/w2_set_tile2tiles.F#4` | -295,7 +292,7 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(A,I5,2(A,I3),A,I5)') 'Tile',is,' neighb:',` |
| `pkg/exch2/w2_set_tile2tiles.F#5` | -318,16 +315,16 | diagnostics | rule: diagnostics statement |  | `- WRITE(msgBuf,'(A,I5,2(A,I3),A)') 'Tile',is,' neighb:',` |
| `eesupp/src/exch1_uv_rx_cube.template#4` | -243,14 +230,17 | forward value | rule: executable statement |  | `+ IF ( OLx.GE.2 .AND. OLy.GE.2 ) THEN` |
| `eesupp/src/exch1_uv_rx_cube.template#5` | -263,8 +253,9 | forward value | rule: executable statement |  | `+ ENDDO` |
| `eesupp/inc/EEPARAMS.h#2` | -29,14 +27,11 | forward value | rule: PARAMETER/DATA |  | `+ PARAMETER ( GSVec_size = 1024 )` |
| `eesupp/inc/CPP_EEOPTIONS.h#3` | -65,9 +62,6 | forward value | rule: #define/#undef | !_CPP_EEOPTIONS_H_ | `- #undef USE_OLD_MACROS_R4R8toRSRL` |
| `eesupp/inc/CPP_EEOPTIONS.h#4` | -82,9 +76,11 | forward value | rule: #define/#undef | !_CPP_EEOPTIONS_H_ | `- #undef FAST_BYTESWAP` |
| `eesupp/inc/CPP_EEOPTIONS.h#5` | -105,7 +101,6 | forward value | rule: #define/#undef | !_CPP_EEOPTIONS_H_ | `- #define ALLOW_ASYNC_COMMUNICATION` |
| `eesupp/inc/CPP_EEOPTIONS.h#7` | -148,7 +143,10 | forward value | rule: #define/#undef | !_CPP_EEOPTIONS_H_ | `+ #undef HACK_FOR_GMAO_CPL` |
| `eesupp/inc/CPP_EEMACROS.h#2` | -128,32 +125,18 | forward value | rule: #define/#undef | !_CPP_EEMACROS_H_ & !(REAL4_IS_SLOW) & USE_OLD_MACROS_R4R8toRSRL / !_CPP_EEMACROS_H_ & REAL4_IS_SLOW & USE_OLD_MACROS_R4R8toRSRL / !_CPP_EEMACROS_H_ & USE_OLD_MACROS_R4R8toRSRL | `- #define _GLOBAL_SUM_R4(a,b) CALL GLOBAL_SUM_R8 ( a, b )` |
| `eesupp/inc/CPP_EEMACROS.h#3` | -172,13 +155,6 | forward value | rule: #define/#undef | !_CPP_EEMACROS_H_ & USE_OLD_MACROS_R4R8toRSRL | `- #define _EXCH_XY_R4(a,b) CALL EXCH_XY_RS ( a, b )` |
| `eesupp/inc/CPP_EEMACROS.h#4` | -210,4 +186,21 | forward value | rule: #define/#undef | !_CPP_EEMACROS_H_ / !_CPP_EEMACROS_H_ & !(EXCLUDE_OPEN_ACTION) / !_CPP_EEMACROS_H_ & EXCLUDE_OPEN_ACTION | `+ #define FMT_PROC_ID 'I9.9'` |
| `eesupp/src/global_sum_vector.F#1` | -1,257 +1,653 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | !(( defined GLOBAL_SUM_ORDER_TILES && defined ALLOW_USE_MPI )) / !(ALLOW_USE_MPI) / ( defined GLOBAL_SUM_ORDER_TILES && defined ALLOW_USE_MPI ) / ALLOW_USE_MPI | `- SUBROUTINE GLOBAL_VEC_SUM_R4(` |
| `pkg/mdsio/mdsio_read_field.F#2` | -270,7 +267,7 | I/O | rule: I/O statement |  | `- OPEN( dUnit, file=dataFName, status='old',` |
| `pkg/mdsio/mdsio_read_field.F#3` | -382,7 +379,7 | I/O | rule: I/O statement |  | `- OPEN( dUnit, file=dataFName, status='old',` |
| `pkg/mdsio/mdsio_read_field.F#4` | -465,7 +462,7 | I/O | rule: I/O statement |  | `- OPEN( dUnit, file=dataFName, status='old',` |
| `pkg/mdsio/mdsio_facef_read.F#2` | -78,7 +75,7 | I/O | rule: I/O statement | ALLOW_EXCH2 | `- OPEN( dUnit, file=fName(1:iLen), status='old',` |
| `pkg/mdsio/mdsio_facef_read.F#3` | -116,7 +113,7 | I/O | rule: I/O statement | !(ALLOW_EXCH2) | `- OPEN( dUnit, file=fName(1:iLen), status='old',` |
| `pkg/mdsio/mdsio_read_meta.F#2` | -193,7 +190,7 | I/O | rule: I/O statement |  | `- OPEN( mUnit, FILE=mFileName, STATUS='old',` |
| `pkg/mdsio/mdsio_read_meta.F#3` | -303,7 +300,11 | I/O | rule: I/O file; I/O statement |  | `- READ(lineBuf(15:iL),'(I5)') nRecords` |
| `pkg/mdsio/MDSIO_OPTIONS.h#2` | -19,19 +28,19 | I/O | rule: #define/#undef in an I/O package | !MDSIO_OPTIONS_H & ALLOW_MDSIO / !MDSIO_OPTIONS_H & ALLOW_MDSIO & !(ALLOW_AUTODIFF) | `- #undef  ALLOW_BROKEN_MDSIO_GL` |
| `pkg/rw/read_rec.F#17` | -564,7 +556,7 | I/O | rule: I/O statement | ALLOW_MDSIO | `- I                      fType, nNz,` |
| `pkg/rw/read_rec.F#19` | -626,7 +618,7 | I/O | rule: I/O statement | ALLOW_MDSIO | `- I                      fType, nNz,` |
| `pkg/rw/read_rec.F#22` | -689,7 +680,7 | I/O | rule: I/O statement | ALLOW_MDSIO | `- I                      fType, nNz,` |
| `pkg/rw/read_rec.F#25` | -752,7 +742,7 | I/O | rule: I/O statement | ALLOW_MDSIO | `- I                      fType, nNz,` |
| `eesupp/src/mds_byteswapr4.F#1` | -1,55 +1,31 | I/O | override: the FAST_BYTESWAP variant (arr as integer(kind=4), shifts) is removed; the CHARACTER*(*) byte swap that remains is the c66g default path (FAST_BYTESWAP was #undef in CPP_EEOPTIONS.h) |  | `- integer(kind=4) arr(n), i32` |
| `eesupp/src/mds_byteswapr8.F#1` | -1,50 +1,24 | I/O | override: same as mds_byteswapr4.F#1 for 8-byte words |  | `- integer(kind=8) arr(n),i64,i1` |
| `eesupp/src/mds_byteswapr8.F#2` | -57,9 +31,7 | I/O | rule: CPP structure change (class of the wrapped statements); I/O file | !(FAST_BYTESWAP) | `- enddo` |
| `eesupp/src/mdsfindunit.F#2` | -29,7 +26,11 | I/O | rule: CPP structure change (class of the wrapped statements); I/O file | HACK_FOR_GMAO_CPL | `+ #ifdef HACK_FOR_GMAO_CPL` |
| `model/src/cg2d.F#7` | -108,7 +98,6 | forward value | rule: executable statement |  | `- cg2dTolerance_sq = cg2dTolerance*cg2dTolerance` |
| `model/src/cg2d.F#9` | -149,18 +138,21 | forward value | rule: block closer; executable statement |  | `+ DO j=0,sNy+1` |
| `model/src/cg2d.F#10` | -180,8 +172,8 | forward value | rule: executable statement | !(CG2D_SINGLECPU_SUM) | `- sumRHStile(bi,bj) = sumRHStile(bi,bj) + cg2d_b(i,j,bi,bj)` |
| `model/src/cg2d.F#11` | -189,11 +181,10 | forward value | rule: executable statement | !(CG2D_SINGLECPU_SUM) / CG2D_SINGLECPU_SUM | `- CALL GLOBAL_SUM_SINGLECPU_RL(cg2d_b, sumRHS, OLx, OLy, myThid)` |
| `model/src/cg2d.F#14` | -394,29 +387,36 | diagnostics | override: new block runs only if debugLevel.GE.debLevE .AND. printResidualFreq.EQ.1 (cg2d.F:392): residual of the returned x into locals errTile/sumRHS, printed; its _EXCH_XY_RL(cg2d_x) is repeated by solve_for_pressure.F:315 anyway; the removed lines were commented out |  | `+ _EXCH_XY_RL(cg2d_x, myThid )` |
| `model/src/cg2d_sr.F#6` | -105,15 +95,20 | forward value | rule: executable statement | ALLOW_SRCG | `- cg2dTolerance_sq = cg2dTolerance*cg2dTolerance` |
| `model/src/cg2d_sr.F#7` | -154,18 +149,28 | forward value | rule: block closer; executable statement | ALLOW_SRCG | `+ DO j=1,sNy` |
| `model/src/cg2d_sr.F#9` | -340,20 +345,17 | forward value | rule: executable statement | ALLOW_SRCG | `- sumPhi(1,bi,bj) = eta_qrNtile(bi,bj)` |
| `model/src/cg2d_nsa.F#1` | -1,11 +1,11 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_CTRL | `+ # include "CTRL_OPTIONS.h"` |
| `model/src/cg2d_nsa.F#3` | -53,13 +41,11 | AD-only | rule: CPP condition differs only in AD macro names; inside an AD-only #if | ALLOW_AUTODIFF | `- #ifdef ALLOW_AUTODIFF` |
| `model/src/cg2d_nsa.F#6` | -91,25 +76,42 | AD-only | rule: inside an AD-only #if | ALLOW_CG2D_NSA & ALLOW_AUTODIFF_TAMC | `+ INTEGER ikey` |
| `model/src/cg2d_nsa.F#7` | -129,15 +131,7 | forward value | rule: executable statement | ALLOW_CG2D_NSA | `- cg2dTolerance_sq = cg2dTolerance*cg2dTolerance` |
| `model/src/cg2d_nsa.F#8` | -180,18 +174,26 | forward value | rule: block closer; executable statement | ALLOW_CG2D_NSA | `- errTile(bi,bj)    = 0. _d 0` |
| `model/src/cg2d_nsa.F#9` | -224,8 +226,8 | diagnostics | rule: diagnostics statement | ALLOW_CG2D_NSA | `- WRITE(standardmessageunit,'(A,1P2E22.14)')` |
| `model/src/cg2d_nsa.F#10` | -242,16 +244,17 | forward value | rule: executable statement | ALLOW_CG2D_NSA | `- IF ( err_sq .GE. cg2dTolerance_sq ) THEN` |
| `model/src/cg2d_nsa.F#11` | -273,15 +276,16 | forward value | rule: executable statement | ALLOW_CG2D_NSA | `- recip_eta_qrNM1 = 1. _d 0/eta_qrN` |
| `model/src/cg2d_nsa.F#12` | -303,7 +307,7 | AD-only | rule: CADJ/TAF directive | ALLOW_CG2D_NSA & ALLOW_AUTODIFF_TAMC & !ALLOW_LOOP_DIRECTIVE | `- CADJ STORE cg2d_s = comlev1_cg2d_iter, key = icg2dkey, byte = isbyte` |
| `model/src/cg2d_nsa.F#13` | -324,13 +328,14 | forward value | rule: executable statement | ALLOW_CG2D_NSA | `- alpha = eta_qrN/alphaSum` |
| `model/src/cg2d_nsa.F#14` | -352,7 +357,7 | diagnostics | rule: diagnostics statement | ALLOW_CG2D_NSA | `- &    ' cg2d: iter=', it2d, ' ; resid.= ', SQRT(err_sq)` |
| `model/src/cg2d_nsa.F#15` | -394,72 +399,72 | AD-only | rule: inside an AD-only #if | ((defined ALLOW_AUTODIFF_TAMC) && (defined ALLOW_LOOP_DIRECTIVE)) | `- print *, 'adstore: ', chardum, int1, idow, int2, int3, icount` |
| `model/src/cg2d_ex0.F#6` | -94,12 +92,14 | forward value | rule: executable statement |  | `- cg2dTolerance_sq = cg2dTolerance*cg2dTolerance` |
| `model/src/cg2d_ex0.F#7` | -141,18 +141,21 | forward value | rule: block closer; executable statement |  | `+ DO j=0,sNy+1` |
| `model/src/ini_cg2d.F#3` | -62,27 +60,15 | forward value | rule: block closer; executable statement | ALLOW_CG2D_NSA / ALLOW_SRCG | `- DO j=1-1,sNy+1` |
| `model/src/ini_cg2d.F#4` | -128,7 +114,7 | forward value | rule: executable statement |  | `- IF ( myNorm .NE. 0. _d 0 ) THEN` |
| `model/src/ini_cg2d.F#5` | -159,15 +145,22 | forward value | rule: block closer; executable statement |  | `- cg2dNormaliseRHS = cg2dTargetResWunit.LE.0.` |
| `model/src/ini_cg2d.F#6` | -177,7 +170,8 | diagnostics | rule: diagnostics statement |  | `- &      'cg2dTolerance =', cg2dTolerance, ' (Area=',globalArea,')'` |
| `model/src/ini_cg2d.F#7` | -202,14 +196,14 | forward value | rule: executable statement |  | `- DO j=0,sNy+1` |
| `model/src/ini_cg2d.F#8` | -218,18 +212,18 | forward value | rule: executable statement |  | `- IF ( aC .EQ. 0. ) THEN` |
| `model/src/solve_for_pressure.F#2` | -27,25 +24,17 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `+ # include "AUTODIFF_PARAMS.h"` |
| `model/src/solve_for_pressure.F#5` | -76,13 +83,6 | forward value | rule: executable statement | ALLOW_NONHYDROSTATIC | `- zeroPsNH = .FALSE.` |
| `model/src/solve_for_pressure.F#6` | -110,9 +110,6 | diagnostics | rule: diagnostics statement | ALLOW_NONHYDROSTATIC | `- WRITE(msgBuf,'(A,2(A,L5))') 'SOLVE_FOR_PRESSURE:',` |
| `model/src/solve_for_pressure.F#7` | -130,16 +127,24 | forward value | rule: block closer; executable statement | ALLOW_NONHYDROSTATIC | `- cg2d_b(i,j,bi,bj) = 0.` |
| `model/src/solve_for_pressure.F#8` | -273,43 +278,47 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | !(DISCONNECTED_TILES) / !(DISCONNECTED_TILES) & !(ALLOW_CG2D_NSA) / !(DISCONNECTED_TILES) & !(ALLOW_CG2D_NSA) & !(ALLOW_SRCG) / !(DISCONNECTED_TILES) & !(ALLOW_CG2D_NSA) & ALLOW_SRCG / !(DISCONNECTED_TILES) & ALLOW_CG2D_NSA / !(DISCONNECTED_TILES) & ALLOW_SRCG | `- #ifdef ALLOW_CG2D_NSA` |
| `model/src/solve_for_pressure.F#9` | -444,7 +453,6 | forward value | rule: executable statement | ALLOW_NONHYDROSTATIC | `- I                  zeroPsNH, zeroMeanPnh,` |
| `model/inc/CPP_OPTIONS.h#2` | -18,19 +15,60 | forward value | rule: #define/#undef | !CPP_OPTIONS_H | `+ #undef ALLOW_FRICTION_HEATING` |
| `model/inc/CPP_OPTIONS.h#3` | -40,49 +78,49 | forward value | rule: #define/#undef | !CPP_OPTIONS_H | `- #define INCLUDE_IMPLVERTADV_CODE` |
| `model/inc/CPP_OPTIONS.h#4` | -93,28 +131,14 | forward value | rule: #define/#undef | !CPP_OPTIONS_H | `+ #undef USE_MASK_AND_NO_IF` |
| `pkg/autodiff/cg2d.flow#1` | -1,22 +1,34 | AD-only | rule: CADJ/TAF directive |  | `- CADJ SUBROUTINE cg2d    ADNAME  = cg2d` |
| `pkg/autodiff/cg2d_mad.F#1` | -0,0 +1,332 | AD-only | rule: AD-only file | ( defined NONLIN_FRSURF \|\| defined ALLOW_DEPTH_CONTROL ) / ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & !(ALLOW_SRCG) / ALLOW_AUTODIFF_TAMC & ( defined NONLIN_FRSURF \|\| defined ALLOW_DEPTH_CONTROL ) / ALLOW_AUTODIFF_TAMC & ALLOW_SRCG / ALLOW_CTRL | `+ #include "AUTODIFF_OPTIONS.h"` |
| `pkg/autodiff/AUTODIFF_PARAMS.h#2` | -14,15 +11,22 | AD-only | rule: AD-only file |  | `- LOGICAL inAdMode, inAdTrue, inAdFalse, inAdExact` |
| `pkg/autodiff/AUTODIFF_PARAMS.h#3` | -30,14 +34,15 | AD-only | rule: AD-only file |  | `- LOGICAL useSmoothCorrel2DinAdMode, useSmoothCorrel2DinFwdMode` |
| `pkg/autodiff/AUTODIFF_PARAMS.h#4` | -45,7 +50,7 | AD-only | rule: AD-only file |  | `- &       useSmoothCorrel2DinAdMode, useSmoothCorrel2DinFwdMode,` |
| `pkg/autodiff/AUTODIFF_PARAMS.h#6` | -65,9 +70,14 | AD-only | rule: AD-only file |  | `- _RL viscFacInAd` |
| `model/src/forward_step.F#2` | -25,15 +22,30 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_BLING / ALLOW_DIC / ALLOW_GCHEM / ALLOW_SALT_PLUME / ALLOW_SHELFICE | `+ # include "SALT_PLUME_OPTIONS.h"` |
| `model/src/forward_step.F#3` | -43,6 +55,9 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_RBCS | `+ # include "RBCS_OPTIONS.h"` |
| `model/src/forward_step.F#6` | -245,15 +260,16 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_CTRL | `- # include "AUTODIFF_MYFIELDS.h"` |
| `model/src/forward_step.F#7` | -271,11 +287,11 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & ALLOW_EXF | `- #  include "EXF_FIELDS.h"` |
| `model/src/forward_step.F#8` | -292,6 +308,7 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & ALLOW_GCHEM | `+ #  include "GCHEM_SIZE.h"` |
| `model/src/forward_step.F#9` | -323,12 +340,17 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_DOWN_SLOPE / ALLOW_AUTODIFF & ALLOW_SEAICE | `+ #  include "SEAICE_GRID.h"` |
| `model/src/forward_step.F#10` | -352,11 +374,25 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & (defined ALLOW_CG2D_NSA \|\| defined NONLIN_FRSURF \|\| \ / ALLOW_TAPENADE / ALLOW_TAPENADE & ALLOW_EXF / ALLOW_TAPENADE & ALLOW_MOM_FLUXFORM | `- # ifdef ALLOW_CG2D_NSA` |
| `model/src/forward_step.F#11` | -388,16 +424,16 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `- CALL AUTODIFF_INADMODE_UNSET( myThid )` |
| `model/src/forward_step.F#12` | -413,6 +449,15 | forward value | rule: block closer; executable statement | ALLOW_SHELFICE_REMESHING | `+ IF ( useShelfIce ) THEN` |
| `model/src/forward_step.F#13` | -460,7 +505,7 | diagnostics | rule: diagnostics statement | ALLOW_DIAGNOSTICS | `- CALL DIAGNOSTICS_SWITCH_ONOFF( myTime, myIter, myThid )` |
| `model/src/forward_step.F#14` | -565,6 +610,14 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `+ # include "check_lev1_dir_forcing.h"` |
| `model/src/forward_step.F#15` | -575,30 +628,25 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & ALLOW_DEPTH_CONTROL / ALLOW_AUTODIFF_TAMC & ALLOW_DEPTH_CONTROL & ALLOW_SEAICE / ALLOW_AUTODIFF_TAMC & ALLOW_KPP / ALLOW_AUTODIFF_TAMC & ALLOW_OBCS / ALLOW_AUTODIFF_TAMC & ALLOW_OBCS & ALLOW_OBCS_STEVENS / ALLOW_AUTODIFF_TAMC & ALLOW_PTRACERS / ALLOW_AUTODIFF_TAMC & EXACT_CONSERV | `- CADJ STORE surfaceForcingTice = comlev1, key = ikey_dynamics,` |
| `model/src/forward_step.F#16` | -617,38 +665,23 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `- #ifdef ALLOW_GCHEM` |
| `model/src/forward_step.F#17` | -656,65 +689,13 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | ALLOW_GCHEM / GCHEM_ADD2TR_TENDENCY & !ALLOW_AUTODIFF | `- ENDIF` |
| `model/src/forward_step.F#18` | -753,6 +734,9 | AD-only | rule: CADJ/TAF directive | (defined ALLOW_AUTODIFF_TAMC) && (defined ALLOW_OBCS) | `+ CADJ STORE salt, theta = comlev1, key = ikey_dynamics, kind = isbyte` |
| `model/src/forward_step.F#19` | -873,16 +857,11 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | ( defined NONLIN_FRSURF \|\| defined ALLOW_SOLVE4_PS_AND_DRAG \|\| \ | `- #if ( defined NONLIN_FRSURF \|\| defined ALLOW_SOLVE4_PS_AND_DRAG )` |
| `model/src/forward_step.F#20` | -890,30 +869,23 | forward value | rule: block closer; executable statement | ALLOW_SHAP_FILT / ALLOW_ZONAL_FILT | `- IF (implicDiv2DFlow.LT.1.) THEN` |
| `model/src/forward_step.F#21` | -925,14 +897,11 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC & (defined NONLIN_FRSURF) \|\| (defined ALLOW_DEPTH_CONTROL) | `- CADJ STORE uVel, vVel` |
| `model/src/forward_step.F#22` | -941,14 +910,8 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_MOM_STEPPING & ALLOW_AUTODIFF_TAMC / ALLOW_MOM_STEPPING & ALLOW_AUTODIFF_TAMC & ALLOW_DEPTH_CONTROL | `- # ifdef ALLOW_DEPTH_CONTROL` |
| `model/src/forward_step.F#23` | -965,6 +928,12 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `+ CADJ STORE wVel = comlev1, key = ikey_dynamics, kind = isbyte` |
| `model/src/forward_step.F#24` | -997,6 +966,9 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | NONLIN_FRSURF & ALLOW_AUTODIFF_TAMC / NONLIN_FRSURF & ALLOW_AUTODIFF_TAMC & ALLOW_PTRACERS | `+ #  ifdef ALLOW_PTRACERS` |
| `model/src/forward_step.F#25` | -1021,7 +993,10 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE wVel = comlev1, key = ikey_dynamics, kind = isbyte` |
| `model/src/forward_step.F#26` | -1032,6 +1007,11 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `+ CADJ STORE salt, theta = comlev1, key = ikey_dynamics, kind = isbyte` |
| `model/src/forward_step.F#27` | -1088,12 +1068,12 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_GCHEM & ALLOW_AUTODIFF_TAMC / ALLOW_GCHEM & ALLOW_AUTODIFF_TAMC & ALLOW_PTRACERS | `- CADJ STORE pTracer  = comlev1, key = ikey_dynamics,` |
| `model/src/forward_step.F#28` | -1112,6 +1092,17 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & ALLOW_PTRACERS | `+ CADJ STORE theta, salt = comlev1, key = ikey_dynamics, kind = isbyte` |
| `model/src/forward_step.F#29` | -1144,13 +1135,6 | diagnostics | rule: diagnostics statement | ALLOW_TIMEAVE | `- CALL TIMER_START('DO_STATEVARS_TAVE   [FORWARD_STEP]',myThid)` |
| `model/src/forward_step.F#30` | -1181,8 +1165,8 | forward value | rule: executable statement | ALLOW_ECCO | `- IF ( useECCO ) CALL ECCO_PHYS( myThid )` |
| `model/src/forward_step.F#31` | -1221,7 +1205,7 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `- CALL AUTODIFF_INADMODE_SET( myThid )` |
| `model/src/the_main_loop.F#1` | -1,32 +1,44 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_BLING / ALLOW_DIC / ALLOW_GCHEM / ALLOW_GENERIC_ADVDIFF / ALLOW_GGL90 / ALLOW_GMREDI / ALLOW_SALT_PLUME / ALLOW_SHELFICE / ALLOW_STREAMICE | `+ # include "GAD_OPTIONS.h"` |
| `model/src/the_main_loop.F#2` | -39,6 +51,12 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_OBSFIT / ALLOW_RBCS | `+ # include "OBSFIT_OPTIONS.h"` |
| `model/src/the_main_loop.F#4` | -78,7 +94,6 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD | `- #  include "AUTODIFF_MYFIELDS.h"` |
| `model/src/the_main_loop.F#5` | -102,6 +117,7 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD & ALLOW_GCHEM | `+ #   include "GCHEM_SIZE.h"` |
| `model/src/the_main_loop.F#6` | -114,10 +130,6 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD / ALLOW_AUTODIFF & !ALLOW_OPENAD & ALLOW_BLING | `- #  ifdef ALLOW_BLING` |
| `model/src/the_main_loop.F#7` | -127,11 +139,16 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD / ALLOW_AUTODIFF & !ALLOW_OPENAD & ALLOW_BLING / ALLOW_AUTODIFF & !ALLOW_OPENAD & ALLOW_EXF | `- #   include "EXF_FIELDS.h"` |
| `model/src/the_main_loop.F#8` | -167,7 +184,8 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD / ALLOW_AUTODIFF & !ALLOW_OPENAD & (defined ALLOW_CG2D_NSA \|\| defined NONLIN_FRSURF \|\| \ | `- #  ifdef ALLOW_CG2D_NSA` |
| `model/src/the_main_loop.F#9` | -176,25 +194,31 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & !ALLOW_OPENAD / ALLOW_AUTODIFF & !ALLOW_OPENAD & ALLOW_OBSFIT / ALLOW_AUTODIFF & ALLOW_CTRL | `+ #  ifdef ALLOW_OBSFIT` |
| `model/src/the_main_loop.F#10` | -203,6 +227,28 | AD-only | rule: inside an AD-only #if | ALLOW_TAPENADE / ALLOW_TAPENADE & ALLOW_DOWN_SLOPE / ALLOW_TAPENADE & ALLOW_EXF / ALLOW_TAPENADE & ALLOW_GMREDI / ALLOW_TAPENADE & ALLOW_KPP / ALLOW_TAPENADE & ALLOW_PTRACERS | `+ # ifdef ALLOW_GMREDI` |
| `model/src/the_main_loop.F#11` | -228,6 +274,18 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_TAMC_CHECKPOINTING / ALLOW_AUTODIFF & ALLOW_TAMC_CHECKPOINTING & !AUTODIFF_2_LEVEL_CHECKPOINT / ALLOW_AUTODIFF & ALLOW_TAMC_CHECKPOINTING & AUTODIFF_4_LEVEL_CHECKPOINT | `+ # ifdef ALLOW_TAMC_CHECKPOINTING` |
| `model/src/the_main_loop.F#12` | -236,69 +294,76 | AD-only | override: TAF tape INIT/STORE directives rewritten; ikey_dynamics = 1 moved from an ALLOW_AUTODIFF to an ALLOW_AUTODIFF_TAMC block; nIter0 assignment unchanged; no plain-build statement changes |  | `+ #endif` |
| `model/src/the_main_loop.F#13` | -320,13 +385,6 | AD-only | rule: CADJ/TAF directive | ECCO_CTRL_DEPRECATED & ALLOW_ECCO | `- CADJ STORE sbar_gen,tbar_gen  = onetape` |
| `model/src/the_main_loop.F#14` | -340,14 +398,30 | forward value | rule: executable statement | USE_PDAF | `+ CALL INIT_PDAF( nIter0, myTime, myIter, myThid )` |
| `model/src/the_main_loop.F#15` | -360,180 +434,186 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | !ALLOW_OPENAD & ALLOW_AUTODIFF / !ALLOW_OPENAD & ALLOW_AUTODIFF & !DISABLE_MULTIDIM_ADVECTION / !ALLOW_OPENAD & ALLOW_AUTODIFF & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OFFLINE ) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OFFLINE ) & !AUTODIFF_USE_STORE_RESTORE / !ALLOW_OPENAD & ALLOW_AUTODIFF & (defined (ALLOW_EXF) && defined (ALLOW_BULKFORMULAE)) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & (defined ALLOW_EXF && defined ALLOW_BULKFORMULAE) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_BLING / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_CG2D_NSA / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_GMREDI / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_MOM_COMMON & !AUTODIFF_DISABLE_LEITH / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_PTRACERS / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE & !DISABLE_MULTIDIM_ADVECTION / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE & SEAICE_CGRID / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE & SEAICE_CGRID & !(SEAICE_LSR_ADJOINT_ITER) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE & SEAICE_CGRID & SEAICE_ALLOW_EVP / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SEAICE & SEAICE_CGRID & SEAICE_LSR_ADJOINT_ITER / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_STEEP_ICECAVITY / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_STREAMICE / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_THSICE / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & ALLOW_THSICE & (defined ALLOW_EXF && defined ALLOW_BULKFORMULAE) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_CG2D_NSA / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_CTRL / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_GENCOST_CONTRIBUTION / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_GMREDI / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_MOM_COMMON & !AUTODIFF_DISABLE_LEITH / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_PTRACERS / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_SEAICE / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_SEAICE & !DISABLE_MULTIDIM_ADVECTION / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_SEAICE & SEAICE_ALLOW_DYNAMICS & !(SEAICE_LSR_ADJOINT_ITER) / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_SEAICE & SEAICE_ALLOW_DYNAMICS & SEAICE_LSR_ADJOINT_ITER / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_SEAICE & SEAICE_ALLOW_EVP / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_STREAMICE / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_TAMC_CHECKPOINTING & !AUTODIFF_2_LEVEL_CHECKPOINT / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_TAMC_CHECKPOINTING & AUTODIFF_4_LEVEL_CHECKPOINT / !ALLOW_OPENAD & ALLOW_AUTODIFF & ALLOW_THSICE | `+ CADJ    INIT tapelvi3 = USER,'adi'` |
| `model/src/the_main_loop.F#17` | -571,15 +651,19 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC & !(ALLOW_TAMC_CHECKPOINTING) | `- # ifndef ALLOW_OPENAD` |
| `model/src/the_main_loop.F#18` | -589,26 +673,40 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_OBSFIT / USE_PDAF | `+ #endif /* ALLOW_PROFILES */` |
| `model/src/the_main_loop.F#19` | -644,30 +742,37 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | ALLOW_COST / ALLOW_ECCO / ALLOW_OBSFIT / ALLOW_PROFILES | `- CALL COST_PROFILES( myIter, myTime, myThid )` |
| `model/src/the_main_loop.F#20` | -675,6 +780,11 | forward value | rule: executable statement | USE_PDAF | `+ CALL FINALIZE_PDAF( )` |
| `model/src/adams_bashforth3.F#5` | -115,13 +112,14 | forward value | rule: executable statement | ALLOW_ADAMSBASHFORTH_3 | `- k = kArg` |
| `pkg/autodiff/tamc.h#2` | -27,8 +24,8 | AD-only | rule: AD-only file | ALLOW_AUTODIFF_TAMC | `- and maxcube in case you are using ptracers or cubed sphere grid)` |
| `pkg/autodiff/tamc.h#3` | -37,53 +34,35 | AD-only | rule: AD-only file | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & !(ALLOW_TAMC_CHECKPOINTING) / ALLOW_AUTODIFF_TAMC & ALLOW_TAMC_CHECKPOINTING / ALLOW_AUTODIFF_TAMC & ALLOW_TAMC_CHECKPOINTING & AUTODIFF_4_LEVEL_CHECKPOINT | `- integer nyears_chkpt` |
| `pkg/autodiff/tamc.h#4` | -91,42 +70,26 | AD-only | rule: AD-only file | ALLOW_AUTODIFF_TAMC | `- common /tamc_keys_i/` |
| `pkg/autodiff/tamc_keys.h#1` | -1,14 +0,0 | AD-only | rule: AD-only file | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & ALLOW_CG2D_NSA | `- #ifdef ALLOW_AUTODIFF_TAMC` |
| `model/src/do_oceanic_phys.F#1` | -1,8 +1,8 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_MOM_COMMON | `+ # include "MOM_COMMON_OPTIONS.h"` |
| `model/src/do_oceanic_phys.F#2` | -32,6 +32,9 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_OBCS | `+ #ifdef ALLOW_OBCS` |
| `model/src/do_oceanic_phys.F#7` | -136,17 +142,14 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `- # include "AUTODIFF_MYFIELDS.h"` |
| `model/src/do_oceanic_phys.F#8` | -163,7 +166,10 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & ALLOW_EXF / ALLOW_AUTODIFF & ALLOW_EXF & ALLOW_CTRL | `- #  include "ctrl.h"` |
| `model/src/do_oceanic_phys.F#9` | -180,13 +186,21 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_ECCO / ALLOW_AUTODIFF & ALLOW_ECCO & ALLOW_SIGMAR_COST_CONTRIBUTION / ALLOW_AUTODIFF & ALLOW_OBCS / ALLOW_TAPENADE / ALLOW_TAPENADE & ALLOW_SHELFICE | `- # ifdef ALLOW_ECCO` |
| `model/src/do_oceanic_phys.F#12` | -223,12 +238,14 | forward value | rule: executable statement |  | `+ kSrf = 1` |
| `model/src/do_oceanic_phys.F#14` | -266,6 +294,27 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_OBCS & ALLOW_AUTODIFF_TAMC / ALLOW_OBCS & ALLOW_AUTODIFF_TAMC & ALLOW_OBCS_STEVENS / ALLOW_OBCS & ALLOW_AUTODIFF_TAMC & ALLOW_OBCS_STEVENS & ALLOW_OBCS_EAST / ALLOW_OBCS & ALLOW_AUTODIFF_TAMC & ALLOW_OBCS_STEVENS & ALLOW_OBCS_NORTH / ALLOW_OBCS & ALLOW_AUTODIFF_TAMC & ALLOW_OBCS_STEVENS & ALLOW_OBCS_SOUTH / ALLOW_OBCS & ALLOW_AUTODIFF_TAMC & ALLOW_OBCS_STEVENS & ALLOW_OBCS_WEST | `+ CADJ STORE etaN  = comlev1, key=ikey_dynamics, kind=isbyte` |
| `model/src/do_oceanic_phys.F#15` | -273,6 +322,22 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_OBCS & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OBCS_BALANCE ) / ALLOW_OBCS & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OBCS_BALANCE ) & ALLOW_OBCS_EAST / ALLOW_OBCS & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OBCS_BALANCE ) & ALLOW_OBCS_NORTH / ALLOW_OBCS & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OBCS_BALANCE ) & ALLOW_OBCS_SOUTH / ALLOW_OBCS & ( defined ALLOW_AUTODIFF_TAMC && defined ALLOW_OBCS_BALANCE ) & ALLOW_OBCS_WEST | `+ #  ifdef ALLOW_OBCS_NORTH` |
| `model/src/do_oceanic_phys.F#16` | -284,33 +349,19 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_ECCO / ALLOW_AUTODIFF & ALLOW_ECCO & ALLOW_SIGMAR_COST_CONTRIBUTION / ALLOW_AUTODIFF & ALLOW_SALT_PLUME | `- # ifdef ALLOW_SALT_PLUME` |
| `model/src/do_oceanic_phys.F#17` | -322,37 +373,24 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `- #ifndef OLD_THSICE_CALL_SEQUENCE` |
| `model/src/do_oceanic_phys.F#18` | -365,108 +403,47 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `- #endif /* ndef OLD_THSICE_CALL_SEQUENCE */` |
| `model/src/do_oceanic_phys.F#19` | -475,9 +452,24 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_SEAICE & ALLOW_AUTODIFF / ALLOW_SEAICE & ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC / ALLOW_SEAICE & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE tices = comlev1, key=ikey_dynamics, kind=isbyte` |
| `model/src/do_oceanic_phys.F#20` | -495,62 +487,43 | forward value | rule: block closer; executable statement | ALLOW_SHELFICE / ALLOW_SHELFICE & !(ALLOW_STEEP_ICECAVITY) / ALLOW_SHELFICE & ALLOW_STEEP_ICECAVITY / OLD_THSICE_CALL_SEQUENCE & (defined ALLOW_THSICE) && !(defined ALLOW_ATM2D) | `- IF ( useThSIce .AND. fluidIsWater ) THEN` |
| `model/src/do_oceanic_phys.F#21` | -572,15 +545,14 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE theta = comlev1, key = ikey_dynamics,` |
| `model/src/do_oceanic_phys.F#22` | -593,8 +565,9 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC | `- CADJ STORE salt, theta = comlev1, key = ikey_dynamics,` |
| `model/src/do_oceanic_phys.F#23` | -606,29 +579,37 | forward value | rule: block closer; executable statement | ALLOW_OBCS | `+ IF (useOBCS) THEN` |
| `model/src/do_oceanic_phys.F#24` | -647,13 +628,13 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `- #ifdef ALLOW_AUTODIFF` |
| `model/src/do_oceanic_phys.F#25` | -695,10 +676,8 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF & ALLOW_GMREDI | `- #  ifdef GM_NON_UNITY_DIAGONAL` |
| `model/src/do_oceanic_phys.F#26` | -734,24 +713,12 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & ALLOW_KPP / ALLOW_AUTODIFF_TAMC & ALLOW_SALT_PLUME | `- CADJ STORE theta(:,:,:,bi,bj) = comlev1_bibj, key=itdkey,` |
| `model/src/do_oceanic_phys.F#28` | -818,6 +786,9 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_AUTODIFF_TAMC | `+ # ifdef ALLOW_AUTODIFF_TAMC` |
| `model/src/do_oceanic_phys.F#29` | -831,21 +802,6 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- kkey = (itdkey-1)*Nr + k` |
| `model/src/do_oceanic_phys.F#30` | -853,44 +809,49 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_ECCO | `- CALL FIND_RHO_2D(` |
| `model/src/do_oceanic_phys.F#31` | -926,33 +887,33 | forward value | rule: executable statement | ALLOW_SALT_PLUME / ALLOW_SALT_PLUME & SALT_PLUME_VOLUME | `- I              rhoInSitu(1-OLx,1-OLy,1,bi,bj), sigmaR,` |
| `model/src/do_oceanic_phys.F#32` | -961,6 +922,10 | AD-only | rule: CADJ/TAF directive | ALLOW_SALT_PLUME & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE saltplumedepth(:,:,bi,bj)= comlev1_bibj,key=tkey,kind=...` |
| `model/src/do_oceanic_phys.F#33` | -975,21 +940,10 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE surfaceForcingU(:,:,bi,bj)` |
| `model/src/do_oceanic_phys.F#34` | -1045,11 +999,10 | forward value | rule: executable statement | ALLOW_GGL90 | `- IF (useGGL90) THEN` |
| `model/src/do_oceanic_phys.F#35` | -1060,16 +1013,6 | forward value | rule: block closer; executable statement | ALLOW_TIMEAVE | `- IF ( taveFreq.GT. 0. _d 0 ) THEN` |
| `model/src/do_oceanic_phys.F#36` | -1077,12 +1020,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GMREDI & ALLOW_AUTODIFF_TAMC & !GM_EXCLUDE_CLIPPING | `- CADJ STORE sigmaX(:,:,:)        = comlev1_bibj, key=itdkey,` |
| `model/src/do_oceanic_phys.F#37` | -1164,6 +1104,11 | forward value | rule: executable statement | ALLOW_GGL90 | `+ IF ( useGGL90 )` |
| `model/src/dynamics.F#1` | -1,11 +1,11 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_CTRL | `+ # include "CTRL_OPTIONS.h"` |
| `model/src/dynamics.F#3` | -41,9 +42,9 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `- # include "tamc_keys.h"` |
| `model/src/dynamics.F#5` | -189,9 +190,14 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `+ INTEGER tkey, kkey` |
| `model/src/dynamics.F#8` | -245,10 +251,10 | forward value | rule: executable statement | ALLOW_DIAGNOSTICS | `- dPhiHydDiagIsOn = .FALSE.` |
| `model/src/dynamics.F#10` | -279,16 +285,7 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `model/src/dynamics.F#11` | -336,12 +333,19 | forward value | rule: block closer; executable statement |  | `+ DO j=1-OLy,sNy+OLy` |
| `model/src/dynamics.F#12` | -355,11 +359,10 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC / ALLOW_AUTODIFF_TAMC & ALLOW_KPP | `- CADJ STORE uVel (:,:,:,bi,bj) = comlev1_bibj, key=idynkey, byte=i...` |
| `model/src/dynamics.F#13` | -396,10 +399,8 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE kappaRU(:,:,:)` |
| `model/src/dynamics.F#14` | -430,7 +431,7 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- kkey = (idynkey-1)*Nr + k` |
| `model/src/dynamics.F#15` | -480,19 +481,9 | forward value | rule: block closer; executable statement | ALLOW_DIAGNOSTICS | `- I        theta, salt,` |
| `model/src/dynamics.F#16` | -591,7 +582,7 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE gU(:,:,:,bi,bj) = comlev1_bibj , key=idynkey, byte=isbyte` |
| `model/src/dynamics.F#17` | -599,7 +590,7 | AD-only | rule: CADJ/TAF directive | ALLOW_AUTODIFF_TAMC | `- CADJ STORE gV(:,:,:,bi,bj) = comlev1_bibj , key=idynkey, byte=isbyte` |
| `model/src/dynamics.F#18` | -623,7 +614,7 | AD-only | rule: CADJ/TAF directive | ALLOW_CD_CODE & ALLOW_AUTODIFF_TAMC | `- CADJ STORE vVelD(:,:,:,bi,bj) = comlev1_bibj , key=idynkey, byte=...` |
| `model/src/dynamics.F#19` | -631,7 +622,7 | AD-only | rule: CADJ/TAF directive | ALLOW_CD_CODE & ALLOW_AUTODIFF_TAMC | `- CADJ STORE uVelD(:,:,:,bi,bj) = comlev1_bibj , key=idynkey, byte=...` |
| `model/src/dynamics.F#20` | -680,15 +671,18 | forward value | rule: executable statement | INCLUDE_SOUNDSPEED_CALC_CODE | `+ CALL DIAGS_SOUND_SPEED( myThid )` |
| `model/src/dynamics.F#21` | -696,6 +690,16 | forward value | rule: block closer; executable statement | ALLOW_DIAGNOSTICS | `+ IF ( selectImplicitDrag.EQ.0 .AND.` |
| `model/src/thermodynamics.F#2` | -17,7 +14,13 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_CTRL | `+ # include "CTRL_OPTIONS.h"` |
| `model/src/thermodynamics.F#3` | -50,10 +53,14 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF | `- # include "tamc_keys.h"` |
| `model/src/thermodynamics.F#4` | -69,6 +76,12 | AD-only | rule: inside an AD-only #if | ALLOW_TAPENADE / ALLOW_TAPENADE & ALLOW_GENERIC_ADVDIFF | `+ # ifdef ALLOW_GENERIC_ADVDIFF` |
| `model/src/thermodynamics.F#5` | -104,6 +117,10 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `+ INTEGER tkey` |
| `model/src/thermodynamics.F#6` | -122,12 +139,9 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- ikey = 1` |
| `model/src/thermodynamics.F#7` | -142,13 +156,13 | forward value | rule: block closer; executable statement | ALLOW_GENERIC_ADVDIFF / ALLOW_GENERIC_ADVDIFF & ALLOW_LAYERS | `- ENDIF` |
| `model/src/thermodynamics.F#8` | -168,16 +182,7 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `model/src/thermodynamics.F#9` | -284,26 +289,23 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ((defined NONLIN_FRSURF) \|\| (defined ALLOW_DEPTH_CONTROL)) && (defined ALLOW_GMREDI) / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ((defined NONLIN_FRSURF) \|\| (defined ALLOW_DEPTH_CONTROL)) && (defined ALLOW_GMREDI) & GM_EXTRA_DIAGONAL / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ((defined NONLIN_FRSURF) \|\| (defined ALLOW_DEPTH_CONTROL)) && (defined ALLOW_GMREDI) & GM_NON_UNITY_DIAGONAL / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & (defined NONLIN_FRSURF \|\| defined ALLOW_DEPTH_CONTROL) \ / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & (defined NONLIN_FRSURF \|\| defined ALLOW_DEPTH_CONTROL) \ & GM_EXTRA_DIAGONAL / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ALLOW_SALT_PLUME | `- CADJ STORE recip_hFacNew(:,:,:) = comlev1_bibj , key=itdkey, byte...` |
| `model/src/temp_integrate.F#2` | -62,12 +59,8 | AD-only | rule: CPP condition differs only in AD macro names; inside an AD-only #if | ALLOW_AUTODIFF | `- #ifdef ALLOW_AUTODIFF` |
| `model/src/temp_integrate.F#3` | -144,8 +137,10 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `+ INTEGER tkey, kkey` |
| `model/src/temp_integrate.F#4` | -177,16 +172,7 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `model/src/temp_integrate.F#5` | -231,15 +217,17 | AD-only | override: #endif of an ALLOW_AUTODIFF block moved above the CADJ STOREs, which get their own ALLOW_AUTODIFF_TAMC block; only CADJ lines change build membership, no executable statement | ALLOW_GENERIC_ADVDIFF | `+ #endif /* ALLOW_AUTODIFF */` |
| `model/src/temp_integrate.F#6` | -247,6 +235,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & INCLUDE_CALC_DIFFUSIVITY_CALL & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE kappaRk = comlev1_bibj, key = tkey, kind = isbyte` |
| `model/src/temp_integrate.F#7` | -262,7 +253,8 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & !DISABLE_MULTIDIM_ADVECTION & GAD_ALLOW_TS_SOM_ADV & ALLOW_AUTODIFF_TAMC | `- CADJ STORE som_T = comlev1_bibj, key=itdkey, byte=isbyte` |
| `model/src/temp_integrate.F#8` | -297,25 +289,20 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & !(ALLOW_ADAMSBASHFORTH_3) / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ALLOW_ADAMSBASHFORTH_3 | `- kkey = (itdkey-1)*Nr + k` |
| `model/src/temp_integrate.F#9` | -356,7 +343,7 | forward value | rule: executable statement | ALLOW_GENERIC_ADVDIFF & ALLOW_ADAMSBASHFORTH_3 | `- I           tempVertDiff4, useGMRedi, useKPP,` |
| `model/src/temp_integrate.F#10` | -371,11 +358,11 | forward value | rule: executable statement | ALLOW_GENERIC_ADVDIFF & !(ALLOW_ADAMSBASHFORTH_3) | `- I           tempVertDiff4, useGMRedi, useKPP,` |
| `model/src/temp_integrate.F#11` | -417,6 +404,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gT_loc(:,:,k) = comlev1_bibj_k, key = kkey, kind = isbyte` |
| `model/src/temp_integrate.F#12` | -425,10 +415,8 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & ALLOW_ADAMSBASHFORTH_3 & ALLOW_AUTODIFF_TAMC | `- CADJ STORE gtNm(:,:,k,bi,bj,1) = comlev1_bibj_k, key=kkey,` |
| `model/src/temp_integrate.F#13` | -439,6 +427,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & !(ALLOW_ADAMSBASHFORTH_3) & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gtNm1(:,:,k,bi,bj) = comlev1_bibj_k, key=kkey, kind = ...` |
| `model/src/temp_integrate.F#14` | -453,6 +444,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_DOWN_SLOPE & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE recip_hFac = comlev1_bibj, key = tkey, kind = isbyte` |
| `model/src/temp_integrate.F#15` | -480,16 +474,17 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC / ALLOW_GENERIC_ADVDIFF & INCLUDE_IMPLVERTADV_CODE & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gT_loc (:,:,:)     = comlev1_bibj, key = tkey, kind = ...` |
| `model/src/temp_integrate.F#16` | -501,10 +496,6 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- CADJ STORE kappaRk(:,:,:) = comlev1_bibj , key=itdkey, byte=isbyte` |
| `model/src/temp_integrate.F#17` | -512,17 +503,6 | forward value | rule: block closer; executable statement | ALLOW_GENERIC_ADVDIFF & ALLOW_TIMEAVE | `- useVariableK = useKPP .OR. usePP81 .OR. useKL10 .OR. useMY82` |
| `model/src/salt_integrate.F#2` | -62,9 +59,8 | AD-only | rule: CPP condition differs only in AD macro names; inside an AD-only #if | ALLOW_AUTODIFF | `- #ifdef ALLOW_AUTODIFF` |
| `model/src/salt_integrate.F#3` | -140,6 +136,11 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `+ INTEGER tkey, kkey` |
| `model/src/salt_integrate.F#4` | -169,16 +170,7 | AD-only | rule: inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `model/src/salt_integrate.F#5` | -223,15 +215,17 | AD-only | override: same restructuring as temp_integrate.F#5 for salt | ALLOW_GENERIC_ADVDIFF | `+ #endif /* ALLOW_AUTODIFF */` |
| `model/src/salt_integrate.F#6` | -239,6 +233,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & INCLUDE_CALC_DIFFUSIVITY_CALL & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE kappaRk = comlev1_bibj, key = tkey, kind = isbyte` |
| `model/src/salt_integrate.F#7` | -254,7 +251,8 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & !DISABLE_MULTIDIM_ADVECTION & GAD_ALLOW_TS_SOM_ADV & ALLOW_AUTODIFF_TAMC | `- CADJ STORE som_S = comlev1_bibj, key=itdkey, byte=isbyte` |
| `model/src/salt_integrate.F#8` | -289,25 +287,20 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & !(ALLOW_ADAMSBASHFORTH_3) / ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC & ALLOW_ADAMSBASHFORTH_3 | `- kkey = (itdkey-1)*Nr + k` |
| `model/src/salt_integrate.F#9` | -348,7 +341,7 | forward value | rule: executable statement | ALLOW_GENERIC_ADVDIFF & ALLOW_ADAMSBASHFORTH_3 | `- I           saltVertDiff4, useGMRedi, useKPP,` |
| `model/src/salt_integrate.F#10` | -363,7 +356,7 | forward value | rule: executable statement | ALLOW_GENERIC_ADVDIFF & !(ALLOW_ADAMSBASHFORTH_3) | `- I           saltVertDiff4, useGMRedi, useKPP,` |
| `model/src/salt_integrate.F#11` | -409,6 +402,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gS_loc(:,:,k) = comlev1_bibj_k, key = kkey, kind = isbyte` |
| `model/src/salt_integrate.F#12` | -417,10 +413,8 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & ALLOW_ADAMSBASHFORTH_3 & ALLOW_AUTODIFF_TAMC | `- CADJ STORE gsNm(:,:,k,bi,bj,1) = comlev1_bibj_k, key=kkey,` |
| `model/src/salt_integrate.F#13` | -431,6 +425,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & NONLIN_FRSURF & !(ALLOW_ADAMSBASHFORTH_3) & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gsNm1(:,:,k,bi,bj) = comlev1_bibj_k, key=kkey, kind = ...` |
| `model/src/salt_integrate.F#14` | -445,6 +442,9 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_DOWN_SLOPE & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE recip_hFac = comlev1_bibj, key = tkey, kind = isbyte` |
| `model/src/salt_integrate.F#15` | -472,17 +472,18 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC / ALLOW_GENERIC_ADVDIFF & INCLUDE_IMPLVERTADV_CODE & ALLOW_AUTODIFF_TAMC | `+ CADJ STORE gS_loc(:,:,:)     = comlev1_bibj, key = tkey, kind = i...` |
| `model/src/salt_integrate.F#16` | -493,10 +494,6 | AD-only | rule: CADJ/TAF directive | ALLOW_GENERIC_ADVDIFF & ALLOW_AUTODIFF_TAMC | `- CADJ STORE kappaRk(:,:,:) = comlev1_bibj , key=itdkey, byte=isbyte` |
| `pkg/exf/exf_getforcing.F#2` | -132,6 +129,15 | forward value | rule: executable statement | ( defined ALLOW_DOWNWARD_RADIATION ) \|\| \ | `+ ( defined ALLOW_ATM_TEMP && defined ALLOW_BULKFORMULAE )` |
| `pkg/exf/exf_getforcing.F#3` | -142,21 +148,53 | forward value | rule: block closer; executable statement | ( defined ALLOW_DOWNWARD_RADIATION ) \|\| \ | `+ ( defined ALLOW_ATM_TEMP && defined ALLOW_BULKFORMULAE )` |
| `pkg/exf/exf_getforcing.F#4` | -166,37 +204,84 | forward value | rule: executable statement | ALLOW_ATM_TEMP & ALLOW_BULKFORMULAE / ALLOW_DOWNWARD_RADIATION | `- CALL EXF_RADIATION( myTime, myIter, myThid )` |
| `pkg/exf/exf_getforcing.F#5` | -205,35 +290,34 | forward value | rule: block closer; executable statement |  | `- k = 1` |
| `pkg/exf/exf_radiation.F#1` | -1,106 +1,75 | forward value | rule: block closer; executable statement | ALLOW_DOWNWARD_RADIATION & ALLOW_ATM_TEMP | `- SUBROUTINE EXF_RADIATION( myTime, myIter, myThid )` |
| `pkg/exf/exf_radiation.F#2` | -108,9 +77,8 | forward value | rule: executable statement | ALLOW_DOWNWARD_RADIATION & ALLOW_ATM_TEMP | `- ENDDO` |
| `pkg/exf/exf_radiation.F#4` | -140,7 +108,7 | AD-only | rule: CPP condition differs only in AD macro names | ALLOW_DOWNWARD_RADIATION & defined(ALLOW_ATM_TEMP) \|\| defined(SHORTWAVE_HEATING) & ALLOW_ZENITHANGLE | `- #ifdef ALLOW_AUTODIFF_TAMC` |
| `pkg/exf/exf_bulkformulae.F#2` | -9,7 +6,7 | forward value | rule: executable statement |  | `- SUBROUTINE EXF_BULKFORMULAE( myTime, myIter, myThid )` |
| `pkg/exf/exf_bulkformulae.F#5` | -216,15 +215,14 | AD-only | rule: inside an AD-only #if | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP & ALLOW_AUTODIFF_TAMC | `- INTEGER ikey_1` |
| `pkg/exf/exf_bulkformulae.F#6` | -239,7 +237,15 | forward value | rule: block closer; executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP | `+ IF ( usingPCoords ) THEN` |
| `pkg/exf/exf_bulkformulae.F#7` | -248,35 +254,24 | forward value | rule: executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP | `- ksrf   = 1` |
| `pkg/exf/exf_bulkformulae.F#8` | -287,14 +282,7 | forward value | rule: block closer; executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP | `- Tsf = theta(i,j,ksrf,bi,bj) + cen2kel` |
| `pkg/exf/exf_bulkformulae.F#9` | -319,8 +307,18 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP / ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP & ALLOW_DRAG_LARGEYEAGER09 | `+ #ifdef  ALLOW_DRAG_LARGEYEAGER09` |
| `pkg/exf/exf_bulkformulae.F#10` | -360,13 +358,8 | AD-only | rule: inside an AD-only #if | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP & ALLOW_AUTODIFF_TAMC | `- ikey_2 = i` |
| `pkg/exf/exf_bulkformulae.F#11` | -420,7 +413,9 | AD-only | rule: CPP structure change (class of the wrapped statements) | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP | `+ #ifdef ALLOW_AUTODIFF_TAMC` |
| `pkg/exf/exf_bulkformulae.F#12` | -437,8 +432,18 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP / ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP & ALLOW_DRAG_LARGEYEAGER09 | `+ #ifdef  ALLOW_DRAG_LARGEYEAGER09` |
| `pkg/exf/exf_bulkformulae.F#13` | -476,12 +481,7 | AD-only | rule: inside an AD-only #if | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP & ALLOW_AUTODIFF_TAMC | `- ikey_1 = i` |
| `pkg/exf/exf_bulkformulae.F#14` | -498,7 +498,13 | forward value | rule: executable statement | ALLOW_BULKFORMULAE & ALLOW_ATM_TEMP | `- IF ( useAtmWind ) THEN` |
| `pkg/seaice/seaice_model.F#1` | -1,7 +1,7 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_EXF | `+ # include "EXF_OPTIONS.h"` |
| `pkg/seaice/seaice_model.F#3` | -40,7 +41,6 | forward value | rule: #include of an options header (may change the compiled code) | ALLOW_EXF | `- # include "EXF_OPTIONS.h"` |
| `pkg/seaice/seaice_model.F#4` | -48,42 +48,94 | forward value | rule: block closer; executable statement | ALLOW_DIAGNOSTICS / ALLOW_EXF | `- CALL EXCH_UV_AGRID_3D_RL( uwind, vwind, .TRUE., 1, myThid )` |
| `pkg/seaice/seaice_model.F#5` | -99,112 +151,153 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | !(DISABLE_SEAICE_GROWTH) / !(DISABLE_SEAICE_GROWTH) & SEAICE_USE_GROWTH_ADX / !(SEAICE_BGRID_DYNAMICS) / !DISABLE_SEAICE_GROWTH / ALLOW_EXF / ALLOW_THSICE / ALLOW_THSICE & !OLD_THSICE_CALL_SEQUENCE / DISABLE_SEAICE_GROWTH / SEAICE_BGRID_DYNAMICS / SEAICE_CGRID | `+ #endif /* ALLOW_AUTODIFF */` |
| `pkg/seaice/seaice_model.F#6` | -272,16 +365,30 | forward value | rule: block closer; executable statement | ALLOW_DIAGNOSTICS | `+ IF ( diag_SIenph_isOn ) THEN` |
| `pkg/seaice/seaice_model.F#7` | -293,9 +400,9 | AD-only | rule: CPP condition differs only in AD macro names; inside an AD-only #if | ALLOW_EXF / ALLOW_EXF & ALLOW_AUTODIFF / ALLOW_EXF & ALLOW_AUTODIFF & ALLOW_AUTODIFF_MONITOR / ALLOW_EXF & ALLOW_AUTODIFF_TAMC / ALLOW_EXF & ALLOW_AUTODIFF_TAMC & (defined (ALLOW_AUTODIFF_MONITOR)) | `- # ifdef ALLOW_AUTODIFF_TAMC` |
| `pkg/seaice/seaice_dynsolver.F#4` | -49,228 +46,262 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & !ALLOW_AUTODIFF / SEAICE_CGRID & !ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & ATMOSPHERIC_LOADING / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- #ifdef SEAICE_CGRID` |
| `pkg/seaice/seaice_dynsolver.F#5` | -289,84 +320,77 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- ENDIF` |
| `pkg/seaice/seaice_dynsolver.F#6` | -384,8 +408,336 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ( defined ALLOW_DIAGNOSTICS && defined SEAICE_CGRID ) / ( defined ALLOW_DIAGNOSTICS && defined SEAICE_CGRID ) & !(ALLOW_AUTODIFF) / ( defined ALLOW_DIAGNOSTICS && defined SEAICE_CGRID ) & !(SEAICE_ALLOW_EVP) / ( defined ALLOW_DIAGNOSTICS && defined SEAICE_CGRID ) & SEAICE_ALLOW_EVP / ( defined ALLOW_DIAGNOSTICS && defined SEAICE_CGRID ) & SEAICE_ALLOW_SIDEDRAG / SEAICE_CGRID | `- #endif /* SEAICE_ALLOW_DYNAMICS */` |
| `pkg/seaice/seaice_lsr.F#2` | -52,10 +49,10 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- # include "AUTODIFF_PARAMS.h"` |
| `pkg/seaice/seaice_lsr.F#3` | -70,7 +67,6 | forward value | rule: CPP structure change (class of the wrapped statements) | SEAICE_CGRID | `- #ifdef SEAICE_ALLOW_DYNAMICS` |
| `pkg/seaice/seaice_lsr.F#5` | -121,21 +128,28 | AD-only | rule: inside an AD-only #if | SEAICE_CGRID & ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & ALLOW_AUTODIFF_TAMC | `- INTEGER itmpkey, itmpkey2, itmpkey3` |
| `pkg/seaice/seaice_lsr.F#7` | -163,17 +178,27 | forward value | rule: block closer; executable statement | SEAICE_CGRID | `+ IF ( usingPCoords ) THEN` |
| `pkg/seaice/seaice_lsr.F#8` | -200,63 +225,86 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS / SEAICE_CGRID & SEAICE_ALLOW_LSR_FLEX | `- areaW(I,J,bi,bj) = 1. _d 0` |
| `pkg/seaice/seaice_lsr.F#9` | -267,56 +315,54 | forward value | rule: block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- IF ( ipass .EQ. 1 ) THEN` |
| `pkg/seaice/seaice_lsr.F#10` | -325,42 +371,54 | forward value | rule: executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_BOTTOMDRAG / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & SEAICE_ALLOW_BOTTOMDRAG / SEAICE_CGRID & SEAICE_ALLOW_SIDEDRAG | `- I     e11, e22, e12, zMin, zMax, hEffM, press0, tensileStrFac,` |
| `pkg/seaice/seaice_lsr.F#11` | -372,60 +430,54 | forward value | rule: executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- FORCEX(I,J,bi,bj)=FORCEX0(I,J,bi,bj)+` |
| `pkg/seaice/seaice_lsr.F#13` | -447,14 +499,36 | forward value | rule: block closer; executable statement | SEAICE_CGRID & SEAICE_ALLOW_MOM_ADVECTION | `+ IF ( SEAICEmomAdvection ) THEN` |
| `pkg/seaice/seaice_lsr.F#14` | -465,32 +539,49 | forward value | rule: block closer; executable statement | SEAICE_CGRID & SEAICE_ALLOW_SIDEDRAG | `+ IF ( SEAICEsideDrag .NE. 0. _d 0 ) THEN` |
| `pkg/seaice/seaice_lsr.F#15` | -512,8 +603,9 | forward value | rule: executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- IF ( printResidual .OR. LSR_mixIniGuess.GE.1 )` |
| `pkg/seaice/seaice_lsr.F#16` | -530,13 +622,13 | forward value | rule: executable statement | SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & SEAICE_ALLOW_FREEDRIFT / SEAICE_CGRID & SEAICE_ALLOW_FREEDRIFT | `- &                 * areaW(i,j,bi,bj)` |
| `pkg/seaice/seaice_lsr.F#17` | -553,10 +645,8 | AD-only | rule: CADJ/TAF directive | SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & SEAICE_ALLOW_FREEDRIFT & ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_FREEDRIFT & ALLOW_AUTODIFF_TAMC | `- CADJ STORE uice, vice = comlev1_dynsol, kind=isbyte,` |
| `pkg/seaice/seaice_lsr.F#18` | -612,44 +702,95 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & !((defined (ALLOW_AUTODIFF_TAMC) && defined (SEAICE_LSR_ADJOINT_ITER))) / SEAICE_CGRID & SEAICE_ALLOW_LSR_FLEX | `- #if (defined (ALLOW_AUTODIFF_TAMC) && defined (SEAICE_LSR_ADJOINT...` |
| `pkg/seaice/seaice_lsr.F#19` | -660,11 +801,11 | AD-only | rule: CADJ/TAF directive | SEAICE_CGRID & ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & ALLOW_AUTODIFF_TAMC | `- CADJ STORE uice, vice = comlev1_lsr, kind=isbyte, key = itmpkey2` |
| `pkg/seaice/seaice_lsr.F#20` | -688,9 +829,9 | forward value | rule: executable statement | SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & SEAICE_GLOBAL_3DIAG_SOLVER / SEAICE_CGRID & SEAICE_GLOBAL_3DIAG_SOLVER | `- &             + vRt1(i,j,bi,bj)*vIce(i-1,J,bi,bj)` |
| `pkg/seaice/seaice_lsr.F#21` | -698,7 +839,7 | forward value | rule: executable statement | SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & SEAICE_GLOBAL_3DIAG_SOLVER / SEAICE_CGRID & SEAICE_GLOBAL_3DIAG_SOLVER | `- iSubIter = SOLV_MAX_TMP*(ipass-1) + m` |
| `pkg/seaice/seaice_lsr.F#22` | -735,25 +876,16 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | SEAICE_CGRID & ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `pkg/seaice/seaice_lsr.F#24` | -781,14 +913,34 | forward value | rule: block closer; executable statement | SEAICE_CGRID & SEAICE_ALLOW_LSR_FLEX | `+ IF ( SEAICEuseLSRflex ) THEN` |
| `pkg/seaice/seaice_lsr.F#26` | -828,6 +980,9 | forward value | rule: executable statement | SEAICE_CGRID & SEAICE_ALLOW_LSR_FLEX | `+ ENDIF` |
| `pkg/seaice/seaice_lsr.F#27` | -836,6 +991,13 | forward value | rule: block closer; executable statement | SEAICE_CGRID & SEAICE_ALLOW_LSR_FLEX | `+ IF ( SEAICEuseLSRflex ) THEN` |
| `pkg/seaice/seaice_lsr.F#28` | -849,8 +1011,9 | diagnostics | rule: diagnostics statement | SEAICE_CGRID & !ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & !ALLOW_AUTODIFF_TAMC | `- WRITE(standardMessageUnit,'(A,1X,I4,1P3E16.8)')` |
| `pkg/seaice/seaice_lsr.F#29` | -863,10 +1026,10 | diagnostics | rule: diagnostics statement | SEAICE_CGRID & !ALLOW_AUTODIFF_TAMC / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS & !ALLOW_AUTODIFF_TAMC | `- WRITE(standardMessageUnit,'(A,I4,A,I6,1P2E16.8)')` |
| `pkg/seaice/seaice_lsr.F#32` | -935,24 +1098,49 | forward value | rule: block closer; executable statement | SEAICE_CGRID & ALLOW_DIAGNOSTICS | `+ CALL EXCH_UV_XY_RL( uTmp, vTmp,.TRUE.,myThid)` |
| `pkg/seaice/seaice_lsr.F#33` | -963,57 +1151,18 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- DO j=1-OLy,sNy+OLy` |
| `pkg/seaice/seaice_lsr.F#34` | -1077,9 +1226,6 | forward value | rule: CPP structure change (class of the wrapped statements) | SEAICE_CGRID | `- #ifdef SEAICE_CGRID` |
| `pkg/seaice/seaice_lsr.F#35` | -1139,16 +1285,13 | forward value | rule: CPP structure change (class of the wrapped statements) | SEAICE_CGRID | `- #endif /* SEAICE_ALLOW_DYNAMICS */` |
| `pkg/seaice/seaice_lsr.F#37` | -1207,8 +1351,6 | forward value | rule: CPP structure change (class of the wrapped statements) | SEAICE_CGRID | `- #ifdef SEAICE_CGRID` |
| `pkg/seaice/seaice_lsr.F#44` | -1511,89 +1654,112 | forward value | rule: block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- kSrf = 1` |
| `pkg/seaice/seaice_lsr.F#46` | -1642,90 +1809,112 | forward value | rule: block closer; executable statement | SEAICE_CGRID / SEAICE_CGRID & SEAICE_ALLOW_DYNAMICS | `- kSrf = 1` |
| `pkg/seaice/seaice_lsr.F#52` | -1939,94 +2128,92 | forward value | rule: CPP structure change (class of the wrapped statements) | SEAICE_CGRID | `- #endif /* SEAICE_ALLOW_DYNAMICS */` |
| `pkg/seaice/seaice_advdiff.F#4` | -71,10 +64,9 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- INTEGER itmpkey` |
| `pkg/seaice/seaice_advdiff.F#6` | -102,92 +91,70 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_GENERIC_ADVDIFF / SEAICE_BGRID_DYNAMICS | `- ks = 1` |
| `pkg/seaice/seaice_advdiff.F#7` | -209,16 +176,125 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_GENERIC_ADVDIFF / ALLOW_GENERIC_ADVDIFF & !(SEAICE_ITD) / ALLOW_GENERIC_ADVDIFF & SEAICE_ITD / SEAICE_ITD | `+ #ifdef SEAICE_ITD` |
| `pkg/seaice/seaice_advdiff.F#8` | -242,26 +318,10 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO j=1-OLy,sNy+OLy` |
| `pkg/seaice/seaice_advdiff.F#9` | -285,50 +345,10 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO j=1-OLy,sNy+OLy` |
| `pkg/seaice/seaice_advdiff.F#10` | -352,21 +372,7 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO j=1-OLy,sNy+OLy` |
| `pkg/seaice/seaice_advdiff.F#14` | -544,25 +551,35 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | SEAICE_ITD | `+ #else /* not ALLOW_GENERIC_ADVDIFF */` |
| `pkg/seaice/seaice_advdiff.F#15` | -577,35 +594,11 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO bj=myByLo(myThid),myByHi(myThid)` |
| `pkg/seaice/seaice_advdiff.F#16` | -620,50 +613,11 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO bj=myByLo(myThid),myByHi(myThid)` |
| `pkg/seaice/seaice_advdiff.F#17` | -678,31 +632,8 | forward value | rule: block closer; executable statement | SEAICE_ITD | `- DO bj=myByLo(myThid),myByHi(myThid)` |
| `pkg/seaice/seaice_growth.F#3` | -56,6 +54,7 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `+ #ifndef SEAICE_USE_GROWTH_ADX` |
| `pkg/seaice/seaice_growth.F#6` | -160,11 +164,12 | AD-only | rule: inside an AD-only #if | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- INTEGER ilockey` |
| `pkg/seaice/seaice_growth.F#11` | -325,23 +332,31 | forward value | rule: block closer; executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SHORTWAVE_HEATING / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- IF ( buoyancyRelation .EQ. 'OCEANICP' ) THEN` |
| `pkg/seaice/seaice_growth.F#12` | -395,110 +410,94 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & defined ( ALLOW_SITRACER ) && defined ( SEAICE_GREASE ) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & (defined (ALLOW_MEAN_SFLUX_COST_CONTRIBUTION) \|\| defined (ALLOW_SSH_GLOBMEAN_COST_CONTRIBUTION)) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SEAICE_GREASE | `- #ifdef SEAICE_GREASE` |
| `pkg/seaice/seaice_growth.F#15` | -586,23 +585,23 | AD-only | rule: CPP condition differs only in AD macro names | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- #if (defined ALLOW_AUTODIFF_TAMC && defined SEAICE_MODIFY_GROWTH_...` |
| `pkg/seaice/seaice_growth.F#16` | -610,76 +609,68 | forward value | rule: block closer; executable statement | (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & (defined (ALLOW_MEAN_SFLUX_COST_CONTRIBUTION) \|\| defined (ALLOW_SSH_GLOBMEAN_COST_CONTRIBUTION)) | `- DO J=1,sNy` |
| `pkg/seaice/seaice_growth.F#18` | -733,15 +724,15 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE qnet(:,:,bi,bj) = comlev1_bibj, key = iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#19` | -755,43 +746,39 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- &          (uWind(I,J,bi,bj)` |
| `pkg/seaice/seaice_growth.F#21` | -814,19 +801,19 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC & SEAICE_CAP_SUBLIM / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC & SEAICE_CAP_SUBLIM | `- CADJ STORE heffActualMult = comlev1_bibj, key = iicekey, byte = i...` |
| `pkg/seaice/seaice_growth.F#23` | -843,24 +831,24 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC & SEAICE_CAP_SUBLIM / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC & SEAICE_CAP_SUBLIM | `- CADJ STORE heffActualMult = comlev1_bibj, key = iicekey, byte = i...` |
| `pkg/seaice/seaice_growth.F#26` | -918,104 +906,104 | AD-only | rule: CADJ/TAF directive; CPP condition differs only in AD macro names | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE AREApreTH       = comlev1_bibj, key = iicekey, byte = ...` |
| `pkg/seaice/seaice_growth.F#27` | -1028,28 +1016,28 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- tmpscal1 =SEAICE_frazilFrac*drF(kSurface)/SEAICE_deltaTtherm` |
| `pkg/seaice/seaice_growth.F#28` | -1057,8 +1045,8 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- &         * (theta(I,J,kSurface,bi,bj)-tempFrz)` |
| `pkg/seaice/seaice_growth.F#30` | -1086,41 +1074,41 | AD-only | rule: CPP condition differs only in AD macro names | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- #if (defined ALLOW_AUTODIFF_TAMC && defined SEAICE_MODIFY_GROWTH_...` |
| `pkg/seaice/seaice_growth.F#31` | -1128,28 +1116,39 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & defined ( ALLOW_SITRACER ) && defined ( SEAICE_GREASE ) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SEAICE_GREASE | `- #ifdef SEAICE_GREASE` |
| `pkg/seaice/seaice_growth.F#33` | -1177,8 +1176,8 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & defined ( ALLOW_SITRACER ) && defined ( SEAICE_GREASE ) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SEAICE_GREASE | `- &         ( tmpscal1 * uWind(I,J,bi,bj) + tmpscal4 * tmpscal2 )**2` |
| `pkg/seaice/seaice_growth.F#36` | -1220,67 +1219,67 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE hsnow(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#37` | -1293,48 +1292,48 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE heff(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#38` | -1346,45 +1345,45 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE hsnow(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#39` | -1393,8 +1392,8 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE heff(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#41` | -1456,56 +1455,56 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE a_QbyATM_cover = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#42` | -1522,46 +1521,46 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING & ALLOW_AUTODIFF_TAMC | `- CADJ STORE HSNOW(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#43` | -1574,107 +1573,115 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !(SHORTWAVE_HEATING) / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SHORTWAVE_HEATING / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- tmpscal1=r_QbyATM_open(I,J)+r_QbyOCN(i,j) *` |
| `pkg/seaice/seaice_growth.F#44` | -1685,41 +1692,41 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE heff(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#46` | -1753,23 +1760,23 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE d_HEFFbyATMonOCN = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#47` | -1777,45 +1784,45 | forward value | rule: CPP structure change (class of the wrapped statements) | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- #ifdef SEAICE_GREASE` |
| `pkg/seaice/seaice_growth.F#50` | -1917,24 +1924,24 | AD-only | rule: CPP condition differs only in AD macro names | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- #if (defined ALLOW_AUTODIFF_TAMC && defined SEAICE_MODIFY_GROWTH_...` |
| `pkg/seaice/seaice_growth.F#51` | -1950,17 +1957,17 | forward value | rule: CPP structure change (class of the wrapped statements) | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- #ifdef SEAICE_GREASE` |
| `pkg/seaice/seaice_growth.F#52` | -1972,45 +1979,45 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC / !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC & ALLOW_SALT_PLUME / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC & ALLOW_SALT_PLUME | `- CADJ &                              key = iicekey, byte = isbyte` |
| `pkg/seaice/seaice_growth.F#54` | -2044,51 +2051,51 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & SEAICE_VARIABLE_SALINITY & ALLOW_AUTODIFF_TAMC | `- CADJ STORE hsalt(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#58` | -2193,39 +2200,39 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- &             snowPrecip(i,j,bi,bj) * (ONE-AREApreTH(I,J))` |
| `pkg/seaice/seaice_growth.F#59` | -2234,15 +2241,15 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_DISABLE_HEATCONSFIX & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_DISABLE_HEATCONSFIX & ALLOW_AUTODIFF_TAMC | `- CADJ &                              key = iicekey, byte = isbyte` |
| `pkg/seaice/seaice_growth.F#60` | -2254,18 +2261,18 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_DISABLE_HEATCONSFIX / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !SEAICE_DISABLE_HEATCONSFIX | `- tmpscal3=rhoConstFresh*maskC(I,J,kSurface,bi,bj)*(` |
| `pkg/seaice/seaice_growth.F#63` | -2311,24 +2318,24 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) | `- SIatmQnt(I,J,bi,bj) =` |
| `pkg/seaice/seaice_growth.F#65` | -2357,84 +2364,47 | forward value | rule: block closer; executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & !(ALLOW_SSH_GLOBMEAN_COST_CONTRIBUTION) & ALLOW_MEAN_SFLUX_COST_CONTRIBUTION / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_SSH_GLOBMEAN_COST_CONTRIBUTION | `- &             +r_FWbySublim(I,J)` |
| `pkg/seaice/seaice_growth.F#66` | -2447,20 +2417,20 | AD-only | rule: CADJ/TAF directive | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_AUTODIFF_TAMC | `- CADJ STORE heff(:,:,bi,bj) = comlev1_bibj,key=iicekey,byte=isbyte` |
| `pkg/seaice/seaice_growth.F#67` | -2471,7 +2441,7 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- IF ( balanceEmPmR ) THEN` |
| `pkg/seaice/seaice_growth.F#68` | -2482,11 +2452,12 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- IF ( balanceEmPmR.AND.(temp_EvPrRn.EQ.UNSET_RL) ) THEN` |
| `pkg/seaice/seaice_growth.F#69` | -2539,50 +2510,50 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_DIAGNOSTICS / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_DIAGNOSTICS | `- DO J=1,sNy` |
| `pkg/seaice/seaice_growth.F#70` | -2602,18 +2573,19 | forward value | rule: block closer; executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- FWFsiGlob=0. _d 0` |
| `pkg/seaice/seaice_growth.F#71` | -2628,10 +2600,10 | forward value | rule: executable statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- IF ( balanceEmPmR ) tmpscal2=tmpscal2-tmpscal1` |
| `pkg/seaice/seaice_growth.F#73` | -2661,8 +2633,8 | diagnostics | rule: diagnostics statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- WRITE(msgBuf,'(a,a,e24.17)') 'rm Global mean of ',` |
| `pkg/seaice/seaice_growth.F#74` | -2682,8 +2654,8 | diagnostics | rule: diagnostics statement | !SEAICE_USE_GROWTH_ADX & (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES / (defined ALLOW_EXF) && (defined ALLOW_ATM_TEMP) & ALLOW_BALANCE_FLUXES | `- WRITE(msgBuf,'(a,a,e24.17)') 'rm Global mean of ',` |
| `pkg/seaice/seaice_growth.F#75` | -2709,6 +2681,7 | forward value | rule: CPP structure change (class of the wrapped statements) |  | `+ #endif /* ndef SEAICE_USE_GROWTH_ADX */` |
| `model/inc/SIZE.h#3` | -51,17 +49,17 | forward value | rule: PARAMETER/DATA |  | `- &           sNx =  20,` |
| `pkg/exf/EXF_PARAM.h#44` | -755,210 +792,8 | forward value | rule: PARAMETER/DATA | !USE_EXF_INTERPOLATION / USE_EXF_INTERPOLATION | `- PARAMETER(MAX_LAT_INC = 1)` |
| `pkg/seaice/SEAICE_SIZE.h#2` | -32,7 +29,6 | AD-only | rule: inside an AD-only #if | ALLOW_SEAICE & ALLOW_AUTODIFF | `- INTEGER iicekey` |
| `pkg/seaice/SEAICE.h#2` | -28,119 +25,190 | forward value | rule: CPP structure change (class of the wrapped statements); PARAMETER/DATA | SEAICE_CGRID & ( defined SEAICE_ALLOW_JFNK \|\| defined SEAICE_ALLOW_KRYLOV ) | `+ #ifdef SEAICE_CGRID` |
| `pkg/seaice/SEAICE.h#4` | -173,91 +244,6 | forward value | rule: PARAMETER/DATA | (defined SEAICE_ALLOW_JFNK) \|\| (defined SEAICE_ALLOW_KRYLOV) | `- PARAMETER ( nVec=2*sNx*sNy )` |
| `eesupp/src/print.F#2` | -70,9 +67,13 | diagnostics | override: idString CHARACTER*9 -> *13 and a format string: process/thread tag of messages |  | `- CHARACTER*9 idString` |
| `eesupp/src/print.F#3` | -108,13 +109,23 | diagnostics | rule: diagnostics statement; message-printing file |  | `- WRITE(idString,'(I4.4,A,I4.4)') myProcId,'.',myThid` |
| `eesupp/src/print.F#4` | -198,9 +209,13 | diagnostics | override: same as print.F#2 in the second message routine |  | `- CHARACTER*9 idString` |
| `eesupp/src/print.F#5` | -229,15 +244,24 | diagnostics | rule: diagnostics statement; message-printing file |  | `- WRITE(idString,'(I4.4,A,I4.4)') myProcId,'.',myThid` |
| `eesupp/src/print.F#6` | -394,13 +418,13 | diagnostics | rule: diagnostics statement (logical IF) |  | `- &  WRITE(msgBuf(45:),'(A,1X,A,I3,1X,A)')` |
| `pkg/autodiff/autodiff_readparms.F#1` | -1,7 +1,7 | AD-only | rule: AD-only file | ALLOW_MOM_COMMON | `+ #ifdef ALLOW_MOM_COMMON` |
| `pkg/autodiff/autodiff_readparms.F#2` | -19,6 +19,13 | AD-only | rule: AD-only file | ALLOW_GENERIC_ADVDIFF / ALLOW_SEAICE | `+ #ifdef ALLOW_GENERIC_ADVDIFF` |
| `pkg/autodiff/autodiff_readparms.F#3` | -32,15 +39,24 | AD-only | rule: AD-only file | ALLOW_AUTODIFF / ALLOW_AUTODIFF & ALLOW_GENERIC_ADVDIFF | `+ #ifdef ALLOW_GENERIC_ADVDIFF` |
| `pkg/autodiff/autodiff_readparms.F#4` | -68,14 +84,19 | AD-only | rule: AD-only file | ALLOW_AUTODIFF | `+ cg2dFullAdjoint    = .FALSE.` |
| `pkg/autodiff/autodiff_readparms.F#5` | -92,15 +113,23 | AD-only | rule: AD-only file | ALLOW_AUTODIFF / ALLOW_AUTODIFF & !(SINGLE_DISK_IO) | `+ #ifdef SINGLE_DISK_IO` |
| `pkg/autodiff/autodiff_readparms.F#6` | -119,6 +148,9 | AD-only | rule: AD-only file | ALLOW_AUTODIFF | `+ viscFacAdj = viscFacInFw` |
| `pkg/autodiff/autodiff_readparms.F#7` | -127,7 +159,24 | AD-only | rule: AD-only file | ALLOW_AUTODIFF | `+ nRetired = 0` |
| `pkg/autodiff/autodiff_readparms.F#8` | -139,6 +188,11 | AD-only | rule: AD-only file | ALLOW_AUTODIFF | `+ CALL WRITE_0D_L( useApproxAdvectionInAdMode, INDEX_NONE,` |
| `pkg/autodiff/autodiff_readparms.F#9` | -165,13 +219,52 | AD-only | rule: AD-only file | ALLOW_AUTODIFF / ALLOW_AUTODIFF & !AUTODIFF_ALLOW_VISCFACADJ / ALLOW_AUTODIFF & (!defined ALLOW_3D_VISCAH && !defined ALLOW_3D_VISCA4) / ALLOW_AUTODIFF & ALLOW_3D_VISCA4 / ALLOW_AUTODIFF & ALLOW_3D_VISCAH | `+ CALL WRITE_0D_RL( viscFacInFw, INDEX_NONE,` |
| `pkg/autodiff/autodiff_readparms.F#10` | -190,6 +283,65 | AD-only | rule: AD-only file | ALLOW_AUTODIFF / ALLOW_AUTODIFF & !(ALLOW_GENERIC_ADVDIFF) / ALLOW_AUTODIFF & ALLOW_GENERIC_ADVDIFF / ALLOW_AUTODIFF & ALLOW_GENERIC_ADVDIFF & ALLOW_SEAICE | `+ IF ( useApproxAdvectionInAdMode ) THEN` |
| `pkg/autodiff/autodiff_inadmode_set_ad.F#1` | -1,16 +1,20 | AD-only | rule: AD-only file |  | `- SUBROUTINE ADAUTODIFF_INADMODE_SET( myThid )` |
| `pkg/autodiff/autodiff_inadmode_set_ad.F#2` | -18,28 +22,51 | AD-only | rule: AD-only file | (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED)) / ALLOW_CTRL / ALLOW_DIAGNOSTICS | `- #include "ctrl.h"` |
| `pkg/autodiff/autodiff_inadmode_set_ad.F#3` | -49,10 +76,13 | AD-only | rule: AD-only file | ALLOW_SEAICE | `+ IF ( SIregFacInAd .NE. UNSET_RL ) SINegFac = SIregFacInAd` |
| `pkg/autodiff/autodiff_inadmode_set_ad.F#4` | -81,14 +111,6 | AD-only | rule: AD-only file | (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED)) | `- #if (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED))` |
| `pkg/autodiff/autodiff_inadmode_unset_ad.F#1` | -1,16 +1,14 | AD-only | rule: AD-only file |  | `- SUBROUTINE ADAUTODIFF_INADMODE_UNSET( myThid )` |
| `pkg/autodiff/autodiff_inadmode_unset_ad.F#2` | -22,24 +20,47 | AD-only | rule: AD-only file | (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED)) / ALLOW_AUTODIFF_MONITOR / ALLOW_AUTODIFF_MONITOR & ALLOW_DIAGNOSTICS / ALLOW_CTRL | `- #include "ctrl.h"` |
| `pkg/autodiff/autodiff_inadmode_unset_ad.F#3` | -49,8 +70,10 | AD-only | rule: AD-only file | ALLOW_SEAICE | `+ IF ( SIregFacInFw .NE. UNSET_RL ) SINegFac = SIregFacInFw` |
| `pkg/autodiff/autodiff_inadmode_unset_ad.F#4` | -80,14 +103,6 | AD-only | rule: AD-only file | (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED)) | `- #if (defined (ALLOW_CTRL) && defined (ECCO_CTRL_DEPRECATED))` |
| `pkg/autodiff/autodiff_inadmode_set.F#1` | -1,21 +1,23 | AD-only | rule: AD-only file |  | `- SUBROUTINE AUTODIFF_INADMODE_SET( myThid )` |
| `pkg/autodiff/autodiff_inadmode_unset.F#1` | -1,21 +1,23 | AD-only | rule: AD-only file |  | `- SUBROUTINE AUTODIFF_INADMODE_UNSET( myThid )` |
| `pkg/autodiff/autodiff_inadmode.flow#1` | -1,13 +1,10 | AD-only | rule: CADJ/TAF directive |  | `- cadj SUBROUTINE autodiff_inadmode_set INPUT   = 1` |
| `pkg/autodiff/autodiff_inadmode.flow#2` | -18,10 +15,10 | AD-only | rule: CADJ/TAF directive |  | `- cadj SUBROUTINE autodiff_inadmode_unset INPUT   = 1` |
| `pkg/autodiff/AUTODIFF_OPTIONS.h#1` | -1,5 +1,7 | AD-only | rule: AD-only file | !AUTODIFF_OPTIONS_H | `+ #ifndef AUTODIFF_OPTIONS_H` |
| `pkg/autodiff/AUTODIFF_OPTIONS.h#2` | -13,11 +15,6 | AD-only | rule: AD-only file | !AUTODIFF_OPTIONS_H | `- #ifndef AUTODIFF_OPTIONS_H` |
| `pkg/autodiff/AUTODIFF_OPTIONS.h#3` | -29,34 +26,61 | AD-only | rule: AD-only file | !AUTODIFF_OPTIONS_H & ALLOW_AUTODIFF & !(ECCO_CPPOPTIONS_H) | `+ #undef AUTODIFF_TAMC_COMPATIBILITY` |
| `pkg/autodiff/g_zero_adj.F#1` | -0,0 +1,122 | AD-only | rule: AD-only file |  | `+ #include "CPP_EEOPTIONS.h"` |
| `pkg/mom_common/mom_calc_visc.F#2` | -14,7 +11,7 | forward value | rule: executable statement |  | `- I        hDiv,vort3,tension,strain,KE,hFacZ,` |
| `pkg/mom_common/mom_calc_visc.F#4` | -81,10 +79,9 | AD-only | rule: CPP condition differs only in AD macro names; inside an AD-only #if | ALLOW_AUTODIFF | `- #ifdef ALLOW_AUTODIFF` |
| `pkg/mom_common/mom_calc_visc.F#6` | -108,13 +106,16 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- INTEGER lockey_1, lockey_2` |
| `pkg/mom_common/mom_calc_visc.F#7` | -138,24 +139,21 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- act1 = bi - myBxLo(myThid)` |
| `pkg/mom_common/mom_calc_visc.F#8` | -182,11 +180,13 | forward value | rule: executable statement |  | `+ calcLeithQG = (viscC2LeithQG.NE.zeroRL)` |
| `pkg/mom_common/mom_calc_visc.F#9` | -202,19 +202,25 | forward value | rule: executable statement |  | `+ leithQG2fac = (viscC2LeithQG/pi)**6` |
| `pkg/mom_common/mom_calc_visc.F#10` | -227,25 +233,27 | forward value | rule: executable statement | ALLOW_LEITH_QG | `- visca4_zsmg(i,j) = 0. _d 0` |
| `pkg/mom_common/mom_calc_visc.F#11` | -307,6 +315,56 | forward value | rule: block closer; executable statement | ALLOW_LEITH_QG | `+ IF ( calcLeithQG ) THEN` |
| `pkg/mom_common/mom_calc_visc.F#12` | -316,12 +374,10 | AD-only | rule: CADJ/TAF directive; inside an AD-only #if | ALLOW_AUTODIFF_TAMC & !AUTODIFF_DISABLE_LEITH | `- lockey_2 = i+olx + (sNx+2*olx)*(j+oly-1)` |
| `pkg/mom_common/mom_calc_visc.F#13` | -361,14 +417,40 | forward value | rule: executable statement | !AUTODIFF_DISABLE_LEITH / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) & ALLOW_LEITH_QG / !AUTODIFF_DISABLE_LEITH & ALLOW_LEITH_QG | `- viscAh_DLth(i,j)=` |
| `pkg/mom_common/mom_calc_visc.F#14` | -380,22 +462,35 | forward value | rule: executable statement | !AUTODIFF_DISABLE_LEITH / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) / !AUTODIFF_DISABLE_LEITH & ALLOW_LEITH_QG | `- viscAh_Dlth(i,j)=(leith2fac*grdVrt+(leithD2fac*grdDiv))*L3` |
| `pkg/mom_common/mom_calc_visc.F#15` | -406,13 +501,16 | forward value | rule: executable statement |  | `- &          +viscAh_DLth(i,j)+viscAh_DSmg(i,j)` |
| `pkg/mom_common/mom_calc_visc.F#16` | -420,13 +518,13 | forward value | rule: executable statement |  | `- &          +viscA4Dfld(i,j,k,bi,bj)` |
| `pkg/mom_common/mom_calc_visc.F#17` | -472,14 +570,38 | forward value | rule: executable statement | !AUTODIFF_DISABLE_LEITH / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) & ALLOW_LEITH_QG / !AUTODIFF_DISABLE_LEITH & ALLOW_LEITH_QG | `- viscAh_ZLth(i,j)=` |
| `pkg/mom_common/mom_calc_visc.F#18` | -495,18 +617,30 | forward value | rule: executable statement | !AUTODIFF_DISABLE_LEITH / !AUTODIFF_DISABLE_LEITH & !(ALLOW_AUTODIFF) / !AUTODIFF_DISABLE_LEITH & ALLOW_LEITH_QG | `+ viscAh_ZLthQG(i,j)=leithQG2fac*(grdVrt + grdDiv)*L3` |
| `pkg/mom_common/mom_calc_visc.F#19` | -514,9 +648,15 | forward value | rule: executable statement |  | `- &           +viscAh_ZLth(i,j)+viscAh_ZSmg(i,j)` |
| `pkg/mom_common/mom_calc_visc.F#20` | -525,9 +665,12 | forward value | rule: executable statement |  | `- &          +viscA4Zfld(i,j,k,bi,bj)` |
| `pkg/mom_common/mom_calc_visc.F#21` | -618,13 +761,18 | diagnostics | rule: diagnostics statement | ALLOW_DIAGNOSTICS & ALLOW_LEITH_QG | `+ CALL DIAGNOSTICS_FILL(viscAh_DLthQG,'VAHDLTHQ',` |
| `pkg/gmredi/GMREDI_OPTIONS.h#2` | -21,16 +27,21 | forward value | rule: #define/#undef | !GMREDI_OPTIONS_H & ALLOW_GMREDI | `+ #undef GM_READ_K3D_REDI` |
| `pkg/gmredi/GMREDI_OPTIONS.h#3` | -47,9 +58,12 | forward value | rule: #define/#undef | !GMREDI_OPTIONS_H & ALLOW_GMREDI | `+ #undef ALLOW_GM_LEITH_QG` |
| `pkg/gmredi/gmredi_slope_limit.F#2` | -15,8 +12,8 | forward value | rule: executable statement |  | `- I             Lrho, hMixLay, depthZ, kLow,` |
| `pkg/gmredi/gmredi_slope_limit.F#4` | -53,17 +51,17 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- #include "tamc_keys.h"` |
| `pkg/gmredi/gmredi_slope_limit.F#6` | -90,55 +88,77 | forward value | rule: block closer; executable statement | ALLOW_GMREDI | `+ IF (kPos.EQ.3 ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#7` | -165,7 +185,7 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_CLIPPING) | `- IF ( tmpFld(i,j) .EQ. 0. ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#8` | -173,50 +193,34 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_CLIPPING) | `- IF (dSigmMod(i,j) .NE. 0.) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#9` | -243,15 +247,15 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_FM07_TAP) | `- IF ( dSigmaDr(i,j).GE. -GM_Small_Number )` |
| `pkg/gmredi/gmredi_slope_limit.F#10` | -286,18 +290,18 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_FM07_TAP) | `- ELSEIF ( dTransLay+hMixLay(i,j)+depthZ(k) .GE. 0. ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#12` | -398,32 +402,34 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_AC02_TAP) / ALLOW_GMREDI & !(GM_EXCLUDE_AC02_TAP) & !ALLOWW_AUTODIFF_TAMC / ALLOW_GMREDI & !(GM_EXCLUDE_AC02_TAP) & !ALLOW_AUTODIFF_TAMC | `- maxSlopeSqr = GM_maxSlope*GM_maxSlope` |
| `pkg/gmredi/gmredi_slope_limit.F#13` | -443,53 +449,52 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- IF ( dSigmaDr(i,j) .NE. 0. ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#14` | -497,9 +502,17 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- IF ( SlopeSqr(i,j) .GT. GM_slopeSqCutoff ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#15` | -509,13 +522,12 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- maxSlopeSqr = GM_maxSlope*GM_maxSlope` |
| `pkg/gmredi/gmredi_slope_limit.F#16` | -527,13 +539,12 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- maxSlopeSqr = GM_maxSlope*GM_maxSlope` |
| `pkg/gmredi/gmredi_slope_limit.F#17` | -547,11 +558,11 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- IF ( SlopeSqr(i,j) .EQ. 0. ) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#18` | -562,18 +573,19 | forward value | rule: executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) | `- IF (SlopeSqr(i,j) .EQ. 0.) THEN` |
| `pkg/gmredi/gmredi_slope_limit.F#19` | -581,36 +593,48 | forward value | rule: CPP structure change (class of the wrapped statements); block closer; executable statement | ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) / ALLOW_GMREDI & !(GM_EXCLUDE_TAPERING) & GMREDI_WITH_STABLE_ADJOINT | `- #ifndef GMREDI_WITH_STABLE_ADJOINT` |
| `pkg/gmredi/gmredi_slope_psi.F#1` | -1,32 +1,36 | forward value | rule: executable statement |  | `- U             dSigmaDrW,dSigmaDrS,` |
| `pkg/gmredi/gmredi_slope_psi.F#2` | -34,14 +38,11 | AD-only | rule: inside an AD-only #if | ALLOW_AUTODIFF_TAMC | `- #include "tamc_keys.h"` |
| `pkg/gmredi/gmredi_slope_psi.F#3` | -50,41 +51,45 | forward value | rule: PARAMETER/DATA; executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC | `- PARAMETER(fpi=3.141592653589793047592d0)` |
| `pkg/gmredi/gmredi_slope_psi.F#4` | -112,8 +117,8 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- dSigmaDrLtd(i,j) = -(GM_Small_Number+` |
| `pkg/gmredi/gmredi_slope_psi.F#5` | -122,7 +127,7 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- IF (dSigmaDrW(i,j).GE.dSigmaDrLtd(i,j))` |
| `pkg/gmredi/gmredi_slope_psi.F#6` | -131,7 +136,7 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- SlopeX(i,j) = -SlopeX(i,j)/dSigmaDrW(i,j)` |
| `pkg/gmredi/gmredi_slope_psi.F#7` | -147,8 +152,8 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- dSigmaDrLtd(i,j) = -(GM_Small_Number+` |
| `pkg/gmredi/gmredi_slope_psi.F#8` | -157,7 +162,7 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- IF (dSigmaDrS(i,j).GE.dSigmaDrLtd(i,j))` |
| `pkg/gmredi/gmredi_slope_psi.F#9` | -166,7 +171,7 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_CLIPPING) | `- SlopeY(i,j) = -SlopeY(i,j)/dSigmaDrS(i,j)` |
| `pkg/gmredi/gmredi_slope_psi.F#10` | -191,34 +196,34 | forward value | rule: block closer; executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- IF (dSigmaDrW(i,j).GE.-GM_Small_Number)` |
| `pkg/gmredi/gmredi_slope_psi.F#11` | -228,31 +233,31 | forward value | rule: block closer; executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- IF (dSigmaDrS(i,j).GE.-GM_Small_Number)` |
| `pkg/gmredi/gmredi_slope_psi.F#12` | -268,37 +273,40 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- IF ( Smod .GT. GM_maxSlope .AND.` |
| `pkg/gmredi/gmredi_slope_psi.F#13` | -308,14 +316,14 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- Smod = ABS(SlopeX(i,j))` |
| `pkg/gmredi/gmredi_slope_psi.F#14` | -327,18 +335,19 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- f1=op5*( 1. _d 0 + TANH( (GM_Scrit-Smod)/GM_Sd ))` |
| `pkg/gmredi/gmredi_slope_psi.F#15` | -347,18 +356,19 | forward value | rule: executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) | `- f1=op5*( 1. _d 0 + TANH( (GM_Scrit-Smod)/GM_Sd ))` |
| `pkg/gmredi/gmredi_slope_psi.F#16` | -376,29 +386,30 | forward value | rule: block closer; executable statement | ALLOW_GMREDI & GM_BOLUS_ADVEC & !(GM_EXCLUDE_TAPERING) & GMREDI_WITH_STABLE_ADJOINT | `- slopeTmpSpec=ABS(SlopeX(i,j))` |
| `model/src/set_defaults.F#2` | -55,6 +52,7 | forward value | rule: executable statement |  | `+ useMin4hFacEdges    = .FALSE.` |
| `model/src/set_defaults.F#3` | -92,7 +90,6 | forward value | rule: executable statement |  | `- use3dCoriolis       = .TRUE.` |
| `model/src/set_defaults.F#4` | -103,6 +100,7 | forward value | rule: executable statement |  | `+ surf_pRef           = 101325. _d 0` |
| `model/src/set_defaults.F#5` | -119,6 +117,7 | forward value | rule: executable statement |  | `+ smag3D_diffCoeff    = 0. _d 0` |
| `model/src/set_defaults.F#6` | -126,6 +125,7 | forward value | rule: executable statement |  | `+ viscC2LeithQG       = 0. _d 0` |
| `model/src/set_defaults.F#7` | -135,6 +135,7 | forward value | rule: executable statement |  | `+ zRoughBot           = 0. _d 0` |
| `model/src/set_defaults.F#8` | -175,6 +176,7 | forward value | rule: executable statement |  | `+ sIceLoadFac         = 1. _d 0` |
| `model/src/set_defaults.F#9` | -184,6 +186,7 | forward value | rule: executable statement |  | `+ momTidalForcing     = .TRUE.` |
| `model/src/set_defaults.F#10` | -191,12 +194,14 | forward value | rule: executable statement |  | `+ temp_stayPositive   = .FALSE.` |
| `model/src/set_defaults.F#11` | -223,10 +228,10 | forward value | rule: executable statement |  | `- useEnergyConservingCoriolis = .FALSE.` |
| `model/src/set_defaults.F#12` | -260,7 +265,8 | forward value | rule: executable statement |  | `- balanceEmPmR        = .FALSE.` |
| `model/src/set_defaults.F#13` | -278,15 +284,16 | forward value | rule: executable statement |  | `+ cg2dMinItersNSA    = 0` |
| `model/src/set_defaults.F#14` | -345,12 +352,8 | forward value | rule: executable statement |  | `- taveFreq          = deltaT*0` |
| `model/src/set_defaults.F#15` | -382,12 +385,14 | forward value | rule: executable statement |  | `+ geoPotAnomFile  = ' '` |
| `model/src/ini_parms.F#7` | -186,25 +207,35 | I/O | rule: I/O statement |  | `- & useAreaViscLength,` |
| `model/src/ini_parms.F#8` | -215,37 +246,29 | I/O | rule: I/O statement |  | `- & viscAz, diffKzT, diffKzS, viscAp, diffKpT, diffKpS,` |
| `model/src/ini_parms.F#9` | -276,6 +299,9 | I/O | rule: I/O statement |  | `+ & useMin4hFacEdges, interViscAr_pCell, interDiffKr_pCell,` |
| `model/src/ini_parms.F#10` | -288,10 +314,11 | I/O | rule: I/O statement |  | `- & lambdaThetaFile, lambdaSaltFile,` |
| `model/src/ini_parms.F#11` | -311,7 +338,13 | forward value | rule: executable statement |  | `+ useJamartWetPoints = .FALSE.` |
| `model/src/ini_parms.F#12` | -337,8 +370,13 | forward value | rule: executable statement |  | `+ writeStatePrec  = UNSET_I` |
| `model/src/ini_parms.F#14` | -511,8 +549,21 | forward value | rule: block closer; executable statement | ALLOW_MOM_COMMON | `+ IF ( selectBotDragQuadr.EQ.-1 .AND. zRoughBot.NE.0. )` |
| `model/src/ini_parms.F#15` | -599,7 +650,47 | forward value | rule: block closer; executable statement | SHORTWAVE_HEATING | `+ IF ( selectPenetratingSW.EQ.UNSET_I ) THEN` |
| `model/src/ini_parms.F#16` | -611,6 +702,38 | forward value | rule: block closer; executable statement |  | `+ IF ( select3dCoriScheme.EQ.UNSET_I ) THEN` |
| `model/src/ini_parms.F#17` | -662,7 +785,7 | diagnostics | override: default printResidualFreq 0 -> -1 (1 if debugLevel>=debLevE, ini_parms.F:787-789): selects cg2d/cg3d residual printing only (cg2d.F:198,330,392) and its config_summary line |  | `- printResidualFreq = 0` |
| `model/src/ini_parms.F#18` | -800,6 +923,12 | forward value | rule: block closer; executable statement |  | `+ IF ( writeStatePrec .NE. UNSET_I ) THEN` |
| `model/src/ini_parms.F#19` | -818,6 +947,19 | forward value | rule: block closer; executable statement |  | `+ IF ( cg2dChkResFreq .NE. UNSET_I ) THEN` |
| `model/src/ini_parms.F#20` | -860,6 +1002,23 | forward value | rule: block closer; executable statement |  | `+ IF ( taveFreq .NE. UNSET_RL ) THEN` |
| `model/src/ini_parms.F#21` | -1030,8 +1189,6 | forward value | rule: executable statement |  | `- IF (taveFreq.NE.0..AND.taveFreq.LT.monitorFreq)` |
| `model/src/ini_parms.F#22` | -1053,10 +1210,21 | forward value | rule: block closer; executable statement |  | `+ interViscAr_pCell = .FALSE.` |
| `model/src/ini_parms.F#23` | -1195,6 +1363,10 | forward value | rule: executable statement |  | `+ IF ( phiEuler .NE. 0. _d 0 .OR. thetaEuler .NE. 0. _d 0` |
| `model/src/ini_parms.F#24` | -1233,44 +1405,6 | forward value | rule: block closer; executable statement |  | `- IF ( usingCartesianGrid ) THEN` |
| `model/src/ini_parms.F#26` | -1455,7 +1589,11 | I/O | rule: CPP structure change (class of the wrapped statements); I/O statement | !(SINGLE_DISK_IO) | `+ #ifdef SINGLE_DISK_IO` |
| `model/src/packages_boot.F#2` | -68,6 +65,7 | I/O | rule: I/O statement |  | `+ &          useOBSFIT,` |
| `model/src/packages_boot.F#3` | -81,11 +79,12 | I/O | rule: I/O statement |  | `+ &          useSTIC,` |
| `model/src/packages_boot.F#4` | -132,6 +131,7 | forward value | rule: executable statement |  | `+ useOBSFIT       =.FALSE.` |
| `model/src/packages_boot.F#5` | -145,11 +145,12 | forward value | rule: executable statement |  | `+ useSTIC         =.FALSE.` |
| `model/src/packages_boot.F#6` | -181,7 +182,11 | I/O | rule: CPP structure change (class of the wrapped statements); I/O statement | !(SINGLE_DISK_IO) | `+ #ifdef SINGLE_DISK_IO` |
| `model/src/packages_boot.F#7` | -200,10 +205,26 | forward value | rule: block closer; executable statement | !HAVE_NETCDF / ALLOW_CAL | `+ IF (useOBSFIT) THEN` |
| `model/src/packages_boot.F#8` | -313,6 +334,9 | diagnostics | rule: diagnostics statement | ALLOW_OBSFIT | `+ CALL PACKAGES_PRINT_MSG( useOBSFIT,     'OBSFIT',      ' ' )` |
| `model/src/packages_boot.F#9` | -355,6 +379,9 | diagnostics | rule: diagnostics statement | ALLOW_STEEP_ICECAVITY | `+ CALL PACKAGES_PRINT_MSG( useSTIC,   'STEEP_ICECAVITY', ' ' )` |
| `model/src/packages_boot.F#10` | -368,7 +395,7 | diagnostics | rule: diagnostics statement | ALLOW_ATM2D | `- CALL PACKAGES_PRINT_MSG( useATM2D,      'ATM2D',       ' ' )` |
| `model/src/packages_boot.F#11` | -430,10 +457,6 | forward value | rule: executable statement | ALLOW_TIMEAVE | `- locFlag = taveFreq.GT.0.` |
| `model/src/packages_readparms.F#5` | -264,17 +268,12 | forward value | rule: executable statement | ALLOW_MATRIX / ALLOW_SALT_PLUME / ALLOW_SEAICE / ALLOW_STREAMICE | `- CALL MATRIX_READPARMS ( myThid )` |
| `model/src/packages_readparms.F#6` | -282,9 +281,9 | forward value | rule: executable statement | ALLOW_STEEP_ICECAVITY / ALLOW_STREAMICE | `- CALL STREAMICE_READPARMS( myThid )` |
| `model/src/packages_readparms.F#7` | -292,6 +291,16 | forward value | rule: executable statement | ALLOW_SALT_PLUME / ALLOW_SEAICE | `+ CALL SEAICE_READPARMS( myThid )` |
| `model/src/packages_readparms.F#8` | -326,17 +335,22 | forward value | rule: executable statement | ALLOW_OBSFIT / ALLOW_PROFILES | `- CALL PROFILES_READPARMS ( myThid )` |
| `model/src/packages_readparms.F#9` | -372,21 +386,31 | forward value | rule: executable statement | ALLOW_NEST2W_CHILD / ALLOW_NEST2W_PARENT | `+ IF (useNest2W_child) CALL NEST2W_C_READPARMS( myThid )` |
| `eesupp/src/eeset_parms.F#1` | -1,13 +1,10 | forward value | rule: executable statement |  | `- SUBROUTINE EESET_PARMS ( doReport )` |
| `eesupp/src/eeset_parms.F#3` | -52,20 +51,23 | I/O | rule: I/O statement |  | `- & useCoupler, useNEST_PARENT, useNEST_CHILD, useOASIS,` |
| `eesupp/src/eeset_parms.F#4` | -110,6 +112,8 | forward value | rule: executable statement |  | `+ useNest2W_parent           = .FALSE.` |
| `eesupp/src/eeset_parms.F#5` | -129,29 +133,33 | forward value | rule: CPP structure change (class of the wrapped statements); executable statement | !(SINGLE_DISK_IO) / !(SINGLE_DISK_IO) & !(USE_FORTRAN_SCRATCH_FILES) & USE_PDAF | `+ # ifdef USE_FORTRAN_SCRATCH_FILES` |
| `eesupp/src/eeset_parms.F#7` | -176,20 +185,9 | forward value | rule: block closer; executable statement |  | `- 1000 CONTINUE` |
| `eesupp/src/eeset_parms.F#8` | -200,20 +198,27 | forward value | rule: block closer; executable statement |  | `+ ENDIF` |
| `eesupp/src/eeset_parms.F#10` | -250,7 +255,11 | I/O | rule: CPP structure change (class of the wrapped statements); I/O statement | !(SINGLE_DISK_IO) | `+ #ifdef SINGLE_DISK_IO` |
| `eesupp/src/open_copy_data_file.F#4` | -77,32 +76,37 | I/O | rule: I/O file; I/O statement | !(SINGLE_DISK_IO) & !(USE_FORTRAN_SCRATCH_FILES) / !(SINGLE_DISK_IO) & !(USE_FORTRAN_SCRATCH_FILES) & USE_PDAF / !(SINGLE_DISK_IO) & !(defined (TARGET_BGL) \|\| defined (TARGET_CRAYXT)) / !(SINGLE_DISK_IO) & USE_FORTRAN_SCRATCH_FILES / !(SINGLE_DISK_IO) & defined (TARGET_BGL) \|\| defined (TARGET_CRAYXT) / SINGLE_DISK_IO | `- OPEN(UNIT=scrUnit1, FILE=scratchFile1, STATUS='UNKNOWN')` |
| `eesupp/src/open_copy_data_file.F#5` | -112,20 +116,8 | I/O | rule: I/O file; I/O statement |  | `- DO WHILE ( .TRUE. )` |
| `eesupp/src/open_copy_data_file.F#6` | -137,17 +129,22 | I/O | rule: I/O file; I/O statement |  | `- iUnit = scrUnit2` |
| `eesupp/src/open_copy_data_file.F#7` | -166,7 +163,8 | I/O | rule: I/O statement | SINGLE_DISK_IO | `- OPEN(UNIT=scrUnit1, FILE=scratchFile1, STATUS='OLD')` |

### Cosmetic hunks

- `eesupp/src/exch_xy_rx.template`: 1
- `eesupp/src/exch_uv_xy_rx.template`: 1
- `eesupp/src/exch_z_3d_rx.template`: 1
- `eesupp/src/exch_uv_agrid_3d_rx.template`: 1
- `eesupp/src/exch_uv_bgrid_3d_rx.template`: 1
- `eesupp/src/exch_3d_rx.template`: 1
- `eesupp/src/exch_sm_3d_rx.template`: 1
- `eesupp/src/exch_uv_dgrid_3d_rx.template`: 1
- `eesupp/src/exch_uv_3d_rx.template`: 1
- `eesupp/src/exch_s3d_rx.template`: 1
- `eesupp/src/exch_xyz_rx.template`: 1
- `eesupp/src/exch_uv_xyz_rx.template`: 1
- `pkg/exch2/exch2_3d_rx.template`: 1
- `pkg/exch2/exch2_z_3d_rx.template`: 1
- `pkg/exch2/exch2_sm_3d_rx.template`: 1
- `pkg/exch2/exch2_s3d_rx.template`: 1
- `pkg/exch2/exch2_uv_3d_rx.template`: 1, 2, 3, 4
- `pkg/exch2/exch2_uv_agrid_3d_rx.template`: 1
- `pkg/exch2/exch2_uv_bgrid_3d_rx.template`: 1
- `pkg/exch2/exch2_uv_cgrid_3d_rx.template`: 1
- `pkg/exch2/exch2_uv_dgrid_3d_rx.template`: 1
- `pkg/exch2/exch2_rx1_cube.template`: 1
- `pkg/exch2/exch2_rx2_cube.template`: 1
- `pkg/exch2/exch2_get_rx1.template`: 1
- `pkg/exch2/exch2_get_rx2.template`: 1
- `pkg/exch2/exch2_put_rx1.template`: 1
- `pkg/exch2/exch2_put_rx2.template`: 1
- `pkg/exch2/exch2_send_rx1.template`: 1
- `pkg/exch2/exch2_send_rx2.template`: 1
- `pkg/exch2/exch2_recv_rx1.template`: 1
- `pkg/exch2/exch2_recv_rx2.template`: 1
- `pkg/exch2/exch2_get_scal_bounds.F`: 1
- `pkg/exch2/exch2_get_uv_bounds.F`: 1
- `pkg/exch2/W2_EXCH2_SIZE.h`: 1
- `pkg/exch2/W2_EXCH2_TOPOLOGY.h`: 1
- `pkg/exch2/W2_EXCH2_PARAMS.h`: 1
- `pkg/exch2/W2_EXCH2_BUFFER.h`: 1
- `pkg/exch2/W2_OPTIONS.h`: 1
- `pkg/exch2/w2_readparms.F`: 1
- `pkg/exch2/w2_map_procs.F`: 1
- `pkg/exch2/w2_e2setup.F`: 1
- `pkg/exch2/w2_eeboot.F`: 1, 2
- `pkg/exch2/w2_set_single_facet.F`: 1
- `pkg/exch2/w2_set_gen_facets.F`: 1
- `pkg/exch2/w2_set_cs6_facets.F`: 1, 3
- `pkg/exch2/w2_set_myown_facets.F`: 1
- `pkg/exch2/w2_set_map_tiles.F`: 1
- `pkg/exch2/w2_set_map_cumsum.F`: 1
- `pkg/exch2/w2_set_tile2tiles.F`: 1
- `pkg/exch2/w2_set_f2f_index.F`: 1
- `eesupp/src/fill_cs_corner_ag_rl.F`: 1
- `eesupp/src/fill_cs_corner_tr_rl.F`: 1
- `eesupp/src/fill_cs_corner_uv_rl.F`: 1
- `eesupp/src/fill_cs_corner_uv_rs.F`: 1
- `eesupp/src/exch0_rx.template`: 1
- `eesupp/src/exch1_rx.template`: 1
- `eesupp/src/exch1_rx_cube.template`: 1, 2
- `eesupp/src/exch1_uv_rx_cube.template`: 1, 2, 3, 6
- `eesupp/src/exch1_z_rx_cube.template`: 1, 2
- `eesupp/src/exch1_bg_rx_cube.template`: 1
- `eesupp/src/exch_rx_send_put_x.template`: 1
- `eesupp/src/exch_rx_send_put_y.template`: 1
- `eesupp/src/exch_rx_recv_get_x.template`: 1
- `eesupp/src/exch_rx_recv_get_y.template`: 1
- `eesupp/src/exch_init.F`: 1
- `eesupp/src/ini_communication_patterns.F`: 1
- `eesupp/src/exch_cycle_ebl.F`: 1
- `eesupp/inc/EXCH.h`: 1
- `eesupp/inc/EESUPPORT.h`: 1, 2
- `eesupp/inc/EEPARAMS.h`: 1, 3, 4, 5, 6
- `eesupp/inc/CPP_EEOPTIONS.h`: 1, 2, 6
- `eesupp/inc/CPP_EEMACROS.h`: 1
- `eesupp/src/global_sum_tile.F`: 1, 2, 3
- `eesupp/src/global_sum.F`: 1, 2, 3, 4, 5, 6, 7, 8
- `eesupp/src/global_max.F`: 1
- `eesupp/src/global_sum_singlecpu.F`: 1
- `eesupp/inc/GLOBAL_SUM.h`: 1, 2
- `eesupp/inc/GLOBAL_MAX.h`: 1
- `pkg/mdsio/mdsio_read_field.F`: 1
- `pkg/mdsio/mdsio_facef_read.F`: 1
- `pkg/mdsio/mdsio_rd_rec_rl.F`: 1
- `pkg/mdsio/mdsio_rd_rec_rs.F`: 1
- `pkg/mdsio/mdsio_seg4torl.F`: 1
- `pkg/mdsio/mdsio_seg8torl.F`: 1
- `pkg/mdsio/mdsio_seg4tors.F`: 1
- `pkg/mdsio/mdsio_seg8tors.F`: 1
- `pkg/mdsio/mdsio_pass_r4torl.F`: 1, 2
- `pkg/mdsio/mdsio_pass_r8torl.F`: 1, 2
- `pkg/mdsio/mdsio_pass_r4tors.F`: 1, 2
- `pkg/mdsio/mdsio_pass_r8tors.F`: 1, 2
- `pkg/mdsio/mdsio_buffertorl.F`: 1
- `pkg/mdsio/mdsio_buffertors.F`: 1
- `pkg/mdsio/mdsio_read_meta.F`: 1
- `pkg/mdsio/mdsio_check4file.F`: 1
- `pkg/mdsio/MDSIO_OPTIONS.h`: 1
- `pkg/mdsio/MDSIO_BUFF_3D.h`: 1
- `pkg/rw/read_fld_xy_rl.F`: 1
- `pkg/rw/read_fld_xy_rs.F`: 1
- `pkg/rw/read_fld_xyz_rl.F`: 1
- `pkg/rw/read_fld_xyz_rs.F`: 1
- `pkg/rw/read_rec.F`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 23, 24
- `pkg/rw/RW_OPTIONS.h`: 1
- `eesupp/src/mds_reclen.F`: 1
- `eesupp/src/mdsfindunit.F`: 1
- `model/src/cg2d.F`: 1, 2, 3, 4, 5, 6, 8, 12, 13
- `model/src/cg2d_sr.F`: 1, 2, 3, 4, 5, 8
- `model/src/cg2d_nsa.F`: 2, 4, 5
- `model/src/cg2d_ex0.F`: 1, 2, 3, 4, 5, 8
- `model/src/ini_cg2d.F`: 1, 2
- `model/src/update_cg2d.F`: 1
- `model/src/solve_for_pressure.F`: 1, 3, 4
- `model/inc/CG2D.h`: 1, 2, 3
- `model/inc/SOLVE_FOR_PRESSURE.h`: 1
- `model/inc/CPP_OPTIONS.h`: 1, 5
- `pkg/autodiff/AUTODIFF_PARAMS.h`: 1, 5
- `model/src/forward_step.F`: 1, 4, 5
- `model/src/the_main_loop.F`: 3, 16
- `model/src/adams_bashforth2.F`: 1
- `model/src/adams_bashforth3.F`: 1, 2, 3, 4
- `pkg/autodiff/tamc.h`: 1
- `model/src/do_oceanic_phys.F`: 3, 4, 5, 6, 10, 11, 13, 27
- `model/src/dynamics.F`: 2, 4, 6, 7, 9
- `model/src/thermodynamics.F`: 1
- `model/src/temp_integrate.F`: 1
- `model/src/salt_integrate.F`: 1
- `pkg/exf/exf_getforcing.F`: 1, 6, 7, 8
- `pkg/exf/exf_radiation.F`: 3
- `pkg/exf/exf_bulkformulae.F`: 1, 3, 4
- `pkg/seaice/seaice_model.F`: 2
- `pkg/seaice/seaice_dynsolver.F`: 1, 2, 3
- `pkg/seaice/seaice_lsr.F`: 1, 4, 6, 12, 23, 25, 30, 31, 36, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51
- `pkg/seaice/seaice_advdiff.F`: 1, 2, 3, 5, 11, 12, 13
- `pkg/seaice/seaice_growth.F`: 1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 17, 20, 22, 24, 25, 29, 32, 34, 35, 40, 45, 48, 49, 53, 55, 56, 57, 61, 62, 64, 72
- `model/inc/SIZE.h`: 1, 2
- `model/inc/PARAMS.h`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
- `model/inc/SURFACE.h`: 1, 2
- `model/inc/GRID.h`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- `model/inc/DYNVARS.h`: 1, 2, 3, 4, 5
- `model/inc/FFIELDS.h`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- `pkg/exf/EXF_PARAM.h`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
- `pkg/exf/EXF_FIELDS.h`: 1, 2, 3, 4, 5, 6, 7
- `pkg/seaice/SEAICE_SIZE.h`: 1
- `pkg/seaice/SEAICE_PARAMS.h`: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
- `pkg/seaice/SEAICE.h`: 1, 3
- `eesupp/src/print.F`: 1
- `pkg/autodiff/autodiff_inadmode_set_ad.F`: 5
- `pkg/autodiff/zero_adj.F`: 1
- `pkg/mom_common/mom_calc_visc.F`: 1, 3, 5
- `pkg/gmredi/GMREDI_OPTIONS.h`: 1
- `pkg/gmredi/gmredi_slope_limit.F`: 1, 3, 5, 11
- `model/src/set_defaults.F`: 1
- `model/src/ini_parms.F`: 1, 2, 3, 4, 5, 6, 13, 25
- `model/src/packages_boot.F`: 1
- `model/src/packages_readparms.F`: 1, 2, 3, 4
- `pkg/diagnostics/diagnostics_is_on.F`: 1
- `eesupp/src/nml_set_terminator.F`: 1
- `eesupp/src/nml_change_syntax.F`: 1
- `eesupp/src/eeset_parms.F`: 2, 6, 9
- `eesupp/src/open_copy_data_file.F`: 1, 2, 3

<!-- END AUDIT -->
