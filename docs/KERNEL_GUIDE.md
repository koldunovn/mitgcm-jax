# Kernel guide (mitjax physics code)

How a Fortran routine of MITgcm master becomes Python in `mitjax/model/src/` or `mitjax/pkg/<pkg>/`. Binding together
with `docs/PORTING_RULES.md`; the canonical examples are the Task 8 prototype kernels in `dev/prototype/style_farray.py`
(`mom_calc_ke`, `gad_dst3_adv_x`, `mom_vi_hdissip`), shown next to their Fortran in `dev/prototype/side_by_side.html`.

**Decision (Nikolay, 2026-10-01): Fortran-index arrays (`mitjax/farray.py`) for all physics code.** Basis (Task 8,
gates job 27826890): both styles were bitwise equal to gfortran, compiled to the same jaxpr and optimized HLO,
gave bitwise-equal gradients and the same A100 device time; the Fortran-index style reads in Fortran's index order and
bounds, keeps unwritten points automatically, and catches i/j swaps, out-of-bounds shifts and `Nr` vs `Nr+1` shapes
that plain slicing accepts silently.

## 1. Files and functions
- One `.py` per `.F`, same directory structure (`model/src/ini_grid.F` → `mitjax/model/src/ini_grid.py`,
  `pkg/mom_common/mom_calc_ke.F` → `mitjax/pkg/mom_common/mom_calc_ke.py`), file and function names in lower case.
- One function per subroutine, same name, the Fortran arguments in the Fortran order minus `bi, bj, myThid`, Fortran
  spelling (`uFld`, `KEscheme`). Keyword-only after `*`: `cfg` (static), `grid` (GRID.h), `params` (PARAMS.h by
  Fortran name, `Params` of `mitjax/model/src/ini_parms.py`: REAL values traced, LOGICAL/INTEGER/CHARACTER values
  static), and whatever common-block state the routine reads (`state`), each field with its Fortran name.
- An output argument (`O`) is also an input: points the Fortran does not write keep their prior values. Return the
  outputs in the Fortran argument order.
- Docstring: the Fortran routine header (`CBOP`…`CEOP` description and argument comments), the line
  `@63cdc0b <path>:<first>-<last>`, then notes: which `#ifdef` branches are ported for which experiment, which are not
  (each raises), and any statement-order or vectorisation argument (§4).

## 2. Declarations
- Every array is an `FArray` with its Fortran declaration: `FArray(data, "hFacW", i=(1-OLx, sNx+OLx),
  j=(1-OLy, sNy+OLy), k=(1, Nr))`. Bounds come from the declaration, never from a loop range (`Nr` vs `Nr+1` [P§5]).
  Storage is `[tile, k, j, i]`; arrays without `bi,bj` in their declaration are `tiled=False`.
- Locals: `uD4 = uDissip.local("uD4")` (same bounds, every point NaN, so a read of a point the Fortran never wrote shows
  up). Where the Fortran initialises a local, port the initialisation loop literally.
- `_RL` and `_RS` are float64 (no `-use_real4` build); `INTEGER` fields are int32/int64 arrays or static Python ints.

## 3. Statements
- One Python assignment per Fortran assignment, same operands, same order, same parentheses — never re-associate,
  factor or simplify; keep tendency arrays and summation order [P§3, F§2]. Port the live line, never a commented-out
  alternative. Add `# :NN` with the Fortran line number at each block.
- Loops: `j = loop_j(lo, hi)`, `i = loop_i(lo, hi)` with the DO bounds verbatim; shifts `i+1`, `j-1` inline; fixed
  indices as Python ints (`uT.at[sNx+OLx, j]`); writes `A = A.at[i, j].set(expr)`.
- **Literals:** the build has no `-fdefault-real-8` (`reference/optfile_levante_gfortran`), so a literal without a `D`
  exponent or `_d` is REAL*4: `0.125`, `0.5`, `2.` are exact, but `0.1` in Fortran is `float32(0.1)` promoted to
  double, not Python's `0.1`. Write such constants through `mitjax.config.fortran.real4("0.1")` (one rounding from
  the exact decimal, as gfortran; `mitjax/tests/test_real4.py`) and cite the line. `0.1 _d 0` / `0.1D0` is the
  double `0.1`.
  `PARAMETER`s are module constants with their citation (e.g. `oneSixth = 1.0 / 6.0  # GAD.h:96-97`).
- Pointwise `IF`/`ELSE` inside a loop → `jnp.where(cond, a, b)` with **both branches finite**: guard the operation
  (safe ops, `mitjax/ops/safe.py`) before it, because a forward `where` does not stop a backward 0·inf [E§6, F§3].
- `IF` on a logical or integer namelist value or a routine argument that selects a branch → a Python `if` on a static
  value (`cfg.use_flag("useShelfIce")`, a static namelist value through `mitjax/config`, or a static argument). CPP
  `#ifdef NAME` → `if cfg.cpp.NAME:`; SIZE.h values are `cfg.size.sNx`, `cfg.size.OLx`, … (the real config object of
  `mitjax/config`; the prototype's `cfg.sNx`/`cfg.NAME` was a stand-in). Float namelist values come from `params`
  (traced) and never decide a branch.
- An unported branch or option raises at trace time (`NotImplementedError` naming the routine and the option); a
  Fortran `STOP` raises with the Fortran message.
- Global sums and maxima go through `mitjax/eesupp` (fixed tile order, Fortran loop order) [E§5]; exchanges are called
  exactly where the Fortran calls `EXCH_*`, through `mitjax/eesupp` (Task 7b).
- **MAX/MIN** of REAL values: `MAX(a, b, p="a"|"b")` / `MIN(...)` of `mitjax/ops/fortran_minmax.py` (host side:
  `mitjax/ops/fortran_minmax_host.py`, with `max_chain`/`min_chain` for running extrema), with `# :NN` on the
  statement; never `jnp.maximum`/`jnp.minimum`. gfortran (11.2.0, -O0, x86-64) emits one `maxsd`/`minsd` per
  argument step and these return their source operand on a tie (+0 vs -0) or a NaN; which Fortran argument is the
  source differs from site to site (operand canonicalisation, then the RTL expander and register allocator: the
  GIMPLE order is wrong at about one site in four, e.g. `MAX(x, 0.D0)` keeps 0 in DIAGS_PHI_HYD but
  `max(abs(dfds(+0)), epsil)` keeps the first argument in GAD_PLM_FUN). So `p` is never chosen by hand: it is the
  winner recorded for that statement in `$MJX_REFERENCE/minmax_sites/<build>.json` (`tools/fortran_minmax_sites.py
  run <build dir>`; "a" = the first argument wins, "b" = the second; n arguments: one letter per step of the left
  fold, "a" = the running value). A statement function (e.g. `Limiter`) takes the winners of each caller's
  expansion. The derivative is JAX's own (`jnp.maximum`'s, half/half at an exact tie).
  A running MAX over tiles and a DO nest (`m = MAX(ABS(x), m)`, e.g. cg2d's rhsMax) is `MAX_CHAIN(init, values,
  p=, acc=, ex=)`: one chain in tile order (`ex.all_tiles`), exact on NaN (the new value winning NaN keeps only the
  elements after the last NaN), never an order-free `jnp.max`.
  `mitjax/tests/test_minmax_sites.py` checks every call in `mitjax/model` and `mitjax/pkg` against the table; a
  `jnp.maximum`/`jnp.minimum`/`jnp.clip` that is not a Fortran MAX/MIN (an index clamp) carries
  `# MINMAX-RAW: <reason>`, a builtin `max`/`min` of host integers `# MINMAX-INT: <why>` (integers have no tie or
  NaN case).
- Library functions: XLA's `sin`, `cos`, `tan`, `pow` are bitwise glibc 2.28; `exp` and `tanh` are not — use
  `mitjax/ops/libm.py` (`glibc_exp`, `glibc_tanh`) [Task 7c]. `x**n` with an integer literal `n` is `lax.integer_pow`
  (the same multiplication sequence as gfortran's `__powidf2`).

## 4. Vertical loops
- A loop over `k` whose iterations are independent may be vectorised with `k, j, i = loops_kji((1, Nr), (jlo, jhi),
  (ilo, ihi))`; say in the docstring why the iterations are independent.
- A recursion in `k` (tridiagonal solves, integrations from the surface or the bottom) uses `scan_k`
  (`mitjax/ops/scan_k.py`, the allow-listed helper; M1 sub-lane COL); never raw `lax.scan` in physics code. The
  loop is written as the Fortran DO statement, `range(2, Nr+1)` for `DO k=2,Nr`, `range(Nr-1, 0, -1)` for
  `DO k=Nr-1,1,-1`; `level_k(k)` returns (at trace time, k a Python int) what iteration k reads, as `level(A, k)`
  slices (FArrays declared (i, j)) and scalars such as `recip_drF[k]`, with the Fortran index (`level(c, k-1)`);
  the body `iteration(carry, x)` is one pass of the loop (carry = what level k reads from level k-1, or k+1 going
  up) and returns the new carry and the level-k results, written back with `set_levels(A, out, ks)` (results
  come back in increasing k for a downward loop too). The body is traced once, so it never branches on k. A
  single level outside the recursion is an integer index: `bet.at[i, j, 1].set(...)`. Measured
  (`mitjax/tests/test_scan_k.py`): bitwise equal to the Python loop in the Fortran order, gradients equal to the
  unrolled loop's; examples: `solve_tridiagonal.py`, `impldiff.py`, `calc_oce_mxlayer.py`.
- A routine the Fortran calls once per level (`CALL MOM_FLUXFORM(bi,bj,k,…)` inside `DO k=1,Nr`) keeps `k` as its
  argument and is written exactly as for one level; **its caller runs the level loop as a scan** with
  `scan_levels` (`mitjax/ops/scan_k.py`; decided by Nikolay 2026-10-02 from the R5 measurement below, replacing the
  2026-10-01 rule "unrolled Python loop"). The caller writes the loop body once, as `def level_k(k, c): ... return c`,
  where `c` is the tuple of what the Fortran carries from one level to the next (the carried locals, the State); the
  `DO` statement becomes static calls of the body for the levels whose code branches on `k` plus one `scan_levels`
  call for the others (`down=True` for `DO k=Nr,1,-1`). Inside the scan `k` is a traced level index `KIdx`
  (`mitjax/farray.py`): it indexes FArrays like an integer (a dynamic level), `k±n`, `n*k` and `MOD(k,2)` work (the
  parity is static: the body is traced for two consecutive levels per scan iteration, so the kUp/kDown slot
  rotations stay static), a comparison with an integer is decided from the range of levels the scan covers -- one the
  range does not decide raises (`k == 1`, `MAX(1,k-1)` at k = 2, `MIN(Nr,k+1)` at k = Nr-1, ...): those levels are
  peeled, visibly, next to the scan -- and a comparison with an array is pointwise. A Python list or tuple indexed
  with `k` raises (keep level data in FArrays). Per-level dump-gate probes: the body returns `(c, y)` and
  `scan_levels(..., with_out=True)` returns every level's `y`; the caller calls the probe after the loop. Values are
  bitwise those of the unrolled loop; reverse mode may sum a value's cotangent contributions from several levels in
  another order (gradients equal to rounding). Readability, Fortran `DO k=1,Nr / CALL CALC_PHI_HYD(..k..) /
  CALL MOM_FLUXFORM(..k..) / CALL TIMESTEP(..k..) / ENDDO` (`dynamics.F:422-562`):

  ```python
  def level_k(k, c):                     # the loop body as written for one level
      state, phiHydF, ..., gvDissip = c
      state, phiHydF, phiHydC, dPhiHydX, dPhiHydY = calc_phi_hyd(iMin, iMax, jMin, jMax, k, ...)
      ...                                # MOM_FLUXFORM(k), TIMESTEP(k): unchanged
      return (state, phiHydF, ..., gvDissip), y
  c, y1 = level_k(1, c)                                       # k = 1: CALC_PHI_HYD / MOM_FLUXFORM branch on it
  c, mid = scan_levels(level_k, c, 2, Nr-1, with_out=True)    # k = 2 .. Nr-1
  c, yN = level_k(Nr, c)                                      # k = Nr
  ```

  `scan_levels(..., peel=(a, b))` runs the a lowest and b highest levels of the range as static calls around the
  scan, in the loop's order (all of them static when nothing is left between). Where the per-level callers stand
  (branch scan-levels, then scan-sweep, 2026-10-02; static levels = the peeled ones):

  | caller (file) | loop | status | static levels / reason |
  |---|---|---|---|
  | DYNAMICS (`dynamics.py`; CALC_PHI_HYD, MOM_FLUXFORM/VECINV incl. MOM_CALC_RTRANS, TIMESTEP inside) | `DO k=1,Nr` | scan | 1, Nr |
  | TEMP_INTEGRATE, SALT_INTEGRATE, PTRACERS_INTEGRATE | `DO k=Nr,1,-1` | scan | Nr, 2, 1 |
  | DO_OCEANIC_PHYS: FIND_RHO_2D; GRAD_SIGMA + CALC_IVDC | `DO k=1,Nr`; `DO k=Nr,2,-1` | scan | -- |
  | GMREDI_CALC_TENSOR (four loops) | `DO k=Nr,2,-1`, `DO k=1,Nr`, 2x `DO k=Nr,1,-1` | scan | --; 1; Nr, Nr-1 |
  | GMREDI_CALC_PSI_BOLUS | `DO k=2,Nr` | scan | -- |
  | GMREDI_RESIDUAL_FLOW | `DO k=1,Nr` | scan | Nr-1, Nr (`MIN(k+1,Nr)`, `k.GE.Nr`) |
  | GAD_ADVECTION horizontal / vertical | `DO k=1,Nr` / `DO k=Nr,1,-1` | scan | -- / Nr, 1 (with DST3FL also Nr-1, 3, 2) |
  | GAD_SOM_ADVECT horizontal / vertical | `DO k=1,Nr` / `DO k=Nr,1,-1` | scan | -- / Nr, 1 |
  | GAD_IMPLICIT_R (implicit advection) | `DO k=Nr,1,-1` | scan | 1 (FLUXLIMIT, U3C4 also 2, 3, Nr-1, Nr) |
  | CONVECTIVE_ADJUSTMENT, CONVECTIVE_ADJUSTMENT_INI | `DO k=kTop,kBottom,kDir` | scan | -- |
  | INTEGR_CONTINUITY: INTEGRATE_FOR_W; the hDivFlow sum | `DO k=Nr,1,-1`; `DO k=1,Nr` | scan | Nr, 1; -- |
  | SOLVE_FOR_PRESSURE (CALC_DIV_GHAT), CORRECTION_STEP | `DO k=..` | scan | -- |
  | PP81_CALC, MY82_CALC (first loop) | `DO K=2,Nr` | scan | 2 (`MAX(1,K-1)` in *_RI_NUMBER) |
  | KPP (`kpp_routines.py`), DO_OCEANIC_PHYS' vermix wiring, CALC_3D_DIFFUSIVITY, CALC_VISCOSITY, TRACERS_CORRECTION_STEP | | unrolled | KPP lane's files this round |
  | UPDATE_CG2D, PP81/MY82/GMREDI_CALC_DIFF, DYNAMICS' kappaRU copy, SBO_CALC, GAD_PPM/PQM_ADV_R's `vsum` | | unrolled | one or two statements per level: nothing to gain |
  | IMPLDIFF's deltaTX, INI_*, SET_GRID_FACTORS, *_INIT_FIXED, INI_FORCING, INI_CG2D | | unrolled | host-side set-up (numpy / eager, once) |
  | MON_CALC_ADVCFL, MON_KE, COST_ATLANTIC_HEAT | | unrolled | output and cost-end code, outside the step's level loops |
  | DO_ATMOSPHERIC_PHYS | | unrolled | lane B |

  Earlier options, measured 2026-10-01 (MOM lane, job 27829003, a MOM_FLUXFORM-shaped composition):
  (a) unrolled loop -- literal, fastest to run, HLO and compile grow with Nr; (b) driver `vmap` over level windows --
  fragile, ~5x slower; (c) k-vectorised kernels -- constant compile, every kernel rewritten.
  **Measured** (CPU node, gate XLA flags, `.lower().compile()` in a fresh process, memory mappings counted in
  `/proc/self/maps`, vm.max_map_count = 65530; fusions in the compiled `as_text()`):

  | gradient program (10 steps, per-step checkpoint) | lower | compile | maps added | fusions | peak RSS | run (warm) |
  |---|---:|---:|---:|---:|---:|---:|
  | optim, master `a016bef` (unrolled; dev job 27840007) | 120.7 s | 532.6 s | 20 520 | 21 155 | 13.8 GB | 2.26 s |
  | optim, exp-scank (DYNAMICS + TEMP/SALT; job 27840008) | 98.1 s | 233.5 s | 13 465 | 15 671 | 5.2 GB | 2.28 s |
  | optim, scan-levels (+ DO_OCEANIC_PHYS, GMREDI_CALC_TENSOR) | 79.6 s | 110.6 s | 10 230 | 11 617 | 3.5 GB | 2.53 s |
  | global_ocean.90x40x15/input_ad, scan-levels | 139.6 s | 414.8 s | 22 156 | 23 082 | 7.6 GB | 4.64 s |
  | *scan sweep (branch scan-sweep; each pair same day, master `4d6148b` vs the sweep, dev jobs as listed):* | | | | | | |
  | optim, master (27841496) / sweep (27841494) | 80.4 / 76.0 s | 106.1 / 95.8 s | 10 819 / 9 874 | 11 617 / 10 932 | 3.7 / 3.5 GB | 3.71 / 3.65 s |
  | global_ocean input_ad, master (27841495) / sweep (27841493) | 143.0 / 123.5 s | 407.3 / 203.4 s | 22 892 / 15 660 | 23 082 / 16 939 | 7.8 / 5.4 GB | 6.04 / 5.39 s |
  | tutorial_tracer_adjsens input_ad, master (27841334) / sweep (27841490) | 79.1 / 63.2 s | 500.0 / 91.6 s | 13 012 / 8 089 | 14 717 / 9 716 | 11.3 / 3.6 GB | 1.01 / 1.07 s |
  | tutorial_tracer_adjsens input_ad.som81, master / sweep (27841491) | -- / 90.1 s | > 30 min / 158.5 s | -- / 9 830 | -- / 13 763 | -- / 4.4 GB | -- / 2.11 s |
  | *cube (GO lane session 10; 5 steps, 12 tiles):* | | | | | | |
  | global_ocean.cs32x15/input_ad, master `1ab2254` (27843047) / corner gathers (27843193) | 190.3 / 152.4 s | 475.7 / 306.0 s | 21 459 / 18 438 | 36 564 / 21 384 | 13.2 / 7.9 GB | 13.3 / 6.9 s |

  (input_ad on master: "~27 min, ~31 GB" for its gradient program, test_goadk_adjoint.py.) optim fc and admGrd
  bitwise master's (vs TAF 1.25e-15, 1.22e-15, 1.73e-15); input_ad admGrd vs TAF 7.0e-16, 2.7e-14, 4.2e-14,
  1.2e-13 (12-15 digits; GOADK measured 11-14 on the unrolled program; same-day master values not taken). The warm
  run of the optim gradient is 12 % slower with the scans (2.53 vs 2.26 s). Scan sweep: fc bitwise master's in all
  four programs; admGrd vs TAF bitwise unchanged for optim (1.25e-15, 1.22e-15, 1.73e-15) and tracer_adjsens
  input_ad (1.2e-16 .. 6.9e-16 at its 5 points), global_ocean input_ad 2.6e-15, 2.7e-14, 4.4e-14, 1.4e-13 (master 7.0e-16, 2.7e-14, 4.2e-14,
  1.2e-13: reverse-mode summation order), som81 1e-16 .. 7e-16 at its 5 points (som81's gradient program did not
  finish in 30-min dev jobs 27840319, 27840429 before the sweep; GAD_SOM_ADVECT's two level loops were half of its
  step's equations). Forward: every gated stage of every experiment bitwise (the whole-run and step gates of all
  experiments, branches scan-levels and scan-sweep; the sweep's gate jobs 27841483-27841588 and 27842024, tier 1
  27841628 and 27842025).
- CPP flags: a routine sees the macros of **its own** include list (Task 6 finding: optim's `code_ad/CPP_OPTIONS.h` does
  not include PACKAGES_CONFIG.h; packages override their `*_OPTIONS.h`). Read a package option as
  `cfg.cpp.flag(NAME, "<PKG>_OPTIONS.h")` when the routine includes that header; `cfg.cpp.flag(NAME)` is the model/src
  view (PACKAGES_CONFIG.h + CPP_OPTIONS.h). `cfg.cpp.NAME` is True when NAME is defined after any of the build's
  headers, and raises when a header that includes CPP_OPTIONS.h changes one of its definitions
  (`mitjax/config/cpp_options.py`, `CppOptions.__getattr__`).
- A Fortran `IF` on a REAL namelist value (reconciled 2026-10-01 from the MOM and core lanes' practice):
  - if it **chooses between two values** of an expression (e.g. `sideDragFactor.LE.0.` in `mom_u_sidedrag`), it stays
    traced: a `jnp.where` with both branches finite;
  - if it **selects which code runs** (calls a routine, skips a block: `bottomDragLinear.NE.0.` in MOM_FLUXFORM,
    `implicSurfPress.NE.1.` in DYNAMICS, `cg2dTargetResWunit.LE.0.`), it is decided once on the host from the namelist
    value, as a static, named flag in `ini_parms` (`bottomDragLinear_ne_0`) with the Fortran citation; a gradient or a
    perturbation of that parameter must not cross the threshold (the flag would be stale) — the docstring says so.
  Run-time integer/logical tests of the model clock (`myIter.EQ.nIter0`) on traced counters stay `jnp.where` over finite
  values; `myIter.LT.0` with myIter ≥ nIter0 ≥ 0 in every forward run is resolved statically (nIter0 < 0 raises).

### Patterns found in real code (Task 10, grid)
- A recursion along i or j (a loop whose iteration reads what the previous one wrote, e.g. `INI_LOCAL_GRID`) is a
  Python loop over a one-point loop index `i = loop_i(ii, ii)`, in the Fortran's order.
- Per-tile scalars (`iG0`, `xG0`, …) are `[tile, 1, 1]` arrays so that they broadcast over `(j, i)`; an integer read of
  a 1-D *tiled* array gives a `[tile]` array, which broadcasts on the wrong axis — reshape explicitly.
- Fortran `MOD`-indexed gathers (`delX(1+MOD(...))`) index the storage `.data[...]` with a numpy integer array of
  storage positions (`fortran_index - lo`), written next to the Fortran expression; loop indices used as numbers come
  from a separate numpy `arange` over the DO bounds.
- **Arguments the Fortran overwrites by reference are returned** (accepted by Nikolay 2026-10-01): a routine that writes
  into an `INTENT`-less argument (e.g. `fell, ferr` in `GAD_PPM_FUN_MONO`, `aa` in `QUADROOT`) returns the new value(s)
  after its declared outputs, in the Fortran argument order; a Fortran early `RETURN` becomes "compute every lane, select
  last" (`jnp.where` on the RETURN condition).
- **Vectorised row/column routines take the caller's loop index as an extra keyword** (accepted 2026-10-01): a routine
  the Fortran calls once per row or column inside the caller's DO loop (e.g. `GAD_OSC_LOC_X`, `GAD_OSC_MUL_X`) is
  vectorised over that loop and receives the caller's loop index as `ix=`/`iy=` (keyword-only, documented in the
  docstring as the one addition to the Fortran signature). Leading coefficient dimensions of point-local arrays are
  dicts keyed by the Fortran index (`fhat[1][ix-1, iy]`).
- A routine that receives one level of a 3-D array as a 2-D argument (Fortran sequence association) reads and writes it
  through small `_level`/`_set_level` helpers next to the call, never by reshaping the caller's storage.
- Host-side builders (the grid, `INITIALISE_FIXED`) run eagerly on the CPU backend under the gate flags: each jax
  operation is its own XLA computation, nothing is fused or re-associated; all nine M1 grids were bitwise at first try.

## 5. Halos, masks, tiles
- Compute every point the Fortran computes, halos included, so that gates compare all points (tolerance classes in
  `docs/PORTING_RULES.md` §5). Masked, halo and padding lanes compute finite values.
- The tile loop is implicit: every FArray carries all tiles; never loop over tiles in Python. The same code runs at
  P=1 and P=N (sharding lives in `mitjax/eesupp` and `mitjax/drivers`).
- At a jit boundary (drivers), pass plain arrays and build FArrays inside the traced program: an FArray passed as a jit
  argument costs ~5 µs of host flattening per call (Task 8 A100 measurement); inside the program it costs nothing.

## 6. What is banned in `mitjax/model/` and `mitjax/pkg/`
JAX transforms (`jit`, `grad`, `vjp`, `jvp`, `custom_jvp/vjp`, `custom_linear_solve`, `checkpoint`/`remat`,
`shard_map`, `pmap`, raw `lax.scan`) and `optimization_barrier` (Nikolay 2026-10-01); `ragged_all_to_all` anywhere.
`mitjax/tests/test_banned_transforms.py` enforces it. `stop_gradient` appears only in TAF-compatible switches, which
live in `mitjax/ad/`.

## 7. Gates for every kernel (gate first)
- Replay gate against the Fortran on synthetic inputs over the real grid (`reference/replay/`, Task 8 harness) and/or
  substep dumps (`reference/jaxdump/`, Task 4) at the gate XLA flags (`mitjax/xla_flags.py`), float parameters passed
  as traced jit arguments; bitwise or the tolerance class with a recorded reason.
- A negative control measured to bite in that experiment (no assumed controls) [E§2, P§4].
- Gradient: finite on all lanes (halos, land, padding); FD at smooth points; tangent vs adjoint dot test.
- Register each test file in the area's manifest fragment with an honest cost group (tier 1 has a budget of 100 tests).
