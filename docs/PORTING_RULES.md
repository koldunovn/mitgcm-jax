# Porting rules (mitjax)

The binding rules for every change in this repository, approved with the project's design on 2026-10-01. Lessons
behind each rule: `docs/LESSONS_CARRIED.md` (the [E§n], [P§n], [F§n] tags resolve through its mapping table). A change
to a rule needs Nikolay's approval and is made here.

## 1. What is ported
- **Faithfulness to the original model comes first** (Nikolay 2026-10-01): when fidelity to the Fortran and any other
  goal (speed, AD convenience, code simplicity) conflict, fidelity wins. Forward values the model uses are exactly the
  Fortran's on every lane the Fortran computes, halos included; AD-only representations (e.g. halo zeroing inside the
  cg2d derivative solves) never replace a forward value.
- **Target:** MITgcm master at commit `63cdc0b` (branch `pinned` of the clone at `$MJX_UPSTREAM`). A newer upstream is
  an explicit sync task (diff `pinned..<new>`, port changed routines, re-run all gates).
- **Literal translation.** Only mechanical changes; any difference from the Fortran is a bug. No "simplifications", no
  algebraic restructuring: keep the tendency arrays, the summation order and the association. Port the live line, never
  a commented-out alternative. [P§3, F§2]
- **Citations.** Every constant and branch cites `file:line` at `63cdc0b` with its literal value; every routine's
  docstring is the Fortran header plus `@63cdc0b file:line`.
- **Configuration.** Values only from the experiment's `data*`, `data.pkg`, `data.diagnostics`, `eedata` and the
  preprocessed CPP options of its `code/` (or `code_ad/`) build. A Fortran default is used only with its citation
  (`set_defaults.F`, `ini_parms.F`, `<pkg>_readparms.F`); no Python defaults. `code_ad` builds are different forward
  models (ALLOW_AUTODIFF branches, their own CPP options and namelists). [E§3]
- **Unported means error.** A package switched on at run time (`use<Pkg>=.TRUE.`) that is not ported, or an unported
  option value, is a hard error at setup. Compiled-but-off packages are fine.
- **Deviations.** A deviation from the Fortran needs Nikolay's approval and a `# DEVIATION:` banner at the site; until
  approved, stop and report it — never implement it first. [E§1, E§3]

## 2. Readability (for a Fortran MITgcm developer who knows Python but not JAX)
- One `.py` per `.F`, one Python function per Fortran subroutine with the same name and the same arguments minus
  `bi, bj, myThid`; the tree mirrors MITgcm (`mitjax/eesupp`, `mitjax/model/src`, `mitjax/pkg/<pkg>`).
- CPP options become static `cfg.<NAME>` flags (`if cfg.ALLOW_AUTODIFF:` mirrors `#ifdef ALLOW_AUTODIFF`); common-block
  variables are fields with their Fortran names; `bi, bj` loops are implicit (tile axis); `k` loops are vectorised only
  when the iterations are independent (said in the docstring); a pointwise `IF` becomes `where`; recursions use `scan_k`.
- Array style: decided by the Task 8 prototype (Fortran-index arrays vs plain slices) — no M1 kernel before the decision.
- **No JAX transforms in `mitjax/model/` and `mitjax/pkg/`:** `jit`, `grad`, `vjp`, `jvp`, `custom_jvp/vjp`,
  `custom_linear_solve`, `checkpoint`/`remat`, `shard_map`, raw `lax.scan` and `optimization_barrier` (no barriers in
  physics code, Nikolay 2026-10-01 [L-CONF-7]) live in `drivers/`, `ad/`, `eesupp/` and `ops/`. Physics code uses jax.numpy and the allow-listed helpers (`scan_k`, `where`, safe ops).
  `mitjax/tests/test_banned_transforms.py` enforces it (AST scan with import aliases resolved).

## 3. Differentiability (every kernel)
- Masked, halo and padding lanes compute finite values: guard before the operation. A forward `where` does not stop a
  backward 0·inf, and the sharded backward does not fold a 0·inf that one device folds. [E§6, F§3]
- Never differentiate solver iterations: fixed iteration counts or implicit derivatives (cg2d through
  `ad/cg2d_rule.py`). Static configuration is never an argument of a differentiated function; never branch on a traced
  parameter inside a rule. [E§6, F§1]
- The exact JAX gradient is the default. Each `data.autodiff` approximation a ladder experiment uses gets a
  TAF-compatible backward-only switch whose forward is byte-identical, with an effect test on a live fixture.
- Take the JAX-optimal route by default, and keep the comparison with TAF as close as possible (Nikolay 2026-10-01): at
  an exact kink (a tie in a max/min, a limiter switch) the default is JAX's own derivative; where a TAF comparison lands
  on one, a TAF-compatible backward-only switch selects TAF's branch [L-CONF-12].
- Gradient claims need the trust protocol (FD h-sweep plateau above the forward-noise floor, TL vs adjoint dot test,
  repeats); FD agreement alone is not proof of correct physics. [E§6]

## 4. Parallelism: one code path
- Arrays `[tile, k, j, i]` (or `[tile, j, i]`) with halos `OLx, OLy` from the experiment's `SIZE.h`; Fortran index `i`
  at array index `i-1+OLx`; tiles in the experiment's order (exch2 W2 numbering when present).
- The same code runs at P=1 and P=N: exchanges from one gather map (exch1 and exch2 alike), `ppermute` inside
  `shard_map(check_vma=True)`; `ragged_all_to_all` is banned (wrong transpose). Global sums in fixed tile order, written
  as explicit add chains in the Fortran loop order. [E§5, E§7]
- Every forward kernel gate also runs at P=4 (fake CPU devices) once the experiment has ≥ 4 tiles. Sharded gradients
  are gated on real GPUs (tier 2): fake CPU devices gave NaN or deadlocked for the ECCO gradient driver. [E§7]

## 5. Gates and tests
- **Gate first (TDD).** Write the gate against the oracle (substep dump or replay, plus a gradient check), then port
  until it passes. A new gate is trusted only after it fails on a planted error that was measured to bite in that
  experiment — no assumed negative controls. [E§2, P§4]
- **Never loosen a tolerance.** Fix a failing gate by fixing the code or the experiment (teacher forcing, replay).
- **The Fortran oracle is built as testreport builds it:** `-ieee` (= `-O0`), no MPI, plus `-ffp-contract=off`
  (Nikolay 2026-10-01: keep `-O0`; `reference/README.md`).
- **Bitwise CPU gates** need all three: the one gate `XLA_FLAGS` string (`mitjax/xla_flags.py`: no FMA, no algsimp,
  4 fake devices; set once by `conftest.py` and by every standalone gate script), float parameters passed as traced jit
  arguments (never closed over), and the oracle built with `-ffp-contract=off`. [E§5]
- **Tolerance classes** (relative to the field's max abs over the compared points, all points incl. halos where the
  Fortran computes them): pointwise ~1e-15 or bitwise; stencils/reductions ~1e-13; solver fields at solver tolerance
  with iteration counts gated; check-list digits ≥ yardstick. Compare by element equality with a finite oracle (Python
  `max()` never picks a NaN); comparators refuse to run without masks; gate the number of points over the ceiling, not
  only the maximum. [E§4, P§4]
- **Manifest.** Every test file is listed with its cost group in its area's fragment `mitjax/tests/manifest_<area>.py`
  (merged by `mitjax/tests/manifest.py`; an unlisted file is a collection error). Every test directory is collected by
  the runner. `xfail(strict=True)` only; oracle-dependent tests fail (not skip) when their data is missing unless
  explicitly marked. [F§7]
- **Tiers:** tier 1 < 10 min and < 100 tests on one CPU compute node, every commit (`sbatch scripts/run_tier1.sbatch`;
  verdict from `scripts/check_pytest_report.py` + exit code, never the job state); tier 1x nightly CPU; tier 2 GPU;
  tier 3 milestone twins (cost stated, Nikolay's yes before submission).
- **Standing gates** (re-run by every task once they exist): full-field gradient finiteness incl. halos/padding,
  halo-poison, range checks (NaN/Inf/ZERO-but-expected-nonzero), P=1 vs P=4 forward, banned-transforms scan.
- Every task includes new or updated tests; all tests pass before the next task starts.
