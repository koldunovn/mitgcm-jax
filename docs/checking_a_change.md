# Checking a change

How to tell whether a change of yours (a namelist value, a ported routine, a new port) changed the model, with and
without a Fortran build of MITgcm.

## Without a Fortran build: testreport against `results/`

Every verification experiment carries the output of a reference run in `results/`. mitjax's comparison is a literal
port of `testreport`'s (`mitjax/testreport_jax.py`): it reads the same check-list variables from both outputs (the
experiment's `tr_checklist`, else testreport's default list: the cg2d residuals and the monitor's temperature,
salinity and velocity statistics) and counts the matching digits per variable as testreport does.

```bash
python -m mitjax compare runs/gyre/rundir/output.txt EXP_DIR --variant input        # forward
python -m mitjax compare runs/col-chk/rundir/output.txt EXP_DIR --variant input_ad   # adjoint (output_adm.txt)
```

- 16 means every printed digit equal, 22 that every compared value is zero in both, `--` no comparison. Pass means
  the first variable matches to at least 10 digits (`MATCH_CRIT`), as in testreport; read the other variables too.
- `results/` were made by the MITgcm developers with their compiler, so even a Fortran build does not always
  reproduce every digit. [YARDSTICK.md](YARDSTICK.md) lists, per experiment and variable, how many digits the
  project's own gfortran build reaches against `results/`; in the gated whole runs mitjax reaches at least those
  numbers ([status.md](status.md)).
- **To check that a change is neutral**, compare the run after the change with the run before it, on the same
  machine: `python -m mitjax.testreport_jax after/rundir/output.txt --exp EXPERIMENT --variant input --reference
  before/rundir/output.txt`. A neutral change gives 16 (or 22) everywhere; mitjax is deterministic on one machine,
  so the outputs are then identical. Notebook 02 shows both cases: reordering the two terms of a sum keeps the
  testreport pass at 11-14 digits, halving a viscous flux drops it to 5.
- For gradients, compare a gradient check with `results/output_adm*.txt` (`--kind adm`; [gradients.md](gradients.md)
  lists the experiments where TAF's own adjoint is not the right reference).

## With a Fortran build: the project's reference ("oracle")

The project's reference is its own gfortran build of MITgcm at `63cdc0b`, built as testreport builds it (`-O0`, no
MPI) plus `-ffp-contract=off` (no fused multiply-add) (`reference/README.md`, `reference/build.sh`). You do not need
it to run mitjax; it is how the port was verified, and the scripts to rebuild it are in `reference/`. Three kinds of
checks use it:

1. **Stage dumps.** An instrumented copy of the build (`reference/jaxdump/`) writes named stages of the time step
   (e.g. `S00_begin`, `C01_cg2d_inputs` and `C02_cg2d_solution` around CG2D) for the first steps, with halos. A test
   runs the ported routine on the dumped inputs of a stage and compares with the dumped outputs. The dumps do not
   change the run: every output file and STDOUT are byte for byte the same with dumps on and off.
2. **Replay harnesses.** For a routine that needs inputs no experiment produces (an unstable column for `CALC_IVDC`,
   every `KEscheme`), a build replaces `THE_MAIN_LOOP` by a driver that reads synthetic inputs on the experiment's
   real grid, calls the Fortran routine, and writes all outputs (`reference/replay*/`).
3. **Whole runs.** A run of mitjax is compared with the Fortran run's STDOUT (every `%MON` record and solver line)
   and pickup files, and both with `results/`.

The criterion is **bit for bit** wherever possible: equal 64-bit patterns (so `+0` and `-0` differ) on every point the
Fortran computes, halos and land included. That needs one fixed `XLA_FLAGS` string ([troubleshooting.md](troubleshooting.md)),
REAL parameters passed to the compiled program as arguments, and the Fortran built without FMA. Where a value cannot
be bit for bit (a solver at its tolerance, a gradient), the test states its tolerance and why. A check counts only
after a **negative control**: a planted error (one ulp in an input, a reversed summation order, a skipped exchange)
that was measured to make it fail. Gradients are checked by finite differences at several step sizes, a dot test of
the tangent against the adjoint, and finite values on every point.

Where XLA flushes subnormal numbers to zero and gfortran does not (sea-ice velocities decaying below 2^-1021 in
`global_ocean.cs32x15/input.seaice`), the reference is a variant of the same Fortran build with flush-to-zero set at
program start (plan decision 13); its difference from the standard build is measured and confined to the subnormal
range ([troubleshooting.md](troubleshooting.md#subnormal-numbers)).

## The project's test tiers

Every test file has a cost group (`mitjax/tests/manifest_*.py`; a file without one is a collection error):

| group | what | where | how |
|---|---|---|---|
| `smoke` | seconds | a login node | `pytest -m smoke -q` |
| `tier1` (with smoke) | the fast suite, under 10 minutes, under 100 tests | one CPU node, every commit | `sbatch scripts/run_tier1.sbatch` |
| `tier1x` | stage-dump and replay checks, whole runs, P = N, finite differences, negative controls, notebooks 01-04 | CPU nodes, before a merge | `sbatch scripts/run_tier1x.sbatch` |
| `tier2` | GPU: sharded gradients on 4 A100s against 1, repeat floors | GPU node | `sbatch scripts/run_tier2.sbatch` |
| `tier3` | milestone runs and the cluster notebooks 05-07 | CPU node, before a release | per test file |

The verdict of a tier comes from the pytest report and exit code (`scripts/check_pytest_report.py`), never from the
batch job's state.

Most tests compare with the Fortran reference and need its data (`MJX_REFERENCE`, its replay runs; a few need the
project's lessons digests, `MJX_LESSONS`). Without that data -- a user's machine -- those test files are not
imported: each shows up as one test, `needs_oracle_data`, that is skipped with a reason naming the variable that is
missing, and they carry the `oracle` marker. So, with only `MJX_UPSTREAM` set:

```bash
pytest -m smoke -q -rs                    # seconds: the unit checks run, the oracle files are skipped by name
pytest -m "smoke or tier1" -q -rs         # the fast suite (a CPU node or a workstation): same, plus tier 1
pytest -q mitjax/tests/test_docs_pages.py mitjax/tests/test_docs_generated.py      # the page checks, by file
```

Which files need the data is found by reading their code (`mitjax/tests/oracle.py`), so a new test needs no
registration. The project's own runs set `MJX_REQUIRE_ORACLE=1`, which turns a missing reference into a failure
instead of a skip.

## Finding your way

- **From a `.F` file to its port:** [fortran_map.md](fortran_map.md); inside the Python file, search for the `.F` line
  number (`# :NN`).
- **What an experiment executes:** `docs/coverage/<experiment>.md` lists the routines the Fortran build executed in
  each variant (gcov), the lines never executed in them and the branches never taken. A routine or branch that is not
  executed is usually not ported, and the port stops if it is reached.
- **The live code of a routine:** `python tools/cpp_live.py EXPERIMENT ROUTINE` prints the lines of a routine that
  survive the experiment's CPP options with their `.F` line numbers. It uses the Fortran build's preprocessor
  (`MJX_REFERENCE`).
- **Rules and decisions:** `docs/PORTING_RULES.md` (binding rules for the port), `docs/KERNEL_GUIDE.md` (how a
  kernel is written), [ISSUES_UPSTREAM.md](ISSUES_UPSTREAM.md) (what the port found in MITgcm itself).
