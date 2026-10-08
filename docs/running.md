# Running the model

Two ways in, with the same results: a small Python API (`import mitjax`) and its command-line mirror
(`python -m mitjax`). Both run an MITgcm experiment directory as `testreport` would: the experiment's `code/`
(or `code_ad/`) options and `SIZE.h`, an input directory (`input`, `input.<v>`, `input_ad`, ...), and the
experiment's `results/` as the reference. The experiment can be one of MITgcm's verification experiments or a copy
of one anywhere on disk ([configurations.md](configurations.md)). Gradients are on [gradients.md](gradients.md),
several devices on [parallel.md](parallel.md).

## The API

```python
import mitjax

exp = mitjax.load("MITgcm/verification/tutorial_barotropic_gyre", variant="input")
run = exp.run(out="runs/gyre")              # the forward run
rep = mitjax.compare(run, exp.results())    # testreport's comparison with results/
print(rep.summary)
```

| call | what it does |
|---|---|
| `mitjax.load(exp_dir, variant="input")` | reads the experiment: `SIZE.h`, the preprocessed `*_OPTIONS.h`, `packages.conf`, the namelists of the input directory. An unknown input directory or an experiment routine without a port stops here; an unported package or option stops the run when the model is set up or its time step is compiled. |
| `exp.run(out, devices=1)` | the forward run in a new directory `out` (must not exist); returns a `Run`. `devices=N` splits the tiles over N JAX devices ([parallel.md](parallel.md)); the result is the same. More devices than JAX has stops with `mitjax.api.DeviceCountError` (a `ValueError`) before anything is written. |
| `exp.results()` | the experiment's own `results/` files for this input directory (`output`, `output_adm`, `output_tlm`, or `None` where there is none), named as testreport names them. |
| `mitjax.compare(run, results)` | testreport's comparison: the check-list variables, their matching digits, pass or fail (`MATCH_CRIT = 10`, the first variable decides). Takes a `Run`, a `Grdchk` or a path to an `output.txt`. |
| `exp.grid()` | the grid (`GRID.h` arrays by their Fortran names: `xC`, `yC`, `rA`, `drF`, `hFacC`, `maskC`, ...) as a `GridArrays` dict of numpy arrays, with the same `interior(name)` and `global_field(name)` as a `Run`. It makes no run directory; for an input directory with a `prepare_run` the files are prepared once in a directory under `$MJX_CACHE/api_grid/`. |
| `exp.gradient(...)`, `exp.grdchk(...)` | adjoint gradient and gradient check: [gradients.md](gradients.md). |

A `Run` holds:

- `run.output`: `<out>/rundir/output.txt`, the lines MITgcm writes to STDOUT that testreport reads (the cg2d solver
  lines and the `%MON` monitor blocks, also the cost lines where the run has a cost).
- `run.pickups`: the pickup files the run's own schedule writes (`pChkptFreq`, `chkptFreq`, the end of the run), in
  MITgcm's format, in `run.rundir`.
- `run.fields`: the final state as numpy arrays (the ocean state, `uVel`, `vVel`, `theta`, `salt`, `etaN`, ...;
  with `pkg/seaice` on also the sea-ice state by its `SEAICE.h` names, `AREA`, `HEFF`, `HSNOW`, `UICE`, `VICE`, ...). They are in mitjax's storage layout
  `[tile, (k,) j, i]` with halos: Fortran index `i` is storage index `i - 1 + OLx`.
- `run.interior(name)`: the same without halos, `[tile, (k,) sNy, sNx]`.
- `run.global_field(name)`: the tiles put together (`run.tilemap` says where each tile goes). On a domain with the
  plain exchange (exch1: one rectangle) this is one global array `[(k,) Ny, Nx]`, tile `bi + (bj-1)*nSx` at
  `((bj-1)*sNy, (bi-1)*sNx)`; on a `pkg/exch2` domain (the cubed sphere) one array per face, `[face, (k,) j, i]`.
- `run.devices`: the number of JAX devices the run used.

API calls do not print the run directory (the command line's `run` prints a `RUNDIR` line for an input directory with
a `prepare_run`); a warning about a MAX/MIN default
(`mitjax.MinMaxDefaultWarning`, [configurations.md](configurations.md#maxmin-at-build-dependent-statements)) comes
through Python's `warnings`.

## The command line

```bash
python -m mitjax run EXP_DIR --variant input --out runs/gyre
python -m mitjax run EXP_DIR --variant input --out runs/gyre-2dev --devices 2
python -m mitjax compare runs/gyre/rundir/output.txt EXP_DIR --variant input
python -m mitjax gradient EXP_DIR --variant input_ad --out runs/grad
python -m mitjax grdchk EXP_DIR --variant input_ad --out runs/grdchk
```

- `run` is `mitjax.load(EXP_DIR, variant).run(out, devices=N)`: it writes `<out>/rundir/output.txt` and the
  pickups and prints where it wrote them. `--devices N` (default 1) splits the tiles over N JAX devices, with the
  same files ([parallel.md](parallel.md)); more devices than JAX has stops with exit code 2 and the
  `DeviceCountError` message before anything is written. It sets `XLA_FLAGS` as the API does
  ([troubleshooting.md](troubleshooting.md#xla-flags)).
- `compare` prints the digits per check-list variable and testreport's summary line; its exit code is 0 for a pass,
  1 for a fail and 3 when there is nothing to compare ("N/O"). `--kind fwd|adm` overrides the kind the input
  directory implies.
- `gradient` and `grdchk` take `--mode run|exact`, `--cg2d-derivative run|exact` and `--lsr-derivative run|sweeps`;
  `gradient` also `--devices N` and `--control NAME` ([gradients.md](gradients.md)).
- `python -m mitjax <command> --help` lists the options.

testreport's comparison also runs on its own, on any MITgcm output: `python -m mitjax.testreport_jax OUTPUT --exp
EXPERIMENT --variant input` (with `--reference FILE` for another reference than `results/`).

## Run directories

Each run gets a new directory `<out>`; nothing is overwritten or deleted. Inside it, the run directory is made as
testreport makes it (`mitjax/make_rundir.py`):

- `<out>/rundir` points to the run directory (`run` for the main input directory, `tr_run.<v>` for a variant), inside
  a mirror of the experiment under `<out>/verification/<experiment>/`.
- testreport's `linkdata`: every file of `input` (and first of the variant's own `input.<v>`, which shadows `input`)
  is linked into the run directory.
- the input directory's `prepare_run` script, if it has one, runs there; its output is kept in
  `<out>/prepare_run.log`. Some `prepare_run` scripts reach other experiments by relative paths
  (`../../tutorial_global_oce_latlon/input/...`); for a copied experiment those experiments must sit next to the
  copy, as they would for testreport.
- the input files the namelists name (keys ending in `File` in `data`, the weights of `data.ctrl`, the start pickup
  when `nIter0 > 0`) must exist, or the directory is refused before the model starts.
- the model then reads its namelists and files from the run directory as MITgcm does, and writes its output there.

Output files are written as MITgcm writes them (MDS `.data`/`.meta`, big-endian). `pkg/diagnostics` and `pkg/mnc`
output is not written (both packages are accepted but not ported: [configurations.md](configurations.md)).

## Run time and memory

Measured on 8 cores of an AMD EPYC 7763 CPU node with the notebooks (the first three rows: the notebooks' own
executions; the last: a run of the same calls before the notebook was executed). Each includes compiling:

| what | wall time | peak memory |
|---|---|---|
| `tutorial_barotropic_gyre` run, comparison and two plots (notebook 01) | 109-127 s | 2.0 GiB |
| `1D_ocean_ice_column/input_ad` gradient, gradient check and a dot test (notebook 03) | 824-825 s | 5.3 GiB |
| a copied and retiled gyre, a namelist change and a box from scratch (notebook 04) | 343-345 s | 5.2 GiB |
| `lab_sea/input` forward and `lab_sea/input_ad` gradient check (notebook 05) | 1745 s | 21.5 GiB |

Most of the time of a short run is XLA compiling the time step; [troubleshooting.md](troubleshooting.md) says what
drives compile time and memory.
