# Installing mitjax

mitjax runs on your own machine or cluster. It needs a conda environment, an MITgcm checkout at the commit it ports,
and a C preprocessor. It does not need a Fortran compiler or any Fortran build of MITgcm.

## Platforms

- **Linux x86_64** is the tested platform: every test, notebook and number in these pages was measured there.
- **macOS on Apple silicon (arm64)** is best effort: nothing has been run on it yet. XLA may round some operations
  differently on another CPU, so the notebooks check platform-tolerant numbers (testreport digits and gradient-check
  digits above a stated minimum); the bit-for-bit checks of the test suite are made on Linux x86_64.
- **GPU:** NVIDIA GPUs with CUDA 12, through the CUDA wheels of jax (below). The GPU tests ran on NVIDIA A100 80 GB.

## Five steps

```bash
git clone https://github.com/koldunovn/mitgcm-jax
git clone https://github.com/MITgcm/MITgcm && git -C MITgcm checkout 63cdc0b   # next to mitgcm-jax, not inside it
conda env create -f mitgcm-jax/environment.yml   # Python 3.12 and the pinned dependencies (jax 0.10.1)
conda activate mitjax
pip install -e mitgcm-jax                    # mitjax itself, editable
export MJX_UPSTREAM=$PWD/MITgcm              # required: the MITgcm checkout at 63cdc0b
```

The commands here and in the other pages run from this directory, the one that holds both checkouts. MITgcm is
cloned next to the mitgcm-jax checkout, not inside it, so that the tests that walk the repository do not walk
MITgcm too.

If you already have an MITgcm clone at another commit, keep it and add a checkout of `63cdc0b` next to it instead
of cloning again: `git -C MITgcm worktree add MITgcm-63cdc0b 63cdc0b`, then
`export MJX_UPSTREAM=$PWD/MITgcm-63cdc0b`.

Then try a run (see [running.md](running.md)):

```bash
python -m mitjax run $MJX_UPSTREAM/verification/tutorial_barotropic_gyre --variant input --out runs/gyre
python -m mitjax compare runs/gyre/rundir/output.txt $MJX_UPSTREAM/verification/tutorial_barotropic_gyre --variant input
```

The second command prints testreport's digits against the experiment's `results/output.txt`; on Linux x86_64 every
variable matches to 16 digits. Notebook 01 makes this run, the comparison and two plots in 109-127 s with a peak
memory of 2.0 GiB (measured three times on 8 cores of an AMD EPYC 7763 CPU node).

## What each piece is

- **`environment.yml`** pins every dependency to the versions the project tests with (the same as
  `constraints.txt`; `mitjax/tests/test_env_pins.py` checks that the two agree). Use `mamba env create` instead of
  `conda env create` if you have mamba. jax is pinned on purpose: an upgrade can change rounding, so it is a
  deliberate, tested step for this project.
- **`pip install -e .`** installs the package `mitjax` from the checkout, editable: a change to a `.py` file is used
  by the next run.
- **The notebooks** need four more packages (matplotlib, nbclient, nbformat, ipykernel), the `notebooks` extra:
  `pip install -e "mitgcm-jax[notebooks]"`. Jupyter itself is not included (`pip install jupyterlab`, or open the notebooks
  in VS Code). The cube-sphere maps of notebook 06 use `nereus` when it is installed and plain matplotlib otherwise.
  See [notebooks/README.md](../notebooks/README.md).
- **GPU:** in the same environment, `pip install "jax[cuda12]==0.10.1"` (CUDA 12 and cuDNN come as pip wheels).
- **`MJX_UPSTREAM`** is the MITgcm checkout at commit `63cdc0b` (checkpoint69q plus 9 commits).
  mitjax reads the Fortran sources from it (the packages' `*_OPTIONS.h`, the `NAMELIST` statements, the default
  values it cites) and the verification experiments. Another commit is refused: every citation in the code refers
  to `63cdc0b`. The error gives the `git worktree` commands above for your clone.
- **A C preprocessor** is needed at run time: mitjax preprocesses the experiment's `*_OPTIONS.h` files as MITgcm's
  `genmake2` does (`cpp -traditional -P`). On Linux this is GNU cpp (the `cpp` of gcc); on macOS clang's cpp from
  the Xcode command line tools. `MJX_CPP` names another preprocessor, `MJX_CPP_DEFINES` replaces the default macro
  set, `MJX_CPP_INCLUDES` gives the include directories and `MJX_HAVE_NETCDF` (1 or 0, default 1) is genmake2's
  `HAVE_NETCDF` (`mitjax/config/cpp_options.py`). The macro set is part of a build's identity (see
  [configurations.md](configurations.md#maxmin-at-build-dependent-statements)).

## Where mitjax writes

| variable | what | default |
|---|---|---|
| `MJX_UPSTREAM` | MITgcm checkout at `63cdc0b` | required |
| `MJX_RUNS` | run output of the tools and tests | `./runs` |
| `MJX_CACHE` | derived tables (preprocessor link farms, exchange maps) | `~/.cache/mitjax` (`$XDG_CACHE_HOME/mitjax`) |
| `MJX_REFERENCE` | the project's Fortran reference runs (developers only) | unset; not needed to run |

All locations are read once, at import (`mitjax/paths.py`; `python mitjax/paths.py` prints the values it resolved).
An API call or `python -m mitjax` writes its run under the `--out` / `out=` directory you give; that directory must
not exist yet (nothing is ever overwritten).

## Developer mode (not installed)

The project's own test runs never install mitjax: each git worktree is run with `PYTHONPATH=<worktree>` (pytest puts
the repository root on `sys.path`), so that a test always imports the checkout it tests and never an installed copy
from another one. The environment then holds only the dependencies, and a small shell file sourced by the batch
scripts sets the `MJX_*` paths. Use this mode if you work in several checkouts at once; otherwise
`pip install -e .` is simpler. Details: [ENV.md](ENV.md).
