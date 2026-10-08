# mitjax notebooks

Walk-throughs for an MITgcm user. Each notebook states its tier, measured run time and peak memory in its first
cell; the committed copies hold the outputs of a run on a Linux CPU node pinned to 8 cores.

| notebook | tier | what it shows |
|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | laptop | run `tutorial_barotropic_gyre`, compare with `results/` as testreport does, plot; the command line |
| [02_fortran_to_jax](02_fortran_to_jax.ipynb) | laptop | read a routine next to its `.F`; change one line, rerun, see what testreport says |
| [03_first_gradient](03_first_gradient.ipynb) | laptop | adjoint gradient and gradient check of `1D_ocean_ice_column` against TAF; a dot test |
| [04_own_configuration](04_own_configuration.ipynb) | laptop | a changed copy of an experiment (run length, namelist, forcing, tiling); a box from scratch; unported options |

| [05_lab_sea](05_lab_sea.ipynb) | cluster | ocean + sea ice forward vs `results/`, sea-ice maps; the adjoint's gradient check vs TAF's tangent-linear model and the Tapenade adjoint; TAF's adjoint off by 1e-4 (MITgcm#1041) |
| [06_adjoint_modes_cs32x15](06_adjoint_modes_cs32x15.ipynb) | cluster | `mode="run"` (TAF's adjoint-mode switches, backward only) vs `mode="exact"`, each against its own TAF reference; one switch at a time below the API |
| [07_parallel](07_parallel.ipynb) | cluster | forward and gradient split over 1-4 devices: bitwise forward, gradient to 1e-14; global sums in tile order; GPUs |

The cluster notebooks (05-07) need one CPU node for about 20-30 min each and up to about 23 GB of memory.

**Before you start:** the conda environment of `environment.yml`, mitjax installed with the notebook extra
(`pip install -e "mitgcm-jax[notebooks]"`), and an MITgcm checkout at commit `63cdc0b`:

```bash
git clone https://github.com/MITgcm/MITgcm && git -C MITgcm checkout 63cdc0b   # next to the mitgcm-jax checkout
export MJX_UPSTREAM=$PWD/MITgcm
jupyter lab mitgcm-jax/notebooks/   # needs `pip install jupyterlab`; or open the notebooks in VS Code
```

The notebooks write their runs and copied experiments under `notebooks/runs/` and `notebooks/my_experiments/`
(new directories each time; nothing is overwritten). The sources are the plain-text files in `src/`
(`python tools/build_notebooks.py` rebuilds the notebooks from them); `mitjax/tests/test_notebooks_laptop.py`
executes the laptop notebooks and checks their asserts; `mitjax/tests/test_notebooks_cluster.py` the cluster ones.
