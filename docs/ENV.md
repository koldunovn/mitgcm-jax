# Environment (Levante)

Env `mitjax` (this project's: `$MJX_PYTHON` in `levante.env`), created 2026-10-01 by job 27825269
(`scripts/make_env.sbatch`, partition `shared`, 9.5 min), pinned by `constraints.txt` (copied from the ECCO port's env
`mitgcm-jax`, itself the fesom-jax known-good set). `test_env.py` fails if jax/jaxlib/numpy/scipy differ from
`constraints.txt`.

**A JAX upgrade is a deliberate, gated step**, never a drift: the canary is tier 1 + the gradient gates + a short GPU
run, old vs new (plan Post-Completion). jax 0.10.2 deadlocked in-process CPU collectives at npes ≥ 4 (fesom_jax lessons
§8); `ragged_all_to_all` has a wrong transpose up to at least 0.11.1 and is banned.

## mitjax is not installed into the env
The env holds only the dependencies. Tests import the checkout under test: pytest puts the repository root (the
directory of `conftest.py`) first on `sys.path`, and the tier-1 runner also exports `PYTHONPATH=<repository>` for
subprocess tests. An installed or editable mitjax would make a worktree silently test another checkout (fesom_jax
lessons §7). `test_env.py::test_imports_this_checkout` checks it.

## Install for a user (any machine; docs plan 20261006 Task 5)
```bash
git clone https://github.com/koldunovn/mitgcm-jax
git clone https://github.com/MITgcm/MITgcm && git -C MITgcm checkout 63cdc0b   # next to mitgcm-jax, not inside it
conda env create -f mitgcm-jax/environment.yml   # Python 3.12.13 + the pinned dependencies (= constraints.txt)
conda activate mitjax
pip install -e mitgcm-jax                    # mitjax itself, editable
export MJX_UPSTREAM=$PWD/MITgcm              # required; MJX_CACHE / MJX_RUNS default to user directories
python -m mitjax run $MJX_UPSTREAM/verification/tutorial_barotropic_gyre --variant input --out runs/gyre
python -m mitjax.testreport_jax runs/gyre/rundir/output.txt --exp tutorial_barotropic_gyre --variant input
```
`mitjax/tests/test_env_pins.py` checks that environment.yml agrees with constraints.txt and pyproject.toml;
`scripts/pip_install_check.sbatch` installs a snapshot editable into a new venv on top of env `mitjax` and runs the
two commands above from outside the repository. A C preprocessor is needed at run time (`cpp -traditional -P`, found
as genmake2 finds it; `MJX_CPP`, `MJX_CPP_DEFINES`, `MJX_CPP_INCLUDES`, `MJX_HAVE_NETCDF` override it,
`mitjax/config/cpp_options.py`). The oracle tools (`reference/`, `tools/`) are not part of the package; the two that
mitjax uses at run time moved into it (`mitjax/make_rundir.py`, `mitjax/testreport_jax.py`; shims stay at the old
paths).

## The notebooks (docs plan 20261006 Task 8)
The laptop notebooks `notebooks/01-04` need four more packages, the `notebooks` extra of `pyproject.toml`
(matplotlib, nbclient, nbformat, ipykernel; pinned in `environment.yml` and `constraints.txt`):
```bash
pip install -e ".[notebooks]"                # in the env above; then open notebooks/ in Jupyter or VS Code
```
The notebooks are built from the plain-text sources `notebooks/src/*.py` (`python tools/build_notebooks.py`); the
committed `.ipynb` files hold the outputs of an execution on a compute node (`scripts/nb_gate.sbatch`,
`mitjax/tests/test_notebooks_laptop.py`, tier1x).

On Levante env `mitjax` stays as it is (other sessions use it): the notebook packages live in an overlay venv that
sees the env's packages, `$MJX_WORK/envs/mitjax-nb` (`python -m venv --system-site-packages`, then
`pip install --no-cache-dir -c constraints.txt` the extra; jax, jaxlib, numpy and scipy resolve to the env's own
copies, checked by the script):
```bash
sbatch scripts/make_nb_venv.sbatch <repository>     # on `shared` (downloads); refuses an existing venv
```
The test and the gate find it as `MJX_NB_PYTHON` (default: that venv where it exists, else the running python). Built
2026-10-07 by job 27938753: matplotlib 3.10.9, nbclient 0.11.0, nbformat 5.10.4, ipykernel 7.3.0 and their
dependencies, every one at its constraints.txt version (log `$MJX_WORK/logs/nb_venv-27938753.out`).

## Developer mode (this project, Levante): not installed
Worktrees are not installed anywhere: run with `PYTHONPATH=<worktree>` (pytest's rootdir does this for tests), see
"mitjax is not installed into the env" above. Machine paths come from `levante.env` (sourced by every sbatch script
and tier runner; `mitjax/paths.py` has neutral defaults). `levante.env` is this project's DKRZ Levante site file,
published as it is (Nikolay 2026-10-08): on another cluster, copy it and adapt the paths (`MJX_WORK`, `MJX_PYTHON`,
`MJX_CONDA_BASE`; `SBATCH_OUTPUT` sends the Slurm logs to `$MJX_WORK/logs/<job name>-<job id>.out`) and the
`#SBATCH --account` / `--partition` lines of the batch scripts.

## Create
```bash
sbatch scripts/make_env.sbatch [repository]     # refuses if the env exists; log in $MJX_WORK/logs/mjx_make_env-<job>.out
```
It runs `mamba create -p .../envs/mitjax python=3.12.13 pip`, then `pip install --no-cache-dir -c constraints.txt` the
dependencies listed in `pyproject.toml` (with the `cuda` and `dev` extras), then a self-check (pins, float64, mitjax
not importable from site-packages; last line `ENV OK`). `--no-cache-dir`: pip's cache would land in `~/.cache/pip`,
and home is ~full. CUDA 12 + cuDNN come as pip wheels; no system CUDA module.

The first run (job 27825269) ended `ENV FAIL [('mitjax importable', ...)]`: the self-check ran with the current
directory = the submit directory (`~/MITjax`), so `python -` found the package there. The env itself was complete and
correct (verified from `/`: `find_spec('mitjax')` is None, all pins match); the script now `cd /` before the check.

## Run
- Smoke (seconds; allowed on a login node, restrict cores):
  `. ./levante.env && JAX_PLATFORMS=cpu taskset -c 0-7 $MJX_PYTHON -m pytest -m smoke -q` (levante.env: the machine
  paths; without it MJX_UPSTREAM is unset and nothing runs, and MJX_REFERENCE is unset so the oracle tests skip)
  (add `PYTHONPYCACHEPREFIX=$MJX_RUNS/pyc/<name>` to keep bytecode out of the repository).
- Tier 1 (compute node): `sbatch scripts/run_tier1.sbatch [repository]` → `$MJX_RUNS/tier1/<jobid>/`
  (`provenance.txt`, `pytest.log`, `report.xml`, `verdict.txt`, `git_diff.patch`). The verdict comes from
  `scripts/check_pytest_report.py` + pytest's exit code, never from the job state.
- `conftest.py` sets the gate `XLA_FLAGS` once (`mitjax/xla_flags.py`: `--xla_cpu_max_isa=AVX
  --xla_disable_hlo_passes=algsimp,multi_output_fusion --xla_force_host_platform_device_count=4`; multi_output_fusion
  since 2026-10-09, the GPU race in the module docstring); a preset `XLA_FLAGS` that sets one of
  these flags to another value is refused. Standalone gate scripts call `mitjax.xla_flags.set_gate_xla_flags()` before
  their first jax computation. No persistent XLA compilation cache (Nikolay's decision in the ECCO port).
- `conftest.py` also sets `MJX_MINMAX_STRICT=1`: in our tests a verification build without a measured MAX/MIN
  winner at a build-dependent statement is refused; a user's run (unset) takes the documented default with one
  `MinMaxDefaultWarning` (plan decision 7, revised 2026-10-07; `mitjax/ops/fortran_minmax.py`).
- Tests that need the Fortran oracle's data (`$MJX_REFERENCE`, its replay runs) or the lessons digests
  (`$MJX_LESSONS`) are found by static reading (`mitjax/tests/oracle.py`) and carry the `oracle` marker; where the
  data is absent such a file is not imported and its one test `needs_oracle_data` skips, naming the variable (a
  user's machine). `levante.env` sets `MJX_REQUIRE_ORACLE=1`: on Levante a missing oracle FAILS those files instead,
  so a lost `$MJX_REFERENCE` can never pass a tier as skips. To see the user's behaviour here, set
  `MJX_REQUIRE_ORACLE=0` and point `MJX_REFERENCE` and `MJX_LESSONS` at an empty directory.
- Fat login node: XLA sizes compile thread pools from the core count and can abort on `ulimit -u`
  (`pthread_create ... failed`) — use `taskset` (fesom_jax lessons §8).

## Recorded versions (installed 2026-10-01, job 27825269)
Python **3.12.13**; jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt **0.10.1**; numpy **2.4.6**; scipy **1.17.1**;
netCDF4 **1.7.4**; pytest **9.0.3**. Every installed package matches `constraints.txt` except conda's own packaging /
setuptools 84.0.0 / wheel 0.48.0. Full freeze:

```
certifi==2026.5.20
cftime==1.6.5
iniconfig==2.3.0
jax==0.10.1
jax-cuda12-pjrt==0.10.1
jax-cuda12-plugin==0.10.1
jaxlib==0.10.1
ml_dtypes==0.5.4
netCDF4==1.7.4
numpy==2.4.6
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-cccl-cu12==12.9.27
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.23.0.39
nvidia-cufft-cu12==11.4.1.4
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.30.4
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.6.5
opt_einsum==3.4.0
packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1785888127/work
pluggy==1.6.0
Pygments==2.20.0
pytest==9.0.3
scipy==1.17.1
setuptools==84.0.0
wheel==0.48.0
```
