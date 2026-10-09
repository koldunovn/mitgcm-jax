# mitjax: MITgcm in JAX

mitjax is a port of the MITgcm ocean model (master at commit `63cdc0b`, checkpoint69q plus 9 commits) to
[JAX](https://github.com/jax-ml/jax). It is

- **literal:** one Python file per `.F` file, the same routines, statements and order, every routine citing the
  Fortran lines it ports; in the checked runs its output equals a gfortran build of MITgcm in every printed digit;
- **differentiable:** gradients by JAX, following each experiment's own adjoint set-up, compared with TAF's
  `results/output_adm*.txt` as testreport compares them;
- **parallel:** MITgcm's tiles split over CPU or GPU devices, the forward bit for bit the same on any device count;
- **readable** for a Fortran MITgcm developer: [docs/fortran_map.md](docs/fortran_map.md) takes you from any `.F` file
  to its Python port.

It runs MITgcm experiment directories as they are (`code/`, `input/`, `results/`), inside or outside the MITgcm tree.
Anything not ported stops with an error that names the Fortran routine.

## Status

What is checked against MITgcm, per verification experiment (from [docs/status.md](docs/status.md)):

<!-- BEGIN GENERATED: status summary (python tools/gen_docs.py) -->

| experiment | variants | forward | gradient | sharded | GPU |
|---|---|---|---|---|---|
| `tutorial_barotropic_gyre` | 1 | 1 | 1 | 1 |  |
| `tutorial_baroclinic_gyre` | 1 | 1 | 1 | 1 | 1 |
| `advect_xy` | 2 | 2 | 2 | 2 |  |
| `advect_xz` | 3 | 3 | 3 | 3 |  |
| `global_ocean.90x40x15` | 7 | 5 | 5 | 2 |  |
| `tutorial_global_oce_optim` | 1 | 1 | 1 | 1 | 1 |
| `adjustment.cs-32x32x1` | 3 | 2 |  | 1 |  |
| `solid-body.cs-32x32x1` | 1 | 1 | 1 | 1 |  |
| `advect_cs` | 1 | 1 | 1 | 1 |  |
| `global_ocean.cs32x15` | 10 | 5 | 3 | 2 |  |
| `tutorial_global_oce_latlon` | 1 | 1 |  |  |  |
| `tutorial_advection_in_gyre` | 1 | 1 |  |  |  |
| `tutorial_tracer_adjsens` | 2 | 1 | 1 |  |  |
| `vermix` | 7 | 7 | 5 |  |  |
| `front_relax` | 5 |  |  |  |  |
| `ideal_2D_oce` | 2 |  |  |  |  |
| `tutorial_reentrant_channel` | 1 | 1 |  | 1 |  |
| `MLAdjust` | 7 | 7 |  |  |  |
| `1D_ocean_ice_column` | 2 | 2 | 2 | 1 |  |
| `offline_exf_seaice` | 11 | 2 | 1 | 2 |  |
| `lab_sea` | 9 | 2 | 2 | 2 |  |
| `seaice_itd` | 3 |  |  |  |  |

Numbers of input directories with at least one test of that kind; which tests, and how the table is made:
[docs/status.md](docs/status.md).

<!-- END GENERATED: status summary -->

## Install

```bash
git clone https://github.com/koldunovn/mitgcm-jax
git clone https://github.com/MITgcm/MITgcm && git -C MITgcm checkout 63cdc0b   # next to mitgcm-jax
conda env create -f mitgcm-jax/environment.yml && conda activate mitjax
pip install -e mitgcm-jax
export MJX_UPSTREAM=$PWD/MITgcm
python -m mitjax run $MJX_UPSTREAM/verification/tutorial_barotropic_gyre --variant input --out runs/gyre
```

Linux x86_64 is the tested platform, macOS arm64 best effort: [docs/install.md](docs/install.md).

## Use it

```python
import mitjax
exp = mitjax.load("MITgcm/verification/tutorial_barotropic_gyre", variant="input")
run = exp.run(out="runs/gyre")
print(mitjax.compare(run, exp.results()).summary)      # testreport's digits against results/
```

- Notebooks: [notebooks/README.md](notebooks/README.md): four laptop notebooks (quick start, Fortran next to JAX,
  a first gradient, your own configuration) and three cluster notebooks (lab_sea with sea ice, adjoint modes on the
  cubed sphere, parallel runs).
- [docs/running.md](docs/running.md): the API, the command line, outputs and run directories.
- [docs/gradients.md](docs/gradients.md): adjoint modes, TAF's adjoint-mode switches, the solvers' derivatives,
  checkpointing, the known differences from TAF.
- [docs/configurations.md](docs/configurations.md): your own experiment, `SIZE.h`, namelists, own `.F` files, adding a
  port, and the generated list of supported and refused options.
- [docs/parallel.md](docs/parallel.md): devices, what is bit for bit and what agrees to rounding.
- [docs/checking_a_change.md](docs/checking_a_change.md): testreport against `results/`, the Fortran reference, the
  test tiers.
- [docs/troubleshooting.md](docs/troubleshooting.md): "not ported" errors, compile time and memory, float64, XLA
  flags, subnormal numbers.
- [docs/READING_GUIDE.md](docs/READING_GUIDE.md): how to read a mitjax routine next to its `.F`, with worked
  examples; [docs/fortran_map.md](docs/fortran_map.md): where each MITgcm file is.
- [docs/ISSUES_UPSTREAM.md](docs/ISSUES_UPSTREAM.md): what the port found in MITgcm itself.

## For developers of the port

The rules of the port are in `docs/PORTING_RULES.md` and `docs/KERNEL_GUIDE.md`, the environment and the test tiers
in `docs/ENV.md` and [docs/checking_a_change.md](docs/checking_a_change.md), the Fortran reference builds in
`reference/README.md`. Every test file has a cost group in `mitjax/tests/manifest_<area>.py`.

## License

MIT, see [LICENSE](LICENSE). mitjax is a translation of MITgcm and keeps MITgcm's copyright and permission notice
(MIT, Copyright (c) 2018 MITgcm Developers and Contributors) in [LICENSE-MITgcm](LICENSE-MITgcm).
