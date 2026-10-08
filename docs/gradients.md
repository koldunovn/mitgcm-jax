# Gradients

mitjax has no hand-written adjoint: JAX differentiates the ported forward code. What it differentiates follows the
experiment's own adjoint set-up (`data.ctrl`, `data.cost`/`data.ecco`, `data.autodiff`, `data.grdchk`), so that the
result can be compared with TAF's `results/output_adm*.txt` as testreport compares it.

## The calls

```python
import mitjax

ad = mitjax.load("MITgcm/verification/1D_ocean_ice_column", variant="input_ad")
g = ad.gradient(out="runs/col-grad")        # fc and dfc/dxx of the run's own cost and control
chk = ad.grdchk(out="runs/col-chk")         # the gradient check at data.grdchk's points
print(mitjax.compare(chk, ad.results()).summary)   # testreport's adm digits against results/output_adm.txt
```

- `exp.gradient(out, mode="run", devices=1, control=None, lsr_derivative=None, cg2d_derivative=None)` returns a
  `Gradient`: `fc` (the cost `COST_FINAL` prints), `adxx` (`{record: numpy array [tile, (k,) j, i]}`) and `files`:
  `adxx_<control>.<optimcycle>.data/.meta` in `<out>/rundir`, written as MITgcm writes them. `control` is a control
  file name of `data.ctrl` (default: `data.grdchk`'s `grdchkvarname`).
- `exp.grdchk(out, mode="run", lsr_derivative=None, cg2d_derivative=None)` returns a `Grdchk`: at each point of
  `data.grdchk` the adjoint gradient and the centred finite difference of the cost (`grdchk_eps`), written to
  `<out>/rundir/output.txt` in the format of MITgcm's `GRDCHK_PRINT`, so `mitjax.compare` and
  `python -m mitjax compare` (option `--kind adm`) read it as testreport reads `output_adm.txt`.
- Command line: `python -m mitjax gradient EXP_DIR --variant input_ad --out DIR [--mode run|exact] [--devices N]
  [--control NAME] [--cg2d-derivative run|exact] [--lsr-derivative run|sweeps]`, and `python -m mitjax grdchk` with
  the same `--mode`, `--cg2d-derivative` and `--lsr-derivative`.

Measured: `1D_ocean_ice_column/input_ad` matches TAF's `results/output_adm.txt` to 13 digits (`admGrd`), and the dot
test of the tangent against the adjoint agrees to a relative 2.2e-14 (notebook 03, 8 cores of an AMD EPYC 7763 CPU
node: gradient 322 s, gradient check 365 s, peak memory 5.3 GiB).

## Controls and costs

- **Controls** are the experiment's `data.ctrl` controls of the generic kinds: `xx_genarr2d`, `xx_genarr3d` and
  `xx_gentim2d` (one gradient program per time record). The first guess must be zero, as in every verification
  adjoint run (`doInitXX` with `optimcycle = 0`); another first guess stops with an error naming `CTRL_INIT_CTRLVAR`.
- **The cost** is the run's own cost as MITgcm computes it in `COST_FINAL` (`pkg/cost`, and the `pkg/ecco` gencost
  terms that the ported experiments use). A cost written in Python is not part of the API.
- **Not supported**, each stopping with an error that names the MITgcm routine: `grdchkvarindex` in `data.grdchk`,
  one-sided differences (`useCentralDiff = .FALSE.`), the `xx_fu`/`xx_fv` grdchk variables (`ncvargrd` 'w'/'s'), a
  gentim2d control with `startrec` other than 1 or with `docycle`, several MPI processes (`nPx*nPy > 1`) in the
  gradient check, and TAF's tangent-linear driver (ADMTLM).

## Modes: `mode="run"` and `mode="exact"`

TAF's adjoint of a verification experiment is not always the exact derivative of its forward run: `data.autodiff`
switches change what the reverse sweep does, and some solvers have hand-written adjoints. mitjax follows the run's
own settings by default and offers the exact derivative as the other choice. A mode only changes the backward pass:
the forward run, and `fc`, are the same in both modes, bit for bit.

- **`mode="run"`** (default): the run's `data.autodiff` switches act in the backward pass as they act in TAF's
  reverse sweep, and the CG2D derivative follows the run's `cg2dFullAdjoint` (below). This is the gradient to
  compare with TAF's `output_adm*.txt`.
- **`mode="exact"`**: no adjoint-mode switch, and the full derivative of the CG2D solve: the derivative of the forward
  run as it is computed.

The `data.autodiff` switches (namelist `AUTODIFF_PARM01`) that `mode="run"` follows:

| switch | what TAF does in the reverse sweep | in mitjax |
|---|---|---|
| `useApproxAdvectionInAdMode` | advection schemes 33 (DST3 with flux limiter) differentiated as scheme 30 (no limiter), for temperature, salinity and sea ice | a backward-only hook in `GAD_ADVECTION` and `SEAICE_ADVECTION` (`mitjax/ad/approx_advection.py`) |
| `SEAICEuseFREEDRIFTswitchInAd` | the sea-ice dynamics differentiated as free drift instead of the LSR solve | a backward-only hook (`mitjax/ad/freedrift_switch.py`) |
| `cg2dFullAdjoint` | `.FALSE.`: the CG2D operator passive; `.TRUE.`: its full adjoint (`CG2D_MAD`) | the CG2D derivative's setting (below) |

The other switches of that namelist are refused in `mode="run"` when they would make TAF's reverse sweep differ from
the forward (an error lists them): an `use<Pkg>inAdMode` switch that differs from the forward's `use<Pkg>`,
`inAdExact = .FALSE.`, `SEAICEapproxLevInAd` other than 0, `viscFacInAd` different from `viscFacInFw`,
`SIregFacInAd`/`SIregFacInFw` set, `SEAICEuseDYNAMICSswitchInAd`, and `useApproxAdvectionInAdMode` where scheme 33
is used outside `GAD_ADVECTION`, with implicit vertical advection, or with passive tracers
(`mitjax/drivers/ad_switches.py`).

## The CG2D solve: an implicit derivative

The pressure solver's iterations are never differentiated. The derivative of CG2D is the derivative of its converged
solution: the adjoint system is solved with the same CG2D (`mitjax/ad/cg2d_rule.py`), as TAF's hand-written
`CG2D_MAD` (`pkg/autodiff/cg2d_mad.F`) does. What it includes is `cg2d_derivative`:

- `"run"`: as the run's TAF adjoint. The operator is active only in a TAF build with a moving operator
  (`NONLIN_FRSURF` or `ALLOW_DEPTH_CONTROL` compiled) and `cg2dFullAdjoint = .TRUE.`; otherwise it is passive
  (`CG2D_MAD`'s `ELSE` branch).
- `"exact"`: the operator always active: the exact gradient of the converged solve.
- `None` (default): the mode's choice, `"run"` for `mode="run"` and `"exact"` for `mode="exact"`.

On a run without `data.autodiff` switches, `mode="exact", cg2d_derivative="run"` gives the gradient of `mode="run"`
bit for bit (`global_ocean.cs32x15/input_ad`, `nonlinFreeSurf = 4` and `cg2dFullAdjoint = .FALSE.`, is the cheapest
verification case where the two CG2D choices differ).

Where no control reaches the operator (a linear free surface, or `nonlinFreeSurf` up to 2 without
`selectImplicitDrag = 2`) both settings give the same gradient. With `cg2dFullAdjoint = .TRUE.` the implicit rule
matched TAF's full `CG2D_MAD` to 14 digits on `global_ocean.90x40x15/input_ad.bottomdrag`.

**TAF's tangent-linear model is not `mode="exact"` when the free surface is nonlinear.** TAF's tangent of CG2D is CG2D
applied to the tangents, with the operator passive, even with `cg2dFullAdjoint = .TRUE.`; `mode="exact"` includes
the operator's derivative. On `global_ocean.cs32x15/input_ad.seaice` (`nonlinFreeSurf = 4`, r* coordinate),
`mode="exact"` agrees with TAF's `output_tlm.seaice.txt` to 7-8 digits, and the same program with the CG2D operator
passive to 11-12 digits. Where `cg2dFullAdjoint = .FALSE.` (as there), `mode="exact", cg2d_derivative="run"` is that
program: TAF's tangent-linear model. Where `cg2dFullAdjoint = .TRUE.` and the operator moves, TAF's adjoint includes
the operator and its tangent does not, and mitjax's gradient follows the adjoint.

## The sea-ice LSR solve: the executed sweeps

`SEAICE_LSR` iterates line relaxations until a tolerance. Where the build defines `SEAICE_LSR_ADJOINT_ITER`
(`lab_sea/code_ad`), TAF tapes every executed sweep, and finite differences converge to the derivative of that sweep
sequence; the implicit derivative of the converged solve is about 1e-3 away from it there. So mitjax differentiates
the executed sweeps too: the LSR iteration runs as a fixed-length scan of `SOLV_MAX_FIXED` sweeps (the Fortran's cap,
500) in which a sweep after convergence does nothing, with a recomputation per sweep in the backward pass. The forward
is bit for bit the forward of the plain iteration. One LSR call costs about 11.5 s of compile time and 30.5 MB of
temporaries (measured on `lab_sea/input_ad`, CPU). This is the one exception to "solver iterations are never
differentiated".

`lsr_derivative` selects it (decided: docs plan 20261006 decision 12 a, which extends master plan decision 17 to
every build without `SEAICE_LSR_ADJOINT_ITER`):

- `None` (default): the executed sweeps, for every build. Where the build defines `SEAICE_LSR_ADJOINT_ITER` this is
  what TAF tapes; where it does not (`global_ocean.cs32x15/input_ad.seaice`, `input_ad.seaice_dynmix`) it is the
  project's choice (plan decisions 14 and 17): the exact derivative of the executed sweeps, which TAF's
  tangent-linear model also computes.
- `"sweeps"`: the same, stated explicitly.
- `"run"`: as the build says: the executed sweeps with `SEAICE_LSR_ADJOINT_ITER`; otherwise the LSR solve is forward
  only and a gradient through it stops with an error, so that no gradient is taken through an untaped solve.

Without `SEAICE_LSR_ADJOINT_ITER` TAF's own LSR adjoint is wrong by construction (it overwrites its per-sweep tape;
`pkg/seaice/seaice_lsr.F:805-807` says so). For those runs mitjax's reference is TAF's tangent-linear model.

## Known differences from TAF's results

Each of these is measured, and reported to the MITgcm developers where it concerns MITgcm itself
([MITgcm#1041](https://github.com/MITgcm/MITgcm/issues/1041); details in [ISSUES_UPSTREAM.md](ISSUES_UPSTREAM.md)).

| run | TAF reference | mitjax |
|---|---|---|
| `lab_sea/input_ad` | TAF's adjoint (`output_adm.txt`) is 1e-4 to 9e-4 off TAF's own tangent-linear model (`output_tlm.txt`), the Tapenade adjoint (`output_tap_adj.txt`) and finite differences: TAF's adjoint of `SEAICE_LSR` | matches the tangent-linear model to 12 digits and the Tapenade adjoint to 12-13 digits; against `output_adm.txt` 3-4 digits (notebook 05) |
| `global_ocean.cs32x15/input_ad.seaice` | no `SEAICE_LSR_ADJOINT_ITER`, and `useApproxAdvectionInAdMode`: TAF's adjoint and tangent differentiate different operators (2-3 digits apart) | the reference is TAF's tangent-linear model and finite differences; `mode="run"` against TAF's adjoint 7-8 digits |
| `global_ocean.cs32x15/input_ad.seaice_dynmix` | `SEAICEuseFREEDRIFTswitchInAd`: TAF's adjoint and tangent agree to 7-8 digits | `mode="run"` against TAF's adjoint 11-12 digits, `mode="exact"` against TAF's tangent 10-11 digits (notebook 06) |

The digits of the last two rows were measured with `lsr_derivative="sweeps"` (now also the default there) at the
four `data.grdchk` points.

## Checkpointing

A gradient over N time steps stores the model state at every step boundary and recomputes each step once in the
backward pass (`mitjax/drivers/checkpoint.py`, schedule `"step"`, the API's choice). Two-level schedules (about
sqrt(N) stored states, every step recomputed twice) and windows split into chunks with the boundary states on the
host exist in the drivers for long windows; the API does not expose them.

## Checking a gradient yourself

- **Finite differences:** `exp.grdchk` at the `data.grdchk` points, or change `grdchk_eps` and the points in
  `data.grdchk` of a copy of the experiment. Agreement with finite differences at one step size is not a proof: test
  several step sizes, and prefer points where the cost is smooth.
- **Dot test:** the tangent `J v` against the adjoint `v . g` for a random direction `v`; notebook 03 shows it with
  the drivers below the API (`mitjax/drivers/adjoint_run.py`).
- **Finite everywhere:** the gradient has finite values on every point, land, halos and padding included; a NaN in
  the gradient of a masked point is a bug.
