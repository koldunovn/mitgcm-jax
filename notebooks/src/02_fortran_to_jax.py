# Source of notebooks/02_fortran_to_jax.ipynb (python tools/build_notebooks.py writes the notebook from this file).

# %% [markdown]
# # 02 From Fortran to JAX: read a routine, change it, check it
#
# - **Tier:** laptop (CPU only)
# - **Run time:** about 4 min (measured: 251 s on a compute node pinned to 8 cores; three runs of the gyre, mostly
#   compiling)
# - **Peak memory:** about 2 GB (measured: 2.0 GiB)
# - **Experiment:** `verification/tutorial_barotropic_gyre`, variant `input`
# - **Shows:** how a `.F` file reads in mitjax (Fortran-index arrays, loops, `# :NN` line citations, vertical loops
#   as `scan_k` / `scan_levels`, the per-statement MAX/MIN convention); then one line of one routine changed, the
#   model rerun and compared with the Fortran's `results/`.
#
# Where to look things up:
# - [`docs/fortran_map.md`](../docs/fortran_map.md): every MITgcm file and routine and the Python file that ports it,
#   by directory, e.g. [`model/src`](../docs/fortran_map.md#modelsrc),
#   [`pkg/mom_fluxform`](../docs/fortran_map.md#pkgmom_fluxform),
#   [`pkg/generic_advdiff`](../docs/fortran_map.md#pkggeneric_advdiff),
#   [`pkg/seaice`](../docs/fortran_map.md#pkgseaice); its first section is the short version of the conventions
#   below.
# - [`docs/READING_GUIDE.md`](../docs/READING_GUIDE.md): the long version, with worked examples (section numbers are
#   quoted below as §n).
# - [`docs/review/M1_SIDE_BY_SIDE.html`](../docs/review/M1_SIDE_BY_SIDE.html): the core routines of the dynamics
#   side by side with their Fortran, line by line (open the file in a browser).

# %%
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import html
import inspect
from pathlib import Path

from IPython.display import HTML, display

import mitjax
from mitjax import paths

UPSTREAM = paths.UPSTREAM                          # $MJX_UPSTREAM: MITgcm at 63cdc0b
PKG = Path(mitjax.__file__).parent                 # the mitjax package


def fortran(rel, a, b):
    """Lines a..b of an MITgcm source file, numbered as in the file."""
    lines = (UPSTREAM / rel).read_text().splitlines()
    return "\n".join(f"{n:4d}  {lines[n - 1]}" for n in range(a, b + 1))


def python(rel, start, stop=None):
    """The lines of mitjax/<rel> from the first line containing `start` up to (not including) the first after it
    containing `stop` (to the end of the file without `stop`)."""
    lines = (PKG / rel).read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if start in ln)
    j = next((n for n in range(i + 1, len(lines)) if stop and stop in lines[n]), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def side_by_side(left_title, left, right_title, right):
    cell = '<td style="vertical-align:top;text-align:left"><b>{}</b><pre style="font-size:11px">{}</pre></td>'
    display(HTML("<table><tr>" + cell.format(html.escape(left_title), html.escape(left))
                 + cell.format(html.escape(right_title), html.escape(right)) + "</tr></table>"))

# %% [markdown]
# ## One routine: `MOM_U_XVISCFLUX`
#
# `pkg/mom_fluxform/mom_u_xviscflux.F` computes the zonal viscous flux of `u` at one level. Its port is
# `mitjax/pkg/mom_fluxform/mom_u_xviscflux.py` (one `.py` per `.F`, same path, lower-case names).

# %%
side_by_side("pkg/mom_fluxform/mom_u_xviscflux.F", fortran("pkg/mom_fluxform/mom_u_xviscflux.F", 52, 71),
             "mitjax/pkg/mom_fluxform/mom_u_xviscflux.py",
             python("pkg/mom_fluxform/mom_u_xviscflux.py", "def mom_u_xviscflux"))

# %% [markdown]
# How to read it (READING_GUIDE §1-4):
#
# - **The docstring head** names the Fortran call and the span it ports: `@63cdc0b
#   pkg/mom_fluxform/mom_u_xviscflux.F:6-74`. In the body, `# :54-71` points at lines of that same file.
# - **Arguments** are the Fortran's in the Fortran order, minus `bi, bj, myThid`; outputs are returned. The common
#   blocks come in as objects: `grid` is `GRID.h`, `cfg.size` is `SIZE.h`, `cfg.cpp` the preprocessed CPP options.
# - **Fortran-index arrays.** `uFld[i+1, j]` is `uFld(i+1,j)` with Fortran index values: every array keeps its
#   declared bounds (`1-OLx:sNx+OLx`), and an index outside them is an error.
# - **Loops.** `j = loop_j(1-OLy, sNy+OLy-1)` is the `DO j` statement and `.at[i, j].set(...)` the assignment
#   inside the loops: the whole loop nest at once. The tile loop `DO bj / DO bi` is implicit (every statement acts
#   on all tiles; storage is `[tile, k, j, i]`).
# - **CPP options** are Python `if`s on `cfg.cpp` (`#ifdef COSINEMETH_III`); integer, logical and character
#   namelist values are static, REAL values are traced (they can be differentiated).
# - **Same statements, same order, same parentheses.** Floating-point addition is not associative; the port keeps
#   the Fortran's expression tree so that it reproduces the Fortran to the last bit.

# %% [markdown]
# ## Vertical loops: `scan_k` and `scan_levels`
#
# A recursion in `k` (a tridiagonal solve, an integration from the surface) is written with `scan_k`
# (READING_GUIDE §7). Here the forward sweep of `model/src/solve_tridiagonal.F`, used by the implicit vertical
# diffusion: `k = 1` is written out, then `scan_k` runs `k = 2..Nr` in the Fortran order; its carry `(cpm1, ypm1)`
# is level `k-1`, `x[...]` the arrays at level `k`. A pivot test `IF (tmpVar .NE. 0)` becomes `jnp.where` with a
# guarded division (`safe_div`), so that the branch not taken stays finite for the derivative (READING_GUIDE §6).

# %%
side_by_side("model/src/solve_tridiagonal.F", fortran("model/src/solve_tridiagonal.F", 237, 275),
             "mitjax/model/src/solve_tridiagonal.py",
             python("model/src/solve_tridiagonal.py", "# :241-254", "# :279-295"))

# %% [markdown]
# A routine the Fortran calls once per level (`DO k ... CALL INTEGRATE_FOR_W(..k..)`) is written for one level,
# and its caller runs it with `scan_levels`; levels the Fortran treats differently (here `k = 1` and `k = Nr`) run
# as separate calls next to the scan (`peel`):

# %%
side_by_side("model/src/integr_continuity.F", fortran("model/src/integr_continuity.F", 266, 281),
             "mitjax/model/src/integr_continuity.py",
             python("model/src/integr_continuity.py", "# :267  DO k=Nr,1,-1", "if params.implicitIntGravWave"))

# %% [markdown]
# ## MAX and MIN: one convention per statement
#
# Every Fortran `MAX`/`MIN` of REAL values is `MAX(a, b, p="a")` or `p="b"` (READING_GUIDE §9). `p` names the
# argument gfortran returns at that statement when the two tie with different signs of zero (`+0` vs `-0`) or one
# is NaN; it was measured, statement by statement, from our gfortran builds. It matters only for bitwise agreement
# (and for the derivative at a tie). The lopping factor `hFacC` of `model/src/ini_masks_etc.F`:

# %%
side_by_side("model/src/ini_masks_etc.F", fortran("model/src/ini_masks_etc.F", 105, 121),
             "mitjax/model/src/ini_masks_etc.py",
             python("model/src/ini_masks_etc.py", "# :105-124", "# :126-143"))

# %% [markdown]
# At a few statements gfortran's choice depends on the build (the options and `SIZE.h`). There mitjax looks your
# build up by a hash of its preprocessed options and `SIZE.h`; where our Fortran reference never measured the
# statement in your build, mitjax takes a documented default and prints one warning naming the statement. Notebook
# 04 shows it.

# %% [markdown]
# ## Change one line and check it against the Fortran
#
# Suppose you want to try a change in `MOM_U_XVISCFLUX`, in its Laplacian term `-viscAh_D*(uFld(i+1,j)-uFld(i,j))`.
# We try two changes of that one line:
#
# - **reordered:** `-viscAh_D*uFld(i+1,j) + viscAh_D*uFld(i,j)`, the same mathematics with a different order of
#   floating-point operations;
# - **halved:** `-0.5*viscAh_D*(uFld(i+1,j)-uFld(i,j))`, half the zonal Laplacian flux of `u`: a real change.
#
# In your own copy of mitjax you would edit `mitjax/pkg/mom_fluxform/mom_u_xviscflux.py`. In this notebook we leave
# the files alone and replace the function where its caller looks it up: `mom_fluxform.py` imported it by name, so
# its module attribute `mom_u_xviscflux` is what the model calls when the time step is traced.

# %%
import jax
import numpy as np

from mitjax.farray import loop_i, loop_j
import mitjax.pkg.mom_fluxform.mom_fluxform as mom_fluxform

original = mom_fluxform.mom_u_xviscflux


def changed_xviscflux(laplacian):
    """MOM_U_XVISCFLUX as in mitjax, with the Laplacian term (the one line marked CHANGED) given by `laplacian`."""
    def mom_u_xviscflux(k, uFld, del2u, xViscFluxU, viscAh_D, viscA4_D, *, cfg, grid):
        sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
        dyF, drF, hFacC, recip_dxF = grid.dyF, grid.drF, grid.hFacC, grid.recip_dxF
        cosFacU = grid.cosFacU
        cF4 = grid.sqCosFacU if cfg.cpp.flag("COSINEMETH_III", "MOM_COMMON_OPTIONS.h") else grid.cosFacU
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        xViscFluxU = xViscFluxU.at[i, j].set(
            dyF[i, j]*drF[k]*hFacC[i, j, k]
            * (
               laplacian(viscAh_D[i, j], uFld[i+1, j], uFld[i, j])     # CHANGED: -viscAh_D*(u(i+1)-u(i))
               * cosFacU[j]
               + viscA4_D[i, j]*(del2u[i+1, j]-del2u[i, j])
               * cF4[j]
              )*recip_dxF[i, j])
        return xViscFluxU
    return mom_u_xviscflux


CHANGES = {
    "reordered": lambda A, u_ip1, u_i: -A*u_ip1 + A*u_i,
    "halved": lambda A, u_ip1, u_i: -0.5*A*(u_ip1 - u_i),
}


def new_out(name):
    n = 1
    while Path(f"runs/{name}-{n}").exists():
        n += 1
    return Path(f"runs/{name}-{n}")


VERIFICATION = UPSTREAM / "verification"
exp = mitjax.load(VERIFICATION / "tutorial_barotropic_gyre", variant="input")
assert exp.config.cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h")   # so the options header above is right

runs = {"original": exp.run(out=new_out("02_gyre"))}
for name, laplacian in CHANGES.items():
    mom_fluxform.mom_u_xviscflux = changed_xviscflux(laplacian)
    jax.clear_caches()                                          # trace the time step again, with the change
    try:
        runs[name] = exp.run(out=new_out(f"02_gyre_{name}"))
    finally:
        mom_fluxform.mom_u_xviscflux = original                 # put the original back
        jax.clear_caches()

# %%
reports = {name: mitjax.compare(run, exp.results()) for name, run in runs.items()}
for name, rep in reports.items():
    print(f"{name:10s}", rep.summary)
print("\nvariable   " + "".join(f"{n:>11s}" for n in reports))
for k, var in enumerate(reports["original"].run.variables):
    print(f"{var.name:10s} " + "".join(f"{rep.run.variables[k].digits:11d}" for rep in reports.values()))
eta0 = runs["original"].fields["etaN"]
for name in CHANGES:
    print(f"max |etaN {name} - etaN original| = {np.abs(runs[name].fields['etaN'] - eta0).max():.2e} m "
          f"(|etaN| <= {np.abs(eta0).max():.2e} m)")

# %% [markdown]
# What this says:
#
# - The original code reproduces the Fortran's `results/` to every printed digit (16; 22 is testreport's count for
#   two zeros).
# - The reordering changes the last digits after ten time steps: testreport still passes (its criterion is 10
#   digits), but the result is no longer the Fortran's bit for bit. That is why the port keeps the Fortran's
#   statements, order and parentheses, and why its own tests compare with the Fortran bit for bit on our reference
#   machine: there a harmless reordering and a real error are not confused.
# - Halving the flux fails testreport: the velocity and surface-height statistics keep about 5 digits.
#
# The asserts below hold on any platform (plan decision 10: another CPU may round differently, so nothing here is
# asserted bit for bit).

# %%
MIN_DIGITS = 10
for name in ("original", "reordered"):
    assert reports[name].verdict == "pass", name
    assert reports[name].run.variables[0].digits >= MIN_DIGITS, name
assert reports["halved"].verdict == "FAIL"
assert np.abs(runs["halved"].fields["etaN"] - eta0).max() > 0
