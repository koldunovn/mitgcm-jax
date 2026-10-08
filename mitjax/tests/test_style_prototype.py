"""Task 8 gates of the two code styles (dev/prototype/style_farray.py, style_slices.py) against the gfortran replay
harness (reference/replay; the run named by reference/replay/CURRENT, relative to $MJX_REFERENCE):

  * bitwise: every output of MOM_CALC_KE (KEscheme -1..3), GAD_DST3_ADV_X (calcCFL T/F) and MOM_VI_HDISSIP (all 8
    harmonic/biharmonic/useVariableViscosity cases), all 36 tiles x 15 levels, ALL points incl. halos and the points
    the routine does not write (they keep the prior field), equal bit for bit, under the gate XLA flags (conftest.py);
  * identical programs: the jaxprs are equal and the optimized HLO is equal after removing debug locations
    (`metadata={...}` and the FileNames/FunctionNames/FileLocations/StackFrames tables);
  * FD: the JAX directional derivative of a weighted sum of all outputs agrees with central differences (h sweep);
  * finite gradients on all lanes (halos, land, grid fields, parameters);
  * XLA:CPU float64 divide vs gfortran (dev/prototype/divide_probe.py, fresh processes per flag set).
Every gate has a planted error that is shown to bite. The replay output must exist: these tests fail, not skip,
without it.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "prototype"))

import proto_run as pr  # noqa: E402
import proto_setup as ps  # noqa: E402


@pytest.fixture(scope="module")
def R():
    return pr.current_replay()          # FileNotFoundError (a failure) when the replay run is missing


def run_case(R, mod, style, routine, case):
    f = jax.jit(pr.all_levels_fn(mod, style, R.cfg, routine, case))
    return [np.asarray(o) for o in f(*R.args(style, routine))]


def diffs_vs_fortran(R, mod, style, routine, case):
    return {n: pr.bit_diff(o, R.out[n]) for n, o in zip(pr.out_names(routine, case), run_case(R, mod, style, routine,
                                                                                               case))}


def test_replay_setup(R):
    """The build's options give the expected cfg; the grid is realistic (land, partial cells); inputs are intact."""
    c = R.cfg
    assert (c.sNx, c.sNy, c.OLx, c.OLy, c.Nr) == (10, 10, 3, 3, 15)            # global_ocean.90x40x15/code/SIZE.h
    assert R.size["nSx"] * R.size["nSy"] == 36
    assert (c.ALLOW_AUTODIFF, c.ALLOW_DIAGNOSTICS, c.MOM_VI_ORIGINAL_VISCA4, c.OLD_DST3_FORMULATION) == (
        False, True, False, False)
    h = R.grid_np["hFacC"]
    assert (h == 0).sum() > 10000 and ((h > 0) & (h < 1)).sum() > 1000          # land and partial cells
    assert (R.grid_np["maskW"] == 0).sum() > 10000
    meta = json.loads((R.rundir / "replay_in.json").read_text())
    import hashlib
    for name, sha in meta["sha256"].items():
        assert hashlib.sha256((R.rundir / name).read_bytes()).hexdigest() == sha
    # the harness's overrides reached the routines (a missing factor or a D/Z swap is visible)
    assert np.all(R.grid_np["recip_deepFacC"] != 1.0) and R.grid_np["viscAhD"] != R.grid_np["viscAhZ"]
    for name in R.out:
        assert np.all(np.isfinite(R.out[name])), name


@pytest.mark.parametrize("style", ["farray", "slices"])
def test_replay_bitwise(R, style):
    bad = {}
    for routine, case in pr.CASES:
        for name, n in diffs_vs_fortran(R, pr.STYLES[style], style, routine, case).items():
            if n:
                bad[name] = n
    assert bad == {}, f"{style}: points differing from gfortran (of {R.out['KE_0'].size} each): {bad}"


# planted errors, each a bug class the gate must see: (style, routine, case, old text, new text)
PLANTS = {
    "rim not kept [E§2]": ("farray", "mom_calc_ke", 0, "    j = loop_j(1-OLy, sNy+OLy-1)                                    # DO j",
                           "    KE = KE.local('KE')  # planted\n    j = loop_j(1-OLy, sNy+OLy-1)  # DO j"),
    "association": ("farray", "mom_vi_hdissip", (True, False, False),
                    "+ viscAhD*              (Dij-Dim)*recip_dyC[i, j])",
                    "+ viscAhD*((Dij-Dim)*recip_dyC[i, j]))"),
    "mask dropped": ("farray", "gad_dst3_adv_x", True, "Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]",
                     "Rjm = (tracer[i-1, j]-tracer[i-2, j])"),
    "factor dropped": ("farray", "gad_dst3_adv_x", True, "*recip_dxC[i, j]*recip_deepFacC[k])",
                       "*recip_dxC[i, j])"),
    "D/Z swap": ("slices", "mom_vi_hdissip", (True, False, False), "uD2 = (viscAhD*", "uD2 = (viscAhZ*"),
    "transposed shift": ("slices", "mom_calc_ke", -1, "(uFld[:, j, i]+uFld[:, j, ip1])**2",
                         "(uFld[:, j, i]+uFld[:, jp1, i])**2"),
}


def test_replay_negative_controls(R):
    bitten = {}
    for label, (style, routine, case, old, new) in PLANTS.items():
        mod = pr.planted(pr.STYLES[style], old, new, name=re.sub(r"\W", "_", label))
        bitten[label] = sum(diffs_vs_fortran(R, mod, style, routine, case).values())
    print("negative controls, points differing from gfortran:", bitten)
    assert all(n > 0 for n in bitten.values()), bitten


def programs(R, mod, style, routine, case, k=2, names=True):
    f = pr.level_fn(mod, style, R.cfg, routine, case, k)
    fields, grid, params, dt = R.args(style, routine)
    fk = {n: a[:, k - 1] for n, a in fields.items()}
    return str(jax.make_jaxpr(f)(fk, grid, params, dt)), pr.normalize_hlo(jax.jit(f).lower(fk, grid, params, dt)
                                                                       .compile().as_text(), names)


def test_identical_programs(R):
    differ, names_only = [], []
    for routine, case in pr.CASES:
        ja, ha = programs(R, pr.style_farray, "farray", routine, case)
        jb, hb = programs(R, pr.style_slices, "slices", routine, case)
        assert "metadata=" not in ha and "FileNames" not in ha
        if ja != jb or ha != hb:
            differ.append((routine, case, ja == jb, ha == hb))
        _, ha_raw = programs(R, pr.style_farray, "farray", routine, case, names=False)
        _, hb_raw = programs(R, pr.style_slices, "slices", routine, case, names=False)
        if ha_raw != hb_raw:
            names_only.append((routine, case))
    print(f"optimized HLO identical modulo names in all {len(pr.CASES)} cases; textually identical (debug "
          f"locations removed) except parameter names in: {names_only}")
    assert differ == [], differ
    # the whole k loop (15 levels in one program) for one routine
    fa = pr.all_levels_fn(pr.style_farray, "farray", R.cfg, "mom_vi_hdissip", (True, True, True))
    fb = pr.all_levels_fn(pr.style_slices, "slices", R.cfg, "mom_vi_hdissip", (True, True, True))
    ha = pr.normalize_hlo(jax.jit(fa).lower(*R.args("farray", "mom_vi_hdissip")).compile().as_text())
    hb = pr.normalize_hlo(jax.jit(fb).lower(*R.args("slices", "mom_vi_hdissip")).compile().as_text())
    assert ha == hb
    # the Fortran-index program called with plain arrays (FArrays built inside the traced program, as in the A100
    # plain-array-signature run) is the same program as well
    from mitjax.farray import FArray
    b = ps.bounds(R.cfg)

    def fc(fl, gr, pa, dt):
        wrapped = ps.Common(**{n: FArray(getattr(gr, n), n, tiled=(n != "recip_deepFacC"), **{d: b[d] for d in dims})
                               for n, dims in ps.GRID_DECL.items()})
        return fa(fl, wrapped, pa, dt)
    hc = pr.normalize_hlo(jax.jit(fc).lower(*R.args("slices", "mom_vi_hdissip")).compile().as_text())
    assert hc == hb
    # negative control: a re-associated statement is a different program
    style, routine, case, old, new = PLANTS["association"]
    jc, hc = programs(R, pr.planted(pr.style_farray, old, new, "assoc_hlo"), "farray", routine, case)
    jb, hb = programs(R, pr.style_slices, "slices", routine, case)
    assert jc != jb and hc != hb


def unphysical_cfl(R, k):
    """Points of level k where GAD_DST3_ADV_X's computed CFL number |uFld*deltaTloc*recip_dxC*recip_deepFacC| > 1:
    the halo rows beyond the domain's northern and southern edges of the spherical-polar grid, where INI_GRID's
    recip_dxC reaches 3.7e10 1/m (uCFL up to 6e14, the upwind factor A up to 1e29)."""
    g = R.grid_np
    cfl = np.abs(R.fields["uVel"][:, k - 1] * R.scal["deltaTloc"] * g["recip_dxC"] * g["recip_deepFacC"][k - 1])
    return cfl > 1.0


def _loss_fn(R, mod, style, routine, case, k, physical_only=False):
    """Scalar loss: every output point weighted by r / (|Fortran output| + median), so all points count, as a
    function of (fields_k, params, deltaTloc, grid). physical_only (DST3 with calcCFL): weight 0 where the CFL number
    is unphysical (> 1), see test_fd_gradients."""
    f = pr.level_fn(mod, style, R.cfg, routine, case, k)
    rng = np.random.default_rng(3)
    w = []
    for n in pr.out_names(routine, case):
        ref = np.abs(R.out[n][:, k - 1])
        wn = rng.standard_normal(ref.shape) / (ref + np.median(ref) + 1e-300)
        if physical_only and routine == "gad_dst3_adv_x" and case:
            wn = np.where(unphysical_cfl(R, k), 0.0, wn)
        w.append(jnp.asarray(wn))

    def loss(fields, params, dt, grid):
        outs = f(fields, grid, params, dt)
        return sum(jnp.sum(wi * o) for wi, o in zip(w, outs))
    return loss


def _fd_check(R, mod, style, routine, case, k=7, physical_only=True, with_tangent=False):
    """(|tangent - adjoint| / |adjoint|, {h: |FD - adjoint| / |adjoint|}[, {h: |FD - tangent| / |tangent|}]) for the
    directional derivative along a random direction scaled by each input's magnitude."""
    loss = _loss_fn(R, mod, style, routine, case, k, physical_only)
    fields, grid, params, dt = R.args(style, routine)
    fk = {n: a[:, k - 1] for n, a in fields.items()}
    x = (fk, params, dt)
    rng = np.random.default_rng(11)
    d = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(np.shape(a))) * (jnp.abs(a) + 1e-30), x)
    lin = jax.jit(lambda x, d: jax.jvp(lambda y: loss(*y, grid), (x,), (d,))[1])(x, d)
    g = jax.jit(jax.grad(lambda y: loss(*y, grid)))(x)
    gd = sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(d)))
    lj = jax.jit(lambda y: loss(*y, grid))
    errs, errs_tangent = {}, {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        xp = jax.tree_util.tree_map(lambda a, b: a + h * b, x, d)
        xm = jax.tree_util.tree_map(lambda a, b: a - h * b, x, d)
        fd = (lj(xp) - lj(xm)) / (2 * h)
        errs[h] = float(abs(fd - gd) / abs(gd))
        errs_tangent[h] = float(abs(fd - lin) / abs(lin))
    if with_tangent:
        return float(abs(lin - gd) / abs(gd)), errs, errs_tangent
    return float(abs(lin - gd) / abs(gd)), errs


@pytest.mark.parametrize("style", ["farray", "slices"])
def test_fd_gradients(R, style):
    """At smooth points (random inputs: no |uTrans| or uCFL at the kink of ABS), central FD of the weighted output
    sum along a random direction of all inputs of the routine (fields, PARAMS.h floats, deltaTloc) agrees with the
    JAX gradient: best relative error over the h sweep < 1e-8; tangent == adjoint to 1e-12.

    DST3 with calcCFL: output points with an unphysical CFL number (> 1; 340 of 9216 points at level 7, all in the halo rows
    beyond the domain's N/S edges) get weight 0. There uT = 0.5*(u+|u|)*A + 0.5*(u-|u|)*B with |A| ~ 1e29 and
    |B| ~ 4 (measured at tile 27, level 7, j-row 15): forward mode forms t - t = 0 before multiplying by A, but
    reverse mode accumulates +-0.5*A*ct through both uses of |u|, and the 0.5*B*ct terms vanish against them, so
    the gradient w.r.t. uTrans is exactly 0 instead of B*ct at 3 of 9216 points (jvp 1.1e-7, vjp 0.0). That is the
    formula in floating point, the same in both styles (test_gradients_identical_between_styles) and in a TAF-style
    adjoint of the same statement; test_reverse_mode_cancellation_at_unphysical_cfl keeps the measurement."""
    worst = {}
    for routine, case in [("mom_calc_ke", c) for c in (-1, 0, 1, 2, 3)] + [("gad_dst3_adv_x", True),
                                                                          ("gad_dst3_adv_x", False),
                                                                          ("mom_vi_hdissip", (True, True, True)),
                                                                          ("mom_vi_hdissip", (True, True, False))]:
        dot, errs = _fd_check(R, pr.STYLES[style], style, routine, case)
        worst[(routine, case)] = (dot, min(errs.values()))
    print(style, "tangent-vs-adjoint and best FD relative errors:", worst)
    assert all(dot < 1e-12 and fd < 1e-8 for dot, fd in worst.values()), worst


def test_reverse_mode_cancellation_at_unphysical_cfl(R):
    """The measurement behind the physical-CFL restriction of the FD gate: with all points weighted, tangent and
    adjoint differ (> 1e-3), FD sides with the tangent, and the excluded points are a small set of halo points."""
    k = 7
    bad = unphysical_cfl(R, k)
    jj = np.nonzero(bad)[1]
    assert 0 < bad.sum() < 0.05 * bad.size              # measured: 340 of 9216 at k = 7
    Ny = R.cfg.sNy + 2 * R.cfg.OLy
    assert set(jj.tolist()) <= set(range(R.cfg.OLy)) | set(range(Ny - R.cfg.OLy, Ny))   # halo rows only
    dot, errs, errs_t = _fd_check(R, pr.style_farray, "farray", "gad_dst3_adv_x", True, k=k, physical_only=False,
                                  with_tangent=True)
    print(f"all points weighted: tangent-vs-adjoint {dot:.3e}, FD-vs-adjoint {errs}, FD-vs-tangent {errs_t}")
    assert dot > 1e-3 and min(errs.values()) > 1e-3 and min(errs_t.values()) < 1e-8


def test_gradients_identical_between_styles(R):
    """Identical programs give identical gradients: for every case the full gradient (fields, grid, params,
    deltaTloc) of the two styles is equal bit for bit."""
    differ = []
    for routine, case in pr.CASES:
        ga = jax.tree_util.tree_leaves(_grad_all_lanes(R, pr.style_farray, "farray", routine, case))
        gb = jax.tree_util.tree_leaves(_grad_all_lanes(R, pr.style_slices, "slices", routine, case))
        assert len(ga) == len(gb)
        n = sum(pr.bit_diff(np.asarray(a), np.asarray(b)) for a, b in zip(ga, gb))
        if n:
            differ.append((routine, case, n))
    assert differ == [], differ


def test_fd_negative_control(R):
    """A derivative cut that keeps the forward (stop_gradient on |uTrans|) is seen by the FD gate."""
    mod = pr.planted(pr.style_farray, "0.5*(uTrans[i, j]+jnp.abs(uTrans[i, j]))",
                     "0.5*(uTrans[i, j]+jax.lax.stop_gradient(jnp.abs(uTrans[i, j])))", "fd_cut")
    mod.jax = jax
    assert sum(diffs_vs_fortran(R, mod, "farray", "gad_dst3_adv_x", True).values()) == 0   # forward unchanged
    _, errs = _fd_check(R, mod, "farray", "gad_dst3_adv_x", True)
    assert min(errs.values()) > 1e-3, errs


def _grad_all_lanes(R, mod, style, routine, case, k=4):
    loss = _loss_fn(R, mod, style, routine, case, k)
    fields, grid, params, dt = R.args(style, routine)
    fk = {n: a[:, k - 1] for n, a in fields.items()}
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(fk, params, dt, grid)


@pytest.mark.parametrize("style", ["farray", "slices"])
def test_gradients_finite_all_lanes(R, style):
    bad = []
    for routine, case in pr.CASES:
        g = _grad_all_lanes(R, pr.STYLES[style], style, routine, case)
        for path, leaf in jax.tree_util.tree_leaves_with_path(g):
            if not np.all(np.isfinite(np.asarray(leaf))):
                bad.append((routine, case, jax.tree_util.keystr(path)))
    assert bad == [], bad


def test_gradients_finite_negative_control(R):
    """A forward-only guard (`where(h > 0, 1/h, 0)`) is finite forward but 0*inf = NaN backward at land [E§6]."""
    mod = pr.planted(pr.style_farray, "recip_hFacC[i, j, k])",
                     "jnp.where(hFacW[i, j, k] > 0, 1./hFacW[i, j, k], 0.))", "nan_grad")
    out = run_case(R, mod, "farray", "mom_calc_ke", 2)[0]
    assert np.all(np.isfinite(out))
    g = _grad_all_lanes(R, mod, "farray", "mom_calc_ke", 2)
    assert not np.all(np.isfinite(np.asarray(g[3].hFacW.data)))


def _divide(R, flags):
    cmd = [sys.executable, str(REPO / "dev" / "prototype" / "divide_probe.py"), str(R.rundir), "--flags", flags]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_divide_xla_cpu_vs_gfortran(R):
    """Measurement [F§2, L-CONF-2] recorded as a gate. Under the gate flags XLA:CPU's float64 divide equals
    gfortran's on every pair with normal operands and quotient, for array/array and for a literal divisor given as
    a traced 0-d operand, a traced array or a Python constant (one divide, no multiply in the optimized HLO); every
    differing value involves a subnormal (dividend or gfortran's quotient) and is XLA's +-0: XLA:CPU flushes
    subnormals to zero, gfortran does not. Negative control: with algsimp on, a scalar divisor is rewritten to
    x*(1/d) and normal values differ."""
    g = _divide(R, "gate")
    print("divide probe (gate flags):", json.dumps(g))
    assert g["numpy_vs_gfortran"]["total"] == 0
    aa = g["array_array"]
    assert aa["vs_gfortran"]["normal_only"] == 0 and aa["vs_gfortran"]["subnormal_diffs_are_zero"]
    assert aa["ops"] == {"divide": 1, "multiply": 0}
    for c, v in g["literal"].items():
        assert v["numpy"] == 0, c
        assert v["traced_scalar_normal_only"] == v["traced_array_normal_only"] == v["python_constant_normal_only"] == 0
        assert v["traced_scalar_detail"]["subnormal_diffs_are_zero"], c
        assert v["traced_scalar_ops"] == v["python_constant_ops"] == {"divide": 1, "multiply": 0}, c
    a = _divide(R, "algsimp_on")
    print("divide probe (algsimp on):", json.dumps(a))
    for c, v in a["literal"].items():
        assert v["traced_scalar_normal_only"] > 0 and v["python_constant_normal_only"] > 0, c
        assert v["traced_array_normal_only"] == 0, c
        assert v["python_constant_ops"]["divide"] == 0 and v["python_constant_ops"]["multiply"] >= 1, c


def test_style_kernels_use_no_transforms():
    """The style kernels read like physics code: no JAX transform, no optimization_barrier (the rule for
    mitjax/model and mitjax/pkg; test_banned_transforms.py), in both styles and in farray.py."""
    from mitjax.tests.test_banned_transforms import BANNED_IN_PHYSICS, jax_names
    banned = set(BANNED_IN_PHYSICS) | {"optimization_barrier"}
    for rel in ("dev/prototype/style_farray.py", "dev/prototype/style_slices.py", "mitjax/farray.py"):
        used = [(line, n) for line, n in jax_names((REPO / rel).read_text()) if n.rsplit(".", 1)[-1] in banned]
        assert used == [], (rel, used)
    planted = "import jax\ndef k(x):\n    return jax.lax.optimization_barrier(x)\n"
    assert [n for _, n in jax_names(planted) if n.rsplit(".", 1)[-1] in banned] == ["jax.lax.optimization_barrier"]
