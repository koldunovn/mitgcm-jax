"""MOM-lane gates (M1 sub-lane MOM): the leaf kernels of pkg/mom_common and pkg/mom_fluxform against the gfortran
replay harness reference/replay_mom (runs named by reference/replay_mom/CURRENT, relative to $MJX_REFERENCE):

  * bitwise: every one of the 79 harness outputs (each kernel under each branch selector the harness sets), all
    tiles and levels, ALL points incl. halos and the points a routine does not write (they keep the prior field),
    element-wise equal and finite, under the gate XLA flags (conftest.py) with the REAL parameters traced, on the
    real grids of tutorial_barotropic_gyre (Cartesian), tutorial_baroclinic_gyre and global_ocean.90x40x15
    (spherical; NONLIN_FRSURF, ALLOW_QHYD_STAGGER_TS) and tutorial_global_oce_optim/code_ad (ALLOW_AUTODIFF);
  * negative controls: planted errors, each measured to bite;
  * gradients: finite on all lanes (halos, land) w.r.t. the fields, the grid and the parameters; directional
    derivative vs central differences (h sweep); tangent vs adjoint dot test.
The replay outputs must exist: these tests fail, not skip, without them.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mitjax.tests import mom_replay as mr

RUNS = {exp: (inp, rundir) for exp, inp, rundir in mr.current_runs()}
EXPS = tuple(RUNS)
_R = {}


def replay(exp):
    if exp not in _R:
        inp, rundir = RUNS[exp]
        _R[exp] = mr.load(exp, inp, rundir)
    return _R[exp]


def run_output(R, name, mods, levels=None):
    grid, params, deepFacA = mr.grid_and_params(R)
    f = jax.jit(mr.all_levels_fn(name, R.cfg, R.size, mods, levels))
    return np.asarray(f(mr.field_arrays(R), grid, params, deepFacA))


def bitwise(exp, mods=None):
    R = replay(exp)
    mods = mods or mr.kernel_modules()
    bad, bits = {}, {}
    for name in mr.rio.OUT3:
        ne, nf, nb = mr.compare(run_output(R, name, mods), R.out[name])
        if ne or nf:
            bad[name] = (ne, nf)
        if nb:
            bits[name] = nb
    return bad, bits


@pytest.mark.parametrize("exp", [e for e in EXPS if e != "tutorial_barotropic_gyre"])   # barotropic: _quick.py
def test_mom_replay_bitwise(exp):
    bad, bits = bitwise(exp)
    print(f"{exp}: outputs differing from gfortran {bad}; bit patterns differing {bits}")
    assert bad == {} and bits == {}, (bad, bits)


def test_mom_replay_setup():
    """The harness ran what the gate assumes: the experiments' options, real grids with land and partial cells,
    the overrides reached the routines, the preprocessed routines are the oracle's (build check), finite outputs."""
    expect = {"tutorial_barotropic_gyre": dict(NONLIN_FRSURF=False, ALLOW_AUTODIFF=False),
              "tutorial_baroclinic_gyre": dict(NONLIN_FRSURF=False, ALLOW_AUTODIFF=False),
              "global_ocean.90x40x15": dict(NONLIN_FRSURF=True, ALLOW_AUTODIFF=False),
              "tutorial_global_oce_optim": dict(NONLIN_FRSURF=False, ALLOW_AUTODIFF=True)}
    assert set(EXPS) == set(expect)
    for exp in EXPS:
        R = replay(exp)
        for flag, v in expect[exp].items():
            assert R.cfg.cpp.flag(flag, "MOM_COMMON_OPTIONS.h") == v, (exp, flag)
        assert R.cfg.cpp.flag("COSINEMETH_III", "MOM_COMMON_OPTIONS.h")
        assert R.g["NONLIN_FRSURF"] == expect[exp]["NONLIN_FRSURF"]
        assert (R.g["maskW"] == 0).sum() > 100, exp                              # land
        assert np.all(R.g["recip_deepFacC"] != 1.0) and np.all(R.g["rhoFacF"] != 1.0)
        assert R.g["sideDragFactor"] == 1.75 and R.g["bottomDragQuadratic"] == 2.3e-3
        top = R.rundir
        while top.name != "runs":                                               # OUT/runs/EXP/INPUT/job<id>/rundir
            top = top.parent
        ident = (top.parent / "bin" / "oracle_identity.txt").read_text().split("\n")
        assert sum(line.startswith("identical ") for line in ident) == 29, exp
        for name, a in R.out.items():
            assert np.all(np.isfinite(a)), (exp, name)
    assert (replay("global_ocean.90x40x15").g["hFacC"] > 0).sum() > 1000


# planted errors (kernel, old text, new text, outputs checked, experiments where it must bite)
PLANTS = {
    "transposed shift (adv_vu)": ("mom_u_adv_vu", "* (uFld[i, j] + uFld[i, j-1]))", "* (uFld[i, j] + uFld[i-1, j]))",
                                  ("fMer_vu_mt0",), EXPS),
    "factor dropped (sqCosFacU)": ("mom_u_xviscflux", "           * cF4[j]\n", "\n", ("xViscU",), EXPS),
    "rim not kept (hfacz)": ("mom_calc_hfacz", "    hFacZ = hFacZ.at[1-OLx, j].set(0.)\n", "", ("hFacZ",), EXPS),
    "free-surface term dropped (adv_wu)": ("mom_u_adv_wu", "params.select_rStar == 0 and not params.rigidLid",
                                           "False", ("fVerU_c0",),
                                           # Nr = 1 (barotropic) has no interior interface; the baroclinic gyre's
                                           # flat bottom makes maskC(k)-maskC(k-1) = 0, so the term is exactly 0
                                           # there (measured: 0 points, job 27829027)
                                           ("global_ocean.90x40x15", "tutorial_global_oce_optim")),
    "wet-point scaling dropped (coriolis)": ("mom_u_coriolis", "    if sel == 1 or sel == 3:", "    if False:",
                                             ("uCf_s1",), EXPS),
    "asymmetry removed (v_sidedrag old: A4tmp for viscA4)": (
        "mom_v_sidedrag", "                          -viscA4*del2v[i, j]*sqCosFacV[j]",
        "                          -A4tmp*del2v[i, j]*sqCosFacV[j]", ("vSD_c1",), EXPS),
    "bottom override ignored (metric_nh)": ("mom_u_metric_nh", "        wVelBottomOverride = 0.\n", "        pass\n",
                                            ("uMetNH",), EXPS),
    "association (metric_nh)": ("mom_u_metric_nh", "uFld[i, j]*recip_rSphere*recip_deepFacC[k]",
                                "uFld[i, j]*(recip_rSphere*recip_deepFacC[k])", ("uMetNH",), EXPS),
    "association (botdrag pCell; partial cells)": ("mom_u_botdrag_coeff", "+ kappaRU[i, j, kLowF]*recDrC*viscFac\n"
                                                   "                                   * recip_hFacW[i, j, k])",
                                                   "+ kappaRU[i, j, kLowF]*(recDrC*viscFac\n"
                                                   "                                   * recip_hFacW[i, j, k]))",
                                                   ("cDragU_b1",), ("global_ocean.90x40x15",)),
    "ALLOW_AUTODIFF init dropped (calc_ke)": ("mom_calc_ke", 'if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):', "if False:",
                                              ("KE_0",), ("tutorial_global_oce_optim",)),
    "wrong neighbour (v_botdrag KE)": ("mom_v_botdrag_coeff", "s = KE[i, j]+KE[i, j-1]", "s = KE[i, j]+KE[i, j+1]",
                                            ("cDragV_b3",), EXPS),
}


def test_mom_negative_controls():
    mods = mr.kernel_modules()
    bitten = {}
    for label, (routine, old, new, names, exps) in PLANTS.items():
        pm = dict(mods)
        pm[routine] = mr.planted(mods[routine], old, new, name=routine + "_planted")
        for exp in exps:
            R = replay(exp)
            bitten[label, exp] = sum(mr.compare(run_output(R, n, pm), R.out[n])[0] for n in names)
    print("negative controls, points differing from gfortran:")
    for key, n in bitten.items():
        print(f"  {key[0]:55s} {key[1]:28s} {n}")
    assert all(n > 0 for n in bitten.values()), {k: n for k, n in bitten.items() if n == 0}


# ---------------------------------------------------------------------------------------------------------------
# gradients (global_ocean.90x40x15: land, partial cells, halos, NONLIN_FRSURF; levels 1, 2 and Nr: the k = 1,
# interior and k = Nr branches)

GRAD_EXP = "global_ocean.90x40x15"


def _grad_setup(name, levels):
    R = replay(GRAD_EXP)
    grid, params, deepFacA = mr.grid_and_params(R)
    fields = mr.field_arrays(R)
    f = mr.all_levels_fn(name, R.cfg, R.size, mr.kernel_modules(), levels)
    w = jnp.asarray(np.random.default_rng(len(name)).standard_normal(f(fields, grid, params, deepFacA).shape))
    return R, f, w, fields, grid, params, deepFacA


def test_mom_gradients_finite():
    """d(sum w*out)/d(fields, grid, params) finite on every lane, for every output (all branches)."""
    R = replay(GRAD_EXP)
    levels = (1, 2, R.Nr)
    bad = {}
    for name in mr.rio.OUT3:
        _, f, w, fields, grid, params, deepFacA = _grad_setup(name, levels)
        g = jax.jit(jax.grad(lambda a, b, c, d: jnp.sum(w * f(a, b, c, d)), argnums=(0, 1, 2, 3)))(
            fields, grid, params, deepFacA)
        n = sum(int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(g))
        if n:
            bad[name] = n
    assert bad == {}, bad


def test_mom_fd_and_dot():
    """Directional derivative (jvp) of sum(w*out) w.r.t. the fields vs central differences (best of an h sweep),
    and the tangent/adjoint dot test <J v, w> = <v, J^T w>, for every output."""
    R = replay(GRAD_EXP)
    levels = (1, 2, R.Nr)
    worst_fd, worst_dot = {}, {}
    for name in mr.rio.OUT3:
        _, f, w, fields, grid, params, deepFacA = _grad_setup(name, levels)
        rng = np.random.default_rng(7)
        v = {n: rng.standard_normal(a.shape) * (np.std(np.asarray(a)) + 1e-30) for n, a in fields.items()}
        # smooth points only: KEin = 0 (exact zeros, the KE > 0 IF of MOM_U/V_BOTDRAG_COEFF and SQRT at 0) is not
        # perturbed
        v["KEin"] = np.where(np.asarray(fields["KEin"]) == 0.0, 0.0, v["KEin"])
        v = {n: jnp.asarray(a) for n, a in v.items()}
        fx = jax.jit(lambda a: f(a, grid, params, deepFacA))
        J = lambda a: jnp.sum(w * fx(a))
        _, jv = jax.jvp(fx, (fields,), (v,))
        d_jvp = float(jnp.sum(w * jv))
        best = np.inf
        for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            p = jax.tree_util.tree_map(lambda a, b: a + h * b, fields, v)
            m = jax.tree_util.tree_map(lambda a, b: a - h * b, fields, v)
            fd = (float(J(p)) - float(J(m))) / (2 * h)
            best = min(best, abs(fd - d_jvp) / max(abs(d_jvp), 1e-300))
        worst_fd[name] = best
        _, vjp = jax.vjp(fx, fields)
        (ct,) = vjp(w)
        lhs = d_jvp
        rhs = float(sum(jnp.sum(ct[n] * v[n]) for n in fields))
        worst_dot[name] = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    print("FD best relative error per output:", {k: f"{x:.1e}" for k, x in worst_fd.items()})
    print("dot test relative error per output:", {k: f"{x:.1e}" for k, x in worst_dot.items()})
    assert max(worst_fd.values()) < 1e-6, {k: x for k, x in worst_fd.items() if x >= 1e-6}
    assert max(worst_dot.values()) < 1e-12, {k: x for k, x in worst_dot.items() if x >= 1e-12}
