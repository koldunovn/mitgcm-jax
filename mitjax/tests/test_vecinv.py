"""VECINV-lane gates (M2 sub-lane VECINV, plan Task 23): pkg/mom_vecinv (MOM_VECINV and the mom_vi_* routines) and
the pkg/mom_common routines it calls, against the gfortran replay harness reference/replay_vecinv (runs named by
reference/replay_vecinv/CURRENT, relative to $MJX_REFERENCE):

  * bitwise: every one of the 91 harness outputs (each leaf routine under each branch selector the harness sets, and
    MOM_VECINV itself with the run's own PARAMS.h (m0) and with the vorticity selectors flipped (m1)), all tiles and
    levels, ALL points incl. halos and the points a routine does not write (they keep the prior field), element-wise
    equal, equal bit patterns and finite, under the gate XLA flags (conftest.py) with every REAL parameter traced
    (jit arguments), on the real grids of global_ocean.90x40x15/code_ad (input_ad: Leith, viscA4GridMax,
    useAreaViscLength, quadratic bottom drag, useCDscheme; ALLOW_AUTODIFF), global_ocean.cs32x15/code (cube, 12
    tiles: the 24 tile corners of the 8 cube corners; NONLIN_FRSURF) and solid-body.cs-32x32x1 (cube, 6 face tiles,
    every tile has all 4 corners), FILL_CS_CORNER_TR_RL (lane B's eesupp port) included;
  * negative controls: planted errors, each measured to bite;
  * gradients: finite on all lanes (halos, land, corners) w.r.t. fields, grid, parameters; directional derivative vs
    central differences at smooth points (h sweep); tangent vs adjoint dot test;
  * in-dump gate: MOM_VECINV inside the model, against stage D00c_mom_vecinv of lane A's dumps-on runs of the same
    three variants (every dumped iteration, every level), from the dumped state and geometry (vecinv_dump.py);
  * MOM_INIT_FIXED's viscosity length scales against the harness's MOM_VISC.h.
The replay outputs must exist: these tests fail, not skip, without them.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mitjax.tests import vecinv_replay as vr


def _keyed(runs):
    """{key: (experiment, input dir, run dir)}: the key is the experiment for its first run in CURRENT, else
    "experiment/input dir" (a second variant of the same build, e.g. global_ocean.cs32x15/input.viscA4)."""
    out = {}
    for exp, inp, rundir in runs:
        out[exp if exp not in out else f"{exp}/{inp}"] = (exp, inp, rundir)
    return out


RUNS = _keyed(vr.current_runs())
EXPS = tuple(RUNS)
BASE = tuple(k for k in EXPS if "/" not in k)                    # one run per build
_R = {}


def replay(key):
    if key not in _R:
        _R[key] = vr.load(*RUNS[key])
    return _R[key]


def _args(R):
    grid, params, visc, deepFacA = vr.grid_params_visc(R)
    return vr.field_arrays(R), grid, params, visc, deepFacA, vr.case_arrays(R)


def leaf_outputs():
    return tuple(n for n in vr.rio.OUT3 if not n.endswith(("_m0", "_m1")))


def run_output(R, name, mods, levels=None):
    f = jax.jit(vr.all_levels_fn(name, R, mods, vr.w2_view(R), levels))
    return np.asarray(f(*_args(R)))


def run_composition(R, m, mods, levels=None):
    f = jax.jit(vr.composition_fn(R, mods, vr.w2_view(R), m, levels))
    return {n: np.asarray(a) for n, a in f(*_args(R)).items()}


def bitwise(exp, mods=None):
    """({output: (points !=, non-finite points)}, {output: bit-pattern differences})."""
    R = replay(exp)
    mods = mods or vr.kernel_modules()
    bad, bits = {}, {}

    def check(name, ours):
        ne, nf, nb = vr.compare(ours, R.out[name])
        if ne or nf:
            bad[name] = (ne, nf)
        if nb:
            bits[name] = nb

    for name in leaf_outputs():
        check(name, run_output(R, name, mods))
    for m in (0, 1):
        outs = run_composition(R, m, mods)
        for n in vr.COMPOSITION:
            check(f"{n}_m{m}", outs[n])
    return bad, bits


@pytest.mark.parametrize("exp", EXPS)
def test_vecinv_replay_bitwise(exp):
    bad, bits = bitwise(exp)
    print(f"{exp}: outputs differing from gfortran {bad}; bit patterns differing {bits}")
    assert bad == {} and bits == {}, (bad, bits)


EXPECT = {"global_ocean.cs32x15/input.viscA4": dict(cube=True, ALLOW_AUTODIFF=False, NONLIN_FRSURF=True,
                                                    useVariableVisc=False, useBiharmonicVisc=True,
                                                    useHarmonicVisc=True, useCDscheme=False),
          "global_ocean.90x40x15": dict(cube=False, ALLOW_AUTODIFF=True, NONLIN_FRSURF=True, useVariableVisc=True,
                                         useBiharmonicVisc=True, useHarmonicVisc=False, useCDscheme=True),
          "global_ocean.cs32x15": dict(cube=True, ALLOW_AUTODIFF=False, NONLIN_FRSURF=True, useVariableVisc=False,
                                        useBiharmonicVisc=False, useHarmonicVisc=True, useCDscheme=False),
          "solid-body.cs-32x32x1": dict(cube=True, ALLOW_AUTODIFF=False, NONLIN_FRSURF=False, useVariableVisc=False,
                                         useBiharmonicVisc=False, useHarmonicVisc=False, useCDscheme=False)}


def test_vecinv_replay_setup():
    """The harness ran what the gate assumes: the experiments' options, real grids with land and partial cells, the
    cube corners on every face, the overrides reached the routines, the preprocessed routines are the oracle's (build
    check), finite outputs, every branch of the selectors taken where the data reach it."""
    # the three base runs always; global_ocean.cs32x15/input.viscA4 once its harness run is in CURRENT
    assert set(BASE) == {k for k in EXPECT if "/" not in k} and set(EXPS) <= set(EXPECT), EXPS
    for exp in EXPS:
        R = replay(exp)
        want = EXPECT[exp]
        assert R.cube == want["cube"], exp
        for flag in ("ALLOW_AUTODIFF", "NONLIN_FRSURF"):
            assert R.cfg.cpp.flag(flag, "MOM_COMMON_OPTIONS.h") == want[flag], (exp, flag)
        assert not R.cfg.cpp.flag("MOM_VI_ORIGINAL_VISCA4", "MOM_VECINV_OPTIONS.h")
        for n in ("useVariableVisc", "useBiharmonicVisc", "useHarmonicVisc", "useCDscheme"):
            assert R.g[n] == want[n], (exp, n)
        assert np.all(R.g["recip_deepFacC"] != 1.0) and np.all(R.g["rhoFacF"] != 1.0)
        top = R.rundir
        while top.name != "runs":
            top = top.parent
        ident = (top.parent / "bin" / "oracle_identity.txt").read_text().split("\n")
        assert sum(line.startswith("identical ") for line in ident) == 30, exp
        for name, a in R.out.items():
            assert np.all(np.isfinite(a)), (exp, name)
        if R.cube:
            t = vr.w2_view(R)
            corners = {c: int(np.sum((np.asarray(getattr(t, "exch2_is" + a)) == 1)
                                     & (np.asarray(getattr(t, "exch2_is" + b)) == 1)))
                       for c, a, b in (("SW", "Wedge", "Sedge"), ("SE", "Eedge", "Sedge"),
                                       ("NE", "Eedge", "Nedge"), ("NW", "Wedge", "Nedge"))}
            assert corners == {"SW": 6, "SE": 6, "NE": 6, "NW": 6}, (exp, corners)
            assert sorted(set(np.asarray(t.exch2_myFace).tolist())) == [1, 2, 3, 4, 5, 6]
    R = replay("global_ocean.90x40x15")
    assert (R.g["maskW"] == 0).sum() > 1000 and ((R.g["hFacC"] > 0) & (R.g["hFacC"] < 1)).sum() > 100
    assert R.g["viscC4leith"] == 1.5 and R.g["viscC4leithD"] == 1.5 and R.g["viscA4GridMax"] == 0.5
    assert R.g["selectBotDragQuadr"] >= 0 and R.g["bottomDragQuadratic"] == 0.0021


def test_mom_init_fixed_bitwise():
    """MOM_INIT_FIXED's length scales (L2/L3/L4rdt at divergence and vorticity points) equal the harness's MOM_VISC.h
    (written by the build's own INITIALISE_FIXED), all points incl. halos: useAreaViscLength = T on
    global_ocean.90x40x15/input_ad (L2 = rA, rAz) and F on the cube runs (2/(recip_dxF**2+recip_dyF**2) arms)."""
    from mitjax.pkg.mom_common.mom_init_fixed import mom_init_fixed
    seen = set()
    for exp in EXPS:
        R = replay(exp)
        grid, params, _, _ = vr.grid_params_visc(R)
        out = mom_init_fixed(cfg=R.cfg, grid=grid, params=params)
        seen.add(bool(R.g["useAreaViscLength"]))
        for n in ("L2_D", "L2_Z", "L3_D", "L3_Z", "L4rdt_D", "L4rdt_Z"):
            assert vr.compare(out[n].data, R.g[n]) == (0, 0, 0), (exp, n)
    assert seen == {True, False}


def test_vecinv_params_from_the_run():
    """The values the kernels read come from the run's own files and equal the harness's record of the run's
    PARAMS.h (replay_grid.bin, written by the build itself): ini_parms_vecinv (the VECINV arm of ini_parms) on every
    run; ini_parms_dyn as a whole where it is ported (every SCALARS/FLAGS value it holds); lane B's W2 tile view equals
    the build's W2_EXCH2_TOPOLOGY.h (replay_w2.bin) on the cube runs."""
    from mitjax.model.src.ini_parms import ini_parms_vecinv
    checked, sources = {}, {}
    for key in EXPS:
        R = replay(key)
        vs, vt = ini_parms_vecinv(R.exp)
        n = 0
        for name, v in list(vs.items()) + [(k, float(x)) for k, x in vt.items()]:
            if name.endswith("_ne_0"):
                assert v == (R.g[name[:-5]] != 0.), (key, name)
            else:
                assert v == R.g[name], (key, name, v, R.g[name])
            n += 1
        params, src = vr.run_params(R)
        sources[key] = src.split(" (")[0]
        if src == "ini_parms_dyn":
            have = {**params.static_items(), **params.traced_items()}
            for name in vr.rio.FLAGS + vr.rio.SCALARS:
                if name in have and name not in ("pi",):
                    v = have[name]
                    assert (float(v) if name in vr.rio.SCALARS else v) == R.g[name], (key, name, v, R.g[name])
                    n += 1
        if R.cube:
            ref = vr.rio.read_w2(R.rundir)
            t = vr.w2_view(R)
            for name in vr.rio.W2_TOPO[1:]:
                assert np.array_equal(np.asarray(getattr(t, name)), ref[name]), (key, name)
        checked[key] = n
    print("values checked per run:", checked, "parameter source:", sources)
    assert sources["global_ocean.90x40x15"] == "ini_parms_dyn"


# planted errors (kernel, old text, new text, outputs checked, experiments where it must bite)
CUBES = ("global_ocean.cs32x15", "solid-body.cs-32x32x1")
PLANTS = {
    "cube corner block skipped (relvort3)": ("mom_calc_relvort3", "    if params.useCubedSphereExchange:",
                                             "    if False:", ("vort3",), CUBES),
    "sign error in the face-2 SE arm (relvort3)": (
        "mom_calc_relvort3", "               -vFld[I-1, J]*dyC[I-1, J])\n              + uFld[I, J-1]*dxC[I, J-1]",
        "               +vFld[I-1, J]*dyC[I-1, J])\n              + uFld[I, J-1]*dxC[I, J-1]", ("vort3",), CUBES),
    "NE corner association (relvort3, odd faces)": (
        "mom_calc_relvort3", "                (-uFld[I, J]*dxC[I, J]\n                 -vFld[I-1, J]*dyC[I-1, J])\n"
                             "                + uFld[I, J-1]*dxC[I, J-1]",
        "                -uFld[I, J]*dxC[I, J]\n                 + (-vFld[I-1, J]*dyC[I-1, J]\n"
        "                + uFld[I, J-1]*dxC[I, J-1])", ("vort3",), ("global_ocean.cs32x15",)),
    # a SW/SE swap is a no-op on these tilings (a tile holding one southern corner holds both); SW/NE is not on cs32
    "corner fill from the wrong corner (del2uv: SW and NE flags swapped)": (
        "mom_vi_del2uv", "corners = jnp.stack([isW & isS, isE & isS, isW & isN, isE & isN], axis=1)",
        "corners = jnp.stack([isE & isN, isE & isS, isW & isN, isW & isS], axis=1)", ("del2u", "hDiv_del2"),
        ("global_ocean.cs32x15",)),
    "corner fill direction 2 for 1 (del2uv, del2u)": (
        "mom_vi_del2uv", "        hDiv = _fill(1, hDiv, cfg, params, w2)", "        hDiv = _fill(2, hDiv, cfg, params, w2)",
        ("del2u",), CUBES),
    "corner fill direction 2 for 1 (calc_visc divDx)": (
        "mom_calc_visc", "            hDiv = _fill(1, hDiv, cfg, p, w2)", "            hDiv = _fill(2, hDiv, cfg, p, w2)",
        ("viscA4_D_v1",), CUBES),
    "Leith neighbour shifted (calc_visc, non-full)": (
        "mom_calc_visc", 'grdVrt = MAX(jnp.abs(vrtDx[i, j+1]), jnp.abs(vrtDx[i, j]), p="a")   # :456',
        'grdVrt = MAX(jnp.abs(vrtDx[i+1, j]), jnp.abs(vrtDx[i, j]), p="a")   # :456', ("viscA4_D_v1",),
        ("global_ocean.90x40x15",)),
    # the Reynolds-number limit at vorticity points: with the first case values it won the MAX at a few points only
    # (79 changed points); with the bounds set from the measured term quantiles (replay_io.py VPAR_VALUES) 5518 of the
    # 9408 written points of levels 1 and 8 change (90x40x15)
    "wrong KE neighbour in keZpt (calc_visc Reynolds limit)": (
        "mom_calc_visc", "    keZpt = 0.25*((KE[i, j]+KE[i-1, j-1])", "    keZpt = 0.25*((KE[i, j]+KE[i+1, j-1])",
        ("viscAh_Z_v3",), ("global_ocean.90x40x15",)),
    "cosFacU dropped (hdissip, variable)": ("mom_vi_hdissip", "                       cosFacU[j]*(Dij-Dmj)*recip_dxC[i, j]\n"
                                            "             -recip_hFacW[i, j, k]*(Zip-Zij)*recip_dyG[i, j])",
                                            "                       (Dij-Dmj)*recip_dxC[i, j]\n"
                                            "             -recip_hFacW[i, j, k]*(Zip-Zij)*recip_dyG[i, j])",
                                            ("uDissip_h0",), BASE),
    "association (vi_u_coriolis, scheme 3)": (
        "mom_vi_u_coriolis",
        "        vort3mj = ((r_hFacZ[i, j]*omega3[i, j]\n                    +(r_hFacZ[i, j+1]*omega3[i, j+1]\n"
        "                      +r_hFacZ[i-1, j]*omega3[i-1, j]\n                      ))*oneThird",
        "        vort3mj = (((r_hFacZ[i, j]*omega3[i, j]\n                    +r_hFacZ[i, j+1]*omega3[i, j+1])\n"
        "                      +r_hFacZ[i-1, j]*omega3[i-1, j]\n                      )*oneThird",
        ("uVort_c3",), BASE),
    # not on solid-body: Nr = 1, so both w averages are multiplied by mask_Km1 = mask_Kp1 = 0 (measured: 0 points)
    "rAdvAreaWeight ignored (vi_v_vertshear)": ("mom_vi_v_vertshear", "        rAdvAreaWeight = False",
                                                "        rAdvAreaWeight = True", ("vShear_s3",),
                                                ("global_ocean.90x40x15", "global_ocean.cs32x15")),
    "angleSinC -> angleCosC (v_coriolis_nh)": ("mom_v_coriolis_nh", "fCoriCos, angleSinC, rA, deepFac2F = "
                                               "grid.fCoriCos, grid.angleSinC,", "fCoriCos, angleSinC, rA, deepFac2F = "
                                               "grid.fCoriCos, grid.angleCosC,", ("vCfNH_s1",), BASE),
    "no-slip vorticity BC dropped (vecinv sideMaskFac)": ("mom_vecinv", "        sideMaskFac = p.sideDragFactor",
                                                          "        sideMaskFac = 0.", ("guDiss_m0",),
                                                          ("global_ocean.90x40x15",)),
    "land vort3 not masked (vecinv)": ("mom_vecinv", "    vort3 = vort3.at[iA, jA].set(jnp.where(land, 0., vort3[iA, jA]))",
                                       "", ("gU_m0",), ("global_ocean.90x40x15", "global_ocean.cs32x15")),
}

def _bite(R, routine, pm, names):
    if any(n.endswith(("_m0", "_m1")) for n in names):
        m = int(names[0][-1])
        outs = run_composition(R, m, pm)
        return sum(vr.compare(outs[n[:-3]], R.out[n])[0] for n in names)
    return sum(vr.compare(run_output(R, n, pm), R.out[n])[0] for n in names)


def test_vecinv_negative_controls():
    """Each plant changes one kernel module (MOM_VECINV imports its callees by name, so a planted callee is seen only
    by the leaf outputs; the composition controls plant mom_vecinv itself)."""
    mods = vr.kernel_modules()
    bitten = {}
    for label, (routine, old, new, names, exps) in PLANTS.items():
        pm = dict(mods)
        pm[routine] = vr.planted(mods[routine], old, new, name=routine + "_planted")
        for exp in exps:
            bitten[label, exp] = _bite(replay(exp), routine, pm, names)
    print("negative controls, points differing from gfortran:")
    for key, n in bitten.items():
        print(f"  {key[0]:55s} {key[1]:28s} {n}")
    assert all(n > 0 for n in bitten.values()), {k: n for k, n in bitten.items() if n == 0}


# ---------------------------------------------------------------------------------------------------------------
# gradients: levels 1, 2, Nr (the k = 1, interior and k = Nr branches); global_ocean.90x40x15 (land, partial cells,
# Leith, bottom drag) and the cube outputs of global_ocean.cs32x15 that run (corners)

def _grad_cases():
    out = []
    for exp in ("global_ocean.90x40x15", "global_ocean.cs32x15"):
        out += [(exp, name) for name in leaf_outputs()] + [(exp, "m0"), (exp, "m1")]
    return out


def _fn(R, name, levels):
    mods = vr.kernel_modules()
    w2 = vr.w2_view(R)
    if name in ("m0", "m1"):
        g = vr.composition_fn(R, mods, w2, int(name[1]), levels)
        return lambda *a: jnp.stack([g(*a)[n] for n in vr.COMPOSITION])
    return vr.all_levels_fn(name, R, mods, w2, levels)


def test_vecinv_gradients_finite():
    """d(sum w*out)/d(fields, grid, params, MOM_VISC.h, deepFacA, case parameters) finite on every lane, for every
    output that runs (all branches)."""
    bad = {}
    for exp, name in _grad_cases():
        R = replay(exp)
        f = _fn(R, name, (1, 2, R.Nr))
        args = _args(R)
        w = jnp.asarray(np.random.default_rng(len(name)).standard_normal(jax.eval_shape(f, *args).shape))
        g = jax.jit(jax.grad(lambda *a: jnp.sum(w * f(*a)), argnums=tuple(range(6))))(*args)
        n = sum(int(np.count_nonzero(~np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(g))
        if n:
            bad[exp, name] = n
    assert bad == {}, bad


def test_vecinv_fd_and_dot():
    """Directional derivative (jvp) of sum(w*out) w.r.t. the fields vs central differences (best of an h sweep), and
    the tangent/adjoint dot test <J v, w> = <v, J^T w>, for every output that runs. Smooth points only: the exact
    zeros of KE (the KE > 0 branches of MOM_CALC_VISC and the bottom drag) are not perturbed."""
    worst_fd, worst_dot = {}, {}
    for exp, name in _grad_cases():
        R = replay(exp)
        f = _fn(R, name, (1, 2, R.Nr))
        fields, grid, params, visc, deepFacA, cp = _args(R)
        rng = np.random.default_rng(7)
        v = {n: rng.standard_normal(a.shape) * (np.std(np.asarray(a)) + 1e-30) for n, a in fields.items()}
        v["KE"] = np.where(np.asarray(fields["KE"]) == 0.0, 0.0, v["KE"])
        v = {n: jnp.asarray(a) for n, a in v.items()}
        fx = jax.jit(lambda a: f(a, grid, params, visc, deepFacA, cp))
        w = jnp.asarray(np.random.default_rng(len(name)).standard_normal(jax.eval_shape(fx, fields).shape))
        J = lambda a: jnp.sum(w * fx(a))
        _, jv = jax.jvp(fx, (fields,), (v,))
        d_jvp = float(jnp.sum(w * jv))
        best = np.inf
        for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            pp = jax.tree_util.tree_map(lambda a, b: a + h * b, fields, v)
            mm = jax.tree_util.tree_map(lambda a, b: a - h * b, fields, v)
            fd = (float(J(pp)) - float(J(mm))) / (2 * h)
            best = min(best, abs(fd - d_jvp) / max(abs(d_jvp), 1e-300))
        worst_fd[exp, name] = best
        _, vjp = jax.vjp(fx, fields)
        (ct,) = vjp(w)
        rhs = float(sum(jnp.sum(ct[n] * v[n]) for n in fields))
        worst_dot[exp, name] = abs(d_jvp - rhs) / max(abs(d_jvp), 1e-300)
    print("FD best relative error per output:", {k: f"{x:.1e}" for k, x in worst_fd.items()})
    print("dot test relative error per output:", {k: f"{x:.1e}" for k, x in worst_dot.items()})
    assert max(worst_fd.values()) < 1e-6, {k: x for k, x in worst_fd.items() if x >= 1e-6}
    assert max(worst_dot.values()) < 1e-12, {k: x for k, x in worst_dot.items() if x >= 1e-12}


def test_vecinv_sqrt_guard_bites():
    """Negative control of the gradient guards: MOM_CALC_VISC's Reynolds-number SQRT (:394) written as a plain
    jnp.sqrt under the `where` gives NaN gradients on the KE = 0 lanes (0 * inf in the backward pass), measured on
    case v2 of global_ocean.90x40x15; the guarded kernel is finite there (test_vecinv_gradients_finite)."""
    R = replay("global_ocean.90x40x15")
    mods = vr.kernel_modules()
    pm = dict(mods)
    pm["mom_calc_visc"] = vr.planted(mods["mom_calc_visc"],
                                     "    Uscl = jnp.where(cU, safe_sqrt(argU, cU & (argU > 0.))*viscAhRe_max, 0.)",
                                     "    Uscl = jnp.where(cU, jnp.sqrt(argU)*viscAhRe_max, 0.)", name="visc_unguarded")
    f = vr.all_levels_fn("viscAh_D_v2", R, pm, None, (1,))
    args = _args(R)
    w = jnp.ones(jax.eval_shape(f, *args).shape)
    g = jax.jit(jax.grad(lambda a: jnp.sum(w * f(a, *args[1:]))))(args[0])
    n = int(np.count_nonzero(~np.isfinite(np.asarray(g["KE"]))))
    print(f"unguarded SQRT: {n} non-finite d/dKE lanes")
    assert n > 0


# ---------------------------------------------------------------------------------------------------------------
# in-dump gate: MOM_VECINV inside the model (stage D00c_mom_vecinv of lane A's jdon runs; mitjax/tests/vecinv_dump.py)

def _dump_gate(exp, mods):
    from mitjax.tests import vecinv_dump as vd
    R = replay(exp)
    ds = vd.dumpset(R.experiment, R.input_dir)
    f = jax.jit(vd.dynamics_fn(R, mods, vr.w2_view(R)))
    res, peak = {}, 0.
    for it in ds.iterations():
        if vd.STAGE not in ds.stages(it):
            continue
        grid, params, visc = vd.grid_params_visc(R, ds, it)
        outs = f(vd.state_inputs(R, ds, it), grid, params, visc)
        ref = vd.reference(R, ds, it)
        for n, a in ref.items():
            res[it, n] = vr.compare(outs[n], a)
            peak = max(peak, float(np.abs(a).max()))
    return res, peak


@pytest.mark.parametrize("exp", EXPS)
def test_vecinv_dump_D00c(exp):
    """gU, gV, guDissip, gvDissip after MOM_VECINV at every level of every dumped iteration, all points incl. halos,
    bitwise against the oracle's own dump (inputs: the dumped state and geometry of the same iteration)."""
    res, peak = _dump_gate(exp, vr.kernel_modules())
    print(f"{exp}: D00c (points !=, non-finite, bits) per (iteration, field): {res}")
    assert len(res) >= 8 and peak > 0.
    assert all(v == (0, 0, 0) for v in res.values()), {k: v for k, v in res.items() if v != (0, 0, 0)}


def test_vecinv_dump_negative_controls():
    """The in-dump gate bites on planted errors in MOM_VECINV (measured)."""
    mods = vr.kernel_modules()
    bitten = {}
    for label, exp in (("no-slip vorticity BC dropped (vecinv sideMaskFac)", "global_ocean.90x40x15"),
                       ("land vort3 not masked (vecinv)", "global_ocean.cs32x15")):
        routine, old, new, _, _ = PLANTS[label]
        pm = dict(mods)
        pm[routine] = vr.planted(mods[routine], old, new, name=routine + "_planted")
        res, _ = _dump_gate(exp, pm)
        bitten[label, exp] = sum(v[0] for v in res.values())
    print("in-dump negative controls, points differing:", bitten)
    assert all(n > 0 for n in bitten.values()), bitten


# ---------------------------------------------------------------------------------------------------------------
# P=4: the same kernels under jit(shard_map(check_vma=True)) on 4 (fake CPU) devices, tiles split in blocks

def test_vecinv_p4():
    """MOM_VECINV (m0, m1), MOM_CALC_RELVORT3, MOM_VI_DEL2UV and MOM_CALC_VISC (corner fills) at levels 1, 2, Nr on 4
    devices (90x40x15: one tile each; cs32x15:
    three tiles each, the W2 tile view sharded with the fields) equal the P=1 run and the Fortran bit for bit. The
    kernels exchange nothing: FILL_CS_CORNER_TR_RL works within a tile, its corner flags sharded with the tiles)."""
    from jax.sharding import PartitionSpec as P
    from mitjax.eesupp.shard import tile_mesh
    from mitjax.farray import FArray
    mesh = tile_mesh(4)
    ax = mesh.axis_names[0]
    mods = vr.kernel_modules()
    res = {}
    for exp in ("global_ocean.90x40x15", "global_ocean.cs32x15"):
        R = replay(exp)
        assert (R.size["nSx"] * R.size["nSy"]) % 4 == 0
        levels = (1, 2, R.Nr)
        w2 = vr.w2_view(R)
        args = _args(R) + (w2,)

        def spec(tree):
            return jax.tree_util.tree_map(
                lambda x: P(ax) if (isinstance(x, FArray) and x.tiled) else P(), tree,
                is_leaf=lambda x: isinstance(x, FArray))
        specs = (jax.tree_util.tree_map(lambda _: P(ax), args[0]),) + tuple(spec(a) for a in args[1:6]) + (
            None if w2 is None else jax.tree_util.tree_map(lambda _: P(ax), w2),)
        for m in (0, 1):
            def body(fields, grid, params, visc, deepFacA, cp, w2l, m=m):
                return vr.composition_fn(R, mods, w2l, m, levels)(fields, grid, params, visc, deepFacA, cp)
            f4 = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P(ax), check_vma=True))
            out4 = {n: np.asarray(a) for n, a in f4(*args).items()}
            out1 = run_composition(R, m, mods, levels)
            for n in vr.COMPOSITION:
                ref = R.out[f"{n}_m{m}"][:, [k - 1 for k in levels]]
                res[exp, f"{n}_m{m}"] = (vr.compare(out4[n], out1[n]), vr.compare(out4[n], ref))
        if R.cube:
            for name in ("vort3", "del2u", "viscA4_D_v1"):          # cube corners; FILL_CS_CORNER_TR_RL
                def bodyv(fields, grid, params, visc, deepFacA, cp, w2l, name=name):
                    return vr.all_levels_fn(name, R, mods, w2l, levels)(fields, grid, params, visc, deepFacA, cp)
                f4 = jax.jit(jax.shard_map(bodyv, mesh=mesh, in_specs=specs, out_specs=P(ax), check_vma=True))
                ref = R.out[name][:, [k - 1 for k in levels]]
                o4 = f4(*args)
                res[exp, name] = (vr.compare(o4, run_output(R, name, mods, levels)), vr.compare(o4, ref))
    print("P=4 vs P=1, P=4 vs Fortran:", res)
    assert all(v == ((0, 0, 0), (0, 0, 0)) for v in res.values()), {k: v for k, v in res.items()
                                                                    if v != ((0, 0, 0), (0, 0, 0))}
