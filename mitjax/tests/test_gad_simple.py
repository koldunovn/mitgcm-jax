"""GAD-A gates (M1 sub-lane): the simple advective-flux and diffusive-flux leaf kernels of pkg/generic_advdiff
(mitjax/pkg/generic_advdiff/gad_{c2,c4,u3,dst3,dst3fl,fluxlimit}_adv_*, gad_diff_*, gad_u3c4_impl_r,
gad_fluxlimit_impl_r) against the gfortran replay harness reference/replay_gad_a (runs named by its CURRENT, one per
experiment grid: global_ocean.90x40x15, advect_xz/input.nlfs, advect_xy, tutorial_baroclinic_gyre):

  * bitwise: every harness output, all tiles and levels, ALL points incl. halos and the points a routine does not
    write (they keep the prior), equal bit for bit (float64 bit patterns; a NaN never matches), under the gate XLA
    flags (conftest.py) with every float input traced; Fortran and JAX outputs finite;
  * the same at P=4 (tiles split over four fake CPU devices with shard_map) on the experiments with >= 4 tiles;
  * gradients: finite on all lanes (halos, land, grid fields, parameters) on the replay inputs (plateaus, exact
    zeros, land); tangent == adjoint and central FD at smooth points (h sweep); limiters differentiated as written;
  * every gate has planted errors measured to bite; unported branches raise.
The replay runs must exist: these tests fail, not skip, without them.
"""

import hashlib
import json

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec

from mitjax.tests import gad_a_replay as G

EXPERIMENTS = ("global_ocean.90x40x15", "advect_xz", "advect_xy", "tutorial_baroclinic_gyre")
# the M1 variants that execute each routine (docs/coverage/*.md), and the harness grid gating it there
EXECUTED_IN = {
    "gad_c2_adv_x": ("global_ocean.90x40x15", "tutorial_baroclinic_gyre"),   # + tutorial_global_oce_optim/input_ad
    "gad_c2_adv_y": ("global_ocean.90x40x15", "tutorial_baroclinic_gyre"),
    "gad_c2_adv_r": ("global_ocean.90x40x15", "tutorial_baroclinic_gyre"),
    "gad_c4_adv_x": ("advect_xy",), "gad_c4_adv_y": ("advect_xy",),             # input.ab3_c4
    "gad_u3_adv_x": ("advect_xz",), "gad_u3_adv_y": ("advect_xz",), "gad_u3c4_impl_r": ("advect_xz",),
    "gad_dst3fl_adv_x": ("advect_xy",), "gad_dst3fl_adv_y": ("advect_xy",),
    "gad_fluxlimit_adv_x": ("advect_xz",), "gad_fluxlimit_adv_y": ("advect_xz",),
    "gad_fluxlimit_impl_r": ("advect_xz",),
    "gad_diff_x": ("tutorial_baroclinic_gyre", "advect_xz"), "gad_diff_y": ("tutorial_baroclinic_gyre", "advect_xz"),
    "gad_diff_r": ("advect_xy", "advect_xz"),
    "gad_dst3_adv_x": (),                                                       # no M1 variant (Task 8 example)
}


@pytest.fixture(scope="module")
def RS():
    return G.current_replays()          # FileNotFoundError (a failure) when a replay run is missing


def test_replay_setup(RS):
    """Each run: the kernel cfg from mitjax/config equals the harness build's own options (GAD_OPTIONS.h.dM) and
    SIZE.h; inputs intact; overrides reached the routines; Fortran outputs finite; the inputs exercise the branches
    (zero differences with an open mask, exact +-0 transports); realistic grids (land, partial cells)."""
    assert set(RS) == set(EXPERIMENTS)
    for exp, R in RS.items():
        c = R.cfg
        assert (c.sNx, c.sNy, c.OLx, c.OLy, c.Nr) == tuple(R.size[k] for k in ("sNx", "sNy", "OLx", "OLy", "Nr"))
        dm = (R.options_dir / "GAD_OPTIONS.h.dM").read_text().split("\n")
        defined = {ln.split()[1] for ln in dm if ln.startswith("#define ")}
        assert {n: getattr(c, n) for n in G.CPP_NAMES} == {n: n in defined for n in G.CPP_NAMES}, exp
        meta = json.loads((R.rundir / "replay_in.json").read_text())
        assert meta["experiment"] == exp
        for name, sha in meta["sha256"].items():
            assert hashlib.sha256((R.rundir / name).read_bytes()).hexdigest() == sha
        g = R.grid_np
        for n in G.replay_io.OVERRIDE_NR + G.replay_io.OVERRIDE_NR1:
            assert np.all(g[n] != 1.0), (exp, n)
        assert np.all(g["cosFacU"] != 1.0) and np.all(g["cosFacV"] != 1.0)
        assert g["rkSign"] == -1.0                                  # z coordinates (INI_PARMS)
        for name, a in R.out.items():
            assert np.all(np.isfinite(a)), (exp, name)
        t, mW = R.fields["tracer"], g["maskW"]
        Rj = (t[..., 1:] - t[..., :-1]) * mW[..., 1:]
        assert np.sum((Rj == 0) & (mW[..., 1:] == 1)) > 20, exp     # zero differences with an open mask
        u = R.fields["uTrans"]
        assert np.sum(u == 0) > 0 and np.sum(np.signbit(u) & (u == 0)) > 0, exp
    h = RS["global_ocean.90x40x15"].grid_np["hFacC"]
    assert (h == 0).sum() > 10000 and ((h > 0) & (h < 1)).sum() > 1000          # land and partial cells
    assert (RS["advect_xz"].grid_np["maskC"] == 0).sum() > 0                    # bathy_slope.bin: land


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_replay_bitwise(RS, exp):
    R = RS[exp]
    bad, nonfinite, points = {}, [], 0
    for case in G.CASES:
        outs = G.run_case(R, case)
        for name, o in zip(G.case_outputs(case), outs):
            n = G.bit_diff(o, R.out[name])
            points += o.size
            if n:
                bad[name] = n
            if not np.all(np.isfinite(o)):
                nonfinite.append(name)
    print(f"{exp}: {len(G.CASES)} cases, {points} points compared, differing: {bad}")
    assert bad == {} and nonfinite == [], (exp, bad, nonfinite)


def test_every_routine_gated_where_executed(RS):
    """Each ported routine has a harness run on the grid of every M1 experiment that executes it, and its live code
    is the same in all M1 builds except gad_diff_y (ISOTROPIC_COS_SCALING in advect_xy), which both cfgs cover."""
    covered = {G.case_routine(c) for c in G.CASES}
    assert covered == set(G.ROUTINES)
    for r, exps in EXECUTED_IN.items():
        assert set(exps) <= set(RS), r
    assert RS["advect_xy"].cfg.ISOTROPIC_COS_SCALING and not RS["advect_xz"].cfg.ISOTROPIC_COS_SCALING


def _p4_case_fn(R, case, mesh):
    """The case under shard_map over 4 fake CPU devices: every [tile, ...] input split along tiles, the rest
    replicated; check_vma on."""
    f = G.case_fn(case, R.cfg)
    T = R.size["nSx"] * R.size["nSy"]
    args = R.args(G.case_fields(case))

    def spec(a):
        return PartitionSpec("t") if (np.ndim(a) >= 2 and np.shape(a)[0] == T) else PartitionSpec()
    in_specs = jax.tree_util.tree_map(spec, args)
    out_specs = tuple(PartitionSpec("t") for _ in G.case_outputs(case))
    g = jax.jit(jax.shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=True))
    return g, args


@pytest.mark.parametrize("exp", ["global_ocean.90x40x15", "tutorial_baroclinic_gyre"])
def test_replay_bitwise_p4(RS, exp):
    """Forward at P=4 (36 tiles -> 9 per device; 4 -> 1) equals gfortran bit for bit, every case."""
    R = RS[exp]
    mesh = Mesh(np.array(jax.devices()[:4]), ("t",))
    bad = {}
    for case in G.CASES:
        g, args = _p4_case_fn(R, case, mesh)
        for name, o in zip(G.case_outputs(case), g(*args)):
            n = G.bit_diff(np.asarray(o), R.out[name])
            if n:
                bad[name] = n
    assert bad == {}, bad


# planted errors, each a bug class the gate must see: label -> (experiment, case, module, old text, new text)
PLANTS = {
    "rim not zeroed": ("global_ocean.90x40x15", "c2_x", "gad_c2_adv_x",
                       "    uT = uT.at[1-OLx, j].set(0.)\n", ""),
    "k=1 branch dropped": ("global_ocean.90x40x15", "c2_r", "gad_c2_adv_r",
                           "if k == 1 or k > Nr:", "if k > Nr:"),
    "mask dropped": ("global_ocean.90x40x15", "c4_x", "gad_c4_adv_x",
                     "Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]", "Rjm = (tracer[i-1, j]-tracer[i-2, j])"),
    "boundary factor dropped": ("advect_xy", "c4_y", "gad_c4_adv_y",
                                "\n           *(1. - maskS[i, j-1, k]*maskS[i, j+1, k]))", ")"),
    "shift i-2 -> i-1": ("advect_xz", "u3_x", "gad_u3_adv_x",
                         "Rjm = (tracer[i-1, j]-tracer[i-2, j])", "Rjm = (tracer[i-1, j]-tracer[i-1, j])"),
    "y loop bounds": ("advect_xz", "u3_y", "gad_u3_adv_y",
                      "j = loop_j(1-OLy+2, sNy+OLy-1)", "j = loop_j(1-OLy+3, sNy+OLy-1)"),
    "deepFac dropped": ("advect_xy", "dst3fl_x_calcCFL", "gad_dst3fl_adv_x",
                        "*recip_dxC[i, j]*recip_deepFacC[k])", "*recip_dxC[i, j])"),
    "thetaMax branch swapped": ("global_ocean.90x40x15", "dst3fl_y_calcCFL", "gad_dst3fl_adv_y",
                                "bigP = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjm)",
                                "bigP = jnp.abs(Rj)*thetaMax < jnp.abs(Rjm)"),
    "limiter cap dropped": ("advect_xy", "dst3fl_x_givenCFL", "gad_dst3fl_adv_x",
                            'psiP = MAX(0., MIN(MIN(1., psiP, p="b"),', "psiP = MAX(0., MIN(psiP,"),
    "association": ("advect_xz", "fluxlimit_x_calcCFL", "gad_fluxlimit_adv_x",
                    "((oneRL-Cr) + uCFL*Cr)", "(oneRL-(Cr - uCFL*Cr))"),
    "upwind side swapped": ("advect_xz", "fluxlimit_y_givenCFL", "gad_fluxlimit_adv_y",
                            "Cr = jnp.where(vTrans[i, j] > zeroRL, Rjm, Rjp)",
                            "Cr = jnp.where(vTrans[i, j] > zeroRL, Rjp, Rjm)"),
    "limiter Superbee -> Min-Mod": ("advect_xz", "fluxlimit_x_givenCFL", "gad_fluxlimit_adv_x",
                                    '    Cr = Limiter(Cr, p=("b", "a", "b", "b"))  ',
                                    "    Cr = jnp.maximum(0., jnp.minimum(1., Cr))  "),
    "cosFacU dropped": ("tutorial_baroclinic_gyre", "diff_x", "gad_diff_x",
                        "\n                *cosFacU[j]))", "))"),
    "ISOTROPIC cosFacV dropped": ("advect_xy", "diff_y", "gad_diff_y",
                                  "*(tracer[i, j] - tracer[i, j-1])\n                    *cosFacV[j]))",
                                  "*(tracer[i, j] - tracer[i, j-1])))"),
    "rhoFacF dropped": ("advect_xz", "diff_r", "gad_diff_r", "*deepFac2F[k]*rhoFacF[k]", "*deepFac2F[k]"),
    "rkSign dropped": ("advect_xz", "gad_u3c4_impl_r", "gad_u3c4_impl_r",
                       "rCenter = 0.5*rTrans[i, j]*recip_rA[i, j]*rkSign", "rCenter = 0.5*rTrans[i, j]*recip_rA[i, j]"),
    "deltaTarg(k-1) -> (k)": ("advect_xz", "gad_u3c4_impl_r", "gad_u3c4_impl_r",
                              "                         - rC4km\n                          *deltaTarg[k-1]",
                              "                         - rC4km\n                          *deltaTarg[k]"),
    "kp1 -> k mask": ("global_ocean.90x40x15", "gad_fluxlimit_impl_r", "gad_fluxlimit_impl_r",
                      "*maskC[i, j, kp1]", "*maskC[i, j, k]"),
    "upwindFac floor dropped": ("advect_xz", "gad_fluxlimit_impl_r", "gad_fluxlimit_impl_r",
                                'upwindFac = upwindFac.at[i, j].set(MAX(-1., upwindFac[i, j], p="b"))', "pass"),
    "iMin-1 (halo written)": ("global_ocean.90x40x15", "gad_fluxlimit_impl_r", "gad_fluxlimit_impl_r",
                              "i = loop_i(iMin, iMax)", "i = loop_i(iMin-1, iMax)"),
    "jnp.maximum for Fortran MAX (tie -0)": ("global_ocean.90x40x15", "dst3fl_x_givenCFL", "gad_dst3fl_adv_x",
                                             '    psiP = MAX(0., MIN(MIN(1., psiP, p="b"),                       # :86-87\n'
                                             '                       thetaP*(1.-uCFL)/(uCFL+1.e-20), p="b"), p="b")',
                                             '    psiP = jnp.maximum(0., MIN(MIN(1., psiP, p="b"),\n'
                                             '                       thetaP*(1.-uCFL)/(uCFL+1.e-20), p="b"))'),
    "DST3 d1*Rjm -> d1*Rjp": ("global_ocean.90x40x15", "dst3_x_givenCFL", "gad_dst3_adv_x",
                              "(d0*Rj+d1*Rjm)", "(d0*Rj+d1*Rjp)"),
}
# A measured candidate that does NOT change any output on these inputs (recorded, not counted as a control): SIGN's
# negative-zero semantics only decide the sign of thetaP/thetaM where Rj = +-0, and there psiP*Rj = +-0 is added
# to a tracer value (test_signed_zero_plant_invisible).
INVISIBLE = {
    "SIGN without -0": ("advect_xy", "dst3fl_x_calcCFL", "gad_dst3fl_adv_x",
                        "jnp.copysign(thetaMax, Rjm*Rj)", "jnp.where(Rjm*Rj >= 0., thetaMax, -thetaMax)"),
}


def test_replay_negative_controls(RS):
    bitten = {}
    for label, (exp, case, modname, old, new) in PLANTS.items():
        mod = G.planted(modname, old, new, tag="plant")
        bitten[label] = sum(G.diffs_vs_fortran(RS[exp], case, mod).values())
    print("negative controls, points differing from gfortran:", bitten)
    assert all(n > 0 for n in bitten.values()), bitten


def test_signed_zero_plant_invisible(RS):
    """Measurement: copysign replaced by a >= 0 test (SIGN of a negative zero) changes no output of DST3FL on these
    inputs; it is not a negative control. Kept so that a change in this (e.g. inputs where it shows) is noticed."""
    exp, case, modname, old, new = INVISIBLE["SIGN without -0"]
    mod = G.planted(modname, old, new, tag="sign")
    assert sum(G.diffs_vs_fortran(RS[exp], case, mod).values()) == 0


def test_unported_branches_raise(RS):
    R = RS["advect_xz"]
    fields, grid, params, scal = R.args()
    from mitjax.pkg.generic_advdiff import gad_h
    for flag, case in (("TARGET_NEC_SX", "gad_u3c4_impl_r"), ("ALLOW_SMAG_3D_DIFFUSIVITY", "diff_x"),
                       ("ALLOW_SMAG_3D_DIFFUSIVITY", "diff_y"), ("OLD_DST3_FORMULATION", "dst3_x_calcCFL")):
        cfg = R.cfg._replace(**{flag: True}, ALLOW_AUTODIFF=True)
        with pytest.raises(NotImplementedError):
            G.case_fn(case, cfg)(fields, grid, params, scal)
    # GAD_U3C4_IMPL_R with the C4 scheme (interior levels); the DST3 arm (:139-147) is ported (GO lane,
    # cs32x15/input_ad, 42846d0) and gated there
    import mitjax.pkg.generic_advdiff.gad_u3c4_impl_r as m
    g = G.make_common(R.cfg, grid, G.GRID_DECL)
    p = G.make_common(R.cfg, params, G.PARAMS_DECL)
    b = G.bounds(R.cfg)
    from mitjax.farray import FArray
    A3 = FArray(fields["aPrior"], "a5d", i=b["i"], j=b["j"], k=(1, R.cfg.Nr))
    rT = FArray(fields["rTrans"][:, 0], "rTrans", i=b["i"], j=b["j"])
    dT = FArray(scal["deltaTarg"], "deltaTarg", k=(1, R.cfg.Nr), tiled=False)
    for scheme, k in ((gad_h.ENUM_CENTERED_4TH, 3),):
        with pytest.raises(NotImplementedError):
            m.gad_u3c4_impl_r(k, 1, R.cfg.sNx, 1, R.cfg.sNy, scheme, dT, rT, g.recip_hFacC, A3, A3, A3, A3, A3,
                              cfg=R.cfg, grid=g, params=p)


# ------------------------------------------------------------------------------------------------------------------
# gradients

def _loss(R, case, levels, w):
    f = G.case_fn(case, R.cfg, levels=levels)

    def loss(fields, params, scal, grid):
        return sum(jnp.sum(wi * o) for wi, o in zip(w, f(fields, grid, params, scal)))
    return loss


def _weights(R, case, levels, seed=3, exclude=None):
    rng = np.random.default_rng(seed)
    ks = slice(None) if levels is None else [k - 1 for k in levels]
    w = []
    for n in G.case_outputs(case):
        ref = np.abs(R.out[n][:, ks])
        wn = rng.standard_normal(ref.shape) / (ref + np.median(ref) + 1e-300)
        if exclude is not None:
            wn = np.where(exclude, 0.0, wn)
        w.append(jnp.asarray(wn))
    return w


def _levels(R, case):
    return None if case in G.IMPL else (2,) if R.cfg.Nr > 1 else (1,)


def smooth_fields(R, seed=11):
    """Replay inputs moved away from the schemes' kinks and jumps (FD at smooth points): tracer = 10 + X(i) + Y(j) +
    Z(k) + 0.05 N(0,1) per tile, with X, Y, Z random walks whose steps are +-U(0.5,4), so every tracer difference
    in i, j and k is bounded away from 0 (|step| >= 0.5 against noise differences of std 0.07) while its sign varies (slope ratios of both signs reach
    every limiter branch); transports and velocities without exact zeros. Why bounded differences: the limiter's
    slope ratio Cr = Rjp/Rj jumps from -CrMax to +CrMax when Rj changes sign, and GAD_FLUXLIMIT_IMPL_R puts the
    limiter into the matrix without a factor Rj, so its matrix is discontinuous at Rj = 0 (measured: FD errors of
    O(1-25) at h >= 1e-5 on the global_ocean grid with an unstructured random tracer)."""
    rng = np.random.default_rng(seed)
    f = dict(R.fields)
    shp = f["tracer"].shape
    T, Nr, Ny, Nx = shp

    def walk(n):
        return np.cumsum(rng.choice([-1.0, 1.0], (T, n)) * rng.uniform(0.5, 4.0, (T, n)), axis=1)
    f["tracer"] = (10.0 + walk(Nx)[:, None, None, :] + walk(Ny)[:, None, :, None] + walk(Nr)[:, :, None, None]
                   + 0.05 * rng.standard_normal(shp))
    for n in ("uTrans", "vTrans", "rTrans", "uVel", "vVel", "uCFL", "vCFL"):
        a = R.fields[n]
        f[n] = np.where(a == 0, np.std(a) * rng.uniform(0.5, 1.5, shp) * rng.choice([-1.0, 1.0], shp), a)
    return f


def unphysical_cfl(R, k, uVel=None):
    """Points of level k where GAD_DST3_ADV_X's computed CFL number |uVel*deltaTloc*recip_dxC*recip_deepFacC| is
    unphysical (> 1e3): the halo rows beyond the N/S edges of a spherical-polar grid, where INI_GRID's recip_dxC
    reaches 3.7e10 1/m (the interior CFL numbers of these synthetic inputs are O(1), at most ~10)."""
    g = R.grid_np
    u = R.fields["uVel"] if uVel is None else uVel
    return np.abs(u[:, k - 1] * R.scal["deltaTloc"] * g["recip_dxC"] * g["recip_deepFacC"][k - 1]) > 1.0e3


def _unpack(x, grid):
    fl, pa, sc = x
    return fl, grid, pa, sc


def _fd_check(R, case, mod=None, physical_only=True, fields_np=None, with_tangent=False):
    """(|tangent - adjoint| / |adjoint|, {h: |FD - adjoint| / |adjoint|}[, {h: |FD - tangent| / |tangent|}]) of a
    weighted sum of the case's outputs, along a random direction (scaled by each input's magnitude) of all inputs the
    case reads (fields, PARAMS.h floats, deltaTloc, diffKh, deltaTarg). The central difference is taken output
    point by output point before the weighted sum (sum w*(f(x+hd) - f(x-hd)) / 2h), so the round-off of a sum of
    ~1e4-1e5 terms does not set the FD floor."""
    levels = _levels(R, case)
    excl = None
    if physical_only and case == "dst3_x_calcCFL":
        excl = unphysical_cfl(R, levels[0], None if fields_np is None else fields_np["uVel"])[:, None]
    w = _weights(R, case, levels, exclude=excl)
    f = G.case_fn(case, R.cfg, mod, levels)
    fields, grid, params, scal = R.args(G.case_fields(case))
    if fields_np is not None:
        fields = {n: jnp.asarray(fields_np[n]) for n in fields}

    def loss(x):
        fl, pa, sc = x
        return sum(jnp.sum(wi * o) for wi, o in zip(w, f(fl, grid, pa, sc)))

    def wdiff(xp, xm):
        return sum(jnp.sum(wi * (op - om)) for wi, op, om in zip(w, f(*_unpack(xp, grid)), f(*_unpack(xm, grid))))
    x = (fields, params, scal)
    rng = np.random.default_rng(11)
    d = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(np.shape(a))) * (jnp.abs(a) + 1e-30), x)
    dj = jax.jit(wdiff)
    lin = jax.jit(lambda x, d: jax.jvp(loss, (x,), (d,))[1])(x, d)
    g = jax.jit(jax.grad(loss))(x)
    gd = sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(d)))
    errs, errs_t = {}, {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        xp = jax.tree_util.tree_map(lambda a, b: a + h * b, x, d)
        xm = jax.tree_util.tree_map(lambda a, b: a - h * b, x, d)
        fd = dj(xp, xm) / (2 * h)
        errs[h] = float(abs(fd - gd) / abs(gd))
        errs_t[h] = float(abs(fd - lin) / abs(lin))
    dot = float(abs(lin - gd) / abs(gd))
    return (dot, errs, errs_t) if with_tangent else (dot, errs)


@pytest.mark.parametrize("exp", ["global_ocean.90x40x15", "advect_xz"])
def test_fd_gradients(RS, exp):
    """At smooth points (smooth_fields: tracer differences bounded away from 0, no exact zeros), for every case: tangent == adjoint to 1e-12
    and the best central-FD relative error over the h sweep < 1e-8. Limiters and the thetaMax/CrMax caps are
    differentiated as written. DST3 with calcCFL: output points with an unphysical CFL number (> 1e3, the polar halo
    rows of global_ocean's spherical grid) get weight 0 (Task 8 measurement; test_dst3_reverse_mode_cancellation).
    The inputs reach every limiter branch (test_limiters_active_in_fd_inputs)."""
    R = RS[exp]
    sf = smooth_fields(R)
    worst = {}
    for case in G.CASES:
        dot, errs = _fd_check(R, case, fields_np=sf)
        worst[case] = (dot, min(errs.values()))
    print(exp, "tangent-vs-adjoint and best FD relative errors:", worst)
    assert all(dot < 1e-12 and fd < 1e-8 for dot, fd in worst.values()), worst


def test_limiters_active_in_fd_inputs(RS):
    """The FD inputs reach every branch of the Superbee limiter (Cr < 0, 0..1/2, 1/2..1, 1..2, > 2) on the
    global_ocean.90x40x15 grid: in i on the cells GAD_FLUXLIMIT_ADV_X computes (level 2) and in k on the interfaces
    GAD_FLUXLIMIT_IMPL_R computes (k = 2..Nr), so the FD gate tests the limiter where it is active."""
    def bins(Cr):
        return [int(np.sum(Cr < 0)), int(np.sum((Cr > 0) & (Cr < .5))), int(np.sum((Cr > .5) & (Cr < 1))),
                int(np.sum((Cr > 1) & (Cr < 2))), int(np.sum(Cr > 2))]
    R = RS["global_ocean.90x40x15"]
    t = smooth_fields(R)["tracer"][:, 1]
    m = R.grid_np["maskW"][:, 1]
    u = R.fields["uTrans"][:, 1]
    Rjp = (t[..., 3:] - t[..., 2:-1]) * m[..., 3:]
    Rj = (t[..., 2:-1] - t[..., 1:-2]) * m[..., 2:-1]
    Rjm = (t[..., 1:-2] - t[..., :-3]) * m[..., 1:-2]
    Cr = np.where(u[..., 2:-1] > 0, Rjm, Rjp)
    ok = Rj != 0
    bx = bins(Cr[ok] / Rj[ok])
    R = RS["global_ocean.90x40x15"]
    t, mC, r = smooth_fields(R)["tracer"], R.grid_np["maskC"], R.fields["rTrans"]
    bz = [0] * 5
    for k in range(2, R.cfg.Nr + 1):
        km2, km1, kp1 = max(1, k - 2), max(1, k - 1), min(R.cfg.Nr, k + 1)
        Rjp = (t[:, kp1 - 1] - t[:, k - 1]) * mC[:, kp1 - 1]
        Rj = t[:, k - 1] - t[:, km1 - 1]
        Rjm = (t[:, km1 - 1] - t[:, km2 - 1]) * mC[:, km2 - 1]
        Cr = np.where(r[:, k - 1] < 0, Rjm, Rjp) / Rj
        bz = [a + b for a, b in zip(bz, bins(Cr))]
    assert all(b > 0 for b in bx + bz), (bx, bz)


def test_fd_negative_control(RS):
    """A derivative cut that keeps the forward (stop_gradient on the limiter value) is seen by the FD gate."""
    R = RS["advect_xz"]
    mod = G.planted("gad_fluxlimit_adv_x", '    Cr = Limiter(Cr, p=("b", "a", "b", "b"))  ',
                    '    Cr = jax.lax.stop_gradient(Limiter(Cr, p=("b", "a", "b", "b")))  ', tag="fd_cut")
    mod.jax = jax
    assert sum(G.diffs_vs_fortran(R, "fluxlimit_x_calcCFL", mod).values()) == 0        # forward unchanged
    _, errs = _fd_check(R, "fluxlimit_x_calcCFL", mod, fields_np=smooth_fields(R))
    assert min(errs.values()) > 1e-3, errs


def _grad_all_lanes(R, case, mod=None):
    levels = _levels(R, case)
    f = G.case_fn(case, R.cfg, mod, levels)
    w = _weights(R, case, levels)
    fields, grid, params, scal = R.args(G.case_fields(case))

    def loss(fields, grid, params, scal):
        return sum(jnp.sum(wi * o) for wi, o in zip(w, f(fields, grid, params, scal)))
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(fields, grid, params, scal)


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_gradients_finite_all_lanes(RS, exp):
    """Gradient of a weighted sum of all outputs with respect to every input (fields, GRID.h fields incl. masks,
    PARAMS.h floats, deltaTloc, diffKh, deltaTarg) is finite on every lane, on the replay inputs (plateaus, exact
    +-0, land, halos, the polar halo metrics of global_ocean)."""
    R = RS[exp]
    bad = []
    for case in G.CASES:
        g = _grad_all_lanes(R, case)
        for path, leaf in jax.tree_util.tree_leaves_with_path(g):
            if not np.all(np.isfinite(np.asarray(leaf))):
                bad.append((case, jax.tree_util.keystr(path)))
    assert bad == [], bad


def test_gradients_finite_negative_control(RS):
    """A forward-only guard (the ELSE-branch division unguarded inside the `where`) is bitwise the same forward but
    0*inf = NaN backward at Rj = 0 lanes [E§6]."""
    R = RS["advect_xy"]
    mod = G.planted("gad_dst3fl_adv_x", "safe_div(Rjm, Rj, ~bigP)", "Rjm/Rj", tag="nan_grad")
    assert sum(G.diffs_vs_fortran(R, "dst3fl_x_calcCFL", mod).values()) == 0
    g = _grad_all_lanes(R, "dst3fl_x_calcCFL", mod)
    assert not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(g))


def test_dst3_reverse_mode_cancellation(RS):
    """The measurement behind the physical-CFL restriction of the DST3 FD gate (Task 8, now on the package kernel,
    job 27828868 scanned all 15 levels): on the replay inputs of global_ocean (level 12; also 2, 3, 6, 8, 9, 14) with
    all points weighted, tangent and adjoint differ (8e-3 here), FD sides with the tangent, and the points with an
    unphysical CFL number (> 1e3) are halo rows beyond the domain's N/S edges of the spherical grid. There
    uT = 0.5*(u+|u|)*A + 0.5*(u-|u|)*B with |A| ~ 1e29: reverse mode accumulates +-0.5*A*ct through both uses of |u|
    and the 0.5*B*ct terms vanish against them. With smooth_fields (the FD gate's inputs) the scan found no level
    where it happens (tangent == adjoint to 4e-16)."""
    R = RS["global_ocean.90x40x15"]
    k = 12
    bad = unphysical_cfl(R, k)
    jj = np.nonzero(bad)[1]
    assert 0 < bad.sum() < 0.05 * bad.size              # measured: 259 of 9216 at k = 12
    Ny = R.cfg.sNy + 2 * R.cfg.OLy
    assert set(jj.tolist()) <= set(range(R.cfg.OLy)) | set(range(Ny - R.cfg.OLy, Ny))
    lv = _levels
    try:
        globals()["_levels"] = lambda R, case: (k,)
        dot, errs, errs_t = _fd_check(R, "dst3_x_calcCFL", physical_only=False, with_tangent=True)
    finally:
        globals()["_levels"] = lv
    print(f"all points weighted, level {k}: tangent-vs-adjoint {dot:.3e}, FD-vs-adjoint {errs}, "
          f"FD-vs-tangent {errs_t}")
    # FD on the replay inputs (exact zeros: kinks of ABS at the evaluation point) has a higher floor than on
    # smooth_fields; the measurement is the separation: adjoint off by > 1e-3, tangent within 1e-6
    assert dot > 1e-3 and min(errs.values()) > 1e-3 and min(errs_t.values()) < 1e-6
