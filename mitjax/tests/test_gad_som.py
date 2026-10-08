"""M1 sub-lane GAD-B gates: the SOM (second-order moment, Prather 1986) advection path of pkg/generic_advdiff against
the gfortran replay harness (reference/replay_gad_b, runs named by reference/replay_gad_b/CURRENT):

  * bitwise: every output of GAD_SOM_ADVECT (schemes 80 and 81: gTracer and the 9 moments), GAD_SOM_ADV_X/_Y
    (limiter 0 and 1: the 11 updated fields and the flux), GAD_SOM_LIM_R (limiter 1), GAD_SOM_EXCHANGES and
    GAD_EXCH_SOM, ALL points incl. halos and the points a routine does not write, element-equal to gfortran and
    finite, on the advect_xy grid (2 tiles of 20x10, OL 3, Nr 1) and the advect_xz grid (2 tiles of 10x1, OL 4:
    sNy < OLy, multi-wrap halo; Nr 20 with a sloping bottom: land and partial cells), under the gate XLA flags;
  * negative controls: each planted error is measured to bite (points differing > 0) in the named experiment;
  * the run's switches the harness reports equal the plain oracle run's STDOUT printout and the data files;
  * gradients of GAD_SOM_ADVECT: finite on every lane (halos, land, the multi-wrap halo of advect_xz) w.r.t. the
    fields, the grid and PARAMS arrays and deltaTLev; tangent vs adjoint dot test; FD h-sweep at smooth points (the
    limiter differentiated as written: JAX's own derivative at a tie, Nikolay 2026-10-01).
The replay output must exist: these tests fail, not skip, without it.
"""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import gad_som_replay as h

import mitjax.pkg.generic_advdiff.gad_exch_som as m_exch
import mitjax.pkg.generic_advdiff.gad_som_adv_r as m_adv_r
import mitjax.pkg.generic_advdiff.gad_som_adv_x as m_adv_x
import mitjax.pkg.generic_advdiff.gad_som_adv_y as m_adv_y
import mitjax.pkg.generic_advdiff.gad_som_advect as m_advect
import mitjax.pkg.generic_advdiff.gad_som_exchanges as m_exchanges
import mitjax.pkg.generic_advdiff.gad_som_lim_r as m_lim_r

SCHEME = {"advect_xy": 80, "advect_xz": 81}          # the experiment's own SOM scheme (T in advect_xy, S in advect_xz)


@pytest.fixture(scope="module")
def runs():
    return h.current_runs()          # FileNotFoundError (a failure) when a replay run is missing


def test_replay_setup(runs):
    """Inputs intact; the harness used the run's switches (= the plain oracle's STDOUT and the data files); the grid
    has land and partial cells in advect_xz; the factor overrides reached the routines; the synthetic inputs exercise
    both limiter branches; every Fortran output is finite."""
    from mitjax.io import stdout as so
    from mitjax.params_io import RunParams
    from mitjax import paths
    for exp, R in runs.items():
        meta = json.loads((R.rundir / "replay_in.json").read_text())
        for name, sha in meta["sha256"].items():
            assert hashlib.sha256((R.rundir / name).read_bytes()).hexdigest() == sha
        lines = so.read_stdout(paths.REFERENCE_RUNS / exp / "input" / "job27826873-plain" / "rundir" / "output.txt")
        printed = so.parameter_dict(so.parameters(lines))
        for name in ("rigidLid", "nonlinFreeSurf", "select_rStar", "uniformFreeSurfLev", "tempSOM_Advection",
                     "saltSOM_Advection", "tempAdvScheme", "saltAdvScheme", "tempVertAdvScheme", "saltVertAdvScheme"):
            v = printed[name].values()[0]
            want = {"T": 1, "F": 0}[v] if v in "TF" else int(v)
            assert R.header[name] == want, (exp, name, R.header[name], v)
        rp = RunParams(R.experiment.run)
        assert rp.get("data", "PARM01", "rigidLid") is False and R.cfg.rigidLid is False
        assert (rp.get("data", "PARM01", "tempAdvScheme"), rp.get("data", "PARM01", "saltAdvScheme")) == (
            R.header["tempAdvScheme"], R.header["saltAdvScheme"])
        # useCubedSphereExchange: not set in eedata, default .FALSE. (eesupp/src/eeset_parms.F:106)
        assert not rp.has("eedata", "EEPARMS", "useCubedSphereExchange") and R.header["useCubedSphereExchange"] == 0
        assert not R.experiment.cfg.use_flag("useDiagnostics")
        for n in h.rio.FACTORS + h.rio.FACTORS_F:
            assert np.all(R.grid_np[n] != 1.0), n
        for name, a in R.out.items():
            assert np.all(np.isfinite(a)), (exp, name)
        # limiter branches: sm_o <= 0 and clipping both occur in the leaf inputs; clipping changed values
        assert (R.inp["sm_o"] <= 0).sum() > 10 and (R.inp["sm_o"] == 0).sum() > 5
        assert (R.out["advx_l1_x"] != R.out["advx_l0_x"]).sum() > 100
        assert (R.out["limr_z"] != R.inp["sm_z"]).sum() > 10
    hc = runs["advect_xz"].grid_np["hFacC"]
    assert (hc == 0).sum() > 100 and ((hc > 0) & (hc < 1)).sum() > 10, "advect_xz: land and partial cells"
    assert runs["advect_xz"].cfg.sNy < runs["advect_xz"].cfg.OLy, "advect_xz: multi-wrap halo"


@pytest.mark.parametrize("exp", h.EXPERIMENTS)
def test_replay_bitwise(runs, exp):
    res = h.run_all(runs[exp])
    assert len(res) == 106
    bad = {k: v for k, v in res.items() if v}
    n = runs[exp].out["gTracer_80"].size
    assert bad == {}, f"{exp}: points differing from gfortran (of {n} each): {bad}"


# planted errors: label -> (experiment, group, schemes, module key, old text, new text)
PLANTS = {
    "association (adv_x part 3)": (
        "advect_xy", "adv_x", None, "adv_x",
        "sm_xx = sm_xx.at[i, j].set(alf1*alf1*sm_xx[i, j] + alfp*alfp*fp_xx[i, j]",
        "sm_xx = sm_xx.at[i, j].set(alf1*(alf1*sm_xx[i, j]) + alfp*alfp*fp_xx[i, j]"),
    "flux outside its loop range not kept (adv_x)": (
        "advect_xy", "adv_x", None, "adv_x",
        "    uT = uT.at[i, j].set((fp_o[i, j] - fn_o[i, j])*recip_dT)",
        "    uT = uT.local('uT').at[i, j].set((fp_o[i, j] - fn_o[i, j])*recip_dT)"),
    "limiter dropped (adv_y)": (
        "advect_xy", "adv_y", None, "adv_y",
        "    if limiter == 1:                                                # :164-185",
        "    if False:"),
    "moment factor alf1q -> alf1 (adv_y part 2)": (
        "advect_xy", "adv_y", None, "adv_y",
        "sm_yz = sm_yz.at[i, j].set(alf1q*sm_yz[i, j])",
        "sm_yz = sm_yz.at[i, j].set(alf1*sm_yz[i, j])"),
    "limiter constant 1.5 -> 1.25 (lim_r)": (
        "advect_xy", "lim_r", None, "lim_r",
        "s1max = slpmax*1.5", "s1max = slpmax*1.25"),
    "kUp/kDw swap (adv_r part 2)": (
        "advect_xy", "advect", (80,), "adv_r",
        "alf1 = 1.0 - aln[i, j, kDw] - alp[i, j, kUp]", "alf1 = 1.0 - aln[i, j, kUp] - alp[i, j, kUp]"),
    "factor rhoFacC dropped (smVol)": (
        "advect_xy", "advect", (80,), "advect",
        "                                        * rhoFacC[k])", "                                        )"),
    "vertical loop order reversed": (
        "advect_xz", "advect", (81,), "advect",
        "                    down=True, peel=(1, 1))", "                    down=False, peel=(1, 1))"),
    "empty-box fill dropped (land)": (
        "advect_xz", "advect", (81,), "advect",
        "                                        + (1.0 - maskC[iA, jA, k]))", "                                        )"),
    "Sz exchange dropped (multi-wrap halo)": (
        "advect_xz", "exch", None, "exch",
        "    s[2] = ex.EXCH_3D_RL(s[2])", "    s[2] = s[2]"),
    "SOM flag of som_T wrong (exchanges)": (
        "advect_xy", "exch", None, "exchanges",
        "        if cfg.tempSOM_Advection:", "        if cfg.saltSOM_Advection:"),
}
MODULES = {"adv_x": m_adv_x, "adv_y": m_adv_y, "adv_r": m_adv_r, "lim_r": m_lim_r, "advect": m_advect,
           "exch": m_exch, "exchanges": m_exchanges}


def _mods_for(label, key, mod):
    """The {group: module} replacement for run_all: a planted leaf is called directly (adv_x, adv_y, lim_r) or
    through a driver clone (adv_r -> GAD_SOM_ADVECT, exch -> GAD_SOM_EXCHANGES and GAD_EXCH_SOM)."""
    if key == "adv_r":
        return {"advect": h.clone(m_advect, label, gad_som_adv_r=mod.gad_som_adv_r)}
    if key == "exch":
        ex2 = h.clone(m_exchanges, label, gad_exch_som=mod.gad_exch_som)
        return {"exch": _ExchPair(ex2, mod)}
    if key == "exchanges":
        return {"exch": _ExchPair(mod, m_exch)}
    return {key: mod}


class _ExchPair:
    def __init__(self, exchanges_mod, exch_mod):
        self.gad_som_exchanges = exchanges_mod.gad_som_exchanges
        self.gad_exch_som = exch_mod.gad_exch_som


def test_replay_negative_controls(runs):
    bitten = {}
    for label, (exp, group, schemes, key, old, new) in PLANTS.items():
        mod = h.planted(MODULES[key], old, new, name="".join(c if c.isalnum() else "_" for c in label))
        res = h.run_all(runs[exp], mods=_mods_for(label, key, mod), only=(group,), schemes=schemes or (80, 81))
        bitten[label] = sum(res.values())
    print("negative controls, points differing from gfortran:", bitten)
    assert all(n > 0 for n in bitten.values()), bitten


# ---------------------------------------------------------------------------------------------------- gradients
def _loss(R, scheme, seed=3):
    """Scalar loss of GAD_SOM_ADVECT: every output point (gTracer and the 9 moments, all lanes) weighted by
    r / (|Fortran output| + median), as a function of (fields, g, p, dTlev)."""
    f = h.advect_fn(R.cfg, scheme)
    rng = np.random.default_rng(seed)
    w = []
    for n in h.advect_out_names(scheme):
        ref = np.abs(R.out[n])
        w.append(jnp.asarray(rng.standard_normal(ref.shape) / (ref + np.median(ref) + 1e-300)))

    def loss(fields, g, p, dTlev):
        return sum(jnp.sum(wi * o) for wi, o in zip(w, f(fields, g, p, dTlev)))
    return loss


@pytest.fixture(scope="module")
def adjoint(runs):
    """adjoint(exp) -> (loss, inputs, gradient of the loss w.r.t. every input), computed once per experiment on first
    use and shared by the finiteness test and the dot test (the 20-level advect_xz program takes minutes to compile
    under AD)."""
    cache = {}

    def get(exp):
        if exp not in cache:
            R = runs[exp]
            loss = _loss(R, SCHEME[exp])
            fields = h.advect_inputs(R)
            g, p, dTlev = R.grid_args()
            grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(fields, g, p, dTlev)
            cache[exp] = (loss, (fields, g, p, dTlev), grads)
        return cache[exp]
    return get


@pytest.mark.parametrize("exp", h.EXPERIMENTS)
def test_gradient_finite_all_lanes(runs, adjoint, exp):
    """d(loss)/d(every input) is finite on every lane: the fields (incl. halos, land, the multi-wrap halo of
    advect_xz), the GRID.h and PARAMS.h arrays and deltaTLev; the halo lanes are reached (nonzero gradient)."""
    R = runs[exp]
    _, _, grads = adjoint(exp)
    bad = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        a = np.asarray(leaf)
        if not np.all(np.isfinite(a)):
            bad[jax.tree_util.keystr(path)] = int((~np.isfinite(a)).sum())
    assert bad == {}, f"{exp}: non-finite gradient lanes {bad}"
    cfg = R.cfg
    gt = np.asarray(grads[0]["tracer"])
    halo = np.ones(gt.shape[-2:], bool)
    halo[cfg.OLy:cfg.OLy + cfg.sNy, cfg.OLx:cfg.OLx + cfg.sNx] = False
    assert np.count_nonzero(gt[..., halo]) > 0, f"{exp}: no gradient reaches the halo of tracer"


@pytest.mark.parametrize("exp", h.EXPERIMENTS)
def test_tangent_adjoint_fd(runs, adjoint, exp):
    """Tangent vs adjoint (dot test) and central FD along a random direction of the fields, zero on land cells (no
    tie of the limiter or of MAX(0,uLoc) is crossed there), measured as an h sweep."""
    R = runs[exp]
    loss, (fields, g, p, dTlev), grads = adjoint(exp)
    wet = np.asarray(R.grid_np["maskC"]) > 0
    rng = np.random.default_rng(11)
    d = {n: jnp.asarray(np.where(wet, rng.standard_normal(a.shape), 0.0) * (np.abs(np.asarray(a)) + 1e-30))
         for n, a in fields.items()}
    lf = jax.jit(lambda x: loss(x, g, p, dTlev))
    tl = float(jax.jit(lambda x, d: jax.jvp(lf, (x,), (d,))[1])(fields, d))
    ad = float(sum(jnp.sum(grads[0][n] * d[n]) for n in fields))
    dot = abs(tl - ad) / abs(ad)
    errs = {}
    for hh in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        xp = {n: fields[n] + hh * d[n] for n in fields}
        xm = {n: fields[n] - hh * d[n] for n in fields}
        errs[hh] = abs((float(lf(xp)) - float(lf(xm))) / (2 * hh) - ad) / abs(ad)
    print(f"{exp}: dot test {dot:.2e}; FD vs adjoint {errs}")
    assert dot < 1e-13, dot
    assert min(errs.values()) < 1e-8, errs


def test_gradient_guard_deltaT_zero(runs):
    """GAD_SOM_ADV_X/_Y and GAD_SOM_ADV_R set recip_dT = 0 when deltaTloc <= 0 (gad_som_adv_x.F:133-134): with
    deltaTloc = 0 the gradient of the leaf's outputs w.r.t. its inputs and deltaTloc is finite (the division is guarded
    before it is made, safe_div). Negative control: the forward-`where` form of the same statement gives NaN lanes."""
    R = runs["advect_xy"]
    leaf_in = {n: jnp.asarray(R.inp[n]) for n in ("uTransX", "vTransY", "fluxPrior")
               + tuple(f"sm_{m}" for m in h.SM11)}
    mask = jnp.asarray(R.grid_np["maskInC"])
    zero_dt = jnp.zeros((R.cfg.Nr,))

    def n_nonfinite(mod):
        f = h.leaf_fn(R.cfg, "x", 1, mod)
        rng = np.random.default_rng(5)
        w = [jnp.asarray(rng.standard_normal(R.inp["sm_v"].shape)) for _ in range(12)]
        loss = lambda x, dt: sum(jnp.sum(wi * o) for wi, o in zip(w, f(x, mask, dt)))
        g = jax.jit(jax.grad(loss, argnums=(0, 1)))(leaf_in, zero_dt)
        return sum(int((~np.isfinite(np.asarray(a))).sum()) for a in jax.tree_util.tree_leaves(g))

    planted = h.planted(m_adv_x, "    recip_dT = safe_div(1.0, deltaTloc, deltaTloc > zeroRL, 0.)",
                        "    recip_dT = jnp.where(deltaTloc > zeroRL, 1.0/deltaTloc, 0.)", "forward_where")
    bad_real, bad_planted = n_nonfinite(m_adv_x), n_nonfinite(planted)
    print(f"deltaTloc = 0: non-finite gradient lanes {bad_real} (guarded), {bad_planted} (forward where)")
    assert bad_real == 0 and bad_planted > 0, (bad_real, bad_planted)
