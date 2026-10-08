"""pkg/seaice gates of 1D_ocean_ice_column/input (M4 step 3, lane M4COL, session 2).

1. SEAICE_READPARMS (+ SEAICE_INIT_FIXED's parameters) against the oracle's STDOUT SEAICE_SUMMARY: every printed
   scalar the port carries, compared as the Fortran prints it (E23.15 / integer / T,F / quoted string).
2. SEAICE_INIT_FIXED + SEAICE_INIT_VARIA bitwise vs I00b_seaice_begin of iteration 0 (every dumped SEAICE.h /
   SEAICE_GRID.h field, all points) and sIceLoad vs I01b_dynsolver of iteration 0.
3. Teacher-forced, iterations 0, 1, 2, every dumped field, every point (halos included), bitwise: DYNSOLVER (forcing
   part + OSTRES) I00b -> I01b; SEAICE_REG_RIDGE I01b -> I03; SEAICE_GROWTH (+ SOLVE4TEMP, BUDGET_OCEAN) I03 -> I04;
   SEAICE_MODEL from the state before it (S02_load_fields) through every stage I00b, I01b, I03, I04, P13, both
   teacher-forced and as our own sea-ice chain from SEAICE_INIT_VARIA.
4. Negative controls, each measured to bite (counts asserted > 0): a planted summary value, SEAICE_initialHEFF,
   SEAICE_drag, celsius2K, SEAICE_mcPheeTaper, IMAX_TICE, the KGEO level.
5. Gradients of SEAICE_MODEL w.r.t. the sea-ice state, the EXF fields, the FFIELDS.h fluxes and the ocean state:
   finite on every lane (halos included) at the zero-ice state of iteration 0, at the thin-ice state of iteration 1
   and at a calm lane (zero wind: the AAA <= SEAICE_EPS_SQ arm of DYNSOLVER, UG = SEAICE_EPS), tangent vs adjoint dot
   test, FD h-sweep at iteration 1 (smooth: ice present, no switch within reach).
Under the gate XLA flags (conftest.py); REAL parameters are traced jit arguments.
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import seaice_gate as G


def _bites(res):
    """Number of (iteration, field) pairs that differ."""
    return sum(1 for _, r in res for v in G.failures(r).values())


# ----------------------------------------------------------------------------------------------- 1. parameters
def _summary(s):
    lines = (s.rundir / "output.txt").read_text(errors="replace").splitlines()
    lines = [ln.split(")", 1)[1] if ln.startswith("(PID") else ln for ln in lines]
    a = next(k for k, ln in enumerate(lines) if "Seaice configuration (SEAICE_PARM01) >>> START" in ln)
    b = next(k for k, ln in enumerate(lines) if "Seaice configuration (SEAICE_PARM01) >>> END" in ln)
    out = {}
    for k in range(a, b):
        m = re.match(r"\s*(\w+)\s*=\s*/\*", lines[k])
        if m:
            tok = lines[k+1].strip()
            if "/*" in tok or "@" in tok:          # arrays (SEAICE_PDF): not compared here
                continue
            out[m.group(1)] = tok
    return out


def _printed(v):
    if isinstance(v, (bool, np.bool_)):
        return "T" if v else "F"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return f"'{v.strip()}'"
    return f"{float(v):.15E}"


def _summary_mismatches(s, sp):
    printed = _summary(s)
    # the REAL array parameters (SEAICE_PDF since lane M4LAB session 3, the SItr* arrays of ALLOW_SITRACER) are traced
    # vectors in sp.r; arrays are not compared here (_summary skips them)
    have = {**dict(sp.s), **{k: float(v) for k, v in sp.r.items() if np.ndim(v) == 0}}
    have["SEAICE_rhoAir"] = float(sp.r["SEAICE_rhoAir"])
    checked, bad = 0, {}
    for name, tok in printed.items():
        if name not in have:
            continue
        checked += 1
        if _printed(have[name]) != tok:
            bad[name] = (_printed(have[name]), tok)
    return checked, bad


def test_seaice_readparms_vs_stdout_summary():
    s = G.setup()
    checked, bad = _summary_mismatches(s, s.sp)
    assert checked >= 50, checked
    assert not bad, bad
    assert s.sp.SEAICE_PDF[0] == 1.0 and all(x == 0.0 for x in s.sp.SEAICE_PDF[1:])   # "1.0E+00, 6 @ 0.0E+00"
    assert s.sp.facOpenGrow == 1.0 and s.sp.facOpenMelt == 0.0


def test_seaice_summary_negative_control():
    """A planted SEAICE_mcPheeTaper (0.9 instead of data.seaice's 0.92) and a planted IMAX_TICE are caught."""
    s = G.setup()
    _, bad = _summary_mismatches(s, s.sp.replace(SEAICE_mcPheeTaper=np.float64(0.9), IMAX_TICE=10))
    assert set(bad) == {"SEAICE_mcPheeTaper", "IMAX_TICE"}, bad


# ----------------------------------------------------------------------------------------------- 2. initial state
def _init_cmp(s):
    ref = G.stage_values(s, 0, "I00b_seaice_begin", G.SF_DUMPED)
    res = G.compare(G.ours_values(s.sf0, s.ff0, {}, list(ref)), ref)
    ref2 = G.stage_values(s, 0, "I01b_dynsolver", ("sIceLoad",))
    res.update(G.compare(G.ours_values({}, s.ff0, {}, ["sIceLoad"]), ref2))
    return res


def test_seaice_init_bitwise_vs_I00b():
    s = G.setup()
    res = _init_cmp(s)
    assert len(res) == 25, sorted(res)
    assert not G.failures(res), G.failures(res)


def test_seaice_init_negative_control():
    """SEAICE_initialHEFF = 0.5 (the commented-out data.seaice line) instead of 0.0: HEFF, AREA, HSALT and sIceLoad
    differ (measured; HSNOW comes from HsnowFile, PRESS0/ZETA are not in I00b's dump)."""
    s = G.setup(planted=(("SEAICE_initialHEFF", np.float64(0.5)),))
    bad = G.failures(_init_cmp(s))
    assert set(bad) == {"HEFF", "AREA", "HSALT", "sIceLoad"}, sorted(bad)


# ----------------------------------------------------------------------------------------------- 3. routines
def test_dynsolver_ostres_bitwise_vs_I01b():
    s = G.setup()
    res = G.run_dynsolver(s)
    for it, r in res:
        assert len(r) == 43, (it, len(r))
        assert not G.failures(r), (it, G.failures(r))


def test_dynsolver_negative_control():
    """SEAICE_drag doubled: DAIRN and FORCEX/Y (+ FORCEX0/Y0) differ once there is ice (iterations 1, 2)."""
    s = G.setup()
    res = G.run_dynsolver(s, sp=s.sp.replace(SEAICE_drag=np.float64(0.002)))
    assert not G.failures(res[0][1])                         # no ice at iteration 0: AREA = 0
    for it, r in res[1:]:
        assert {"DAIRN", "FORCEX", "FORCEY", "FORCEX0", "FORCEY0"} <= set(G.failures(r)), (it, G.failures(r))


def test_reg_ridge_bitwise_vs_I03():
    s = G.setup()
    for it, r in G.run_stage(s, G.reg_ridge_fn(s), "I01b_dynsolver", "I03_reg_ridge"):
        assert len(r) == 12, (it, len(r))
        assert not G.failures(r), (it, G.failures(r))


def test_reg_ridge_negative_control():
    """celsius2K = 273.15 instead of 273.16: TICES of the ice-free column (iteration 0, :226) differs."""
    s = G.setup()
    op = dataclasses.replace(s.op, celsius2K=np.float64(273.15))
    res = G.run_stage(s, G.reg_ridge_fn(s), "I01b_dynsolver", "I03_reg_ridge", op=op)
    assert "TICES" in G.failures(res[0][1]), G.failures(res[0][1])


def test_growth_bitwise_vs_I04():
    s = G.setup()
    for it, r in G.run_stage(s, G.growth_fn(s), "I03_reg_ridge", "I04_growth"):
        assert len(r) == 20, (it, len(r))
        assert not G.failures(r), (it, G.failures(r))


@pytest.mark.parametrize("name,value,its,must", [
    ("SEAICE_mcPheeTaper", np.float64(0.9), (1, 2), {"HEFF", "Qnet"}),
    # the surface-temperature Newton iteration reaches its fixed point bitwise within 4 passes in this column
    # (measured: IMAX_TICE 4, 5, 7 give the oracle's bits at iterations 0-2); 3 passes differ at iteration 1
    ("IMAX_TICE", 3, (1,), {"TICES", "HEFF"}),
    ("SEAICE_saltFrac", np.float64(0.31), (0, 1, 2), {"HSALT", "saltFlux"}),
], ids=["mcPheeTaper", "IMAX_TICE", "saltFrac"])
def test_growth_negative_controls(name, value, its, must):
    s = G.setup()
    res = G.run_stage(s, G.growth_fn(s), "I03_reg_ridge", "I04_growth", sp=s.sp.replace(**{name: value}))
    for it, r in res:
        if it in its:
            assert must <= set(G.failures(r)), (name, it, G.failures(r))


@pytest.mark.parametrize("teacher", [True, False], ids=["teacher_forced", "own_chain"])
def test_seaice_model_bitwise_all_stages(teacher):
    s = G.setup()
    want = {"I00b_seaice_begin": 51, "I01b_dynsolver": 43, "I03_reg_ridge": 12, "I04_growth": 20,
            "P13_seaice_model": 35}
    for it, cmp in G.run_model(s, teacher=teacher):
        for st, r in cmp.items():
            assert len(r) == want[st], (it, st, len(r))
            assert not G.failures(r), (it, st, G.failures(r))


def test_seaice_model_negative_control_kgeo():
    """KGEO level 2 instead of 1 (SEAICE_BICE_STRESS): GWATX/GWATY differ from I01b on."""
    s = G.setup()
    res = G.run_model(s, its=(1,), kgeo=2)
    assert {"GWATX", "GWATY"} <= set(G.failures(res[0][1]["I01b_dynsolver"])), G.failures(res[0][1]["I01b_dynsolver"])


# ----------------------------------------------------------------------------------------------- 5. gradients
X_SF = ("HEFF", "AREA", "HSNOW", "HSALT", "TICES")
X_EXF = ("atemp", "aqh", "lwdown", "swdown", "wspeed", "uwind", "vwind", "precip", "evap")
X_FF = ("Qnet", "Qsw", "EmPmR", "fu", "fv")
X_ST = ("theta", "salt", "uVel", "vVel", "etaN")
Y_SF = ("HEFF", "AREA", "HSNOW", "HSALT", "TICES", "FORCEX", "FORCEY", "PRESS0", "DAIRN", "WINDX", "WINDY", "AMASS",
        "GWATX", "GWATY")
Y_FF = ("Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad", "fu", "fv")


def _gfun(s, it, calm=False):
    sfd, ffd, exfd, std = G.stage_inputs(s, it, "S02_load_fields")
    if calm:
        for n in ("uwind", "vwind", "wspeed"):
            exfd[n] = np.zeros_like(exfd[n])
    fn = G.model_fn(s)
    myTime = jnp.float64(s.tp.startTime + s.tp.deltaTClock*(it - s.tp.nIter0))
    x0 = {**{n: jnp.asarray(sfd[n]) for n in X_SF}, **{n: jnp.asarray(exfd[n]) for n in X_EXF},
          **{n: jnp.asarray(ffd[n]) for n in X_FF}, **{n: jnp.asarray(std[n]) for n in X_ST}}

    def g(x):
        sf2 = {**sfd, **{n: x[n] for n in X_SF}}
        ex2 = {**exfd, **{n: x[n] for n in X_EXF}}
        ff2 = {**ffd, **{n: x[n] for n in X_FF}}
        st2 = {**std, **{n: x[n] for n in X_ST}}
        osf, off, _ = fn(s.sp, s.op, sf2, ff2, ex2, st2, myTime)["P13_seaice_model"]
        return {**{n: osf[n] for n in Y_SF}, **{n: off[n] for n in Y_FF}}
    return g, x0


def _rand_like(tree, seed):
    leaves, tdef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)])


def _vdot(a, b):
    return sum(float(jnp.vdot(x, y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("it,calm", [(0, False), (1, False), (1, True)], ids=["zero_ice", "thin_ice", "calm"])
def test_seaice_model_gradients_finite_and_dot(it, calm):
    s = G.setup()
    g, x0 = _gfun(s, it, calm)
    y0, vjp = jax.vjp(g, x0)
    w = _rand_like(y0, 1)
    (xbar,) = vjp(w)
    bad = {n: int(np.count_nonzero(~np.isfinite(np.asarray(v)))) for n, v in xbar.items()}
    assert not any(bad.values()), bad
    v = _rand_like(x0, 2)
    _, ydot = jax.jvp(g, (x0,), (v,))
    badt = {n: int(np.count_nonzero(~np.isfinite(np.asarray(a)))) for n, a in ydot.items()}
    assert not any(badt.values()), badt
    lhs, rhs = _vdot(ydot, w), _vdot(v, xbar)
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs), 1.0), (lhs, rhs)


def test_seaice_model_fd_thin_ice():
    """FD h-sweep of SEAICE_MODEL at iteration 1 (thin ice, interior point) along atemp, lwdown and HEFF (interior):
    the best central difference agrees with the tangent to < 1e-7 for each output that responds and whose FD rounding
    floor, cond = 1e-16*|y|/(|dy/dx|*|dx|), is below 1e-9. Measured: 8 of the 11 responding pairs qualify; excluded
    are HEFF -> Qnet (cond 1.6e-6: its best FD error 2.3e-7 sits at that floor), HEFF -> AREA (2.4e-7) and
    HEFF -> EmPmR (1.5e-8); their tangents are covered by the dot test."""
    s = G.setup()
    g, x0 = _gfun(s, 1)
    y0 = g(x0)
    checked = 0
    for name, scale in (("atemp", 1.0), ("lwdown", 1.0), ("HEFF", 1e-3)):
        d = {n: jnp.zeros_like(a) for n, a in x0.items()}
        a = np.zeros(x0[name].shape)
        a[..., 2, 2] = scale                                   # the interior point (OLx = OLy = 2)
        d[name] = jnp.asarray(a)
        _, t = jax.jvp(g, (x0,), (d,))
        for out in ("HEFF", "AREA", "Qnet", "TICES", "EmPmR"):
            tv = np.asarray(t[out])[..., 2, 2].ravel()[0]
            y = np.asarray(y0[out])[..., 2, 2].ravel()[0]
            if tv == 0.0 or 1e-16*abs(y)/(abs(tv)*scale) > 1e-9:
                continue
            errs = []
            for h in (1e-1, 3e-2, 1e-2, 1e-3, 1e-4):
                xp = {n: x0[n] + h*d[n] for n in x0}
                xm = {n: x0[n] - h*d[n] for n in x0}
                fd = (np.asarray(g(xp)[out]) - np.asarray(g(xm)[out]))[..., 2, 2].ravel()[0] / (2*h)
                errs.append(abs(fd - tv) / abs(tv))
            assert min(errs) < 1e-7, (name, out, tv, errs)
            checked += 1
    assert checked == 8, checked


# ------------------------------------------- 5b. Richardson-extrapolated FD of the three HEFF tangents (session 3)
RICH_PAIRS = ("Qnet", "AREA", "EmPmR")


def _smooth_point(s):
    """The well-conditioned smooth point of the HEFF -> Qnet / AREA / EmPmR check: iteration 2's EXF fields, FFIELDS.h
    and ocean state (S02_load_fields) with a thick, nearly closed, snow-free ice cover on every lane: HEFF = 1 m,
    AREA = 0.9, HSNOW = 0, HSALT = HEFF times iteration 2's HSALT/HEFF, TICES = 260 K. Why there: the oracle's own
    state (HEFF 1.7e-3 m, AREA 3.3e-3 at iteration 1) puts a switch within 4e-4 m of HEFF (measured: the central
    difference of AREA jumps by 1e+4 relative at h = 4e-4) and Qnet's FD rounding floor at 1e-8..1e-7; at 1 m every
    MAX/MIN that HEFF reaches is decided by a margin far larger than the steps (AREApreTH 0.9 > area_reg 0.15,
    HEFF >> hice_reg 0.1, MAX(-HEFF, ...) never limited by HEFF, TICES far below the wet-albedo switch), while
    HSNOW = 0 keeps the penetrating shortwave XIO*exp(-1.5*h) (seaice_solve4temp.F:321-325) live, the path by which
    Qnet depends on HEFF (with snow it is exactly 0: measured)."""
    g, x0 = _gfun(s, 2)
    x0 = dict(x0)
    hs = np.asarray(x0["HSALT"]) / np.asarray(x0["HEFF"])
    x0["HEFF"] = jnp.full_like(x0["HEFF"], 1.0)
    x0["AREA"] = jnp.full_like(x0["AREA"], 0.9)
    x0["HSNOW"] = jnp.full_like(x0["HSNOW"], 0.0)
    x0["HSALT"] = jnp.asarray(hs * 1.0)
    x0["TICES"] = jnp.full_like(x0["TICES"], 260.0)
    return g, x0


def _richardson(g, x0, d, outs, h0, n=4):
    """{out: table}: table[m][k] = the m-times Richardson-extrapolated central difference from steps h0/2**k
    (m = 0: the plain central differences), at the interior point (2, 2)."""
    pt = lambda v: np.asarray(v)[..., 2, 2].ravel()[0]
    D = {o: [] for o in outs}
    for k in range(n):
        h = h0 / 2**k
        yp = g({m: x0[m] + h*d[m] for m in x0})
        ym = g({m: x0[m] - h*d[m] for m in x0})
        for o in outs:
            D[o].append((pt(yp[o]) - pt(ym[o])) / (2*h))
    out = {}
    for o in outs:
        T = [D[o]]
        for m in range(1, n):
            T.append([(4**m*T[-1][k+1] - T[-1][k])/(4**m - 1) for k in range(len(T[-1]) - 1)])
        out[o] = T
    return out


def _heff_direction(x0):
    d = {n: jnp.zeros_like(a) for n, a in x0.items()}
    a = np.zeros(x0["HEFF"].shape)
    a[..., 2, 2] = 1.0                                         # the interior point (OLx = OLy = 2), 1 m
    d["HEFF"] = jnp.asarray(a)
    return d


def _tangent_scaled(g, factor):
    """g with the HEFF tangent scaled by `factor` on its way in (identical forward): the negative control's planted
    change in the derivative path."""
    @jax.custom_jvp
    def ident(x):
        return x

    @ident.defjvp
    def _ident_jvp(primals, tangents):
        return primals[0], tangents[0] * factor

    return lambda x: g({**x, "HEFF": ident(x["HEFF"])})


def _rich_errors(g, x0, gt=None):
    """Relative errors |R - t|/|t| of the Richardson table (h0 = 2e-2 m, 4 steps) against the tangent of `gt`."""
    d = _heff_direction(x0)
    _, t = jax.jvp(g if gt is None else gt, (x0,), (d,))
    tv = {o: np.asarray(t[o])[..., 2, 2].ravel()[0] for o in RICH_PAIRS}
    tab = _richardson(g, x0, d, RICH_PAIRS, 2e-2)
    return tv, {o: [[abs(v - tv[o])/abs(tv[o]) for v in row] for row in tab[o]] for o in RICH_PAIRS}


def test_seaice_model_fd_richardson_heff():
    """HEFF -> Qnet, HEFF -> AREA, HEFF -> EmPmR of SEAICE_MODEL (the three pairs session 2 left to the dot test) by
    Richardson-extrapolated central differences at the smooth point `_smooth_point`: steps h = 2e-2 / 2**k m,
    k = 0..3. Asserted: the tangent is nonzero; the plain central differences converge at second order (successive
    error ratios 4 +- 0.4 for the two largest steps: no switch within reach); the once-extrapolated value at the two
    largest steps and the twice-extrapolated one agree with the tangent to < 1e-7 (bar unchanged). Measured on the
    login node (2026-10-03), Qnet / AREA / EmPmR: central at 2e-2 1.7e-4 / 3.9e-4 / 2.4e-4 (ratios 4.0);
    R1(2e-2, 1e-2) 1.6e-9 / 3.7e-8 / 1.4e-8; R2 1.3e-12 / 6.2e-11 / 1.6e-13; best entry (the error floor) 4.8e-13 /
    1.0e-11 / 1.6e-13. With smaller steps (h0 = 4e-3) R1 rises again to 1.6e-11 / 1.7e-9 / 1.1e-11 at 2.5e-4:
    rounding."""
    s = G.setup()
    g, x0 = _smooth_point(s)
    tv, err = _rich_errors(g, x0)
    for o in RICH_PAIRS:
        assert tv[o] != 0.0, o
        c = err[o][0]
        assert 3.6 < c[0]/c[1] < 4.4 and 3.6 < c[1]/c[2] < 4.4, (o, c)
        assert err[o][1][0] < 1e-7 and err[o][2][0] < 1e-7, (o, err[o])


def test_seaice_model_fd_richardson_negative_control():
    """The same check against a tangent planted 1e-6 too large on the HEFF input (custom_jvp identity, forward
    unchanged): every pair fails the 1e-7 bar (measured relative error 1.0e-6 on each)."""
    s = G.setup()
    g, x0 = _smooth_point(s)
    _, err = _rich_errors(g, x0, gt=_tangent_scaled(g, 1.0 + 1e-6))
    for o in RICH_PAIRS:
        assert err[o][1][0] > 1e-7 and err[o][2][0] > 1e-7, (o, err[o])
