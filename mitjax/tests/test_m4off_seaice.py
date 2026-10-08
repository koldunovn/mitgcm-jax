"""offline_exf_seaice/input.thermo (M4 step 4, lane M4OFF session 1): the parts ported so far, gated bitwise against
lane A's dumps-on run job27855986-jdon (iterations 0-2, every point incl. halos), each with a measured negative control
(or the measured statement that no plant can bite in this run: a structural blind spot).

1. The front of FORWARD_STEP (the Model of the -cal build: EXF without pkg/cal, climsst file, surf_pRef with the
   LINEAR EOS): S00_begin, X01..X06, S02_load_fields at iterations 0 (own initial carry), 1, 2 (teacher-forced).
2. SEAICE_INIT_FIXED / SEAICE_INIT_VARIA of the C-grid build (HeffFile, AreaFile, HsnowFile, seaiceMaskU/V, SIMaskU/V,
   k1/k2 metric terms): I00_seaice_begin at iteration 0, sIceLoad at S00_begin.
3. SEAICE_DYNSOLVER without dynamics: SEAICE_GET_DYNFORCING (Y01: TAUX, TAUY), SEAICE_OCEAN_STRESS (Y09: fu, fv),
   from the oracle's I00 / S02 / S00 values.
4. SEAICE_ADVDIFF (multi-dimensional, scheme 77) from I01 -> I02: bitwise, but uIce = vIce = 0 in this run, so every
   flux is 0 and no plant in the advection can bite (asserted: a blind spot; gated in input.dyn_lsr).
5. SEAICE_REG_RIDGE without SEAICE_VARIABLE_SALINITY from I02 -> I03.
Session 2:
6. SEAICE_GROWTH (+ SOLVE4TEMP, BUDGET_OCEAN) of this build from I03 -> I04 (no SEAICE_VARIABLE_SALINITY: SEAICE_salt0;
   sublimation; growMeltByConv; areaLossFormula 2; no SHORTWAVE_HEATING; the heat-conservation fix compiled), with
   measured negative controls of each new arm and the measured blind spots (SEAICEheatConsFix and the ALLOW_DIAGNOSTICS
   locals cannot bite: useRealFreshWaterFlux .FALSE., output only).
7. SEAICE_MODEL from the oracle's fields before I00: every sea-ice stage I00..I04 and P13.
8. The whole FORWARD_STEP (useSEAICE without a mixing package, FORCING_SURF_RELAX with SEAICErestoreUnderIce,
   momStepping .FALSE.): steps 0-2 at every dumped stage, free (chained from the Model's own initial carry) and
   teacher-forced, with a measured negative control.
9. The 120-step run through the run driver, the restart, P=4 == P=1 (test_m4off_run.py).
"""

import numpy as np
import pytest

from mitjax.tests import m4off_gate as G

ITS = (0, 1, 2)


@pytest.fixture(scope="module")
def md():
    return G.model()


def test_front_of_step_exf_bitwise(md):
    m, ds = md
    res = G.run_front(m, ds)
    for it in ITS:
        stages = set(res[it])
        assert set(G.FRONT_STAGES) <= stages, (it, sorted(stages))
        assert sum(len(res[it][s]) for s in G.FRONT_STAGES) >= 190, it
        assert not G.bad(res[it]), (it, G.bad(res[it]))


def test_front_of_step_negative_control_surf_pRef(md):
    """surf_pRef (carried for the LINEAR EOS by the Model, read by EXF_MAPFIELDS' pLoad) x (1 + 1e-12): X06 pLoad
    differs at every iteration (measured: every point of the tiles' interiors)."""
    m, ds = md
    a = m.arrays
    p = a.params.replace(traced={"surf_pRef": a.params.surf_pRef*(1.0 + 1e-12)})
    res = G.run_front(m, ds, its=(0,), arrays=a._replace(params=p) if hasattr(a, "_replace") else
                      type(a)(**{**a.__dict__, "params": p}))
    b = G.bad(res[0])
    assert ("X06_exf_mapfields", "pLoad") in b and ("S02_load_fields", "pLoad") in b, sorted(b)


def test_seaice_init_bitwise(md):
    m, ds = md
    sf = m.pk0["seaice"]
    names = [n for n in ("AREA", "HEFF", "HEFFM", "HSNOW", "SIMaskU", "SIMaskV", "TICES", "UICE", "VICE", "k1AtC",
                         "k1AtU", "k1AtV", "k1AtZ", "k2AtC", "k2AtU", "k2AtV", "k2AtZ", "seaiceMaskU", "seaiceMaskV")]
    d = G.differing(ds, 0, "I00_seaice_begin", {n: sf[n] for n in names})
    assert len(d) == len(names) and not any(d.values()), d
    assert G.differing(ds, 0, "S00_begin", {"sIceLoad": m.ff0.sIceLoad}) == {"sIceLoad": 0}


def test_seaice_init_negative_control(md):
    """SEAICE_INIT_VARIA without the HeffFile read (HeffFile = ' '): HEFF and AREA (:326-327) differ. (sIceLoad is
    not written: useRealFreshWaterFlux is .FALSE. in this run, :447; it stays 0 in both, measured.)"""
    from mitjax.pkg.seaice.seaice_init_varia import seaice_init_varia
    m, ds = md
    sp, op = m.arrays.pkc["sp"], m.arrays.pkc["op"]
    sf, ff = seaice_init_varia(m.pk0["seaice"], m.ff0, cfg=m.cfg, sp=sp.replace(HeffFile=" "), op=op,
                               state=m.state0, params=m.params, tp=m.prm.time, rw=m.rw, ex=m.ex, ip=m.prm.init)
    d = G.differing(ds, 0, "I00_seaice_begin", {n: sf[n] for n in ("HEFF", "AREA", "HSNOW", "TICES")})
    assert d["HEFF"] > 0 and d["AREA"] > 0, d
    assert not np.any(np.asarray(ff.sIceLoad.data))


def _dynsolver(m, ds, it, sp=None):
    from mitjax.pkg.seaice.seaice_dynsolver import seaice_dynsolver
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    sf = dict(m.pk0["seaice"])
    sf.update(G.stage_inputs(m, ds, it, "I00_seaice_begin", ("AREA", "HEFF", "HSNOW", "UICE", "VICE")))
    sf["DWATN"] = G.like(sf["DWATN"], ds.field(it, "I01_dynsolver", "DWATN"))   # not written without dynamics
    exf = dict(m.pk0["exf"])
    exf.update(G.stage_inputs(m, ds, it, "I00_seaice_begin", ("uwind", "vwind")))
    ff = m.ff0.replace(**G.stage_inputs(m, ds, it, "S02_load_fields", ("fu", "fv")))
    st = m.state0.replace(uVel=G.like(m.state0.uVel, ds.field(it, "S00_begin", "uVel")),
                          vVel=G.like(m.state0.vVel, ds.field(it, "S00_begin", "vVel")))
    probes = {}

    def probe(stage, vals, ff_):
        probes[stage] = vals
    _, ff1 = seaice_dynsolver(0., it, sf, ff, exf, cfg=m.cfg, sp=sp, op=pkc["op"], exfp=pkc["exfp"],
                              grid=m.arrays.grid, state=st, ex=m.ex, probe=probe)
    return (G.differing(ds, it, "Y01_get_dynforcing", probes["Y01_get_dynforcing"]),
            G.differing(ds, it, "Y09_ocean_stress", {"fu": ff1.fu, "fv": ff1.fv}))


def test_seaice_dynsolver_bitwise(md):
    m, ds = md
    for it in ITS:
        y01, y09 = _dynsolver(m, ds, it)
        assert y01 == {"TAUX": 0, "TAUY": 0} and y09 == {"fu": 0, "fv": 0}, (it, y01, y09)


def test_seaice_dynsolver_negative_controls(md):
    """SEAICE_drag x (1 + 1e-6): TAUX differs (TAUY = 0: vwind = 0); SEAICEstressFactor 0.999: fu differs (fv = 0)
    (measured at iteration 1: TAUX 2160 points, fu 4548 points)."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    y01, _ = _dynsolver(m, ds, 1, sp.replace(SEAICE_drag=float(sp.SEAICE_drag)*1.000001))
    assert y01["TAUX"] > 0, y01
    _, y09 = _dynsolver(m, ds, 1, sp.replace(SEAICEstressFactor=0.999))
    assert y09["fu"] > 0, y09


def _advdiff(m, ds, it, sp=None):
    from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    sf = dict(m.pk0["seaice"])
    sf.update(G.stage_inputs(m, ds, it, "I01_dynsolver", ("AREA", "HEFF", "HSNOW", "UICE", "VICE")))
    sf2 = seaice_advdiff(sf["UICE"], sf["VICE"], 0., it, sf, cfg=m.cfg, sp=sp, op=pkc["op"], grid=m.arrays.grid)
    return G.differing(ds, it, "I02_advdiff", {n: sf2[n] for n in ("AREA", "HEFF", "HSNOW")}), sf


def test_seaice_advdiff_bitwise_and_blind_spot(md):
    m, ds = md
    sp = m.arrays.pkc["sp"]
    for it in ITS:
        d, sf = _advdiff(m, ds, it)
        assert d == {"AREA": 0, "HEFF": 0, "HSNOW": 0}, (it, d)
        assert not np.any(np.asarray(sf["UICE"].data)) and not np.any(np.asarray(sf["VICE"].data))
    # blind spot (measured): with uIce = vIce = 0 every flux is 0, a planted time step does not bite
    d, _ = _advdiff(m, ds, 1, sp.replace(SEAICE_deltaTtherm=float(sp.SEAICE_deltaTtherm)*1.000001))
    assert d == {"AREA": 0, "HEFF": 0, "HSNOW": 0}, d


def _reg_ridge(m, ds, it, sp=None):
    from mitjax.pkg.seaice.seaice_reg_ridge import seaice_reg_ridge
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    sf = dict(m.pk0["seaice"])
    sf.update(G.stage_inputs(m, ds, it, "I02_advdiff", ("AREA", "HEFF", "HSNOW", "TICES")))
    sf3 = seaice_reg_ridge(0., it, sf, cfg=m.cfg, sp=sp, op=pkc["op"])
    return G.differing(ds, it, "I03_reg_ridge", {n: sf3[n] for n in ("AREA", "HEFF", "HSNOW", "TICES", "d_HEFFbyNEG",
                                                                      "d_HSNWbyNEG")})


def test_seaice_reg_ridge_bitwise(md):
    m, ds = md
    for it in ITS:
        d = _reg_ridge(m, ds, it)
        assert len(d) == 6 and not any(d.values()), (it, d)


def test_seaice_reg_ridge_negative_control(md):
    """SEAICE_area_max 0.999 (:381): AREA differs at every iteration (measured 2960 / 2590 / 2368 points); a planted
    SEAICE_area_floor x 1.5 does not bite (no ice cover between the floors in iterations 0-2: blind spot)."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    for it in ITS:
        assert _reg_ridge(m, ds, it, sp.replace(SEAICE_area_max=0.999))["AREA"] > 0, it
    d = _reg_ridge(m, ds, 1, sp.replace(SEAICE_area_floor=float(sp.SEAICE_area_floor)*1.5))
    assert not any(d.values()), d


# ----------------------------------------------------------------------------------------- session 2: growth
def test_seaice_growth_bitwise(md):
    """SEAICE_GROWTH teacher-forced from the oracle's fields before I04 (I03 and earlier): the 10 SEAICE.h and 8 FFIELDS.h
    fields of I04_growth on every point, iterations 0-2."""
    m, ds = md
    for it, d in G.run_growth(m, ds).items():
        assert len(d) == 18 and not any(d.values()), (it, d)


@pytest.mark.parametrize("plant,must", [
    # measured (points at iterations 0 / 1 / 2): AREA 128/134/132, HEFF 370/896/1078, Qnet 1622/1326/1528,
    # EmPmR 48/520/496, saltFlux 46/426/364
    (dict(sp={"SEAICE_growMeltByConv": False}), {"AREA", "HEFF", "Qnet", "EmPmR", "saltFlux"}),
    # AREA 24/26/26 (formula 1 sums the three MINs separately: differs where the signs mix)
    (dict(sp={"SEAICE_areaLossFormula": 1}), {"AREA"}),
    # saltFlux 3184/3186/3186 (MIN(SEAICE_salt0, salt) = SEAICE_salt0 everywhere: salt > 4)
    (dict(sp={"SEAICE_salt0": 4.0*(1.0 + 1e-6)}), {"saltFlux"}),
    # HEFF and saltFlux 3120/3184/3164 (a_FWbySublim = 0 with SEAICE_DISABLE_SUBLIM, :967)
    (dict(on=("SEAICE_DISABLE_SUBLIM",)), {"HEFF", "saltFlux"}),
], ids=["growMeltByConv", "areaLossFormula1", "salt0", "disable_sublim"])
def test_seaice_growth_negative_controls(md, plant, must):
    from mitjax.tests import m4col_gate as C
    m, ds = md
    sp = m.arrays.pkc["sp"]
    fn = G.growth_fn(m, C.cfg_with(m, on=plant["on"])) if "on" in plant else None
    res = G.run_growth(m, ds, sp=sp.replace(**plant["sp"]) if "sp" in plant else None, fn=fn)
    for it, d in res.items():
        assert must <= {n for n, v in d.items() if v}, (plant, it, d)


def test_seaice_growth_blind_spots(md):
    """Measured: the heat-conservation fix (:2239-2300) changes QNET only with useRealFreshWaterFlux and
    nonlinFreeSurf > 0 (both off in input.thermo), so SEAICEheatConsFix .TRUE. and a build with
    SEAICE_DISABLE_HEATCONSFIX give the same bits; the ALLOW_DIAGNOSTICS locals d_AREAby* are output only."""
    from mitjax.tests import m4col_gate as C
    m, ds = md
    sp = m.arrays.pkc["sp"]
    for kw in (dict(sp=sp.replace(SEAICEheatConsFix=True)),
               dict(fn=G.growth_fn(m, C.cfg_with(m, on=("SEAICE_DISABLE_HEATCONSFIX",)))),
               dict(fn=G.growth_fn(m, C.cfg_with(m, off=("ALLOW_DIAGNOSTICS",))))):
        for it, d in G.run_growth(m, ds, **kw).items():
            assert len(d) == 18 and not any(d.values()), (it, d)


def test_seaice_model_every_stage_bitwise(md):
    """SEAICE_MODEL (C-grid: DYNSOLVER without dynamics, ADVDIFF, REG_RIDGE, GROWTH, the exchanges :317-342) from the
    oracle's fields before I00_seaice_begin, compared at I00, Y01, Y09, I01, I02, I03, I04 and P13 (194 fields per
    iteration), iterations 0-2."""
    m, ds = md
    for it, res in G.run_seaice_model(m, ds).items():
        assert set(res) == set(G.SEAICE_STAGES) | {"P13_seaice_model"}, (it, sorted(res))
        assert sum(len(d) for d in res.values()) == 194, (it, {s: len(d) for s, d in res.items()})
        assert not any(v for d in res.values() for v in d.values()), (it, res)


# ------------------------------------------------------------------------------------- session 2: whole step
def test_steps_0_2_every_dumped_stage_bitwise():
    """FORWARD_STEP of the driver (useSEAICE without a mixing package; momStepping .FALSE.; FORCING_SURF_RELAX with
    SEAICErestoreUnderIce) at iterations 0-2, free (chained from the Model's own initial carry) and teacher-forced:
    every dumped stage (27 stages, 475 fields per iteration; TICES category 1) bitwise on every point."""
    r = G.run()
    for teacher in (False, True):
        for it, (bad, ncmp) in G.steps(r, teacher=teacher).items():
            assert len(ncmp) == 27 and sum(ncmp.values()) == 475, (teacher, it, ncmp)
            assert not bad, (teacher, it, bad)


def test_steps_negative_control_mcphee_taper():
    """SEAICE_mcPheeTaper 0.9 (SEAICE_GROWTH :1036-1039) planted in the Model's arrays: the free steps differ from I04
    on (HEFF, Qnet at I04; theta at T02 through the surface heat flux)."""
    r = G.run()
    m = r.m
    a = m.arrays
    pkc = dict(a.pkc, sp=a.pkc["sp"].replace(SEAICE_mcPheeTaper=0.9))
    res = G.steps(r, its=(1,), arrays=a.replace(pkc=pkc))
    bad = res[1][0]
    assert {"HEFF", "Qnet"} <= set(bad.get("I04_growth", {})), sorted(bad)
    assert "theta" in bad.get("T02_temp_integrate", {}), sorted(bad)


# ----------------------------------------------------------------------------------- session 2: growth gradients
def _rand_like(tree, seed):
    import jax
    leaves, tdef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)])


def _vdot(a, b):
    import jax
    import jax.numpy as jnp
    return sum(float(jnp.vdot(x, y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("it,smooth", [(0, False), (1, False), (2, False), (1, True)],
                         ids=["it0", "it1", "it2", "thick_snow"])
def test_seaice_growth_gradients_finite_and_dot(md, it, smooth):
    """SEAICE_GROWTH of this build (G.growth_gfun: HEFF, AREA, HSNOW, TICES, the EXF fields, Qnet, Qsw, theta, salt on
    every lane -> HEFF, AREA, HSNOW, TICES, Qnet, Qsw, EmPmR, saltFlux): reverse and tangent finite on every lane
    (halos, land), dot test with random directions on every lane <= 1e-12 relative (measured 2.4e-15, 1.9e-15,
    3.2e-16, 0)."""
    import jax
    m, ds = md
    g, x0 = G.growth_gfun(m, ds, it, smooth)
    y0, vjp = jax.vjp(g, x0)
    w = _rand_like(y0, 1)
    (xbar,) = vjp(w)
    v = _rand_like(x0, 2)
    _, ydot = jax.jvp(g, (x0,), (v,))
    for tree in (xbar, ydot):
        bad = {n: int(np.count_nonzero(~np.isfinite(np.asarray(a)))) for n, a in tree.items()}
        assert not any(bad.values()), bad
    lhs, rhs = _vdot(ydot, w), _vdot(v, xbar)
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs), 1.0), (lhs, rhs)


GROWTH_FD_DIRS = (("atemp", 1.0), ("lwdown", 1.0), ("HEFF", 1e-2), ("AREA", 1e-2), ("theta", 1e-2))


def _growth_fd_errors(m, ds):
    """{(input, output): (tangent, [relative errors of the central FD at h = 1e-1 .. 1e-4])} of SEAICE_GROWTH at the
    oracle's own state of iteration 1 at the interior wet point G.PT (thin ice, HEFF 0.20 m, AREA 0.999, no snow),
    for each output that responds and whose FD rounding floor 1e-16*|y|/|dy/dx| is below 1e-9."""
    import jax
    import jax.numpy as jnp
    g, x0 = G.growth_gfun(m, ds, 1)
    y0 = g(x0)

    def pt(a):
        a = np.asarray(a)
        return a[G.PT] if a.ndim == 3 else a[G.PT[0], 0, G.PT[1], G.PT[2]]
    out = {}
    for name, scale in GROWTH_FD_DIRS:
        d = {n: jnp.zeros_like(a) for n, a in x0.items()}
        a = np.zeros(x0[name].shape)
        if a.ndim == 3:
            a[G.PT] = scale
        else:
            a[G.PT[0], 0, G.PT[1], G.PT[2]] = scale
        d[name] = jnp.asarray(a)
        _, t = jax.jvp(g, (x0,), (d,))
        for o in G.GY_SF + G.GY_FF:
            tv, y = float(pt(t[o])), float(pt(y0[o]))
            if tv == 0.0 or 1e-16*abs(y)/abs(tv) > 1e-9:
                continue
            errs = []
            for h in (1e-1, 3e-2, 1e-2, 1e-3, 1e-4):
                fd = (pt(g({n: x0[n] + h*d[n] for n in x0})[o]) - pt(g({n: x0[n] - h*d[n] for n in x0})[o])) / (2*h)
                errs.append(abs(fd - tv) / abs(tv))
            out[(name, o)] = (tv, errs)
    return out


def test_seaice_growth_fd(md):
    """Central FD h-sweep of SEAICE_GROWTH at the oracle's iteration-1 state at G.PT along atemp, lwdown, HEFF, AREA
    and theta of that point (no halo copies: the point is 10 points from every tile edge): the best FD agrees with
    the tangent to < 1e-7 for every responding, well-conditioned pair (measured: 29 pairs, worst 4.0e-9 HEFF -> AREA)."""
    m, ds = md
    res = _growth_fd_errors(m, ds)
    assert len(res) == 29, sorted(res)
    worst = {k: min(e) for k, (_, e) in res.items() if min(e) >= 1e-7}
    assert not worst, worst


def test_seaice_growth_fd_negative_control(md, monkeypatch):
    """Negative control of test_seaice_growth_fd: every Fortran MAX/MIN derivative times (1 + 1e-4), the values
    unchanged (m4col: test_m4col_column._planted_minmax_step): every one of the 29 pairs fails the 1e-7 bar."""
    from mitjax.ops import fortran_minmax as FM
    from mitjax.tests.test_m4col_column import _planted_minmax_step
    for key in list(FM._STEPS):
        monkeypatch.setitem(FM._STEPS, key, _planted_minmax_step(*key, 1.0 + 1e-4))
    m, ds = md
    res = _growth_fd_errors(m, ds)
    worst = {k: min(e) for k, (_, e) in res.items() if min(e) >= 1e-7}
    print("planted MAX/MIN derivatives: pairs above the bar", worst)
    assert len(worst) == 29, (len(worst), worst)        # measured: every pair (worst-case 1.4e-7 HEFF -> HEFF)
