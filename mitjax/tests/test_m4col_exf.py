"""pkg/cal and pkg/exf gates of 1D_ocean_ice_column/input (M4 step 3, lane M4COL).

1. Calendar (CAL_READPARMS, CAL_SET) and the EXF field start times (EXF_READPARMS, EXF_INIT_FIXED /
   EXF_GETFFIELD_START) against the oracle's STDOUT (CAL_SUMMARY, EXF_SUMMARY), incl. data.exf's repeated keys
   (atempperiod, swdownperiod, lwdownperiod assigned twice: the namelist READ keeps the last, 86400).
2. EXF_GETFORCING bitwise vs the dumps X01_exf_getffields .. X06_exf_mapfields at iterations 0, 1, 2, every point
   (halos included): teacher-forced (the oracle's EXF fields at the end of the previous iteration) and our own
   chain from EXF_INIT_VARIA (records and fields carried by the port alone; between iterations SEAICE_MODEL's
   exchange of uwind/vwind, seaice_model.F:124-126, checked against I00b_seaice_begin).
3. Negative controls, each measured to bite (counts asserted > 0): a planted Dalton number, the first (overridden)
   atempperiod of data.exf, cen2kel = celsius2K, the teacher prior taken at X06 instead of after SEAICE_MODEL's
   wind exchange, a planted start date.
4. Gradients of EXF_GETFORCING's traced part (X02..X06) w.r.t. its inputs: finite on every lane at the oracle's
   state and at a lane with atemp = 0 and zero wind (both guarded IF branches), tangent vs adjoint dot test, FD
   h-sweep at the oracle's (smooth) state.
Under the gate XLA flags (conftest.py); REAL parameters are traced jit arguments.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import exf_gate as G


def _stdout(s):
    return (s.rundir / "output.txt").read_text(errors="replace").splitlines()


def _value_after(lines, key):
    """The value printed on the line after `key =` in a CAL/EXF summary block."""
    for k, ln in enumerate(lines):
        if key in ln:
            return lines[k + 1].split(")", 1)[1].strip()
    raise KeyError(key)


# ----------------------------------------------------------------------------------------------- 1. calendar
def test_calendar_and_start_times_vs_stdout():
    s = G.setup()
    out = _stdout(s)
    cal = s.cal
    assert float(_value_after(out, "modelstart =")) == cal.modelStart
    assert float(_value_after(out, "modelend  =")) == cal.modelEnd
    assert float(_value_after(out, "modelStep =")) == cal.modelStep
    assert int(_value_after(out, "modelStartDate YYYYMMDD")) == cal.modelStartDate[0]
    assert int(_value_after(out, "modelStartDate HHMMSS")) == cal.modelStartDate[1]
    assert int(_value_after(out, "modelEndDate   YYYYMMDD")) == cal.modelEndDate[0]
    assert int(_value_after(out, "modelEndDate   HHMMSS")) == cal.modelEndDate[1]
    printed = {}
    for ln in out:
        m = re.search(r"\)\s+(.*?) (starts at|period is)\s+(-?[\d.]+)", ln)
        if m:
            printed[(m.group(1).strip(), m.group(2))] = float(m.group(3))
    names = {"Zonal wind forcing": "uwind", "Meridional wind forcing": "vwind", "Atmospheric temperature": "atemp",
             "Downward shortwave flux": "swdown", "Downward longwave flux": "lwdown"}
    assert len(printed) == 10
    for label, fld in names.items():
        assert printed[(label, "starts at")] == getattr(s.exf, fld + "StartTime"), fld
        assert printed[(label, "period is")] == getattr(s.exf, fld + "period"), fld
    assert s.exf.atempperiod == 86400.0 and s.exf.uwindperiod == 2635200.0


def test_calendar_negative_control_start_date():
    """A planted start date (atempstartdate2 = 0 instead of 180000) moves the start time: the STDOUT check bites."""
    from mitjax.pkg.exf.exf_getffield_start import exf_getffield_start
    from mitjax.model.grid import UNSET_RL
    s = G.setup()
    t, err = exf_getffield_start(False, "exf", "atemp", s.exf.atempperiod, s.exf.atempstartdate1, 0, UNSET_RL, 0,
                                 useCAL=True, cal=s.cal, nIter0=s.tp.nIter0, startTime=s.tp.startTime)
    assert err == 0 and t != s.exf.atempStartTime and t == -1382400.0


# ----------------------------------------------------------------------------------------------- 2. dump gates
@pytest.mark.parametrize("teacher", [True, False], ids=["teacher_forced", "own_chain"])
def test_exf_getforcing_bitwise_vs_dumps(teacher):
    s = G.setup()
    assert s.ds.iterations() == [0, 1, 2]
    res = G.run_steps(s, teacher=teacher)
    assert [it for it, _ in res] == [0, 1, 2]
    for it, cmp in res:
        assert set(cmp) == set(G.STAGES) | {"I00b_winds"}
        for st, c in cmp.items():
            assert len(c) == (2 if st == "I00b_winds" else
                              len(G.EXF_DUMPED) + (len(G.FF_MAPPED) if st == "X06_exf_mapfields" else 0))
            assert all(v[0] == 25 for v in c.values())                       # 1x1 tile with OLx = OLy = 2
            bad = G.failures(c)
            assert not bad, f"iteration {it} {st}: {bad}"


# ----------------------------------------------------------------------------------------------- 3. controls
def _n_bad(res, stage=None):
    return sum(sum(v[1] + v[4] for v in G.failures(c).values())
               for _, cmp in res for st, c in cmp.items() if stage is None or st == stage)


@pytest.mark.parametrize("planted,first_stage", [
    ((("cdalton", 0.0347),), "X04_exf_bulkformulae"),
    ((("atempperiod", 2635200.0),), "X01_exf_getffields"),
    ((("cen2kel", 273.16),), "X02_exf_radiation"),
], ids=["cdalton", "atempperiod_first_assignment", "cen2kel_celsius2K"])
def test_negative_controls_bite(planted, first_stage):
    s = G.setup(planted_exf=planted)
    res = G.run_steps(s)
    assert _n_bad(res, first_stage) > 0
    k = G.STAGES.index(first_stage)
    assert all(_n_bad(res, st) == 0 for st in G.STAGES[:k]), "a control must not bite before its stage"


def test_negative_control_teacher_prior_at_x06():
    """The EXF fields of the previous iteration taken at X06 miss SEAICE_MODEL's exchange of uwind/vwind."""
    s = G.setup()
    res = G.run_steps(s, its=(0, 1), last_stage=False)
    bad = G.failures(res[1][1]["X01_exf_getffields"])
    assert set(bad) == {"uwind", "vwind"} and all(v[1] > 0 for v in bad.values())


# ----------------------------------------------------------------------------------------------- 4. gradients
def _grad_setup(zero_lane=False):
    """(f, x0): f(x) -> X06 outputs of EXF_GETFORCING's traced part as one vector, x = the inputs (atemp, aqh,
    uwind, vwind, lwdown, swdown, exf_Tsf) on every lane, at the oracle's iteration-1 X01 state."""
    from mitjax.model.src.ini_ffields import ini_ffields
    from mitjax.pkg.exf.exf_getforcing import exf_getforcing_fluxes
    from types import SimpleNamespace
    s = G.setup()
    sz = s.sz
    o = G.oracle_stage(s, 1, "X01_exf_getffields")
    theta = s.ds.field(1, "S00_begin", "theta")[:, 0]
    names = ("atemp", "aqh", "uwind", "vwind", "lwdown", "swdown")
    x0 = {n: jnp.asarray(o[n]) for n in names}
    x0["Tsf"] = jnp.asarray(theta) + s.exf.cen2kel
    if zero_lane:
        x0 = {n: v.at[0, sz.OLy, sz.OLx].set(0.) if n in ("atemp", "uwind", "vwind") else v for n, v in x0.items()}
    base = {n: jnp.asarray(o[n]) for n in G.EXF_DUMPED}
    ff0 = ini_ffields(cfg=s.cfg)

    def f(x, exf, params, fp):
        flds = {n: G._xy(n, base[n], sz) for n in G.EXF_DUMPED}
        for n in names:
            flds[n] = G._xy(n, x[n], sz)
        f2, ff = exf_getforcing_fluxes(G._xy("Tsf", x["Tsf"], sz), jnp.float64(3600.), 1, flds, ff0, cfg=s.cfg,
                                       exf=exf, grid=s.grid, params=params, fp=fp,
                                       state=SimpleNamespace(theta=None, uVel=None, vVel=None), tp=s.tp, ex=s.ex)
        outs = [f2[n].data for n in ("hs", "hl", "evap", "lwflux", "swflux", "ustress", "vstress", "hflux", "sflux")]
        outs += [getattr(ff, n).data for n in G.FF_MAPPED]
        return jnp.concatenate([a.ravel() for a in outs])
    return s, f, x0


@pytest.mark.parametrize("zero_lane", [False, True], ids=["oracle_state", "atemp0_zero_wind"])
def test_gradient_finite_every_lane(zero_lane):
    s, f, x0 = _grad_setup(zero_lane)
    y, vjp = jax.vjp(lambda x: f(x, s.exf, s.params, s.fp), x0)
    assert bool(jnp.all(jnp.isfinite(y)))
    ct = jax.random.normal(jax.random.PRNGKey(1), y.shape, jnp.float64)
    (g,) = vjp(ct)
    for n, v in g.items():
        assert bool(jnp.all(jnp.isfinite(v))), n
    assert any(bool(jnp.any(v != 0.)) for v in g.values())


def test_gradient_guard_negative_control():
    """Why the zero-wind lane needs the guard: the unguarded `where(wsSq /= 0, sqrt(wsSq), 0)` of EXF_WIND :133-141
    has a NaN derivative at wsSq = 0 (measured), the guarded form used by the port a finite one."""
    from mitjax.ops.safe import safe_sqrt
    raw = jax.grad(lambda w: jnp.where(w != 0., jnp.sqrt(w), 0.))(0.)
    guarded = jax.grad(lambda w: safe_sqrt(w, w != 0.))(0.)
    assert bool(jnp.isnan(raw)) and bool(jnp.isfinite(guarded))


def test_gradient_dot_test_and_fd():
    s, f, x0 = _grad_setup()
    fx = jax.jit(lambda x: f(x, s.exf, s.params, s.fp))
    key = jax.random.PRNGKey(7)
    ks = jax.random.split(key, len(x0) + 1)
    v = {n: jax.random.normal(k, x0[n].shape, jnp.float64) for n, k in zip(sorted(x0), ks[:-1])}
    # scale the direction to the size of each input (Tsf ~ 271 K, atemp ~ 260 K, winds ~ 1 m/s, fluxes ~ 1e2)
    v = {n: v[n]*(1e-3*jnp.maximum(jnp.abs(x0[n]), 1e-6)) for n in v}   # MINMAX-RAW: test scaling, not Fortran
    y, jv = jax.jvp(fx, (x0,), (v,))
    w = jax.random.normal(ks[-1], y.shape, jnp.float64)
    _, vjp = jax.vjp(fx, x0)
    (wj,) = vjp(w)
    lhs = float(jnp.vdot(w, jv))
    rhs = float(sum(jnp.vdot(wj[n], v[n]) for n in v))
    assert abs(lhs - rhs) <= 1e-12*max(abs(lhs), abs(rhs))
    # FD h-sweep (central differences) at the oracle's state: the best step agrees with the tangent
    errs = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        yp = fx({n: x0[n] + h*v[n] for n in x0})
        ym = fx({n: x0[n] - h*v[n] for n in x0})
        fd = (yp - ym)/(2*h)
        errs.append(float(jnp.max(jnp.abs(fd - jv))/jnp.max(jnp.abs(jv))))
    assert min(errs) < 1e-7, errs


def test_exf_getforcing_eager_whole_routine_iteration0():
    """EXF_GETFORCING as one call (host reads + the rest, eager) at iteration 0 from EXF_INIT_VARIA: every stage
    bitwise (the gate above runs the traced part under jit; this checks the composition of exf_getforcing)."""
    from types import SimpleNamespace
    from mitjax.model.src.ini_ffields import ini_ffields
    from mitjax.pkg.exf.exf_getforcing import exf_getforcing
    s = G.setup()
    fields = G.init_fields(s)
    theta = s.ds.field(0, "S00_begin", "theta")
    ff = ini_ffields(cfg=s.cfg).replace(gcmSST=G._xy("gcmSST", theta[:, 0], s.sz))
    got = {}

    def probe(stage, fl, ff_):
        d = {n: np.asarray(fl[n].data) for n in G.EXF_DUMPED}
        if stage == "X06_exf_mapfields":
            d.update({n: np.asarray(getattr(ff_, n).data) for n in G.FF_MAPPED})
        got[stage] = d
    exf_getforcing(np.float64(s.tp.startTime), 0, fields, ff, cfg=s.cfg, exf=s.exf, cal=s.cal, grid=s.grid,
                   params=s.params, fp=s.fp, state=SimpleNamespace(theta=None, uVel=None, vVel=None), rw=s.rw,
                   tp=s.tp, ex=s.ex, probe=probe)
    assert set(got) == set(G.STAGES)
    for st in G.STAGES:
        bad = G.failures(G.compare(got[st], G.oracle_stage(s, 0, st)))
        assert not bad, (st, bad)
