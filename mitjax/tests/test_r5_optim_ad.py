"""R5 tutorial_global_oce_optim/input_ad (code_ad build), plan Task 16: the forward step's front half against the
oracle's substep dumps, teacher-forced from each step's S00_begin dump (helpers and oracle runs: r5_gate.py).

Gated here (every point of every tile incl. halos, bit patterns; all 10 steps): S02 (periodic forcing through the
host preload), S03 (CTRL_MAP_GENTIM2D + CTRL_MAP_FORCING in the step), P01 (FREEZE_SURFACE + EXTERNAL_FORCING_SURF,
salt relaxation), P02 (ALLOW_AUTODIFF resets, JMD95Z FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC), P03, P05/P06
(GMREDI_CALC_TENSOR dm95, GMREDI_DO_EXCH), S04, T01 (GMREDI_RESIDUAL_FLOW), T11-T13 and T02 (TEMP_INTEGRATE with the
GM/Redi fluxes and GMREDI_CALC_DIFF), T21-T23, T03 (SALT_INTEGRATE, ADVECT lane's port, with GM/Redi) and S05.
Each planted error of NEGATIVE bites at the stage named (measured), with every earlier stage still bitwise. Not yet
reachable (GO lane): the CD scheme in DYNAMICS and INTEGR_CONTINUITY's real-fresh-water branch; hence no whole-run
fc / %MON / admCst gate yet.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

STEPS = tuple(range(10))


def _front(m, until=None, steps=STEPS, fn=None, **kw):
    from mitjax.tests import r5_gate as r5
    fn = fn or m.front_fn(until or r5.UNTIL)
    out = {}
    genarr = None
    for it in range(max(steps) + 1):
        o, probes = m.run_front(it, fn, genarr=genarr, **kw)
        genarr = o[5]["pk"]["genarr"]           # CTRL_GENARR.h carried from step to step (the control's records)
        if it in steps:
            out[it] = r5.compare_front(m, it, probes)
    return out


def test_front_bitwise_zero_control():
    """All FRONT stages of all 10 steps bitwise (the variant as testreport runs it)."""
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    res = _front(m)
    for it, r in res.items():
        assert set(r) == set(r5.FRONT), (it, sorted(set(r5.FRONT) - set(r)))
        assert not r5.missing(m, it, r), (it, r5.missing(m, it, r))
        assert not r5.bad(r), (it, r5.bad(r))


def test_front_bitwise_planted_control():
    """The same with the CTRL lane's planted nonzero xx_qnet (job27829159-ctrlxx): the control enters Qnet at S03
    and every later stage, bitwise; with the zero run's control records instead, S03 Qnet differs (negative
    control), S02 still bitwise."""
    from mitjax.tests import r5_gate as r5
    m = r5.model("xx")
    res = _front(m, steps=(0, 1, 2, 3))
    for it, r in res.items():
        assert not r5.missing(m, it, r) and not r5.bad(r), (it, r5.missing(m, it, r), r5.bad(r))
    z = r5.model("zero")
    res = _front(m, steps=(0,), eff=z.eff)
    b = r5.bad(res[0])
    assert "S02_load_fields" not in b
    assert b["S03_ctrl_map_forcing"]["Qnet"][1] > 1000, b["S03_ctrl_map_forcing"]


def test_whole_step_bitwise_10_steps():
    """The whole code_ad step -- FRONT, DYNAMICS with the CD scheme (S06), SOLVE_FOR_PRESSURE
    with the real-fresh-water source (S09, and the CG2D scalars of C02), MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY
    (exactConserv, real fresh water), TRACERS_CORRECTION_STEP, DO_FIELDS_BLOCKING_EXCHANGES (S16) and COST_TILE
    (S18) -- bitwise on every dumped field, all 10 steps (teacher-forced; CD_CODE_VARS.h, CTRL_GENARR.h, cost.h
    carried from our own previous step: the dumps do not hold them)."""
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    for it, (r, c) in enumerate(r5.run_steps(m, 10)):
        assert not r5.missing(m, it, r, r5.STEP), (it, r5.missing(m, it, r, r5.STEP))
        assert not r5.bad(r), (it, r5.bad(r))
        assert not {k: v for k, v in c.items() if any(v[1:])}, (it, c)


def test_whole_run_monitor_and_cost():
    """The whole 10-step run through the run driver (Model incl. Model._packages, forward incl. COST_FINAL): every
    %MON record and MONITOR banner identical to the oracle STDOUT (11 blocks), and COST_FINAL's lines -- early fc,
    the per-tile objf_temp_tut / objf_hflux_tut lines, local fc, ` global fc =   6.20023228182337E+00` -- identical
    to the oracle's."""
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import r5_gate as r5
    m, res, o = r5.whole_run("t1x")
    kinds = {"mon": lambda r: "%MON " in r, "banner": lambda r: "MONITOR dynamic field statistics" in r,
             "cost": lambda r: " fc = " in r or "--> objf_" in r}
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    for kind, sel in kinds.items():
        a, b = [r for r in res.records if sel(r)], [r for r in o.raw[k0:] if sel(r)]
        # the oracle's STDOUT continues with GRDCHK_MAIN's perturbed runs (their COST_FINAL blocks): the reference
        # run's records are its first ones
        n = len(a) if kind == "cost" else len(b)
        assert a and a == b[:len(a)] and len(a) == n, (kind, len(a), len(b),
                                                         [(x, y) for x, y in zip(a, b) if x != y][:3])
    assert len([r for r in res.records if kinds["cost"](r)]) == 3 + 2 * 4      # early, 4 tiles x 2, local, global
    assert sum("%MON time_tsnumber" in r for r in res.records) == 11
    assert "(PID.TID 0000.0001)  global fc =   6.20023228182337E+00" in res.records
    del ag


def test_ivdc_fires_in_optim():
    """CALC_IVDC's convective branch is exercised: IVDConvCount = 1 at some points of the oracle's P02 in every
    step (and ours equals it: test_front_bitwise_zero_control); with ivdc_kappa = 0 (calcConvect off) P02's
    IVDConvCount and T13's kappaRk differ."""
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    for it in STEPS:
        n = int(np.count_nonzero(m.ds.field(it, "P02_rho_sigma_ivdc", "IVDConvCount")))
        assert n > 0, it
    p = m.params.replace(static={"ivdc_kappa_ne_0": False})
    res = _front(m, steps=(0,), params=p)
    b = r5.bad(res[0])
    assert b["P02_rho_sigma_ivdc"]["IVDConvCount"][1] > 0
    assert b["T13_temp_impl"]["kappaRk"][1] > 0


def test_set_ref_state_eos():
    """phiRef(2*Nr+1) of SET_REF_STATE (set_ref_state.F:85-100) == the oracle's G00 phiRef, bitwise (the dump holds
    it as a tile-shaped record of 2*Nr+1 levels, every point of a level the same value); pRef4EOS is gated through
    FIND_RHO_2D (P02 rhoInSitu)."""
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    sz = m.cfg.size
    r = np.asarray(m.ds.field(0, "G00_geometry", "phiRef"))
    assert r.shape[1] == 2 * sz.Nr + 1
    ref = r[0, :, sz.OLy, sz.OLx]
    assert np.all(r == ref[None, :, None, None])
    ours = np.asarray(m.params.phiRef.data)
    assert np.array_equal(ours.view(np.uint64), ref.view(np.uint64)), (ours, ref)


# planted errors: (name, first stage that must differ, field)
NEGATIVE = (
    ("forcing_weights_swapped", 1, "S02_load_fields", "Qnet"),
    ("freeze_off", 1, "P02_rho_sigma_ivdc", "rhoInSitu"),    # FREEZE_SURFACE fires from step 1 on (10 points)
    ("surf_pRef_plus_1Pa", 0, "P02_rho_sigma_ivdc", "rhoInSitu"),
    ("gm_Sd_plus_1ulp", 0, "P05_gmredi_tensor", None),
    ("gm_fluxes_off", 0, "T11_temp_gT", "gT_loc"),
    ("gm_calc_diff_off", 0, "T13_temp_impl", "kappaRk"),
)


@pytest.mark.parametrize("name,it,stage,field", NEGATIVE, ids=[n[0] for n in NEGATIVE])
def test_negative_control(name, it, stage, field, monkeypatch):
    from mitjax.model.src.external_fields_load import ForcingPreload
    from mitjax.tests import r5_gate as r5
    m = r5.model("zero")
    kw = {}
    if name == "forcing_weights_swapped":
        kw["pre"] = ForcingPreload(m.pre.slots, m.pre.irec, m.pre.aWght, m.pre.bWght)
    elif name == "freeze_off":
        kw["params"] = m.params.replace(static={"allowFreezing": False})
    elif name == "surf_pRef_plus_1Pa":
        kw["params"] = m.params.replace(traced={"surf_pRef": m.params.surf_pRef + 1.0})
    elif name == "gm_Sd_plus_1ulp":
        kw["gm_patch"] = lambda gm: gm.replace(GM_Sd=np.nextafter(np.float64(gm.GM_Sd), np.inf))
    elif name == "gm_fluxes_off":
        import mitjax.pkg.gmredi.gmredi_rtransport as rt
        import mitjax.pkg.gmredi.gmredi_xtransport as xt
        import mitjax.pkg.gmredi.gmredi_ytransport as yt
        monkeypatch.setattr(xt, "gmredi_xtransport", lambda *a, **k: a[-1])
        monkeypatch.setattr(yt, "gmredi_ytransport", lambda *a, **k: a[-1])
        monkeypatch.setattr(rt, "gmredi_rtransport", lambda *a, **k: a[-1])
    elif name == "gm_calc_diff_off":
        import mitjax.pkg.gmredi.gmredi_calc_diff as cd
        monkeypatch.setattr(cd, "gmredi_calc_diff", lambda *a, **k: a[6])
    res = _front(m, steps=(it,), **kw)[it]
    b = r5.bad(res)
    first = next(s for s in r5.FRONT if s in b)
    assert first == stage, (name, first, b[first])
    if field is None:
        assert any(v[1] > 0 for v in b[stage].values()), (name, b[stage])
    else:
        assert b[stage][field][1] > 0, (name, b[stage])
