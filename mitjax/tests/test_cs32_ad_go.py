"""global_ocean.cs32x15/input_ad forward (plan Task 25, M2 adjoint family), GO lane session 9: the code_ad build's step
through the run driver's Model on the cube (nIter0 = 72000 from a pickup without GuNm2 / GvNm2: CHECK_PICKUP's
mom_StartAB = 1 with ALLOW_ADAMSBASHFORTH_3 momentum; DST3 implicit vertical advection; the generic controls xx_theta,
xx_salt, xx_ptr1, xx_diffkr and xx_qnet, xx_empmr, xx_fu, xx_fv at their zero first guess (CTRL_SIZE.h of the build);
COST_TEST's TSQUARED cost) vs lane A's dumps (helpers: mitjax/tests/cs32_gate.py).

* steps 72000-72002 from the initial carry: every dumped stage bitwise (35 stages incl. S03 CTRL_MAP_FORCING, the
  per-level D00a / D00c, T10-T23 with the DST3 matrices, S18 cost.h);
* the whole run (5 steps) through the driver: every %MON record and MONITOR banner of the oracle's reference forward
  sweep, and COST_FINAL's lines (early fc, the 12 objf_test lines, local fc, global fc = 9.62450968706140E+04)
  identical to the oracle STDOUT.
Costs: about 9 min on a CPU compute node. The adjoint (admGrd, FD, ad_dynstat, trust protocol): scripts/cs32_adjoint.py
(a measurement script, not a gate).
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import cs32_gate as S  # noqa: E402

EXP = ("global_ocean.cs32x15", "input_ad")


@pytest.fixture(scope="module")
def r():
    from mitjax.tests import cube_run_gate as C
    return C.CubeRun(*EXP)


def test_steps_bitwise(r):
    p = r.m.params
    assert p.mom_StartAB == 1 and r.m.cfg.cpp.ALLOW_ADAMSBASHFORTH_3        # CHECK_PICKUP: GuNm2 / GvNm2 missing
    out, ncmp, _ = S.run_compare(r, 3)
    print("cs32x15/input_ad stages compared:", len(ncmp), "fields:", sum(ncmp.values()))
    for it in r.its:
        got = {st for (i, st) in ncmp if i == it}
        assert {"S00_begin", "S03_ctrl_map_forcing", "D00c_mom_vecinv", "S06_dynamics", "T13_temp_impl",
                "T23_salt_impl", "S16_blocking_exchanges", "S18_cost_tile"} <= got, (it, sorted(got))
    assert len(ncmp) >= 105 and {k: v for k, v in out.items() if v} == {}


def test_whole_run_monitor_and_cost_lines():
    from mitjax.tests import cube_run_gate as C
    m, res, o = C.whole_run(*EXP, tag="go-adwhole")
    raw = o.raw
    k0 = next(n for n, x in enumerate(raw) if "Begin MONITOR dynamic field statistics" in x) - 1
    k1 = next(n for n, x in enumerate(raw) if "Start of S/R COST_FINAL" in x)
    sel = lambda x: "%MON " in x or "MONITOR dynamic field statistics" in x       # noqa: E731
    ours, theirs = [x for x in res.records if sel(x)], [x for x in raw[k0:k1] if sel(x)]
    assert len(ours) == len(theirs) > 0 and ours == theirs
    k2 = next(n for n, x in enumerate(raw) if " global fc =" in x)
    ca = [x.rstrip() for x in res.records if "fc =" in x or "objf_" in x]
    cb = [x.rstrip() for x in raw[k1:k2 + 1] if "fc =" in x or "objf_" in x]
    assert len(ca) == 15 and ca == cb, (ca, cb)
    assert ca[-1].endswith("9.62450968706140E+04")
