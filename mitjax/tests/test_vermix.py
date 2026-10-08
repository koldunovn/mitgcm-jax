"""vermix end to end (M3 Task 30, vermix lane): the run driver's Model on lane A's dumps-on runs.

For every variant: the initial State vs I02_ini_fields and the INITIALISE_FIXED grid vs G00_geometry (bitwise); the
first three steps with every dumped stage probed and compared on every point of every tile, halos included, bit
patterns too (P01-P11, T-, D-, C-, S-stages); the whole run through the driver: every %MON record and banner identical
to the oracle STDOUT, testreport digits vs results/ >= the yardstick (and vs the oracle), the pickup files byte-identical,
nothing on STDERR. vermix is one tile: P=N == P=1 has no N > 1 to run.
input.dd (LINEAR EOS, KPP + double diffusion) and the MDJWF variants input (KPP), .ggl90, .my82, .opps, .pp81 run
(FIND_ALPHA / FIND_BETA 'MDJWF' of lane EOSAB, wired at the session 3 merge). input.gglLC stops at DYNAMICS'
GGL90_ADD_STOKESDRIFT (useLANGMUIR, mom_fluxform.F:1083-1088), not ported: a strict xfail that turns into a failure
when it lands, and the mark goes.
Costs: about 3 min per variant on a CPU node.
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

LANGMUIR = pytest.mark.xfail(strict=True, raises=NotImplementedError,
                             reason="DYNAMICS: GGL90_ADD_STOKESDRIFT (useLANGMUIR; pkg/mom_fluxform/mom_fluxform.F:"
                                    "1083-1088, pkg/ggl90/ggl90_add_stokesdrift.F) not ported")
VARIANTS = ["input.dd", "input", "input.ggl90", "input.my82", "input.opps", "input.pp81",
            pytest.param("input.gglLC", marks=LANGMUIR)]


ALL = ["input.dd", "input", "input.ggl90", "input.gglLC", "input.my82", "input.opps", "input.pp81"]


@pytest.mark.parametrize("inp", ALL)
def test_initial_state_and_grid_bitwise(inp):
    """Every variant: INI_PRESSURE's storePhiHyd4Phys iteration (MDJWF variants) included (totPhiHyd of I02)."""
    from mitjax.tests import vermix_gate as V
    bad, n = V.initial_bad(V.run(inp))
    assert n > 50 and not bad, (n, bad)


@pytest.mark.parametrize("inp", VARIANTS)
def test_steps_0_2_every_dumped_stage_bitwise(inp):
    from mitjax.tests import vermix_gate as V
    r = V.run(inp)
    bad, ncmp, _ = V.steps(r, 3)
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    miss = V.missing_stages(r, ncmp, 3)
    assert not any(miss.values()), miss
    assert all((it, "P07_kpp") in ncmp for it in r.its[:3]) == (inp in ("input", "input.dd"))


@pytest.mark.parametrize("inp", VARIANTS)
def test_whole_run_monitor_digits_pickups(inp):
    from mitjax.tests import vermix_gate as V
    m, res, o, (diffs, rows, nblocks), (files, missing) = V.whole_run(inp)
    print(f"vermix/{inp} digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert nblocks == 21 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert len(rows) == 17 and all(a >= y for _, a, y, _ in rows), rows
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert pk and all(pk.values()), pk
    assert res.stderr == [], res.stderr[:5]
