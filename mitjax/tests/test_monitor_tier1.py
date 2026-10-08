"""pkg/monitor: the one compact tier-1 gate (plan Task 11, lane MON; the full gates are in test_monitor.py, tier 1x).

tutorial_barotropic_gyre: our MONITOR on the oracle's own state (mitjax/tests/monitor_gate.py) writes the oracle's
STDOUT records character for character.
"""

from mitjax.tests import monitor_gate as mg


def _blocks_report(o, iterations):
    cfg = mg.make_cfg(o)
    out, n = {}, 0
    for t in iterations:
        theirs = o.block_lines(t)
        n += len(theirs)
        out[t] = mg.compare(mg.ours_block(o, t, cfg=cfg), theirs)
    return out, n



def test_monitor_barotropic_gyre_tier1():
    """tutorial_barotropic_gyre: the initial block, the block after step 1, and both grid-statistics blocks are the
    oracle's records character for character (incl. its -0.0 trAdv_CFL_w_max at iteration 1)."""
    o = mg.oracle("tutorial_barotropic_gyre", "input")
    rep, n = _blocks_report(o, iterations=[0, 1])
    assert n == 2 * 57
    assert all(not d for d in rep.values()), rep
    assert "-0.0000000000000E+00" in o.block_lines(1)[37]
    grid = mg.ours_grid_blocks(o)
    loose = o.loose_blocks()
    assert [len(b) for b in loose] == [72, 12]
    assert all(not mg.compare(a, b) for a, b in zip(grid, loose))
