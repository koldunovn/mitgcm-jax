"""pkg/monitor gates (plan Task 11, monitor item; lane MON): the %MON text of our MONITOR on the oracle's own state,
character for character against the oracle's STDOUT, for every M1 variant (mitjax/tests/monitor_gate.py says which
dump stage holds the state of each MONITOR call), plus the gfortran MIN/MAX semantics the extrema depend on and the
negative controls (planted summation order, planted format, planted wrong field), each measured to bite.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from mitjax.pkg.monitor import mon_calc_stats_rl as mcs
from mitjax.pkg.monitor import mon_out
from mitjax.ops import fortran_minmax_host as mi
from mitjax.tests import monitor_gate as mg

HERE = Path(__file__).resolve().parent / "fortran_format"


def _variant_ids():
    return [f"{e}/{i}" for e, i in mg.VARIANTS]


def _blocks_report(o, cfg=None, mutate=None, iterations=None):
    """{tsnumber: diffs} over the gated blocks, and the number of records compared."""
    cfg = cfg or mg.make_cfg(o)
    out, n = {}, 0
    for t in iterations or mg.gated_iterations(o):
        ours = mg.ours_block(o, t, cfg=cfg, mutate=mutate)
        theirs = o.block_lines(t)
        n += len(theirs)
        out[t] = mg.compare(ours, theirs)
    return out, n


# ---------------------------------------------------------------------------------------------------------------
# tier 1x: every variant, every gated block

@pytest.mark.parametrize("exp_inp", mg.VARIANTS, ids=_variant_ids())
def test_monitor_blocks_identical(exp_inp):
    """Every gated block of the variant (`jdon` run, plus the later blocks of the `jdon3` run) is identical."""
    blocks = mg.gated_blocks(*exp_inp)
    assert blocks and blocks[0][1] == mg.oracle(*exp_inp).its[0]
    bad, n = {}, 0
    for kind in ("jdon", "jdon3"):
        its = [t for k, t in blocks if k == kind]
        if not its:
            continue
        rep, nk = _blocks_report(mg.oracle(*exp_inp, kind), iterations=its)
        n += nk
        bad.update({(kind, t): d for t, d in rep.items() if d})
    assert n > 0
    assert not bad, f"{exp_inp}: blocks {sorted(bad)} differ, e.g. {next(iter(bad.values()))[:3]}"


@pytest.mark.parametrize("exp_inp", mg.VARIANTS, ids=_variant_ids())
def test_grid_statistics_identical(exp_inp):
    o = mg.oracle(*exp_inp)
    ours = mg.ours_grid_blocks(o)
    loose = o.loose_blocks()
    assert len(loose) == 2 and len(ours) == 2
    for a, b in zip(ours, loose):
        assert not mg.compare(a, b), mg.compare(a, b)[:3]


def test_later_iterations_gated_where_dumped():
    """Which blocks the dumps allow (reported, and kept from shrinking unnoticed): steps 1-3 of the jdon runs, plus
    the later MONITOR steps of the jdon3 runs (block 10 / 16 of the advect variants, 36003 of global_ocean)."""
    got = {f"{e}/{i}": mg.gated_blocks(e, i) for e, i in mg.VARIANTS}
    for v in ("tutorial_barotropic_gyre/input", "tutorial_baroclinic_gyre/input",
              "tutorial_global_oce_optim/input_ad"):
        assert got[v] == [("jdon", t) for t in (0, 1, 2, 3)], (v, got[v])
    assert got["global_ocean.90x40x15/input"] == [("jdon", 36000), ("jdon", 36001), ("jdon", 36002),
                                                  ("jdon3", 36003)]
    for v in ("advect_xy/input.ab3_c4", "advect_xz/input", "advect_xz/input.nlfs", "advect_xz/input.pqm"):
        assert got[v] == [("jdon", 0), ("jdon3", 10)], (v, got[v])     # monitorFreq = 10 steps
    assert got["advect_xy/input"] == [("jdon", 0), ("jdon3", 16)]      # monitorFreq = dumpFreq/deltaT = 16 steps


# ---------------------------------------------------------------------------------------------------------------
# gfortran MIN/MAX semantics (mitjax/ops/fortran_minmax_host.py)

def _hex(b):
    return struct.unpack(">d", bytes.fromhex(b))[0]


def test_minmax_probe_matches_gimple_order():
    """minmax_probe.F: the host MAX/MIN with the winner of each probe statement reproduce every probe line (the
    winner of a site is the source operand of its maxsd/minsd, tools/fortran_minmax_sites.py; for these statements it
    is also the first GIMPLE operand, minmax_probe_gimple.txt)."""
    gim = (HERE / "minmax_probe_gimple.txt").read_text()
    assert "MAX_EXPR <M.7_100, t.14_28>" in gim and "MAX_EXPR <_119, M.9_118>" in gim
    rows = [ln.split() for ln in (HERE / "minmax_probe_gfortran11.out").read_text().splitlines()]
    d = {r[0]: [] for r in rows}
    for r in rows:
        d[r[0]].append((_hex(r[1]), _hex(r[2])))
    for x, got in d["MAXZ"]:            # T = MAX( R(N), zeroRL ): the first argument wins
        assert np.signbit(mi.MAX(x, 0.0, p="a")) == np.signbit(got) and mi.MAX(x, 0.0, p="a") == got
    for x, got in d["MINZ"]:            # T = MIN( RK(N), zeroRL )
        assert np.signbit(mi.MIN(x, 0.0, p="a")) == np.signbit(got) and mi.MIN(x, 0.0, p="a") == got
    t, mx = d["CHAI"][1]                # MX(3) = MAX( MX(3)=+0, T=-0 ) -> +0: the first argument wins
    assert t == 0 and np.signbit(t) and not np.signbit(mx)
    assert not np.signbit(mi.MAX(0.0, -0.0, p="a"))
    v, mx = d["ARR0"][0]                # MX(1) = MAX( MX(1)=+0, V(2)=-0 ) -> -0: the second argument wins
    assert np.signbit(mx) and np.signbit(mi.MAX(0.0, -0.0, p="b"))


def test_minmax_sites_table():
    """The GIMPLE operand order lane MON recorded from the oracle's own sources (minmax_gimple_sites.txt). The winner
    of a site is the source operand of its maxsd/minsd, which can differ from the first GIMPLE operand (mon_advcflw2:
    the K term wins although it is the second GIMPLE operand): the port takes `p` from $MJX_REFERENCE/minmax_sites,
    checked by mitjax/tests/test_minmax_sites.py."""
    sites = [ln.split(":", 1) for ln in (HERE / "minmax_gimple_sites.txt").read_text().splitlines()
             if not ln.startswith("#")]
    by = {}
    for f, expr in sites:
        by.setdefault(f, []).append(expr.strip())
    new = "<tmpval, M>"
    for f in ("mon_calc_stats_rl", "mon_calc_stats_rs", "mon_vort3"):
        assert all(new in e for e in by[f]), (f, by[f])
    assert all(new in e for e in by["mon_ke"] + by["mon_advcfl"] + by["mon_advcflw"])
    assert all("<M, tmpval>" in e for e in by["mon_stats_rs"])
    assert by["monitor"] == ["M = MAX_EXPR <M, >;"]          # p = dTtracerLev(1)
    assert by["mon_advcflw2"] == ["M = MAX_EXPR <, M>;", "M = MAX_EXPR <tmpval, M>;"]
    assert by["mon_calc_advcfl"][-3:] == ["M = MAX_EXPR <, M>;"] * 3


def test_chain_semantics():
    """min_chain/max_chain against an explicit loop of host MIN/MAX, on signed zeros and NaN, both winners."""
    rng = np.random.default_rng(1)
    pool = np.array([0.0, -0.0, 1.0, -1.0, np.nan, 2.0])
    for _ in range(300):
        xs = rng.choice(pool, size=rng.integers(0, 7))
        init = rng.choice(pool)
        for p in ("a", "b"):
            m_min = m_max = np.float64(init)
            for x in xs:
                m_min = mi.MIN(m_min, x, p=p)
                m_max = mi.MAX(m_max, x, p=p)
            for got, want in ((mi.min_chain(init, xs, p=p), m_min),
                              (mi.max_chain(init, xs, p=p), m_max)):
                assert (np.isnan(got) and np.isnan(want)) or (got == want and np.signbit(got) == np.signbit(want)), \
                    (init, xs, p, got, want)


# ---------------------------------------------------------------------------------------------------------------
# negative controls (each measured to bite)

def _ndiff(rep):
    return sum(len(d) for d in rep.values())


def test_planted_tile_order_bites(monkeypatch):
    """GLOBAL_SUM_TILE_RL of the theMean sum of MON_CALC_STATS_RL with the tiles reversed (global_ocean, 36 tiles)."""
    o = mg.oracle("global_ocean.90x40x15", "input")
    real = mcs.global_sum_tile_rl
    calls = {"n": 0}

    def planted(phiTile, ex=None):
        calls["n"] += 1
        if calls["n"] % 5 == 4:             # the 4th of the 5 sums per call: tileMean (:176)
            return real(phiTile[::-1], ex)
        return real(phiTile, ex)

    monkeypatch.setattr(mcs, "global_sum_tile_rl", planted)
    rep, n = _blocks_report(o, iterations=[36000])
    bad = [d for d in rep[36000]]
    assert bad, "reversing the tile order of the mean changed no record"
    assert all("_mean" in a or "_sd" in a for _, a, _ in bad), bad


def test_planted_kji_order_bites(monkeypatch):
    """The per-tile sums as a pairwise (tree) reduction instead of the k, j, i chain (tutorial_barotropic_gyre, one
    tile): the last digits of the means change."""
    import jax.numpy as jnp
    from mitjax.pkg.monitor import mon_ke, mon_vort3

    def tree(contrib):
        a = jnp.asarray(contrib)
        return jnp.sum(a.reshape(a.shape[0], -1), axis=1)

    for mod in (mcs, mon_ke, mon_vort3):
        monkeypatch.setattr(mod, "tile_sums", tree)
    o = mg.oracle("tutorial_barotropic_gyre", "input")
    rep, _ = _blocks_report(o, iterations=[3])
    assert len(rep[3]) >= 1, "a tree reduction changed no record"


def test_planted_format_bites(monkeypatch):
    """MON_OUT_ALL writing 1PE21.12 instead of 1PE21.13: every real-valued record differs."""
    real = mon_out.fortran_write
    monkeypatch.setattr(mon_out, "fortran_write",
                        lambda fmt, *v: real(fmt.replace("E21.13", "E21.12"), *v))
    o = mg.oracle("tutorial_barotropic_gyre", "input")
    rep, n = _blocks_report(o, iterations=[1])
    n_real = sum(1 for ln in o.block_lines(1) if "%MON" in ln and "tsnumber" not in ln)
    assert len(rep[1]) == n_real, (len(rep[1]), n_real)


def test_planted_wrong_field_bites():
    """uVel replaced by vVel in the state MONITOR reads (tutorial_baroclinic_gyre, after step 1)."""
    def swap(*, state, **_):
        state.uVel = state.vVel

    o = mg.oracle("tutorial_baroclinic_gyre", "input")
    rep, _ = _blocks_report(o, iterations=[1], mutate=swap)
    names = {a.split()[3] for _, a, _ in rep[1] if "%MON" in a}
    assert {"dynstat_uvel_max", "ke_mean", "vort_r_min"} <= names, sorted(names)


# ---------------------------------------------------------------------------------------------------------------
# DIFFERENT_MULTIPLE (mitjax/eesupp/different_multiple.py) decides when MONITOR prints

@pytest.mark.parametrize("exp_inp", mg.VARIANTS, ids=_variant_ids())
def test_monitor_schedule_matches_oracle(exp_inp):
    """The iterations at which DIFFERENT_MULTIPLE(monitorFreq, myTime, deltaTClock) holds (monitor.F:48, myTime of
    forward_step.F:808) are exactly the tsnumbers of the oracle's MONITOR blocks."""
    from mitjax.eesupp.different_multiple import different_multiple

    o = mg.oracle(*exp_inp)
    nIter0, nEnd = o.p("nIter0"), o.p("nEndIter")
    start, dtc, freq = o.p("startTime"), o.p("deltaTClock"), o.p("monitorFreq")
    ours = [n for n in range(nIter0, nEnd + 1)
            if different_multiple(freq, start + dtc*float(n - nIter0), dtc)]
    theirs = [int(b.text["time_tsnumber"]) for b in o.blocks if b.kind == "MONITOR dynamic field statistics"]
    assert ours == theirs, (ours, theirs)


def test_different_multiple_cases():
    from mitjax.eesupp.different_multiple import different_multiple

    assert different_multiple(1200.0, 1200.0, 1200.0)          # |step| = freq: the closest-multiple branch
    assert not different_multiple(12000.0, 1200.0, 1200.0)
    assert different_multiple(12000.0, 12000.0, 1200.0)
    assert different_multiple(1.0, 3.1104e9, 86400.0)          # |step| > freq
    assert not different_multiple(0.0, 1.0, 1.0)               # freq = 0
    with pytest.raises(OverflowError):
        different_multiple(1.0, 3.0e9, 0.5)                    # NINT outside INTEGER*4
