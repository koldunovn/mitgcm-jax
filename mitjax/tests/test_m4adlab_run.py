"""lab_sea/input_ad (M4 step 7 part 2, lane M4ADLAB session 4): the whole 4-step run of the code_ad build through the
run driver on the real Model (no stubs: pkg/ecco, pkg/ctrl, pkg/cost as data.pkg sets them) against lane A's
dumps-on oracle job27855987-jdon (code_ad forward; its STDOUT, cost and ecco files equal the plain run's).

1. %MON records (every block, steps 0-4) identical; every COST_FINAL line of the oracle in order (four f_gencost:
   theta 3.08813215476166E+03 over 1860, salt 1.06770863781960E+03 over 1860, sst 3.08060340883278E+03 over 150,
   mdt 5.02565637443429E-02 over 115; nine f_gentim2d, two f_genarr2d (xx_siarea, xx_siheff), two f_genarr3d, the
   Writing / Reading lines, early / local / global fc = 7.23649445797779E+03); ECCO_OFFSET's two STDOUT lines;
   costfunction.0000, costfunction_ecco.0000, costfunction_ctrl.0000 byte-identical; the pkg/ecco files
   byte-identical: the bar files m_theta / m_salt / m_sst / m_eta_month.0000000000 (ACTIVE_WRITE_XY(Z), float64),
   misfit_theta / salt / sst / mdt (writeBinaryPrec 32) and weight_mdt (outputlevel 5), per tile (.001.001 ..
   .002.002, .data and .meta); the pickups byte-identical.
   NOTE (handoff, decision 11): the f_gencost values the lane's handoff and the dispatch quoted (theta
   3.08813215476105E+03, ...) are the FIRST PERTURBED run's of the grdchk (fc+ at (6,8)), as TAF's output_adm.txt
   prints them; the reference run's lines are the ones above (oracle output.txt :4357-4360).
2. Negative controls of the 2-D gencost path, each measured on the final carry by replaying COST_FINAL (the final
   COST_AVERAGESFIELDS call, COST_GENCOST_ALL, the finals) with one plant: ECCO_MASKMINDEPTH's topomin -200 -> -100
   (f_gencost mdt and its point count change), 'offset' dropped (f_gencost mdt and misfit_mdt change), the m_sst arm
   reading THETA level 2 at the call after the loop (the m_sst bar record and f_gencost sst change), sum1mon = 4 at
   the call after the loop (every bar record changes), the genarr2d control xx_siheff's weight planted to 0
   (f_genarr2d siheff loses its 150 points).
Costs: about 10 min on a CPU node (one run, five replays of COST_FINAL).
"""

import jax
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("lab_sea", "input_ad")

ECCO_BASE = ("m_theta_month.0000000000", "m_salt_month.0000000000", "m_sst_month.0000000000",
             "m_eta_month.0000000000", "misfit_theta", "misfit_salt", "misfit_sst", "misfit_mdt", "weight_mdt")
TILES = ("001.001", "001.002", "002.001", "002.002")
ECCO_FILES = tuple(f"{b}.{t}.{x}" for b in ECCO_BASE for t in TILES for x in ("data", "meta"))
COST_FILES = ("costfunction.0000", "costfunction_ecco.0000", "costfunction_ctrl.0000")


def _oracle_dir():
    from mitjax import paths as P
    return P.REFERENCE_RUNS / EXP[0] / EXP[1] / "job27855987-jdon" / "rundir"


@pytest.fixture(scope="module")
def whole():
    from mitjax import paths as P
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    exp_dir = P.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    rundir = make_rundir(exp_dir, EXP[1], ag.out_dir("adlab-whole"))
    m = Model(e, rundir)
    res = forward(m)
    jax.clear_caches()
    return m, res, mg.oracle(*EXP)


def _cost_lines(records):
    return [r for r in records if " fc = " in r or "--> f_" in r or "cost function info" in r
            or "ecco_offset:" in r]


def test_whole_run_monitor_cost_ecco_files_pickups(whole):
    from mitjax.tests import advect_gate as ag
    m, res, o = whole
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    print(diffs, nblocks, rows)
    ours = _cost_lines(res.records)
    k1 = next(n for n, r in enumerate(o.raw) if "Gradient-check starts" in r)    # the reference run's block only
    theirs = _cost_lines(o.raw[:k1])
    print("\n".join(ours))
    assert ours == theirs and len(ours) == 27, [(a, b) for a, b in zip(ours, theirs) if a != b] or (ours, theirs)
    assert "(PID.TID 0000.0001)  global fc =   7.23649445797779E+03" in ours
    for v in ("3.08813215476166E+03", "1.06770863781960E+03", "3.08060340883278E+03", "5.02565637443429E-02"):
        assert sum(v in r and "--> f_gencost" in r for r in ours) == 1, v
    od = _oracle_dir()
    bad = [f for f in COST_FILES + ECCO_FILES if not (m.rundir / f).exists()
           or (m.rundir / f).read_bytes() != (od / f).read_bytes()]
    assert not bad, bad
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 5, (diffs, nblocks)
    pk, extra = ag.pickup_diffs(m, o)
    assert pk and all(pk.values()) and not extra, (pk, extra)


def _replay(m, res, plant=None):
    """COST_FINAL replayed on the run's final carry (Model.cost_final: the call of COST_AVERAGESFIELDS after the loop,
    COST_DRIVER, the package finals) with `plant(m)` applied to the Model first and undone after. -> (out dict of
    the package finals, global fc)."""
    import copy
    pk = res.carry[4]
    saved = (m.arrays, copy.copy(m.cost_fixed), copy.copy(m.ecco.__dict__), dict(vars(m)))
    try:
        if plant is not None:
            plant(m)
        out = {}
        cost, loc = m.cost_final(pk["cost"], pk.get("genarr"), state=res.carry[0], out=out, ecco=pk.get("ecco"))
        return out, cost
    finally:
        m.arrays, m.cost_fixed = saved[0], saved[1]
        m.ecco.__dict__.update(saved[2])
        for k in list(vars(m)):
            if k not in saved[3]:
                delattr(m, k)
        vars(m).update(saved[3])


def _gencost(out, k):
    nv, f, no, mlt, name = out["ecco"][k - 1]
    return float(f), float(no)


def test_replay_reproduces_and_negative_controls(whole):
    import jax.numpy as jnp
    m, res, _ = whole
    base, _ = _replay(m, res)
    ref = {k: _gencost(base, k) for k in (1, 2, 3, 4)}
    print("replay", ref)
    assert ref[4][1] == 115.0 and ref[3][1] == 150.0, ref
    rec0 = {k: np.asarray(v[0].data) for k, v in base["ecco_files"]["rec"].items()}

    def setup_plant(k, **pp):
        def plant(mm):
            st = []
            for s in mm.ecco.setup:
                if s["g"].k == k:
                    s = dict(s, pp=dict(s["pp"], **pp))
                st.append(s)
            mm.ecco.setup = st
        return plant
    out, _ = _replay(m, res, setup_plant(4, topomin=-100.0))                     # ECCO_MASKMINDEPTH
    print("topomin -100", _gencost(out, 4))
    assert _gencost(out, 4)[1] != ref[4][1] and _gencost(out, 4)[0] != ref[4][0]
    out, _ = _replay(m, res, setup_plant(4, dooffset=False))                     # ECCO_OFFSET
    print("no offset", _gencost(out, 4))
    assert _gencost(out, 4)[0] != ref[4][0]
    assert not np.array_equal(np.asarray(out["ecco_files"]["mis"][4][0][1]),
                              np.asarray(base["ecco_files"]["mis"][4][0][1]))

    def sum1_plant(mm):                                                          # sum1mon at the call after the loop
        a = mm.arrays
        tab = {k: (b, s.at[-1].set(4), r) for k, (b, s, r) in a.pkc["ecco_tab"].items()}
        mm.arrays = a.replace(pkc=dict(a.pkc, ecco_tab=tab))
    out, _ = _replay(m, res, sum1_plant)
    for k in (1, 2, 3, 4):
        assert not np.array_equal(np.asarray(out["ecco_files"]["rec"][k][0].data), rec0[k]), k

    import mitjax.pkg.ecco.cost_averagesfields as CA
    real = CA.cost_gencost_customize

    def level2(g, *, theta, salt, maskC, sz, m_eta=None):                        # m_sst reads THETA(:,:,2)
        if CA.customize_arm(g) == "m_sst":
            from mitjax.farray import loops_kji
            k, j, i = loops_kji((2, 2), (1, sz.sNy), (1, sz.sNx))
            return theta[i, j, k] * maskC[i, j, k]
        return real(g, theta=theta, salt=salt, maskC=maskC, sz=sz, m_eta=m_eta)
    CA.cost_gencost_customize = level2
    try:
        out, _ = _replay(m, res)
    finally:
        CA.cost_gencost_customize = real
    assert not np.array_equal(np.asarray(out["ecco_files"]["rec"][3][0].data), rec0[3])
    assert _gencost(out, 3)[0] != ref[3][0] and _gencost(out, 1) == ref[1]

    def w0_plant(mm):                                                            # wgenarr2d of xx_siheff = 0
        from mitjax.farray import FArray
        w = mm.genarr_w_in[(2, 2)]
        mm.genarr_w_in = {**mm.genarr_w_in, (2, 2): FArray(jnp.zeros_like(w.data), w.name, tiled=w.tiled,
                                                           _dims=w.dims)}
    nb = [x for x in base["ctrl"]["arr2d"] if x[0] == 2][0]
    out, _ = _replay(m, res, w0_plant)
    na = [x for x in out["ctrl"]["arr2d"] if x[0] == 2][0]
    print("siheff", nb, na)
    assert float(nb[2]) == 150.0 and float(na[2]) == 0.0


@pytest.mark.parametrize("nproc", [4, 2])
def test_sharded_equals_single(whole, nproc):
    """The 4 steps of the driver's step on the real Model (State, FFIELDS.h, the package state incl. SEAICE.h, EXF,
    KPP / GMREDI / DOWN_SLOPE, cost.h, CTRL_GENARR.h and the ECCO.h averages / bar records) at P=nproc (TileSharding
    over the 4 tiles of 10x8, jit(shard_map(check_vma=True))) == P=1, every leaf bitwise (m4lab_gate.sharded_vs_single;
    lab_sea's registered map lab_sea-t4_10x8_ol4x4.npz)."""
    from mitjax.tests import m4lab_gate as L
    m = whole[0]
    assert L.sharded_vs_single(m, nproc, 4) == {}
