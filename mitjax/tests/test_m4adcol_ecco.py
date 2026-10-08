"""Lane M4ADCOL (M4 step 7, session 3): pkg/ecco of 1D_ocean_ice_column/input_ad (code_ad build) against the oracle
job27856057-jdon -- the gencost path as the run executes it (mitjax/pkg/ecco: ECCO_READPARMS, ECCO_COST_INIT_FIXED,
COST_AVERAGESFIELDS at every step start and after the loop, COST_GENCOST_ALL -> COST_GENERIC -> COST_GENLOOP,
ECCO_COST_FINAL), through the run driver with useECCO as data.pkg sets it:
  * the bar files m_theta_month.0000000000 / m_salt_month.0000000000 (.data float64, .meta) byte-identical;
  * the misfit files misfit_theta / misfit_salt (.data float32 with the -0. of the masked points, .meta)
    byte-identical;
  * the f_gencost lines (2.05294994115046E-02 over 20 points, 2.70356960765386E-03 over 23) and costfunction_ecco.0000
    byte-identical; the averaging table of the ten steps (first call ASSIGN, nine ACCUMULATE, after the loop DIVIDE by
    sum1mon = 11, record 1).
Negative controls (measured, each a whole run on the planted Model): mult_gencost = 1 (fc gains f_gencost), the data
levels shifted by one (f_gencost and misfits change), sum1mon = 10 at the final call (the bar files change). Blind
spot asserted: the 13 monthly data records are identical, so COST_GENCAL's record choice cannot show (a planted
record 2 left every line unchanged, dev job 27880878); its host arithmetic is checked directly. Costs: about 10 min on
a CPU node (four runs).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import adcol_gate as A  # noqa: E402

ECCO_FILES = ("m_theta_month.0000000000.data", "m_theta_month.0000000000.meta", "m_salt_month.0000000000.data",
              "m_salt_month.0000000000.meta", "misfit_theta.data", "misfit_theta.meta", "misfit_salt.data",
              "misfit_salt.meta", "costfunction_ecco.0000")


def _run(tag, plant=None):
    """A whole run of input_ad through the run driver on a fresh run directory -> (Model, records, {file: bytes})."""
    import jax
    from mitjax import paths as P
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    exp_dir = P.UPSTREAM / "verification" / A.EXP[0]
    e = load_experiment(exp_dir, A.EXP[1])
    rundir = make_rundir(exp_dir, A.EXP[1], ag.out_dir(f"adcol-ecco-{tag}"))
    m = Model(e, rundir)
    if plant is not None:
        plant(m)
    res = forward(m, write_pickups=False)
    files = {f: (rundir / f).read_bytes() for f in ECCO_FILES if (rundir / f).exists()}
    jax.clear_caches()
    return m, res.records, files


def _oracle_dir():
    from mitjax import paths as P
    return P.REFERENCE_RUNS / A.EXP[0] / A.EXP[1] / "job27856057-jdon" / "rundir"


def _lines(records, key):
    return [r for r in records if key in r]


@pytest.fixture(scope="module")
def base():
    return _run("base")


def test_averaging_table(base):
    from mitjax.pkg.ecco.cost_averagesfields import ACCUMULATE, ASSIGN, DIVIDE
    m = base[0]
    for k in (1, 2):
        t = m.ecco.table[k]
        assert list(t["branch"]) == [ASSIGN] + [ACCUMULATE]*9 + [DIVIDE], t
        assert list(t["sum1"]) == [0]*10 + [11] and t["rec"][-1] == 1 and t["writes"] == [(11, 1)], t
        assert t["nrec"] == 1


def test_ecco_files_byte_identical(base):
    _, _, files = base
    od = _oracle_dir()
    bad = [f for f in ECCO_FILES if files.get(f) != (od / f).read_bytes()]
    assert not bad, bad


def test_f_gencost_lines(base):
    _, recs, _ = base
    from mitjax.tests import monitor_gate as mg
    o = mg.oracle(*A.EXP)
    ours, theirs = _lines(recs, "--> f_gencost"), _lines(o.raw, "--> f_gencost")
    print("\n".join(ours))
    assert ours == theirs and len(ours) == 2, (ours, theirs)
    assert "2.05294994115046E-02" in ours[0] and "2.70356960765386E-03" in ours[1]
    cf = base[2]["costfunction_ecco.0000"].decode().splitlines()
    assert "2.00000000000000E+01" in cf[0] and "2.30000000000000E+01" in cf[1], cf


def test_negative_control_mult_gencost(base):
    """mult_gencost(1:2) = 1 planted (ECCO_COST_FINAL's tile_fc term): global fc gains f_gencost(1) + f_gencost(2)."""
    import jax.numpy as jnp

    def plant(m):
        m.cost_fixed["ecco"]["mult"] = {k: jnp.float64(1.0) if k in (1, 2) else v
                                        for k, v in m.cost_fixed["ecco"]["mult"].items()}
    _, recs, _ = _run("mult", plant)
    f0 = float(_lines(base[1], " global fc = ")[0].split("=")[1])
    f1 = float(_lines(recs, " global fc = ")[0].split("=")[1])
    print(f0, f1, f1 - f0)
    assert f1 != f0
    assert abs((f1 - f0) - (2.05294994115046E-02 + 2.70356960765386E-03)) < 1e-9


def test_record_blind_spot_and_level_shift_control(base):
    """COST_GENCAL's record choice (monthly fields: localrec = 1 + MOD(modm-1+irec-1, 12), cost_gencal.F:100-106) is
    invisible in this run: the 13 records of t_ref_1x1x23x13 / s_ref_1x1x23x13 are identical (measured; a planted
    record 2 cannot change anything), so the host arithmetic is checked directly (irec = 2 -> localrec 2, the cyclic
    file name). The data path itself is controlled by a level shift: obs(k) = file level k+1 (level Nr kept):
    f_gencost and the misfits change, the bar files do not."""
    import jax.numpy as jnp

    from mitjax.pkg.ecco.cost_gencost_all import cost_gencal
    m = base[0]
    sz = m.cfg.size
    for name in ("t_ref_1x1x23x13", "s_ref_1x1x23x13"):
        a = np.fromfile(m.rundir / name, ">f4").reshape(13, sz.Nr)
        assert all(np.array_equal(a[0], a[r]) for r in range(13)), name
    g = m.ecco.ep.gencost[0]
    f1, f2, lrec, orec, exst = cost_gencal(g.gencost_barfile, g.gencost_datafile, 2, g.gencost_startdate,
                                          g.gencost_period, eccoiter=0, cal=m.cal, dTtracerLev1=3600.0,
                                          rundir=m.rundir)
    assert (f1, f2, lrec, orec, exst) == ("m_theta_month.0000000000", "t_ref_1x1x23x13", 2, 2, True)

    def plant(mm):
        for k in (1, 2):
            o = np.asarray(mm.cost_fixed["ecco"]["obs"][k][0])
            o2 = o.copy()
            o2[:, :-1] = o[:, 1:]
            mm.cost_fixed["ecco"]["obs"][k] = [jnp.asarray(o2)]
    _, recs, files = _run("levshift", plant)
    a, b = _lines(base[1], "--> f_gencost"), _lines(recs, "--> f_gencost")
    print(a, b)
    assert a[0] != b[0] and a[1] != b[1]
    assert files["misfit_theta.data"] != base[2]["misfit_theta.data"]
    assert files["m_theta_month.0000000000.data"] == base[2]["m_theta_month.0000000000.data"]


def test_negative_control_sum1(base):
    """sum1mon = 10 at the call after the loop (one step short): the bar files change."""
    import jax.numpy as jnp

    def plant(m):
        a = m.arrays
        tab = {k: (b, s.at[-1].set(10), r) for k, (b, s, r) in a.pkc["ecco_tab"].items()}
        m.arrays = a.replace(pkc=dict(a.pkc, ecco_tab=tab))
        assert int(jnp.asarray(tab[1][1])[-1]) == 10
    _, recs, files = _run("sum1", plant)
    for f in ("m_theta_month.0000000000.data", "m_salt_month.0000000000.data"):
        assert files[f] != base[2][f], f
