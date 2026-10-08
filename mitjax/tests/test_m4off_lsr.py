"""offline_exf_seaice/input.dyn_lsr (M4 step 4, lane M4OFF session 3): the C-grid sea-ice dynamics with the LSR solver,
gated bitwise against lane A's dumps-on run job27855986-jdon (iterations 0-2, every point incl. halos), each with a
measured negative control or an asserted blind spot.

1. SEAICE_LSR (Picard passes with SEAICEuseLSRflex, the Gauss-Seidel line sweeps, SEAICE_RESIDUAL, the FLEX_FACTOR
   criterion) teacher-forced from Y04_solver_inputs -> Y06_lsr, and its STDOUT lines (5 per pass, 20 passes) equal to
   the oracle's, iterations 0-2.
2. SEAICE_MODEL from the oracle's fields before I00_seaice_begin: every sea-ice stage Y01, Y02 (forcing set-up, ice
   strength), Y04, Y06, Y09 (ocean stress with ice velocities), I01, I02 (SEAICE_ADVDIFF with the PPM scheme 41: the
   advection blind in input.thermo is live here), I03, P13 (the no-thermodynamics arm: sIceLoad).
3. glibc's log10 (LSRflexFac, seaice_lsr.F:754) transcribed in mitjax/ops/libm.py: bitwise /lib64/libm-2.28.so.
4. The LSR solve is forward only (its derivative is Nikolay's decision): JVP and VJP through it raise
   NotImplementedError.
"""

import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import m4off_gate as G
from mitjax.tests import m4off_lsr_gate as L

ITS = (0, 1, 2)
STAGES = ("Y01_get_dynforcing", "Y02_ice_strength", "Y04_solver_inputs", "Y06_lsr", "Y09_ocean_stress",
          "I01_dynsolver", "I02_advdiff", "I03_reg_ridge", "P13_seaice_model")


@pytest.fixture(scope="module")
def md():
    return L.model()


@pytest.fixture(scope="module")
def lsr(md):
    m, ds = md
    return L.run_lsr(m, ds, its=ITS)


def test_lsr_bitwise(lsr):
    for it in ITS:
        d, _ = lsr[it]
        assert len(d) >= 18, (it, sorted(d))
        assert not {k: v for k, v in d.items() if v}, (it, d)


def test_lsr_stdout_lines(lsr):
    for it in ITS:
        _, out = lsr[it]
        mine, ref = L.lsr_lines(out), L.oracle_lsr_lines(it)
        assert len(ref) == 100 and mine == ref, (it, len(mine), [(a, b) for a, b in zip(mine, ref) if a != b][:4])


def test_lsr_negative_control_relax(md):
    """SEAICE_LSRrelaxU x (1 + 1e-12): Y06 UICE differs (measured, dev job 27870971); the printed lines do not
    resolve it (E12.6 / 1PE16.8: the same 100 lines), so the field gate is the sensitive one."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    sp2 = sp.replace(SEAICE_LSRrelaxU=float(sp.SEAICE_LSRrelaxU)*(1.0 + 1e-12))
    res = L.run_lsr(m, ds, its=(0,), sp=sp2)
    d, out = res[0]
    assert d["UICE"] > 1000, d


def test_lsr_negative_control_ncheck(md):
    """SOLV_NCHECK 3 instead of 2: the convergence tests fall on other sweeps, the iteration counts of the STDOUT
    lines and UICE / VICE differ (measured)."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    res = L.run_lsr(m, ds, its=(0,), sp=sp.replace(SOLV_NCHECK=3))
    d, out = res[0]
    assert d["UICE"] > 1000 and d["VICE"] > 1000, d
    assert L.lsr_lines(out) != L.oracle_lsr_lines(0)


@pytest.fixture(scope="module")
def model_res(md):
    m, ds = md
    return G.run_seaice_model(m, ds, its=ITS)


def test_seaice_model_dyn_lsr_bitwise(model_res):
    for it in ITS:
        r = model_res[it]
        assert set(STAGES) <= set(r), (it, sorted(r))
        bad = {st: {k: v for k, v in d.items() if v} for st, d in r.items()}
        assert not {st: d for st, d in bad.items() if d}, (it, bad)
        for st in ("Y02_ice_strength", "Y06_lsr", "I02_advdiff", "P13_seaice_model"):
            assert len(r[st]) >= 6, (it, st, r[st])


def test_seaice_model_negative_control_strength(md):
    """SEAICE_strength x (1 + 1e-12): Y02 PRESS0 / SEAICE_zMax differ (the ice strength gate bites). Measured blind
    spot at iteration 0: SEAICE_cStar cannot bite there (AREA = 1 everywhere, AreaFile const100: EXP(-cStar*0) = 1,
    dev job 27870971)."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    r = G.run_seaice_model(m, ds, its=(0,), sp=sp.replace(SEAICE_strength=float(sp.SEAICE_strength)*(1.0 + 1e-12)))[0]
    assert r["Y02_ice_strength"]["PRESS0"] > 100 and r["Y02_ice_strength"]["SEAICE_zMax"] > 100, r["Y02_ice_strength"]
    r = G.run_seaice_model(m, ds, its=(0,), sp=sp.replace(SEAICE_cStar=float(sp.SEAICE_cStar)*(1.0 + 1e-12)))[0]
    assert not r["Y02_ice_strength"]["PRESS0"], r["Y02_ice_strength"]          # the blind spot, asserted


def test_seaice_model_negative_control_advection(md):
    """SEAICEadvScheme 77 instead of 41 (all three fields): I02 HEFF and AREA differ (the advection gate, blind in
    input.thermo, bites here: uIce /= 0)."""
    m, ds = md
    sp = m.arrays.pkc["sp"]
    sp2 = sp.replace(SEAICEadvSchHeff=77, SEAICEadvSchArea=77, SEAICEadvSchSnow=77)
    r = G.run_seaice_model(m, ds, its=(1,), sp=sp2)[1]
    assert r["I02_advdiff"]["HEFF"] > 100 and r["I02_advdiff"]["AREA"] > 100, r["I02_advdiff"]
    assert not r["Y06_lsr"]["UICE"], r["Y06_lsr"]


def test_glibc_log10_bitwise():
    from mitjax.ops.libm import glibc_log10
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    libm = ctypes.CDLL("/lib64/libm-2.28.so")
    libm.log10.restype, libm.log10.argtypes = ctypes.c_double, [ctypes.c_double]
    rng = np.random.default_rng(0)
    f = jax.jit(glibc_log10)
    nd_xla = 0
    for lo, hi in ((1e-30, 1e-10), (1e-10, 1e-3), (1e-3, 10.), (0.5, 2.), (10., 1e30)):
        x = np.exp(rng.uniform(np.log(lo), np.log(hi), 100000))
        g = np.array([libm.log10(v) for v in x])
        assert np.array_equal(np.asarray(f(x)).view(np.int64), g.view(np.int64)), (lo, hi)
        nd_xla += np.count_nonzero(np.asarray(jax.jit(jnp.log10)(x)).view(np.int64) != g.view(np.int64))
    assert nd_xla > 10000, nd_xla          # control: XLA's log10 is not glibc's (measured 16 % of the arguments)


def test_lsr_forward_only(md):
    """Any derivative through SEAICE_LSR raises NotImplementedError (the LSR derivative is Nikolay's decision)."""
    m, ds = md
    pkc = m.arrays.pkc
    sf, ff, exf, st = G.inputs_at(m, ds, 0, "Y06_lsr")
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only
    myTime, myIter = L.clock(m, 0)

    def J(heff):
        sf2 = {**sf, "HEFF": type(sf["HEFF"])(heff, "HEFF", tiled=True, _dims=sf["HEFF"].dims)}
        out, _ = seaice_lsr_forward_only(myTime, myIter, sf2, cfg=m.cfg, sp=pkc["sp"], op=pkc["op"],
                                         grid=m.arrays.grid, state=st, ex=m.ex)
        return jnp.sum(out["UICE"].data)
    h = sf["HEFF"].data
    with pytest.raises(NotImplementedError, match="SEAICE_LSR"):
        jax.grad(J)(h)
    with pytest.raises(NotImplementedError, match="SEAICE_LSR"):
        jax.jvp(J, (h,), (jnp.ones_like(h),))
