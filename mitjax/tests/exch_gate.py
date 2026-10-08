"""Exchange gate helpers shared by test_exchange.py, test_exchange_controls.py and test_sharded_exchange.py (not a test
file). The oracle: lane A's exchange-probe dumps (plan Task 4, run job 27826873; reference/jaxdump/jaxdump.F
JAXDUMP_EXCH_PROBE), one per M1 layout."""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.eesupp import exch_maps as EM
from mitjax.io.dump import DumpSet

# probe outputs -> the Fortran call, as jaxdump.F:913-981 makes it (3-D routines on [tile, k=1, j, i] arrays)
ROUTINES = (
    (("xT",), lambda ex, u, v: (ex.EXCH_XY_RL(u),)),
    (("xUVs_u", "xUVs_v"), lambda ex, u, v: ex.EXCH_UV_XY_RL(u, v, True)),
    (("xUVn_u", "xUVn_v"), lambda ex, u, v: ex.EXCH_UV_XY_RL(u, v, False)),
    (("xZ",), lambda ex, u, v: (ex.EXCH_Z_3D_RL(u[:, None])[:, 0],)),
    (("xAs_u", "xAs_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_AGRID_3D_RL(u[:, None], v[:, None], True))),
    (("xAn_u", "xAn_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_AGRID_3D_RL(u[:, None], v[:, None], False))),
    (("xBs_u", "xBs_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_BGRID_3D_RL(u[:, None], v[:, None], True))),
    (("xBn_u", "xBn_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_BGRID_3D_RL(u[:, None], v[:, None], False))),
    (("x3D",), lambda ex, u, v: (ex.EXCH_3D_RL(u[:, None])[:, 0],)),
    (("xSMs",), lambda ex, u, v: (ex.EXCH_SM_3D_RL(u[:, None], True)[:, 0],)),
    (("xDs_u", "xDs_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_DGRID_3D_RL(u[:, None], v[:, None], True))),
    (("xUV3s_u", "xUV3s_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_3D_RL(u[:, None], v[:, None], True))),
    (("xS3D",), lambda ex, u, v: (ex.EXCH_S3D_RL(u[:, None])[:, 0],)),
)
# signed-zero probes (jaxdump.F:1003-1027): the same routines on fields of +0 (zp*) and -0 (zm*)
ZERO_ROUTINES = (
    ("UVs", ("_u", "_v"), lambda ex, u, v: ex.EXCH_UV_XY_RL(u, v, True)),
    ("As", ("_u", "_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_AGRID_3D_RL(u[:, None], v[:, None], True))),
    ("Bs", ("_u", "_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_BGRID_3D_RL(u[:, None], v[:, None], True))),
    ("Ds", ("_u", "_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_DGRID_3D_RL(u[:, None], v[:, None], True))),
    ("UV3s", ("_u", "_v"), lambda ex, u, v: _lev0(ex.EXCH_UV_3D_RL(u[:, None], v[:, None], True))),
    ("SMs", ("_u",), lambda ex, u, v: (ex.EXCH_SM_3D_RL(u[:, None], True)[:, 0],)),
)


def _lev0(uv):
    return tuple(a[:, 0] for a in uv)


def probe_dumps(exp):
    inp, job = EM.M1_PROBES[exp]
    d = paths.REFERENCE_RUNS / exp / inp / f"{job}-jdon" / "dumps"
    assert d.is_dir(), f"missing probe dumps {d}"
    ds = DumpSet(d)
    return ds, ds.iterations()[0]


def bits(x):
    return np.asarray(x, np.float64).view(np.int64)


def probe_mismatches(ex, ds, it, apply=None):
    """{probe field: points where the exchanger's output differs bitwise from the Fortran probe} for every routine and
    signed-zero probe. apply(fn, ex, u, v) runs one routine (default: jit with the exchanger as an argument)."""
    apply = apply or (lambda fn, ex, u, v: jax.jit(fn)(ex, u, v))
    L = ex.layout
    cbase = ds.scalar(it, EM.STAGE, "xCbase")
    pu, pv = (jnp.asarray(x) for x in EM.probe_inputs(L, cbase))
    out = {}
    for flds, fn in ROUTINES:
        got = apply(fn, ex, pu, pv)
        for f, g in zip(flds, got):
            out[f] = int(np.sum(bits(g) != bits(ds.field(it, EM.STAGE, f)[:, 0])))
    for name, sufs, fn in ZERO_ROUTINES:
        for pm, z in (("p", 0.0), ("m", -0.0)):
            zz = jnp.full(L.shape2d, z)
            got = apply(fn, ex, zz, zz)
            for suf, g in zip(sufs, got):
                f = f"z{pm}{name}{suf}"
                out[f] = int(np.sum(bits(g) != bits(ds.field(it, EM.STAGE, f)[:, 0])))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# mixed signed-zero probes (lane A, jaxdump3 runs: reference/jaxdump/jaxdump.F, `zu*`: u = -0, v = +0; `zv*`: u = +0,
# v = -0) through the vector exchanges with signs; one run per experiment layout
MIXED = ("UVs", "As", "Bs", "Ds", "UV3s")
MIXED_ROUTINES = {"UVs": lambda ex, u, v: ex.EXCH_UV_XY_RL(u, v, True),
                  "As": lambda ex, u, v: _lev0(ex.EXCH_UV_AGRID_3D_RL(u[:, None], v[:, None], True)),
                  "Bs": lambda ex, u, v: _lev0(ex.EXCH_UV_BGRID_3D_RL(u[:, None], v[:, None], True)),
                  "Ds": lambda ex, u, v: _lev0(ex.EXCH_UV_DGRID_3D_RL(u[:, None], v[:, None], True)),
                  "UV3s": lambda ex, u, v: _lev0(ex.EXCH_UV_3D_RL(u[:, None], v[:, None], True))}
JD3_PROBES = {"advect_xy": "input", "advect_xz": "input", "global_ocean.90x40x15": "input"}


def jd3_probe_dumps(exp):
    """(DumpSet, first iteration) of the registered jdon3 run of the experiment (reference/reference_runs.py)."""
    from mitjax.tests.grid_gate import _registry
    rr = _registry()
    run = rr.find_run(exp, JD3_PROBES[exp], "jdon3")
    ds = DumpSet(rr.run_top(run) / "dumps")
    return ds, ds.iterations()[0]


def mixed_zero_mismatches(ex, ds, it, apply=None):
    """{probe field: points differing bitwise} of the mixed signed-zero probes zu*/zv* (every point of the level)."""
    apply = apply or (lambda fn, ex, u, v: jax.jit(fn)(ex, u, v))
    L = ex.layout
    out = {}
    for g in MIXED:
        for tag, (zu, zv) in (("zu", (-0.0, 0.0)), ("zv", (0.0, -0.0))):
            got = apply(MIXED_ROUTINES[g], ex, jnp.full(L.shape2d, zu), jnp.full(L.shape2d, zv))
            for c, a in zip("uv", got):
                f = f"{tag}{g}_{c}"
                out[f] = int(np.sum(bits(a) != bits(ds.field(it, EM.STAGE, f)[:, 0])))
    return out


def rx2_apply_numpy(passes, swap, u, v):
    """exchange.apply_rx2 in numpy (host replay of the tables; the gates' reference)."""
    a1, a2 = (v, u) if swap else (u, v)
    for p in passes:
        outs = []
        for c, own in ((1, a1), (2, a2)):
            src, written, sa1, sa2 = p[c]
            with np.errstate(invalid="ignore"):
                val = np.asarray(sa1, np.float64) * a1[..., src] + np.asarray(sa2, np.float64) * a2[..., src]
            outs.append(np.where(written, val, own))
        a1, a2 = outs
    return (a2, a1) if swap else (a1, a2)


def rx2_transpose_numpy(passes, swap, ybar_u, ybar_v):
    """The transpose of an exch2 C-grid vector exchange, derived by hand (not JAX's): one pass is
        y_c[p] = written_c[p] ? sa1_c[p]*x1[src_c[p]] + sa2_c[p]*x2[src_c[p]] : x_c[p]      (c = 1, 2)
    so, with cotangents ybar_c,
        xbar_c[q] = (not written_c[q]) * ybar_c[q]
                    + sum over c' and the points p with written_c'[p] and src_c'[p] = q of sa_c(c')[p] * ybar_c'[p]
    where sa_1(c') = sa1_c', sa_2(c') = sa2_c' -- the zero coefficients included: they put 0*ybar (a signed zero, or
    NaN where ybar is Inf/NaN) into the other component's source points, which a copy's transpose never touches.
    Passes in reverse order (E = E2 E1, E^T = E1^T E2^T); swap: array1 is v."""
    b1, b2 = (ybar_v, ybar_u) if swap else (ybar_u, ybar_v)
    for p in reversed(passes):
        x1 = np.where(p[1][1], 0.0, b1)
        x2 = np.where(p[2][1], 0.0, b2)
        for c, bc in ((1, b1), (2, b2)):
            src, written, sa1, sa2 = p[c]
            idx = np.nonzero(written)[0]
            with np.errstate(invalid="ignore"):
                np.add.at(x1, src[idx], np.asarray(sa1, np.float64)[idx] * bc[idx])
                np.add.at(x2, src[idx], np.asarray(sa2, np.float64)[idx] * bc[idx])
        b1, b2 = x1, x2
    return (b2, b1) if swap else (b1, b2)
