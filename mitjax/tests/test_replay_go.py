"""GAD_DST2U1_IMPL_R and SOLVE_PENTADIAGONAL vs gfortran (M1 lane GO, session 4): the replay harness
reference/replay_go (THE_MAIN_LOOP replaced in a genmake2 build of the experiment's code/ dir, the replayed .f byte-
identical to the oracle build's) on synthetic inputs over the real grids of global_ocean.90x40x15 and advect_xz; runs
named by reference/replay_go/CURRENT (relative to $MJX_RUNS).

* GAD_DST2U1_IMPL_R for k = 1..Nr into prior-initialised matrices, advectionScheme = ENUM_UPWIND_1RST and ENUM_DST2:
  a3d, b3d, c3d bitwise on every point incl. halos (element equality, bit patterns, finite);
* SOLVE_PENTADIAGONAL (errCode = -1 on entry): the solution bitwise on every point and the per-tile errCode equal
  (the zero-pivot branch is exercised: c5d = 0 at level 1 on ~1 % of the points);
* negative controls: the other scheme's matrices, and the solve with a5d := 0, each differ from the oracle.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax import paths  # noqa: E402
from mitjax.farray import FArray  # noqa: E402

HERE = Path(__file__).resolve().parents[2] / "reference" / "replay_go"


def _io():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_replay_go_io", HERE / "replay_io.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runs():
    out = []
    for line in (HERE / "CURRENT").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            exp, inp, rel = line.split()
            out.append((exp, inp, paths.RUNS / rel))
    return out


def _case(exp, inp, rundir):
    io = _io()
    size, f, one_d = io.read_inputs(rundir)
    out = io.read_outputs(rundir, size)
    g = io.read_grid(rundir, size)
    b = dict(i=(1 - size["OLx"], size["sNx"] + size["OLx"]), j=(1 - size["OLy"], size["sNy"] + size["OLy"]))
    Nr = size["Nr"]
    A3 = lambda a, n: FArray(jnp.asarray(a), n, k=(1, Nr), **b)            # noqa: E731
    A2 = lambda a, n: FArray(jnp.asarray(a), n, **b)                       # noqa: E731
    A1 = lambda a, n: FArray(jnp.asarray(a), n, k=(1, Nr), tiled=False)    # noqa: E731
    grid = SimpleNamespace(recip_rA=A2(g["recip_rA"], "recip_rA"), recip_drF=A1(g["recip_drF"], "recip_drF"),
                           recip_deepFac2C=A1(g["recip_deepFac2C"], "recip_deepFac2C"), rkSign=g["rkSign"])
    params = SimpleNamespace(recip_rhoFacC=A1(g["recip_rhoFacC"], "recip_rhoFacC"))
    return SimpleNamespace(size=size, f=f, one_d=one_d, out=out, A3=A3, A2=A2, A1=A1, grid=grid, params=params,
                           recip_hFac=A3(g["recip_hFacC"], "recip_hFacC"), exp=exp, inp=inp)


def _bits(ours, ref):
    o = np.ascontiguousarray(np.asarray(ours), np.float64)
    r = np.ascontiguousarray(ref, np.float64)
    assert o.shape == r.shape
    return int(np.count_nonzero(o.view(np.int64) != r.view(np.int64))), int(np.count_nonzero(~(o == r))), \
        int(np.count_nonzero(~np.isfinite(o)))


def _dst2u1(c, scheme):
    from mitjax.pkg.generic_advdiff.gad_dst2u1_impl_r import gad_dst2u1_impl_r
    sz = c.size
    kcfg = SimpleNamespace(Nr=sz["Nr"])

    def run(rT, a, b_, cc, dT):
        a, b_, cc = c.A3(a, "a3d"), c.A3(b_, "b3d"), c.A3(cc, "c3d")
        for k in range(1, sz["Nr"] + 1):                                    # the harness's DO k = 1, Nr
            a, b_, cc = gad_dst2u1_impl_r(k, 1, sz["sNx"], 1, sz["sNy"], scheme, c.A1(dT, "deltaTarg"),
                                          c.A2(rT[:, k - 1], "rTrans"), c.recip_hFac, a, b_, cc, cfg=kcfg,
                                          grid=c.grid, params=c.params)
        return a.data, b_.data, cc.data
    return jax.jit(run)(c.f["rTrans"], c.f["aPrior"], c.f["bPrior"], c.f["cPrior"], c.one_d["deltaTarg"])


def _penta(c, a5=None):
    from mitjax.model.src.solve_pentadiagonal import solve_pentadiagonal
    from mitjax.tests import grid_gate as gg
    cfg = gg.experiment(c.exp, c.inp).cfg
    sz = c.size
    a5 = c.f["a5d"] if a5 is None else a5

    def run(a, b_, cc, d, e, y):
        y, err = solve_pentadiagonal(1, sz["sNx"], 1, sz["sNy"], c.A3(a, "a5d"), c.A3(b_, "b5d"), c.A3(cc, "c5d"),
                                     c.A3(d, "d5d"), c.A3(e, "e5d"), c.A3(y, "y5d"), -1, cfg=cfg)
        return y.data, err
    return jax.jit(run)(a5, c.f["b5d"], c.f["c5d"], c.f["d5d"], c.f["e5d"], c.f["y5d"])


@pytest.mark.parametrize("run", runs(), ids=lambda r: f"{r[0]}/{r[1]}")
def test_dst2u1_impl_r_bitwise(run):
    from mitjax.pkg.generic_advdiff.gad_h import ENUM_DST2, ENUM_UPWIND_1RST
    c = _case(*run)
    for scheme, pre in ((ENUM_UPWIND_1RST, "up1"), (ENUM_DST2, "dst2")):
        res = _dst2u1(c, scheme)
        for name, ours in zip(("a3d", "b3d", "c3d"), res):
            assert _bits(ours, c.out[f"{pre}_{name}"]) == (0, 0, 0), (scheme, name)
    # negative control: the other scheme's matrices differ (rUpwind = |rCenter|*(1-rLimit) matters)
    res = _dst2u1(c, ENUM_UPWIND_1RST)
    assert _bits(res[0], c.out["dst2_a3d"])[0] > 100        # measured: 276 (advect_xz), more on global_ocean


@pytest.mark.parametrize("run", runs(), ids=lambda r: f"{r[0]}/{r[1]}")
def test_solve_pentadiagonal_bitwise(run):
    c = _case(*run)
    y, err = _penta(c)
    assert _bits(y, c.out["penta_y"])[:2] == (0, 0)
    ref_err = c.out["penta_err"][:, 0, 0, 0].astype(int)
    assert np.array_equal(np.asarray(err), ref_err), (np.asarray(err), ref_err)
    assert (ref_err == 1).any()                     # the zero-pivot branch ran (c5d = 0 at level 1)
    # negative control: the 2nd lower diagonal dropped
    y0, _ = _penta(c, a5=np.zeros_like(c.f["a5d"]))
    assert _bits(y0, c.out["penta_y"])[0] > 1000
