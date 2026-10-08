"""Lane M4ADCOL (M4 step 7, sessions 2-3): the FD-oracle check of 1D_ocean_ice_column/input_ad -- our `global fc` at the
grdchk perturbations of data.grdchk (xx_theta, (i,j) = (1,1), k = 1..4, grdchk_eps = 1.d-7, both signs) against lane
A's FD oracle job27856073-fdzero (the code_ad build's grdchk run, equal to TAF's printed fc+/fc- in every
digit) in every printed digit (`(A,1PE22.14)`, cost_final.F:244).

The perturbation enters as the genarr3d control record (CTRL_MAP_INI_GENARR: theta + xx/sqrt(w) with w = 1 from
ones_64b.bin, bounded by CTRL_BOUND_3D) through drivers.Model(xx={"genarr": ...}), with useECCO as data.pkg sets it
(session 3): pkg/ecco's gencost terms enter fc with mult_gencost = 0 and CTRL_COST_GEN3D's xx**2 = 1e-14 term with
mult_genarr3d = 1 (the useECCO default, ctrl_readparms.F:482-500).
Negative control: the k = 1 perturbation does not reproduce the k = 2 oracle value (the comparison discriminates
between neighbouring levels). Costs: about 15 min on a CPU node (eight set-ups and runs).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import adcol_gate as A  # noqa: E402


def _oracle_fc():
    from mitjax import paths as P
    out = (P.REFERENCE_RUNS / A.EXP[0] / A.EXP[1] / "job27856073-fdzero" / "rundir" / "output.txt").read_text()
    lines = [ln for ln in out.splitlines() if " global fc = " in ln]
    assert len(lines) == 9, lines             # the reference run, then fc+ and fc- for k = 1..4
    return lines[0], {(k, s): lines[1 + 2*(k - 1) + (0 if s > 0 else 1)] for k in (1, 2, 3, 4) for s in (1, -1)}


def _fc(k, sgn):
    from mitjax import paths as P
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.farray import FArray
    from mitjax.tests import advect_gate as ag
    exp_dir = P.UPSTREAM / "verification" / A.EXP[0]
    e = load_experiment(exp_dir, A.EXP[1])
    rundir = make_rundir(exp_dir, A.EXP[1], ag.out_dir(f"adcol-fd-k{k}{'p' if sgn > 0 else 'm'}"))
    sz = e.cfg.size
    shape = (sz.nSx*sz.nSy, sz.Nr, sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx)
    d = np.zeros(shape)
    d[0, k - 1, sz.OLy, sz.OLx] = sgn * 1e-7         # data.grdchk: iGloPos = jGloPos = 1, grdchk_eps = 1.d-7
    dims = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy), k=(1, sz.Nr))
    xx = FArray(jnp.asarray(d), "xx", **dims)
    m = Model(e, rundir, xx={"genarr": {(3, 1): xx}})
    res = forward(m, write_pickups=False)
    jax.clear_caches()
    return [r for r in res.records if " global fc = " in r]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_fc_at_grdchk_perturbations_every_digit(k):
    _, orc = _oracle_fc()
    for sgn in (1, -1):
        got = _fc(k, sgn)
        print(k, sgn, got)
        assert got == [orc[(k, sgn)]], (k, sgn, got, orc[(k, sgn)])
        if k == 1 and sgn == 1:
            assert got != [orc[(2, 1)]]                  # negative control (docstring)
