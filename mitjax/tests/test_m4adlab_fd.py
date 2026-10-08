"""lab_sea/input_ad (M4 step 7 part 2, lane M4ADLAB session 4): the FD oracle of the code_ad build -- our `global fc`
at the ten grdchk perturbations of data.grdchk against lane A's FD oracle job27856073-fdzero in every printed digit
(`(A,1PE22.14)`, cost_final.F:244). The oracle's fc+ / fc- equal TAF's output_adm.txt in every digit
(test_m4adlab_targets.py).

grdchk (data.grdchk: grdchk_eps = 1.d-3, iGloPos = 4, jGloPos = 8, nstep = 1, nend = 4, on xx_atemp = gentim2d 1;
the oracle prints `grdchk pos: i,j,k= 6..10 8 1 ; bi,bj= 1 1 ; rec= 1`): the perturbation +-1e-3 of the control
vector at record 1 of xx_atemp, tile (1,1), (i,j) = (6..10, 8) -- the gentim2d control record through
drivers.Model(xx=...) (CTRL_MAP_INI_GENTIM2D: xx/sqrt(w) with w = 1 from ones_64b.bin; CTRL_GET_GEN interpolates the
two records in time), the whole 4-step run with every package as data.pkg sets it. fc includes CTRL_COST_GEN2D's
xx**2 = 1e-6 of xx_atemp (mult_gentim2d = 1, the useECCO default).
Negative control: the (6,8) perturbation does not reproduce the (7,8) oracle value.
Costs: about 40 min on a CPU node (ten set-ups and runs).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("lab_sea", "input_ad")
POINTS = (6, 7, 8, 9, 10)


def _oracle_fc():
    from mitjax import paths as P
    out = (P.REFERENCE_RUNS / EXP[0] / EXP[1] / "job27856073-fdzero" / "rundir" / "output.txt").read_text()
    lines = [ln for ln in out.splitlines() if " global fc = " in ln]
    assert len(lines) == 11, lines            # the reference run, then fc+ and fc- for i = 6..10
    return lines[0], {(i, s): lines[1 + 2*(i - 6) + (0 if s > 0 else 1)] for i in POINTS for s in (1, -1)}


def _fc(i, sgn):
    from mitjax import paths as P
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.farray import FArray
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d, fstr_blank
    from mitjax.tests import advect_gate as ag
    exp_dir = P.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    rundir = make_rundir(exp_dir, EXP[1], ag.out_dir(f"adlab-fd-i{i}{'p' if sgn > 0 else 'm'}"))
    sz = e.cfg.size
    b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
    z = jnp.zeros((sz.nSx*sz.nSy, sz.sNy + 2*sz.OLy, sz.sNx + 2*sz.OLx), jnp.float64)
    d = np.zeros(z.shape)
    d[0, sz.OLy + 8 - 1, sz.OLx + i - 1] = sgn * 1e-3        # tile (1,1) = index 0, (i, j) = (i, 8), record 1
    gs = [g for g in ctrl_readparms_gentim2d(e.run, e.cfg) if not fstr_blank(g.xx_gentim2d_weight)]
    # records per control: CTRL_INIT_REC (Model._gentim2d_nrec); 2 for every gentim2d of this run (period 864000 s
    # over the 4-hour run from 19790101: the oracle's `ctrl-wet 6: no recs for ivar = 5..13  2`)
    xx = {g.iarr: [FArray(z, "xx", **b), FArray(z, "xx", **b)] for g in gs}
    xx[1] = [FArray(jnp.asarray(d), "xx", **b), FArray(z, "xx", **b)]
    m = Model(e, rundir, xx=xx)
    assert all(len(xx[g.iarr]) == m._gentim2d_nrec(g) for g in gs)
    res = forward(m, write_pickups=False)
    jax.clear_caches()
    return [r for r in res.records if " global fc = " in r]


@pytest.mark.parametrize("i", POINTS)
def test_fc_at_grdchk_perturbations_every_digit(i):
    _, orc = _oracle_fc()
    for sgn in (1, -1):
        got = _fc(i, sgn)
        print(i, sgn, got)
        assert got == [orc[(i, sgn)]], (i, sgn, got, orc[(i, sgn)])
        if i == 6 and sgn == 1:
            assert got != [orc[(7, 1)]]                      # negative control (docstring)
