"""Helpers of the 1D_ocean_ice_column/input_ad forward gates (M4 step 7, lane M4ADCOL; not a test file).

The driver Model of the code_ad build on lane A's dumps-on run directory (job27856057-jdon), its oracle dumps and the
probing step (cube_run_gate / cs32_gate). Session 3: the Model runs with useECCO as data.pkg sets it (pkg/ecco ported,
mitjax/pkg/ecco; session 2's interim `ecco_off` configuration is gone).
"""

import dataclasses
from functools import lru_cache

EXP = ("1D_ocean_ice_column", "input_ad")


class AdColRun:
    """cube_run_gate.CubeRun of input_ad (Model on the jdon run directory, the oracle dumps)."""

    def __init__(self, cfg_map=None):
        from mitjax.config.params import load
        from mitjax.drivers.model import Model
        from mitjax.io.dump import DumpSet
        from mitjax.tests import cube_run_gate as CR
        self.exp, self.inp = EXP
        self.top = CR.run_top(*EXP)
        self.e = load(*EXP)
        if cfg_map is not None:
            self.e = dataclasses.replace(self.e, cfg=cfg_map(self.e.cfg))
        self.m = Model(self.e, self.top / "rundir")
        self.ds = DumpSet(self.top / "dumps")
        self.its = self.ds.iterations()

    def stages(self, it):
        return self.ds.stages(it)


@lru_cache(maxsize=None)
def adcol_run():
    return AdColRun()


# not probed by FORWARD_STEP (compared elsewhere or set-up stages), as test_m4col_column.NOT_PROBED
NOT_PROBED = ("C01_cg2d_inputs", "C02_cg2d_solution", "X00_exch_probe", "I02_ini_fields", "G00_geometry",
              "S17_monitor")


def free_steps(r, n=3):
    """cs32_gate.run_compare: n steps from the Model's initial carry, every dumped stage on every point, bit
    patterns -> (bad {(it, stage): fields}, compared {(it, stage): count}, missing {it: stages not compared})."""
    from mitjax.tests import cs32_gate as CS
    bad, ncmp, _ = CS.run_compare(r, n)
    miss = {it: [s for s in r.stages(it) if s not in NOT_PROBED and (it, s) not in ncmp] for it in r.its[:n]}
    return {k: v for k, v in bad.items() if v}, ncmp, miss


def teacher_step(r, it, *, cfg=None, arrays=None, f=None):
    """m4col_gate.teacher_step on input_ad with an optional planted `cfg` (m4col_gate.cfg_with) and planted
    `arrays` (the Model's Arrays with a traced input replaced): the whole step of iteration `it` from the oracle's
    state at its start (m4col_gate.teacher_carry; the Model's initial carry at it = 0), every dumped stage compared
    on every point. Returns ({stage: bad fields}, {stage: fields compared})."""
    import jax.numpy as jnp
    from mitjax.tests import cube_run_gate as CR
    from mitjax.tests import m4col_gate as C
    m = r.m
    f = C.step_fn(r, cfg) if f is None else f
    tp = m.prm.time
    carry = C.teacher_carry(m, r.ds, it)
    _, probes = f(m.arrays if arrays is None else arrays, carry, jnp.int32(it + 1),
                  jnp.float64(tp.startTime + tp.deltaTClock*it), jnp.int32(tp.nIter0 + it))
    bad, ncmp = {}, {}
    for key, vals in probes.items():
        st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
        name = st[0] if isinstance(st, tuple) else st
        if name in r.stages(it):
            if hasattr(vals, "__dataclass_fields__"):              # S18_cost_tile: cost.h by field
                from mitjax.tests import cs32_gate as CS
                vals = CS._cost_fields(vals, m.cfg.size)
            res = CR.compare_stage(r, it, st, vals)
            bad[name] = {**bad.get(name, {}), **CR.bad(res)}
            ncmp[name] = ncmp.get(name, 0) + len(res)
    return {s: v for s, v in bad.items() if v}, ncmp


def first_bad(r, its=(0, 1, 2), **kw):
    """[(it, first bad stage in dump order, its bad fields)] of teacher_step over `its` (empty: no difference)."""
    out = []
    for it in its:
        bad, _ = teacher_step(r, it, **kw)
        order = r.stages(it)
        for s in order:
            if s in bad:
                out.append((it, s, sorted(bad[s])))
                break
    return out


def with_exfp(m, **values):
    """The Model's Arrays with EXF_PARAM.h REAL values replaced (traced inputs: no new set-up)."""
    import numpy as np
    from mitjax.pkg.exf.exf_param_h import ExfParams
    a = m.arrays
    e = a.pkc["exfp"]
    return a.replace(pkc=dict(a.pkc, exfp=ExfParams(r={**e.r, **{k: np.float64(v) for k, v in values.items()}},
                                                     s=e.s)))
