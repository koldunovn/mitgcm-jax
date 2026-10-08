"""GOADK in-model gates (M2 Task 24 forward): global_ocean.90x40x15/input_ad* through the driver Model.

1. Free-running steps (no teacher forcing): every dumped stage of iterations 0-2 of lane A's dumps-on run (`jdon`),
   and 0-3 of the planted-control runs (`ctrlxx`, job27833840), compared on every point of every tile, halos included
   (element equality, bit patterns, finite): the state, forcing, GM/Redi tensor and bolus stream functions, the
   per-level MOM_VECINV tendencies (D00c) and hydrostatic pressure (D00a), the grid the step sees (G00, NONLIN_FRSURF
   hFac), the tracer stages (DST3 + implicit vertical advection, GM/Redi), the cost (S18).
2. The whole 10-step run through the run driver (`drivers/run.forward`, incl. COST_FINAL with COST_ATLANTIC_HEAT):
   every %MON line, the cg2d lines and the COST_FINAL lines (early / per-tile objf_atl / local / global fc) identical
   to the oracle STDOUT.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

VARIANTS = ("input_ad", "input_ad.kapgm", "input_ad.kapredi", "input_ad.bottomdrag")
# the dumped fields the step does not probe: CG2D's internals (C01/C02: its inputs and solution are compared through
# S09 and the cg2d lines of the whole run), I02_ini_fields (INI_FIELDS' output: INITIALISE_VARIA changes theta and
# wVel, dEtaHdt after it -- the end of INITIALISE_VARIA is the S00_begin record of iteration 0, compared), the exchange
# probe X00, and the G00 fields our Grid does not carry (r* / hybrid-sigma / reference profiles of other routines)
NOT_PROBED = ("C01_cg2d_inputs", "C02_cg2d_solution", "I02_ini_fields", "X00_exch_probe", "G00_geometry")


@pytest.fixture(autouse=True)
def _release_executables():
    """Each test compiles whole-step programs: drop them afterwards, since every XLA:CPU kernel holds memory mappings
    and a process may hold only vm.max_map_count of them (PORTING_LESSONS, 2026-10-02)."""
    import gc

    import jax
    yield
    from mitjax.tests import goadk_model_gate as M
    M.model.cache_clear()
    jax.clear_caches()
    gc.collect()


def _free(inp, kind, nsteps):
    from mitjax.tests import goadk_model_gate as M
    m, ds = M.model(inp, kind)
    out, _ = M.run_free(m, ds, nsteps)
    return m, ds, out


@pytest.mark.parametrize("inp,kind,nsteps", [("input_ad", "jdon", 3), ("input_ad.bottomdrag", "jdon", 3),
                                             ("input_ad", "ctrlxx", 4), ("input_ad.kapgm", "ctrlxx", 4),
                                             ("input_ad.kapredi", "ctrlxx", 4),
                                             ("input_ad.bottomdrag", "ctrlxx", 4)])
def test_free_steps_bitwise_every_dumped_stage(inp, kind, nsteps):
    from mitjax.tests import goadk_model_gate as M
    m, ds, out = _free(inp, kind, nsteps)
    assert [it for it, _, _ in out] == list(range(nsteps))
    for it, res, _ in out:
        assert not M.bad(res), (it, M.bad(res))
        left = [(s, n) for s, n in M.not_compared(ds, it, res) if s not in NOT_PROBED]
        assert not left, (it, left)
        assert sum(len(r) for r in res.values()) > 400, it
        assert len(res.get("D00c_mom_vecinv", {})) == 4 * 15 and "P05_gmredi_tensor" in res and \
            "T13_temp_impl" in res and "S18_cost_tile" in res, (it, sorted(res))


@pytest.mark.parametrize("inp", VARIANTS)
def test_whole_run_monitor_and_cost(inp):
    import os
    import time
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import goadk_gate as G
    exp_dir = paths.UPSTREAM / "verification" / "global_ocean.90x40x15"
    out = paths.RUNS / "goadk" / "tests_whole" / f"{inp}-{os.getpid()}-{time.time_ns()}"      # unique, never reused
    m = Model(load_experiment(exp_dir, inp), make_rundir(exp_dir, inp, out))
    res = forward(m, write_pickups=False)
    raw = [ln.rstrip("\n") for ln in open(G.run_top(inp, "yardstick") / "rundir" / "output.txt")]
    k0 = next(n for n, r in enumerate(raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    kinds = {"mon": lambda r: "%MON " in r, "cost": lambda r: " fc = " in r or "--> objf_" in r,
             "cg2d": lambda r: "cg2d_init_res" in r or "cg2d_iters" in r or "cg2d_res" in r}
    for kind, sel in kinds.items():
        a, b = [r for r in res.records if sel(r)], [r for r in raw[k0:] if sel(r)]
        # the oracle's STDOUT continues with GRDCHK_MAIN's perturbed runs: the reference run's records come first
        assert a and a == b[:len(a)], (kind, len(a), len(b), [(x, y) for x, y in zip(a, b) if x != y][:3])
    assert sum("%MON time_tsnumber" in r for r in res.records) == 11
    assert len([r for r in res.records if kinds["cost"](r)]) == 3 + 4                # early, 4 tiles, local, global
    assert any(r.endswith(" global fc =  -4.50065077785264E-02") or " global fc = " in r for r in res.records)


def test_negative_control_planted_control_vs_zero_oracle():
    """The planted-control run's xx_theta in a Model gated against the zero-control dumps: S00_begin theta of
    iteration 0 differs (the control enters at initialisation) -- the free-step gate sees the control."""
    from mitjax.io.dump import DumpSet
    from mitjax.tests import goadk_gate as G
    from mitjax.tests import goadk_model_gate as M
    m, _ = M.model("input_ad", "ctrlxx")
    ds0 = DumpSet(G.run_top("input_ad", "jdon") / "dumps")
    out, _ = M.run_free(m, ds0, 1)
    b = M.bad(out[0][1])
    assert "S00_begin" in b and b["S00_begin"]["theta"][1] > 1000, b.get("S00_begin")
