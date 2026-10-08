"""Helpers of the vermix end-to-end gates (M3 Task 30, vermix lane; not a test file): the run driver's Model on the
run directory of lane A's dumps-on run (cube_run_gate.CubeRun is generic: any registered dumps-on run), its own step
with forward_step's probe on every dumped stage (cs32_gate.run_compare), the initial state and grid vs I02_ini_fields
and G00_geometry, and the whole run through the driver (cube_run_gate.whole_run, advect_gate.run_verdict)."""

from mitjax.tests import cube_run_gate as C
from mitjax.tests import cs32_gate as CS
from mitjax.tests import grid_gate as gg
from mitjax.tests import r1_gate as rg

EXP = "vermix"
# stages compared elsewhere: the exchange probe (the maps' own gate, scripts/make_exch_maps.py), the monitor (the
# whole run's %MON records); I02/G00: initial_bad
NOT_STEP_STAGES = ("I02_ini_fields", "G00_geometry", "X00_exch_probe", "S17_monitor")


def run(inp):
    return C.cube_run(EXP, inp)


def initial_bad(r):
    """({(stage, field): compare tuple} of the fields that differ, n fields compared): the Model's initial State vs
    I02_ini_fields (INI_FIELDS; vermix's later INITIALISE_VARIA calls write none of these fields) and its grid vs
    G00_geometry (every field the Grid holds; vertical vectors through grid_gate.as_dump_shape)."""
    it = r.its[0]
    out, n = {}, 0
    res = rg.compare_stage(r.ds, it, "I02_ini_fields", r.m.state0)
    # wVel: INITIALISE_VARIA's INTEGR_CONTINUITY (:334) writes it after INI_FIELDS (the dump is before); the State's
    # wVel is gated at S00_begin of the first step
    res.pop("wVel", None)
    n += len(res)
    out.update({("I02_ini_fields", k): v for k, v in C.bad(res).items()})
    import numpy as np
    grid = gg.build_grid(EXP, r.inp)                                    # INITIALISE_FIXED's grid chain
    pairs, _ = gg.gated_pairs(EXP, r.inp, r.ds, it)
    for ours_name, (stage, name) in pairs:
        if stage != "G00_geometry":
            continue
        x = getattr(grid, ours_name)
        ref = r.ds.field(it, stage, name)
        if not hasattr(x, "dims"):                                      # a GRID.h scalar (gravitySign, rkSign)
            ok = bool(np.all(np.asarray(x, np.float64) == ref))
            n += 1
            if not ok:
                out[(stage, ours_name)] = ("scalar", float(np.asarray(x)), float(ref.ravel()[0]))
            continue
        if x.data.ndim == 2:                                           # (j, tile) vectors: cosFacU/V, sqCosFacU/V
            ours = np.broadcast_to(np.asarray(x.data)[:, None, :, None], ref.shape)   # constant along i (dump)
        else:
            ours, ref = gg.as_dump_shape(x.data, ref)
        v = rg.compare_field(ours, ref)
        n += 1
        if v[0] == "shape" or v[2] or v[3]:
            out[(stage, ours_name)] = v
    return out, n


def steps(r, n):
    """CS.run_compare: ({(it, stage): bad fields}, {(it, stage): fields compared}, final carry)."""
    return CS.run_compare(r, n)


def missing_stages(r, ncmp, n):
    """Dumped stages of the first n iterations that no probe compared (beyond NOT_STEP_STAGES)."""
    return {it: [s for s in r.stages(it) if s not in NOT_STEP_STAGES and (it, s) not in ncmp] for it in r.its[:n]}


def whole_run(inp):
    """(Model, forward result, oracle, run_verdict tuple, output_file_diffs tuple)."""
    from mitjax.tests import advect_gate as ag
    m, res, o = C.whole_run(EXP, inp, tag="vermix-whole")
    return m, res, o, ag.run_verdict(EXP, inp, res, o), C.output_file_diffs(m, o)
