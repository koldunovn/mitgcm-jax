"""Time-loop skeleton gates (plan Task 11): the iteration counters of mitjax/drivers/the_main_loop.py against the
oracle's %MON time_tsnumber / time_secondsf lines (every monitor block of every M1 variant, the printed text), and the
split-run identity of the scan driver (k iterations + n-k iterations from the returned carry == n iterations). The
model restart gate (N steps == k steps + pickup + N-k steps, bitwise) needs FORWARD_STEP (Task 12): recorded in the
lane handoff, not faked here."""

import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.drivers.the_main_loop import forward_step_counters, start_counters, the_main_loop
from mitjax.io.stdout import FORWARD_DYNAMICS, monitor_blocks, read_stdout
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

IDS = [f"{e}/{i}" for e, i in gg.VARIANTS]


def _trivial_step(model, st, iloop, myTime, myIter):
    """A FORWARD_STEP that only updates the counters where forward_step.F:807-808 does."""
    myTime, myIter = forward_step_counters(iloop, nIter0=model["nIter0"], startTime=model["startTime"],
                                           deltaTClock=model["deltaTClock"])
    return st, myTime, myIter, (myIter, myTime)


def _model(tp):
    return {"nIter0": jnp.int32(tp.nIter0), "startTime": jnp.float64(tp.startTime),
            "deltaTClock": jnp.float64(tp.deltaTClock)}


def _same_text(printed, x):
    """Our value printed with the printed text's number of mantissa digits (1PE: d.ddd...E+xx) equals the text."""
    mant, _, ex = printed.upper().replace("D", "E").partition("E")
    nd = len(mant.split(".")[1]) if "." in mant else 0
    ours = f"{x:.{nd}E}"
    om, _, oe = ours.partition("E")
    return om == mant.lstrip("+") and int(oe) == int(ex)


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_time_counters_vs_monitor(exp, inp):
    _, _, rundir = gg.oracle(exp, inp)
    blocks = [b for b in monitor_blocks(read_stdout(rundir / "output.txt")) if b.kind == FORWARD_DYNAMICS]
    tp = ig.params(exp, inp).time
    t0, i0 = start_counters(tp)
    _, tN, iN, (iters, times) = the_main_loop(_trivial_step, _model(tp), {}, t0, i0, nTimeSteps=tp.nTimeSteps)
    ours = {int(i0): float(t0)}
    ours.update({int(i): float(t) for i, t in zip(np.asarray(iters), np.asarray(times))})
    assert int(iN) == tp.nEndIter and float(tN) == tp.endTime
    seen = 0
    for b in blocks:
        it = int(b["time_tsnumber"])
        assert it in ours, f"monitor at iteration {it} outside the run"
        assert _same_text(b.text["time_secondsf"], ours[it]), (it, b.text["time_secondsf"], ours[it])
        seen += 1
    assert seen >= 2, f"{exp}/{inp}: only {seen} monitor blocks"


def test_time_counters_negative_control():
    """Planted error: the counter update one iteration late (myIter = nIter0 + iLoop - 1). The monitor check bites."""
    exp, inp = "tutorial_barotropic_gyre", "input"
    _, _, rundir = gg.oracle(exp, inp)
    blocks = [b for b in monitor_blocks(read_stdout(rundir / "output.txt")) if b.kind == FORWARD_DYNAMICS]
    tp = ig.params(exp, inp).time

    def late(model, st, iloop, myTime, myIter):
        t, i = forward_step_counters(iloop - 1, nIter0=model["nIter0"], startTime=model["startTime"],
                                     deltaTClock=model["deltaTClock"])
        return st, t, i, (i, t)
    t0, i0 = start_counters(tp)
    _, _, _, (iters, times) = the_main_loop(late, _model(tp), {}, t0, i0, nTimeSteps=tp.nTimeSteps)
    ours = dict(zip(np.asarray(iters).tolist(), np.asarray(times).tolist()))
    bad = [b for b in blocks if int(b["time_tsnumber"]) not in ours
           or not _same_text(b.text["time_secondsf"], ours[int(b["time_tsnumber"])])]
    assert bad, "the late counter was not detected"


def test_split_run_identity_and_first_eager():
    """k + (n-k) iterations from the returned carry == n iterations; first_eager gives the same outputs."""
    tp = ig.params("global_ocean.90x40x15", "input").time
    m = _model(tp)
    t0, i0 = start_counters(tp)
    st0 = {"x": jnp.zeros(3)}

    def step(model, st, iloop, myTime, myIter):
        t, i = forward_step_counters(iloop, nIter0=model["nIter0"], startTime=model["startTime"],
                                     deltaTClock=model["deltaTClock"])
        return {"x": st["x"] * 0.5 + t}, t, i, (i, t)
    s_all, t_all, i_all, o_all = the_main_loop(step, m, st0, t0, i0, nTimeSteps=10)
    s_a, t_a, i_a, o_a = the_main_loop(step, m, st0, t0, i0, nTimeSteps=4)
    s_b, t_b, i_b, o_b = the_main_loop(step, m, s_a, t_a, i_a, nTimeSteps=6, iloop0=5)
    assert np.array_equal(np.asarray(s_all["x"]), np.asarray(s_b["x"])) and int(i_all) == int(i_b)
    assert float(t_all) == float(t_b)
    s_e, _, _, o_e = the_main_loop(step, m, st0, t0, i0, nTimeSteps=10, first_eager=True)
    assert np.array_equal(np.asarray(s_e["x"]), np.asarray(s_all["x"]))
    assert np.array_equal(np.asarray(o_e[0]), np.asarray(o_all[0]))
