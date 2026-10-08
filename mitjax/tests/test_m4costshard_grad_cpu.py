"""Sharded gradients of the global_ocean.cs32x15 code_ad builds (COST_TEST's cost.h) on fake CPU devices (lane
M4COSTSHARD, tier 1x).

The blocker (lane M4ADCS32ICE session 3, jobs 27907962 / 27907964): GenarrAdjoint's final cost ran Model.cost_final on
the device's own tiles with the set-up's global maskC, so under shard_map COST_TEST mixed [12] and [Tloc] arrays and
did not trace. The fix (drivers/adjoint_run.final_cost_gathered, shared with the gentim2d driver `problem`): COST_FINAL
runs on the tiled fields it reads gathered to every real tile in tile order (ex.all_tiles; the identity on one
device, a psum of the zero-padded blocks under shard_map: exact), from the traced model argument (grid, exchanger,
cost_fixed), so each tile's objf_test chain and GLOBAL_SUM_TILE_RL's tile-order sum are the Fortran's at every P.
The audit (scripts/m4costshard_audit.py, jobs 27908675-77) measured the carried cost state at P = 6: cost.h's plain
[tile] accumulators (tile_fc, objf_*) stay [12] and invariant through the step (no per-step term writes them in
these builds: COST_TILE only accumulates the sharded cMean* FArrays), COST_TEST is the only per-tile chain.

* trace gate (input_ad.seaice, P = 6; the negative control): the objective with the pre-fix final cost (the gather
  reverted) fails to trace as before (incompatible shapes in COST_TEST), the gathered one traces; every plain [tile]
  cost leaf is invariant after a step and the cMean* are sharded.
* sharded gradient (scripts/m4costshard_shardgrad.py in a subprocess with 6 fake devices): GenarrAdjoint's objective
  (CTRL_MAP_INI_GENARR of xx_theta, the per-step checkpointed loop, COST_FINAL) at P = 1 and under
  jit(shard_map(check_vma=True)) at the listed P: fc bitwise vs P = 1, the gradient finite on every lane incl. the
  padded layout, padding tiles exactly 0. P = 5 (Tpad = 15) has padding tiles.
* the R5 criterion max|g_P - g_1| <= TOL max|g_1| (the transposed exchanges sum a point's copies in another order
  than the single-device scatter-add): met by input_ad.seaice (2 steps, P = 6: 1.0e-15) and by input_ad cut to its
  first step (1.6e-17); NOT met by the whole input_ad window (5 steps; dev jobs 27908678 / 27908873: run mode P = 6
  6.0e-14, P = 2 2.2e-13, P = 5 1.8e-13; exact mode 8.5e-14 / 1.2e-13), a strict xfail with the measured values:
  the difference grows with the window, P = 6: 1.6e-17 after one step, 9.2e-16 after three, 6.0e-14 after five
  (jobs 27909127, 27909128, 27908678); input_ad.seaice_dynmix (5 steps) is above it too (run 2.4e-14, exact
  9.3e-14 at P = 6; not a case here). Localisation (lane M4COSTSHARD session 2): the gradient's own conditioning --
  CG2D counts identical at P = 1 and 6, and ONE ulp injected at P = 1 moves the gradient as much as P = 6 does.
* the window's 1-ulp floor criterion (plan decision 18, Nikolay 2026-10-06 "1 a"; replaces the strict xfail for the
  whole input_ad window): max|g_P - g_1| <= C_FLOOR * floor, floor = the largest max|g - g_base| / max|g_base| over
  the 1-ulp (relative +-1 pattern, fixed seed) injections into the adjoint state at every step boundary of the SAME
  window at P = 1 (scripts/m4costshard_cg2dlog.py perturb; its uninstrumented base equals g_1 bit for bit). Measured
  (job 27909580): floor 2.27e-13; P = 6 / 5 / 2: 6.0e-14 / 1.8e-13 / 2.2e-13 (<= 1.0 floor). C_FLOOR = 10 is the
  main session's margin (Nikolay may change it). Negative control: a planted difference of 1e-11 max|g_1| at one
  point of the real g_1 fails the bound. Short windows keep R5's TOL (1e-14), which they meet.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOL = 1e-14
C_FLOOR = 10                       # plan decision 18: the bar as a multiple of the window's P = 1 1-ulp floor
ROOT = Path(__file__).resolve().parents[2]
_R = {}


def _run(script, *args, timeout=7200, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}     # the script sets its own (6 devices)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])


def test_trace_negative_control(tmp_path):
    out = tmp_path / "audit"
    _run("m4costshard_audit.py", out, "seaice", 6, timeout=1800)
    R = json.loads((out / "audit.json").read_text())
    assert not R["J_old"]["ok"] and "incompatible shapes" in R["J_old"]["error"].lower(), R["J_old"]
    assert R["J_gathered"]["ok"], R["J_gathered"]
    assert R["step"]["ok"], R["step"]
    for x in R["step"]["cost"]:
        if x["path"].startswith(".cMean"):
            assert x["shape"][0] == 2 and x["vma"], x                   # sharded: 2 local tiles, varying
        else:
            assert x["shape"] in ([], [12]) and not x["vma"], x         # replicated, invariant
    assert all(x["shape"] in ([], [12]) and not x["vma"] for x in R["J_gathered"]["cost_after_final"])


# (id, variant, mode, P list, steps (0 = the whole window))
CASES = [("seaice-run", "seaice", "run", "6,2", 0), ("ad-run", "ad", "run", "6,5", 0), ("ad-run-1step", "ad", "run",
                                                                                        "6", 1)]
FLOOR_CASES = {"ad-run"}           # the windows judged by the 1-ulp floor (decision 18); the others by TOL


def _result(tmp_path_factory, cid):
    if cid not in _R:
        _, var, mode, procs, nst = next(c for c in CASES if c[0] == cid)
        out = tmp_path_factory.mktemp(cid)
        _run("m4costshard_shardgrad.py", out / "r", var, mode, procs,
             env_extra={"MJX_SHARD_NSTEPS": str(nst)} if nst else None)
        _R[cid] = (procs, json.loads((out / "r" / "result.json").read_text()), out)
    return _R[cid][:2]


def _floor(cid):
    """The window's 1-ulp floor at P = 1 (decision 18) from scripts/m4costshard_cg2dlog.py perturb, whose REF is the
    case's own shardgrad output (g1.npy): (floor, perturb result, REF dir)."""
    procs, R, out = _R[cid]
    if "floor" not in R:
        _run("m4costshard_cg2dlog.py", out / "floor", "perturb", out / "r")
        R["floor"] = json.loads((out / "floor" / "result.json").read_text())
    F = R["floor"]
    bnd = [x["rel"] for x in F["runs"] if x["name"].startswith("bnd K=")]
    return max(bnd), F, out / "r"


@pytest.mark.parametrize("cid", [c[0] for c in CASES])
def test_sharded_gradient(tmp_path_factory, cid):
    """fc bitwise vs P = 1, the gradient finite on every lane incl. padding, padding tiles exactly 0."""
    procs, R = _result(tmp_path_factory, cid)
    assert R["g1_finite"]
    print(f"{cid}: fc {R['J1_printed']}")
    for p in procs.split(","):
        r = R[f"P{p}"]
        print(f"  P={p}: fc bitwise {r['J_bitwise']}, bits differ {r['bits_differ']}, rel {r['rel']:.3e}")
        assert r["J_bitwise"], (p, r["J_printed"], R["J1_printed"])
        assert r["finite"] and r["pad_zero"], (p, r)


@pytest.mark.parametrize("cid", [c[0] for c in CASES])
def test_gradient_difference_criterion(tmp_path_factory, cid):
    """max|g_P - g_1| <= TOL max|g_1| (R5's criterion) for the short windows; for the FLOOR_CASES <= C_FLOOR times
    the window's own 1-ulp floor at P = 1 (decision 18), with its negative control."""
    procs, R = _result(tmp_path_factory, cid)
    rel = {p: R[f"P{p}"]["rel"] for p in procs.split(",")}
    if cid not in FLOOR_CASES:
        print(f"{cid}: rel {rel} (TOL {TOL})")
        assert all(v <= TOL for v in rel.values()), rel
        return
    floor, F, ref = _floor(cid)
    bound = C_FLOOR * floor
    print(f"{cid}: rel {rel}; 1-ulp floor {floor:.3e} (boundary injections "
          f"{[(x['name'], round(x['rel'], 17)) for x in F['runs'] if x['name'].startswith('bnd')]}); bound {bound:.3e}")
    assert F["base"]["grad_bitwise_vs_ref"], F["base"]           # the instruments change no bit of g_1
    assert 0.0 < floor < 1e-10, floor                               # a measured rounding floor, not a broken run
    assert all(v <= bound for v in rel.values()), (rel, bound)
    # negative control: a planted difference of 1e-11 max|g_1| at one point of the real g_1 fails the bound
    g1 = np.load(ref / "g1.npy")
    gp = g1.copy()
    i = np.unravel_index(np.argmax(np.abs(g1)), g1.shape)
    gp[i] += 1e-11 * np.max(np.abs(g1))
    planted = float(np.max(np.abs(gp - g1)) / np.max(np.abs(g1)))
    assert planted > bound, (planted, bound)
