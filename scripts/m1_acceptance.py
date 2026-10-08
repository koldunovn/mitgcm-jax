#!/usr/bin/env python3
"""M1 acceptance (plan Task 18), one variant per call (scripts/m1_acceptance.sbatch runs it on a compute node):

    m1_acceptance.py EXPERIMENT INPUT OUT_DIR [--pn N]

1. the CLI's own function (mitjax.drivers.run.run, what `python -m mitjax run` calls) into OUT_DIR/cli: the whole run,
   its rundir/output.txt and pickups; wall time;
2. tools/testreport_jax.py digits per check-list variable: ours vs results/, the yardstick (our gfortran oracle's
   STDOUT vs results/), ours vs the oracle; every %MON record and MONITOR banner vs the oracle STDOUT
   (advect_gate.run_verdict's selection); every pickup file of the oracle run directory byte for byte;
3. with --pn N (N > 1): the driver's step at P = N (jit(shard_map(check_vma=True)) on N fake CPU devices) vs P = 1
   for the whole run, every carry leaf and per-step output bit for bit (go_gate.run_steps_p), with the exchange maps
   the driver's Model chose (the cube's W2 topology on the cubed sphere, else the build's probed maps).
--pn-only (the M2 acceptance runs) skips 1-2; with N > xla_flags.FAKE_DEVICES it sets the gate
flags with N fake devices (as test_cs32_go_run.py does), so the CLI run of 1 stays a separate process with the plain
gate flags.
Writes OUT_DIR/result.json (every number, the paths of the files they come from). Never deletes anything.
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

from mitjax.xla_flags import FAKE_DEVICES, gate_xla_flags, set_gate_xla_flags  # noqa: E402


def _pn_argv(argv):
    return int(argv[argv.index("--pn") + 1]) if "--pn" in argv else 0


if "--pn-only" in sys.argv and _pn_argv(sys.argv) > FAKE_DEVICES:      # before any backend initialises
    os.environ["XLA_FLAGS"] = gate_xla_flags("").replace(f"device_count={FAKE_DEVICES}",
                                                       f"device_count={_pn_argv(sys.argv)}")
else:
    set_gate_xla_flags()


def model_maps(e):
    """The exchange maps drivers/model.Model.__init__ builds its Exchanger from (drivers/model.py: the cube's W2
    topology when ALLOW_EXCH2 and the cubed-sphere exchange, else the build's probed maps)."""
    from mitjax.eesupp.exch_maps import load_cube_maps, load_maps
    from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
    cfg = e.cfg
    if cfg.cpp.ALLOW_EXCH2 and use_cubed_sphere_exchange(e):
        return load_cube_maps(cfg.experiment, cfg.input_dir)
    return load_maps(cfg.experiment, code=cfg.code_dir)


class _Skip(Exception):
    pass


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--pn", type=int, default=0)
    ap.add_argument("--pn-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True)
    res = {"experiment": a.experiment, "input": a.input}
    from mitjax import paths
    from mitjax.drivers.run import run
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    exp_dir = paths.UPSTREAM / "verification" / a.experiment
    t0 = time.time()
    try:
        if a.pn_only:
            raise _Skip
        output = run(exp_dir, a.input, out / "cli")
        res["wall_s"] = time.time() - t0
        res["output"] = str(output)
        o = mg.oracle(a.experiment, a.input)
        res["oracle_stdout"] = str(o.stdout_path)
        records = Path(output).read_text().splitlines()
        rs = type("R", (), {"records": records})
        diffs, rows, nblocks = ag.run_verdict(a.experiment, a.input, rs, o)
        res["mon_diffs"], res["mon_blocks"] = diffs, nblocks
        res["rows"] = [list(r) for r in rows]
        theirs = sorted(p.name for p in o.stdout_path.parent.iterdir() if p.name.startswith("pickup"))
        rundir = Path(output).parent
        import filecmp
        res["pickups"] = {f: (rundir / f).exists() and filecmp.cmp(rundir / f, o.stdout_path.parent / f,
                                                                     shallow=False) for f in theirs}
        res["pickups_extra"] = sorted(p.name for p in rundir.iterdir() if p.name.startswith("pickup")
                                      and p.name not in theirs)
    except _Skip:
        pass
    except Exception:
        res["error"] = traceback.format_exc()
    if a.pn > 1:
        try:
            import jax
            import numpy as np
            from mitjax.drivers.model import Model
            from mitjax.drivers.run import load_experiment, make_rundir
            from mitjax.tests import go_gate as G
            e = load_experiment(exp_dir, a.input)
            m = Model(e, make_rundir(exp_dir, a.input, out / "pn"))
            n = m.prm.time.nTimeSteps
            t1 = time.time()
            c1, o1 = G.run_steps_p(m, n)
            cN, oN = G.run_steps_p(m, n, a.pn, maps=model_maps(e))
            la, lb = jax.tree.leaves(c1), jax.tree.leaves(cN)
            nd = [int(np.count_nonzero(np.ascontiguousarray(x, np.float64).view(np.int64)
                                       != np.ascontiguousarray(y, np.float64).view(np.int64))) for x, y in zip(la, lb)]
            same_out = all(np.array_equal(np.asarray(x), np.asarray(y))
                           for p_, q_ in zip(o1, oN) for x, y in zip(jax.tree.leaves(p_), jax.tree.leaves(q_)))
            res["pn"] = {"P": a.pn, "steps": n, "leaves": len(la), "differing_leaves": sum(x > 0 for x in nd),
                         "outputs_equal": bool(same_out), "wall_s": time.time() - t1}
        except Exception:
            res["pn_error"] = traceback.format_exc()
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "pickups")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
