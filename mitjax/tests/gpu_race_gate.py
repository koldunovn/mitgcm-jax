"""The program of the GPU race of 2026-10-09 (mitjax/xla_flags.py docstring, `multi_output_fusion`): the vjp forward
(jax.vjp of drivers/grad's objective with schedule "none", so every intermediate is a residual) of the first 3 steps
of tutorial_global_oce_optim/input_ad, gentim2d control xx_qnet record 1: the smallest program that raced on one A100
(136 of its 5534 outputs differed between calls; job 27994893). `program(out)` builds it.

Run as a script (`python -m mitjax.tests.gpu_race_gate --out DIR`, the control of test_gpu_race.py), it compiles the
program with the XLA flags of the environment and prints the races mitjax/tests/hlo_race.py finds in it, as one JSON
line. JAX's backend is started before mitjax.api is called, so the API's own flags (which would add
multi_output_fusion) do not reach XLA: the environment's XLA_FLAGS are the ones compiled with."""

import argparse
import json
import os
from pathlib import Path

EXPERIMENT, VARIANT, STEPS = "tutorial_global_oce_optim", "input_ad", 3


def program(out, steps=STEPS):
    """(vjp_fwd, args): vjp_fwd(*args) = jax.vjp of the objective of the first `steps` steps at theta = 0, i.e. the
    cost fc and the residuals of the reverse sweep (mitjax/drivers/adjoint_run.problem, drivers/grad._objective)."""
    import jax

    from mitjax import api, paths
    from mitjax.drivers import adjoint_run as AR
    from mitjax.drivers import grad as G
    exp = api.load(paths.UPSTREAM / "verification" / EXPERIMENT, VARIANT)
    m = exp.model(Path(out) / "model")
    step, model, st0, xs, th0, params_fn, final_cost = AR.problem(m, 1, 1)
    J = G._objective(step, schedule="none", segments=None, cost=None, final_cost=final_cost,
                     init_fn=lambda t, s: s, params_fn=params_fn)

    def vjp_fwd(th, mo, s0, x):
        return jax.vjp(lambda t: J(t, mo, s0, x), th)
    return vjp_fwd, (th0, model, st0, xs[:steps])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new directory for the model's run directory")
    a = ap.parse_args(argv)
    flags = os.environ.get("XLA_FLAGS", "")
    import jax
    devices = jax.devices()                       # the backend starts here and reads XLA_FLAGS = `flags`
    from mitjax.tests import hlo_race as H
    fn, args = program(a.out)
    races = H.find(jax.jit(fn).lower(*args).compile().as_text())
    print(json.dumps({"xla_flags": flags, "devices": [str(d) for d in devices],
                      "races": [r._asdict() for r in races]}))


if __name__ == "__main__":
    main()
