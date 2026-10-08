#!/usr/bin/env python3
"""XLA:CPU float64 divide vs gfortran (plan Task 8 [F§2]; L-CONF-2 of docs/LESSONS_CARRIED.md).

    divide_probe.py RUNDIR --flags gate|algsimp_on|default      (prints one JSON line)

Runs in a fresh process because XLA reads XLA_FLAGS once: `gate` = mitjax.xla_flags (no FMA, no algsimp, 4 devices),
`algsimp_on` = the gate flags without --xla_disable_hlo_passes=algsimp, `default` = no XLA_FLAGS at all.
Inputs: the replay harness's divide probe (reference/replay/code/the_main_loop.F, section 4c): 65536 pairs (a, b)
over many magnitudes (replay_io.make_divide_pairs: normal range, full exponent range, subnormal-adjacent quotients,
subnormal dividends, large ratios, literal-like divisors) and gfortran's a/b, a/3.D0, a/1000.D0, a/7.D0, a/0.1D0
(-O0, -ffp-contract=off). Measured, as counts of float64 values whose bit patterns differ from gfortran's:
  1. array / array: jit(a / b) with both operands traced arrays, and numpy's a / b;
  2. a / literal c (c = 3, 1000, 7, 0.1): the divisor as a traced 0-d operand (broadcast in the program), as a
     traced full array, and as a Python constant (a literal in the program);
  3. the optimized HLO of each variant: number of divide and multiply instructions (a reciprocal rewrite shows as
     a multiply with no divide).
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATEGORIES = ("normal", "full_range", "to_subnormal", "subnormal_a", "large_ratio", "literal_like")


def set_flags(kind):
    if kind == "default":
        os.environ.pop("XLA_FLAGS", None)
        return ""
    sys.path.insert(0, str(REPO))
    from mitjax.xla_flags import GATE_FLAGS, set_gate_xla_flags
    if kind == "gate":
        return set_gate_xla_flags()
    if kind == "algsimp_on":
        os.environ["XLA_FLAGS"] = " ".join(f for f in GATE_FLAGS if not f.startswith("--xla_disable_hlo_passes"))
        return os.environ["XLA_FLAGS"]
    raise SystemExit(f"unknown flag set {kind}")


def category_slices(n):
    q = n // 8
    edges = [0, 2 * q, 4 * q, 5 * q, 6 * q, 7 * q, n]
    return {c: slice(edges[m], edges[m + 1]) for m, c in enumerate(CATEGORIES)}


def count_ops(hlo_text):
    return {"divide": len(re.findall(r"\bdivide\(", hlo_text)), "multiply": len(re.findall(r"\bmultiply\(", hlo_text))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir")
    ap.add_argument("--flags", required=True, choices=("gate", "algsimp_on", "default"))
    a = ap.parse_args(argv)
    flags = set_flags(a.flags)

    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    spec = importlib.util.spec_from_file_location("_rio", REPO / "reference" / "replay" / "replay_io.py")
    rio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rio)
    A, B, Q = rio.read_divide(a.rundir)
    cats = category_slices(len(A))

    tiny = np.finfo(np.float64).tiny
    subn = lambda v: (v != 0) & (np.abs(v) < tiny)

    def diff(x, ref, a=A):
        """Differing bit patterns: total, per input category, split by whether a subnormal is involved (dividend or
        gfortran's quotient), and whether every differing value with a subnormal involved is XLA's +-0 (FTZ/DAZ)."""
        x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
        bad = x.view(np.int64) != ref.view(np.int64)
        sub = subn(a) | subn(ref)
        return {"total": int(bad.sum()), "normal_only": int((bad & ~sub).sum()), "n_normal_only": int((~sub).sum()),
                "subnormal_involved": int((bad & sub).sum()), "n_subnormal_involved": int(sub.sum()),
                "subnormal_diffs_are_zero": bool(np.all(x[bad & sub] == 0)),
                **{c: int(bad[s].sum()) for c, s in cats.items()}}

    res = {"flags": a.flags, "XLA_FLAGS": flags, "jax": jax.__version__, "n": len(A)}
    aj, bj = jnp.asarray(A), jnp.asarray(B)
    f_aa = jax.jit(lambda x, y: x / y)
    np.seterr(all="ignore")                     # overflow/underflow are part of the probe
    res["numpy_vs_gfortran"] = diff(A / B, Q[:, 0])
    res["array_array"] = {"vs_gfortran": diff(f_aa(aj, bj), Q[:, 0]),
                          "ops": count_ops(f_aa.lower(aj, bj).compile().as_text())}
    res["literal"] = {}
    for m, c in enumerate(rio.DIVIDE_LITERALS):
        ref = Q[:, 1 + m]
        f_s = jax.jit(lambda x, d: x / d)
        f_c = jax.jit(lambda x, c=c: x / c)
        d0 = jnp.asarray(c, jnp.float64)
        dfull = jnp.full_like(aj, c)
        res["literal"][repr(c)] = {
            "numpy": diff(A / c, ref)["total"],
            "traced_scalar_detail": diff(f_s(aj, d0), ref),
            "traced_scalar": diff(f_s(aj, d0), ref)["total"],
            "traced_scalar_ops": count_ops(f_s.lower(aj, d0).compile().as_text()),
            "traced_array": diff(f_s(aj, dfull), ref)["total"],
            "python_constant": diff(f_c(aj), ref)["total"],
            "python_constant_normal_only": diff(f_c(aj), ref)["normal_only"],
            "traced_scalar_normal_only": diff(f_s(aj, d0), ref)["normal_only"],
            "traced_array_normal_only": diff(f_s(aj, dfull), ref)["normal_only"],
            "python_constant_ops": count_ops(f_c.lower(aj).compile().as_text()),
            "reciprocal_multiply_numpy": diff(A * (1.0 / c), ref)["total"],
        }
    print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
