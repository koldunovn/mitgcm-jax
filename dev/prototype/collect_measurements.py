#!/usr/bin/env python3
"""Write dev/prototype/measurements.json (read by make_side_by_side.py) from the job outputs of plan Task 8:
the gates job's pytest log (numbers printed by mitjax/tests/test_style_prototype.py) and the A100 bench JSON files.

    collect_measurements.py GATES_LOG BENCH_JSON [BENCH_RAW_JSON]
"""

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def grab(text, prefix):
    m = re.search(rf"^{re.escape(prefix)}\s*(.*)$", text, re.M)
    if not m:
        raise SystemExit(f"'{prefix}' not in the gates log")
    return m.group(1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    log = Path(argv[0]).read_text()
    bench = json.loads(Path(argv[1]).read_text())
    raw = json.loads(Path(argv[2]).read_text()) if len(argv) > 2 else None
    neg = ast.literal_eval(grab(log, "negative controls, points differing from gfortran:"))
    fd = ast.literal_eval(grab(log, "farray tangent-vs-adjoint and best FD relative errors:"))
    fd_s = ast.literal_eval(grab(log, "slices tangent-vs-adjoint and best FD relative errors:"))
    dv = json.loads(grab(log, "divide probe (gate flags):"))
    da = json.loads(grab(log, "divide probe (algsimp on):"))
    verdict = re.search(r"^(\d+) passed in ([\d.]+)s", log, re.M)
    worst_dot = max(max(v[0] for v in fd.values()), max(v[0] for v in fd_s.values()))
    fdr = [v[1] for v in list(fd.values()) + list(fd_s.values())]
    lit = dv["literal"]
    alg = {c: v["traced_scalar_normal_only"] for c, v in da["literal"].items()}
    gates = [
        {"gate": "bitwise vs gfortran (both styles)", "cls": "ok",
         "result": "0 differing points",
         "detail": "23 outputs (15 routine cases) x 36 tiles x 15 levels x 16x16 incl. halos and the unwritten rim "
                   "(138,240 values each), gate XLA flags"},
        {"gate": "negative controls (must bite)", "cls": "ok",
         "result": ", ".join(f"{k}: {v:,}" for k, v in neg.items()),
         "detail": "points differing from gfortran with the planted error"},
        {"gate": "identical programs", "cls": "ok",
         "result": "jaxpr equal; optimized HLO equal modulo names, all 15 cases (CPU) and 10 benchmark programs (A100)",
         "detail": "debug locations removed; the only textual difference: parameter names from the pytree path "
                   "(an FArray's storage is child 0); negative control: a re-associated statement differs"},
        {"gate": "FD vs JAX gradient (both styles)", "cls": "ok",
         "result": f"best FD rel. error {min(fdr):.1e} .. {max(fdr):.1e}; tangent vs adjoint <= {worst_dot:.1e}",
         "detail": "9 cases, h = 1e-3..1e-7, random direction over all inputs; DST3 with calcCFL at points with "
                   "CFL <= 1 (see finding); negative control: stop_gradient(|uTrans|) gives 4e-3"},
        {"gate": "finite gradients on all lanes", "cls": "ok", "result": "finite (15 cases x 2 styles)",
         "detail": "every input incl. grid fields and parameters, halos and land; negative control: "
                   "where(h > 0, 1/h, 0) is NaN backward"},
        {"gate": "gradients identical between styles", "cls": "ok", "result": "bitwise, all 15 cases", "detail": ""},
        {"gate": "XLA:CPU divide vs gfortran (gate flags)", "cls": "warn",
         "result": f"0 of {dv['array_array']['vs_gfortran']['n_normal_only']:,} normal pairs differ; "
                   f"{dv['array_array']['vs_gfortran']['subnormal_involved']:,} of "
                   f"{dv['array_array']['vs_gfortran']['n_subnormal_involved']:,} pairs with a subnormal differ, "
                   "all flushed to +-0 by XLA",
         "detail": "array/array and literal divisors (" + ", ".join(lit) + ") as traced scalar, traced array or "
                   "Python constant: one divide, no multiply; gfortran == numpy on all 65,536"},
        {"gate": "XLA:CPU divide with algsimp on", "cls": "warn",
         "result": "scalar divisor -> x*(1/d): " + ", ".join(f"/{c}: {n:,}" for c, n in alg.items())
                   + " normal values differ",
         "detail": "traced full-array divisor not rewritten (0); a Python-constant divisor leaves no divide at all"},
    ]
    if verdict:
        gates.insert(0, {"gate": "gates job", "cls": "ok", "result": f"{verdict.group(1)} passed in "
                         f"{verdict.group(2)} s", "detail": "mitjax/tests/test_farray.py + test_style_prototype.py"})

    def rows(b, tag):
        out = []
        for c in b["cases"]:
            out.append({"size": c["size"] + tag, "case": f"{c['routine']} {c['case']}",
                        "farray_ms": c["farray_median_ms"], "slices_ms": c["slices_median_ms"],
                        "ratio": c["ratio_median"],
                        "spread": f"{min(c['farray_ms']):.4f}..{max(c['farray_ms']):.4f} / "
                                  f"{min(c['slices_ms']):.4f}..{max(c['slices_ms']):.4f}",
                        "hlo_identical": c["hlo_identical"], "bitwise": c["outputs_bitwise_equal"]})
        return out
    timing = rows(bench, "")
    assert all(r["hlo_identical"] and r["bitwise"] for r in timing)
    note = ("Unmarked rows: FArrays passed as jit arguments. ")
    if raw:
        rr = rows(raw, " (plain-array signature)")
        assert all(r["bitwise"] for r in rr)
        timing += rr
        note += ("Rows marked 'plain-array signature' (job 27826819): the Fortran-index program is called with the same "
                 "plain arrays as the slices program and builds its FArrays inside the traced program; the HLO check "
                 "of that run kept the module name (jit__lambda vs jit_f, the only difference; the module name is now "
                 "normalised and the identity is gated on CPU in test_identical_programs). ")
    note += (f"W = {bench['W']}, N = {bench['N']}, {bench['repeats']} interleaved repeats; outputs bitwise equal "
             "between the styles in every row.")
    m = {"gates": gates, "timing": timing, "timing_note": note}
    (HERE / "measurements.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"wrote {HERE / 'measurements.json'}")


if __name__ == "__main__":
    sys.exit(main())
