#!/usr/bin/env python3
"""A100 timing of the two code styles (plan Task 8; [F§6]): warm runs only, per call (t_N - t_W) / (N - W).

    bench_gpu.py OUT_JSON [--repeats 5] [--W 5] [--N 105]

Problem sizes: `go90` = global_ocean.90x40x15's SIZE.h (36 tiles of 10x10, OLx = OLy = 3, Nr = 15) and `llc90` =
an LLC90-like layout (13 tiles of 90x90, OLx = OLy = 4, Nr = 50), on a synthetic grid with land and partial cells
(proto_setup.synthetic_grid) and random inputs. Each program is one routine case for all levels (the caller's k
loop, proto_run.all_levels_fn), jitted with production XLA flags (no gate flags on GPU). The two styles are timed
interleaved (A B A B ...) `repeats` times; per call median and min/max are reported, plus the normalized optimized
HLO comparison of the GPU programs and a finiteness check of every output ([bench-finite]).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import mitjax  # noqa: E402,F401  (x64)
import proto_run as pr  # noqa: E402
import proto_setup as ps  # noqa: E402

SIZES = {"go90": dict(sNx=10, sNy=10, OLx=3, OLy=3, Nr=15, T=36),
         "llc90": dict(sNx=90, sNy=90, OLx=4, OLy=4, Nr=50, T=13)}
CASES = [("mom_calc_ke", 0), ("mom_calc_ke", 3), ("gad_dst3_adv_x", True), ("mom_vi_hdissip", (True, True, False)),
         ("mom_vi_hdissip", (True, True, True))]


def make_problem(size, seed=1):
    s = SIZES[size]
    cfg = ps.Cfg(sNx=s["sNx"], sNy=s["sNy"], OLx=s["OLx"], OLy=s["OLy"], Nr=s["Nr"], ALLOW_AUTODIFF=False,
                 ALLOW_DIAGNOSTICS=True, MOM_VI_ORIGINAL_VISCA4=False, OLD_DST3_FORMULATION=False,
                 useDiagnostics=False)
    g = ps.synthetic_grid(cfg, s["T"], seed=seed)
    rng = np.random.default_rng(seed + 1)
    shp = (s["T"], s["Nr"], s["sNy"] + 2 * s["OLy"], s["sNx"] + 2 * s["OLx"])
    fields = {n: rng.standard_normal(shp) for n in pr.replay_io.IN3}
    for n, scale in (("uFld", 0.3), ("vFld", 0.3), ("uVel", 0.3), ("uTrans", 2e6), ("hDiv", 1e-6), ("vort3", 1e-6),
                     ("dStar", 1e-17), ("zStar", 1e-17), ("viscAh_Z", 5e5), ("viscAh_D", 5e5), ("viscA4_Z", 1e14),
                     ("viscA4_D", 1e14)):
        fields[n] = fields[n] * scale
    fields["hFacZ"] = np.abs(fields["hFacZ"]).clip(0, 1)
    return cfg, g, fields


def args_for(cfg, g, fields, style, routine):
    need = {"mom_calc_ke": ("uFld", "vFld", "KEprior"),
            "gad_dst3_adv_x": ("uTrans", "uVel", "uCFL", "tracer", "uTprior"),
            "mom_vi_hdissip": ("hDiv", "vort3", "dStar", "zStar", "hFacZ", "viscAh_Z", "viscAh_D", "viscA4_Z",
                               "viscA4_D", "uDissipPrior", "vDissipPrior")}[routine]
    return ({n: jnp.asarray(fields[n]) for n in need}, ps.make_grid(cfg, g, style), ps.make_params(g),
            jnp.asarray(86400.0))


def wrap_grid(cfg, grid_raw):
    """GRID.h common block of plain arrays -> of FArrays (inside a traced program)."""
    from mitjax.farray import FArray
    b = ps.bounds(cfg)
    return ps.Common(**{n: FArray(getattr(grid_raw, n), n, tiled=(n != "recip_deepFacC"), **{d: b[d] for d in dims})
                        for n, dims in ps.GRID_DECL.items()})


def per_call(f, args, W, N):
    def run(n):
        t0 = time.perf_counter()
        out = None
        for _ in range(n):
            out = f(*args)
        jax.block_until_ready(out)
        return time.perf_counter() - t0, out
    tW, _ = run(W)
    tN, out = run(N)
    return (tN - tW) / (N - W), out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_json")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--N", type=int, default=105)
    ap.add_argument("--sizes", default="go90,llc90")
    ap.add_argument("--raw-signature", action="store_true",
                    help="farray style called with plain arrays (FArrays built inside the jitted program)")
    a = ap.parse_args(argv)
    res = {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__, "raw_signature": a.raw_signature,
           "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""), "W": a.W, "N": a.N, "repeats": a.repeats, "cases": []}
    print(json.dumps({k: res[k] for k in ("devices", "jax", "XLA_FLAGS")}), flush=True)
    for size in a.sizes.split(","):
        cfg, g, fields = make_problem(size)
        for routine, case in CASES:
            fs, hlo, outs, argsd = {}, {}, {}, {}
            for style in ("farray", "slices"):
                body = pr.all_levels_fn(pr.STYLES[style], style, cfg, routine, case)
                if a.raw_signature and style == "farray":
                    # same jit signature as the slices style: plain arrays in, FArrays built inside the program
                    body = (lambda body: lambda fl, gr, pa, dt: body(fl, wrap_grid(cfg, gr), pa, dt))(body)
                    argsd[style] = args_for(cfg, g, fields, "slices", routine)
                else:
                    argsd[style] = args_for(cfg, g, fields, style, routine)
                f = jax.jit(body)
                t0 = time.perf_counter()
                comp = f.lower(*argsd[style]).compile()
                hlo[style] = pr.normalize_hlo(comp.as_text())
                outs[style] = jax.block_until_ready(f(*argsd[style]))
                fs[style] = f
                print(f"compiled {size} {routine} {case} {style} in {time.perf_counter() - t0:.1f} s", flush=True)
            times = {"farray": [], "slices": []}
            for _ in range(a.repeats):
                for style in ("farray", "slices"):
                    t, _ = per_call(fs[style], argsd[style], a.W, a.N)
                    times[style].append(t)
            finite = all(bool(jnp.all(jnp.isfinite(o))) for s in outs for o in outs[s])
            same = all(bool(jnp.array_equal(x, y)) for x, y in zip(outs["farray"], outs["slices"]))
            row = {"size": size, "routine": routine, "case": repr(case), "hlo_identical": hlo["farray"] == hlo["slices"],
                   "outputs_bitwise_equal": same, "finite": finite,
                   **{f"{s}_ms": [1e3 * t for t in times[s]] for s in times},
                   **{f"{s}_median_ms": 1e3 * float(np.median(times[s])) for s in times}}
            row["ratio_median"] = row["farray_median_ms"] / row["slices_median_ms"]
            res["cases"].append(row)
            print(f"[bench-finite] {finite}  {size} {routine} {case}: farray {row['farray_median_ms']:.4f} ms "
                  f"slices {row['slices_median_ms']:.4f} ms ratio {row['ratio_median']:.4f} "
                  f"(farray {min(row['farray_ms']):.4f}..{max(row['farray_ms']):.4f}, slices "
                  f"{min(row['slices_ms']):.4f}..{max(row['slices_ms']):.4f}) hlo_identical {row['hlo_identical']} "
                  f"bitwise {same}", flush=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1) + "\n")
    print(f"BENCH DONE -> {a.out_json}")


if __name__ == "__main__":
    sys.exit(main())
