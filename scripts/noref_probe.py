"""Portability probe (docs plan 20261006, Task 1): run mitjax the way a user without our machine would, and record
every failure with the file:line that raised it.

    python scripts/noref_probe.py --out <new dir> [--cases a,b] [--levels L0,L1,...]

Set-up (all under <out>, never deleted): `MJX_REFERENCE` and `MJX_REFERENCE_RUNS` point to empty directories,
`MJX_WORK` to a new directory, `MJX_RUNS` to <out>/runs; `MJX_UPSTREAM` stays the MITgcm clone (the one thing a user
has). Every experiment a case uses is COPIED to <out>/exps/<experiment> (outside the MITgcm tree; only its code*,
input* and results directories, symlinks dereferenced), with the sibling experiments its prepare_run reaches.

Each (case, level) runs in its own subprocess (the paths are read once, at import). A level adds stand-ins for the
fixes not made yet, so that the failures behind the first one become visible:
    L0  nothing: the user's situation
    L1  the experiment in the MITgcm tree instead of the copy (stand-in for plan Task 2)
    L2  + L1 + toolchain: cpp_options.default_toolchain from the real oracle build records (stand-in for plan Task 4)
    L3  + L2 + exchange maps: exch_maps.map_dir() = the real $MJX_REFERENCE/exch_maps (stand-in for plan Task 3)
    PLANT  the negative control of mitjax/tests/test_no_reference.py: the toolchain read re-pointed at
           $MJX_REFERENCE (cpp_options.default_toolchain = oracle_toolchain), counted as a foreign read
(Since docs plan 20261006 Tasks 2-4 the stand-ins are no longer needed: L0 runs; mitjax/tests/test_no_reference.py
runs the L0 children of the forward, gradient and sharded cases and of the API calls (Task 6) as a tier-1x gate.)
An audit hook records every file the case opens, lists or executes outside <out>, the MITgcm clone, the repository
and the Python installation (`foreign` in the record), with the innermost mitjax frame that did it. A stand-in's own
reads are listed under `shim`.

Results: <out>/results/<case>-<level>.json (status, exception, raise site, innermost mitjax frame, traceback, foreign
reads) and <out>/SUMMARY.txt (one line per case and level). Exit code 0 always (it measures; it does not gate).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# case -> (experiment, variant, sibling experiments its prepare_run reaches (copied next to it), what it does)
CASES = {
    "fwd_gyre": ("tutorial_barotropic_gyre", "input", (), "forward, whole run (python -m mitjax run)"),
    "fwd_col": ("1D_ocean_ice_column", "input", (), "forward, whole run (python -m mitjax run)"),
    "fwd_lab": ("lab_sea", "input", (), "forward, 2 steps (drivers.run.forward)"),
    "fwd_cs32": ("global_ocean.cs32x15", "input", ("tutorial_held_suarez_cs",), "forward, 2 steps"),
    "grad_col": ("1D_ocean_ice_column", "input_ad", ("isomip",), "gradient (GenarrAdjoint, xx_theta)"),
    "shard_p2": ("tutorial_baroclinic_gyre", "input", (), "sharded forward P=2 (4 tiles), 2 steps vs P=1"),
    # lane API (docs plan 20261006 Task 6): the API calls (mitjax/api.py)
    "api_gyre": ("tutorial_barotropic_gyre", "input", (), "API: load, run, results, compare"),
    "api_col": ("1D_ocean_ice_column", "input_ad", ("isomip",), "API: gradient (adxx file), grdchk, compare (adm)"),
    "api_shard": ("tutorial_global_oce_optim", "input_ad", ("tutorial_global_oce_latlon",),
                  "API: gradient on devices=2 (4 tiles)"),
}
LEVELS = ("L0", "L1", "L2", "L3")
COPY_DIRS = ("code", "input", "results")       # prefixes of the experiment directories copied


def copy_experiment(upstream, exp, dest_root):
    """<dest_root>/<exp> with the experiment's code*, input* and results directories (symlinks dereferenced)."""
    src = Path(upstream) / "verification" / exp
    dst = Path(dest_root) / exp
    if dst.exists():
        return dst
    dst.mkdir(parents=True)
    for d in sorted(src.iterdir()):
        if d.is_dir() and d.name.startswith(COPY_DIRS):
            shutil.copytree(d, dst / d.name, symlinks=False)
    return dst


# ------------------------------------------------------------------------------------------------------ the child

class Reads:
    """sys.addaudithook: file opens / listings / executions outside the allowed roots."""

    def __init__(self, allowed):
        self.allowed = [str(Path(a).resolve()) for a in allowed]
        self.allowed += [str(Path(a)) for a in allowed]
        self.seen = {}
        self.shim = False

    def __call__(self, event, args):
        if event not in ("open", "os.listdir", "os.scandir", "subprocess.Popen", "os.exec", "glob.glob"):
            return
        p = args[0] if args else None
        if event == "subprocess.Popen":     # (executable, args, cwd, env)
            p = args[0] if args[0] is not None else (list(args[1])[0] if args[1] else None)
        if isinstance(p, (bytes, bytearray)):
            p = p.decode(errors="replace")
        if not isinstance(p, (str, os.PathLike)):
            return
        p = os.fspath(p)
        if not os.path.isabs(p) or any(p.startswith(a + os.sep) or p == a for a in self.allowed):
            return
        if p.startswith(("/proc", "/sys", "/dev", "/tmp", "/etc", "/usr", "/lib", "/lib64", "/scratch")):
            return
        key = (event, p)
        if key in self.seen:
            return
        site = None
        for fr in reversed(traceback.extract_stack()[:-1]):
            if _ours(fr.filename) and not fr.filename.endswith("noref_probe.py"):
                site = f"{os.path.relpath(fr.filename, REPO)}:{fr.lineno}"
                break
        self.seen[key] = {"event": event, "path": p, "site": site, "shim": self.shim}


def _ours(filename):
    """A frame of this repository's mitjax/, reference/ or tools/ (not the env's site-packages)."""
    return any(filename.startswith(str(REPO / d) + os.sep) for d in ("mitjax", "reference", "tools"))


def _site(tb_list, pred):
    for fr in reversed(tb_list):
        if pred(fr.filename):
            return f"{os.path.relpath(fr.filename, REPO)}:{fr.lineno} ({fr.name})"
    return None


def apply_shims(level, real_ref, reads):
    from mitjax.config import cpp_options
    if level == "PLANT":
        cpp_options.default_toolchain = cpp_options.oracle_toolchain       # a read of $MJX_REFERENCE/bin (planted)
        return
    if level >= 2:
        reads.shim = True
        recs = sorted(p for p in (Path(real_ref) / "bin").glob("*") if (p / "Makefile").is_file())
        reads.shim = False

        def default_toolchain():
            reads.shim = True
            try:
                return cpp_options.toolchain_from_record(recs[0])
            finally:
                reads.shim = False
        cpp_options.default_toolchain = default_toolchain
    if level >= 3:
        from mitjax.eesupp import exch_maps
        exch_maps.map_dir = lambda: Path(real_ref) / "exch_maps"


def run_case(case, exp_dir, out):
    import jax
    import numpy as np
    exp, inp, _, _ = CASES[case]
    from mitjax.drivers.run import forward, load_experiment, make_rundir, run
    info = {}
    if case.startswith("api_"):                     # lane API: the API calls (mitjax/api.py)
        import mitjax
        exp = mitjax.load(exp_dir, inp)
        if case == "api_gyre":
            r = exp.run(out=out / "run")
            rep = mitjax.compare(r, exp.results())
            info.update(output_lines=sum(1 for _ in open(r.output)), verdict=rep.verdict, nfields=len(r.fields))
            gr = exp.grid()                         # lane APIF: the grid without a run directory, global fields
            sz = exp.config.cfg.size
            info.update(grid_xC=list(gr.global_field("xC").shape) == [sz.Ny, sz.Nx],
                        global_eta=list(r.global_field("etaN").shape) == [sz.Ny, sz.Nx])
        elif case == "api_col":
            g = exp.gradient(out=out / "grad")
            a = g.adxx[1]
            info.update(fc=g.fc, finite=bool(np.all(np.isfinite(a))), gnorm=float(np.sqrt(np.sum(a * a))),
                        files=[p.name for p in g.files])
            jax.clear_caches()
            chk = exp.grdchk(out=out / "grdchk")
            info.update(verdict=mitjax.compare(chk, exp.results()).verdict, nchecks=len(chk.checks))
        elif case == "api_shard":
            g = exp.gradient(out=out / "grad", devices=2)
            info.update(fc=g.fc, finite=bool(all(np.all(np.isfinite(v)) for v in g.adxx.values())))
        return info
    if case in ("fwd_gyre", "fwd_col"):
        o = run(exp_dir, inp, out / "run")
        info["output_lines"] = sum(1 for _ in open(o))
        return info
    from mitjax.drivers.model import Model
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    e = load_experiment(exp_dir, inp)
    m = Model(e, make_rundir(exp_dir, inp, out / "run"))
    if case in ("fwd_lab", "fwd_cs32"):
        res = forward(m, nTimeSteps=2)
        info["records"] = len(res.records)
    elif case == "grad_col":
        from mitjax.drivers.adjoint_run import GenarrAdjoint
        fc, g = GenarrAdjoint(m, key=(3, 1)).value_and_grad()
        g = np.asarray(g)
        info.update(fc=float(fc), gnorm=float(np.sqrt(np.sum(g * g))), finite=bool(np.all(np.isfinite(g))))
    elif case == "shard_p2":
        from mitjax.tests.go_gate import run_steps_p
        c1, _ = run_steps_p(m, 2)
        c2, _ = run_steps_p(m, 2, nproc=2, maps=m.exch_maps if getattr(m, "exch_maps", None) else None)
        l1, l2 = jax.tree.leaves(c1), jax.tree.leaves(c2)
        info["bitwise_p2_p1"] = len(l1) == len(l2) and all(
            np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)) for a, b in zip(l1, l2))
    return info


def child(a):
    out = Path(a.out)
    level = "PLANT" if a.level == "PLANT" else LEVELS.index(a.level)
    case = a.child
    exp, inp, sibs, what = CASES[case]
    work = out / "work" / f"{case}-{a.level}"
    allowed = [out, a.upstream, REPO, sys.prefix, sys.base_prefix, Path(sys.executable).resolve().parents[1]]
    reads = Reads(allowed + ([sys.pycache_prefix] if sys.pycache_prefix else []))
    sys.addaudithook(reads)
    rec = {"case": case, "level": a.level, "experiment": exp, "variant": inp, "what": what}
    t0 = time.time()
    try:
        apply_shims(level, a.real_reference, reads)
        exp_dir = (Path(a.upstream) / "verification" / exp) if level != "PLANT" and level >= 1 else out / "exps" / exp
        rec["experiment_dir"] = str(exp_dir)
        rec["info"] = run_case(case, exp_dir, work)
        rec["status"] = "OK"
    except BaseException as err:      # noqa: BLE001 -- a probe records every failure
        tb = traceback.extract_tb(err.__traceback__)
        rec["status"] = "FAIL"
        rec["exception"] = f"{type(err).__name__}: {err}"[:2000]
        rec["raised_at"] = _site(tb, lambda f: True)
        rec["mitjax_frame"] = _site(tb, _ours)
        rec["traceback"] = traceback.format_exception(err)[-30:]
    rec["seconds"] = round(time.time() - t0, 1)
    rec["foreign"] = sorted(reads.seen.values(), key=lambda r: (r["shim"], r["path"]))
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / f"{case}-{a.level}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("case", "level", "status", "exception", "raised_at", "mitjax_frame",
                                               "seconds")}))


# ------------------------------------------------------------------------------------------------------ the parent

def parent(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (a new directory per probe; nothing is overwritten)")
    sys.path.insert(0, str(REPO))
    from mitjax import paths       # the real locations, before the overrides
    upstream, real_ref = paths.UPSTREAM.resolve(), paths.REFERENCE.resolve()
    for d in ("empty_reference", "empty_reference_runs", "work", "runs", "exps", "results"):
        (out / d).mkdir(parents=True)
    cases = a.cases.split(",") if a.cases else list(CASES)
    levels = a.levels.split(",") if a.levels else list(LEVELS)
    for c in cases:
        exp, _, sibs, _ = CASES[c]
        for e in (exp,) + sibs:
            copy_experiment(upstream, e, out / "exps")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MJX_")}
    env.update(MJX_WORK=str(out / "work"), MJX_UPSTREAM=str(upstream), MJX_REFERENCE=str(out / "empty_reference"),
               MJX_REFERENCE_RUNS=str(out / "empty_reference_runs"), MJX_RUNS=str(out / "runs"),
               PYTHONPATH=str(REPO), JAX_PLATFORMS="cpu")
    lines = []
    for c in cases:
        for lv in levels:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", c, "--level", lv,
                                "--out", str(out), "--upstream", str(upstream), "--real-reference", str(real_ref)],
                               env=env, cwd=str(out / "work"), capture_output=True, text=True)
            (out / "results" / f"{c}-{lv}.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr[-20000:])
            p = out / "results" / f"{c}-{lv}.json"
            rec = json.loads(p.read_text()) if p.exists() else {"status": f"CRASH rc={r.returncode}"}
            nf = sum(1 for f in rec.get("foreign", []) if not f["shim"])
            line = (f"{c:9s} {lv}  {rec['status']:5s} {rec.get('seconds', '-')!s:>7}s  foreign={nf}  "
                    f"{rec.get('mitjax_frame') or ''}  {(rec.get('exception') or '')[:160]}")
            print(line, flush=True)
            lines.append(line)
            if rec["status"] == "OK" and not a.all_levels:
                break       # a passing level: the higher stand-ins add nothing
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases")
    ap.add_argument("--levels")
    ap.add_argument("--all-levels", action="store_true", help="run the higher levels after a passing one too")
    ap.add_argument("--child")
    ap.add_argument("--level")
    ap.add_argument("--upstream")
    ap.add_argument("--real-reference")
    a = ap.parse_args(argv)
    if a.child:
        sys.path.insert(0, str(REPO))
        child(a)
    else:
        parent(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
