"""mitjax without the Fortran oracle and with the experiment outside the MITgcm tree (docs plan 20261006 Task 5, the
no-reference test; built from scripts/noref_probe.py, plan Task 1).

A user's situation: only MJX_UPSTREAM (the MITgcm checkout), MJX_RUNS and MJX_CACHE (new directories) are set;
MJX_REFERENCE, MJX_REFERENCE_RUNS and MJX_WORK are unset; every experiment is a copy outside the tree (code*, input*,
results, symlinks dereferenced). Each case runs as scripts/noref_probe.py's child at level L0 (no stand-in) in its own
process, under an audit hook that records every file opened, listed or executed outside the probe directory, the
MITgcm checkout, the repository and the Python installation (system directories /usr, /lib, /etc, /proc, /tmp... not
counted):
    fwd_gyre   forward, whole run of tutorial_barotropic_gyre/input through `python -m mitjax run`'s driver
    fwd_cs32   forward, 2 steps of global_ocean.cs32x15/input: cubed sphere, prepare_run reaching the sibling
               experiment tutorial_held_suarez_cs (copied next to it), mitjax/make_rundir.py's run directory
    grad_col   the gradient of 1D_ocean_ice_column/input_ad (GenarrAdjoint, xx_theta): finite; its prepare_run
               links ones_64b.bin from the sibling experiment isomip/input_ad (copied next to it)
    shard_p2   tutorial_baroclinic_gyre, 2 steps sharded on P=2 devices bitwise equal to P=1
    api_gyre   the API (docs plan Task 6): mitjax.load + exp.run + mitjax.compare(run, exp.results()) passes;
               exp.grid() and Run.global_field give [Ny, Nx] arrays (lane APIF)
    api_col    the API: exp.gradient (adxx_theta written, finite) and exp.grdchk of 1D_ocean_ice_column/input_ad,
               compare(chk, exp.results()) passes testreport's adm check
    api_shard  the API: exp.gradient(devices=2) of tutorial_global_oce_optim/input_ad (4 tiles), finite
Each must finish OK with no foreign access except the C preprocessor the toolchain found.
Negative control: the same forward case with one read re-pointed at $MJX_REFERENCE (the toolchain from the oracle's
build records, level PLANT) and MJX_REFERENCE set to the real oracle is caught (foreign reads under the oracle).
Costs: about 15 min wall on a CPU node (the eight children in parallel; the API gradient + grdchk the longest).
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mitjax import paths

CASES = ("fwd_gyre", "fwd_cs32", "grad_col", "shard_p2", "api_gyre", "api_col", "api_shard")


def _probe():
    spec = importlib.util.spec_from_file_location("_mjx_noref_probe", paths.REPO / "scripts" / "noref_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _user_env(out, reference=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MJX_")}  # paths.py ignores the ECCO prefix
    env.update(MJX_UPSTREAM=str(paths.UPSTREAM.resolve()), MJX_RUNS=str(out / "runs"), MJX_CACHE=str(out / "cache"),
               PYTHONPATH=str(paths.REPO), JAX_PLATFORMS="cpu")
    if reference is not None:
        env["MJX_REFERENCE"] = str(reference)
    return env


def _child(probe, out, case, level, env):
    r = subprocess.run([sys.executable, probe.__file__, "--child", case, "--level", level, "--out", str(out),
                        "--upstream", env["MJX_UPSTREAM"], "--real-reference", str(out / "unused")],
                       env=env, cwd=str(out / "work"), capture_output=True, text=True)
    (out / "results" / f"{case}-{level}.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr[-20000:])
    p = out / "results" / f"{case}-{level}.json"
    return json.loads(p.read_text()) if p.exists() else {"status": f"CRASH rc={r.returncode}", "foreign": []}


def _cpp_program(env):
    """The C preprocessor the user's toolchain finds (the one foreign program a case may run)."""
    r = subprocess.run([sys.executable, "-c", "from mitjax.config import cpp_options as C; "
                        "print(C.default_toolchain().cpp[0])"], env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _foreign(rec, allowed_programs):
    return [f for f in rec.get("foreign", []) if not (f["event"] in ("subprocess.Popen", "os.exec")
                                                       and f["path"] in allowed_programs)]


def test_no_reference_forward_gradient_sharded():
    probe = _probe()
    real_ref = paths.REFERENCE.resolve()
    out = paths.RUNS / "tests_port" / f"noref-{os.getpid()}-{time.time_ns()}"
    for d in ("work", "runs", "cache", "exps", "results"):
        (out / d).mkdir(parents=True)
    for c in CASES:
        exp, _, sibs, _ = probe.CASES[c]
        for e in (exp,) + sibs:
            probe.copy_experiment(paths.UPSTREAM, e, out / "exps")
    env = _user_env(out)
    allowed = {_cpp_program(env)}
    jobs = [(c, "L0", env) for c in CASES] + [("fwd_gyre", "PLANT", _user_env(out, reference=real_ref))]
    with ThreadPoolExecutor(len(jobs)) as ex:
        recs = dict(zip([(c, lv) for c, lv, _ in jobs], ex.map(lambda j: _child(probe, out, *j), jobs)))
    summary = {k: (r["status"], r.get("exception"), r.get("seconds"), len(_foreign(r, allowed)))
               for k, r in recs.items()}
    (out / "SUMMARY.json").write_text(json.dumps({f"{c}-{lv}": v for (c, lv), v in summary.items()}, indent=1))
    for c in CASES:
        r = recs[(c, "L0")]
        assert r["status"] == "OK", (c, r.get("exception"), r.get("raised_at"), r.get("mitjax_frame"))
        assert _foreign(r, allowed) == [], (c, _foreign(r, allowed)[:5])
        assert str(Path(r["experiment_dir"])).startswith(str(out / "exps")), r["experiment_dir"]
    assert recs[("fwd_gyre", "L0")]["info"]["output_lines"] > 0
    g = recs[("grad_col", "L0")]["info"]
    assert g["finite"] and g["gnorm"] > 0, g
    assert recs[("shard_p2", "L0")]["info"]["bitwise_p2_p1"] is True
    a = recs[("api_gyre", "L0")]["info"]
    assert a["verdict"] == "pass" and a["output_lines"] > 0 and a["nfields"] > 0, a
    assert a["grid_xC"] and a["global_eta"], a                      # lane APIF: exp.grid(), Run.global_field
    a = recs[("api_col", "L0")]["info"]
    assert a["finite"] and a["gnorm"] > 0 and "adxx_theta.0000000000.data" in a["files"], a
    assert a["verdict"] == "pass" and a["nchecks"] == 4, a
    assert recs[("api_shard", "L0")]["info"]["finite"], recs[("api_shard", "L0")]["info"]
    # negative control: the planted read of $MJX_REFERENCE (it succeeds against the real oracle) is caught
    bad = _foreign(recs[("fwd_gyre", "PLANT")], allowed)
    assert any(f["path"].startswith(str(real_ref)) for f in bad), recs[("fwd_gyre", "PLANT")].get("exception")
