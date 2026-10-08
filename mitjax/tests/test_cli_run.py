"""Lane CLI (docs plan 20261006 decision 12 b): `python -m mitjax run` goes through the API
(mitjax/__main__.py: the API's Experiment.run(out, devices=N), the API's XLA flags), with `--devices N`.

* XLA flags: with no user XLA_FLAGS the API's string (set_api_xla_flags) is the gates' string (set_gate_xla_flags) on
  x86-64, under MJX_XLA_FLAG_SET unset and "gpu"; a CLI started by a test inherits the gate XLA_FLAGS and the API's
  merge leaves them as they are. Control: the arm64 API set differs from the gate set (no --xla_cpu_max_isa).
* CLI = API on tutorial_barotropic_gyre/input: `python -m mitjax run` in a subprocess with XLA_FLAGS and
  MJX_XLA_FLAG_SET removed (a user's shell) writes output.txt and pickups byte for byte the in-process exp.run's
  (gate flags); its stdout ends with the path of output.txt; exit code 0.
* `--devices 2` = `--devices 1` (mitjax.__main__.main in this process) on global_ocean.90x40x15/input_ad (4 tiles, the
  exch1 case of test_api_followups.py's sharded forward): every file the run writes byte for byte.
  Control: one ulp planted into tile 4's theta in drivers/run.sharded_step (the P > 1 step only) -> the `--devices 2`
  output differs from `--devices 1`; this also shows that `--devices 2` reaches the sharded step.
* `--devices 5` (JAX has 4 CPU devices under the API flags) in a subprocess with no user XLA_FLAGS: exit code 2,
  stderr carries DeviceCountError's text "devices=5: JAX has 4 device(s)" (so the API's device-count flag reached XLA
  before its backend started), and no output directory is made.
* RUNDIR line: the CLI's Experiment (mitjax.__main__.cli_experiment) prints make_rundir's `RUNDIR <path>` line for a
  prepare_run variant (advect_cs/input), as the CLI did before. Control: the API's own Experiment prints none.
About 10 min on a CPU node (two gyre runs, three 10-step 90x40x15 runs).
"""

import os
import platform
import subprocess
import sys

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax import paths  # noqa: E402
from mitjax import xla_flags as XF  # noqa: E402

GYRE = ("tutorial_barotropic_gyre", "input")
GO90 = ("global_ocean.90x40x15", "input_ad")


def _out(tag):
    from mitjax.tests import advect_gate as ag
    return ag.out_dir(f"cli-{tag}")


def _exp(name):
    return paths.UPSTREAM / "verification" / name


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _user_env():
    """os.environ without XLA_FLAGS and MJX_XLA_FLAG_SET (a user's shell), the repository on PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "MJX_XLA_FLAG_SET")}
    env["PYTHONPATH"] = str(paths.REPO)
    return env


def _written(rundir):
    """{name: bytes} of every regular file the run wrote into rundir (the linked inputs are symlinks)."""
    return {p.name: p.read_bytes() for p in sorted(rundir.iterdir()) if p.is_file() and not p.is_symlink()}


def _same(a, b):
    diff = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
    return not diff, f"{len(a)} / {len(b)} files, differing {diff}"


# ------------------------------------------------------------------------------------------------------ XLA flags

def test_api_flags_equal_gate_flags_on_x86(monkeypatch):
    for which in (None, "gpu"):
        if which is None:
            monkeypatch.delenv("MJX_XLA_FLAG_SET", raising=False)
        else:
            monkeypatch.setenv("MJX_XLA_FLAG_SET", which)
        gate = XF.gate_xla_flags("", XF.flag_set())
        for machine in ("x86_64", "AMD64", None):
            if machine is None and platform.machine().lower() not in XF.X86_MACHINES:
                continue
            assert XF.merge_user_flags("", XF.api_flag_set(machine)) == gate, (which, machine)
        # a CLI subprocess of a test inherits the test's gate XLA_FLAGS: both merges leave them unchanged
        assert XF.merge_user_flags(gate, XF.api_flag_set("x86_64")) == gate == XF.gate_xla_flags(gate, XF.flag_set())
        # control: off x86-64 the API set drops --xla_cpu_max_isa, so the comparison above can fail
        assert XF.merge_user_flags("", XF.api_flag_set("aarch64")) != gate


# --------------------------------------------------------------------------------------------- CLI = API (gyre)

@pytest.fixture(scope="module")
def gyre():
    import mitjax
    exp = mitjax.load(_exp(GYRE[0]), variant=GYRE[1])
    r = exp.run(_out("gyre-api"))
    _release()
    out = _out("gyre-cli")
    p = subprocess.run([sys.executable, "-m", "mitjax", "run", str(_exp(GYRE[0])), "--variant", GYRE[1], "--out",
                        str(out)], env=_user_env(), capture_output=True, text=True)
    return dict(api=r, cli_out=out, proc=p)


def test_cli_run_bitwise_vs_api(gyre):
    p, out, r = gyre["proc"], gyre["cli_out"], gyre["api"]
    assert p.returncode == 0, (p.stdout[-2000:], p.stderr[-3000:])
    lines = p.stdout.strip().splitlines()
    assert lines and lines[-1] == str(out / "rundir" / "output.txt"), p.stdout[-2000:]
    assert not any(ln.startswith("RUNDIR ") for ln in lines), p.stdout[-2000:]     # no prepare_run: none before
    ours, api = _written(out / "rundir"), _written(r.rundir)
    assert "output.txt" in api and any(n.startswith("pickup") for n in api), sorted(api)
    ok, info = _same(ours, api)
    print("gyre CLI vs API:", info)
    assert ok, info


# ---------------------------------------------------------------------------------------- --devices 2 = 1 (GO90)

def _cli_run(tag, devices):
    from mitjax.__main__ import main
    out = _out(tag)
    rc = main(["run", str(_exp(GO90[0])), "--variant", GO90[1], "--out", str(out), "--devices", str(devices)])
    _release()
    assert rc == 0
    return _written(out / "rundir")


@pytest.fixture(scope="module")
def go90():
    return {P: _cli_run(f"go90-p{P}", P) for P in (1, 2)}


def test_cli_devices2_bitwise_vs_1(go90):
    assert "output.txt" in go90[1], sorted(go90[1])
    ok, info = _same(go90[2], go90[1])
    print("GO90 --devices 2 vs 1:", info)
    assert ok, info


def test_cli_devices2_planted_ulp(go90, monkeypatch):
    """Control: one ulp in tile 4's theta (its first wet interior point at k = 1) in the sharded step only -> the
    `--devices 2` output differs from `--devices 1`."""
    import jax.numpy as jnp

    from mitjax.drivers import run as R
    orig = R.sharded_step

    def planted(m, sh, carry):
        st = carry[0]
        th = np.array(st.theta.data)
        sz = m.cfg.size
        wet = np.argwhere(th[3, 0, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] != 0)
        assert len(wet), "tile 4 has no wet point at k = 1"
        j, i = int(wet[0][0]) + sz.OLy, int(wet[0][1]) + sz.OLx
        th[3, 0, j, i] = np.nextafter(th[3, 0, j, i], np.inf)
        st = st.replace(theta=type(st.theta)(jnp.asarray(th), st.theta.name, tiled=st.theta.tiled,
                                             _dims=st.theta.dims))
        return orig(m, sh, (st,) + tuple(carry[1:]))
    monkeypatch.setattr(R, "sharded_step", planted)
    bad = _cli_run("go90-ctl-p2", 2)
    ok, info = _same(bad, go90[1])
    print("planted ulp, --devices 2 vs 1:", info)
    assert not ok, info


# ------------------------------------------------------------------------------------------ --devices above count

def test_cli_devices_above_count():
    out = _out("gyre-devices5")
    p = subprocess.run([sys.executable, "-m", "mitjax", "run", str(_exp(GYRE[0])), "--variant", GYRE[1], "--out",
                        str(out), "--devices", "5"], env=_user_env(), capture_output=True, text=True)
    print(p.stderr[-1000:])
    assert p.returncode == 2, (p.stdout[-2000:], p.stderr[-3000:])
    assert "DeviceCountError: devices=5: JAX has 4 device(s)" in p.stderr, p.stderr[-3000:]
    assert not out.exists() and not out.is_symlink()


# ------------------------------------------------------------------------------------------------------ RUNDIR line

def test_cli_keeps_rundir_line(capsys):
    import mitjax
    from mitjax.__main__ import cli_experiment
    exp = cli_experiment(_exp("advect_cs"), "input")                 # input/prepare_run links the cube grid
    rd = exp._rundir(_out("advect-rundir-cli"))
    out = capsys.readouterr().out
    assert rd.exists() and any(ln.startswith("RUNDIR ") for ln in out.splitlines()), out
    # control: the API's own Experiment prints no RUNDIR line
    rd2 = mitjax.load(_exp("advect_cs"), variant="input")._rundir(_out("advect-rundir-api"))
    out2 = capsys.readouterr().out
    assert rd2.exists() and "RUNDIR" not in out2, out2
