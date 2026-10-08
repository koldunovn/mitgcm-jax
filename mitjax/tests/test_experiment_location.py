"""Experiments anywhere on disk; an experiment's own Fortran must be ported (docs plan 20261006, Task 2, decision 11).

1. A verification experiment COPIED outside the MITgcm tree (its code*, input* and results directories) runs bitwise
   equal to the in-tree run through `python -m mitjax run`'s driver: output.txt and every written .data / .meta file
   byte for byte -- tutorial_barotropic_gyre (also as a renamed copy) and tutorial_global_oce_latlon (own
   ptracers_*.F ports, PTRACERS_SIZE.h read from the code directory).
2. The configuration of the copy equals the in-tree one except for `exp_dir` and SIZE.h's provenance.
3. Decision 11: a planted unknown code/foo.F is refused naming FOO; a ported experiment routine changed in one
   statement is refused naming it; an unchanged copy of an MITgcm routine is accepted.
Negative controls: the old lookup (load_experiment requiring <tree>/verification/<exp>) fails on the copy; the old
kernel-site lookup (MJX_UPSTREAM/verification/<experiment>/<code dir>) reads the tree, not the copy (a copy with
its own PTRACERS_SIZE.h).
Costs: about 15 min on a CPU node (six whole runs; latlon ~9 min, job 27924823).
"""

import dataclasses
import filecmp
import os
import shutil
import time
from pathlib import Path

import pytest

from mitjax import paths
from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()


def _new_dir(tag):
    """A new directory for this test file (never reused, never removed)."""
    d = paths.RUNS / "tests_port" / f"{tag}-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    return d


def copy_experiment(exp, dest_root, name=None):
    """<dest_root>/<name or exp> with the experiment's code*, input* and results directories, symlinks dereferenced
    (scripts/noref_probe.copy_experiment)."""
    src = paths.UPSTREAM / "verification" / exp
    dst = Path(dest_root) / (name or exp)
    dst.mkdir(parents=True)
    for d in sorted(src.iterdir()):
        if d.is_dir() and d.name.startswith(("code", "input", "results")):
            shutil.copytree(d, dst / d.name, symlinks=False)
    return dst


def _old_load_experiment(exp_dir, variant):
    """drivers/run.load_experiment before Task 2 (54ada69, verbatim): the experiment had to be <tree>/verification/."""
    from mitjax.config.params import load
    exp_dir = Path(exp_dir).resolve()
    if exp_dir.parent.name != "verification":
        raise ValueError(f"{exp_dir}: expected <MITgcm tree>/verification/<experiment>")
    return load(exp_dir.name, variant, upstream=exp_dir.parents[1])


def _run(exp_dir, variant, tag):
    from mitjax.drivers.run import run
    return run(exp_dir, variant, _new_dir(tag) / "out").parent


def _same_outputs(a, b):
    """{file: identical?} for output.txt and every regular .data / .meta file either run directory wrote."""
    names = {p.name for d in (a, b) for p in Path(d).iterdir()
             if p.is_file() and not p.is_symlink() and (p.suffix in (".data", ".meta") or p.name == "output.txt")}
    return {n: (a / n).is_file() and (b / n).is_file() and filecmp.cmp(a / n, b / n, shallow=False)
            for n in sorted(names)}


@pytest.mark.parametrize("exp,variant,name", [("tutorial_barotropic_gyre", "input", None),
                                              ("tutorial_barotropic_gyre", "input", "my_gyre"),
                                              ("tutorial_global_oce_latlon", "input", None)])
def test_copy_runs_bitwise(exp, variant, name):
    root = _new_dir(f"copy-{name or exp}")
    cp = copy_experiment(exp, root / "exps", name)
    a = _run(paths.UPSTREAM / "verification" / exp, variant, f"tree-{exp}")
    b = _run(cp, variant, f"copy-{name or exp}")
    same = _same_outputs(a, b)
    print(exp, name, len(same), "files compared")
    assert "output.txt" in same and len(same) > 1 and all(same.values()), same


def test_config_of_copy_equals_tree():
    from mitjax.config.params import load
    exp = "tutorial_global_oce_latlon"
    cp = copy_experiment(exp, _new_dir("cfg") / "exps")
    t, c = load(exp, "input"), load(exp, "input", exp_dir=cp)
    assert c.cfg.exp_dir == str(cp.resolve()) and t.cfg.exp_dir == str((paths.UPSTREAM / "verification" / exp).resolve())
    assert c.cfg.size.source == str(cp.resolve() / "code" / "SIZE.h")         # provenance: the copy's SIZE.h
    assert dataclasses.replace(c.cfg, exp_dir=t.cfg.exp_dir,
                               size=dataclasses.replace(c.cfg.size, source=t.cfg.size.source)) == t.cfg
    assert c.params.values.keys() == t.params.values.keys()
    assert all(c.params.values[k].tobytes() == t.params.values[k].tobytes() for k in t.params.values)
    assert c.run.vars == t.run.vars


def test_old_lookup_fails_on_copy():
    """Negative control 1: the pre-Task-2 loader refuses the copied experiment."""
    from mitjax.drivers.run import load_experiment
    cp = copy_experiment("tutorial_barotropic_gyre", _new_dir("old") / "exps")
    assert load_experiment(cp, "input").cfg.exp_dir == str(cp.resolve())
    with pytest.raises(ValueError, match="expected <MITgcm tree>/verification"):
        _old_load_experiment(cp, "input")


def test_sites_read_the_copy(monkeypatch):
    """The kernel sites read the copy's code directory: a PTRACERS_SIZE.h added to the copy (pkg/ptracers' file with
    PTRACERS_num + 1) is the one _num reads; negative control 2: the old lookup planted at the one helper reads the
    tree's code directory (which has none) and falls back to pkg/ptracers' value."""
    import re
    from mitjax.config import params
    from mitjax.pkg.ptracers.ptracers_readparms import _num
    exp = "tutorial_global_oce_latlon"
    cp = copy_experiment(exp, _new_dir("sites") / "exps")
    txt = (paths.UPSTREAM / "pkg" / "ptracers" / "PTRACERS_SIZE.h").read_text()
    n0 = int(re.search(r"PTRACERS_num\s*=\s*(\d+)", txt).group(1))
    (cp / "code" / "PTRACERS_SIZE.h").write_text(re.sub(r"(PTRACERS_num\s*=\s*)\d+", rf"\g<1>{n0 + 1}", txt))
    cfg = params.load(exp, "input", exp_dir=cp).cfg
    assert _num(cfg) == n0 + 1
    monkeypatch.setattr(params, "code_path",
                        lambda cfg: paths.UPSTREAM / "verification" / cfg.experiment / cfg.code_dir)
    assert _num(cfg) == n0                                    # the old lookup: not the copy's file


FOO = """C     a routine no port knows
      SUBROUTINE FOO( myThid )
      IMPLICIT NONE
      INTEGER myThid
      RETURN
      END
"""


def test_unknown_routine_refused():
    from mitjax.config.own_code import UnportedRoutine
    from mitjax.drivers.run import load_experiment
    cp = copy_experiment("tutorial_barotropic_gyre", _new_dir("foo") / "exps")
    (cp / "code" / "foo.F").write_text(FOO)
    with pytest.raises(UnportedRoutine, match=r"foo\.F \(routine FOO\)"):
        load_experiment(cp, "input")


def test_changed_ported_routine_refused():
    from mitjax.config.own_code import UnportedRoutine
    from mitjax.drivers.run import load_experiment
    cp = copy_experiment("advect_xy", _new_dir("xy") / "exps")
    f = cp / "code" / "ini_vel.F"
    txt = f.read_text()
    assert load_experiment(cp, "input").cfg.code_dir == "code"           # the copy as it is: accepted
    f.write_text(txt.replace("      IMPLICIT NONE", "      IMPLICIT NONE\nC     changed by the user", 1))
    with pytest.raises(UnportedRoutine, match=r"ini_vel\.F \(routine INI_VEL\)"):
        load_experiment(cp, "input")


def test_unchanged_upstream_copy_accepted():
    from mitjax.config.own_code import classify
    from mitjax.drivers.run import load_experiment
    cp = copy_experiment("tutorial_barotropic_gyre", _new_dir("same") / "exps")
    shutil.copyfile(paths.UPSTREAM / "model" / "src" / "ini_theta.F", cp / "code" / "ini_theta.F")
    assert classify(cp, "code", paths.UPSTREAM)["ini_theta.F"] == ("unchanged", "model/src/ini_theta.F")
    assert load_experiment(cp, "input").cfg.experiment == "tutorial_barotropic_gyre"
