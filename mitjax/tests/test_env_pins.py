"""environment.yml agrees with constraints.txt and pyproject.toml (docs plan 20261006 Task 5).

environment.yml is written by hand for users (conda env create + pip install -e .); constraints.txt is the exact set
of our env `mitjax` (docs/ENV.md). Every pip pin of environment.yml equals constraints.txt, the Python version equals
the one constraints.txt was taken from, and every runtime dependency of pyproject.toml is pinned there.
Negative controls: a planted version, an unpinned or missing dependency, another Python are reported.
"""

import re
import tomllib

from mitjax import paths

_PIN = re.compile(r"^\s*-\s+([A-Za-z0-9_.\-]+)==([^\s#]+)\s*(?:#.*)?$")
_PY = re.compile(r"^\s*-\s+python=([0-9.]+)\s*$")


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def constraints():
    text = (paths.REPO / "constraints.txt").read_text()
    pins = {_norm(m.group(1)): m.group(2) for m in (re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", ln) for ln in
                                                    text.splitlines()) if m}
    py = re.search(r"Python (\d+\.\d+\.\d+)", text).group(1)
    return pins, py


def env_errors(env_text, pyproject=None):
    """Disagreements of an environment.yml text with constraints.txt and pyproject.toml ([] = none)."""
    pins, py = constraints()
    pyproject = pyproject or tomllib.loads((paths.REPO / "pyproject.toml").read_text())
    errs, got, pys = [], {}, []
    for ln in env_text.splitlines():
        if ln.strip().startswith("#") or not ln.strip():
            continue
        m = _PIN.match(ln)
        if m:
            got[_norm(m.group(1))] = m.group(2)
        elif _PY.match(ln):
            pys.append(_PY.match(ln).group(1))
        elif re.match(r"^\s*-\s+[A-Za-z]", ln) and ln.strip() not in ("- pip", "- conda-forge", "- pip:"):
            errs.append(f"unpinned entry {ln.strip()!r}")
    if pys != [py]:
        errs.append(f"python {pys} != constraints.txt's Python {py}")
    for name, v in sorted(got.items()):
        if pins.get(name) != v:
            errs.append(f"{name}=={v} but constraints.txt has {pins.get(name)}")
    for dep in pyproject["project"]["dependencies"]:
        name = _norm(re.split(r"[<>=!\[ ]", dep, maxsplit=1)[0])
        if name not in got:
            errs.append(f"pyproject dependency {name} not pinned in environment.yml")
    for dep in pyproject["project"].get("optional-dependencies", {}).get("notebooks", []):
        # docs plan 20261006 Task 8: the notebooks extra is part of the user's env (docs/ENV.md)
        name = _norm(re.split(r"[<>=!\[ ]", dep, maxsplit=1)[0])
        if name not in got:
            errs.append(f"pyproject notebooks extra {name} not pinned in environment.yml")
    if "jax" in got and got.get("jaxlib") != got["jax"]:
        errs.append("jaxlib must be pinned with jax")
    return errs


def test_environment_yml_agrees_with_constraints():
    text = (paths.REPO / "environment.yml").read_text()
    assert env_errors(text) == []
    # negative controls
    assert env_errors(text.replace("numpy==2.4.6", "numpy==2.4.5")) == [
        "numpy==2.4.5 but constraints.txt has 2.4.6"]
    errs = env_errors(text.replace("      - scipy==1.17.1\n", ""))
    assert errs == ["pyproject dependency scipy not pinned in environment.yml"], errs
    errs = env_errors(text.replace("netCDF4==1.7.4", "netCDF4"))
    assert errs[0] == "unpinned entry '- netCDF4'", errs
    assert env_errors(text.replace("python=3.12.13", "python=3.13.1")) == [
        "python ['3.13.1'] != constraints.txt's Python 3.12.13"]
    errs = env_errors(text.replace("      - matplotlib==3.10.9\n", ""))
    assert errs == ["pyproject notebooks extra matplotlib not pinned in environment.yml"], errs
    assert env_errors(text.replace("nbclient==0.11.0", "nbclient==0.10.0")) == [
        "nbclient==0.10.0 but constraints.txt has 0.11.0"]
