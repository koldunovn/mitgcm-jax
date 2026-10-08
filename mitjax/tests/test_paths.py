"""mitjax/paths.py: the one place for machine paths. Its defaults are neutral (docs plan 20261006 Task 5: MJX_UPSTREAM
required, MJX_RUNS ./runs, MJX_CACHE the user cache, the python the running one); with MJX_WORK (levante.env, this
project's values) every location derives from the work root as before; each variable overrides its own path and the
ones derived from it, the ECCO port's `MITJAX_*` variables are ignored, and no file of the repository but levante.env
hard-codes the work root (paths.py and the batch scripts included: their defaults are neutral since 2026-10-08),
reads `MITJAX_*` or points into the ECCO port (`~/MIT`). The expected root is the one levante.env sets."""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _levante_root():
    """The work root levante.env sets (its MJX_WORK default): this project's root on Levante."""
    m = re.search(r"^export MJX_WORK=\$\{MJX_WORK:-(/[^}]+)\}$", (REPO / "levante.env").read_text(), re.M)
    assert m, "levante.env: no `export MJX_WORK=${MJX_WORK:-<path>}` line"
    return m.group(1)


LEVANTE = _levante_root()


def resolved(levante=False, cwd=None, **env):
    """paths.py --sh in a clean process with only the given MJX_*/MITJAX_* variables set (and levante.env sourced
    first if `levante`); unset locations are absent."""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("MJX_", "MITJAX_"))}
    e.update(env)
    cmd = f'{"" if not levante else ". " + str(REPO / "levante.env") + "; "}"{sys.executable}" ' \
          f'"{REPO / "mitjax" / "paths.py"}" --sh'
    out = subprocess.run(["bash", "-c", cmd], env=e, check=True, capture_output=True, text=True, cwd=cwd).stdout
    return dict(line[len("export "):].split("=", 1) for line in out.splitlines() if line.startswith("export "))


def _repo_files():
    skip = {".git", "work", "__pycache__"}
    for p in REPO.rglob("*"):
        rel = p.relative_to(REPO)
        if p.is_file() and not skip.intersection(rel.parts) and p.suffix in {".py", ".sh", ".sbatch", ".toml", ""}:
            yield rel.as_posix(), p.read_text(errors="replace")


def scan(files, this_test="mitjax/tests/test_paths.py"):
    """Return (work_root, mitjax_prefix, ecco_tree): files that hard-code the work root (levante.env is the one place
    for it), read MITJAX_*, or point into the ECCO port's tree."""
    root_ok = {"levante.env", this_test}
    work_root, prefix, ecco = [], [], []
    for rel, text in files:
        if rel == this_test:
            continue
        if LEVANTE in text and rel not in root_ok:
            work_root.append(rel)
        if "MITJAX_" in text and rel != "mitjax/paths.py":  # its docstring names the prefix it ignores
            prefix.append(rel)
        if re.search(r"~/MIT\b(?!jax)|/MIT/|/home/[a]/a27[0]088/MIT\b(?!jax)", text):    # [x]: the public-tree scan
            ecco.append(rel)
    return work_root, prefix, ecco


def test_paths():
    """One test (tier 1 counts tests against a budget of 100): defaults, overrides, foreign prefix, hard-coded roots."""
    # neutral defaults: nothing of this machine; MJX_UPSTREAM required (MissingPath names it)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    p = resolved(cwd="/")
    assert p == {"MJX_RUNS": "/runs", "MJX_CACHE": str((Path(xdg) if xdg else Path.home() / ".cache") / "mitjax"),
                 "MJX_PYTHON": sys.executable}, p
    e = {k: v for k, v in os.environ.items() if not k.startswith(("MJX_", "MITJAX_"))}
    r = subprocess.run([sys.executable, "-c", "import importlib.util as u; s = u.spec_from_file_location('p', "
                        f"{str(REPO / 'mitjax' / 'paths.py')!r}); m = u.module_from_spec(s); s.loader.exec_module(m); "
                        "m.UPSTREAM"], env=e, capture_output=True, text=True)
    assert r.returncode != 0 and "MissingPath: MJX_UPSTREAM is not set" in r.stderr, r.stderr[-300:]
    # this project: levante.env gives every location of the work root
    p = resolved(levante=True)
    assert p["MJX_WORK"] == LEVANTE
    assert p["MJX_UPSTREAM"] == f"{LEVANTE}/upstream/MITgcm"
    assert p["MJX_REFERENCE_RUNS"] == f"{LEVANTE}/reference/runs" and p["MJX_CACHE"] == f"{LEVANTE}/cache"
    assert p["MJX_RUNS"] == f"{LEVANTE}/runs" and p["MJX_DEV"] == f"{LEVANTE}/dev"
    q = resolved(MJX_WORK="/x")
    assert q["MJX_UPSTREAM"] == "/x/upstream/MITgcm" and q["MJX_REFERENCE_RUNS"] == "/x/reference/runs"
    assert q["MJX_RUNS"] == "/x/runs" and q["MJX_LESSONS"] == "/x/lessons"

    # each variable overrides its own subtree only; empty means unset
    p = resolved(MJX_WORK="/x", MJX_REFERENCE="/r", MJX_RUNS="")
    assert p["MJX_REFERENCE_RUNS"] == "/r/runs" and p["MJX_RUNS"] == "/x/runs"
    assert resolved(MJX_UPSTREAM="~/u")["MJX_UPSTREAM"] == str(Path("~/u").expanduser())

    # the ECCO port's prefix steers nothing
    assert resolved(MITJAX_WORK="/ecco", MITJAX_REFERENCE="/ecco/r", MITJAX_UPSTREAM="/e") == resolved()

    # no other file hard-codes the work root, reads MITJAX_*, or points into ~/MIT
    work_root, prefix, ecco = scan(_repo_files())
    assert (work_root, prefix, ecco) == ([], [], []), (work_root, prefix, ecco)
    # negative controls: each planted line is caught
    planted = [("mitjax/a.py", f'W = "{LEVANTE}/runs"'), ("tools/b.py", 'os.environ["MITJAX_WORK"]'),
               ("scripts/e.sbatch", f"#SBATCH --output={LEVANTE}/logs/%x-%j.out"),
               ("mitjax/paths.py", f'WORK = "{LEVANTE}"'),
               ("scripts/c.sh", "cp ~/MIT/scripts/x ."), ("mitjax/d.py", '"' + LEVANTE.rsplit("/", 1)[0] + '/MIT/reference"'),
               ("mitjax/ok.py", "~/MITjax/docs")]
    assert scan(planted) == (["mitjax/a.py", "scripts/e.sbatch", "mitjax/paths.py"], ["tools/b.py"],
                             ["scripts/c.sh", "mitjax/d.py"])
