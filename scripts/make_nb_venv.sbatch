#!/bin/bash
#SBATCH --job-name=mjx_nb_venv
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The notebook overlay venv (docs plan 20261006 Task 8; docs/ENV.md): env `mitjax` stays untouched; the venv sees its
# packages (--system-site-packages) and adds only pyproject's `notebooks` extra, pinned by constraints.txt.
#     sbatch scripts/make_nb_venv.sbatch <repository> [venv]      (venv default $MJX_WORK/envs/mitjax-nb)
# Refuses an existing venv; never deletes. Checks that jax / jaxlib / numpy / scipy resolve to the env's own copies.
set -eu
REPO=${1:?repository}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"
PY=${MJX_PYTHON:?MJX_PYTHON}
VENV=${2:-$MJX_WORK/envs/mitjax-nb}
[ -e "$VENV" ] && { echo "FAIL: $VENV exists; not touching it"; exit 1; }
echo "job $SLURM_JOB_ID $(date -Is) repo $REPO ($(git -C "$REPO" rev-parse HEAD)) venv $VENV base $PY"
"$PY" -m venv --system-site-packages "$VENV"
mapfile -t DEPS < <("$PY" - "$REPO/pyproject.toml" <<'PYEOF'
import sys, tomllib
for d in tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["notebooks"]:
    print(d)
PYEOF
)
echo "notebooks extra: ${DEPS[*]}"
"$VENV/bin/python" -m pip install --no-cache-dir -c "$REPO/constraints.txt" "${DEPS[@]}"
"$VENV/bin/python" - "$PY" <<'PYEOF'
import importlib, subprocess, sys
base = sys.argv[1]
bad = []
for m in ("jax", "jaxlib", "numpy", "scipy"):
    mod = importlib.import_module(m)
    ref = subprocess.run([base, "-c", f"import {m}; print({m}.__version__, {m}.__file__)"], capture_output=True,
                         text=True, check=True).stdout.split()
    print(m, mod.__version__, mod.__file__, "| env:", *ref)
    if [mod.__version__, mod.__file__] != ref:
        bad.append(m)
for m in ("matplotlib", "nbclient", "nbformat", "ipykernel"):
    mod = importlib.import_module(m)
    print(m, mod.__version__)
if bad:
    sys.exit(f"FAIL: {bad} differ from the env's own copies")
print("VENV OK")
PYEOF
"$VENV/bin/python" -m ipykernel install --prefix "$VENV" --name python3 --display-name "mitjax-nb"
"$VENV/bin/jupyter" kernelspec list
