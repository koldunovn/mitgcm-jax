#!/bin/bash
#SBATCH --job-name=mjx_fresh_env
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=fresh_install-%j.out
#
# Fresh-install check, step 1 of 2 (docs plan 20261006 Task 11): a user's install from the PUBLIC tree, nothing of
# this project's environment inherited. In a NEW directory WORK (refused if it exists; nothing is ever deleted):
#   1. clone the public tree (a git repository made by tools/make_public_tree.py) -> WORK/mitgcm-jax
#   2. conda env from its environment.yml, by prefix -> WORK/env (package cache WORK/pkgs, no pip cache)
#   3. MITgcm from GitHub at 63cdc0b, next to the clone -> WORK/MITgcm
#   4. `pip install -e .` in WORK/mitgcm-jax (the documented install), then the import checks
# Step 2 (scripts/fresh_install_nb01.sbatch, partition compute, submitted with --dependency=afterok on this job)
# executes notebook 01 and the user's smoke + tier-1 selection in that environment.
#
#   sbatch [-o LOG] scripts/fresh_install.sbatch PUBLIC_TREE WORK [CONDA]
#   bash scripts/fresh_install.sbatch --dry-run PUBLIC_TREE WORK [CONDA]   # checks and prints the plan, runs nothing
#
# CONDA: the conda (or mamba) executable; default `conda` on PATH. Every inherited MJX_* variable is cleared, and
# PYTHONPATH, PYTHONHOME, PYTHONSTARTUP, XLA_FLAGS, JAX_* and CONDA_PREFIX with them: the user has none of ours.
# Result: WORK/env_step.txt (one line per step) and WORK/ENV_OK when every step passed; exit code 0 only then.
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
PUBLIC=${1:?public tree (a git repository made by tools/make_public_tree.py)}
WORK=${2:?new work directory}
CONDA=${3:-conda}
MITGCM_URL=https://github.com/MITgcm/MITgcm
MITGCM_COMMIT=63cdc0b

# ---- nothing inherited
for v in $(compgen -e); do
  case "$v" in
    MJX_*|JAX_*|PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|PYTHONPYCACHEPREFIX|XLA_FLAGS|CONDA_PREFIX|CONDA_DEFAULT_ENV|\
    VIRTUAL_ENV) unset "$v" ;;
  esac
done
left=$(compgen -e | grep -E '^(MJX_|JAX_)' || true)
[ -z "$left" ] || { echo "FAIL: still set: $left"; exit 1; }

# ---- checks before anything is written
[ -d "$PUBLIC/.git" ] ||
  { echo "FAIL: $PUBLIC is not a git repository (build it with tools/make_public_tree.py)"; exit 1; }
[ "$(git -C "$PUBLIC" rev-list --count HEAD)" = 1 ] || { echo "FAIL: $PUBLIC must hold exactly one commit"; exit 1; }
[ -z "$(git -C "$PUBLIC" status --porcelain)" ] || { echo "FAIL: $PUBLIC has uncommitted changes"; exit 1; }
[ -z "$(git -C "$PUBLIC" remote)" ] || { echo "FAIL: $PUBLIC has a remote"; exit 1; }
[ -f "$PUBLIC/environment.yml" ] && [ -f "$PUBLIC/pyproject.toml" ] ||
  { echo "FAIL: no environment.yml / pyproject.toml in $PUBLIC"; exit 1; }
[ -e "$WORK" ] && { echo "FAIL: $WORK exists (a fresh install needs a new directory)"; exit 1; }
command -v "$CONDA" > /dev/null || { echo "FAIL: no conda executable '$CONDA'"; exit 1; }
PUBLIC_COMMIT=$(git -C "$PUBLIC" rev-parse HEAD)

cat <<EOF
fresh install: public tree $PUBLIC (commit $PUBLIC_COMMIT) -> $WORK
  1. git clone $PUBLIC $WORK/mitgcm-jax
  2. $CONDA env create -p $WORK/env -f $WORK/mitgcm-jax/environment.yml   (CONDA_PKGS_DIRS=$WORK/pkgs)
  3. git clone $MITGCM_URL $WORK/MITgcm && checkout $MITGCM_COMMIT
  4. $WORK/env/bin/python -m pip install --no-cache-dir -e $WORK/mitgcm-jax; import checks
EOF
if [ "$DRY" = 1 ]; then echo "dry run: nothing done"; exit 0; fi

mkdir -p "$WORK"
LOG=$WORK/env_step.txt
step() { echo "$(date -Is) $*" | tee -a "$LOG"; }
step "job ${SLURM_JOB_ID:-none} on $(hostname); public commit $PUBLIC_COMMIT"
export CONDA_PKGS_DIRS=$WORK/pkgs PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

git clone -q --no-hardlinks "$PUBLIC" "$WORK/mitgcm-jax"
step "1 clone OK ($(git -C "$WORK/mitgcm-jax" ls-files | wc -l) files)"

"$CONDA" env create -q -p "$WORK/env" -f "$WORK/mitgcm-jax/environment.yml" > "$WORK/conda.log" 2>&1 ||
  { step "2 conda env create FAILED (conda.log)"; exit 1; }
PY=$WORK/env/bin/python
step "2 env OK: $("$PY" --version 2>&1)"

git clone -q "$MITGCM_URL" "$WORK/MITgcm" > "$WORK/mitgcm_clone.log" 2>&1 ||
  { step "3 MITgcm clone FAILED"; exit 1; }
git -C "$WORK/MITgcm" checkout -q "$MITGCM_COMMIT"
step "3 MITgcm OK at $(git -C "$WORK/MITgcm" rev-parse --short=7 HEAD)"

(cd "$WORK/mitgcm-jax" && "$PY" -m pip install --no-cache-dir -e . > "$WORK/pip.log" 2>&1) ||
  { step "4 pip install -e . FAILED (pip.log)"; exit 1; }
"$PY" -m pip check >> "$WORK/pip.log" 2>&1 || { step "4 pip check FAILED (pip.log)"; exit 1; }
"$PY" -m pip freeze > "$WORK/pip_freeze.txt"
cd "$WORK"
MJX_UPSTREAM=$WORK/MITgcm "$PY" - <<'EOF' | tee -a "$LOG"
import os, sys
import jax, mitjax, numpy
from mitjax import paths
work = os.getcwd()
assert mitjax.__file__.startswith(os.path.join(work, "mitgcm-jax")), mitjax.__file__
assert jax.__version__ == "0.10.1", jax.__version__
assert getattr(paths, "REFERENCE", None) is None, "MJX_REFERENCE leaked into the user's environment"
print(f"4 import OK: mitjax {mitjax.__file__}, jax {jax.__version__}, numpy {numpy.__version__}, "
      f"python {sys.version.split()[0]}, UPSTREAM {paths.UPSTREAM}")
EOF
touch "$WORK/ENV_OK"
step "ENV OK -> $WORK (next: scripts/fresh_install_nb01.sbatch $WORK)"
