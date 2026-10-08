#!/bin/bash
#SBATCH --job-name=mjx_kpp_dev
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The KPP lane's tests alone on a compute node (development runs between tier jobs):
#   sbatch reference/replay_kpp/jobs/pytest_kpp.sbatch REPO [PYTEST_ARGS...]
# e.g. REPO=$MJX_DEV/wt_m3_kpp, PYTEST_ARGS mitjax/tests/test_kpp.py -k gradient.
# Runs on a frozen snapshot (git archive HEAD) in $MJX_RUNS/kpp/<job>/src; never deletes anything.
set -u
REPO=${1:?repository}; shift
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT=$MJX_RUNS/kpp/${SLURM_JOB_ID:?run this through sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src"
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD)
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
git -C "$REPO" status --porcelain > "$OUT/git_status.txt"
echo "job $SLURM_JOB_ID $(hostname) commit $COMMIT dirty $(wc -l < "$OUT/git_status.txt") args $*"
cd "$OUT/src"
JAX_PLATFORMS=cpu PYTHONPATH="$OUT/src" PYTHONPYCACHEPREFIX="$OUT/pyc" "$PY" -m pytest -p no:cacheprovider -s -rA \
  --junitxml="$OUT/junit.xml" "${@:-mitjax/tests/test_kpp.py}"
echo "pytest exit $?"
