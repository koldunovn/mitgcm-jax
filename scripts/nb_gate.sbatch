#!/bin/bash
#SBATCH --job-name=mjx_nb_gate
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The laptop notebooks' gate (docs plan 20261006 Task 8): mitjax/tests/test_notebooks_laptop.py (or the node ids
# given) on a frozen snapshot of the repository's HEAD, pinned to 8 cores (a laptop's count), with the notebook
# python (MJX_NB_PYTHON, default the overlay venv $MJX_WORK/envs/mitjax-nb of docs/ENV.md).
#   sbatch scripts/nb_gate.sbatch <repository> [pytest node ids...]
# Output: $MJX_RUNS/nb/gate-<commit7>-<job>/ (provenance.txt, pytest.log, report.xml, verdict.txt; notebooks/: the
# executed notebooks in executed/, measures.jsonl with wall time and peak RSS per notebook). The verdict is
# scripts/check_pytest_report.py on the report AND pytest's exit code, never the job state. Nothing is deleted.
# To commit the executed notebooks of a passing run (their cells equal HEAD's sources):
#   cp $OUT/notebooks/executed/0[1-4]_*.ipynb <repository>/notebooks/   (not the *-planted one)
set -u
REPO=${1:?repository}; shift
[ $# -eq 0 ] && set -- mitjax/tests/test_notebooks_laptop.py
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
eval "$("$MJX_PYTHON" "$REPO/mitjax/paths.py" --sh)"
export MJX_NB_PYTHON=${MJX_NB_PYTHON:-$MJX_WORK/envs/mitjax-nb/bin/python}
PY=$MJX_NB_PYTHON
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD) || exit 1
OUT=$MJX_RUNS/nb/gate-${COMMIT:0:7}-${SLURM_JOB_ID:?sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src" || exit 1
git -C "$REPO" status --porcelain > "$OUT/git_status.txt"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) commit $COMMIT python $PY tests $*" | tee "$OUT/provenance.txt"
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONPYCACHEPREFIX=$OUT/pycache PYTHONPATH=$OUT/src MJX_NB_OUT=$OUT/notebooks
T0=$(date +%s)
taskset -c 0-7 "$PY" -m pytest -q -rfEs -s -p no:cacheprovider --basetemp="$OUT/pytest_tmp" \
    --junitxml="$OUT/report.xml" --durations=20 "$@" > "$OUT/pytest.log" 2>&1
RC=$?
echo "pytest exit code $RC, wall $(( $(date +%s) - T0 )) s" | tee -a "$OUT/provenance.txt"
cat "$OUT/notebooks/measures.jsonl" 2>/dev/null | tee -a "$OUT/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT/report.xml" > "$OUT/verdict.txt"
CRC=$?
cat "$OUT/verdict.txt"
if [ "$RC" -ne 0 ] || [ "$CRC" -ne 0 ]; then echo "GATE FAIL (pytest rc $RC, check rc $CRC) -> $OUT"; exit 1; fi
echo "GATE PASS -> $OUT"
