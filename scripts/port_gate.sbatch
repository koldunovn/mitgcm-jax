#!/bin/bash
#SBATCH --job-name=mjx_port_gate
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Lane PORT gates (docs plan 20261006 S1): the named test files on a frozen snapshot of the repository's HEAD.
#   sbatch scripts/port_gate.sbatch <repository> <test file or node id>...
# Output: $MJX_RUNS/port/gate-<commit7>-<job>/ (provenance.txt, pytest.log, report.xml, verdict.txt); the verdict is
# scripts/check_pytest_report.py on the report AND pytest's exit code, never the job state. Nothing is deleted.
set -u
REPO=${1:?repository}; shift
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD) || exit 1
OUT=$MJX_RUNS/port/gate-${COMMIT:0:7}-${SLURM_JOB_ID:?sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src" || exit 1
git -C "$REPO" status --porcelain > "$OUT/git_status.txt"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) commit $COMMIT tests $*" | tee "$OUT/provenance.txt"
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPYCACHEPREFIX=$OUT/pycache PYTHONPATH=$OUT/src
T0=$(date +%s)
"$PY" -m pytest -q -rfEs -p no:cacheprovider --basetemp="$OUT/pytest_tmp" --junitxml="$OUT/report.xml" \
    --durations=20 "$@" > "$OUT/pytest.log" 2>&1
RC=$?
echo "pytest exit code $RC, wall $(( $(date +%s) - T0 )) s" | tee -a "$OUT/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT/report.xml" > "$OUT/verdict.txt"
CRC=$?
cat "$OUT/verdict.txt"
if [ "$RC" -ne 0 ] || [ "$CRC" -ne 0 ]; then echo "GATE FAIL (pytest rc $RC, check rc $CRC) -> $OUT"; exit 1; fi
echo "GATE PASS -> $OUT"
