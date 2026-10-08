#!/bin/bash
#SBATCH --job-name=mjx_gates_c
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Task 8 gates on a CPU compute node from a frozen snapshot of one commit:
#   sbatch -J mjx_gates_c_<commit> dev/prototype/run_gates.sbatch COMMIT [REPO]
# Unpacks `git archive COMMIT` into OUT=$MJX_RUNS/lane_c/gates-<commit>-job<id>/src (new; refuses an existing one) and
# runs pytest there on mitjax/tests/test_farray.py and mitjax/tests/test_style_prototype.py (the gate XLA flags
# come from conftest.py). Verdict: scripts/check_pytest_report.py on the JUnit report AND pytest's exit code, never
# the job state; the gates' measurements (negative-control counts, FD errors, divide probe) are in pytest.log
# (-rP). Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${2:-${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}}
[ -f "$REPO/mitjax/paths.py" ] || { echo "GATES FAIL: no mitjax repository at $REPO"; exit 1; }
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}") || { echo "GATES FAIL: no commit $1"; exit 1; }
OUT=$MJX_RUNS/lane_c/gates-${COMMIT:0:7}-job${SLURM_JOB_ID:?run through sbatch}
[ -e "$OUT" ] && { echo "GATES FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src"
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
cd "$OUT/src"
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) commit $COMMIT -> $OUT" | tee "$OUT/provenance.txt"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pycache
T0=$(date +%s)
"$PY" -m pytest -q -rfEsP mitjax/tests/test_farray.py mitjax/tests/test_style_prototype.py \
    --basetemp="$OUT/pytest_tmp" --junitxml="$OUT/report.xml" --durations=25 > "$OUT/pytest.log" 2>&1
RC=$?
cat "$OUT/pytest.log"
echo "pytest exit code $RC, wall $(( $(date +%s) - T0 )) s" | tee -a "$OUT/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT/report.xml" > "$OUT/verdict.txt"
CRC=$?
cat "$OUT/verdict.txt"
if [ "$RC" -ne 0 ] || [ "$CRC" -ne 0 ]; then echo "GATES FAIL (pytest rc $RC, report rc $CRC) -> $OUT"; exit 1; fi
echo "GATES PASS -> $OUT"
