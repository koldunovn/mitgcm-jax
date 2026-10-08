#!/bin/bash
#SBATCH --job-name=mjx_tier1
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00          # short limit: backfills sooner; tier-1 budget is 10 min
#SBATCH --output=slurm-%x-%j.out
#
# Tier 1: every smoke + tier1 test on one CPU compute node (budget: < 10 min, < 100 tests).
#   sbatch scripts/run_tier1.sbatch [repository]        (default: the checkout holding this script, or $MJX_TREE)
# Pass/fail is decided by scripts/check_pytest_report.py on the JUnit report AND pytest's exit code, never by the
# job state. Each job writes into its own new OUT_DIR and never deletes anything; the tests run on a frozen
# snapshot of the repository's HEAD in $OUT_DIR/src (uncommitted changes are not tested and are reported). Tests
# are selected by marker over pyproject's test directories (all of them), never from a hand-kept directory list.
set -u

# repository: argument 1, else $MJX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy:
# scontrol knows the original). Paths: mitjax/paths.py (MJX_WORK, ...); python: $MJX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${1:-${MJX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (pass it or set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT_DIR=$MJX_RUNS/tier1/${SLURM_JOB_ID:?run this through sbatch}
if [ -e "$OUT_DIR" ]; then echo "FAIL: OUT_DIR $OUT_DIR already exists"; exit 1; fi
mkdir -p "$OUT_DIR"

# Frozen snapshot (the milestone-0 review found the runners testing the live checkout): the tests run on
# `git archive HEAD` of the repository, unpacked into $OUT_DIR/src (never deleted), not on the live checkout;
# uncommitted changes are NOT tested (said loudly below).
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD) || { echo "FAIL: $REPO is not a git checkout"; exit 1; }
git -C "$REPO" status --porcelain > "$OUT_DIR/git_status.txt"
git -C "$REPO" diff > "$OUT_DIR/git_diff.patch"
DIRTY=$(wc -l < "$OUT_DIR/git_status.txt")
SRC=$OUT_DIR/src
mkdir "$SRC" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$SRC" ||
  { echo "FAIL: cannot unpack the snapshot of $COMMIT into $SRC"; exit 1; }
echo "$COMMIT" > "$SRC/COMMIT"
{
  echo "job      $SLURM_JOB_ID on $(hostname), $(date -Is)"
  echo "repo     $REPO"
  echo "commit   $COMMIT"
  echo "snapshot $SRC (git archive of the commit; the tests run here)"
  echo "dirty    $DIRTY paths (NOT tested)"
  echo "python   $PY"
} | tee "$OUT_DIR/provenance.txt"
if [ "$DIRTY" -ne 0 ]; then
  echo "WARNING: $REPO has $DIRTY uncommitted paths (git_status.txt, git_diff.patch): they are NOT tested;" \
       "this job tests commit $COMMIT only" | tee "$OUT_DIR/DIRTY_NOT_TESTED.txt"
fi
cd "$SRC" || exit 1

# XLA_FLAGS (gate flags, 4 fake devices) is set by conftest.py. No persistent compilation cache (user decision).
# PYTHONPATH pinned to the snapshot for subprocess tests; bytecode outside the repository (home quota, stale NFS).
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=$SRC
export PYTHONPYCACHEPREFIX=$OUT_DIR/pycache

# the tier-1 test budget: TIER1_MAX_TESTS of the module mitjax.tests.manifest (test_manifest.py checks it is read)
MAX_TESTS=$("$PY" -c "from mitjax.tests.manifest import TIER1_MAX_TESTS; print(TIER1_MAX_TESTS)") ||
  { echo "FAIL: cannot read TIER1_MAX_TESTS from the snapshot"; exit 1; }

T0=$(date +%s)
"$PY" -m pytest -m "smoke or tier1" -q -rfEs \
    --basetemp="$OUT_DIR/pytest_tmp" --junitxml="$OUT_DIR/report.xml" --durations=15 \
    > "$OUT_DIR/pytest.log" 2>&1
PYTEST_RC=$?
cat "$OUT_DIR/pytest.log"
WALL=$(( $(date +%s) - T0 ))

echo "pytest exit code $PYTEST_RC, wall ${WALL} s" | tee -a "$OUT_DIR/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT_DIR/report.xml" --max-tests "$MAX_TESTS" > "$OUT_DIR/verdict.txt"
CHECK_RC=$?
cat "$OUT_DIR/verdict.txt"
if [ "$WALL" -gt 600 ]; then
  echo "WARNING: tier 1 took ${WALL} s, over the 600 s budget" | tee -a "$OUT_DIR/verdict.txt"
fi

if [ "$PYTEST_RC" -ne 0 ] || [ "$CHECK_RC" -ne 0 ]; then
  echo "TIER1 FAIL (pytest rc $PYTEST_RC, report check rc $CHECK_RC) -> $OUT_DIR (commit $COMMIT)"
  exit 1
fi
echo "TIER1 PASS -> $OUT_DIR (commit $COMMIT; uncommitted paths not tested: $DIRTY)"
