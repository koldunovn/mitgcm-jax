#!/bin/bash
#SBATCH --job-name=mjx_tier1x
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00          # tier 1x: nightly and whenever a kernel or reference/ changes (3:33 on 3e44b5d, job 27917464)
#SBATCH --output=slurm-%x-%j.out
#
# Tier 1x: every tier1x test on one CPU compute node (substep and replay gates, negative controls, FD sweeps,
# frozen-oracle checks; no test-count budget).
#   sbatch scripts/run_tier1x.sbatch [repository]        (default: the checkout holding this script, or $MJX_TREE)
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
OUT_DIR=$MJX_RUNS/tier1x/${SLURM_JOB_ID:?run this through sbatch}
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
export PYTHONPATH=$SRC
export PYTHONPYCACHEPREFIX=$OUT_DIR/pycache

# Parallel shards (default 6, TIER1X_PARALLEL=1 for one process): scripts/tier1x_shards.py splits the tier1x files
# of the manifest by their measured times; each shard runs on its own block of cores and runs its files one by one,
# each FILE in its own pytest process with its own JUnit report, log and basetemp. A process that dies (an XLA
# compile abort killed whole runs twice: 27832296, 27833481) then loses only its file, whose missing report fails
# the verdict by name; the verdict sums all reports (check_pytest_report.py).
NPAR=${TIER1X_PARALLEL:-6}
NCPU=${SLURM_CPUS_PER_TASK:-128}
PER=$(( NCPU / NPAR ))
export OMP_NUM_THREADS=$PER
run_shard() {          # $1 = shard index: its files one per pytest process, in the shard's order
  local k=$1 n=0 rc=0 f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    n=$(( n + 1 ))
    echo "== shard $k file $n: $f" >> "$OUT_DIR/pytest_$k.log"
    taskset -c $(( k*PER ))-$(( (k+1)*PER - 1 )) \
      "$PY" -m pytest -m "tier1x" -q -rfEs -p no:cacheprovider \
        --basetemp="$OUT_DIR/pytest_tmp_${k}_$n" --junitxml="$OUT_DIR/report_${k}_$n.xml" --durations=5 \
        "$f" >> "$OUT_DIR/pytest_$k.log" 2>&1 || rc=1
    echo "$OUT_DIR/report_${k}_$n.xml" >> "$OUT_DIR/reports_$k.txt"
  done < "$OUT_DIR/shard_$k.txt"
  return $rc
}
T0=$(date +%s)
PIDS=()
for (( k=0; k<NPAR; k++ )); do
  "$PY" scripts/tier1x_shards.py --shards "$NPAR" --index "$k" > "$OUT_DIR/shard_$k.txt" ||
    { echo "FAIL: cannot compute shard $k"; exit 1; }
  : > "$OUT_DIR/reports_$k.txt"
  run_shard "$k" &
  PIDS+=($!)
done
PYTEST_RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || PYTEST_RC=1; done
cat "$OUT_DIR"/pytest_*.log > "$OUT_DIR/pytest.log"
cat "$OUT_DIR/pytest.log"
WALL=$(( $(date +%s) - T0 ))
mapfile -t REPORTS < <(cat "$OUT_DIR"/reports_*.txt)

echo "pytest exit code $PYTEST_RC (any shard nonzero -> 1), $NPAR shards, wall ${WALL} s" | tee -a "$OUT_DIR/provenance.txt"
"$PY" scripts/check_pytest_report.py "${REPORTS[@]}" > "$OUT_DIR/verdict.txt"
CHECK_RC=$?
cat "$OUT_DIR/verdict.txt"

if [ "$PYTEST_RC" -ne 0 ] || [ "$CHECK_RC" -ne 0 ]; then
  echo "TIER1X FAIL (pytest rc $PYTEST_RC, report check rc $CHECK_RC) -> $OUT_DIR (commit $COMMIT)"
  exit 1
fi
echo "TIER1X PASS -> $OUT_DIR (commit $COMMIT; uncommitted paths not tested: $DIRTY)"
