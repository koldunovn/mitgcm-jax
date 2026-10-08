#!/bin/bash
#SBATCH --job-name=mjx_tier2
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00          # size it from the measured first-call times (test_sharded_grad_gpu.py docstring)
#SBATCH --output=slurm-%x-%j.out
#
# Tier 2: the GPU tests (sharded gradients on 4 real A100s vs 1, repeats, linearity guards) on one GPU node.
#   sbatch [--gpus=1] [--time=..] scripts/run_tier2.sbatch [repository] [pytest -k expression]
# (default: --gpus=4; repository = the checkout holding this script, or $MJX_TREE). A test that needs more GPUs than
# the job has skips itself (the skip is listed by the verdict). Pass/fail is decided by scripts/check_pytest_report.py
# on the JUnit report AND pytest's exit code, never by the job state. Each job writes into its own new OUT_DIR
# ($MJX_RUNS/tier2/<job>, exported to the tests as MJX_TIER2_OUT for their saved arrays) and never deletes anything;
# the tests run on a frozen `git archive` snapshot of the repository's HEAD in $OUT_DIR/src (uncommitted changes are
# not tested and are reported). Modelled on scripts/run_tier1x.sbatch.
set -u

# repository: argument 1, else $MJX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy:
# scontrol knows the original). Paths: mitjax/paths.py (MJX_WORK, ...); python: $MJX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${1:-${MJX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}}
KEXPR=${2:-}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (pass it or set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT_DIR=$MJX_RUNS/tier2/${SLURM_JOB_ID:?run this through sbatch}
if [ -e "$OUT_DIR" ]; then echo "FAIL: OUT_DIR $OUT_DIR already exists"; exit 1; fi
mkdir -p "$OUT_DIR"

# Frozen snapshot (the milestone-0 review found the runners testing the live checkout): `git archive HEAD` unpacked
# into $OUT_DIR/src (never deleted).
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
  echo "gpus     ${SLURM_GPUS_ON_NODE:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "select   ${KEXPR:-<all tier2>}"
} | tee "$OUT_DIR/provenance.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT_DIR/gpus.csv" 2>&1
if [ "$DIRTY" -ne 0 ]; then
  echo "WARNING: $REPO has $DIRTY uncommitted paths (git_status.txt, git_diff.patch): they are NOT tested;" \
       "this job tests commit $COMMIT only" | tee "$OUT_DIR/DIRTY_NOT_TESTED.txt"
fi
cd "$SRC" || exit 1

# XLA_FLAGS is set by conftest.py (the GPU-only set below). No persistent compilation cache (user decision). The
# CPU backend stays available next to CUDA: the tests build the model on the host CPU and place it on the GPUs.
# PYTHONPATH pinned to the snapshot; bytecode outside the repository.
export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=$SRC
export PYTHONPYCACHEPREFIX=$OUT_DIR/pycache
export MJX_TIER2_OUT=$OUT_DIR
# GPU-only flag set (mitjax/xla_flags.py flag_set): the gate flags without --xla_disable_hlo_passes=algsimp, which more
# than doubles the XLA GPU compile of a gradient program; GPU tests compare against measured repeat floors only.
export MJX_XLA_FLAG_SET=gpu
# Unbuffered output and a stack dump of every thread after 10 min in one test: job 27832889 hit its time limit inside
# the first gradient call and left an EMPTY pytest.log (block-buffered stdout), so where the time went was lost.
export PYTHONUNBUFFERED=1

T0=$(date +%s)
"$PY" -m pytest -m "tier2" ${KEXPR:+-k "$KEXPR"} -q -rfEs -s -o faulthandler_timeout=600 \
    --basetemp="$OUT_DIR/pytest_tmp" --junitxml="$OUT_DIR/report.xml" --durations=15 \
    > "$OUT_DIR/pytest.log" 2>&1
PYTEST_RC=$?
cat "$OUT_DIR/pytest.log"
WALL=$(( $(date +%s) - T0 ))

echo "pytest exit code $PYTEST_RC, wall ${WALL} s" | tee -a "$OUT_DIR/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT_DIR/report.xml" > "$OUT_DIR/verdict.txt"
CHECK_RC=$?
cat "$OUT_DIR/verdict.txt"

if [ "$PYTEST_RC" -ne 0 ] || [ "$CHECK_RC" -ne 0 ]; then
  echo "TIER2 FAIL (pytest rc $PYTEST_RC, report check rc $CHECK_RC) -> $OUT_DIR (commit $COMMIT)"
  exit 1
fi
echo "TIER2 PASS -> $OUT_DIR (commit $COMMIT; uncommitted paths not tested: $DIRTY)"
