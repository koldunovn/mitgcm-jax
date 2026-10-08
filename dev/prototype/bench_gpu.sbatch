#!/bin/bash
#SBATCH --job-name=mjx_bench_c
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#
# A100 timing of the Task 8 code styles (dev/prototype/bench_gpu.py) from a frozen snapshot of one commit:
#   sbatch -J mjx_bench_c_<commit> dev/prototype/bench_gpu.sbatch COMMIT [REPO [raw]]
# (third argument `raw`: also the run with the Fortran-index style called through plain-array signatures)
# OUT = $MJX_RUNS/lane_c/bench-<commit>-job<id> (new). Production XLA flags (XLA_FLAGS unset). Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${2:-${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}}
[ -f "$REPO/mitjax/paths.py" ] || { echo "BENCH FAIL: no mitjax repository at $REPO"; exit 1; }
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}") || { echo "BENCH FAIL: no commit $1"; exit 1; }
OUT=$MJX_RUNS/lane_c/bench-${COMMIT:0:7}-job${SLURM_JOB_ID:?run through sbatch}
[ -e "$OUT" ] && { echo "BENCH FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src"
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
unset XLA_FLAGS
export PYTHONUNBUFFERED=1 PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pycache
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) commit $COMMIT -> $OUT"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" "$OUT/src/dev/prototype/bench_gpu.py" "$OUT/bench.json" || { echo "BENCH FAIL"; exit 1; }
if [ "${3:-}" = "raw" ]; then
  "$PY" "$OUT/src/dev/prototype/bench_gpu.py" "$OUT/bench_raw_signature.json" --raw-signature || { echo "BENCH FAIL raw"; exit 1; }
fi
echo "BENCH OK -> $OUT/bench.json"
