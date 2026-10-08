#!/bin/bash
#SBATCH --job-name=mjx_r5_gpu
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=slurm-%x-%j.out
#
# R5 optim adjoint on A100s (plan Task 17): scripts/r5_shardgrad/gpu.py of a frozen git archive of REPO's HEAD.
#   sbatch --gpus=1 --time=00:18:00 scripts/r5_shardgrad/gpu.sbatch REPO p1
#   sbatch --gpus=4 --time=00:18:00 scripts/r5_shardgrad/gpu.sbatch REPO pN 4
# (budget: <= 1.5 A100-h for both jobs; the program runs under `timeout` 2 minutes inside the job's limit)
set -u
REPO=${1:?repo}; MODE=${2:?p1|pN}; P=${3:-4}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT=$MJX_RUNS/r5_gpu/${SLURM_JOB_ID}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
echo "$COMMIT" > "$OUT/COMMIT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT/gpus.csv" 2>&1
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cuda,cpu MJX_XLA_FLAG_SET=gpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONUNBUFFERED=1
export PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pycache
LIMIT=$(( $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 120 ))
timeout "$LIMIT" "$PY" -u scripts/r5_shardgrad/gpu.py "$MODE" "$OUT/res" "$P"
echo "exit code $?"
