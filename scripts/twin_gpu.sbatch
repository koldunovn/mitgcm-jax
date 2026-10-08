#!/bin/bash
#SBATCH --job-name=mjx_twinG
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:58:00          # <= 1 A100-h; measured 16 min (job 27840809)
#SBATCH --output=slurm-%x-%j.out
# Tier-3 twin (plan Task 18, lane B): the JAX run of global_ocean.90x40x15/input.twin through `python -m mitjax run`
# on ONE A100 (MJX_XLA_FLAG_SET=gpu), under a 55-minute timeout. Output under $MJX_RUNS/twin. Deletes nothing.
set -u
REPO=${1:?repo}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
SHADOW=$MJX_RUNS/twin/MITgcm
OUT=$MJX_RUNS/twin/jax_gpu-${SLURM_JOB_ID}
[ -e "$OUT" ] && { echo "exists"; exit 1; }
mkdir -p "$OUT/src"
C=$(git -C "$REPO" rev-parse HEAD); git -C "$REPO" archive --format=tar "$C" | tar -x -C "$OUT/src"; echo "$C" > "$OUT/src/COMMIT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT/gpus.csv" 2>&1
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cuda,cpu MJX_XLA_FLAG_SET=gpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
export PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pyc
T0=$(date +%s)
timeout 3300 "$PY" -m mitjax run "$SHADOW/verification/global_ocean.90x40x15" --variant input.twin --out "$OUT/run"
echo "EXIT $? wall $(( $(date +%s) - T0 )) s"
