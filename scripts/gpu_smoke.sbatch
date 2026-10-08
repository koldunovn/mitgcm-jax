#!/bin/bash
#SBATCH --job-name=mjx_gpu_smoke
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00          # <= 0.5 A100-h
#SBATCH --output=slurm-%x-%j.out
#
# M1 GPU smoke (plan Task 18): scripts/gpu_smoke.py of a frozen git archive of REPO's HEAD on one A100, under a
# 25-minute `timeout` (compile canary [F§1]). sbatch scripts/gpu_smoke.sbatch REPO [timing]
# (timing: scripts/step_timing.py --gpu on global_ocean.90x40x15/input instead, 10 steps)
set -u
REPO=${1:?repo}; MODE=${2:-smoke}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT=$MJX_RUNS/gpu_smoke/${SLURM_JOB_ID}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
echo "$COMMIT" > "$OUT/COMMIT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT/gpus.csv" 2>&1
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cuda,cpu MJX_XLA_FLAG_SET=gpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONUNBUFFERED=1
export PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pycache
if [ "$MODE" = timing ]; then
  timeout 1500 "$PY" scripts/step_timing.py global_ocean.90x40x15 input "$OUT/timing" --steps 10 --gpu
else
  timeout 1500 "$PY" scripts/gpu_smoke.py "$OUT/smoke"
fi
echo "exit code $?"
