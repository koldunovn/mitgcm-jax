#!/bin/bash
#SBATCH --job-name=mjx_twinJ
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:55:00          # measured 24.9 min (job 27840490)
#SBATCH --output=slurm-%x-%j.out
# Tier-3 twin (plan Task 18, lane B): the JAX run of global_ocean.90x40x15/input.twin through `python -m mitjax run`
# on CPU (the API's XLA flags since docs plan decision 12 b, equal to the gate flags on x86-64 without user XLA_FLAGS:
# mitjax/xla_flags.py). Output under $MJX_RUNS/twin. Deletes nothing.
set -u
REPO=${1:?repo}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
SHADOW=$MJX_RUNS/twin/MITgcm
OUT=$MJX_RUNS/twin/jax_cpu-${SLURM_JOB_ID}
[ -e "$OUT" ] && { echo "exists"; exit 1; }
mkdir -p "$OUT/src"
C=$(git -C "$REPO" rev-parse HEAD); git -C "$REPO" archive --format=tar "$C" | tar -x -C "$OUT/src"; echo "$C" > "$OUT/src/COMMIT"
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pyc
T0=$(date +%s)
"$PY" -m mitjax run "$SHADOW/verification/global_ocean.90x40x15" --variant input.twin --out "$OUT/run"
echo "EXIT $? wall $(( $(date +%s) - T0 )) s"
