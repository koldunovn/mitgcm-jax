#!/bin/bash
#SBATCH -J mjx_gridshard
#SBATCH -p compute
#SBATCH -A ab0995
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --output=slurm-%x-%j.out
# usage: sbatch scripts/grid_sharding/run.sbatch <frozen checkout dir> [jit limit s]
set -euo pipefail
SRC=$1
LIMIT=${2:-900}
eval "$(python3 "$SRC/mitjax/paths.py" --sh)"
OUT=$MJX_RUNS/grid_sharding/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$SRC"
export JAX_PLATFORMS=cpu PYTHONPATH=$SRC PYTHONPYCACHEPREFIX=$OUT/pyc
"$MJX_PYTHON" scripts/grid_sharding/measure.py "$LIMIT" > "$OUT/measure.log" 2>&1
echo "exit $?" >> "$OUT/measure.log"
