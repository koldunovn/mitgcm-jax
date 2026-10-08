#!/bin/bash
#SBATCH --job-name=mjx_m1acc
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# M1 acceptance (plan Task 18), one variant: sbatch scripts/m1_acceptance.sbatch REPO EXPERIMENT INPUT PN
# runs scripts/m1_acceptance.py of a frozen git archive of REPO's HEAD into $MJX_RUNS/m1_acceptance/<job>-<exp>-<input>;
# PN = timing runs scripts/step_timing.py instead (steady-state step cost for the tier-3 twin estimate).
set -u
REPO=${1:?repo}; EXP=${2:?experiment}; INP=${3:?input}; PN=${4:-0}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse HEAD)
OUT=$MJX_RUNS/m1_acceptance/${SLURM_JOB_ID}-${EXP}-${INP}
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
echo "$COMMIT" > "$OUT/COMMIT"
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cpu PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pycache
if [ "$PN" = timing ]; then
  "$PY" scripts/step_timing.py "$EXP" "$INP" "$OUT/timing" --steps 10
else
  "$PY" scripts/m1_acceptance.py "$EXP" "$INP" "$OUT/run" --pn "$PN"
fi
