#!/bin/bash
#SBATCH --job-name=mjx_vecinv_minmax
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# MAX/MIN site tables (tools/fortran_minmax_sites.py, lane MINMAX) of the VECINV harness builds, and the per-routine
# comparison with the M1 table of global_ocean.90x40x15 (the table test_minmax_sites.py audits the kernels against):
#   sbatch reference/replay_vecinv/jobs/minmax_sites.sbatch REPO BUILD_DIR...
# Tables go to $MJX_REFERENCE/minmax_sites/replay_vecinv__<build>.json (the tool never overwrites a different table).
# Runs the tool from a frozen snapshot (git archive HEAD) in $MJX_RUNS/vecinv/<job>/src; never deletes anything.
set -u
REPO=${1:?repository}; shift
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT=$MJX_RUNS/vecinv/${SLURM_JOB_ID:?run this through sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$OUT/src"
cd "$OUT/src"
GITDIR=$(dirname "$(command -v git)")                 # the tool records the commit; module purge drops git from PATH
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0
set -u
export PATH=$PATH:$GITDIR
JAX_PLATFORMS=cpu PYTHONPATH="$OUT/src" "$PY" tools/fortran_minmax_sites.py run "$@" -j 32 || { echo "MINMAX FAIL run"; exit 1; }
T=$MJX_REFERENCE/minmax_sites
args=("$T/global_ocean.90x40x15-code-63cdc0b-f48c062.json")
for b in "$@"; do args+=("$T/replay_vecinv__$(basename "$b").json"); done
JAX_PLATFORMS=cpu PYTHONPATH="$OUT/src" "$PY" tools/fortran_minmax_sites.py compare "${args[@]}"
echo "minmax exit $?"
