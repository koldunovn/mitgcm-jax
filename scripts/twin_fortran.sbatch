#!/bin/bash
#SBATCH --job-name=mjx_twinF
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:50:00          # one serial 10-year run: ~24 min measured (jobs 27840489, 27840711, 27841031)
#SBATCH --output=slurm-%x-%j.out
# Tier-3 twin (plan Task 18, lane B): one Fortran run of the shadow tree's global_ocean.90x40x15 twin variant
# (scripts/twin_setup.py) with one of lane A's frozen binaries, run directory under $MJX_RUNS/twin (never
# $MJX_REFERENCE). Deletes nothing.
#   sbatch scripts/twin_fortran.sbatch REPO TAG BINARY VARIANT [mpi]
#   F1: F1 global_ocean.90x40x15-code-63cdc0b-f48c062 input.twin
#   F2: F2 global_ocean.90x40x15-code-63cdc0b-d21b643-retile input.twin mpi   (data.exch2 of the retile layout)
#   F3: F3 global_ocean.90x40x15-code-63cdc0b-f48c062 input.twin3
set -u
REPO=${1:?repo}; TAG=${2:?tag}; BIN=${3:?binary}; VAR=${4:?variant}; MPI=${5:-}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
SHADOW=$MJX_RUNS/twin/MITgcm
OUT=$MJX_RUNS/twin/fortran-$TAG-${SLURM_JOB_ID}
[ -e "$OUT" ] && { echo "exists"; exit 1; }
mkdir -p "$OUT/src"
C=$(git -C "$REPO" rev-parse HEAD); git -C "$REPO" archive --format=tar "$C" | tar -x -C "$OUT/src"; echo "$C" > "$OUT/src/COMMIT"
"$PY" "$OUT/src/reference/make_rundir.py" global_ocean.90x40x15 "$VAR" --upstream "$SHADOW" --runs "$OUT/runs" \
    --run-id "$TAG" --binary "$BIN" ${MPI:+--mpi} || exit 1
bash "$OUT/src/reference/run.sh" "$OUT/runs/global_ocean.90x40x15/$VAR/$TAG"
