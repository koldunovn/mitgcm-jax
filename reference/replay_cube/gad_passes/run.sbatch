#!/bin/bash
#SBATCH --job-name=mjx_gad_passes
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
# gfortran replay of GAD_ADVECTION's cubed-sphere pass structure (make_program.py):
#   sbatch reference/replay_cube/gad_passes/run.sbatch MJX_COMMIT
# OUT = $MJX_REFERENCE/replay_cube/gad_passes-<commit>-job<id> (new); scripts from `git archive MJX_COMMIT`; the
# oracle's compiler (gcc/11.2.0 module) and the oracle's FFLAGS family (-O0); output out.txt. Deletes nothing.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=$(cd "$(dirname "$SELF")/../../.." && pwd)
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}") || exit 1
OUT=$MJX_REFERENCE/replay_cube/gad_passes-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
[ -e "$OUT" ] && { echo "GADPASS FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/scripts"
git -C "$REPO" archive --format=tar "$COMMIT" reference/replay_cube/gad_passes | tar -x -C "$OUT/scripts"
"$PY" "$OUT/scripts/reference/replay_cube/gad_passes/make_program.py" "$MJX_UPSTREAM" "$OUT/gadpass.F" ||
  { echo "GADPASS FAIL generate"; exit 1; }
git -C "$MJX_UPSTREAM" rev-parse pinned > "$OUT/upstream.txt"
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u; type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh; module purge; module load gcc/11.2.0-gcc-11.2.0; set -u
cd "$OUT" || exit 1
gfortran --version | head -1 > compiler.txt
gfortran -O0 -fconvert=big-endian -ffp-contract=off -o gadpass gadpass.F || { echo "GADPASS FAIL compile"; exit 1; }
./gadpass > out.txt || { echo "GADPASS FAIL run"; exit 1; }
echo "GADPASS OK $OUT ($(grep -c '^P' out.txt) pass records)"
