#!/bin/bash
#SBATCH --job-name=mjx_gsumref
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#
# Fortran-order global-sum reference (plan Task 7b): compile scripts/global_sum_ref/gsumref.F with the oracle's
# compiler (reference/build.sh: gcc/11.2.0 module) and its FFLAGS/FOPTIM (the Makefile of every oracle build:
# -O0 -ffp-contract=off -fconvert=big-endian ...), run it on make_input.py's input, write
#   $MJX_REFERENCE/exch_maps/global_sum_ref-job<ID>/{gsumref.F, make_input.py, input.bin, output.bin, SHA256SUMS,
#   provenance.txt}
# Read by mitjax/tests/test_global_sum.py. Never deletes or overwrites anything (a new directory per job).
#   sbatch scripts/global_sum_ref/run.sbatch [repository]
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${1:-${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
OUT=$MJX_REFERENCE/exch_maps/global_sum_ref-job${SLURM_JOB_ID:?run this through sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT"
cp "$REPO/scripts/global_sum_ref/gsumref.F" "$REPO/scripts/global_sum_ref/make_input.py" "$OUT/"
FFLAGS="-fallow-argument-mismatch -fconvert=big-endian -fimplicit-none -mcmodel=medium -Wall"
FFLAGS="$FFLAGS -Wno-unused-dummy-argument -ffp-contract=off"
FOPTIM="-O0"
cd "$OUT" || exit 1
GIT=$(command -v git)            # git comes from a module: resolve it before `module purge` (as reference/build.sh)
COMMIT=$("$GIT" -C "$REPO" rev-parse HEAD); DIRTY=$("$GIT" -C "$REPO" status --porcelain | wc -l)
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0
{
  echo "job       $SLURM_JOB_ID on $(hostname), $(date -Is)"
  echo "repo      $REPO at $COMMIT ($DIRTY dirty paths)"
  echo "gfortran  $(command -v gfortran): $(gfortran --version | head -1)"
  echo "compile   gfortran $FFLAGS $FOPTIM -o gsumref gsumref.F"
} | tee provenance.txt
gfortran $FFLAGS $FOPTIM -o gsumref gsumref.F || { echo "FAIL: compile"; exit 1; }
"$PY" make_input.py . || exit 1
./gsumref || { echo "FAIL: gsumref"; exit 1; }
sha256sum gsumref.F make_input.py input.bin output.bin | tee SHA256SUMS
echo "GSUMREF OK $OUT"
