#!/bin/bash
#SBATCH --job-name=mjx_minmax_probe
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#
# gfortran's REAL*8 MAX/MIN on ties (+0/-0) and NaN with the oracle's compiler and flags (GAD-A lane):
#   sbatch reference/replay_gad_a/jobs/minmax_probe.sbatch MJX_COMMIT
# OUT = $MJX_REFERENCE/replay_gad_a/minmax_probe-<commit>-job<id>/{minmax_probe.F (from git archive), mmprobe,
# minmax.txt}. Verdict line "PROBE OK <file>". Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../../.." && pwd)}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}") || { echo "PROBE FAIL: no commit $1"; exit 1; }
OUT=$MJX_REFERENCE/replay_gad_a/minmax_probe-${COMMIT:0:7}-job${SLURM_JOB_ID:?run through sbatch}
[ -e "$OUT" ] && { echo "PROBE FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT"
git -C "$REPO" archive --format=tar "$COMMIT" reference/replay_gad_a/probe/minmax_probe.F | tar -x -O > "$OUT/minmax_probe.F"
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0                  # as reference/build.sh
set -u
cd "$OUT"
FLAGS="-O0 -ffp-contract=off -fconvert=big-endian -fimplicit-none -mcmodel=medium"   # optfile_levante_gfortran, -ieee
{ echo "gfortran $(command -v gfortran): $(gfortran --version | head -1)"; echo "flags $FLAGS"; } > provenance.txt
gfortran $FLAGS -o mmprobe minmax_probe.F || { echo "PROBE FAIL compile"; exit 1; }
{ echo 0 | ./mmprobe; echo 1 | ./mmprobe; } > minmax.txt || { echo "PROBE FAIL run"; exit 1; }
[ "$(wc -l < minmax.txt)" -eq 100 ] || { echo "PROBE FAIL: $(wc -l < minmax.txt) lines"; exit 1; }
echo "PROBE OK $OUT/minmax.txt"
