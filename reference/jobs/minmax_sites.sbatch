#!/bin/bash
#SBATCH --job-name=mjx_minmax
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%x-%j.out
#
# MAX/MIN site tables (tools/fortran_minmax_sites.py, lane MINMAX's rule) of oracle builds under $MJX_REFERENCE/build,
# then the cross-build comparison of these tables with the reference tables named on the command line:
#   sbatch -J mjx_minmax_<tag> reference/jobs/minmax_sites.sbatch MJX_COMMIT BUILD_NAME... [-- TABLE.json...]
# BUILD_NAME: a directory under $MJX_REFERENCE/build (e.g. advect_cs-code-63cdc0b-06b4417). Tables go to
# $MJX_REFERENCE/minmax_sites/<build>.json + .sha256 (the tool never overwrites a different table; an identical one is
# accepted). The tool runs from a frozen snapshot (`git archive MJX_COMMIT tools mitjax reference` into
# $MJX_RUNS/snapshots/<job name>-<job id>); its scratch compiles go to $MJX_RUNS/minmax_sites/ (symlinks + objects).
# Verdict lines: the tool's `OK`/`FAIL <build>` per build and the compare summary (`... N with a different p between
# builds`); exit 1 if the run failed or any site has a different p. Arguments after the script name only. Never
# deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
shift
BUILDS=(); EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do BUILDS+=("$MJX_REFERENCE/build/$1"); shift; done
[ $# -gt 0 ] && shift
EXTRA=("$@")
[ ${#BUILDS[@]} -gt 0 ] || { echo "FAIL: no BUILD_NAME given"; exit 2; }
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_minmax}-${SLURM_JOB_ID:?run this through sbatch}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" tools mitjax reference | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) mitjax $COMMIT builds ${#BUILDS[@]}"
GITDIR=$(dirname "$(command -v git)")            # the tool records the commit; module purge drops git from PATH
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0
set -u
export PATH=$PATH:$GITDIR JAX_PLATFORMS=cpu PYTHONPYCACHEPREFIX=$MJX_RUNS/pyc/minmax-$SLURM_JOB_ID
cd "$SNAP"
PYTHONPATH="$SNAP" "$PY" tools/fortran_minmax_sites.py run "${BUILDS[@]}" -j "${SLURM_CPUS_PER_TASK:-32}"
RC=$?
T=$MJX_REFERENCE/minmax_sites
TABLES=("${EXTRA[@]+"${EXTRA[@]}"}")
for b in "${BUILDS[@]}"; do TABLES+=("$T/$(basename "$b").json"); done
OUT=$(PYTHONPATH="$SNAP" "$PY" tools/fortran_minmax_sites.py compare "${TABLES[@]}")
CRC=$?
echo "$OUT"
N=$(echo "$OUT" | sed -n 's/.* \([0-9][0-9]*\) with a different p between builds.*/\1/p' | tail -1)
echo "run exit $RC, compare exit $CRC, sites with a different p: ${N:-?}"
[ "$RC" -eq 0 ] && [ "$CRC" -eq 0 ] && [ "${N:-x}" = 0 ] || { echo "MINMAX FAILED"; exit 1; }
echo "MINMAX OK"
