#!/bin/bash
#SBATCH --job-name=mjx_ftzbuild
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Flush-to-zero (FTZ/DAZ) variants of frozen oracle builds (plan decision 13; reference/relink_ftz.sh):
#   sbatch reference/jobs/ftz_build.sbatch MJX_COMMIT SRC_NAME [SRC_NAME ...]
# e.g.  sbatch reference/jobs/ftz_build.sbatch $(git rev-parse --short HEAD) \
#         global_ocean.cs32x15-code-63cdc0b-06b4417 global_ocean.cs32x15-code-63cdc0b-704fd6b-jaxdump
# For each SRC_NAME (a directory under $MJX_REFERENCE/bin with its build directory under $MJX_REFERENCE/build) the
# job creates $MJX_REFERENCE/build/SRC_NAME-ftz-<MJX_COMMIT short>/ (refuses an existing one or an existing bin),
# unpacks `git archive MJX_COMMIT reference mitjax/paths.py` into it and runs its reference/relink_ftz.sh: the source
# build's objects linked with the compiler's crtfastmath.o, verified (objects identical, the standard relink reproduces the source binary, MXCSR FTZ
# and DAZ at MAIN__, the probe), frozen into $MJX_REFERENCE/bin/SRC_NAME-ftz/.
# Verdict: "BUILD OK <name>" / "BUILD FAIL <name> rc=<n>" per build, "ALL BUILDS OK"; exit 1 if any failed.
# Never deletes anything.
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
[ $# -ge 1 ] || { echo "FAIL: no source build given"; exit 2; }
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT"
FAILED=0
for SRC in "$@"; do
  NAME=$SRC-ftz
  BUILD=$MJX_REFERENCE/build/$NAME-${COMMIT:0:7}     # one build directory per attempt; the binary: bin/$NAME
  if [ -e "$BUILD" ] || [ -e "$MJX_REFERENCE/bin/$NAME" ]; then
    echo "BUILD FAIL $NAME: build or bin directory exists (nothing is overwritten)"; FAILED=1; continue
  fi
  mkdir -p "$BUILD/scripts"
  git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$BUILD/scripts"
  echo "$COMMIT" > "$BUILD/scripts/COMMIT"
  bash "$BUILD/scripts/reference/relink_ftz.sh" "$BUILD" "$SRC" "$COMMIT" > "$BUILD/build_job.out" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ] && [ -x "$MJX_REFERENCE/bin/$NAME/mitgcmuv" ]; then
    echo "BUILD OK $NAME -> $MJX_REFERENCE/bin/$NAME"
    grep -E "^(objects|relink std|ftz binary|MXCSR|probe) " "$MJX_REFERENCE/bin/$NAME/ftz_provenance.txt"
  else
    echo "BUILD FAIL $NAME rc=$RC (log $BUILD/build.log)"; tail -15 "$BUILD/build_job.out"; FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] || { echo "BUILDS FAILED"; exit 1; }
echo "ALL BUILDS OK"
