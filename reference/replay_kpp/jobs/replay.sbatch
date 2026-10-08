#!/bin/bash
#SBATCH --job-name=mjx_replay_kpp
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build and run the KPP replay harness (M3 sub-lane KPP; reference/replay_kpp/build.sh, code/the_main_loop.F; a copy
# of reference/replay_col, which stays lane COL's):
#   sbatch -J mjx_replay_kpp_<tag> reference/replay_kpp/jobs/replay.sbatch MJX_COMMIT EXP:CODE INPUT_DIR [INPUT_DIR...]
# e.g. vermix:code input input.dd. Arguments go after the script name, never through --export.
#
# OUT = $MJX_REFERENCE/replay_kpp/<EXP>-<CODE>-<upstream short>-<MJX_COMMIT short>-job<id> (new; refuses an existing
# one).
#   1. git archive MJX_COMMIT reference mitjax/paths.py -> OUT/scripts (the frozen snapshot that builds and runs)
#   2. OUT/scripts/reference/replay_kpp/build.sh (genmake2 + make, preprocessed-source identity with the M3 oracle
#      build $MJX_REFERENCE/build/<EXP>-<CODE>-<upstream short>-463504e, frozen OUT/bin)
#   3. per INPUT_DIR: run directory with lane A's reference/make_rundir.py (--binary OUT/bin, --runs OUT/runs, run id
#      job<id>), synthetic inputs (reference/replay_kpp/replay_io.py write-inputs), lane A's reference/run.sh (exit 0)
#   4. the outputs must exist and the harness line must be in STDOUT; verdict line "REPLAY OK <run dir>" per input
#      directory, or "REPLAY FAIL ...".
# Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "REPLAY FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"

COMMIT_ARG=${1:?mitjax commit}
SPEC=${2:?EXP:CODE}
shift 2
[ $# -ge 1 ] || { echo "REPLAY FAIL: no INPUT_DIR"; exit 1; }
IFS=: read -r EXP CODE <<< "$SPEC"
[ -n "$EXP" ] && [ -n "$CODE" ] || { echo "REPLAY FAIL: bad spec $SPEC (EXP:CODE)"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}") || { echo "REPLAY FAIL: no commit $COMMIT_ARG"; exit 1; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "REPLAY FAIL: no branch pinned"; exit 1; }
NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
OUT=$MJX_REFERENCE/replay_kpp/$NAME
ORACLE=$MJX_REFERENCE/build/$EXP-$CODE-${UP:0:7}-463504e
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP -> $OUT"
[ -e "$OUT" ] && { echo "REPLAY FAIL: $OUT exists"; exit 1; }
[ -d "$ORACLE/bld" ] || { echo "REPLAY FAIL: no oracle build $ORACLE (the preprocessed-source identity check needs it)"; exit 1; }
mkdir -p "$OUT/scripts"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$OUT/scripts"
echo "$COMMIT" > "$OUT/scripts/COMMIT"

bash "$OUT/scripts/reference/replay_kpp/build.sh" "$OUT" "$EXP" "$CODE" "$COMMIT" "$UP" "$ORACLE" \
  || { echo "REPLAY FAIL build (log $OUT/build.log)"; exit 1; }

RUNID=job${SLURM_JOB_ID:-none}
FAIL=0
for INP in "$@"; do
  "$PY" "$OUT/scripts/reference/make_rundir.py" "$EXP" "$INP" --binary "$OUT/bin" --runs "$OUT/runs" --run-id "$RUNID" \
    || { echo "REPLAY FAIL make_rundir $INP"; FAIL=1; continue; }
  TOP=$OUT/runs/$EXP/$INP/$RUNID
  "$PY" "$OUT/scripts/reference/replay_kpp/replay_io.py" write-inputs "$TOP/rundir" --size "$OUT/mods/SIZE.h" \
    || { echo "REPLAY FAIL write-inputs $INP"; FAIL=1; continue; }
  bash "$OUT/scripts/reference/run.sh" "$TOP"
  RC=$?
  [ "$RC" -eq 0 ] || { echo "REPLAY FAIL run $INP rc $RC ($TOP/rundir/output.txt)"; FAIL=1; continue; }
  for f in kpp_out.bin kpp_fixed.bin; do
    [ -s "$TOP/rundir/$f" ] || { echo "REPLAY FAIL $INP: no $f"; FAIL=1; }
  done
  grep "REPLAY KPP harness" "$TOP/rundir/output.txt" || { echo "REPLAY FAIL $INP: no harness line in STDOUT"; FAIL=1; continue; }
  echo "REPLAY OK $INP $(cd "$TOP/rundir" && pwd -P)"
done
exit $FAIL
