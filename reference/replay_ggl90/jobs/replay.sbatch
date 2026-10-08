#!/bin/bash
#SBATCH --job-name=mjx_replay_ggl90
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build and run the GGL90 replay harness (M3 sub-lane GGL90; reference/replay_ggl90/build.sh,
# code/the_main_loop.F; a copy of the COL harness reference/replay_col):
#   sbatch -J mjx_replay_ggl90_<tag> reference/replay_ggl90/jobs/replay.sbatch MJX_COMMIT EXP:CODE:INPUT_DIR ORACLE_TAG
# e.g. vermix:code:input.gglLC 463504e, global_ocean.90x40x15:code:input.idemix f48c062 (ORACLE_TAG: the lane-A
# oracle build of that code, $MJX_REFERENCE/build/<EXP>-<CODE>-<upstream short>-<ORACLE_TAG>, whose preprocessed
# routines must be byte-identical). Arguments go after the script name, never through --export.
#
# OUT = $MJX_REFERENCE/replay_ggl90/<EXP>-<CODE>-<upstream short>-<MJX_COMMIT short>-job<id> (new; refuses an existing
# one).
#   1. git archive MJX_COMMIT reference mitjax/paths.py -> OUT/scripts (the frozen snapshot that builds and runs)
#   2. OUT/scripts/reference/replay_ggl90/build.sh (genmake2 + make, identity check vs the oracle build, frozen OUT/bin)
#   3. run directory with lane A's reference/make_rundir.py (testreport's linkdata + prepare_run of EXP/INPUT_DIR),
#      synthetic inputs (reference/replay_ggl90/replay_io.py write-inputs), then lane A's reference/run.sh: exit 0
#   4. the outputs must exist and the harness line must be in STDOUT; verdict line "REPLAY OK <run dir>" or
#      "REPLAY FAIL ...".
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
SPEC=${2:?EXP:CODE:INPUT_DIR}
OTAG=${3:?oracle build tag}
IFS=: read -r EXP CODE INP <<< "$SPEC"
[ -n "$EXP" ] && [ -n "$CODE" ] && [ -n "$INP" ] || { echo "REPLAY FAIL: bad spec $SPEC (EXP:CODE:INPUT_DIR)"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}") || { echo "REPLAY FAIL: no commit $COMMIT_ARG"; exit 1; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "REPLAY FAIL: no branch pinned"; exit 1; }
NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
OUT=$MJX_REFERENCE/replay_ggl90/$NAME
ORACLE=$MJX_REFERENCE/build/$EXP-$CODE-${UP:0:7}-$OTAG
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP -> $OUT"
[ -e "$OUT" ] && { echo "REPLAY FAIL: $OUT exists"; exit 1; }
[ -d "$ORACLE/bld" ] || { echo "REPLAY FAIL: no oracle build $ORACLE (the preprocessed-source identity check needs it)"; exit 1; }
mkdir -p "$OUT/scripts"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$OUT/scripts"
echo "$COMMIT" > "$OUT/scripts/COMMIT"

bash "$OUT/scripts/reference/replay_ggl90/build.sh" "$OUT" "$EXP" "$CODE" "$COMMIT" "$UP" "$ORACLE" \
  || { echo "REPLAY FAIL build (log $OUT/build.log)"; exit 1; }

RUNID=job${SLURM_JOB_ID:-none}
"$PY" "$OUT/scripts/reference/make_rundir.py" "$EXP" "$INP" --binary "$OUT/bin" --runs "$OUT/runs" --run-id "$RUNID" \
  || { echo "REPLAY FAIL make_rundir"; exit 1; }
TOP=$OUT/runs/$EXP/$INP/$RUNID
"$PY" "$OUT/scripts/reference/replay_ggl90/replay_io.py" write-inputs "$TOP/rundir" --size "$OUT/mods/SIZE.h" \
  --exp "$EXP" || { echo "REPLAY FAIL write-inputs"; exit 1; }
bash "$OUT/scripts/reference/run.sh" "$TOP"
RC=$?
[ "$RC" -eq 0 ] || { echo "REPLAY FAIL run rc $RC ($TOP/rundir/output.txt)"; exit 1; }
for f in ggl_parm.bin ggl_out.bin ggl_grid.bin; do
  [ -s "$TOP/rundir/$f" ] || { echo "REPLAY FAIL: no $f"; exit 1; }
done
grep "REPLAY GGL90 harness" "$TOP/rundir/output.txt" || { echo "REPLAY FAIL: no harness line in STDOUT"; exit 1; }
echo "REPLAY OK $(cd "$TOP/rundir" && pwd -P)"
