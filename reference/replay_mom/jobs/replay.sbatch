#!/bin/bash
#SBATCH --job-name=mjx_replay_mom
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build and run the MOM replay harness (M1 sub-lane MOM; reference/replay_mom/build.sh, code/the_main_loop.F;
# a copy of the Task 8 harness reference/replay, which stays lane C's):
#   sbatch -J mjx_replay_mom_<tag> reference/replay_mom/jobs/replay.sbatch MJX_COMMIT EXP:CODE:INPUT_DIR
# e.g. tutorial_barotropic_gyre:code:input, tutorial_global_oce_optim:code_ad:input_ad. Arguments go after the script name, never
# through --export.
#
# OUT = $MJX_REFERENCE/replay_mom/<EXP>-<CODE>-<upstream short>-<MJX_COMMIT short>-job<id> (new; refuses an
# existing one).
#   1. git archive MJX_COMMIT reference mitjax/paths.py -> OUT/scripts (the frozen snapshot that builds and runs)
#   2. OUT/scripts/reference/replay_mom/build.sh (genmake2 + make, checks vs the oracle build of the same code at
#      $MJX_REFERENCE/build/<EXP>-<CODE>-<upstream short>-f48c062, frozen OUT/bin)
#   3. run directory with lane A's reference/make_rundir.py (testreport's linkdata + prepare_run of EXP/INPUT_DIR,
#      --binary OUT/bin, --runs OUT/runs, run id job<id>), synthetic inputs (reference/replay_mom/replay_io.py
#      write-inputs), then lane A's reference/run.sh (exit 0 + "Execution ended Normally")
#   4. the outputs must exist; verdict line "REPLAY OK <run dir>" or "REPLAY FAIL ...".
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
IFS=: read -r EXP CODE INP <<< "$SPEC"
[ -n "$EXP" ] && [ -n "$CODE" ] && [ -n "$INP" ] || { echo "REPLAY FAIL: bad spec $SPEC (EXP:CODE:INPUT_DIR)"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}") || { echo "REPLAY FAIL: no commit $COMMIT_ARG"; exit 1; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "REPLAY FAIL: no branch pinned"; exit 1; }
NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
OUT=$MJX_REFERENCE/replay_mom/$NAME
ORACLE=$MJX_REFERENCE/build/$EXP-$CODE-${UP:0:7}-f48c062
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP -> $OUT"
[ -e "$OUT" ] && { echo "REPLAY FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/scripts"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$OUT/scripts"
echo "$COMMIT" > "$OUT/scripts/COMMIT"
[ -d "$ORACLE/bld" ] || { echo "REPLAY FAIL: no oracle build $ORACLE (the preprocessed-source identity check needs it)"; exit 1; }

bash "$OUT/scripts/reference/replay_mom/build.sh" "$OUT" "$EXP" "$CODE" "$COMMIT" "$UP" "$ORACLE" \
  || { echo "REPLAY FAIL build (log $OUT/build.log)"; exit 1; }

RUNID=job${SLURM_JOB_ID:-none}
"$PY" "$OUT/scripts/reference/make_rundir.py" "$EXP" "$INP" --binary "$OUT/bin" --runs "$OUT/runs" --run-id "$RUNID" \
  || { echo "REPLAY FAIL make_rundir"; exit 1; }
TOP=$OUT/runs/$EXP/$INP/$RUNID
"$PY" "$OUT/scripts/reference/replay_mom/replay_io.py" write-inputs "$TOP/rundir" --size "$OUT/mods/SIZE.h" \
  --experiment "$EXP" || { echo "REPLAY FAIL write-inputs"; exit 1; }
bash "$OUT/scripts/reference/run.sh" "$TOP" || { echo "REPLAY FAIL run ($TOP/rundir/output.txt)"; exit 1; }
for f in replay_out.bin replay_grid.bin; do
  [ -s "$TOP/rundir/$f" ] || { echo "REPLAY FAIL: no $f"; exit 1; }
done
grep "REPLAY MOM harness" "$TOP/rundir/output.txt"
echo "REPLAY OK $(cd "$TOP/rundir" && pwd -P)"
