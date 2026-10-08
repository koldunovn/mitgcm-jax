#!/bin/bash
#SBATCH --job-name=mjx_replay_ptracers
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build and run the PTRACERS replay harness (M2 sub-lane PTRACERS; reference/replay_ptracers/build.sh,
# code/the_main_loop.F; a copy of the COL harness reference/replay_col, which stays lane COL's):
#   sbatch -J mjx_replay_ptracers_<tag> reference/replay_ptracers/jobs/replay.sbatch MJX_COMMIT EXP:CODE:INPUT_DIR
# i.e. tutorial_tracer_adjsens:code_ad:input_ad (the harness calls the routines of that build). Arguments go after the
# script name, never through --export.
#
# OUT = $MJX_REFERENCE/replay_ptracers/<EXP>-<CODE>-<upstream short>-<MJX_COMMIT short>-job<id> (new; refuses an existing
# one).
#   1. git archive MJX_COMMIT reference mitjax/paths.py -> OUT/scripts (the frozen snapshot that builds and runs)
#   2. OUT/scripts/reference/replay_ptracers/build.sh (genmake2 + make, checks vs the oracle build of the same code at
#      $MJX_REFERENCE/build/<EXP>-<CODE>-<upstream short>-06b4417 (the M2 oracle builds, lane A session 6), frozen
#      OUT/bin)
#   3. run directory with lane A's reference/make_rundir.py (testreport's linkdata + prepare_run of EXP/INPUT_DIR,
#      --binary OUT/bin, --runs OUT/runs, run id job<id>), synthetic inputs (reference/replay_ptracers/replay_io.py
#      write-inputs), then lane A's reference/run.sh: exit 0 (RUN OK) or 3 (RUN EXPECTED-STOP: the forward-only
#      code_ad build of tutorial_tracer_adjsens stops in GRDCHK after THE_MAIN_LOOP, reference/run_verdict.py)
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
IFS=: read -r EXP CODE INP <<< "$SPEC"
[ -n "$EXP" ] && [ -n "$CODE" ] && [ -n "$INP" ] || { echo "REPLAY FAIL: bad spec $SPEC (EXP:CODE:INPUT_DIR)"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}") || { echo "REPLAY FAIL: no commit $COMMIT_ARG"; exit 1; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "REPLAY FAIL: no branch pinned"; exit 1; }
NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
OUT=$MJX_REFERENCE/replay_ptracers/$NAME
ORACLE=$MJX_REFERENCE/build/$EXP-$CODE-${UP:0:7}-06b4417
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP -> $OUT"
[ -e "$OUT" ] && { echo "REPLAY FAIL: $OUT exists"; exit 1; }
[ -d "$ORACLE/bld" ] || { echo "REPLAY FAIL: no oracle build $ORACLE (the preprocessed-source identity check needs it)"; exit 1; }
mkdir -p "$OUT/scripts"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$OUT/scripts"
echo "$COMMIT" > "$OUT/scripts/COMMIT"

bash "$OUT/scripts/reference/replay_ptracers/build.sh" "$OUT" "$EXP" "$CODE" "$COMMIT" "$UP" "$ORACLE" \
  || { echo "REPLAY FAIL build (log $OUT/build.log)"; exit 1; }

RUNID=job${SLURM_JOB_ID:-none}
"$PY" "$OUT/scripts/reference/make_rundir.py" "$EXP" "$INP" --binary "$OUT/bin" --runs "$OUT/runs" --run-id "$RUNID" \
  || { echo "REPLAY FAIL make_rundir"; exit 1; }
TOP=$OUT/runs/$EXP/$INP/$RUNID
"$PY" "$OUT/scripts/reference/replay_ptracers/replay_io.py" write-inputs "$TOP/rundir" --size "$OUT/mods/SIZE.h" \
  || { echo "REPLAY FAIL write-inputs"; exit 1; }
bash "$OUT/scripts/reference/run.sh" "$TOP"
RC=$?
# the harness runs no time step, so gfortran's IEEE_UNDERFLOW_FLAG note of the declared stop (reference/run_verdict.py
# _TA_TR, measured on the oracle's 4 steps) is absent: accept that one line less (job 27833187, the first run)
if [ "$RC" -eq 1 ] && [ "$EXP:$CODE" = "tutorial_tracer_adjsens:code_ad" ] && \
   [ "$(cat "$TOP/rundir/run.tr_log")" = "$(printf '%s\n' 'STOP ABNORMAL END: S/R MDS_READ_FIELD' \
     ' MDS_READ_FIELD: filename: adxx_ptr1.0000000000 , adxx_ptr1.0000000000.001.001.data' \
     ' MDS_READ_FIELD: Files DO not exist')" ]; then
  echo "declared GRDCHK stop without the underflow note (no time step in the harness)"; RC=3
fi
[ "$RC" -eq 0 ] || [ "$RC" -eq 3 ] || { echo "REPLAY FAIL run rc $RC ($TOP/rundir/output.txt)"; exit 1; }
for f in pt_out.bin pt_grid.bin; do
  [ -s "$TOP/rundir/$f" ] || { echo "REPLAY FAIL: no $f"; exit 1; }
done
grep "REPLAY PTRACERS harness" "$TOP/rundir/output.txt" || { echo "REPLAY FAIL: no harness line in STDOUT"; exit 1; }
echo "REPLAY OK $(cd "$TOP/rundir" && pwd -P)"
