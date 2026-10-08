#!/bin/bash
#SBATCH --job-name=mjx_jdrun
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Invisibility runs of the jaxdump shim (plan Task 4, lesson [E§2]); also the first half of Task 3b's oracle runs.
#   sbatch -J mjx_jdrun reference/jobs/jaxdump_runs.sbatch MJX_COMMIT EXP INPUT_DIR PLAIN_BIN JD_BIN [...]
# (groups of four after MJX_COMMIT; for chaining after the build use --dependency=afterany:<build job>: a missing
# binary fails here.) For each variant three run directories are made from the frozen scripts of MJX_COMMIT
# (reference/make_rundir.py, run ids job<ID>-plain, job<ID>-jdoff, job<ID>-jdon under
# $MJX_REFERENCE_RUNS/<EXP>/<INPUT_DIR>/) and run with reference/run.sh, one after another:
#   plain   PLAIN_BIN, JAXDUMP_DIR unset
#   jdoff   JD_BIN,    JAXDUMP_DIR unset
#   jdon    JD_BIN,    JAXDUMP_DIR=<run top>/dumps (default steps nIter0, nIter0+1, nIter0+2)
# then reference/jaxdump/invisibility.py writes $MJX_REFERENCE_RUNS/<EXP>/<INPUT_DIR>/invisibility-job<ID>.txt.
# Each MANIFEST.json records the binary's sha256. Verdict lines: "RUN OK/EXPECTED-STOP/FAIL ..." per run
# (reference/run.sh, reference/run_verdict.py) and "INVISIBLE/VISIBLE ..." per variant; exit 1 if any comparison
# failed or any run FAILED. The forward-only build of tutorial_global_oce_optim/code_ad stops in GRDCHK_MAIN after the
# reference run (it reads adxx_qnet, which only an adjoint run writes): that stop is declared in run_verdict.py and
# matched by its exact signature (RUN EXPECTED-STOP, not a failure); any other abnormal end fails the job. The
# comparison runs whenever all three runs wrote STDOUT; a run that crashes in only one mode shows up as a STDOUT
# difference. Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
shift
[ $(( $# % 4 )) -eq 0 ] && [ $# -gt 0 ] || { echo "FAIL: groups of EXP INPUT_DIR PLAIN_BIN JD_BIN"; exit 2; }
ID=job${SLURM_JOB_ID:?run this through sbatch}
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_jdrun}-${SLURM_JOB_ID}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
FAILED=0
while [ $# -gt 0 ]; do
  EXP=$1; INP=$2; PBIN=$3; JBIN=$4; shift 4
  BASE=$MJX_REFERENCE_RUNS/$EXP/$INP
  OKV=1; RUNV=""; STOPV=""
  for MODE in plain jdoff jdon; do
    BIN=$JBIN; [ "$MODE" = plain ] && BIN=$PBIN
    OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$BIN" --run-id "$ID-$MODE" 2>&1)
    echo "$OUT" | tail -3
    TOP=$BASE/$ID-$MODE
    echo "$OUT" | grep -q '^RUNDIR ' || { echo "RUN FAIL $EXP/$INP $MODE: no run directory"; OKV=0; FAILED=1; continue; }
    if [ "$MODE" = jdon ]; then
      mkdir "$TOP/dumps"
      env JAXDUMP_DIR="$TOP/dumps" bash "$SNAP/reference/run.sh" "$TOP"
    else
      env -u JAXDUMP_DIR -u JAXDUMP_STEPS bash "$SNAP/reference/run.sh" "$TOP"
    fi
    RC=$?    # run.sh: 0 RUN OK, 3 RUN EXPECTED-STOP (declared and matched: not a failure), else RUN FAIL
    if [ "$RC" -eq 3 ]; then STOPV="$STOPV $MODE"; elif [ "$RC" -ne 0 ]; then RUNV="$RUNV $MODE"; FAILED=1; fi
    [ -s "$TOP/rundir/output.txt" ] || OKV=0
  done
  [ -n "$STOPV" ] && echo "NOTE $EXP/$INP: expected stop (RUN EXPECTED-STOP) for:$STOPV"
  [ -n "$RUNV" ] && echo "NOTE $EXP/$INP: run.sh verdict FAIL for:$RUNV (compared anyway when all wrote STDOUT)"
  if [ "$OKV" -eq 1 ]; then
    "$PY" "$SNAP/reference/jaxdump/invisibility.py" "$BASE/$ID-plain" "$BASE/$ID-jdoff" "$BASE/$ID-jdon" \
      --out "$BASE/invisibility-$ID.txt" || FAILED=1
  else
    echo "VISIBLE $EXP/$INP: not all three runs succeeded (no comparison)"
  fi
done
[ "$FAILED" -eq 0 ] || { echo "JAXDUMP RUNS FAILED"; exit 1; }
echo "ALL INVISIBLE"
