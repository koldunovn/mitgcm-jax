#!/bin/bash
#SBATCH --job-name=mjx_restart
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The Fortran oracle's own restart (lane A session 5): run A and its restart B, both dumps on, then
# reference/restart_compare.py A B.
#   sbatch -J mjx_restart_<tag> reference/jobs/restart_runs.sbatch MJX_COMMIT EXP INPUT_DIR JD_BIN PCHKPT K NB STEPS
#     A: make_rundir.py --set data:PARM03:pChkptFreq=PCHKPT (a permanent pickup every PCHKPT seconds,
#        DO_WRITE_PICKUP: permPickup = DIFFERENT_MULTIPLE(pChkPtFreq,myTime,deltaTClock), do_write_pickup.F:60-61),
#        the experiment's own nIter0 and nTimeSteps; run id job<ID>-restartA.
#     B: the same overlay plus nIter0=K and nTimeSteps=NB (a restarted run's `data`), and every regular file
#        pickup.<K as I10.10>.* that A wrote copied into its run directory (make_rundir.py --copy); run id
#        job<ID>-restartB. B is made only after A ran (its pickup must exist).
#   Both: JD_BIN (a jaxdump build under $MJX_REFERENCE/bin), JAXDUMP_DIR=<top>/dumps, JAXDUMP_STEPS=STEPS (':'-list).
# Runs with overlays and copied pickups: GATE FIXTURES, never yardsticks or plain oracle runs.
# Output: $MJX_REFERENCE_RUNS/<EXP>/<INPUT_DIR>/restart-job<ID>.{txt,json} (reference/restart_compare.py).
# Frozen scripts: `git archive MJX_COMMIT reference mitjax/paths.py mitjax/io/dump.py` into
# $MJX_RUNS/snapshots/<job name>-<job id>.
# Verdict: "RESTART PAIR OK" (both runs RUN OK, comparison written) or "RESTART PAIR FAILED"; exit 1 on failure.
# Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO"
[ $# -eq 8 ] || { echo "FAIL: MJX_COMMIT EXP INPUT_DIR JD_BIN PCHKPT K NB STEPS"; exit 2; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$1^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
EXP=$2; INP=$3; JBIN=$4; PCHKPT=$5; K=$6; NB=$7; STEPS=$8
ID=job${SLURM_JOB_ID:?run this through sbatch}
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_restart}-${SLURM_JOB_ID}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) mitjax/io/dump.py | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
BASE=$MJX_REFERENCE_RUNS/$EXP/$INP
SET=(--set "data:PARM03:pChkptFreq=$PCHKPT")

run_one() {   # TOP: make_rundir output already checked; run with dumps on
  mkdir "$1/dumps"
  env JAXDUMP_DIR="$1/dumps" JAXDUMP_STEPS="$STEPS" bash "$SNAP/reference/run.sh" "$1"
}

OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$JBIN" --run-id "$ID-restartA" "${SET[@]}" 2>&1)
echo "$OUT" | tail -3
echo "$OUT" | grep -q '^RUNDIR ' || { echo "RESTART PAIR FAILED: no run directory A"; exit 1; }
A=$BASE/$ID-restartA
run_one "$A" || { echo "RESTART PAIR FAILED: run A"; exit 1; }

PK=pickup.$(printf %010d "$K")
COPIES=()
for f in "$A/rundir/$PK".*; do
  [ -f "$f" ] && [ ! -L "$f" ] && COPIES+=(--copy "$f:$(basename "$f")")
done
[ ${#COPIES[@]} -gt 0 ] || { echo "RESTART PAIR FAILED: A wrote no $PK.*"; exit 1; }
OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$JBIN" --run-id "$ID-restartB" "${SET[@]}" \
      --set "data:PARM03:nIter0=$K" --set "data:PARM03:nTimeSteps=$NB" "${COPIES[@]}" 2>&1)
echo "$OUT" | tail -3
echo "$OUT" | grep -q '^RUNDIR ' || { echo "RESTART PAIR FAILED: no run directory B"; exit 1; }
B=$BASE/$ID-restartB
run_one "$B" || { echo "RESTART PAIR FAILED: run B"; exit 1; }

"$PY" "$SNAP/reference/restart_compare.py" "$A" "$B" --steps "$STEPS" --out "$BASE/restart-$ID.txt" \
  --json "$BASE/restart-$ID.json" || { echo "RESTART PAIR FAILED: comparison"; exit 1; }
echo "RESTART PAIR OK"
