#!/bin/bash
#SBATCH --job-name=mjx_ctrl_oracle
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Oracle runs for the ctrl/cost gates of tutorial_global_oce_optim/input_ad (lane ctrl; mitjax/tests/test_ctrl_cost.py):
#   sbatch mitjax/tests/ctrl_oracle.sbatch MJX_COMMIT JD_BIN SOURCE_TOP [MODES]
#     JD_BIN      jaxdump binary under $MJX_REFERENCE/bin (registered, e.g. ...-5f16129-jaxdump)
#     SOURCE_TOP  an existing forward-only run of the same build (its xx_qnet.0000000000.*.meta are copied)
#     MODES       which runs (default "ctrlzero ctrlxx")
# Two runs, dumps on at every step (JAXDUMP_STEPS=0:...:9; the default is steps 0-2 only, which holds no
# E01_cost_final), run ids under $MJX_REFERENCE_RUNS/tutorial_global_oce_optim/input_ad/:
#   job<ID>-ctrlzero  the variant as testreport runs it (zero first-guess control)
#   job<ID>-ctrlxx    planted nonzero xx_qnet (mitjax/tests/ctrl_xx_files.py, seed 20261001, +-50 W/m^2) with the
#                     overlays data.ctrl:CTRL_NML:doInitXX=.FALSE. and doMainUnpack=.FALSE. (not testreport's run:
#                     never a yardstick). doMainUnpack: with doInitXX=.FALSE. THE_MODEL_MAIN would otherwise call
#                     CTRL_UNPACK, which reads the packed vector ecco_ctrl_MIT_CE_000.opt0000 (the_model_main.F:637-641;
#                     job 27829092 stopped there)
# Both end in the declared grdchk stop (RUN EXPECTED-STOP: no adxx file, so no grdchk forward runs add dumps).
# Frozen scripts: git archive MJX_COMMIT (reference, mitjax) into $MJX_RUNS/snapshots/<job>-<id>. Deletes nothing.
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
JBIN=${2:?jaxdump binary name}
SRC=${3:?source run top}
MODES=${4:-ctrlzero ctrlxx}
ID=job${SLURM_JOB_ID:?run this through sbatch}
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_ctrl_oracle}-${SLURM_JOB_ID}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
EXP=tutorial_global_oce_optim; INP=input_ad
BASE=$MJX_REFERENCE_RUNS/$EXP/$INP
STEPS=0:1:2:3:4:5:6:7:8:9
FAILED=0

XX=$MJX_RUNS/ctrl/xx_planted-$ID
"$PY" "$SNAP/mitjax/tests/ctrl_xx_files.py" "$SRC" "$XX" || { echo "FAIL: xx files"; exit 1; }
COPIES=()
for f in "$XX"/xx_qnet.0000000000.*; do COPIES+=(--copy "$f:$(basename "$f")"); done

run_one() {   # MODE, make_rundir extra args...
  local MODE=$1; shift
  local TOP=$BASE/$ID-$MODE
  OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$JBIN" --run-id "$ID-$MODE" "$@" 2>&1)
  echo "$OUT" | tail -3
  echo "$OUT" | grep -q '^RUNDIR ' || { echo "RUN FAIL $MODE: no run directory"; FAILED=1; return; }
  mkdir "$TOP/dumps"
  env JAXDUMP_DIR="$TOP/dumps" JAXDUMP_STEPS="$STEPS" bash "$SNAP/reference/run.sh" "$TOP"
  RC=$?
  [ "$RC" -eq 0 ] || [ "$RC" -eq 3 ] || FAILED=1
  du -sh "$TOP/dumps"
}
for M in $MODES; do
  case $M in
    ctrlzero) run_one ctrlzero ;;
    ctrlxx) run_one ctrlxx --set "data.ctrl:CTRL_NML:doInitXX=.FALSE." --set "data.ctrl:CTRL_NML:doMainUnpack=.FALSE." \
              "${COPIES[@]}" ;;
    *) echo "FAIL: unknown mode $M"; FAILED=1 ;;
  esac
done
[ "$FAILED" -eq 0 ] || { echo "CTRL ORACLE RUNS FAILED"; exit 1; }
echo "CTRL ORACLE RUNS OK"
