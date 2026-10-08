#!/bin/bash
#SBATCH --job-name=mjx_orun
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Oracle runs of plan Task 3b (debugMode call-order runs, diagnostics/mnc-off runs, second-tiling runs), one serial
# process per run, one after another:
#   sbatch -J mjx_orun_<tag> reference/jobs/oracle_runs.sbatch MJX_COMMIT SPEC [SPEC ...]
#   SPEC = EXP@INPUT_DIR@BIN_NAME@LABEL[@MAKE_RUNDIR_OPTION ...]
# e.g.  tutorial_baroclinic_gyre@input@tutorial_baroclinic_gyre-code-63cdc0b-f48c062@diagoff@--set@data.pkg:PACKAGES:useDiagnostics=.FALSE.
#       global_ocean.90x40x15@input@global_ocean.90x40x15-code-63cdc0b-<commit>-retile@retile@--mpi
# Fields are separated by `@` (the options of reference/make_rundir.py: --mpi, --set FILE:GROUP:KEY=VALUE), never by
# commas (arguments go after the script name; sbatch --export splits at commas). Each run directory is made from the
# frozen scripts of MJX_COMMIT (`git archive` into $MJX_RUNS/snapshots/<job name>-<job id>) with run id
# job<ID>-<LABEL> under $MJX_REFERENCE_RUNS/<EXP>/<INPUT_DIR>/ and run by reference/run.sh. For chaining after a
# build use --dependency=afterany:<build job>: a missing binary is a failure here (make_rundir refuses).
# Verdict lines "RUN OK ..." / "RUN EXPECTED-STOP ..." / "RUN FAIL ..." per run (reference/run.sh; the forward-only
# optim build's grdchk STOP is declared in reference/run_verdict.py and reported as RUN EXPECTED-STOP -- the STDOUT
# is complete up to that point); exit 1 if any run failed. Never deletes anything.
# `@--dumps@STEPS` in a SPEC (not a make_rundir option; lane A session 8): run the (jaxdump) binary with dumps on,
# JAXDUMP_DIR=<run top>/dumps and JAXDUMP_STEPS=STEPS (colon-separated iterations), e.g. the planted-control runs
# EXP@input_ad@<jaxdump bin>@ctrlxx@--set@data.ctrl:CTRL_NML:doInitXX=.FALSE.@...@--copy@...@--dumps@0:1:2:3
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
[ $# -gt 0 ] || { echo "FAIL: no SPEC given"; exit 2; }
ID=job${SLURM_JOB_ID:?run this through sbatch}
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_orun}-${SLURM_JOB_ID}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
FAILED=0
for SPEC in "$@"; do
  IFS=@ read -r -a F <<< "$SPEC"
  [ ${#F[@]} -ge 4 ] || { echo "RUN FAIL $SPEC: expected EXP@INPUT_DIR@BIN_NAME@LABEL[@OPTION ...]"; FAILED=1; continue; }
  EXP=${F[0]}; INP=${F[1]}; BIN=${F[2]}; LABEL=${F[3]}; RAW=("${F[@]:4}"); OPTS=(); STEPS=""; DUMPS=0
  for ((k = 0; k < ${#RAW[@]}; k++)); do
    if [ "${RAW[$k]}" = --dumps ]; then DUMPS=1; k=$((k + 1)); STEPS=${RAW[$k]:-}; else OPTS+=("${RAW[$k]}"); fi
  done
  if [ "$DUMPS" = 1 ] && ! [[ $STEPS =~ ^[0-9]+(:[0-9]+)*$ ]]; then
    echo "RUN FAIL $SPEC: --dumps needs colon-separated iterations"; FAILED=1; continue
  fi
  echo "== $EXP/$INP $LABEL: $BIN ${OPTS[*]}"
  OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$BIN" --run-id "$ID-$LABEL" ${OPTS[@]+"${OPTS[@]}"} 2>&1)
  echo "$OUT" | tail -3
  TOP=$MJX_REFERENCE_RUNS/$EXP/$INP/$ID-$LABEL
  echo "$OUT" | grep -q '^RUNDIR ' || { echo "RUN FAIL $EXP/$INP $LABEL: no run directory"; FAILED=1; continue; }
  if [ -n "$STEPS" ]; then
    mkdir "$TOP/dumps"
    env JAXDUMP_DIR="$TOP/dumps" JAXDUMP_STEPS="$STEPS" bash "$SNAP/reference/run.sh" "$TOP"
  else
    env -u JAXDUMP_DIR -u JAXDUMP_STEPS bash "$SNAP/reference/run.sh" "$TOP"
  fi
  RC=$?    # run.sh: 0 RUN OK, 3 RUN EXPECTED-STOP (declared in reference/run_verdict.py: not a failure), else FAIL
  [ "$RC" -eq 0 ] || [ "$RC" -eq 3 ] || FAILED=1
done
[ "$FAILED" -eq 0 ] || { echo "RUNS FAILED"; exit 1; }
echo "ALL RUNS OK"
