#!/bin/bash
#SBATCH --job-name=mjx_run
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#
# Run the Fortran oracle (plan Task 3a; reference/run.sh), one serial process per run, runs one after another:
#   sbatch -J mjx_run_<tag> reference/jobs/run.sbatch RUN_TOP [RUN_TOP ...]
#       RUN_TOP: directories made by reference/make_rundir.py
#   sbatch -J mjx_run_<tag> reference/jobs/run.sbatch --make MJX_COMMIT EXPERIMENT INPUT_DIR BIN_NAME [...]
#       (groups of three after --make) make each run directory in the job, with run id job<SLURM_JOB_ID>, from the
#       frozen scripts of MJX_COMMIT, then run it; for chaining after a build job
#       (`--dependency=afterany:<build job>`: afterany, so a failed build ends this job quickly instead of
#       leaving it pending forever; a missing binary is a failure here).
# Frozen scripts: `git archive MJX_COMMIT reference mitjax/paths.py` into $MJX_RUNS/snapshots/<job name>-<job id>.
# Arguments after the script name only (sbatch --export splits at commas). Verdict lines "RUN OK <dir>" /
# "RUN EXPECTED-STOP ..." (declared in reference/run_verdict.py, not a failure) / "RUN FAIL ..."; exit 1 if any run failed. Repository: $MJX_TREE, else the checkout holding this script.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO"

TOPS=()
if [ "${1:-}" = "--make" ]; then
  shift
  COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
  shift
  SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_run}-${SLURM_JOB_ID:?run this through sbatch}
  [ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
  mkdir -p "$SNAP"
  git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$SNAP"
  echo "$COMMIT" > "$SNAP/COMMIT"
  [ $(( $# % 3 )) -eq 0 ] && [ $# -gt 0 ] || { echo "FAIL: --make takes groups of EXPERIMENT INPUT_DIR BIN_NAME"; exit 2; }
  while [ $# -gt 0 ]; do
    EXP=$1; INP=$2; BIN=$3; shift 3
    OUT=$("$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$BIN" --run-id "job$SLURM_JOB_ID" 2>&1)
    echo "$OUT"
    TOP=$MJX_REFERENCE_RUNS/$EXP/$INP/job$SLURM_JOB_ID
    if echo "$OUT" | grep -q '^RUNDIR '; then TOPS+=("$TOP"); else echo "RUN FAIL $EXP/$INP: no run directory"; FAILED=1; fi
  done
  RUNSH=$SNAP/reference/run.sh
else
  TOPS=("$@")
  RUNSH=$REPO/reference/run.sh
fi

FAILED=${FAILED:-0}
for TOP in "${TOPS[@]}"; do
  bash "$RUNSH" "$TOP"
  RC=$?    # run.sh: 0 RUN OK, 3 RUN EXPECTED-STOP (declared in reference/run_verdict.py: not a failure), else FAIL
  [ "$RC" -eq 0 ] || [ "$RC" -eq 3 ] || FAILED=1
done
[ "$FAILED" -eq 0 ] || { echo "RUNS FAILED"; exit 1; }
echo "ALL RUNS OK"
