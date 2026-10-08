#!/bin/bash
#SBATCH --job-name=mjx_cov_report
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#
# Re-run the coverage report (tools/coverage.py report) on runs and gcov collections that already exist (plan Task 5):
#   sbatch -J mjx_cov_report reference/coverage/report_gcov.sbatch MJX_COMMIT RUN_ID COLLECT_NAME EXP:INPUT_DIR [...]
# RUN_ID: the run id of reference/coverage/run_gcov.sbatch (job<id>); COLLECT_NAME: the collect directory inside each
# run (gcov-<commit7>). Runs: $MJX_REFERENCE/coverage/<EXP>/<INPUT_DIR>/<RUN_ID>; build: from the run's MANIFEST.json
# binary. Output: $MJX_REFERENCE/coverage/<EXP>/coverage-<RUN_ID>-r<this job id>.{md,json.gz} (new names; nothing
# is overwritten). Frozen scripts: git archive MJX_COMMIT into $MJX_RUNS/snapshots/<job name>-<job id>. Experiments
# in parallel. Verdict lines "REPORT OK"/"REPORT FAIL"; exit 1 if any failed. Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
RUN_ID=${2:?run id}; CNAME=${3:?collect dir name}; shift 3
[ $# -ge 1 ] || { echo "FAIL: no EXP:INPUT_DIR given"; exit 2; }
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_cov_report}-${SLURM_JOB_ID:?run this through sbatch}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference tools mitjax | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
COV=$MJX_REFERENCE/coverage
export JAX_PLATFORMS=cpu PYTHONPYCACHEPREFIX=$MJX_RUNS/pyc/cov-$SLURM_JOB_ID
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) mitjax $COMMIT runs $RUN_ID collect $CNAME"

declare -A VARGS BUILDS
ORDER=()
for spec in "$@"; do
  IFS=: read -r EXP INP <<< "$spec"
  TOP=$COV/$EXP/$INP/$RUN_ID
  [ -f "$TOP/$CNAME/COLLECTED" ] || { echo "REPORT FAIL $EXP/$INP: no $TOP/$CNAME/COLLECTED"; continue; }
  BIN=$("$PY" -c 'import json,os,sys; print(os.path.basename(os.path.dirname(json.load(open(sys.argv[1]))["binary"]["path"])))' "$TOP/MANIFEST.json" 2>/dev/null) ||
    BIN=$(basename "$(dirname "$(readlink -f "$TOP/rundir/mitgcmuv")")")
  # one report per experiment AND build (lane A session 11: the gcov lines of a code_ad variant must be mapped with
  # the code_ad build's preprocessed sources; before, the last variant's build served all, which broke
  # 1D_ocean_ice_column, whose code is SEAICE_BGRID_DYNAMICS and code_ad is not); a code_ad report gets the suffix
  # -code_ad (likewise any code dir other than `code`)
  CODE=$(echo "$BIN" | sed -n 's/.*-\(code[a-z_]*\)-[0-9a-f]\{7\}-[0-9a-f]\{7\}-gcov$/\1/p')
  KEY=$EXP@${CODE:-code}
  [ -n "${VARGS[$KEY]+x}" ] || ORDER+=("$KEY")
  if [ -n "${BUILDS[$KEY]+x}" ] && [ "${BUILDS[$KEY]}" != "$MJX_REFERENCE/build/$BIN" ]; then
    echo "REPORT FAIL $EXP/$INP: two builds for $KEY (${BUILDS[$KEY]}, $BIN)"; continue
  fi
  VARGS[$KEY]="${VARGS[$KEY]-} --variant $INP $TOP $TOP/$CNAME"
  BUILDS[$KEY]=$MJX_REFERENCE/build/$BIN
done
declare -A PIDS
for KEY in "${ORDER[@]}"; do
  EXP=${KEY%@*}; CODE=${KEY#*@}; SFX=""; [ "$CODE" = code ] || SFX=-$CODE
  # shellcheck disable=SC2086
  "$PY" "$SNAP/tools/coverage.py" report "$EXP" ${VARGS[$KEY]} --build "${BUILDS[$KEY]}" \
    --md "$COV/$EXP/coverage-$RUN_ID-r$SLURM_JOB_ID$SFX.md" \
    --json "$COV/$EXP/coverage-$RUN_ID-r$SLURM_JOB_ID$SFX.json.gz" \
    --commit "${COMMIT:0:7}" > "$COV/log-report-$EXP-$RUN_ID-r$SLURM_JOB_ID$SFX.txt" 2>&1 &
  PIDS[$KEY]=$!
done
FAILED=0
for KEY in "${ORDER[@]}"; do
  EXP=${KEY%@*}; CODE=${KEY#*@}; SFX=""; [ "$CODE" = code ] || SFX=-$CODE
  if wait "${PIDS[$KEY]}"; then :; else echo "REPORT FAIL $KEY"; FAILED=1; fi
  grep -E "^(REPORT|Traceback|[A-Za-z]*Error)" "$COV/log-report-$EXP-$RUN_ID-r$SLURM_JOB_ID$SFX.txt"
done
[ "$FAILED" -eq 0 ] && [ ${#ORDER[@]} -gt 0 ] || { echo "REPORTS FAILED"; exit 1; }
echo "ALL REPORTS OK"
