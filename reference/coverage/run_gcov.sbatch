#!/bin/bash
#SBATCH --job-name=mjx_cov_run
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Coverage runs of MITgcm verification variants with the gcov binaries (plan Task 5), then the coverage analysis:
#   sbatch -J mjx_cov_run reference/coverage/run_gcov.sbatch MJX_COMMIT BIN_COMMIT EXP:INPUT_DIR [EXP:INPUT_DIR ...]
# MJX_COMMIT: the mitjax commit whose scripts and tools run (frozen: `git archive MJX_COMMIT reference tools mitjax`
# into $MJX_RUNS/snapshots/<job name>-<job id>); BIN_COMMIT: the commit of the gcov binaries
# (<EXP>-<code>-<upstream7>-<BIN_COMMIT7>-gcov, reference/coverage/build_gcov.sbatch; their COVERAGE.txt must say
# COVERAGE OK). Arguments after the script name only (sbatch --export splits values at commas). For chaining after a
# build job use --dependency=afterany (a missing binary fails here at once).
#
# Per variant (experiments in parallel, the variants of one experiment one after another):
#   1. reference/make_rundir.py EXP INPUT_DIR --binary <gcov bin> --run-id job<id> --runs $MJX_REFERENCE/coverage
#      -> RUN_TOP = $MJX_REFERENCE/coverage/<EXP>/<INPUT_DIR>/job<id> (testreport's run directory, lane A's script);
#   2. GCOV_PREFIX=RUN_TOP/gcda reference/run.sh RUN_TOP (lane A's script; counts land under RUN_TOP/gcda, never in
#      the build directory); a run that does not end normally (the forward-only optim build stops in GRDCHK_MAIN)
#      is reported and still analysed: gfortran's STOP exits through exit(), which writes the counts;
#   3. tools/coverage.py collect RUN_TOP --build <build dir> --out RUN_TOP/gcov-<MJX_COMMIT7> (gcov 11.2.0, the
#      compiler's own).
# Per experiment: tools/coverage.py report -> $MJX_REFERENCE/coverage/<EXP>/coverage-job<id>.{md,json.gz}
# (copied to docs/coverage/<EXP>.md by hand after review). Verdict lines: "RUN OK|RUN FAIL", "COLLECT OK",
# "REPORT OK"/"REPORT FAIL"; exit 1 if a collect or report failed. Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"

COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
BINC=$(git -C "$REPO" rev-parse --verify "${2:?binary commit}^{commit}") || { echo "FAIL: no commit $2"; exit 1; }
shift 2
[ $# -ge 1 ] || { echo "FAIL: no EXP:INPUT_DIR given"; exit 2; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "FAIL: no branch pinned"; exit 1; }
ID=job${SLURM_JOB_ID:?run this through sbatch}
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_cov_run}-$SLURM_JOB_ID
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference tools mitjax | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
COV=$MJX_REFERENCE/coverage
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) repo $REPO mitjax $COMMIT binaries $BINC upstream $UP snapshot $SNAP"

set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module load gcc/11.2.0-gcc-11.2.0                     # gcov of the compiler that built the binaries
set -u
export JAX_PLATFORMS=cpu PYTHONPYCACHEPREFIX=$MJX_RUNS/pyc/cov-$SLURM_JOB_ID

declare -A INPUTS
ORDER=()
# grouped by experiment AND code dir (lane A session 11): one report per build (a code_ad variant's gcov lines are
# mapped with the code_ad build's preprocessed sources); the code_ad group's files get the suffix -code_ad
for spec in "$@"; do
  IFS=: read -r EXP INP <<< "$spec"
  case $INP in input_ad*) CODE=code_ad ;; input_min*) CODE=code_min ;; *) CODE=code ;; esac
  KEY=$EXP@$CODE
  [ -n "${INPUTS[$KEY]+x}" ] || ORDER+=("$KEY")
  INPUTS[$KEY]="${INPUTS[$KEY]-} $INP"
done

one_experiment() {
  local EXP=${1%@*} SFX=""; [ "${1#*@}" = code ] || SFX=-${1#*@}; shift
  local VARGS=() BUILD="" FAIL=0 INP CODE BIN TOP OUTD
  for INP in "$@"; do
    case $INP in input_ad*) CODE=code_ad ;; input_min*) CODE=code_min ;; *) CODE=code ;; esac
    BIN=$EXP-$CODE-${UP:0:7}-${BINC:0:7}-gcov
    if ! grep -q '^COVERAGE OK' "$MJX_REFERENCE/bin/$BIN/COVERAGE.txt" 2>/dev/null; then
      echo "RUN FAIL $EXP/$INP: no gcov binary $BIN with COVERAGE OK"; FAIL=1; continue
    fi
    BUILD=$MJX_REFERENCE/build/$BIN
    "$PY" "$SNAP/reference/make_rundir.py" "$EXP" "$INP" --binary "$BIN" --run-id "$ID" --runs "$COV" ||
      { echo "RUN FAIL $EXP/$INP: make_rundir"; FAIL=1; continue; }
    TOP=$COV/$EXP/$INP/$ID
    GCOV_PREFIX=$TOP/gcda bash "$SNAP/reference/run.sh" "$TOP"
    OUTD=$TOP/gcov-${COMMIT:0:7}
    if "$PY" "$SNAP/tools/coverage.py" collect "$TOP" --build "$BUILD" --out "$OUTD"; then
      VARGS+=(--variant "$INP" "$TOP" "$OUTD")
    else
      echo "COLLECT FAIL $EXP/$INP"; FAIL=1
    fi
  done
  if [ ${#VARGS[@]} -gt 0 ]; then
    if "$PY" "$SNAP/tools/coverage.py" report "$EXP" "${VARGS[@]}" --build "$BUILD" \
         --md "$COV/$EXP/coverage-$ID$SFX.md" --json "$COV/$EXP/coverage-$ID$SFX.json.gz" \
         --commit "${COMMIT:0:7}"; then :
    else echo "REPORT FAIL $EXP"; FAIL=1; fi
  fi
  return $FAIL
}

mkdir -p "$COV"
declare -A PIDS
for KEY in "${ORDER[@]}"; do
  # shellcheck disable=SC2086
  one_experiment "$KEY" ${INPUTS[$KEY]} > "$COV/log-${KEY/@/-}-$ID.txt" 2>&1 &
  PIDS[$KEY]=$!
done
FAILED=0
for KEY in "${ORDER[@]}"; do
  wait "${PIDS[$KEY]}" || FAILED=1
  echo "=== $KEY (log $COV/log-${KEY/@/-}-$ID.txt)"
  grep -E "^(RUN OK|RUN FAIL|COLLECT|REPORT|Traceback|[A-Za-z]*Error)" "$COV/log-${KEY/@/-}-$ID.txt"
done
[ "$FAILED" -eq 0 ] || { echo "COVERAGE JOB FAILED"; exit 1; }
echo "ALL COVERAGE REPORTS OK"
