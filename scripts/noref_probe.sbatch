#!/bin/bash
#SBATCH --job-name=mjx_port_noref
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Portability probe (docs plan 20261006, Task 1) on a frozen snapshot of the repository's HEAD:
#   sbatch scripts/noref_probe.sbatch <repository> probe [probe args...]   scripts/noref_probe.py (forward, gradient,
#                                                                            sharded cases at levels L0-L3)
#   sbatch scripts/noref_probe.sbatch <repository> tier1                   tier 1 (smoke or tier1) with
#                                                                            MJX_REFERENCE(_RUNS) empty, MJX_WORK new
# Output: $MJX_RUNS/port/noref-<mode>-<commit7>-<job>/ (new; nothing deleted). The tier-1 mode measures which tests
# need the oracle: its verdict is a measurement, not a gate.
set -u
REPO=${1:?repository}; MODE=${2:?probe or tier1}; shift 2
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD) || exit 1
OUT=$MJX_RUNS/port/noref-$MODE-${COMMIT:0:7}-${SLURM_JOB_ID:?sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src" || exit 1
git -C "$REPO" status --porcelain > "$OUT/git_status.txt"
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) commit $COMMIT mode $MODE" | tee "$OUT/provenance.txt"
cd "$OUT/src" || exit 1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPYCACHEPREFIX=$OUT/pycache PYTHONPATH=$OUT/src
if [ "$MODE" = probe ]; then
  "$PY" scripts/noref_probe.py --out "$OUT/probe" "$@" > "$OUT/probe.log" 2>&1
  echo "probe exit code $?" | tee -a "$OUT/provenance.txt"
  cat "$OUT/probe/SUMMARY.txt"
elif [ "$MODE" = tier1 ]; then
  mkdir -p "$OUT/empty_reference" "$OUT/empty_reference_runs" "$OUT/work" "$OUT/runs"
  export MJX_WORK=$OUT/work MJX_REFERENCE=$OUT/empty_reference MJX_REFERENCE_RUNS=$OUT/empty_reference_runs \
         MJX_RUNS=$OUT/runs
  "$PY" -m pytest -m "smoke or tier1" -q -rfEs --basetemp="$OUT/pytest_tmp" --junitxml="$OUT/report.xml" \
      > "$OUT/pytest.log" 2>&1
  echo "pytest exit code $?" | tee -a "$OUT/provenance.txt"
  "$PY" scripts/check_pytest_report.py "$OUT/report.xml" --max-tests 100 > "$OUT/verdict.txt"
  echo "report check exit code $?" | tee -a "$OUT/provenance.txt"
  tail -3 "$OUT/verdict.txt"
else
  echo "unknown mode $MODE"; exit 1
fi
