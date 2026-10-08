#!/bin/bash
#SBATCH --job-name=mjx_fresh_nb01
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:45:00
#SBATCH --output=fresh_install_nb01-%j.out
#
# Fresh-install check, step 2 of 2 (docs plan 20261006 Task 11): in the environment step 1
# (scripts/fresh_install.sbatch) built in WORK, as the user would, with nothing of this project inherited:
#   a. notebook 01 executed in place (WORK/mitgcm-jax/notebooks) by mitjax.tests.nb_exec, its asserts the check,
#      pinned to 8 cores (a laptop's count); the executed copy and its measures go to WORK/nb01/
#   b. the user's tests: `pytest -m "smoke or tier1"` in WORK/mitgcm-jax without any oracle data (oracle tests skip,
#      naming what is missing); a finding, reported, not part of this job's verdict
#
#   sbatch [-o LOG] --dependency=afterok:<step-1 job> scripts/fresh_install_nb01.sbatch WORK
#   bash scripts/fresh_install_nb01.sbatch --dry-run WORK      # checks and prints the plan, runs nothing
#
# Every inherited MJX_* variable is cleared (and PYTHONPATH, PYTHONHOME, XLA_FLAGS, JAX_*); the user's three are set:
# MJX_UPSTREAM=WORK/MITgcm, MJX_RUNS=WORK/runs, MJX_CACHE=WORK/cache. Exit code 0 only when notebook 01 ran
# through. Nothing is deleted.
set -uo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
WORK=${1:?work directory of scripts/fresh_install.sbatch}

for v in $(compgen -e); do
  case "$v" in
    MJX_*|JAX_*|PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|PYTHONPYCACHEPREFIX|XLA_FLAGS|CONDA_PREFIX|CONDA_DEFAULT_ENV|\
    VIRTUAL_ENV) unset "$v" ;;
  esac
done
left=$(compgen -e | grep -E '^(MJX_|JAX_)' || true)
[ -z "$left" ] || { echo "FAIL: still set: $left"; exit 1; }

if [ "$DRY" = 0 ]; then
  [ -f "$WORK/ENV_OK" ] || { echo "FAIL: $WORK/ENV_OK missing (step 1 did not pass)"; exit 1; }
  [ -e "$WORK/nb01" ] && { echo "FAIL: $WORK/nb01 exists"; exit 1; }
fi
PY=$WORK/env/bin/python
TREE=$WORK/mitgcm-jax
export MJX_UPSTREAM=$WORK/MITgcm MJX_RUNS=$WORK/runs MJX_CACHE=$WORK/cache OMP_NUM_THREADS=8
cat <<EOF
fresh install, step 2: $WORK
  a. taskset -c 0-7 $PY -m mitjax.tests.nb_exec $TREE/notebooks/01_quickstart.ipynb -> $WORK/nb01/
  b. $PY -m pytest -m "smoke or tier1" in $TREE (user environment; no oracle) -> $WORK/user_tests/
  MJX_UPSTREAM=$MJX_UPSTREAM MJX_RUNS=$MJX_RUNS MJX_CACHE=$MJX_CACHE
EOF
if [ "$DRY" = 1 ]; then echo "dry run: nothing done"; exit 0; fi

mkdir -p "$WORK/nb01" "$WORK/user_tests"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) $(date -Is)" | tee "$WORK/nb01/provenance.txt"
cd "$TREE/notebooks" || exit 1
taskset -c 0-7 "$PY" -m mitjax.tests.nb_exec 01_quickstart.ipynb "$WORK/nb01/01_quickstart.executed.ipynb" \
    "$WORK/nb01/measures.json" --timeout 1800 > "$WORK/nb01/nb_exec.log" 2>&1
NB_RC=$?
echo "notebook 01 exit code $NB_RC" | tee -a "$WORK/nb01/provenance.txt"
cat "$WORK/nb01/measures.json" 2>/dev/null | tee -a "$WORK/nb01/provenance.txt"

cd "$TREE" || exit 1
taskset -c 0-7 "$PY" -m pytest -m "smoke or tier1" -q -rfEs -p no:cacheprovider \
    --basetemp="$WORK/user_tests/pytest_tmp" --junitxml="$WORK/user_tests/report.xml" \
    > "$WORK/user_tests/pytest.log" 2>&1
T_RC=$?
echo "user tests (smoke or tier1) pytest exit code $T_RC" | tee -a "$WORK/nb01/provenance.txt"
"$PY" scripts/check_pytest_report.py "$WORK/user_tests/report.xml" > "$WORK/user_tests/verdict.txt"
cat "$WORK/user_tests/verdict.txt"

if [ "$NB_RC" -ne 0 ]; then echo "FRESH INSTALL FAIL: notebook 01 (nb_exec rc $NB_RC) -> $WORK/nb01"; exit 1; fi
echo "FRESH INSTALL PASS: notebook 01 ran in the user's environment -> $WORK/nb01 (user tests: rc $T_RC, see" \
     "$WORK/user_tests/verdict.txt)"
