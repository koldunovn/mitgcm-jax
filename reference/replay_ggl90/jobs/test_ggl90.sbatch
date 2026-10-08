#!/bin/bash
#SBATCH --job-name=mjx_test_ggl90
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The GGL90 lane's tests (mitjax/tests/test_ggl90.py, tier1x) on a frozen snapshot of a commit:
#   sbatch reference/replay_ggl90/jobs/test_ggl90.sbatch MJX_COMMIT [pytest args...]
# OUT = $MJX_RUNS/ggl90/test-<commit short>-job<id> (new); src = git archive of the commit; JUnit report + pytest
# exit code checked by scripts/check_pytest_report.py. Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../../.." && pwd)}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
shift
OUT=$MJX_RUNS/ggl90/test-${COMMIT:0:7}-job${SLURM_JOB_ID:-none}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src"
git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src"
cd "$OUT/src"
export JAX_PLATFORMS=cpu PYTHONPATH=$OUT/src PYTHONPYCACHEPREFIX=$OUT/pyc
"$PY" -m pytest -q -p no:cacheprovider --junitxml="$OUT/report.xml" mitjax/tests/test_ggl90.py "$@" -s > "$OUT/pytest.log" 2>&1
RC=$?
cat "$OUT/pytest.log"
"$PY" scripts/check_pytest_report.py "$OUT/report.xml" 2>&1; echo "check_pytest_report rc=$?"
echo "TEST GGL90 rc=$RC out=$OUT"
