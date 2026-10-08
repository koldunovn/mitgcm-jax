#!/bin/bash
#SBATCH --job-name=mjx_pip_install
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# `pip install -e .` works (docs plan 20261006 Task 5): a NEW venv on top of env mitjax (--system-site-packages: the
# pinned dependencies, no download; compute nodes have no internet), mitjax installed editable from a frozen snapshot
# of HEAD (--no-deps --no-build-isolation: setuptools of the env), then, from a directory outside the repository and
# with PYTHONPATH unset: mitjax imports from the snapshot, the moved modules (mitjax.make_rundir,
# mitjax.testreport_jax) run as `python -m`, and `python -m mitjax run` of tutorial_barotropic_gyre/input followed by
# `python -m mitjax.testreport_jax` against the experiment's results/ passes.
#   sbatch scripts/pip_install_check.sbatch <repository>
# Output: $MJX_RUNS/port/s2/pip-<commit7>-<job>/ (venv, snapshot, logs, verdict.txt). Nothing is deleted.
set -u
REPO=${1:?repository}
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify HEAD) || exit 1
OUT=$MJX_RUNS/port/s2/pip-${COMMIT:0:7}-${SLURM_JOB_ID:?sbatch}
[ -e "$OUT" ] && { echo "FAIL: $OUT exists"; exit 1; }
mkdir -p "$OUT/src" "$OUT/elsewhere" && git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$OUT/src" || exit 1
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) commit $COMMIT" | tee "$OUT/provenance.txt"
fail() { echo "PIP INSTALL CHECK FAIL: $*" | tee "$OUT/verdict.txt"; exit 1; }
unset PYTHONPATH
export JAX_PLATFORMS=cpu PYTHONPYCACHEPREFIX=$OUT/pycache
"$PY" -m venv --system-site-packages "$OUT/venv" || fail "venv"
V=$OUT/venv/bin/python
"$V" -m pip install --no-cache-dir --no-deps --no-build-isolation -e "$OUT/src" > "$OUT/pip.log" 2>&1 ||
  fail "pip install -e (pip.log)"
cd "$OUT/elsewhere" || exit 1
"$V" - "$OUT/src" > "$OUT/imports.log" 2>&1 <<'PYEOF' || fail "imports (imports.log)"
import sys
from pathlib import Path
import mitjax, mitjax.make_rundir, mitjax.testreport_jax, mitjax.config.params, mitjax.drivers.run
src = Path(sys.argv[1]).resolve()
for m in (mitjax, mitjax.make_rundir, mitjax.testreport_jax):
    assert Path(m.__file__).resolve().is_relative_to(src), m.__file__
assert not any(k.startswith(("reference", "tools")) for k in sys.modules), sorted(k for k in sys.modules
                                                                                 if k.startswith(("reference", "tools")))
print("imports OK from", src)
PYEOF
"$V" -m mitjax.make_rundir --help > "$OUT/make_rundir_help.log" 2>&1 || fail "python -m mitjax.make_rundir --help"
T0=$(date +%s)
"$V" -m mitjax run "$MJX_UPSTREAM/verification/tutorial_barotropic_gyre" --variant input --out "$OUT/run" \
  > "$OUT/run.log" 2>&1 || fail "python -m mitjax run (run.log)"
echo "run wall $(( $(date +%s) - T0 )) s" | tee -a "$OUT/provenance.txt"
"$V" -m mitjax.testreport_jax "$OUT/run/rundir/output.txt" --exp tutorial_barotropic_gyre --variant input \
  > "$OUT/testreport.log" 2>&1 || fail "testreport_jax (testreport.log)"
tail -1 "$OUT/testreport.log" | tee -a "$OUT/provenance.txt"
echo "PIP INSTALL CHECK PASS" | tee "$OUT/verdict.txt"
