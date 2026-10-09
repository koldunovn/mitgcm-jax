#!/bin/bash
#SBATCH --job-name=mjx_example_gpu
#SBATCH --partition=gpu
#SBATCH --account=ab0995_gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Example: mitjax on GPUs (docs/parallel.md, "Running on GPUs"). The #SBATCH lines are for DKRZ Levante (one node,
# 4 NVIDIA A100 80 GB); on another cluster change partition, account and constraint. The python environment needs
# jax's CUDA wheels: pip install "jax[cuda12]==0.10.1" (docs/install.md).
#   sbatch scripts/example_gpu.sbatch [repository]      (default: the checkout holding this script, or $MJX_TREE)
# Experiment: verification/tutorial_global_oce_optim, variant input_ad (90 x 40 x 15 in 4 tiles of 45 x 20, 10 steps;
# cost and control xx_qnet of data.ctrl), through the CLI (python -m mitjax, docs/running.md):
#   1. the forward run on 1 GPU and on 4 GPUs (one tile per GPU); output.txt compared
#   2. the gradient dfc/dxx_qnet on 1 GPU and on 4 GPUs; the adxx files compared
#   3. the gradient check at data.grdchk's points on 1 GPU, compared with TAF's results/output_adm.txt (testreport)
# Output in a new directory $MJX_RUNS/example_gpu/<job id>; nothing is deleted.
set -u

# repository: argument 1, else $MJX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy:
# scontrol knows the original). Paths: mitjax/paths.py (MJX_UPSTREAM, MJX_RUNS, ...); python: $MJX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${1:-${MJX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (pass it or set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}    # not needed after `pip install -e .`, harmless then
EXP=$MJX_UPSTREAM/verification/tutorial_global_oce_optim
OUT=$MJX_RUNS/example_gpu/${SLURM_JOB_ID:?run this through sbatch}
if [ -e "$OUT" ]; then echo "FAIL: $OUT already exists"; exit 1; fi
mkdir -p "$OUT"
{
  echo "job      $SLURM_JOB_ID on $(hostname), $(date -Is)"
  echo "repo     $REPO, commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)," \
       "$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) uncommitted paths"
  echo "python   $PY"
  nvidia-smi -L
  "$PY" -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
} 2>&1 | tee "$OUT/provenance.txt"
"$PY" -c "import jax, sys; sys.exit(jax.default_backend() != 'gpu')" ||
  { echo "FAIL: JAX sees no GPU (pip install \"jax[cuda12]==0.10.1\" in this environment)"; exit 1; }

FAILED=0
step() {     # step NAME COMMAND...: run, time and log one command; a failure is counted, the next step still runs
  local name=$1; shift
  local t0=$(date +%s)
  echo "== $name: $*"
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "== $name: exit $rc, $(( $(date +%s) - t0 )) s" | tee -a "$OUT/summary.txt"
  [ "$rc" -eq 0 ] || FAILED=$(( FAILED + 1 ))
}

# 1. forward on 1 and on 4 GPUs
step run-1gpu "$PY" -m mitjax run "$EXP" --variant input_ad --out "$OUT/run-1gpu" --devices 1
step run-4gpu "$PY" -m mitjax run "$EXP" --variant input_ad --out "$OUT/run-4gpu" --devices 4
if cmp -s "$OUT/run-1gpu/rundir/output.txt" "$OUT/run-4gpu/rundir/output.txt"; then
  echo "forward: output.txt identical on 1 and 4 GPUs" | tee -a "$OUT/summary.txt"
else
  echo "forward: output.txt differs between 1 and 4 GPUs" | tee -a "$OUT/summary.txt"
fi

# 2. gradient on 1 and on 4 GPUs (a GPU's reverse sweep need not be bitwise reproducible: scatter-add atomics)
step gradient-1gpu "$PY" -m mitjax gradient "$EXP" --variant input_ad --out "$OUT/gradient-1gpu" --devices 1
step gradient-4gpu "$PY" -m mitjax gradient "$EXP" --variant input_ad --out "$OUT/gradient-4gpu" --devices 4
"$PY" - "$OUT/gradient-1gpu/rundir" "$OUT/gradient-4gpu/rundir" <<'EOF' 2>&1 | tee -a "$OUT/summary.txt"
import sys
from pathlib import Path

import numpy as np

from mitjax.io.mds import read_meta

one, four = (Path(d) for d in sys.argv[1:3])
for f in sorted(one.glob("adxx_*.data")):
    dtype = ">f8" if read_meta(f.with_suffix(".meta"))["filePrec"] == 64 else ">f4"
    g1, g4 = np.fromfile(f, dtype), np.fromfile(four / f.name, dtype)
    rel = np.max(np.abs(g4 - g1)) / np.max(np.abs(g1))
    print(f"gradient {f.name}: max|g_4 - g_1| / max|g_1| = {rel:.1e} "
          f"({np.count_nonzero(g4 != g1)} of {g1.size} values differ)")
EOF

# 3. gradient check on 1 GPU against TAF's output_adm.txt
step grdchk-1gpu "$PY" -m mitjax grdchk "$EXP" --variant input_ad --out "$OUT/grdchk-1gpu"
step compare-taf "$PY" -m mitjax compare "$OUT/grdchk-1gpu/rundir/output.txt" "$EXP" --variant input_ad

echo; echo "summary ($OUT/summary.txt):"; cat "$OUT/summary.txt"
[ "$FAILED" -eq 0 ] || { echo "FAIL: $FAILED step(s) failed"; exit 1; }
