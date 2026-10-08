#!/bin/bash
#SBATCH --job-name=mjx_make_env
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Create env `mitjax` (docs/ENV.md):  sbatch scripts/make_env.sbatch  [repository]
# Python 3.12.13 from conda-forge, then the pyproject dependencies (+ cuda, dev extras) pinned by constraints.txt.
# mitjax itself is NOT installed: tests import the checkout under test (pytest rootdir / PYTHONPATH), never a copy in
# site-packages. pip runs with --no-cache-dir (home quota ~full). Refuses to touch an existing env; never deletes.
set -eu

# repository: argument 1, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows it)
SELF=$0; [ -f "$(dirname "$SELF")/../pyproject.toml" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${1:-$(cd "$(dirname "$SELF")/.." && pwd)}
[ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/constraints.txt" ] || { echo "FAIL: no mitjax repository at $REPO"; exit 1; }

[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (the conda installation: MJX_CONDA_BASE)
E=${MJX_CONDA_BASE:?set MJX_CONDA_BASE to the conda/mamba installation that holds the env (levante.env)}
ENV=${MJX_ENV-$E/envs/mitjax}
if [ -e "$ENV" ]; then echo "FAIL: $ENV exists; not touching it"; exit 1; fi

echo "job $SLURM_JOB_ID on $(hostname), $(date -Is); repo $REPO ($(git -C "$REPO" rev-parse HEAD)); env $ENV"
"$E/bin/mamba" create -y -p "$ENV" python=3.12.13 pip

# Dependencies from pyproject.toml (one source of truth), with the cuda and dev extras.
mapfile -t DEPS < <("$ENV/bin/python" - "$REPO/pyproject.toml" <<'EOF'
import sys, tomllib
p = tomllib.load(open(sys.argv[1], "rb"))["project"]
for d in p["dependencies"] + p["optional-dependencies"]["cuda"] + p["optional-dependencies"]["dev"]:
    print(d)
EOF
)
echo "dependencies: ${DEPS[*]}"
"$ENV/bin/pip" install --no-cache-dir -c "$REPO/constraints.txt" "${DEPS[@]}"

echo "---- pip freeze"
"$ENV/bin/pip" freeze
echo "---- check"
# from a neutral directory: `python -` puts the current directory on sys.path, and sbatch starts in the submit
# directory (the repository), which made the first run (job 27825269) report mitjax as importable.
cd /
JAX_PLATFORMS=cpu "$ENV/bin/python" - "$REPO/constraints.txt" <<'EOF'
import sys, importlib.metadata as md
pins = {}
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if "==" in line:
        n, v = line.split("==")
        pins[n.strip().lower()] = v.strip()
bad = [(n, md.version(n), pins[n]) for n in ("jax", "jaxlib", "numpy", "scipy", "pytest") if md.version(n) != pins[n]]
import jax
jax.config.update("jax_enable_x64", True)
x = jax.numpy.arange(3.0)
print("python", sys.version.split()[0], "jax", jax.__version__, "dtype", x.dtype, "devices", jax.devices())
try:
    import mitjax  # must NOT be importable from site-packages
    bad.append(("mitjax importable", mitjax.__file__, "absent"))
except ImportError:
    pass
print("ENV FAIL", bad) if bad or x.dtype != "float64" else print("ENV OK")
sys.exit(1 if bad else 0)
EOF
