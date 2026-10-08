#!/bin/bash
#SBATCH --job-name=mjx_build
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# Build several MITgcm verification codes at once (plan Task 3a), each from a frozen snapshot:
#   sbatch -J mjx_build_<tag> reference/jobs/build.sbatch MJX_COMMIT EXP:CODE[:TAG] [EXP:CODE[:TAG] ...]
# e.g.  sbatch -J mjx_build_m1 reference/jobs/build.sbatch $(git rev-parse --short HEAD) \
#         tutorial_barotropic_gyre:code advect_xy:code tutorial_global_oce_optim:code_ad
# Arguments go after the script name (never through `sbatch --export`, which splits values at commas).
#
# TAG `jaxdump` builds the instrumented oracle of plan Task 4 (reference/build.sh, step 1b); TAG `retile` the second
# tiling of plan Task 3b (reference/retile/<EXP>-<CODE>/SIZE.h, step 1c). CODE is any code* directory of the
# experiment at the pinned commit (code, code_ad forward only, code_min; lane A session 7).
# For each EXP:CODE the build is named <EXP>-<CODE>-<upstream short hash>-<MJX_COMMIT short>[-TAG], with the
# upstream commit = branch `pinned` of $MJX_UPSTREAM. The job creates $MJX_REFERENCE/build/<name>/ (refuses an
# existing one), unpacks `git archive MJX_COMMIT reference mitjax/paths.py` of this repository into
# <name>/scripts, and runs <name>/scripts/reference/build.sh there; all builds run in parallel (make -j 16 each),
# their transcripts in <name>/build.log. Frozen binaries: $MJX_REFERENCE/bin/<name>/ (reference/build.sh).
# Verdict: one line per build, "BUILD OK <name>" or "BUILD FAIL <name> rc=<n>"; exit 1 if any failed.
# Repository: $MJX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"

COMMIT_ARG=${1:?mitjax commit}; shift
[ $# -ge 1 ] || { echo "FAIL: no EXP:CODE given"; exit 2; }
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}") || { echo "FAIL: no commit $COMMIT_ARG"; exit 1; }
UP=$(git -C "$MJX_UPSTREAM" rev-parse --verify "pinned^{commit}") || { echo "FAIL: no branch pinned"; exit 1; }
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP"

declare -A PIDS
for spec in "$@"; do
  IFS=: read -r EXP CODE TAG <<< "$spec"
  # any code* directory of the experiment at the pinned commit (code, code_ad, code_min, ...); build.sh knows the
  # genmake2 options a code directory needs beyond testreport's (code_min: -standarddirs eesupp)
  case $CODE in code*) ;; *) echo "BUILD FAIL $spec: code dir must start with 'code'"; continue ;; esac
  if [ "$(git -C "$MJX_UPSTREAM" cat-file -t "$UP:verification/$EXP/$CODE" 2>/dev/null)" != tree ]; then
    echo "BUILD FAIL $spec: no directory verification/$EXP/$CODE at pinned $UP"; continue
  fi
  case $TAG in ""|jaxdump|retile) ;; *) echo "BUILD FAIL $spec: tag must be empty, jaxdump or retile"; continue ;; esac
  NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}${TAG:+-$TAG}
  BUILD=$MJX_REFERENCE/build/$NAME
  if [ -e "$BUILD" ] || [ -e "$MJX_REFERENCE/bin/$NAME" ]; then
    echo "BUILD FAIL $NAME: build or bin directory exists (nothing is overwritten)"; continue
  fi
  mkdir -p "$BUILD/scripts"
  git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) | tar -x -C "$BUILD/scripts"
  echo "$COMMIT" > "$BUILD/scripts/COMMIT"
  bash "$BUILD/scripts/reference/build.sh" "$BUILD" "$EXP" "$CODE" "$COMMIT" "$UP" "$TAG" > "$BUILD/build_job.out" 2>&1 &
  PIDS[$NAME]=$!
  echo "started $NAME pid ${PIDS[$NAME]}"
done

FAILED=0
for NAME in "${!PIDS[@]}"; do
  wait "${PIDS[$NAME]}"; RC=$?
  if [ "$RC" -eq 0 ] && [ -x "$MJX_REFERENCE/bin/$NAME/mitgcmuv" ]; then
    echo "BUILD OK $NAME -> $MJX_REFERENCE/bin/$NAME"
  else
    echo "BUILD FAIL $NAME rc=$RC (log $MJX_REFERENCE/build/$NAME/build.log)"
    tail -15 "$MJX_REFERENCE/build/$NAME/build_job.out"
    FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] && [ ${#PIDS[@]} -eq $# ] || { echo "BUILDS FAILED"; exit 1; }
echo "ALL BUILDS OK"
