#!/bin/bash
#SBATCH --job-name=mjx_cov_build
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# gcov builds of MITgcm verification codes (plan Task 5): lane A's oracle build (reference/build.sh, unchanged) with
# the coverage optfile reference/coverage/optfile_levante_gfortran_gcov (= the oracle optfile + `--coverage`).
#   sbatch -J mjx_cov_build reference/coverage/build_gcov.sbatch MJX_COMMIT EXP:CODE [EXP:CODE ...]
# Arguments after the script name only (sbatch --export splits values at commas).
#
# For each EXP:CODE the build is named <EXP>-<CODE>-<upstream short hash>-<MJX_COMMIT short>-gcov. As
# reference/jobs/build.sbatch does, the job creates $MJX_REFERENCE/build/<name>/ (refuses an existing one) and unpacks
# `git archive MJX_COMMIT reference mitjax/paths.py` into <name>/scripts — except reference/optfile_levante_gfortran,
# which is not extracted: the coverage optfile is placed under that name instead, so the frozen build.sh (which
# always reads $SELF_DIR/optfile_levante_gfortran) builds with it. Everything else is reference/build.sh as committed:
# upstream snapshot, genmake2 -ieee, package and flag checks (FOPTIM must still be exactly -O0), make, freeze to
# $MJX_REFERENCE/bin/<name>/. Builds run in parallel.
# After build.sh, this job checks the result and writes <bin>/COVERAGE.txt:
#   * FFLAGS has --coverage; every Fortran object of the build has its .gcno (the notes files stay in
#     $MJX_REFERENCE/build/<name>/bld, read by tools/coverage.py);
#   * CPPCMD, DEFINES and INCLUDES of the Makefile equal those of the plain oracle record of the same code
#     (mitjax/config/cpp_options.default_toolchain() requires all records under bin/ to agree).
# Verdict lines: "COVERAGE BUILD OK <name>" / "COVERAGE BUILD FAIL <name> ..."; exit 1 if any failed.
# Runs of these binaries MUST set GCOV_PREFIX (reference/coverage/run_gcov.sbatch does): without it the counts go to
# the build directory and accumulate over runs. Never deletes anything.
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
git -C "$REPO" cat-file -e "$COMMIT:reference/coverage/optfile_levante_gfortran_gcov" ||
  { echo "FAIL: commit $COMMIT has no reference/coverage/optfile_levante_gfortran_gcov"; exit 1; }
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT upstream $UP"

declare -A PIDS
for spec in "$@"; do
  IFS=: read -r EXP CODE <<< "$spec"
  case $CODE in code|code_ad|code_min) ;; *) echo "COVERAGE BUILD FAIL $spec: code dir must be code, code_ad or code_min"; continue ;; esac
  NAME=$EXP-$CODE-${UP:0:7}-${COMMIT:0:7}-gcov
  BUILD=$MJX_REFERENCE/build/$NAME
  if [ -e "$BUILD" ] || [ -e "$MJX_REFERENCE/bin/$NAME" ]; then
    echo "COVERAGE BUILD FAIL $NAME: build or bin directory exists (nothing is overwritten)"; continue
  fi
  mkdir -p "$BUILD/scripts"
  git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) |
    tar -x -C "$BUILD/scripts" --exclude=reference/optfile_levante_gfortran
  [ -e "$BUILD/scripts/reference/optfile_levante_gfortran" ] && {
    echo "COVERAGE BUILD FAIL $NAME: oracle optfile was extracted"; continue; }
  cp "$BUILD/scripts/reference/coverage/optfile_levante_gfortran_gcov" "$BUILD/scripts/reference/optfile_levante_gfortran"
  echo "$COMMIT" > "$BUILD/scripts/COMMIT"
  bash "$BUILD/scripts/reference/build.sh" "$BUILD" "$EXP" "$CODE" "$COMMIT" "$UP" > "$BUILD/build_job.out" 2>&1 &
  PIDS[$NAME]=$!
  echo "started $NAME pid ${PIDS[$NAME]}"
done

mkvar() { grep -E "^$1 =" "$2" | head -1; }
FAILED=0
for NAME in "${!PIDS[@]}"; do
  wait "${PIDS[$NAME]}"; RC=$?
  BIN=$MJX_REFERENCE/bin/$NAME; BLD=$MJX_REFERENCE/build/$NAME/bld
  if [ "$RC" -ne 0 ] || [ ! -x "$BIN/mitgcmuv" ]; then
    echo "COVERAGE BUILD FAIL $NAME rc=$RC (log $MJX_REFERENCE/build/$NAME/build.log)"
    tail -15 "$MJX_REFERENCE/build/$NAME/build_job.out"; FAILED=1; continue
  fi
  PROBLEMS=()
  grep -E '^FFLAGS =' "$BIN/Makefile" | grep -q -- ' --coverage' || PROBLEMS+=("FFLAGS lacks --coverage")
  NO=$(find "$BLD" -maxdepth 1 -name '*.o' | wc -l)
  NG=0; MISSING=""
  for o in "$BLD"/*.o; do
    if [ -e "${o%.o}.gcno" ]; then NG=$((NG + 1)); else
      [ -e "${o%.o}.c" ] || MISSING="$MISSING $(basename "$o")"; fi     # C sources: CFLAGS has no --coverage
  done
  [ -z "$MISSING" ] || PROBLEMS+=("Fortran objects without .gcno:$MISSING")
  EXP=${NAME%%-code*}; CODE=$(echo "$NAME" | sed -E 's/^.*-(code_ad|code_min|code)-[0-9a-f]{7}-[0-9a-f]{7}-gcov$/\1/')
  PLAIN=$(ls -d "$MJX_REFERENCE/bin/$EXP-$CODE-${UP:0:7}-"* 2>/dev/null | grep -E -- '-[0-9a-f]{7}$' | head -1)
  if [ -z "$PLAIN" ]; then PROBLEMS+=("no plain oracle record $EXP-$CODE-${UP:0:7}-<commit> to compare the preprocessor")
  else
    for v in CPPCMD DEFINES INCLUDES; do
      [ "$(mkvar $v "$BIN/Makefile")" = "$(mkvar $v "$PLAIN/Makefile")" ] ||
        PROBLEMS+=("$v differs from $(basename "$PLAIN"): '$(mkvar $v "$BIN/Makefile")'")
    done
  fi
  {
    echo "coverage build (plan Task 5): reference/coverage/build_gcov.sbatch job ${SLURM_JOB_ID:-none} $(date -Is)"
    echo "optfile      reference/coverage/optfile_levante_gfortran_gcov at $COMMIT (frozen here as optfile_levante_gfortran)"
    echo "gcno_dir     $BLD"
    echo "objects      $NO .o, $NG with .gcno"
    echo "plain record ${PLAIN:-none}"
    echo "runs MUST set GCOV_PREFIX (reference/coverage/run_gcov.sbatch)"
    for p in "${PROBLEMS[@]+"${PROBLEMS[@]}"}"; do echo "FAIL: $p"; done
    [ ${#PROBLEMS[@]} -eq 0 ] && echo "COVERAGE OK" || echo "COVERAGE BAD"
  } > "$BIN/COVERAGE.txt"
  cat "$BIN/COVERAGE.txt"
  if [ ${#PROBLEMS[@]} -eq 0 ]; then echo "COVERAGE BUILD OK $NAME -> $BIN"
  else echo "COVERAGE BUILD FAIL $NAME (see $BIN/COVERAGE.txt)"; FAILED=1; fi
done
[ "$FAILED" -eq 0 ] && [ ${#PIDS[@]} -eq $# ] || { echo "COVERAGE BUILDS FAILED"; exit 1; }
echo "ALL COVERAGE BUILDS OK"
