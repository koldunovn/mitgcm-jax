#!/bin/bash
# Build one MITgcm verification experiment's code (or code_ad, forward only; or code_min) at the pinned upstream
# commit with gfortran: the Fortran oracle of mitjax (plan Task 3a; reference/README.md).
#
#   build.sh BUILD_DIR EXPERIMENT CODE_DIR MJX_COMMIT UPSTREAM_COMMIT [TAG]
#
# TAG (optional): empty = plain build; `jaxdump` = the instrumented oracle of plan Task 4: step 1b writes
# instrumented copies of model/src files (reference/jaxdump/instrument.py, from the snapshot and the experiment's
# code dir) + jaxdump.F + JAXDUMP.h into BUILD_DIR/jaxdump_mods, which genmake2 gets IN FRONT of the experiment's code
# dir (-mods="BUILD_DIR/jaxdump_mods <code dir>": the first directory holding a file wins, tools/genmake2:2945-2961).
# The snapshot's pkg/ and model/ are never modified. After make, the stages compiled into the binary are listed from
# the preprocessed sources (bld/*.f) in jaxdump_stages.txt.
# TAG `retile` = a second tiling for the invariance checks of plan Task 3b: step 1c copies the committed
# reference/retile/<EXPERIMENT>-<CODE_DIR>/SIZE.h (the experiment's SIZE.h or SIZE.h_mpi with only the tiling
# changed, single process; its header comment says what changed) into BUILD_DIR/retile_mods, in front of the code dir
# (-mods="BUILD_DIR/retile_mods <code dir>"); after make, bld/SIZE.h must equal that file.
#
# Normally started by reference/jobs/build.sbatch, which creates BUILD_DIR, unpacks this script's commit into
# BUILD_DIR/scripts (git archive) and runs it from there: the script that builds is the committed one. It refuses
# to run from anywhere else.
#
# Steps (testreport's sequence for a forward build, verification/testreport:1765-1771 without the run):
#   1. upstream snapshot: `git archive UPSTREAM_COMMIT tools eesupp model pkg doc/tag-index
#      verification/EXPERIMENT/CODE_DIR` from $MJX_UPSTREAM into BUILD_DIR/MITgcm. genmake2 writes generated sources
#      into its -rootdir (eesupp/src, pkg/exch2, pkg/regrid, pkg/mnc templates; tools/genmake2:2376-2405, 2571),
#      so the read-only clone is never the rootdir.
#   2. genmake2 in BUILD_DIR/bld with testreport's default options (testreport:381-417; OptLev=1 -> -ieee,
#      MPI=0 -> no -mpi): -ds -m make -mods=<code dir> -optfile=reference/optfile_levante_gfortran -ieee.
#      testreport runs genmake2 in verification/EXPERIMENT/build (testreport:381-390, `cd $1`), where genmake2
#      sources ./genmake_local (tools/genmake2:1447, 1768-1770): if the experiment has build/genmake_local at
#      UPSTREAM_COMMIT it is copied into BUILD_DIR/bld first (lane A session 7).
#      Code directories testreport does not build: code_min (adjustment.cs-32x32x1, "minimal" test case) gets
#      -standarddirs eesupp, the command line of its README ("to build": `genmake2 -standarddirs eesupp
#      -mods ../code_min`): eesupp/src and the packages of its packages.conf only, no model/src; its main.F skips
#      THE_MODEL_MAIN. No jaxdump build of it (nothing of model/src is compiled or called). Other code* directories
#      than code, code_ad, code_min are refused (code_tap/code_oad need -tap/-oad and their tools; the others have
#      no recorded command).
#   3. checks before compiling (reference/check_build.py): no requested package dropped (packages.conf or the
#      default list vs ENABLED_PACKAGES and PACKAGES_CONFIG.h), NetCDF found, strict flags (-O0 from -ieee,
#      -ffp-contract=off, -fconvert=big-endian, no fast-math/FMA/-march options).
#   4. make depend; make (testreport:519-540, 542-600; the build directory is new, so no `make Clean`).
#   5. checks after compiling: packages again, AD_CONFIG.h forward version (no ALLOW_ADJOINT_RUN /
#      ALLOW_TANGENTLINEAR_RUN), the effective macros of every *OPTIONS.h (cpp -dM) kept in BUILD_DIR/options.
#   6. freeze: $MJX_REFERENCE/bin/<name>/ (name = basename of BUILD_DIR) gets mitgcmuv (read-only), sha256, the
#      logs, genmake2 options and state, PACKAGES_CONFIG.h, AD_CONFIG.h, options/, provenance.txt. The build
#      directory stays: the preprocessed sources (*.f) in BUILD_DIR/bld are read by later tasks.
# Nothing is deleted or overwritten: an existing bin directory is a failure.
# Compiler: the Levante modules below (gfortran 11.2.0, Spack); serial netCDF-Fortran 4.5.3 (no MPI build).
set -euo pipefail

BUILD=${1:?build dir}; EXP=${2:?experiment}; CODE=${3:?code dir: code, code_ad or code_min}
MJX_COMMIT=${4:?mitjax commit}; UP_COMMIT=${5:?upstream commit}; TAG=${6-}
case $TAG in ""|jaxdump|retile) ;; *) echo "FAIL: unknown build tag '$TAG' (empty, jaxdump or retile)"; exit 2 ;; esac
EXTRA_GENMAKE=()
case $CODE in
  code|code_ad) ;;
  code_min)  # verification/adjustment.cs-32x32x1/README ("minimal" test case, "to build")
    EXTRA_GENMAKE=(-standarddirs eesupp)
    [ "$TAG" = jaxdump ] && { echo "FAIL: no jaxdump build of $CODE: model/src is not compiled, THE_MODEL_MAIN not called"; exit 2; } ;;
  *) echo "FAIL: no recorded genmake2 command for code directory '$CODE' (code, code_ad, code_min)"; exit 2 ;;
esac
SELF_DIR=$(cd "$(dirname "$0")" && pwd -P)
BUILD=$(cd "$BUILD" && pwd -P)
[ "$SELF_DIR" = "$BUILD/scripts/reference" ] || {
  echo "FAIL: build.sh runs only from its frozen copy $BUILD/scripts/reference (this is $SELF_DIR)"; exit 2; }
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$BUILD/scripts/mitjax/paths.py" --sh)"
NAME=$(basename "$BUILD")
BIN=$MJX_REFERENCE/bin/$NAME
ROOT=$BUILD/MITgcm
MODS=$ROOT/verification/$EXP/$CODE
CHECK="$PY $SELF_DIR/check_build.py"
[ -e "$BIN" ] && { echo "FAIL: $BIN exists (nothing is overwritten; a new commit gives a new name)"; exit 1; }
[ -e "$ROOT" ] && { echo "FAIL: $ROOT exists: a build directory is used once"; exit 1; }
exec > >(tee -a "$BUILD/build.log") 2>&1
step() { echo "== $(date +%H:%M:%S) $*"; }

step "upstream snapshot $UP_COMMIT -> $ROOT"
GIT=$(command -v git)            # git comes from a module: resolve it before `module purge`
mkdir -p "$ROOT" "$BUILD/bld"
"$GIT" -C "$MJX_UPSTREAM" archive --format=tar "$UP_COMMIT" tools eesupp model pkg doc/tag-index \
  "verification/$EXP/$CODE" | tar -x -C "$ROOT"
[ -d "$MODS" ] || { echo "FAIL: no $EXP/$CODE at $UP_COMMIT"; exit 1; }
GMLOCAL=verification/$EXP/build/genmake_local
if [ "$("$GIT" -C "$MJX_UPSTREAM" cat-file -t "$UP_COMMIT:$GMLOCAL" 2>/dev/null)" = blob ]; then
  "$GIT" -C "$MJX_UPSTREAM" archive --format=tar "$UP_COMMIT" "$GMLOCAL" | tar -x -C "$ROOT"
  cp "$ROOT/$GMLOCAL" "$BUILD/bld/genmake_local"
  echo "genmake_local from $GMLOCAL:"; cat "$BUILD/bld/genmake_local"
fi
ALLMODS=$MODS
if [ "$TAG" = jaxdump ]; then
  step "instrument (jaxdump) -> $BUILD/jaxdump_mods"
  JDMODS=$BUILD/jaxdump_mods
  [ -e "$JDMODS" ] && { echo "FAIL: $JDMODS exists"; exit 1; }
  "$PY" "$SELF_DIR/jaxdump/instrument.py" --root "$ROOT" --mods "$MODS" "$JDMODS" | tee "$BUILD/instrument_report.txt"
  ALLMODS="$JDMODS $MODS"
fi
if [ "$TAG" = retile ]; then
  step "second tiling (retile) -> $BUILD/retile_mods"
  RTSRC=$SELF_DIR/retile/$EXP-$CODE/SIZE.h
  RTMODS=$BUILD/retile_mods
  [ -f "$RTSRC" ] || { echo "FAIL: no $RTSRC"; exit 1; }
  [ -e "$RTMODS" ] && { echo "FAIL: $RTMODS exists"; exit 1; }
  mkdir "$RTMODS"
  cp "$RTSRC" "$RTMODS/SIZE.h"
  diff "$MODS/SIZE.h" "$RTMODS/SIZE.h" | tee "$BUILD/retile_diff.txt" || true
  ALLMODS="$RTMODS $MODS"
fi

step "modules"
# keep conda/mamba tools (an nc-config of another netCDF-C, a second cpp) off PATH
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0
set -u
unset NETCDF_ROOT NETCDF_HOME NETCDF_INC NETCDF_LIB NETCDF_INCDIR NETCDF_LIBDIR MPI_HOME MPI_INC_DIR
export MJX_NF_PREFIX; MJX_NF_PREFIX=$(nf-config --prefix)
export MJX_NC_LIBDIR; MJX_NC_LIBDIR=$(dirname "$(ldd "$MJX_NF_PREFIX/lib/libnetcdff.so" | awk '/libnetcdf\.so/ {print $3}')")
{
  echo "name        $NAME"
  echo "experiment  $EXP/$CODE"
  echo "upstream    $UP_COMMIT ($("$GIT" -C "$MJX_UPSTREAM" describe --tags --always "$UP_COMMIT"))"
  echo "mitjax      $MJX_COMMIT (scripts: $BUILD/scripts, git archive of that commit)"
  echo "host        $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
  echo "modules     $(module -t list 2>&1 | tr '\n' ' ')"
  echo "gfortran    $(command -v gfortran): $(gfortran --version | head -1)"
  echo "gcc         $(gcc --version | head -1)"
  echo "cpp         $(command -v cpp): $(cpp --version | head -1)"
  echo "netcdf      fortran $MJX_NF_PREFIX ($(nf-config --version)); c $MJX_NC_LIBDIR"
  echo "glibc       $(ldd --version | head -1)"
  echo "PATH        $PATH"
} | tee "$BUILD/provenance.txt"

step "genmake2"
cd "$BUILD/bld"
GENMAKE=("$ROOT/tools/genmake2" -ds -m make -rootdir="$ROOT" -mods="$ALLMODS"
         -optfile="$SELF_DIR/optfile_levante_gfortran" -ieee ${EXTRA_GENMAKE[@]+"${EXTRA_GENMAKE[@]}"})
printf '%q ' "${GENMAKE[@]}" > ../genmake2_command.txt; echo >> ../genmake2_command.txt
"${GENMAKE[@]}" > ../genmake2.out 2>&1 || { echo "FAIL: genmake2 (see $BUILD/genmake2.out)"; tail -30 ../genmake2.out; exit 1; }
grep -E "NetCDF-enabled" ../genmake2.out
if grep -q 'package is now DISABLED' ../genmake2.out; then
  grep -B3 -A3 'DISABLED' ../genmake2.out; echo "FAIL: genmake2 disabled a package"; exit 1
fi

step "checks before make"
$CHECK packages . --rootdir "$ROOT" --mods "$MODS" --report ../packages_check.txt
$CHECK flags . --ieee | tee ../flags_check.txt

step "make depend"
make depend > ../make_depend.log 2>&1 || { echo "FAIL: make depend"; tail -30 ../make_depend.log; exit 1; }
step "make"
make -j "${MJX_MAKE_JOBS:-16}" > ../make.log 2>&1 || {
  echo "FAIL: make"; grep -n -i -E "error" ../make.log | head -40; exit 1; }
[ -x mitgcmuv ] || { echo "FAIL: no mitgcmuv"; exit 1; }

step "checks after make"
$CHECK packages . --rootdir "$ROOT" --mods "$MODS" --report ../packages_check_after_make.txt
$CHECK forward . | tee ../forward_check.txt
$CHECK options . --out ../options
if [ "$TAG" = jaxdump ]; then
  # stages compiled into this binary (after CPP): "<stage> <file.f>" per generated call, from the preprocessed sources
  grep -o -E "CALL JAXDUMP_[A-Z]+\( *'[A-Za-z0-9_]+'" ./*.f | sed -E "s/^\.\/([^:]+):.*'([A-Za-z0-9_]+)'$/\2 \1/" \
    | sort -u > ../jaxdump_stages.txt
  grep -q -E "^S00_begin forward_step\.f$" ../jaxdump_stages.txt || { echo "FAIL: S00_begin not compiled"; exit 1; }
  grep -c "JAXDUMP_ITERSHIFT( 1 )" forward_step.f > /dev/null || { echo "FAIL: iteration shift not compiled"; exit 1; }
  echo "compiled stages: $(cut -d' ' -f1 ../jaxdump_stages.txt | sort -u | tr '\n' ' ')"
fi

if [ "$TAG" = retile ]; then
  cmp SIZE.h ../retile_mods/SIZE.h || { echo "FAIL: bld/SIZE.h is not the retile SIZE.h"; exit 1; }
fi

step "freeze -> $BIN"
mkdir -p "$MJX_REFERENCE/bin"
mkdir "$BIN"                      # fails if it appeared meanwhile
cp mitgcmuv "$BIN/mitgcmuv"
chmod a-w "$BIN/mitgcmuv"
(cd "$BIN" && sha256sum mitgcmuv > sha256)
ldd mitgcmuv > ../ldd.txt
mkdir "$BIN/logs"
cp ../genmake2.out genmake.log ../make_depend.log ../make.log "$BIN/logs/"
[ -f genmake_local ] && cp genmake_local "$BIN/genmake_local"
cp ../genmake2_command.txt genmake_state Makefile PACKAGES_CONFIG.h AD_CONFIG.h ../packages_check.txt \
   ../packages_check_after_make.txt ../flags_check.txt ../forward_check.txt ../ldd.txt ../provenance.txt "$BIN/"
cp -r ../options "$BIN/options"
if [ "$TAG" = jaxdump ]; then
  cp ../instrument_report.txt ../jaxdump_stages.txt "$BIN/"
  cp -r ../jaxdump_mods "$BIN/jaxdump_mods"
fi
if [ "$TAG" = retile ]; then
  cp ../retile_mods/SIZE.h "$BIN/SIZE.h"
  cp ../retile_diff.txt "$BIN/"
fi
cp "$SELF_DIR/optfile_levante_gfortran" "$BIN/"
echo "build dir   $BUILD" >> "$BIN/provenance.txt"
echo "sha256      $(cut -c1-64 "$BIN/sha256")" >> "$BIN/provenance.txt"
step "BUILT $BIN"
cp "$BUILD/build.log" "$BIN/build.log"
