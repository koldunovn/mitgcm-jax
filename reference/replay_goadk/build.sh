#!/bin/bash
# Build the GOADK replay harness (M2 sub-lane GOADK, plan Task 24; a copy of the GM/Redi harness reference/replay_gmredi,
# reference/replay_goadk/code/the_main_loop.F, R's adx_harness pattern [E§2]): MITgcm at the pinned commit, a
# verification experiment's code/ (or code_ad/, forward only) directory plus reference/replay_goadk/code
# (THE_MAIN_LOOP replaced), genmake2 and make exactly as the oracle (reference/build.sh: testreport's -ieee, i.e.
# FOPTIM=-O0; reference/optfile_levante_gfortran: -ffp-contract=off, -fconvert=big-endian; gfortran 11.2.0 module).
#
#   build.sh OUT EXPERIMENT CODE_DIR MJX_COMMIT UPSTREAM_COMMIT [ORACLE_BUILD_DIR]
#
# Started by reference/replay_goadk/jobs/replay.sbatch, which unpacks this script's commit into OUT/scripts (git archive) and
# runs it from there; it refuses to run from anywhere else. Layout:
#   OUT/MITgcm   upstream snapshot (git archive UPSTREAM_COMMIT tools eesupp model pkg doc/tag-index
#                verification/EXPERIMENT/CODE_DIR; genmake2 writes into its rootdir, so never the clone)
#   OUT/mods     the experiment's CODE_DIR files + reference/replay_goadk/code/* (a name in both is a failure)
#   OUT/bld      genmake2 + make; the preprocessed *.f stay
#   OUT/options  effective macros of every *OPTIONS.h (reference/check_build.py options)
#   OUT/bin      mitgcmuv (read-only) + sha256 + provenance.txt
# Checks: reference/check_build.py flags --ieee (strict flags, FOPTIM -O0); with ORACLE_BUILD_DIR (the oracle's build
# of the same experiment, reference/build.sh), every preprocessed gmredi_*.f, gad_dst3_adv_*.f, mom_*_botdrag_coeff.f and
# cost_atlantic_heat.f must be byte-identical to the oracle's: the harness compiles the oracle's code with its options.
# Nothing is deleted or overwritten: existing targets are a failure.
set -euo pipefail

OUT=${1:?out dir}; EXP=${2:?experiment}; CODE=${3:?code dir}
MJX_COMMIT=${4:?mitjax commit}; UP_COMMIT=${5:?upstream commit}; ORACLE=${6:-}
SELF_DIR=$(cd "$(dirname "$0")" && pwd -P)
OUT=$(cd "$OUT" && pwd -P)
[ "$SELF_DIR" = "$OUT/scripts/reference/replay_goadk" ] || {
  echo "FAIL: build.sh runs only from its frozen copy $OUT/scripts/reference/replay_goadk (this is $SELF_DIR)"; exit 2; }
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$OUT/scripts/mitjax/paths.py" --sh)"
ROOT=$OUT/MITgcm; MODS=$OUT/mods; BLD=$OUT/bld; BIN=$OUT/bin
CHECK="$PY $OUT/scripts/reference/check_build.py"
for d in "$ROOT" "$MODS" "$BLD" "$BIN"; do [ -e "$d" ] && { echo "FAIL: $d exists"; exit 1; }; done
exec > >(tee -a "$OUT/build.log") 2>&1
step() { echo "== $(date +%H:%M:%S) $*"; }

step "upstream snapshot $UP_COMMIT -> $ROOT"
GIT=$(command -v git)
mkdir -p "$ROOT" "$MODS" "$BLD"
"$GIT" -C "$MJX_UPSTREAM" archive --format=tar "$UP_COMMIT" tools eesupp model pkg doc/tag-index \
  "verification/$EXP/$CODE" | tar -x -C "$ROOT"
[ -d "$ROOT/verification/$EXP/$CODE" ] || { echo "FAIL: no $EXP/$CODE at $UP_COMMIT"; exit 1; }
cp "$ROOT/verification/$EXP/$CODE/"* "$MODS/"
for f in "$SELF_DIR/code/"*; do
  [ -e "$MODS/$(basename "$f")" ] && { echo "FAIL: $(basename "$f") is in both $EXP/$CODE and replay_goadk/code"; exit 1; }
  cp "$f" "$MODS/"
done

step "modules (as reference/build.sh)"
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0
set -u
unset NETCDF_ROOT NETCDF_HOME NETCDF_INC NETCDF_LIB NETCDF_INCDIR NETCDF_LIBDIR MPI_HOME MPI_INC_DIR
export MJX_NF_PREFIX; MJX_NF_PREFIX=$(nf-config --prefix)
export MJX_NC_LIBDIR; MJX_NC_LIBDIR=$(dirname "$(ldd "$MJX_NF_PREFIX/lib/libnetcdff.so" | awk '/libnetcdf\.so/ {print $3}')")
{
  echo "replay harness $EXP/$CODE (reference/replay_goadk)"
  echo "upstream    $UP_COMMIT ($("$GIT" -C "$MJX_UPSTREAM" describe --tags --always "$UP_COMMIT"))"
  echo "mitjax      $MJX_COMMIT (scripts: $OUT/scripts, git archive of that commit)"
  echo "host        $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
  echo "modules     $(module -t list 2>&1 | tr '\n' ' ')"
  echo "gfortran    $(command -v gfortran): $(gfortran --version | head -1)"
  echo "replay code $(cd "$SELF_DIR/code" && sha256sum ./* | awk '{print $2 "=" substr($1,1,12)}' | tr '\n' ' ')"
} | tee "$OUT/provenance.txt"

step "genmake2"
cd "$BLD"
GENMAKE=("$ROOT/tools/genmake2" -ds -m make -rootdir="$ROOT" -mods="$MODS"
         -optfile="$OUT/scripts/reference/optfile_levante_gfortran" -ieee)
printf '%q ' "${GENMAKE[@]}" > ../genmake2_command.txt; echo >> ../genmake2_command.txt
"${GENMAKE[@]}" > ../genmake2.out 2>&1 || { echo "FAIL: genmake2"; tail -30 ../genmake2.out; exit 1; }
step "flags check"
$CHECK flags . --ieee | tee ../flags_check.txt
step "make depend"
make depend > ../make_depend.log 2>&1 || { echo "FAIL: make depend"; tail -30 ../make_depend.log; exit 1; }
step "make"
make -j "${MJX_MAKE_JOBS:-16}" > ../make.log 2>&1 || {
  echo "FAIL: make"; grep -n -i -E "error" ../make.log | head -40; exit 1; }
[ -x mitgcmuv ] || { echo "FAIL: no mitgcmuv"; exit 1; }
$CHECK options . --out ../options
grep -q "THE_MAIN_LOOP of the mitjax GOADK replay" the_main_loop.f || {
  echo "FAIL: the build did not take reference/replay_goadk/code/the_main_loop.F"; exit 1; }

if [ -n "$ORACLE" ]; then
  step "preprocessed routines vs the oracle build $ORACLE"
  for f in $(cd "$BLD" && ls gmredi_*.f gad_dst3_adv_*.f mom_*_botdrag_coeff.f cost_atlantic_heat.f); do
    cmp "$BLD/$f" "$ORACLE/bld/$f" || { echo "FAIL: $f differs from the oracle's $ORACLE/bld/$f"; exit 1; }
    echo "identical $f ($(sha256sum "$f" | cut -c1-12))"
  done | tee ../oracle_identity.txt
fi

step "freeze -> $BIN"
mkdir "$BIN"
cp mitgcmuv "$BIN/mitgcmuv"
chmod a-w "$BIN/mitgcmuv"
(cd "$BIN" && sha256sum mitgcmuv > sha256)
ldd mitgcmuv > "$BIN/ldd.txt"
cp ../provenance.txt ../genmake2_command.txt ../flags_check.txt Makefile PACKAGES_CONFIG.h "$BIN/"
[ -n "$ORACLE" ] && cp ../oracle_identity.txt "$BIN/"
echo "sha256      $(cut -c1-64 "$BIN/sha256")" >> "$BIN/provenance.txt"
step "BUILT $BIN"
