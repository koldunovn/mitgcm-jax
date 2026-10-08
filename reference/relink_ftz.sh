#!/bin/bash
# The flush-to-zero (FTZ/DAZ) variant of a frozen oracle build (plan decision 13, Nikolay 2026-10-04).
#
#   relink_ftz.sh BUILD_DIR SRC_NAME MJX_COMMIT
#
# XLA:CPU computes with the x86 MXCSR flags FTZ (flush-to-zero) and DAZ (denormals-are-zero) set (L-CONF-2; no XLA
# flag turns them off); gfortran's binaries keep subnormals. Where a gate meets subnormals the port is gated bitwise
# against an oracle with FTZ/DAZ set at program start. That oracle is the SAME build: the object files of the frozen
# standard build SRC_NAME ($MJX_REFERENCE/build/SRC_NAME/bld/*.o, compiled by reference/build.sh), linked once more
# with the standard build's own link command plus the crtfastmath.o of the SAME compiler (gfortran 11.2.0 of the
# Levante module, `gfortran -print-file-name=crtfastmath.o`; its constructor set_fast_math sets FTZ and DAZ in MXCSR
# before main, libgcc/config/i386/crtfastmath.c). No -ffast-math code generation: no object is compiled here.
#
# Normally started by reference/jobs/ftz_build.sbatch, which creates BUILD_DIR ($MJX_REFERENCE/build/SRC_NAME-ftz-<MJX_COMMIT
# short>: one directory per attempt; the frozen binary is $MJX_REFERENCE/bin/SRC_NAME-ftz),
# unpacks this script's commit into BUILD_DIR/scripts and runs it from there. Steps:
#   1. checks: the source bin directory's mitgcmuv has the sha256 of its sha256 file; the source build directory
#      holds the objects; the compiler of the Levante modules is the one of the source provenance.txt.
#   2. crtfastmath.o: `gfortran -print-file-name=crtfastmath.o` must be an existing absolute path under that
#      compiler's prefix (not the system gcc 8 one); its sha256 is recorded.
#   3. the link command: the line after `Creating mitgcmuv ...` of the source make.log (Makefile rule
#      `$(LINK) -o $@ $(FFLAGS) $(FOPTIM) $(OBJFILES) $(LIBS)`); it must name `-o mitgcmuv` and no fast-math option.
#   4. the objects: every *.o of the source bld/ copied (cp -p) into BUILD_DIR/bld; sha256 of each, in both
#      directories, written to objects_sha256.txt / objects_sha256_src.txt: identical lists, and every object the
#      link command names is there.
#   5. standard relink: the link command unchanged in BUILD_DIR/bld must reproduce the frozen standard binary
#      byte for byte (same sha256): the objects and the command are those of the standard build.
#   6. FTZ link: the same command with crtfastmath.o appended -> mitgcmuv. The FTZ binary holds set_fast_math (nm),
#      the relinked standard binary does not.
#   7. run-time check of MXCSR at the entry of the main program (MAIN__) under gdb, both binaries: the FTZ binary
#      has DAZ and FZ set, the standard one neither. The probe reference/ftz/ftz_probe.F, compiled with the source
#      build's FFLAGS/FOPTIM and linked both ways: 1e-300*1e-10 and 1e-310*1 are 0 with FTZ/DAZ, nonzero without.
#   8. freeze -> $MJX_REFERENCE/bin/SRC_NAME-ftz/: mitgcmuv (read-only), sha256, every other file of the source bin
#      directory (options/, logs/, jaxdump_stages.txt, ... : the build record is the source build's), provenance.txt
#      (the source's with the name and sha256 lines of this binary), ftz_provenance.txt (everything above).
# Nothing is deleted or overwritten: an existing bin or bld directory is a failure.
set -euo pipefail

BUILD=${1:?build dir}; SRC=${2:?source build name}; MJX_COMMIT=${3:?mitjax commit}
SELF_DIR=$(cd "$(dirname "$0")" && pwd -P)
BUILD=$(cd "$BUILD" && pwd -P)
[ "$SELF_DIR" = "$BUILD/scripts/reference" ] || {
  echo "FAIL: relink_ftz.sh runs only from its frozen copy $BUILD/scripts/reference (this is $SELF_DIR)"; exit 2; }
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$BUILD/scripts/mitjax/paths.py" --sh)"
NAME=$SRC-ftz
case $(basename "$BUILD") in "$NAME-${MJX_COMMIT:0:7}") ;; *) echo "FAIL: build dir is not $NAME-${MJX_COMMIT:0:7}"; exit 2 ;; esac
SBIN=$MJX_REFERENCE/bin/$SRC
SBLD=$MJX_REFERENCE/build/$SRC/bld
BIN=$MJX_REFERENCE/bin/$NAME
[ -e "$BIN" ] && { echo "FAIL: $BIN exists (nothing is overwritten)"; exit 1; }
[ -e "$BUILD/bld" ] && { echo "FAIL: $BUILD/bld exists: a build directory is used once"; exit 1; }
exec > >(tee -a "$BUILD/build.log") 2>&1
step() { echo "== $(date +%H:%M:%S) $*"; }
FP=$BUILD/ftz_provenance.txt
rec() { echo "$*" | tee -a "$FP"; }

step "1. source build $SRC"
[ -f "$SBIN/mitgcmuv" ] && [ -f "$SBIN/sha256" ] || { echo "FAIL: no frozen binary $SBIN"; exit 1; }
SSHA=$(cut -d' ' -f1 "$SBIN/sha256")
[ "$(sha256sum "$SBIN/mitgcmuv" | cut -d' ' -f1)" = "$SSHA" ] || { echo "FAIL: $SBIN/mitgcmuv != its sha256 file"; exit 1; }
[ -f "$SBLD/../make.log" ] || { echo "FAIL: no $SBLD/../make.log"; exit 1; }

PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0
set -u
FCLINE="$(command -v gfortran): $(gfortran --version | head -1)"
SFCLINE=$(sed -n 's/^gfortran *//p' "$SBIN/provenance.txt")
[ "$FCLINE" = "$SFCLINE" ] || { echo "FAIL: compiler '$FCLINE' is not the source build's '$SFCLINE'"; exit 1; }
FCPREFIX=$(dirname "$(dirname "$(command -v gfortran)")")
: > "$FP"
rec "name        $NAME"
rec "variant     FTZ/DAZ at program start (plan decision 13): the objects of $SRC linked with crtfastmath.o"
rec "source      $SBIN (sha256 $SSHA)"
rec "mitjax      $MJX_COMMIT (scripts: $BUILD/scripts, git archive of that commit)"
rec "host        $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
rec "gfortran    $FCLINE"

step "2. crtfastmath.o of this compiler"
CRT=$(gfortran -print-file-name=crtfastmath.o)
case $CRT in "$FCPREFIX"/*) ;; *) echo "FAIL: crtfastmath.o '$CRT' is not under $FCPREFIX"; exit 1 ;; esac
[ -f "$CRT" ] || { echo "FAIL: no file $CRT"; exit 1; }
rec "crtfastmath $CRT sha256 $(sha256sum "$CRT" | cut -d' ' -f1)"
rec "            set_fast_math in it: $(nm "$CRT" | grep -c -w set_fast_math) symbol(s)"
rec "            (system gcc: $(/usr/bin/gcc -print-file-name=crtfastmath.o 2>/dev/null || echo none); not used)"

step "3. link command of the source build"
LINKCMD=$(awk '/^Creating mitgcmuv \.\.\.$/ {getline; print; exit}' "$SBLD/../make.log")
case $LINKCMD in "gfortran "*" -o mitgcmuv "*) ;; *) echo "FAIL: no link command after 'Creating mitgcmuv' in make.log"; exit 1 ;; esac
echo "$LINKCMD" | grep -q -E -- '-ffast-math|-Ofast|-funsafe-math|crtfastmath|-ffinite-math|-fno-signed-zeros' &&
  { echo "FAIL: fast-math option in the standard link command"; exit 1; }
echo "$LINKCMD" > "$BUILD/link_command_std.txt"
echo "$LINKCMD $CRT" > "$BUILD/link_command_ftz.txt"
rec "link std    $(echo "$LINKCMD" | cut -c1-160) ... ($(echo "$LINKCMD" | wc -w) words; $BUILD/link_command_std.txt)"
rec "link ftz    the same + $CRT"

step "4. objects"
mkdir "$BUILD/bld"
cp -p "$SBLD"/*.o "$BUILD/bld/"
(cd "$SBLD" && sha256sum ./*.o) > "$BUILD/objects_sha256_src.txt"
(cd "$BUILD/bld" && sha256sum ./*.o) > "$BUILD/objects_sha256.txt"
cmp "$BUILD/objects_sha256_src.txt" "$BUILD/objects_sha256.txt" || { echo "FAIL: copied objects differ"; exit 1; }
NOBJ=$(wc -l < "$BUILD/objects_sha256.txt")
for o in $(echo "$LINKCMD" | tr ' ' '\n' | grep -E '\.o$'); do
  [ -f "$BUILD/bld/$o" ] || { echo "FAIL: the link command names $o, not in $SBLD"; exit 1; }
done
rec "objects     $NOBJ object files, sha256 identical to $SBLD (objects_sha256.txt = objects_sha256_src.txt);" \
    "$(echo "$LINKCMD" | tr ' ' '\n' | grep -c -E '\.o$') named by the link command"

step "5. standard relink reproduces $SRC"
cd "$BUILD/bld"
bash -c "$LINKCMD"
RSHA=$(sha256sum mitgcmuv | cut -d' ' -f1)
[ "$RSHA" = "$SSHA" ] || { echo "FAIL: the relinked standard binary $RSHA is not $SRC's $SSHA"; exit 1; }
mv mitgcmuv mitgcmuv_std_relink
rec "relink std  sha256 $RSHA = $SRC (the objects and the link command reproduce the standard binary)"

step "6. FTZ link"
bash -c "$LINKCMD $CRT"
FSHA=$(sha256sum mitgcmuv | cut -d' ' -f1)
[ "$FSHA" != "$SSHA" ] || { echo "FAIL: the FTZ binary equals the standard one"; exit 1; }
NF=$(nm mitgcmuv | grep -c -w set_fast_math || true)
NS=$(nm mitgcmuv_std_relink | grep -c -w set_fast_math || true)
[ "$NF" -ge 1 ] && [ "$NS" -eq 0 ] || { echo "FAIL: set_fast_math in FTZ $NF, in standard $NS"; exit 1; }
rec "ftz binary  sha256 $FSHA; set_fast_math: FTZ $NF, standard $NS"

step "7. run-time MXCSR and probe"
mkdir "$BUILD/mxcsr_check"
cd "$BUILD/mxcsr_check"
mx() { gdb -nx -batch -ex 'break MAIN__' -ex run -ex 'p $mxcsr' -ex kill "$1" 2>&1 | grep -E '^\$1 = ' | sed 's/^\$1 = //' || true; }
MXF=$(mx "$BUILD/bld/mitgcmuv"); MXS=$(mx "$BUILD/bld/mitgcmuv_std_relink")
rec "MXCSR       at MAIN__ (gdb): FTZ binary $MXF; standard binary $MXS"
echo "$MXF" | grep -q -w DAZ && echo "$MXF" | grep -q -w FZ || { echo "FAIL: FTZ binary MXCSR '$MXF' lacks DAZ/FZ"; exit 1; }
echo "$MXS" | grep -q -w -E 'DAZ|FZ' && { echo "FAIL: standard binary MXCSR '$MXS' has DAZ/FZ"; exit 1; }
FFL=$(sed -n 's/^FFLAGS *= *//p' "$SBLD/Makefile"); FOP=$(sed -n 's/^FOPTIM *= *//p' "$SBLD/Makefile")
rec "probe       gfortran $FFL $FOP -c reference/ftz/ftz_probe.F"
# shellcheck disable=SC2086
gfortran $FFL $FOP -c "$SELF_DIR/ftz/ftz_probe.F" -o ftz_probe.o
gfortran -o probe_std ftz_probe.o
gfortran -o probe_ftz ftz_probe.o "$CRT"
for args in "1.D-300 1.D-10" "1.D-310 1.D0"; do
  # shellcheck disable=SC2086
  PS=$(./probe_std $args | sed 's/^FTZPROBE x, y, x\*y: *//'); PF=$(./probe_ftz $args | sed 's/^FTZPROBE x, y, x\*y: *//')
  rec "probe       $args: standard -> $PS ; ftz -> $PF"
  # the product token: exactly zero is 0.00000000000000000E+000 (by string: awk's own conversion underflows)
  ZS=$(echo "$PS" | awk '{print ($3 ~ /^-?0\.0+E\+000$/) ? "zero" : "nonzero"}')
  ZF=$(echo "$PF" | awk '{print ($3 ~ /^-?0\.0+E\+000$/) ? "zero" : "nonzero"}')
  [ "$ZS" = nonzero ] && [ "$ZF" = zero ] || { echo "FAIL: probe $args: standard $ZS, ftz $ZF"; exit 1; }
done

step "8. freeze -> $BIN"
mkdir "$BIN"
(cd "$SBIN" && find . -mindepth 1 -maxdepth 1 ! -name mitgcmuv ! -name sha256 ! -name provenance.txt -print0 |
  xargs -0 -I{} cp -r {} "$BIN/")
cp "$BUILD/bld/mitgcmuv" "$BIN/mitgcmuv"
chmod a-w "$BIN/mitgcmuv"
(cd "$BIN" && sha256sum mitgcmuv > sha256)
[ "$(cut -d' ' -f1 "$BIN/sha256")" = "$FSHA" ] || { echo "FAIL: frozen copy differs"; exit 1; }
ldd "$BIN/mitgcmuv" > "$BIN/ldd.txt"
sed -e "s|^name .*|name        $NAME|" -e '/^sha256 /d' "$SBIN/provenance.txt" > "$BIN/provenance.txt"
echo "ftz         FTZ/DAZ variant of $SRC (crtfastmath.o linked; ftz_provenance.txt)" >> "$BIN/provenance.txt"
echo "sha256      $FSHA" >> "$BIN/provenance.txt"
cp "$BUILD/link_command_std.txt" "$BUILD/link_command_ftz.txt" "$BUILD/objects_sha256.txt" "$BIN/"
cp "$SELF_DIR/ftz/ftz_probe.F" "$BIN/"
rec "frozen      $BIN"
cp "$FP" "$BIN/ftz_provenance.txt"
step "BUILT FTZ $BIN"
cp "$BUILD/build.log" "$BIN/build_ftz.log"
