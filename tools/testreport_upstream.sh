#!/bin/bash
# Run MITgcm's own testreport code (verification/testreport at the pinned commit), not a re-implementation of it:
# the independent side of the cross-check of tools/testreport_jax.py.
#
#   testreport_upstream.sh build [out_dir]
#       Compile the comparison program tr_cmpnum exactly as testreport does: its function createcodelet
#       (testreport:950-998) is extracted from the file and run, with CC=${CC:-cc} as testreport sets it
#       (testreport:1128-1130). Default out_dir: $MJX_REFERENCE/tools/<7-char upstream commit>. Refuses to touch an
#       existing out_dir. Writes BUILD_INFO.txt (commit, compiler, sha256 of tr_cmpnum.c and tr_cmpnum).
#
#   testreport_upstream.sh run <tr_cmpnum_dir> <kind> <exp_src_dir> <variant> <output_file> <reference_file> <work_dir>
#       kind 0 = forward, 1 = tangent linear (TAF), 2 = adjoint (TAF) (testreport:1146-1149). Builds a miniature
#       experiment under the new directory <work_dir>: results/<reference name> -> reference_file,
#       run/<OUTPUTFILE> -> output_file, and input dirs holding only the check lists (tr_checklist*) of
#       <exp_src_dir>/<variant> and of its base dir (input, input_ad), which testreport's own linkdata links into
#       run/ (testreport:771-832; prepare_run and data files are left out on purpose). Then runs testreport's
#       testoutput_run (with testoutput_var) and formatresults from inside <tr_cmpnum_dir> (testoutput_var calls
#       ./tr_cmpnum) and prints three lines:
#           LISTVAR=<listVar as testoutput_run wrote it to summary.txt>
#           RESULTS=<the result string testoutput_run returns>
#           SUMMARY=<the summary.txt line: formatresults output through testreport's seds (testreport:1779)>
#       testoutput_var's messages (verbose=1) go to stderr.
#
# Paths: mitjax/paths.py (MJX_UPSTREAM, MJX_REFERENCE), run by $MJX_PYTHON, else python3 (paths.py is stdlib only).
# Never deletes anything: every directory it writes is new.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
eval "$(${MJX_PYTHON:-python3} "$REPO/mitjax/paths.py" --sh)"
TESTREPORT=$MJX_UPSTREAM/verification/testreport

die() { echo "testreport_upstream.sh: $*" >&2; exit 1; }
[ -r "$TESTREPORT" ] || die "no testreport at $TESTREPORT"

# extract_function NAME: the text of testreport's shell function NAME, from the line "NAME()" up to the next
# function definition (not to the first "}" in column 1: createcodelet's here-document ends its C code with one)
extract_function() {
    local text
    text=$(awk -v n="$1()" '$0 == n {f = 1; print; next}
                            f && /^[A-Za-z_][A-Za-z0-9_]*\(\)$/ {exit}
                            f {print}' "$TESTREPORT")
    [ -n "$text" ] || die "function $1 not found in $TESTREPORT"
    printf '%s\n' "$text"
}

cmd=${1:-}
case $cmd in
build)
    commit=$(git -C "$MJX_UPSTREAM" rev-parse HEAD) || die "no git checkout at $MJX_UPSTREAM"
    pinned=$(git -C "$MJX_UPSTREAM" rev-parse pinned) || die "no branch pinned at $MJX_UPSTREAM"
    [ "$commit" = "$pinned" ] || die "upstream HEAD $commit is not pinned $pinned"
    git -C "$MJX_UPSTREAM" diff --quiet HEAD -- verification/testreport || die "verification/testreport is modified"
    out=${2:-$MJX_REFERENCE/tools/${commit:0:7}}
    [ -e "$out" ] && die "$out exists; not touching it"
    mkdir -p "$out" || die "cannot create $out"
    cd "$out" || exit 1
    eval "$(extract_function createcodelet)"
    if test "x${CC:-}" = x ; then CC=cc ; fi        # testreport:1128-1130
    createcodelet || die "createcodelet failed"
    {
        echo "upstream     $MJX_UPSTREAM"
        echo "commit       $commit ($(git -C "$MJX_UPSTREAM" describe --tags "$commit"))"
        echo "source       verification/testreport, function createcodelet"
        echo "compiler     CC=$CC: $($CC --version 2>&1 | head -1) ($(readlink -f "$(command -v "$CC")"))"
        echo "command      $CC -o tr_cmpnum tr_cmpnum.c -lm"
        echo "host         $(hostname), $(date -Is)"
        sha256sum tr_cmpnum.c tr_cmpnum | sed 's/^/sha256       /'
    } > BUILD_INFO.txt
    cat BUILD_INFO.txt
    ;;
run)
    [ $# -eq 8 ] || die "usage: $0 run <tr_cmpnum_dir> <kind> <exp_src_dir> <variant> <output> <reference> <work_dir>"
    trdir=$(cd "$2" && pwd) || die "no directory $2"
    KIND=$3; src=$(cd "$4" && pwd) || die "no directory $4"; variant=$5
    out_file=$(readlink -f "$6"); ref_file=$(readlink -f "$7"); work=$8
    [ -x "$trdir/tr_cmpnum" ] || die "no executable $trdir/tr_cmpnum (run: $0 build)"
    [ -r "$out_file" ] && [ -r "$ref_file" ] || die "unreadable $out_file or $ref_file"
    case $KIND in
        0) inputdir=input ;    OUTPUTFILE=output.txt ;;       # testreport:1347-1351, 1416
        1) inputdir=input_ad ; OUTPUTFILE=output_tlm.txt ;;   # testreport:1319-1326
        2) inputdir=input_ad ; OUTPUTFILE=output_adm.txt ;;   # testreport:1327-1334
        *) die "kind $KIND not supported (0, 1, 2)" ;;
    esac
    case $variant in $inputdir|$inputdir.*) ;; *) die "variant $variant does not belong to kind $KIND" ;; esac
    [ -e "$work" ] && die "$work exists; not touching it"
    mkdir -p "$work/exp/results" "$work/exp/run" "$work/locdir" || die "cannot create $work"
    work=$(cd "$work" && pwd)
    EXP=$work/exp
    refname=$(basename "$ref_file")
    ln -s "$ref_file" "$EXP/results/$refname"
    ln -s "$out_file" "$EXP/run/$OUTPUTFILE"
    # input dirs with the check lists only; linkdata takes the variant's first, then the base dir's
    for d in $variant $inputdir ; do
        [ -d "$src/$d" ] && [ ! -e "$EXP/$d" ] || continue
        mkdir -p "$EXP/$d"
        for f in "$src/$d"/tr_checklist* ; do
            [ -e "$f" ] && ln -s "$f" "$EXP/$d/$(basename "$f")"
        done
    done
    # testreport's globals for these functions (testreport:1145-1154, 1413-1432) and its default check list
    debug=0 ; verbose=1 ; MPI=0 ; MULTI_THREAD=f ; PTRACERS_NUM="1 2 3 4 5" ; MATCH_CRIT=10
    TMP=$work/tr_tmp ; locDIR=$work/locdir ; CDIR=$work/locdir
    eval "$(sed -n '/^# set the Default List of output variables to be checked/,/^fi$/p' "$TESTREPORT")"
    [ -n "${DEF_CHECK_LIST:-}" ] || die "default check list not found in $TESTREPORT"
    for fn in linkdata testoutput_var testoutput_run formatresults ; do
        eval "$(extract_function $fn)"
    done
    if [ "$variant" = "$inputdir" ] ; then
        linkdata "$EXP/run" "$inputdir" 2>/dev/null
    else
        linkdata "$EXP/run" "$variant" "$inputdir" 2>/dev/null
    fi
    expname=$(basename "$src")
    [ "$variant" = "$inputdir" ] || expname=$expname.${variant#$inputdir.}
    cd "$trdir" || exit 1                              # testoutput_var runs ./tr_cmpnum
    results=$(testoutput_run "$EXP" run "$refname")
    fres=$(formatresults "$expname" Y Y Y Y $results)
    echo "LISTVAR=$(sed -n "s/^listVar='\(.*\)'$/\1/p" "$locDIR/summary.txt")"
    echo "RESULTS=$results"
    echo "SUMMARY=$(echo "$fres" | sed 's/ 99/ --/g' | sed 's/  > />/' | sed 's/  < /</')"
    ;;
*)
    sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
