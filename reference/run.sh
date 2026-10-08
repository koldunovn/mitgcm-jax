#!/bin/bash
# Run the Fortran oracle in a run directory made by reference/make_rundir.py (plan Task 3a; reference/README.md).
#
#   run.sh RUN_TOP        RUN_TOP = $MJX_REFERENCE_RUNS/<experiment>/<input dir>/<run id> (holds READY, MANIFEST.json,
#                         rundir -> verification/<experiment>/<run | tr_run.v>)
#
# As testreport runs a serial (MPI=0) test (verification/testreport:1414-1417, 896-899):
#   ( ./mitgcmuv > output.txt ) >> run.tr_log 2>&1
# so the model's standard output (the STDOUT testreport compares, with the %MON and cg2d_init_res lines) lands in
# <rundir>/output.txt and anything on stderr in <rundir>/run.tr_log. Success = exit code 0 AND
# "PROGRAM MAIN: Execution ended Normally" in the last lines of output.txt (testreport:899). Both are checked
# (reference/run_verdict.py). Exit codes: 0 RUN OK, 3 RUN EXPECTED-STOP (an abnormal end declared for the variant in
# run_verdict.EXPECTED_STOPS and matched exactly, e.g. the forward-only optim build's grdchk stop), 1 RUN FAIL.
# Refuses a directory without READY, with REFUSED.txt, or that already ran (output.txt exists); checks the
# executable against the sha256 recorded by make_rundir.py. Writes RUN_TOP/run_provenance.txt. Deletes nothing.
set -u
TOP=$(cd "${1:?run dir (RUN_TOP)}" && pwd -P) || exit 1
[ -f "$TOP/READY" ] || { echo "RUN FAIL $TOP: no READY (make_rundir.py did not finish)"; exit 1; }
[ -e "$TOP/REFUSED.txt" ] && { echo "RUN FAIL $TOP: REFUSED.txt present"; exit 1; }
RUN=$(cd "$TOP/rundir" && pwd -P) || exit 1
cd "$RUN" || exit 1
[ -e output.txt ] && { echo "RUN FAIL $RUN already ran (output.txt exists); make a new run dir"; exit 1; }
[ -e mitgcmuv ] || { echo "RUN FAIL $RUN: no mitgcmuv (make_rundir.py --binary)"; exit 1; }
PY=${MJX_PYTHON:-python}
WANT=$("$PY" -c 'import json,sys; b=json.load(open(sys.argv[1]))["binary"]; print(b["sha256"] if b else "")' \
       "$TOP/MANIFEST.json")
GOT=$(sha256sum "$(readlink -f mitgcmuv)" | cut -c1-64)
[ -n "$WANT" ] && [ "$GOT" = "$WANT" ] || { echo "RUN FAIL: mitgcmuv sha256 $GOT != recorded '$WANT'"; exit 1; }

PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'mambaforge|conda' | paste -sd: -)
set +u
type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge; module load gcc/11.2.0-gcc-11.2.0 netcdf-fortran/4.5.3-gcc-11.2.0   # as reference/build.sh
set -u
ulimit -s unlimited
{
  echo "job ${SLURM_JOB_ID:-none} host $(hostname) $(date -Is)"
  echo "run dir $RUN"
  echo "binary $(readlink -f mitgcmuv) sha256 $GOT"
  echo "command ( ./mitgcmuv > output.txt ) >> run.tr_log 2>&1   (testreport:1417, 899)"
  echo "ldd:"; ldd ./mitgcmuv
} > "$TOP/run_provenance.txt"
T0=$(date +%s)
( ./mitgcmuv > output.txt ) >> run.tr_log 2>&1
RC=$?
END=$(tail output.txt | grep -c 'PROGRAM MAIN: Execution ended Normally')
echo "exit $RC normal_end $END wall $(( $(date +%s) - T0 )) s" | tee -a "$TOP/run_provenance.txt"
# verdict: RUN OK (exit 0), RUN EXPECTED-STOP (a stop declared for this variant, matched by its exact signature:
# exit 3), RUN FAIL (exit 1) -- reference/run_verdict.py, next to this script (also in a job's snapshot)
"$PY" "$(dirname "$0")/run_verdict.py" "$TOP" "$RC" | tee -a "$TOP/run_provenance.txt"
VRC=${PIPESTATUS[0]}
[ "$VRC" -eq 1 ] && tail -20 output.txt run.tr_log
[ "$VRC" -eq 0 ] || [ "$VRC" -eq 3 ] || VRC=1
exit "$VRC"
