#!/usr/bin/env python3
"""Verdict of one oracle run made by reference/make_rundir.py and run by reference/run.sh (REVIEW_M0 #12).

    run_verdict.py RUN_TOP EXIT_CODE

As testreport (verification/testreport:899): success = exit code 0 AND "PROGRAM MAIN: Execution ended Normally" in
the last lines of <rundir>/output.txt -> `RUN OK`, exit 0. An abnormal end DECLARED for the run's variant in
EXPECTED_STOPS, matched by its exact signature (exit code, the whole run.tr_log, the last STDOUT lines, and STDOUT
lines that must precede them), -> `RUN EXPECTED-STOP` with the reason, exit 3 (callers that accept it say so). Anything
else -> `RUN FAIL`, exit 1: a real failure is never mistaken for the expected one. Stdlib only; deletes nothing.
"""

import json
import sys
from pathlib import Path

NORMAL_END = "PROGRAM MAIN: Execution ended Normally"

# (experiment, input dir): the declared abnormal end, measured in the 12 forward-only runs of the code_ad build under
# $MJX_REFERENCE_RUNS/tutorial_global_oce_optim/input_ad (jobs 27826794, 27826871, 27826873, 27826998, 27827363,
# 27827364: all exit 0 with this run.tr_log byte for byte)
EXPECTED_STOPS = {
    ("tutorial_global_oce_optim", "input_ad"): {
        "why": "forward-only code_ad build: GRDCHK_MAIN reads adxx_qnet (MDS_READ_FIELD), which only an adjoint run "
               "writes; the STDOUT is complete up to the gradient check",
        "exit": 0,
        "tr_log": "STOP ABNORMAL END: S/R MDS_READ_FIELD\n"
                  " MDS_READ_FIELD: filename: adxx_qnet.0000000000 , adxx_qnet.0000000000.001.001.data\n"
                  " MDS_READ_FIELD: Files DO not exist\n",
        "stdout_tail": ["(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_qnet.0000000000 , "
                        "adxx_qnet.0000000000.001.001.data",
                        "(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist"],
        "stdout_has": ["// CONFIG_CHECK : Normal End",
                       "(PID.TID 0000.0001) ====== Starts gradient-check number   1 (=ichknum) ======="],
    },
}


# M2 forward-only code_ad variants (lane A session 6): the same stop, measured in the plain, jdoff and jdon runs of
# job 27832229 (identical run.tr_log in all three). global_ocean.90x40x15/input_ad* run with useSingleCpuIO=.TRUE.
# (one global file, adxx_<name>.0000000000.data) and print three warnings first; tutorial_tracer_adjsens reads tiled
# files and gfortran notes a signalling underflow flag at the STOP.
_GO_TR = ("STOP ABNORMAL END: S/R MDS_READ_FIELD\n"
          "** WARNING ** CTRL_CHECK: relying on mdsio_gl.F to pack/unpack the controlvector is unsafe when "
          "useSingleCpuIO is true.\n"
          "** WARNING ** CONFIG_CHECK: nonlinFreeSurf might cause problems\n"
          "** WARNING ** with different FreeSurf & Tracer time-steps\n"
          " MDS_READ_FIELD: filename: adxx_{n}.0000000000.data\n"
          " MDS_READ_FIELD: File does not exist\n")
_GO_TAIL = ["(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_{n}.0000000000.data",
            "(PID.TID 0000.0001)  MDS_READ_FIELD: File does not exist"]
_TA_TR = ("Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG\n"
          "STOP ABNORMAL END: S/R MDS_READ_FIELD\n"
          " MDS_READ_FIELD: filename: adxx_{n}.0000000000 , adxx_{n}.0000000000.001.001.data\n"
          " MDS_READ_FIELD: Files DO not exist\n")
_TA_TAIL = ["(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_{n}.0000000000 , adxx_{n}.0000000000.001.001.data",
            "(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist"]
for _key, _name, _tr, _tail in (
        (("global_ocean.90x40x15", "input_ad"), "xx_theta", _GO_TR, _GO_TAIL),
        (("global_ocean.90x40x15", "input_ad.kapgm"), "xx_kapgm", _GO_TR, _GO_TAIL),
        (("global_ocean.90x40x15", "input_ad.kapredi"), "xx_kapredi", _GO_TR, _GO_TAIL),
        (("global_ocean.90x40x15", "input_ad.bottomdrag"), "xx_bottomdrag", _GO_TR, _GO_TAIL),
        (("tutorial_tracer_adjsens", "input_ad"), "xx_ptr1", _TA_TR, _TA_TAIL),
        (("tutorial_tracer_adjsens", "input_ad.som81"), "xx_ptr1", _TA_TR, _TA_TAIL)):
    EXPECTED_STOPS[_key] = {
        "why": f"forward-only code_ad build: GRDCHK_MAIN reads ad{_name} (MDS_READ_FIELD), which only an adjoint run "
               "writes; the STDOUT is complete up to the gradient check",
        "exit": 0, "tr_log": _tr.replace("xx_{n}", _name),
        "stdout_tail": [s.replace("xx_{n}", _name) for s in _tail],
        "stdout_has": list(EXPECTED_STOPS[("tutorial_global_oce_optim", "input_ad")]["stdout_has"])}


# global_ocean.cs32x15/input_ad (lane A session 7): the same stop, measured in the plain, jdoff and jdon runs of job
# 27832624 (identical run.tr_log in all three). Tiled files (useSingleCpuIO not set); before the stop the pickup check
# notes that pickup.0000072000 (written without AB3 fields) lacks GuNm2/GvNm2: code_ad/CPP_OPTIONS.h defines
# ALLOW_ADAMSBASHFORTH_3, so the restart is "approximated" with mom_StartAB=1.
EXPECTED_STOPS[("global_ocean.cs32x15", "input_ad")] = {
    "why": "forward-only code_ad build: GRDCHK_MAIN reads adxx_theta (MDS_READ_FIELD), which only an adjoint run "
           "writes; the STDOUT is complete up to the gradient check",
    "exit": 0,
    "tr_log": "STOP ABNORMAL END: S/R MDS_READ_FIELD\n"
              "READ_MFLDS_CHECK: reading from file: pickup.0000072000\n"
              "READ_MFLDS_CHECK: which contains   11 fields :\n"
              " >Uvel    < >GuNm1   < >Vvel    < >GvNm1   < >Theta   < >GtNm1   < >Salt    < >GsNm1   < >EtaN    < "
              ">dEtaHdt < >EtaH    <\n"
              "READ_MFLDS_CHECK:    2 field(s) is/are missing :\n"
              " >GuNm2   < >GvNm2   <\n"
              "** WARNING ** CHECK_PICKUP: Will get only an approximated Restart\n"
              " Continue with mom_StartAB =         1 ; nHydStartAB =     72000\n"
              "          with tempStartAB =     72000 ; saltStartAB =     72000\n"
              + _TA_TR.split("\n", 2)[2].replace("xx_{n}", "xx_theta"),
    "stdout_tail": [s.replace("xx_{n}", "xx_theta") for s in _TA_TAIL],
    "stdout_has": list(EXPECTED_STOPS[("tutorial_global_oce_optim", "input_ad")]["stdout_has"])}


# Output-request overlays of the code_ad variants (lane A session 8, job 27833807): the same stop; the package that is
# switched off prints its two warning lines on stderr after the STOP line (measured: run.tr_log of the overlay run =
# the variant's declared run.tr_log with these two lines inserted after its first line)
_OFF_WARN = {
    "useMNC=.FALSE.": "** Warning ** MNC_READPARMS: ignores \"data.mnc\" file since\n"
                      "** Warning ** MNC_READPARMS: useMNC= F (set from \"data.pkg\")\n",
    "useDiagnostics=.FALSE.": "** Warning ** DIAGNOSTICS_READPARMS: ignores \"data.diagnostics\" file since\n"
                              "** Warning ** DIAGNOSTICS_READPARMS: useDiagnostics= F (set from \"data.pkg\")\n"}
for _key, _ov in ((("global_ocean.90x40x15", "input_ad.bottomdrag"), "useMNC=.FALSE."),
                  (("global_ocean.cs32x15", "input_ad"), "useDiagnostics=.FALSE.")):
    _first, _rest = EXPECTED_STOPS[_key]["tr_log"].split("\n", 1)
    EXPECTED_STOPS[_key + (_ov,)] = dict(EXPECTED_STOPS[_key], tr_log=_first + "\n" + _OFF_WARN[_ov] + _rest,
                                         why=EXPECTED_STOPS[_key]["why"] + f" (overlay {_ov}: the package's "
                                         "two warnings on stderr)")


# M4 forward-only code_ad variants (lane A session 11): the same grdchk stop, measured in the plain, jdoff and jdon runs of
# jobs 27855986 (1D_ocean_ice_column, offline_exf_seaice), 27855987 (lab_sea) and 27855988 (global_ocean.cs32x15
# input_ad.seaice, .seaice_dynmix, .thsice); run.tr_log identical in all three
# modes of each variant (md5 compared). Entries: (key, control, job, run.tr_log, last STDOUT lines). Warnings on stderr
# differ per variant (packages switched off in data.pkg, SEAICE_READPARMS advection notes); lab_sea reads its controls
# from ctrlDir='./ctrl_variables' (data.ctrl); 1D_ocean_ice_column/input_ad reads one global file (useSingleCpuIO).
M4_STOPS = [
    (('1D_ocean_ice_column', 'input_ad'), 'xx_theta', 27855986,
     'Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG\nSTOP ABNORMAL END: S/R MDS_READ_FIELD\n** WARNING ** SEAICE_READPARMS: will use AdvScheme = 2 for HEFF  without any diffusion\n** WARNING ** SEAICE_READPARMS: will use AdvScheme = 2 for AREA  without any diffusion\n** WARNING ** SEAICE_READPARMS: will use AdvScheme = 2 for HSNOW without any diffusion\n** WARNING ** SEAICE_READPARMS: since DIFF1 is set to 0 (= new DIFF1 default value)\n MDS_READ_FIELD: filename: adxx_theta.0000000000.data\n MDS_READ_FIELD: File does not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_theta.0000000000.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: File does not exist']),
    (('offline_exf_seaice', 'input_ad'), 'xx_atemp', 27855986,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n MDS_READ_FIELD: filename: adxx_atemp.0000000000 , adxx_atemp.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_atemp.0000000000 , adxx_atemp.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('offline_exf_seaice', 'input_ad.thsice'), 'xx_atemp', 27855986,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n** Warning ** SEAICE_READPARMS: ignores "data.seaice" file since\n** Warning ** SEAICE_READPARMS: useSEAICE= F (set from "data.pkg")\n** Warning ** DIAGNOSTICS_READPARMS: ignores "data.diagnostics" file since\n** Warning ** DIAGNOSTICS_READPARMS: useDiagnostics= F (set from "data.pkg")\n MDS_READ_FIELD: filename: adxx_atemp.0000000000 , adxx_atemp.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_atemp.0000000000 , adxx_atemp.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('lab_sea', 'input_ad'), 'xx_atemp', 27855987,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n MDS_READ_FIELD: filename: ./ctrl_variables/adxx_atemp.0000000000 , ./ctrl_variables/adxx_atemp.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: ./ctrl_variables/adxx_atemp.0000000000 , ./ctrl_variables/adxx_atemp.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('lab_sea', 'input_ad.noseaice'), 'xx_atemp', 27855987,
     '** Warning ** DWNSLP_READPARMS: ignores "data.down_slope" file since\n** Warning ** DWNSLP_READPARMS: useDOWN_SLOPE= F (set from "data.pkg")\n** Warning ** SEAICE_READPARMS: ignores "data.seaice" file since\n** Warning ** SEAICE_READPARMS: useSEAICE= F (set from "data.pkg")\nSTOP ABNORMAL END: S/R MDS_READ_FIELD\n MDS_READ_FIELD: filename: ./ctrl_variables/adxx_atemp.0000000000 , ./ctrl_variables/adxx_atemp.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: ./ctrl_variables/adxx_atemp.0000000000 , ./ctrl_variables/adxx_atemp.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('lab_sea', 'input_ad.noseaicedyn'), 'xx_salt', 27855987,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n MDS_READ_FIELD: filename: ./ctrl_variables/adxx_salt.0000000000 , ./ctrl_variables/adxx_salt.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: ./ctrl_variables/adxx_salt.0000000000 , ./ctrl_variables/adxx_salt.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    # global_ocean.cs32x15 sea-ice code_ad variants (job 27855988; input_ad.seaice restarts from pickup.0000036000
    # without the AB3 fields, as input_ad)
    (('global_ocean.cs32x15', 'input_ad.seaice'), 'xx_theta', 27855988,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n** WARNING ** PACKAGES_BOOT: useCAL no longer set to T when using EXF (useEXF=T)\n** WARNING ** PACKAGES_BOOT:  as it used to be before checkpoint66d (2017/02/13)\n** WARNING ** PACKAGES_BOOT: To continue to use pkg/cal with EXF, need to add:\n** WARNING ** PACKAGES_BOOT: > useCAL=.TRUE., < in file "data.pkg"\n** Warning ** CAL_READPARMS: ignores "data.cal" file since\n** Warning ** CAL_READPARMS: useCAL= F (set from "data.pkg")\nREAD_MFLDS_CHECK: reading from file: pickup.0000036000\nREAD_MFLDS_CHECK: which contains   11 fields :\n >Uvel    < >GuNm1   < >Vvel    < >GvNm1   < >Theta   < >GtNm1   < >Salt    < >GsNm1   < >EtaN    < >dEtaHdt < >EtaH    <\nREAD_MFLDS_CHECK:    2 field(s) is/are missing :\n >GuNm2   < >GvNm2   <\n** WARNING ** CHECK_PICKUP: Will get only an approximated Restart\n Continue with mom_StartAB =         1 ; nHydStartAB =     36000\n          with tempStartAB =     36000 ; saltStartAB =     36000\n MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('global_ocean.cs32x15', 'input_ad.seaice_dynmix'), 'xx_theta', 27855988,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n** WARNING ** PACKAGES_BOOT: useCAL no longer set to T when using EXF (useEXF=T)\n** WARNING ** PACKAGES_BOOT:  as it used to be before checkpoint66d (2017/02/13)\n** WARNING ** PACKAGES_BOOT: To continue to use pkg/cal with EXF, need to add:\n** WARNING ** PACKAGES_BOOT: > useCAL=.TRUE., < in file "data.pkg"\n** Warning ** DIAGNOSTICS_READPARMS: ignores "data.diagnostics" file since\n** Warning ** DIAGNOSTICS_READPARMS: useDiagnostics= F (set from "data.pkg")\n MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
    (('global_ocean.cs32x15', 'input_ad.thsice'), 'xx_theta', 27855988,
     'STOP ABNORMAL END: S/R MDS_READ_FIELD\n** WARNING ** PACKAGES_BOOT: useCAL no longer set to T when using EXF (useEXF=T)\n** WARNING ** PACKAGES_BOOT:  as it used to be before checkpoint66d (2017/02/13)\n** WARNING ** PACKAGES_BOOT: To continue to use pkg/cal with EXF, need to add:\n** WARNING ** PACKAGES_BOOT: > useCAL=.TRUE., < in file "data.pkg"\n** Warning ** DIAGNOSTICS_READPARMS: ignores "data.diagnostics" file since\n** Warning ** DIAGNOSTICS_READPARMS: useDiagnostics= F (set from "data.pkg")\n MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data\n MDS_READ_FIELD: Files DO not exist\n',
     ['(PID.TID 0000.0001)  MDS_READ_FIELD: filename: adxx_theta.0000000000 , adxx_theta.0000000000.001.001.data', '(PID.TID 0000.0001)  MDS_READ_FIELD: Files DO not exist']),
]
for _key, _name, _job, _tr, _tail in M4_STOPS:
    EXPECTED_STOPS[_key] = {
        "why": f"forward-only code_ad build: GRDCHK_MAIN reads ad{_name} (MDS_READ_FIELD), which only an adjoint run "
               f"writes; the STDOUT is complete up to the gradient check (measured: job {_job})",
        "exit": 0, "tr_log": _tr, "stdout_tail": list(_tail),
        "stdout_has": list(EXPECTED_STOPS[("tutorial_global_oce_optim", "input_ad")]["stdout_has"])}


def stop_key(man):
    """EXPECTED_STOPS key of a run from its MANIFEST.json: (experiment, input dir) plus, for a run with namelist
    overlays, the sorted `KEY=VALUE` of its overlays (an overlay can change the stderr lines before the stop, e.g. a
    package's warnings); classify falls back to (experiment, input dir) when no overlay-specific stop is declared."""
    ov = tuple(sorted(f"{o['key']}={o['value']}" for o in man.get("overlays") or []))
    return (man.get("experiment"), man.get("input_dir")) + ov


def classify(top, rc):
    """-> (verdict, message): OK, EXPECTED-STOP or FAIL."""
    top = Path(top)
    out = top / "rundir" / "output.txt"
    log = top / "rundir" / "run.tr_log"
    text = out.read_text(encoding="latin-1") if out.is_file() else ""
    lines = text.rstrip("\n").split("\n") if text else []
    normal = sum(NORMAL_END in ln for ln in lines[-10:])
    if rc == 0 and normal >= 1:
        return "OK", f"{out}"
    man = json.loads((top / "MANIFEST.json").read_text()) if (top / "MANIFEST.json").is_file() else {}
    exp = EXPECTED_STOPS.get(stop_key(man), EXPECTED_STOPS.get((man.get("experiment"), man.get("input_dir"))))
    if exp is not None:
        problems = []
        if rc != exp["exit"]:
            problems.append(f"exit {rc} != {exp['exit']}")
        tr = log.read_text(encoding="latin-1") if log.is_file() else None
        if tr != exp["tr_log"]:
            problems.append("run.tr_log differs from the declared stop")
        if lines[-len(exp["stdout_tail"]):] != exp["stdout_tail"]:
            problems.append("STDOUT does not end with the declared lines")
        problems += [f"STDOUT lacks {s!r}" for s in exp["stdout_has"] if s not in text]
        if not problems:
            return "EXPECTED-STOP", f"{out}: {exp['why']}"
        return "FAIL", f"{top} (exit {rc}, normal end lines {normal}; not the declared stop: {'; '.join(problems)})"
    return "FAIL", f"{top} (exit {rc}, normal end lines {normal})"


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        print("usage: run_verdict.py RUN_TOP EXIT_CODE")
        return 2
    verdict, msg = classify(a[0], int(a[1]))
    print(f"RUN {verdict} {msg}")
    return {"OK": 0, "EXPECTED-STOP": 3}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
