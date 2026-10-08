#!/usr/bin/env python3
"""The Fortran oracle runs the project relies on, their checks, and the yardstick (plan Task 3b).

    reference_runs.py --check            check every registered run (files, binary sha256, recorded sha256)
    reference_runs.py --lock             write reference/reference_runs.lock.json (sha256 of every reference file)
    reference_runs.py --docs             write docs/REFERENCE_RUNS.md and docs/YARDSTICK.md from the runs

Registry: RUNS below (variant, run id, build, job, purpose). Builds: BUILDS (frozen binaries
$MJX_REFERENCE/bin/<exp>-<code>-63cdc0b-<tag>, reference/build.sh). A run is checked by `check_run`: run directory,
READY, MANIFEST.json naming the expected binary, the MANIFEST's binary sha256 equal to the bin directory's sha256 file
and to the sha256 of the frozen mitgcmuv, run_provenance.txt with exit 0, STDOUT present, and every file recorded in
the lock file with the same sha256 (STDOUT of every run; the copied adjoint control files of the FD runs; for the
dumps-on runs the list of dump files with their sizes), and for a dumps-on run the dumped iterations
(dumps/jaxdump_info.txt) equal to the registered JAXDUMP_STEPS. A missing file or a different hash is a problem, never
a skip.

Dumps-on runs (DUMP_KINDS): `jdon` (build jaxdump, Task 4: the oracle of the substep gates, one per variant), `jdon2`
(build jaxdump2: the extended G00 groups G/V of the core lane), `jdon3` (build jaxdump3: initialisation stages
I00-I02, rStarDhCDt at S12_calc_rstar, mixed signed-zero probes; steps reaching a later MONITOR block for the advect
variants). Each comes with its dumps-off (`jdoff*`) and plain (`jdplain*`; the job27826873 plain run is the
`yardstick`) runs of the same job: the invisibility triple of reference/jobs/jaxdump_runs.sbatch.
FTZ oracle (plan decision 13; FTZ_VARIANTS): `ftz_plain`, `ftz_jdoff`, `ftz_jdon` = the invisibility triple of the
FTZ/DAZ variants of the builds (reference/relink_ftz.sh: crtfastmath.o linked, no object recompiled); the oracle of the
gates that meet subnormals, never a yardstick; their written pickups go into the lock; the band measurement against
the standard triple (reference/ftz_band.py) lies next to the runs.
Restart pair (`restart_a`, `restart_b`; reference/jobs/restart_runs.sbatch): the Fortran oracle's own restart, run A
with permanent pickups and run B restarted from A's pickup, both dumps on (jaxdump2 binary); gate fixtures with
namelist overlays (B also with the copied pickup), never yardsticks; their written pickup files go into the lock.

Yardstick (docs/YARDSTICK.md): tools/testreport_jax.py digits of the yardstick run's STDOUT vs the variant's results/
file for every check-list variable (testreport's own list for the variant), plus for tutorial_global_oce_optim/input_ad
the adm list of the FD run (option (i): zero adxx_qnet, reference/grdchk_adxx.py) and testreport's default forward list
of the forward-only build vs the forward %MON lines of results/output_adm.txt. Stdlib only; machine paths from
mitjax/paths.py (loaded by path).
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paths = _load("_mjx_paths", REPO / "mitjax" / "paths.py")
trj = _load("_mjx_testreport_jax", REPO / "tools" / "testreport_jax.py")
cmp_runs = _load("_mjx_compare_runs", REPO / "reference" / "compare_runs.py")

UPSTREAM_SHORT = "63cdc0b"
LOCK = REPO / "reference" / "reference_runs.lock.json"

# build tag -> (binary name suffix, build job, what)
BUILDS = {
    "plain": ("f48c062", 27826031, "plain build, testreport's options (-ieee = -O0, no MPI), Task 3a"),
    "jaxdump": ("5f16129-jaxdump", 27826872, "instrumented build (jaxdump shim), Task 4"),
    "retile": ("d21b643-retile", 27827362, "second tiling: reference/retile/<exp>-<code>/SIZE.h, Task 3b"),
    "jaxdump2": ("b73c05c-jaxdump", 27828735, "instrumented build with the extended G00 groups G/V (kSurf*, kLowC, "
                 "topoZ, cosFac*, deepFac*, hybrid sigma, rkSign, gravitySign), core lane Task 11"),
    "jaxdump3": ("275a02b-jaxdump", 27829310, "instrumented build: initialisation stages I00_pickup_read, "
                 "I01_read_pickup, I02_ini_fields; rStarDhCDt in group r (S12_calc_rstar); mixed signed-zero "
                 "exchange probes zu*/zv*, lane A session 4"),
    "plain_m2": ("06b4417", 27832154, "plain builds of the M2 codes, testreport's options (as Task 3a), lane A "
                 "session 6"),
    "jaxdump_m2": ("8316e95-jaxdump", 27832225, "instrumented builds of the M2 codes (jaxdump3 stages + group P "
                   "ptracers, stage T04_ptracers_integrate), lane A session 6"),
    "plain_m2b": ("78ca390", 27832600, "plain builds of global_ocean.cs32x15/code_ad (forward only) and "
                  "adjustment.cs-32x32x1/code_min (genmake2 -standarddirs eesupp, its README), testreport's options "
                  "otherwise, lane A session 7"),
    "jaxdump_m2b": ("78ca390-jaxdump", 27832600, "instrumented build of global_ocean.cs32x15/code_ad (the jaxdump_m2 "
                    "instrumentation), lane A session 7"),
    "plain_m3": ("463504e", 27840380, "plain builds of the M3 codes (vermix, front_relax, ideal_2D_oce, "
                 "tutorial_reentrant_channel, MLAdjust), testreport's options, lane A session 9"),
    "jaxdump_m3": ("463504e-jaxdump", 27840380, "instrumented builds of the M3 codes and of global_ocean.90x40x15/code "
                   "(column-mixing groups K/Q/Y, IDEMIX in k; stages P07-P11, T05_opps, T06_convective_adjustment), "
                   "lane A session 9"),
    "plain_m4": ("704fd6b", 27855975, "plain builds of the M4 codes (1D_ocean_ice_column, offline_exf_seaice, "
                 "seaice_itd, lab_sea; code and code_ad forward only), testreport's options, lane A session 11"),
    "jaxdump_m4": ("704fd6b-jaxdump", 27855975, "instrumented builds of the M4 codes and of global_ocean.cs32x15 "
                   "code/code_ad (EXF X01-X06, sea ice I00-I04/Y01-Y09/P13, thsice P12/I02a, salt plume P14), lane A "
                   "session 11"),
    "jaxdump_m4b": ("1ac79cb-jaxdump", 27856045, "instrumented builds of 1D_ocean_ice_column code/code_ad with the "
                    "B-grid sea-ice stages I00b_seaice_begin/I01b_dynsolver and group B, lane A session 11"),
    # plan decision 13 (FTZ oracle): the objects of the standard build, linked with gfortran 11.2.0's crtfastmath.o
    # (reference/relink_ftz.sh; ftz_provenance.txt in the bin directory: objects identical, the standard relink
    # reproduces the source binary, MXCSR DAZ+FZ at MAIN__, the FTZ probe)
    "plain_m2_ftz": ("06b4417-ftz", 27878830, "FTZ/DAZ variant of plain_m2 (global_ocean.cs32x15/code): crtfastmath.o "
                     "linked, no object recompiled, lane A session 12"),
    "jaxdump_m4_ftz": ("704fd6b-jaxdump-ftz", 27878830, "FTZ/DAZ variant of jaxdump_m4 (global_ocean.cs32x15/code): "
                       "crtfastmath.o linked, no object recompiled, lane A session 12"),
}
# FTZ build -> its standard source build
FTZ_SOURCE = {"plain_m2_ftz": "plain_m2", "jaxdump_m4_ftz": "jaxdump_m4"}
DUMP_KINDS = ("jdon", "jdon2", "jdon3", "ctrlzero", "ctrlxx", "restart_a", "restart_b", "ftz_jdon")
# namelist overlays (make_rundir.py --set, recorded in MANIFEST.json) each kind must have; every other kind has none
OVERLAYS = {"debug": (("eedata", "EEPARMS", "debugMode", ".TRUE."),),
            "diagoff": (("data.pkg", "PACKAGES", "useDiagnostics", ".FALSE."),),
            "mncoff": (("data.pkg", "PACKAGES", "useMNC", ".FALSE."),),
            "ctrlxx": (("data.ctrl", "CTRL_NML", "doInitXX", ".FALSE."), ("data.ctrl", "CTRL_NML", "doMainUnpack",
                                                                          ".FALSE.")),
            # the Fortran restart pair (reference/jobs/restart_runs.sbatch): A writes a permanent pickup every 5 steps,
            # B is the restart from A's pickup.0000000005 (a restarted run's data: nIter0=5, nTimeSteps=5)
            "restart_a": (("data", "PARM03", "pChkptFreq", "6000.0"),),
            "restart_b": (("data", "PARM03", "pChkptFreq", "6000.0"), ("data", "PARM03", "nIter0", "5"),
                          ("data", "PARM03", "nTimeSteps", "5")),
            }
# kinds whose run directory holds copied input files (make_rundir.py --copy); their sha256 go into the lock
COPY_KINDS = ("fdzero", "fdrandom", "ctrlxx", "restart_b")
# kinds that must never serve as a yardstick or a plain oracle run: planted inputs and namelist overlays
FIXTURE_KINDS = ("ctrlxx", "restart_a", "restart_b")
# kinds whose written pickup files (rundir/pickup.*, regular files) go into the lock: the restart pair's state
RESTART_KINDS = ("restart_a", "restart_b")
RESTART_JOB, RESTART_STEPS = 27831497, (5, 6, 7, 8, 9)

# the M1 variants: (experiment, input dir, code dir); one build per code dir serves its variants
VARIANTS = [
    ("tutorial_barotropic_gyre", "input", "code"),
    ("tutorial_baroclinic_gyre", "input", "code"),
    ("advect_xy", "input", "code"),
    ("advect_xy", "input.ab3_c4", "code"),
    ("advect_xz", "input", "code"),
    ("advect_xz", "input.nlfs", "code"),
    ("advect_xz", "input.pqm", "code"),
    ("global_ocean.90x40x15", "input", "code"),
    ("tutorial_global_oce_optim", "input_ad", "code_ad"),
]
OPTIM = ("tutorial_global_oce_optim", "input_ad")
# the M2 variants (plan M2; lane A session 6): plain build plain_m2, instrumented build jaxdump_m2; one invisibility
# triple per variant (reference/jobs/jaxdump_runs.sbatch: plain = the yardstick, jdoff, jdon with the default dump
# steps nIter0..nIter0+2); the code_ad variants are forward-only builds with the FD oracle runs of Task 3b
# (zero / random adxx file of the variant's control, reference/grdchk_adxx.py)
M2_VARIANTS = [
    ("adjustment.cs-32x32x1", "input", "code"),
    ("adjustment.cs-32x32x1", "input.nlfs", "code"),
    ("solid-body.cs-32x32x1", "input", "code"),
    ("advect_cs", "input", "code"),
    ("global_ocean.cs32x15", "input", "code"),
    ("global_ocean.cs32x15", "input.viscA4", "code"),
    ("global_ocean.cs32x15", "input.in_p", "code"),
    ("global_ocean.90x40x15", "input_ad", "code_ad"),
    ("global_ocean.90x40x15", "input_ad.kapgm", "code_ad"),
    ("global_ocean.90x40x15", "input_ad.kapredi", "code_ad"),
    ("global_ocean.90x40x15", "input_ad.bottomdrag", "code_ad"),
    ("tutorial_global_oce_latlon", "input", "code"),
    ("tutorial_advection_in_gyre", "input", "code"),
    ("tutorial_tracer_adjsens", "input_ad", "code_ad"),
    ("tutorial_tracer_adjsens", "input_ad.som81", "code_ad"),
    # lane A session 7: data.pkg leaves useSEAICE at its default .FALSE. (packages_boot.F:145) and sets useTHSICE,
    # useEXF .FALSE. (seaice, thsice, exf compiled but not used; STDOUT "compiled but not used"); its sea-ice
    # siblings input_ad.seaice*/thsice are M4
    ("global_ocean.cs32x15", "input_ad", "code_ad"),
]
# the M3 variants (plan Task 28; lane A session 9): plain build plain_m3 (global_ocean.90x40x15's input.dwnslp and
# input.idemix: the M1 plain build of its code), instrumented build jaxdump_m3; one invisibility triple per variant
# (reference/jobs/jaxdump_runs.sbatch at 463504e: plain = the yardstick, jdoff, jdon with steps nIter0..nIter0+2)
M3_VARIANTS = [("vermix", i, "code") for i in ("input", "input.dd", "input.ggl90", "input.gglLC", "input.my82",
                                                "input.opps", "input.pp81")]
M3_VARIANTS += [("front_relax", i, "code") for i in ("input", "input.bvp", "input.in_p", "input.mxl", "input.top")]
M3_VARIANTS += [("ideal_2D_oce", "input", "code"), ("ideal_2D_oce", "input.geom", "code"),
                ("tutorial_reentrant_channel", "input", "code")]
M3_VARIANTS += [("MLAdjust", i, "code") for i in ("input", "input.A4FlxF", "input.AhFlxF", "input.AhStTn",
                                                   "input.AhVrDv", "input.QGLeith", "input.QGLthGM")]
M3_VARIANTS += [("global_ocean.90x40x15", "input.dwnslp", "code"), ("global_ocean.90x40x15", "input.idemix", "code")]
M3_BUILD = {(e, i): ("plain" if e == "global_ocean.90x40x15" else "plain_m3", "jaxdump_m3") for e, i, _ in M3_VARIANTS}
M3_TRIPLE = {(e, i): (27840385 if e in ("vermix", "front_relax") else 27840387 if e == "MLAdjust" else 27840386)
             for e, i, _ in M3_VARIANTS}
# the M4 variants (plan M4; lane A session 11): plain build plain_m4, instrumented build jaxdump_m4
# (1D_ocean_ice_column: jaxdump_m4b, the B-grid stages; global_ocean.cs32x15: the M2 plain binaries plain_m2 /
# plain_m2b); one invisibility triple per variant (reference/jobs/jaxdump_runs.sbatch: plain = the yardstick, jdoff,
# jdon with steps nIter0..+2).
# offline_exf_seaice/input_ad.obcs is M5 (useOBCS). The code_ad variants are forward-only builds with FD oracle runs.
M4_VARIANTS = [("1D_ocean_ice_column", "input", "code"), ("1D_ocean_ice_column", "input_ad", "code_ad")]
M4_VARIANTS += [("offline_exf_seaice", i, "code") for i in (
    "input", "input.thermo", "input.dyn_lsr", "input.dyn_ellnnfr", "input.dyn_jfnk", "input.dyn_mce",
    "input.dyn_paralens", "input.dyn_teardrop", "input.thsice")]
M4_VARIANTS += [("offline_exf_seaice", i, "code_ad") for i in ("input_ad", "input_ad.thsice")]
M4_VARIANTS += [("lab_sea", i, "code") for i in ("input", "input.fd", "input.hb87", "input.longstep", "input.natl_box",
                                                  "input.salt_plume")]
M4_VARIANTS += [("lab_sea", i, "code_ad") for i in ("input_ad", "input_ad.noseaice", "input_ad.noseaicedyn")]
M4_VARIANTS += [("seaice_itd", i, "code") for i in ("input", "input.lipscomb07", "input.thermo")]
M4_VARIANTS += [("global_ocean.cs32x15", i, "code") for i in ("input.seaice", "input.icedyn", "input.thsice")]
M4_VARIANTS += [("global_ocean.cs32x15", i, "code_ad") for i in ("input_ad.seaice", "input_ad.seaice_dynmix",
                                                                  "input_ad.thsice")]
M4_BUILD = {(e, i): (("plain_m2" if c == "code" else "plain_m2b") if e == "global_ocean.cs32x15" else "plain_m4",
                     "jaxdump_m4b" if e == "1D_ocean_ice_column" else "jaxdump_m4") for e, i, c in M4_VARIANTS}
# invisibility-triple job per M4 variant (sacct-checked; jaxdump_runs.sbatch at 704fd6b, 1D_ocean_ice_column at
# 3e077ed): 27855986 1D (superseded by 27856057, B-grid build; not registered) + offline_exf_seaice thermo/dyn_lsr/
# input/input_ad*, 27855987 lab_sea, 27855988 cs32x15, 27855989 seaice_itd + the other offline_exf_seaice variants.
# 27855986-27855988 are FAILED in sacct only because the code_ad grdchk stops were not yet declared (declared in
# run_verdict.py from these runs: 3e077ed, 03f6b85); all comparisons INVISIBLE.
M4_TRIPLE = {}
for _e, _i, _c in M4_VARIANTS:
    M4_TRIPLE[(_e, _i)] = (27856057 if _e == "1D_ocean_ice_column" else 27855987 if _e == "lab_sea" else
                           27855988 if _e == "global_ocean.cs32x15" else 27855989 if _e == "seaice_itd" or _i in (
                               "input.dyn_ellnnfr", "input.dyn_jfnk", "input.dyn_mce", "input.dyn_paralens",
                               "input.dyn_teardrop", "input.thsice") else 27855986)
# FD oracle runs of the M4 code_ad variants: (job, control); zero and random ad<control> files written by
# reference/grdchk_adxx.py from the triple's plain run into $MJX_REFERENCE/grdchk_adxx/d018053-<exp>-<input>-{zero,
# random20261001} (lab_sea: from and into ctrlDir ./ctrl_variables), copied with make_rundir --copy
# (reference/jobs/oracle_runs.sbatch at d018053)
M4_FD = {("1D_ocean_ice_column", "input_ad"): (27856073, "xx_theta"),
         ("offline_exf_seaice", "input_ad"): (27856073, "xx_atemp"),
         ("offline_exf_seaice", "input_ad.thsice"): (27856073, "xx_atemp"),
         ("lab_sea", "input_ad"): (27856073, "xx_atemp"),
         ("lab_sea", "input_ad.noseaice"): (27856073, "xx_atemp"),
         ("lab_sea", "input_ad.noseaicedyn"): (27856073, "xx_salt"),
         ("global_ocean.cs32x15", "input_ad.seaice"): (27856074, "xx_theta"),
         ("global_ocean.cs32x15", "input_ad.seaice_dynmix"): (27856074, "xx_theta"),
         ("global_ocean.cs32x15", "input_ad.thsice"): (27856074, "xx_theta")}
# FTZ oracle variants (plan decision 13, Nikolay 2026-10-04; lane A session 12). XLA:CPU computes with FTZ/DAZ set
# (L-CONF-2; no XLA flag clears them), gfortran keeps subnormals: where a gate meets subnormals the port is gated
# bitwise against the FTZ variant of the same builds (FTZ_SOURCE). One invisibility triple per variant
# (reference/jobs/jaxdump_runs.sbatch with the -ftz binaries): kinds ftz_plain, ftz_jdoff, ftz_jdon (dumps at
# nIter0..+2). Never yardsticks: the standard oracle stays the reference. Its difference from the FTZ oracle is
# measured by reference/ftz_band.py (reference/jobs/ftz_band.sbatch: ftzband-job<triple>-{plain,jdon}.{txt,json} next
# to the runs) and must stay below 2**-1021. (variant) -> (triple job, plain build, jaxdump build, band job, standard
# triple job)
FTZ_VARIANTS = {("global_ocean.cs32x15", "input.seaice"): (27878831, "plain_m2_ftz", "jaxdump_m4_ftz", 27878832,
                                                           27855988)}
FTZ_KINDS = ("ftz_plain", "ftz_jdoff", "ftz_jdon")


def ftz_band_file(exp, inp, mode, suffix="txt"):
    """Path of the FTZ-vs-standard band measurement of a variant (mode plain or jdon)."""
    return paths.REFERENCE_RUNS / exp / inp / f"ftzband-job{FTZ_VARIANTS[(exp, inp)][0]}-{mode}.{suffix}"


# builds of the M2 variants added in session 7 (default: plain_m2, jaxdump_m2)
M2_BUILD = {("global_ocean.cs32x15", "input_ad"): ("plain_m2b", "jaxdump_m2b")}
# adjustment.cs-32x32x1's "minimal" case (plan Task 25; its README): code_min compiles eesupp + exch2 + debug only and
# its main.F skips THE_MODEL_MAIN, so the run sets up the execution environment and the W2 topology and ends; no %MON,
# no results/ file (testreport does not run it), no jaxdump build (nothing of model/src). One plain run, kind
# `minimal` (reference/jobs/oracle_runs.sbatch at 78ca390, job 27832625).
M2_MIN = [("adjustment.cs-32x32x1", "input_min", "code_min")]
M2_MIN_JOB = 27832625
# invisibility-triple job per M2 variant (sacct-checked; reference/jobs/jaxdump_runs.sbatch at 8316e95)
M2_TRIPLE = {(e, i): (27832226 if e in ("adjustment.cs-32x32x1", "solid-body.cs-32x32x1", "advect_cs") else
                      27832228 if e == "global_ocean.cs32x15" else 27832229 if c == "code_ad" else 27832231)
             for e, i, c in M2_VARIANTS}
# input.in_p: job 27832228's directories lacked prepare_run's sibling inputs (../input.icedyn, ../input.seaice; the
# model stopped reading data.exf) -> rerun after the make_rundir fix 30695ea (job27832228-* in_p: not registered)
M2_TRIPLE[("global_ocean.cs32x15", "input.in_p")] = 27832348
# session 7 (reference/jobs/jaxdump_runs.sbatch at 78ca390): FAILED in sacct only because the grdchk stop was not yet
# declared; the stop measured in its three runs is declared in run_verdict.py (d117407), all three EXPECTED-STOP
M2_TRIPLE[("global_ocean.cs32x15", "input_ad")] = 27832624
# invisibility verdict files that are not invisibility-job<job>.txt: cs32x15/input re-checked with the
# DIAGSTATS_UNSET_REGIONS exemption (reference/jaxdump/invisibility.py at 30b10ae; the job's own verdict file says
# VISIBLE for the unset regions of dynStDiag.0000072000.t001.nc and stays as it is)
M2_INVISIBILITY = {("global_ocean.cs32x15", "input"): "invisibility-job27832228-r30b10ae.txt"}


def invisibility_file(exp, inp, ftz=False):
    """Path of the invisibility verdict of an M2, M3 or M4 variant's triple (ftz: of its FTZ triple)."""
    if ftz:
        return paths.REFERENCE_RUNS / exp / inp / f"invisibility-job{FTZ_VARIANTS[(exp, inp)][0]}.txt"
    job = M2_TRIPLE.get((exp, inp)) or M3_TRIPLE.get((exp, inp)) or M4_TRIPLE[(exp, inp)]
    name = M2_INVISIBILITY.get((exp, inp), f"invisibility-job{job}.txt")
    return paths.REFERENCE_RUNS / exp / inp / name


# FD oracle runs of the M2 code_ad variants: (job, control name); zero and random ad<name> files written by
# reference/grdchk_adxx.py from the job27832229-plain runs into $MJX_REFERENCE/grdchk_adxx/30b10ae-<exp>-<input>-{zero,
# random20261001}, copied with make_rundir --copy (reference/jobs/oracle_runs.sbatch at 30b10ae)
M2_FD = {("global_ocean.90x40x15", "input_ad"): (27832347, "xx_theta"),
         ("global_ocean.90x40x15", "input_ad.kapgm"): (27832347, "xx_kapgm"),
         ("global_ocean.90x40x15", "input_ad.kapredi"): (27832347, "xx_kapredi"),
         ("global_ocean.90x40x15", "input_ad.bottomdrag"): (27832347, "xx_bottomdrag"),
         ("tutorial_tracer_adjsens", "input_ad"): (27832347, "xx_ptr1"),
         ("tutorial_tracer_adjsens", "input_ad.som81"): (27832347, "xx_ptr1"),
         # session 7: adxx files from job27832624-plain into $MJX_REFERENCE/grdchk_adxx/d117407-<exp>-<input>-{zero,
         # random20261001} (12 tiled files + meta), reference/jobs/oracle_runs.sbatch at d117407
         ("global_ocean.cs32x15", "input_ad"): (27832649, "xx_theta")}
# forward-only code_ad variants with an FD oracle (fdzero run, results/output_adm*.txt)
AD_VARIANTS = (OPTIM,) + tuple((e, i) for e, i, c in M2_VARIANTS + M4_VARIANTS if c == "code_ad")
NITER0 = {("global_ocean.90x40x15", "input"): 36000,     # data, PARM03 nIter0 (others start at 0)
          ("global_ocean.cs32x15", "input"): 72000, ("global_ocean.cs32x15", "input.viscA4"): 86400,
          ("global_ocean.cs32x15", "input_ad"): 72000,
          ("ideal_2D_oce", "input"): 36000, ("ideal_2D_oce", "input.geom"): 36000, ("MLAdjust", "input.A4FlxF"): 36,
          ("global_ocean.90x40x15", "input.dwnslp"): 36000,
          ("tutorial_advection_in_gyre", "input"): 259200,
          # M4 (measured: dumps/jaxdump_info.txt of the jdon runs)
          **{("global_ocean.cs32x15", i): 36000 for i in ("input.seaice", "input.icedyn", "input.thsice",
                                                          "input_ad.seaice", "input_ad.seaice_dynmix",
                                                          "input_ad.thsice")},
          **{("lab_sea", i): 1 for i in ("input", "input.fd", "input.salt_plume")}}
# jaxdump3 runs (build 275a02b): job and JAXDUMP_STEPS per variant. advect_xz (all three) and advect_xy/input.ab3_c4
# print a %MON block every 10 steps (monitorFreq 12000 / deltaT 1200; dumpFreq 27500 / deltaT 2750 as monitorFreq,
# ini_parms.F:1187-1196), advect_xy/input every 16 (dumpFreq 40000 / deltaT 2500): the block of iteration n+1 is
# written in the step that starts at n, so steps 9 and 15 hold it. global_ocean: default steps (36000-36002).
JD3 = {("advect_xz", "input"): (27829313, (0, 1, 2, 9)), ("advect_xz", "input.nlfs"): (27829313, (0, 1, 2, 9)),
       ("advect_xz", "input.pqm"): (27829313, (0, 1, 2, 9)), ("advect_xy", "input.ab3_c4"): (27829313, (0, 1, 2, 9)),
       ("advect_xy", "input"): (27829314, (0, 1, 2, 15)),
       ("global_ocean.90x40x15", "input"): (27829315, (36000, 36001, 36002))}
# variants with diagnostics / mnc switched on in data.pkg (the output-request invariance runs, item 3b)
OUTPUT_OFF = {("tutorial_baroclinic_gyre", "input"): ("diagoff", "mncoff"), ("advect_xz", "input.nlfs"): ("diagoff",),
              ("global_ocean.90x40x15", "input"): ("diagoff",)}
# M2 (lane A session 8; brainstorm section 3: diagnostics and mnc are not ported, gate = the Fortran's model numbers do
# not change): every M2 variant whose STDOUT says "pkg/mnc compiled and used" (useMNC=T: cs32x15/input,
# advection_in_gyre, 90x40x15/input_ad.bottomdrag) or "pkg/diagnostics compiled and used", measured on the registered
# plain runs; reference/jobs/oracle_runs.sbatch at 7ff3e89 with --set data.pkg:PACKAGES:use...=.FALSE.
OUTPUT_OFF_M2 = {("global_ocean.cs32x15", "input"): (("mncoff", "diagoff"), 27833806),
                 ("tutorial_advection_in_gyre", "input"): (("mncoff", "diagoff"), 27833806),
                 ("adjustment.cs-32x32x1", "input"): (("diagoff",), 27833806),
                 ("solid-body.cs-32x32x1", "input"): (("diagoff",), 27833806),
                 ("advect_cs", "input"): (("diagoff",), 27833806),
                 ("global_ocean.cs32x15", "input.viscA4"): (("diagoff",), 27833806),
                 ("global_ocean.cs32x15", "input.in_p"): (("diagoff",), 27833806),
                 ("global_ocean.90x40x15", "input_ad.bottomdrag"): (("mncoff",), 27833807),
                 ("global_ocean.cs32x15", "input_ad"): (("diagoff",), 27833807)}
OUTPUT_OFF_WHAT = {"diagoff": "useDiagnostics=.FALSE.", "mncoff": "useMNC=.FALSE."}
# planted first-guess controls of the global_ocean.90x40x15 code_ad family (lane A session 8; as job27829159-ctrlxx for
# optim; the GOADK gates S02 + control = S03): (job, control name). Seeded uniform values (seed 20261001) written by
# reference/grdchk_adxx.py --prefix '' --random 20261001 --amp A from the job27832229-plain runs into
# $MJX_REFERENCE/ctrl_planted/0542f7a-<exp>-<input>-<name>-seed20261001 (A: xx_theta 0.5 K, xx_kapgm and xx_kapredi
# 100 m^2/s, xx_bottomdrag 2e-4), one global file each (useSingleCpuIO); read with the overlays data.ctrl doInitXX=F,
# doMainUnpack=F; jaxdump_m2 binary, dumps on at steps 0-3 (reference/jobs/oracle_runs.sbatch at 0542f7a)
M2_CTRLXX = {("global_ocean.90x40x15", "input_ad"): (27833840, "xx_theta"),
             ("global_ocean.90x40x15", "input_ad.kapgm"): (27833840, "xx_kapgm"),
             ("global_ocean.90x40x15", "input_ad.kapredi"): (27833840, "xx_kapredi"),
             ("global_ocean.90x40x15", "input_ad.bottomdrag"): (27833840, "xx_bottomdrag")}
CTRLXX_STEPS = (0, 1, 2, 3)


def code_of(exp, inp):
    return next(c for e, i, c in VARIANTS + M2_VARIANTS + M2_MIN + M3_VARIANTS + M4_VARIANTS
                if (e, i) == (exp, inp))


def bin_name(exp, code, build):
    return f"{exp}-{code}-{UPSTREAM_SHORT}-{BUILDS[build][0]}"


def _runs():
    runs = []

    def add(exp, inp, run_id, build, job, purpose, kind, ends_normally=True, steps=None):
        runs.append({"exp": exp, "input": inp, "run_id": run_id, "build": build, "job": job, "purpose": purpose,
                     "kind": kind, "ends_normally": ends_normally, "steps": steps})

    for exp, inp, code in VARIANTS:
        optim = (exp, inp) == OPTIM
        j = 27826998 if optim else 27826873
        end = not optim     # the forward-only optim build stops in GRDCHK_MAIN without an adxx file
        n0 = NITER0.get((exp, inp), 0)
        default = (n0, n0 + 1, n0 + 2)
        add(exp, inp, f"job{j}-plain", "plain", j, "yardstick: STDOUT vs results/ (docs/YARDSTICK.md)", "yardstick",
            end)
        add(exp, inp, f"job{j}-jdoff", "jaxdump", j, "jaxdump binary, dumps off (Task 4 invisibility)", "jdoff", end)
        add(exp, inp, f"job{j}-jdon", "jaxdump", j, "jaxdump binary, dumps on: substep dumps in <run>/dumps "
            "(oracle of the substep gates; Task 4 invisibility)", "jdon", end, default)
        j2 = 27828737
        add(exp, inp, f"job{j2}-plain", "plain", j2, "plain binary, invisibility triple of the jaxdump2 build "
            "(core lane, Task 11)", "jdplain2", end)
        add(exp, inp, f"job{j2}-jdoff", "jaxdump2", j2, "jaxdump2 binary, dumps off (invisibility)", "jdoff2", end)
        add(exp, inp, f"job{j2}-jdon", "jaxdump2", j2, "jaxdump2 binary, dumps on: G00 groups G/V extended "
            "(oracle of test_grid.py::test_grid_extra_fields_bitwise)", "jdon2", end, default)
        if (exp, inp) in JD3:
            j3, steps = JD3[(exp, inp)]
            add(exp, inp, f"job{j3}-plain", "plain", j3, "plain binary, invisibility triple of the jaxdump3 build",
                "jdplain3", end)
            add(exp, inp, f"job{j3}-jdoff", "jaxdump3", j3, "jaxdump3 binary, dumps off (invisibility)", "jdoff3",
                end)
            add(exp, inp, f"job{j3}-jdon", "jaxdump3", j3, "jaxdump3 binary, dumps on, JAXDUMP_STEPS="
                + ":".join(map(str, steps)) + ": initialisation stages I00-I02, rStarDhCDt at S12_calc_rstar, "
                "later MONITOR block (lane MON)", "jdon3", end, steps)
        if exp == "tutorial_barotropic_gyre":
            add(exp, inp, "job27826032", "plain", 27826032, "repeat of the plain run (Task 3a): run-to-run "
                "determinism", "repeat")
        else:
            add(exp, inp, "job27826871", "plain", 27826871, "repeat of the plain run (lane D, Task 6): run-to-run "
                "determinism", "repeat", end)
        add(exp, inp, "job27827363-debug", "plain", 27827363, "debugMode=.TRUE. overlay (eedata): call order "
            "reference/call_order/; not a yardstick", "debug", end)
        add(exp, inp, "job27827364-retile", "retile", 27827364, "second tiling (invariance check, docs/YARDSTICK.md)",
            "retile", end)
        for lab in OUTPUT_OFF.get((exp, inp), ()):
            what = OUTPUT_OFF_WHAT[lab]
            add(exp, inp, f"job27827363-{lab}", "plain", 27827363, f"{what} overlay (data.pkg): output-request "
                "invariance; not a yardstick", lab)
    for lab, what in (("fdzero", "zero adxx_qnet (option (i)): the FD gradient oracle and admCst/admFwd yardstick"),
                      ("fdrandom", "random adxx_qnet: shows the FD lines do not depend on the adxx file")):
        add(*OPTIM, f"job27827478-{lab}", "plain", 27827478, what, lab)
    # CTRL lane (forward-only optim with the jaxdump binary, dumps on, steps 0-9; both end in the declared grdchk stop,
    # run verdict EXPECTED-STOP). Job 27829092 is FAILED in sacct because of its other directory job27829092-ctrlxx
    # (exit 2 in ctrl unpacking; not registered, a deletion candidate of the main session).
    add(*OPTIM, "job27829092-ctrlzero", "jaxdump", 27829092, "CTRL lane: zero control (no overlay, no copied file), "
        "dumps on, JAXDUMP_STEPS=0:...:9: oracle of the ctrl gates", "ctrlzero", False, tuple(range(10)))
    add(*OPTIM, "job27829159-ctrlxx", "jaxdump", 27829159, "GATE FIXTURE, NEVER A YARDSTICK: planted nonzero "
        "xx_qnet (make_rundir --copy from $MJX_RUNS/ctrl/xx_planted-job27829159) read with the namelist overlays "
        "data.ctrl doInitXX=.FALSE., doMainUnpack=.FALSE.; dumps on, JAXDUMP_STEPS=0:...:9 (CTRL lane gates)",
        "ctrlxx", False, tuple(range(10)))
    # the Fortran oracle's own restart (lane A session 5): jaxdump2 binary, dumps on at the start iterations 5-9 of
    # both runs; reference/restart_compare.py output: restart-job<RESTART_JOB>.{txt,json} next to the runs
    exp, inp = "tutorial_barotropic_gyre", "input"
    add(exp, inp, f"job{RESTART_JOB}-restartA", "jaxdump2", RESTART_JOB, "GATE FIXTURE, NEVER A YARDSTICK: run A of "
        "the Fortran restart pair: 10 steps with the overlay data pChkptFreq=6000.0 (permanent pickups at 5 and 10), "
        "dumps on, JAXDUMP_STEPS=5:6:7:8:9 (scripts/tests/test_restart_oracle.py)", "restart_a", True, RESTART_STEPS)
    add(exp, inp, f"job{RESTART_JOB}-restartB", "jaxdump2", RESTART_JOB, "GATE FIXTURE, NEVER A YARDSTICK: run B of "
        "the Fortran restart pair: restart from run A's pickup.0000000005 (copied), overlays data pChkptFreq=6000.0, "
        "nIter0=5, nTimeSteps=5; dumps on, JAXDUMP_STEPS=5:6:7:8:9 (scripts/tests/test_restart_oracle.py)",
        "restart_b", True, RESTART_STEPS)
    # M2 (lane A session 6)
    for exp, inp, code in M2_VARIANTS:
        j = M2_TRIPLE[(exp, inp)]
        ad = code == "code_ad"
        end = not ad        # forward-only code_ad builds stop in GRDCHK_MAIN without an adxx file (run_verdict.py)
        n0 = NITER0.get((exp, inp), 0)
        bp, bj = M2_BUILD.get((exp, inp), ("plain_m2", "jaxdump_m2"))
        add(exp, inp, f"job{j}-plain", bp, j, "yardstick: STDOUT vs results/ (docs/YARDSTICK.md, M2)",
            "yardstick", end)
        add(exp, inp, f"job{j}-jdoff", bj, j, "jaxdump binary, dumps off (invisibility)", "jdoff", end)
        add(exp, inp, f"job{j}-jdon", bj, j, "jaxdump binary, dumps on: substep dumps in <run>/dumps "
            "(oracle of the M2 substep gates; invisibility)", "jdon", end, (n0, n0 + 1, n0 + 2))
        if (exp, inp) in M2_FD:
            jf, name = M2_FD[(exp, inp)]
            for lab, what in (("fdzero", f"zero ad{name} (option (i)): the FD gradient oracle and admCst/admFwd "
                               "yardstick"),
                              ("fdrandom", f"random ad{name}: shows the FD lines do not depend on the adxx file")):
                add(exp, inp, f"job{jf}-{lab}", bp, jf, what, lab)
        if (exp, inp) in M2_CTRLXX:
            jc, name = M2_CTRLXX[(exp, inp)]
            add(exp, inp, f"job{jc}-ctrlxx", bj, jc, f"GATE FIXTURE, NEVER A YARDSTICK: planted nonzero {name} "
                "(make_rundir --copy from $MJX_REFERENCE/ctrl_planted/0542f7a-*) read with the namelist overlays "
                "data.ctrl doInitXX=.FALSE., doMainUnpack=.FALSE.; dumps on, JAXDUMP_STEPS=0:1:2:3 (GOADK gates)",
                "ctrlxx", False, CTRLXX_STEPS)
        labs, jo = OUTPUT_OFF_M2.get((exp, inp), ((), None))
        for lab in labs:
            add(exp, inp, f"job{jo}-{lab}", bp, jo, f"{OUTPUT_OFF_WHAT[lab]} overlay (data.pkg): output-request "
                "invariance (model lines and pickups as the plain run); not a yardstick", lab, end)
    # M3 (lane A session 9)
    for exp, inp, code in M3_VARIANTS:
        j = M3_TRIPLE[(exp, inp)]
        n0 = NITER0.get((exp, inp), 0)
        bp, bj = M3_BUILD[(exp, inp)]
        add(exp, inp, f"job{j}-plain", bp, j, "yardstick: STDOUT vs results/ (docs/YARDSTICK.md, M3)", "yardstick")
        add(exp, inp, f"job{j}-jdoff", bj, j, "jaxdump binary, dumps off (invisibility)", "jdoff")
        add(exp, inp, f"job{j}-jdon", bj, j, "jaxdump binary, dumps on: substep dumps in <run>/dumps (oracle of the "
            "M3 substep gates; invisibility)", "jdon", True, (n0, n0 + 1, n0 + 2))
    # M4 (lane A session 11)
    for exp, inp, code in M4_VARIANTS:
        j = M4_TRIPLE[(exp, inp)]
        ad = code == "code_ad"
        end = not ad        # forward-only code_ad builds stop in GRDCHK_MAIN without an adxx file (run_verdict.py)
        n0 = NITER0.get((exp, inp), 0)
        bp, bj = M4_BUILD[(exp, inp)]
        add(exp, inp, f"job{j}-plain", bp, j, "yardstick: STDOUT vs results/ (docs/YARDSTICK.md, M4)", "yardstick",
            end)
        add(exp, inp, f"job{j}-jdoff", bj, j, "jaxdump binary, dumps off (invisibility)", "jdoff", end)
        add(exp, inp, f"job{j}-jdon", bj, j, "jaxdump binary, dumps on: substep dumps in <run>/dumps (oracle of the "
            "M4 substep gates; invisibility)", "jdon", end, (n0, n0 + 1, n0 + 2))
        if (exp, inp) in M4_FD:
            jf, name = M4_FD[(exp, inp)]
            for lab, what in (("fdzero", f"zero ad{name} (option (i)): the FD gradient oracle and admCst/admFwd "
                               "yardstick"),
                              ("fdrandom", f"random ad{name}: shows the FD lines do not depend on the adxx file")):
                add(exp, inp, f"job{jf}-{lab}", bp, jf, what, lab)
    # FTZ oracle (plan decision 13; lane A session 12)
    for (exp, inp), (j, bp, bj, _, _) in FTZ_VARIANTS.items():
        n0 = NITER0.get((exp, inp), 0)
        add(exp, inp, f"job{j}-plain", bp, j, "FTZ oracle (decision 13): the FTZ/DAZ plain binary; the oracle of the "
            "gates that meet subnormals, NEVER A YARDSTICK (the standard yardstick stays the reference)", "ftz_plain")
        add(exp, inp, f"job{j}-jdoff", bj, j, "FTZ oracle: FTZ/DAZ jaxdump binary, dumps off (invisibility)",
            "ftz_jdoff")
        add(exp, inp, f"job{j}-jdon", bj, j, "FTZ oracle: FTZ/DAZ jaxdump binary, dumps on (the oracle of the gates "
            "that meet subnormals; invisibility)", "ftz_jdon", True, (n0, n0 + 1, n0 + 2))
    for exp, inp, code in M2_MIN:
        add(exp, inp, f"job{M2_MIN_JOB}-plain", "plain_m2b", M2_MIN_JOB, "minimal case: EEBOOT + W2 topology only "
            "(THE_MODEL_MAIN skipped); no results/ reference; W2 log vs the code build's (docs/YARDSTICK.md)",
            "minimal")
    return runs


RUNS = _runs()


def run_top(run):
    return paths.REFERENCE_RUNS / run["exp"] / run["input"] / run["run_id"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_key(run):
    return f"{run['exp']}/{run['input']}/{run['run_id']}"


def reference_files(run):
    """{relative path under the run top: path} of the files whose sha256 the lock records."""
    top = run_top(run)
    out = {"rundir/output.txt": top / "rundir" / "output.txt"}
    if run["kind"] in COPY_KINDS:
        # the copied files as MANIFEST.json lists them (the FD runs: every ad<control> file; until lane A session 7
        # only adxx_qnet.* was globbed, so the M2 FD runs' adxx files were not locked)
        man = json.loads((top / "MANIFEST.json").read_text())
        for c in man.get("copies") or []:
            out[f"rundir/{c['name']}"] = top / "rundir" / c["name"]
        if run["kind"] not in ("fdzero", "fdrandom"):
            out["OVERLAY.txt"] = top / "OVERLAY.txt"
    if (run["exp"], run["input"]) in {(e, i) for e, i, _ in M2_VARIANTS + M2_MIN + M3_VARIANTS + M4_VARIANTS}:
        # exch2 tile topology logs (W2_printMsg < 0, pkg/exch2/w2_eeboot.F:84-93): lane B builds the cube maps from them
        for f in sorted((top / "rundir").resolve().glob("w2_tile_topology.*.log")):
            out[f"rundir/{f.name}"] = f
    if run["kind"] in RESTART_KINDS:
        rundir = (top / "rundir").resolve()
        for f in sorted(rundir.glob("pickup.*")):
            if f.is_file() and not f.is_symlink():
                out[f"rundir/{f.name}"] = f
        out["OVERLAY.txt"] = top / "OVERLAY.txt"
    if run["kind"] in FTZ_KINDS:
        # the FTZ oracle's written pickups (the gates compare their end pickups byte for byte)
        for f in sorted((top / "rundir").resolve().glob("pickup*")):
            if f.is_file() and not f.is_symlink():
                out[f"rundir/{f.name}"] = f
    return out


def dump_listing(run):
    d = run_top(run) / "dumps"
    return sorted([p.name, p.stat().st_size] for p in d.iterdir()) if d.is_dir() else None


def check_run(run, lock=None, runs_root=None):
    """List of problems of one registered run (empty: fine)."""
    top = (Path(runs_root) / run["exp"] / run["input"] / run["run_id"]) if runs_root else run_top(run)
    probs = []
    for need in ("READY", "MANIFEST.json", "run_provenance.txt", "rundir/output.txt"):
        if not (top / need).exists():
            probs.append(f"missing {top / need}")
    if probs:
        return probs
    man = json.loads((top / "MANIFEST.json").read_text())
    want = bin_name(run["exp"], code_of(run["exp"], run["input"]), run["build"])
    binp = Path(man["binary"]["path"]) if man.get("binary") else None
    if binp is None or binp.parent.name != want:
        probs.append(f"MANIFEST binary {binp} is not {want}")
    else:
        recorded = (binp.parent / "sha256").read_text().split()[0] if (binp.parent / "sha256").is_file() else None
        if not binp.is_file():
            probs.append(f"missing binary {binp}")
        elif not (man["binary"]["sha256"] == recorded == sha256(binp)):
            probs.append(f"binary sha256 mismatch: MANIFEST {man['binary']['sha256'][:12]}, sha256 file "
                         f"{(recorded or '-')[:12]}, mitgcmuv {sha256(binp)[:12]}")
    overlays = tuple((o["file"], o["group"], o["key"], o["value"]) for o in man.get("overlays") or [])
    if overlays != OVERLAYS.get(run["kind"], ()):
        probs.append(f"MANIFEST overlays {overlays} are not the registered {OVERLAYS.get(run['kind'], ())}")
    if bool(man.get("copies")) != (run["kind"] in COPY_KINDS):
        probs.append(f"MANIFEST copies {len(man.get('copies') or [])} files; kind {run['kind']} "
                     f"{'needs' if run['kind'] in COPY_KINDS else 'has none'}")
    for c in man.get("copies") or []:
        f = top / "rundir" / c["name"]
        if not f.is_file() or f.is_symlink() or sha256(f) != c["sha256"]:
            probs.append(f"copied file {c['name']} missing or not the copied bytes (MANIFEST sha256)")
    prov = (top / "run_provenance.txt").read_text()
    if "\nexit 0 " not in prov:
        probs.append("run_provenance.txt: exit code not 0")
    if run["ends_normally"] and "normal_end 1" not in prov:
        probs.append("run_provenance.txt: no normal end")
    if run.get("steps") is not None:
        info = top / "dumps" / "jaxdump_info.txt"
        got = tuple(int(ln.split()[1]) for ln in info.read_text().split("\n") if ln.startswith("step ")) \
            if info.is_file() else None
        if got != tuple(run["steps"]):
            probs.append(f"dumped iterations {got} (dumps/jaxdump_info.txt) are not the registered steps "
                         f"{tuple(run['steps'])}")
    if lock is not None:
        rec = lock.get(run_key(run))
        if rec is None:
            probs.append("not in the lock file")
        else:
            for rel, h in rec["sha256"].items():
                p = top / rel
                if not p.is_file():
                    probs.append(f"missing {p}")
                elif sha256(p) != h:
                    probs.append(f"sha256 of {p} differs from the lock")
            if rec.get("dumps") is not None:
                d = top / "dumps"
                got = sorted([q.name, q.stat().st_size] for q in d.iterdir()) if d.is_dir() else None
                if got != rec["dumps"]:
                    probs.append(f"dump listing of {d} differs from the lock")
    return probs


def make_lock():
    lock = {}
    for run in RUNS:
        top = run_top(run)
        man = json.loads((top / "MANIFEST.json").read_text())
        lock[run_key(run)] = {
            "binary": Path(man["binary"]["path"]).parent.name, "binary_sha256": man["binary"]["sha256"],
            "job": run["job"], "sha256": {rel: sha256(p) for rel, p in reference_files(run).items()},
            "dumps": dump_listing(run) if run["kind"] in DUMP_KINDS else None}
    return lock


def read_lock(path=LOCK):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------------------------
# yardstick

def find_run(exp, inp, kind):
    return next(r for r in RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, kind))


def yardstick_rows(exp, inp):
    """[(section, name, digits, deciding, pattern)] for one variant: testreport's check list of the yardstick run vs
    results/ (forward variants); for optim the adm list of the FD run and the forward list vs output_adm.txt."""
    rows = []
    if (exp, inp) not in AD_VARIANTS:
        rep = trj.compare(run_top(find_run(exp, inp, "yardstick")) / "rundir" / "output.txt", exp, inp)
        for v in rep.run.variables:
            rows.append(("forward vs results/" + rep.reference.name, v.name, v.digits, v.name == rep.run.s_var,
                         v.pattern))
        return rows
    rep = trj.compare(run_top(find_run(exp, inp, "fdzero")) / "rundir" / "output.txt", exp, inp)
    for v in rep.run.variables:
        rows.append(("adm list (FD run, zero adxx) vs results/output_adm.txt", v.name, v.digits,
                     v.name == rep.run.s_var, v.pattern))
    ref = rep.reference
    for kind in ("yardstick", "fdzero"):
        out = trj.read_lines(run_top(find_run(exp, inp, kind)) / "rundir" / "output.txt")
        run = trj.testoutput_run(out, trj.read_lines(ref), trj.KIND_FWD, {})
        for v in run.variables:
            rows.append((f"forward list ({kind} run) vs forward %MON of results/output_adm.txt", v.name, v.digits,
                         False, v.pattern))
    return rows


_AD_MON = re.compile(r"^\(PID\.TID \d+\.\d+\) %MON (ad_\w+?)\s*=\s*(\S+)\s*$")
AD_STATS = ("max", "min", "mean", "sd", "del2")


def adm_monitor_blocks(lines):
    """Pure: the adjoint %MON blocks of an adjoint build's STDOUT (MONITOR in adjoint mode, `%MON ad_*` lines): a list
    of (ad_time_tsnumber, [(name, value text), ...]) in printed order; a block starts at `ad_time_tsnumber`."""
    blocks = []
    for ln in lines:
        m = _AD_MON.match(ln)
        if not m:
            continue
        if m[1] == "ad_time_tsnumber":
            blocks.append((int(m[2]), []))
        elif blocks:
            blocks[-1][1].append((m[1], m[2]))
    return blocks


def adm_monitor_summary(blocks):
    """{tsnumbers, records_per_step, fields: [(group, stats)], same_fields, all_zero_groups} of adm_monitor_blocks:
    group = name without its statistic suffix (ad_dynstat_adtheta, ad_forcing_adqnet, ...), ad_time_secondsf left
    out; all_zero_groups = groups whose every value is zero in every block."""
    ts = [b[0] for b in blocks]
    names = list(dict.fromkeys(n for _, f in blocks for n, _ in f if n != "ad_time_secondsf"))
    groups = {}
    for n in names:
        g, _, s = n.rpartition("_")
        if s not in AD_STATS:
            g, s = n, ""
        groups.setdefault(g, []).append(s)
    zero = [g for g in groups if all(float(v.replace("D", "E")) == 0.0 for _, f in blocks for n, v in f
                                     if n.rpartition("_")[0] == g or n == g)]
    per = sorted({ts.count(x) for x in ts})
    return {"tsnumbers": ts, "records_per_step": per, "fields": list(groups.items()),
            "same_fields": len({tuple(n for n, _ in f) for _, f in blocks}) <= 1, "all_zero_groups": zero}


_OBJF_LINE = re.compile(r"^\(PID\.TID \d+\.\d+\)\s+--> objf_")
_ADM_LINE = re.compile(r"^\(PID\.TID \d+\.\d+\)\s+ADM ")


def model_lines(lines):
    """Pure: the STDOUT lines that carry the model's numbers, in order: every numeric record of compare_runs.py (%MON,
    incl. the ptracer monitor trcstat_*, %SBO, cg2d_init_res, cg2d_iters, cg2d Sum(rhs), the cost's fc lines, the
    gradient check's fc lines), the cost terms (`--> objf_`) and the gradient check's `ADM` lines."""
    return [ln.rstrip() for ln in lines
            if any(rx.match(ln) for _, rx in cmp_runs.RECORD_PATTERNS) or _OBJF_LINE.match(ln) or _ADM_LINE.match(ln)]


def mds_pickups(top):
    """{name: path} of the MDS pickup files (regular files rundir/pickup*.data|.meta) a run wrote."""
    rd = (Path(top) / "rundir").resolve()
    return {f.name: f for f in sorted(rd.glob("pickup*")) if f.is_file() and not f.is_symlink()
            and f.suffix in (".data", ".meta")}


def output_request_problems(base_top, test_top):
    """(problems, number of model lines, common pickup files) of an output-request overlay run (diagoff, mncoff)
    against its plain run: the model lines (model_lines) identical as text and in order, and every MDS pickup file both
    runs wrote byte-identical (a pickup only one run wrote is listed as a problem too)."""
    a, b = (model_lines(cmp_runs.read_stdout(t)) for t in (base_top, test_top))
    probs = []
    if a != b:
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        probs.append(f"model lines differ ({len(a)} vs {len(b)}), first at {k}: "
                     f"{a[k] if k < len(a) else '-'!r} vs {b[k] if k < len(b) else '-'!r}")
    pa, pb = mds_pickups(base_top), mds_pickups(test_top)
    probs += [f"pickup {n} written by one run only" for n in sorted(set(pa) ^ set(pb))]
    common = sorted(set(pa) & set(pb))
    probs += [f"pickup {n} differs" for n in common if pa[n].read_bytes() != pb[n].read_bytes()]
    return probs, len(a), common


def checklist_names(exp, inp):
    """The variables of testreport's listVar for the variant (what the yardstick must cover)."""
    upstream = paths.UPSTREAM
    kind = trj.kind_of_variant(inp)
    files = trj.resolve_checklists(upstream / "verification" / exp, inp, kind)
    ref = upstream / "verification" / exp / "results" / trj.reference_name(inp, kind)
    run = trj.testoutput_run([], trj.read_lines(ref), kind, {k: p.read_text() for k, p in files.items()})
    return [v.name for v in run.variables]


def fmt_digits(d):
    return "--" if d == 99 else str(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--docs", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    if a.lock:
        LOCK.write_text(json.dumps(make_lock(), indent=1) + "\n")
        print(f"WROTE {LOCK}")
    if a.check:
        lock = read_lock()
        for run in RUNS:
            p = check_run(run, lock)
            print(("OK   " if not p else "FAIL ") + run_key(run) + ("" if not p else ": " + "; ".join(p)))
            rc |= bool(p)
    if a.docs:
        docs = _load("_mjx_reference_docs", REPO / "reference" / "reference_docs.py")
        for name, text in docs.render().items():
            (REPO / "docs" / name).write_text(text)
            print(f"WROTE docs/{name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
