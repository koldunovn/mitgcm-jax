"""M4 Fortran oracle (lane A session 11, plan M4): instrumented builds of the sea-ice / EXF codes, invisibility of the
shim, the forcing and sea-ice records in the dumps-on runs, the FD oracle runs of the code_ad variants, and the M4
coverage worklists.

Oracle-dependent: fails (never skips) while a build, run or report is missing. Builds: plain 704fd6b and jaxdump
704fd6b (job 27855975), 1D_ocean_ice_column jaxdump 1ac79cb (job 27856045, B-grid stages); runs: the invisibility
triples of reference/reference_runs.py M4_TRIPLE and the FD runs of M4_FD.
"""

import gzip
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from mitjax import paths
from mitjax.io.dump import DumpSet

REPO = Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load(REPO / "reference" / "reference_runs.py", "_mjx_rr_m4")
UP7 = "63cdc0b"
CORE = {"S00_begin", "G00_geometry", "X00_exch_probe", "S04_oceanic_phys", "S06_dynamics", "S15_tracers_correction",
        "S16_blocking_exchanges", "I02_ini_fields"}
# the forcing / sea-ice / thsice / salt-plume stages (reference/jaxdump/SUBSTEPS.md "M4")
PKG = re.compile(r"^(X0[1-6]_|I00b?_seaice_begin|I01b?_dynsolver|I02_advdiff|I02a_thsice|I03_reg_ridge|I04a?_growth"
                 r"|Y0\d_|P1[234]_)")
EXF = ("X01_exf_getffields X02_exf_radiation X03_exf_wind X04_exf_bulkformulae X05_exf_hflux_sflux "
       "X06_exf_mapfields").split()
CGRID_BEGIN = ["I00_seaice_begin", "Y01_get_dynforcing", "Y09_ocean_stress", "I01_dynsolver"]
THERMO = ["I02_advdiff", "I03_reg_ridge", "I04_growth", "P13_seaice_model"]

# package stages each M4 jaxdump build compiles (measured from jaxdump_stages.txt): (build, set)
_ALLY = {f"Y0{n}_{s}" for n, s in ((2, "ice_strength"), (3, "freedrift"), (4, "solver_inputs"), (5, "evp"),
                                     (6, "lsr"), (7, "krylov"), (8, "jfnk"))}
_C = set(EXF) | set(CGRID_BEGIN) | {"I02_advdiff", "I03_reg_ridge", "P13_seaice_model"}
COMPILED = {
    ("1D_ocean_ice_column", "code"): ("1ac79cb", set(EXF) | {"I00b_seaice_begin", "I01b_dynsolver", "I03_reg_ridge",
                                                            "I04_growth", "P13_seaice_model", "Y01_get_dynforcing",
                                                            "Y09_ocean_stress"}),
    ("1D_ocean_ice_column", "code_ad"): ("1ac79cb", _C | {"I04_growth"}),
    ("offline_exf_seaice", "code"): ("704fd6b", _C | _ALLY | {"I04_growth", "I02a_thsice_advect", "P12_thsice_main"}),
    ("offline_exf_seaice", "code_ad"): ("704fd6b", _C | _ALLY - {"Y07_krylov", "Y08_jfnk"}
                                        | {"I04a_growth_adx", "I02a_thsice_advect", "P12_thsice_main"}),
    ("lab_sea", "code"): ("704fd6b", _C | _ALLY - {"Y07_krylov", "Y08_jfnk"} | {"I04_growth", "P14_salt_plume_exch"}),
    ("lab_sea", "code_ad"): ("704fd6b", _C | _ALLY - {"Y07_krylov", "Y08_jfnk"} | {"I04_growth",
                                                                                  "P14_salt_plume_exch"}),
    ("seaice_itd", "code"): ("704fd6b", _C | _ALLY | {"I04_growth"}),
    ("global_ocean.cs32x15", "code"): ("704fd6b", _C | _ALLY - {"Y03_freedrift", "Y05_evp"}
                                       | {"I04_growth", "I02a_thsice_advect", "P12_thsice_main"}),
    ("global_ocean.cs32x15", "code_ad"): ("704fd6b", _C | _ALLY - {"Y07_krylov", "Y08_jfnk"}
                                          | {"I04_growth", "I02a_thsice_advect", "P12_thsice_main"}),
}

# package stages that run in each M4 variant (records at the second dumped iteration, in this order)
_LSR = ["Y02_ice_strength", "Y04_solver_inputs", "Y06_lsr"]
RUN = {
    ("1D_ocean_ice_column", "input"): EXF + ["I00b_seaice_begin", "I01b_dynsolver", "I03_reg_ridge", "I04_growth",
                                             "P13_seaice_model"],
    ("1D_ocean_ice_column", "input_ad"): EXF + CGRID_BEGIN + THERMO,
    ("offline_exf_seaice", "input"): EXF + ["P12_thsice_main"] + CGRID_BEGIN + ["I02a_thsice_advect",
                                                                               "P13_seaice_model"],
    ("offline_exf_seaice", "input.thermo"): EXF + CGRID_BEGIN + THERMO,
    ("offline_exf_seaice", "input_ad"): EXF + CGRID_BEGIN + ["I02_advdiff", "I03_reg_ridge", "I04a_growth_adx",
                                                             "P13_seaice_model"],
    ("offline_exf_seaice", "input_ad.thsice"): EXF + ["P12_thsice_main"],
    ("offline_exf_seaice", "input.thsice"): EXF + ["P12_thsice_main"],
    ("lab_sea", "input"): EXF + CGRID_BEGIN[:2] + ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs",
                                                   "Y06_lsr"] + CGRID_BEGIN[2:] + THERMO,
    ("lab_sea", "input.fd"): EXF + CGRID_BEGIN[:2] + ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs"]
    + CGRID_BEGIN[2:] + THERMO,
    ("lab_sea", "input.hb87"): EXF + CGRID_BEGIN[:2] + ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs",
                                                        "Y05_evp"] + CGRID_BEGIN[2:] + THERMO,
    ("lab_sea", "input.salt_plume"): EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + THERMO
    + ["P14_salt_plume_exch"],
    ("lab_sea", "input.longstep"): [],
    ("lab_sea", "input.natl_box"): [],
    ("lab_sea", "input_ad"): EXF + CGRID_BEGIN[:2] + ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs",
                                                      "Y06_lsr"] + CGRID_BEGIN[2:] + THERMO,
    ("lab_sea", "input_ad.noseaice"): EXF,
    ("lab_sea", "input_ad.noseaicedyn"): EXF + CGRID_BEGIN + THERMO + ["P14_salt_plume_exch"],
    ("seaice_itd", "input"): EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + THERMO,
    ("seaice_itd", "input.lipscomb07"): EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + ["I02_advdiff",
                                                                                         "I03_reg_ridge",
                                                                                         "P13_seaice_model"],
    ("seaice_itd", "input.thermo"): EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + THERMO,
    ("global_ocean.cs32x15", "input.seaice"): EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + THERMO,
    ("global_ocean.cs32x15", "input.icedyn"): EXF + ["P12_thsice_main"] + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:]
    + ["I02a_thsice_advect", "P13_seaice_model"],
    ("global_ocean.cs32x15", "input.thsice"): ["P12_thsice_main"],
    ("global_ocean.cs32x15", "input_ad.thsice"): EXF + ["P12_thsice_main"],
}
for _v in ("dyn_lsr", "dyn_ellnnfr", "dyn_mce"):
    RUN[("offline_exf_seaice", f"input.{_v}")] = EXF + CGRID_BEGIN[:2] + _LSR + CGRID_BEGIN[2:] + ["I02_advdiff",
                                                                                                  "I03_reg_ridge",
                                                                                                  "P13_seaice_model"]
for _v, _s in (("dyn_jfnk", ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs", "Y08_jfnk"]),
               ("dyn_paralens", ["Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs", "Y07_krylov"]),
               ("dyn_teardrop", _LSR)):
    RUN[("offline_exf_seaice", f"input.{_v}")] = EXF + ["P12_thsice_main"] + CGRID_BEGIN[:2] + _s + CGRID_BEGIN[2:] \
        + ["I02a_thsice_advect", "P13_seaice_model"]
for _v in ("input_ad.seaice", "input_ad.seaice_dynmix"):
    RUN[("global_ocean.cs32x15", _v)] = EXF + CGRID_BEGIN[:2] + ["Y02_ice_strength", "Y03_freedrift",
                                                                "Y04_solver_inputs", "Y06_lsr"] + CGRID_BEGIN[2:] \
        + THERMO

# stages that run and hold only zeros at every dumped iteration, with the measured cause (every other package stage
# that runs must hold a nonzero value in some field)
ALL_ZERO = {
    # TAUX/TAUY = SIwindTau*SIMaskU (seaice_get_dynforcing.F:216); SIMaskU is set only #ifdef SEAICE_CGRID
    # (seaice_init_fixed.F:245-248) and 1D_ocean_ice_column/code_ad defines neither SEAICE_CGRID nor B grid
    ("1D_ocean_ice_column", "input_ad", "Y01_get_dynforcing"),
    # saltPlumeFlux is set only while ice forms (seaice_growth.F:2085-2086); in steps 0-2 of this variant the ice only
    # melts (HEFF after I04_growth <= after I03_reg_ridge everywhere)
    ("lab_sea", "input_ad.noseaicedyn", "P14_salt_plume_exch"),
}
# fields that must be nonzero in the first three experiments (the porting order of the next cycle)
NONZERO = {
    ("1D_ocean_ice_column", "input"): {
        "X04_exf_bulkformulae": ("hs", "hl", "evap", "ustress"), "X06_exf_mapfields": ("Qnet", "Qsw", "fu", "EmPmR"),
        "I00b_seaice_begin": ("HEFF", "AREA", "TICES", "WINDX", "DAIRN", "AMASS"),
        "I01b_dynsolver": ("FORCEX", "FORCEX0", "PRESS0", "SEAICE_zMax", "GWATX", "fu", "fv"),
        "I04_growth": ("HEFF", "AREA", "HSALT", "Qnet", "Qsw", "saltFlux", "sIceLoad")},
    ("offline_exf_seaice", "input.thermo"): {
        "X04_exf_bulkformulae": ("hs", "hl", "evap"), "I04_growth": ("HEFF", "AREA", "Qnet", "TICES")},
    ("offline_exf_seaice", "input.dyn_lsr"): {
        "Y02_ice_strength": ("PRESS0", "SEAICE_zMax"), "Y06_lsr": ("UICE", "VICE", "ZETA", "ETA"),
        "Y09_ocean_stress": ("fu", "fv")},
    ("offline_exf_seaice", "input"): {"P12_thsice_main": ("iceMask", "iceHeight"),
                                      "I01_dynsolver": ("UICE", "VICE")},
    ("lab_sea", "input"): {"Y06_lsr": ("UICE", "VICE"), "I04_growth": ("HEFF", "AREA", "HSNOW", "Qnet"),
                           "X04_exf_bulkformulae": ("hs", "hl", "evap")},
}


def _stages(exp, code):
    commit, _ = COMPILED[(exp, code)]
    f = paths.REFERENCE / "bin" / f"{exp}-{code}-{UP7}-{commit}-jaxdump" / "jaxdump_stages.txt"
    return {ln.split()[0] for ln in f.read_text().split("\n") if ln.strip()}


def _jdon(exp, inp):
    return DumpSet(rr.run_top(rr.find_run(exp, inp, "jdon")) / "dumps")


def stage_problems(ds, exp, inp):
    """Problems of one dumps-on run: the package stages at the second dumped iteration differ from RUN (order
    included), a stage that runs holds only zeros at every dumped iteration (unless ALL_ZERO), or a NONZERO field is
    zero."""
    probs = []
    its = ds.iterations()
    it = its[1]
    got = [s for s in dict.fromkeys(k[1] for k in ds.keys(it)) if PKG.match(s)]
    if got != RUN[(exp, inp)]:
        probs.append(f"{exp}/{inp}: package stages {got}")
    for s in got:
        flds = sorted({k[2] for k in ds.keys(it) if k[1] == s})
        nz = any(np.any(ds.field(i, s, f) != 0) for i in its for f in flds if s in ds.stages(i))
        if nz == ((exp, inp, s) in ALL_ZERO):
            probs.append(f"{exp}/{inp}: {s} {'nonzero but listed in ALL_ZERO' if nz else 'all zero'}")
    for s, flds in NONZERO.get((exp, inp), {}).items():
        have = {k[2] for k in ds.keys(it) if k[1] == s}
        probs += [f"{exp}/{inp}: {s} {f} {'zero' if f in have else 'missing'}" for f in flds
                  if f not in have or not np.any(ds.field(it, s, f) != 0)]
    return probs


def test_m4_compiled_stages():
    """Core stages in every M4 jaxdump build; the forcing / sea-ice stages exactly as compiled (B-grid stages only in
    1D_ocean_ice_column/code, SEAICE_BGRID_DYNAMICS; GROWTH_ADX only in offline_exf_seaice/code_ad; solver stages per
    the SEAICE_ALLOW_* options of each code)."""
    for (exp, code), (_, want) in COMPILED.items():
        st = _stages(exp, code)
        assert CORE <= st, (exp, code, sorted(CORE - st))
        got = {s for s in st if PKG.match(s)}
        assert got == want, (exp, code, sorted(got ^ want))
    assert {e for e, _, _ in rr.M4_VARIANTS} == {e for e, _ in COMPILED}


def test_m4_invisibility_verdicts():
    """Dumps off and on change no output file and no STDOUT number of any M4 variant (no new exemption)."""
    for exp, inp, _ in rr.M4_VARIANTS:
        f = rr.invisibility_file(exp, inp)
        assert f.is_file(), f"missing {f}"
        lines = f.read_text().split("\n")
        assert not any(ln.startswith("VISIBLE") for ln in lines), (exp, inp, lines[:3])
        for mode in ("dumps-off", "dumps-on"):
            hit = [ln for ln in lines if ln.startswith(f"INVISIBLE {exp}/{inp} {mode}")]
            assert len(hit) == 1 and "DIAGSTATS_UNSET_REGIONS 0" in hit[0], (exp, inp, mode)


def test_m4_stage_presence():
    """Every M4 variant runs exactly its forcing / sea-ice stages (none where the package is off: lab_sea
    input.longstep, input.natl_box, input_ad.noseaice), every stage that runs holds nonzero fields (two measured
    exceptions, ALL_ZERO), and the key fields of the first three experiments are nonzero. Negative controls: the
    1D_ocean_ice_column run of the 704fd6b build (no B-grid stages) and planted RUN / ALL_ZERO entries fail."""
    probs = []
    for exp, inp, _ in rr.M4_VARIANTS:
        probs += stage_problems(_jdon(exp, inp), exp, inp)
    assert not probs, "\n".join(probs)
    # control 1: the superseded 1D run of job 27855986 (704fd6b jaxdump, no I00b/I01b) is caught
    old = DumpSet(paths.REFERENCE_RUNS / "1D_ocean_ice_column" / "input" / "job27855986-jdon" / "dumps")
    assert any("package stages" in p for p in stage_problems(old, "1D_ocean_ice_column", "input"))
    # control 2: a planted expectation (I04_growth where the variant runs only dynamics) is caught
    ds = _jdon("offline_exf_seaice", "input.dyn_lsr")
    saved = RUN[("offline_exf_seaice", "input.dyn_lsr")]
    try:
        RUN[("offline_exf_seaice", "input.dyn_lsr")] = saved + ["I04_growth"]
        assert stage_problems(ds, "offline_exf_seaice", "input.dyn_lsr")
    finally:
        RUN[("offline_exf_seaice", "input.dyn_lsr")] = saved
    # control 3: removing an ALL_ZERO exemption is caught
    ALL_ZERO.discard(("lab_sea", "input_ad.noseaicedyn", "P14_salt_plume_exch"))
    try:
        assert any("all zero" in p for p in stage_problems(_jdon("lab_sea", "input_ad.noseaicedyn"), "lab_sea",
                                                            "input_ad.noseaicedyn"))
    finally:
        ALL_ZERO.add(("lab_sea", "input_ad.noseaicedyn", "P14_salt_plume_exch"))


def _adm(run):
    return [ln.strip() for ln in (rr.run_top(run) / "rundir" / "output.txt").read_text(encoding="latin-1")
            .split("\n") if " ADM " in ln]


def test_m4_fd_oracle_runs():
    """FD oracle of every M4 code_ad variant: the zero and the random ad<control> runs end normally and differ only in
    the `ADM adjoint_gradient` lines (the FD gradient does not depend on the adxx file); the FD gradient is nonzero
    and finite. Control: a planted difference in an FD line is detected by the same comparison."""
    ad = [(e, i) for e, i, c in rr.M4_VARIANTS if c == "code_ad"]
    assert sorted(ad) == sorted(rr.M4_FD)
    for exp, inp in ad:
        z, r = (_adm(rr.find_run(exp, inp, k)) for k in ("fdzero", "fdrandom"))
        assert len(z) == len(r) > 0, (exp, inp)
        diff = {a.split("ADM", 1)[1].split()[0] for a, b in zip(z, r) if a != b}
        assert diff == {"adjoint_gradient"}, (exp, inp, diff)
        fd = [float(a.split("=")[1].replace("D", "E")) for a in z if "finite-diff_grad" in a]
        assert fd and all(np.isfinite(fd)) and any(v != 0 for v in fd), (exp, inp, fd)
    z = _adm(rr.find_run(*ad[0], "fdzero"))
    planted = [ln.replace("finite-diff_grad       = ", "finite-diff_grad       = 1") for ln in z]
    assert {a.split("ADM", 1)[1].split()[0] for a, b in zip(z, planted) if a != b} == {"finite-diff_grad"}


def _cov_current():
    out = []
    for ln in (REPO / "reference" / "coverage" / "CURRENT_M4").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            doc, rep = ln.split()
            out.append((REPO / "docs" / "coverage" / doc, paths.REFERENCE / "coverage" / rep))
    return out


def test_m4_coverage_worklists():
    """docs/coverage/<doc>.md of every M4 build is its gcov report (titled M4); every M4 variant is in a report, ran
    the_model_main once, printed the same %MON/cg2d lines as the plain oracle run, and is physics-active (the M4 rung
    checks of tools/coverage.py)."""
    problems, seen = [], set()
    from tools.coverage import neutral_paths    # the page names $MJX_* locations, the report absolute ones
    for doc, md in _cov_current():
        if not doc.is_file() or doc.read_text() != neutral_paths(md.read_text()):
            problems.append(f"{doc} is not the report {md}")
        if not md.read_text().split("\n")[0].endswith("— M4 porting worklist"):
            problems.append(f"{md}: title")
        rep = json.load(gzip.open(str(md)[:-3] + ".json.gz", "rt"))
        problems += [f"{md.name}: mapping problem {p}" for p in rep["problems"]]
        for v in rep["variants"]:
            seen.add((rep["meta"]["experiment"], v["input_dir"]))
            if v["compare_plain"]["equal"] is not True:
                problems.append(f"{md}: {v['input_dir']} gcov run vs plain {v['compare_plain']}")
            main = [r for r in v["routines"] if r["routine"] == "the_model_main"]
            if not main or main[0]["calls"] != 1:
                problems.append(f"{md}: {v['input_dir']} the_model_main not executed once")
            problems += [f"{md}: {v['input_dir']} inactive {c}" for c in v["physics"] if not c["ok"]]
    missing = {(e, i) for e, i, _ in rr.M4_VARIANTS} - seen
    assert not problems and not missing, "\n".join(problems + [f"not in a report: {m}" for m in sorted(missing)])
