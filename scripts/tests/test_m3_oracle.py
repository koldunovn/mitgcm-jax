"""M3 Fortran oracle (lane A session 9, plan Task 28): instrumented builds of the M3 codes with the column-mixing
stages, invisibility of the shim, the mixing records in the dumps-on runs, and the M3 coverage worklists.

Oracle-dependent: fails (never skips) while a build, run or report is missing. Builds: plain and jaxdump 463504e
(job 27840380; global_ocean.90x40x15's input.dwnslp/input.idemix: the M1 plain build f48c062 and the 463504e jaxdump
build); runs: the invisibility triples of reference/reference_runs.py M3_TRIPLE.
"""

import gzip
import importlib.util
import json
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


rr = _load(REPO / "reference" / "reference_runs.py", "_mjx_rr_m3")
UP7, JD = "63cdc0b", "463504e-jaxdump"
CORE = {"S00_begin", "G00_geometry", "X00_exch_probe", "S04_oceanic_phys", "S06_dynamics", "S15_tracers_correction",
        "S16_blocking_exchanges", "I02_ini_fields"}
MIXING = {"P04_ggl90", "P07_kpp", "P08_pp81", "P09_my82", "P10_kpp_exch", "P11_ggl90_exch", "T05_opps",
          "T06_convective_adjustment"}
# mixing stages each M3 jaxdump build compiles (from its preprocessed sources: jaxdump_stages.txt)
COMPILED = {"vermix": MIXING,
            "global_ocean.90x40x15": {"P04_ggl90", "P11_ggl90_exch", "T06_convective_adjustment"}}
# mixing stages that run (records at the second dumped iteration) and their fields, per variant; every listed field
# is nonzero somewhere (KPPghat, KPPfrac stay zero in vermix: no non-local flux term, no shortwave)
RUN = {("vermix", "input"): {"P07_kpp": ("KPPviscAz", "KPPdiffKzT", "KPPdiffKzS", "KPPhbl"), "P10_kpp_exch": ()},
       ("vermix", "input.dd"): {"P07_kpp": ("KPPviscAz", "KPPdiffKzT", "KPPdiffKzS", "KPPhbl"), "P10_kpp_exch": ()},
       ("vermix", "input.ggl90"): {"P04_ggl90": ("GGL90TKE", "GGL90viscArU", "GGL90diffKr"), "P11_ggl90_exch": ()},
       ("vermix", "input.gglLC"): {"P04_ggl90": ("GGL90TKE", "GGL90viscArU", "GGL90diffKr"), "P11_ggl90_exch": ()},
       ("vermix", "input.my82"): {"P09_my82": ("MYviscAr", "MYdiffKr", "MYhbl")},
       ("vermix", "input.opps"): {"T05_opps": ("theta", "salt")},
       ("vermix", "input.pp81"): {"P08_pp81": ("PPviscAr", "PPdiffKr")},
       ("front_relax", "input.bvp"): {"T06_convective_adjustment": ("theta", "salt")},
       ("front_relax", "input.mxl"): {"T06_convective_adjustment": ("theta", "salt")},
       ("front_relax", "input.top"): {"T06_convective_adjustment": ("theta", "salt")},
       ("global_ocean.90x40x15", "input.idemix"): {"P04_ggl90": ("GGL90TKE", "IDEMIX_E", "IDEMIX_F_B", "IDEMIX_F_S"),
                                                   "P11_ggl90_exch": ()}}
# P11 sits after a one-line IF (useGGL90) and so dumps in every build that compiles GGL90, with zero fields when GGL90
# is off
P11_EVERYWHERE = {"vermix", "global_ocean.90x40x15"}


def _stages(exp):
    f = paths.REFERENCE / "bin" / f"{exp}-code-{UP7}-{JD}" / "jaxdump_stages.txt"
    return {ln.split()[0] for ln in f.read_text().split("\n") if ln.strip()}


def _jdon(exp, inp):
    return DumpSet(rr.run_top(rr.find_run(exp, inp, "jdon")) / "dumps")


def test_m3_compiled_stages():
    """Core stages in every M3 jaxdump build; the column-mixing stages exactly where their package is compiled
    (vermix: KPP, PP81, MY82, GGL90, OPPS; global_ocean: GGL90 with IDEMIX); T06 wherever INCLUDE_CONVECT_CALL."""
    for exp in sorted({e for e, _, _ in rr.M3_VARIANTS}):
        st = _stages(exp)
        assert CORE <= st, (exp, sorted(CORE - st))
        want = COMPILED.get(exp, {"T06_convective_adjustment"})
        assert st & MIXING == want, (exp, sorted(st & MIXING))


def test_m3_invisibility_verdicts():
    """Dumps off and on change no output file and no STDOUT number of any M3 variant (no new exemption)."""
    for exp, inp, _ in rr.M3_VARIANTS:
        f = rr.invisibility_file(exp, inp)
        assert f.is_file(), f"missing {f}"
        lines = f.read_text().split("\n")
        assert not any(ln.startswith("VISIBLE") for ln in lines), (exp, inp, lines[:3])
        for mode in ("dumps-off", "dumps-on"):
            hit = [ln for ln in lines if ln.startswith(f"INVISIBLE {exp}/{inp} {mode}")]
            assert len(hit) == 1 and "DIAGSTATS_UNSET_REGIONS 0" in hit[0], (exp, inp, mode)


def test_m3_mixing_records():
    """The mixing stages that run in each variant (and no other mixing stage) hold their fields, nonzero; a stage
    of a package that is switched off has no record (the planted check: the P07 records of vermix/input are absent in
    vermix/input.pp81, where useKPP is off). The KPP fields after KPP_DO_EXCH equal those after KPP_CALC in the
    interior of the tile."""
    for exp, inp, _ in rr.M3_VARIANTS:
        ds = _jdon(exp, inp)
        it = ds.iterations()[1]
        want = dict(RUN.get((exp, inp), {}))
        if exp in P11_EVERYWHERE:
            want.setdefault("P11_ggl90_exch", ())
        got = set(ds.stages(it)) & MIXING
        assert got == set(want), (exp, inp, sorted(got))
        for stage, flds in want.items():
            for f in flds:
                assert np.any(ds.field(it, stage, f) != 0), (exp, inp, stage, f)
    ds = _jdon("vermix", "input")          # one 1x1 column (sNx = sNy = 1, OLx = OLy = 2)
    it = ds.iterations()[1]
    for f in ("KPPviscAz", "KPPdiffKzT", "KPPdiffKzS", "KPPhbl"):
        (a,), (b,) = ds.tiles(it, "P07_kpp", f).values(), ds.tiles(it, "P10_kpp_exch", f).values()
        assert a.interior.size > 0 and np.array_equal(a.interior.view(np.int64), b.interior.view(np.int64)), f
    pp = _jdon("vermix", "input.pp81")
    assert not any(k[1] == "P07_kpp" for k in pp.keys(pp.iterations()[1]))


# mapping problems of the coverage reports with their cause (strict: any other one fails, and so does a listed one
# that disappears): pkg/layers' unbalanced #ifdef ALLOW_LAYERS (docs/ISSUES_UPSTREAM.md) makes cpp exit nonzero, which
# the Makefile pipe ignores and tools/cpp_live.py does not; the two routines (1 executed function each) are missing
# from the tutorial_reentrant_channel tables
KNOWN_MAPPING = {("tutorial_reentrant_channel", f): "upstream: unbalanced #ifdef ALLOW_LAYERS"
                 for f in ("layers_fluxcalc.F", "layers_fluxcalc.f", "layers_locate.F", "layers_locate.f")}


def _cov_current():
    out = []
    for ln in (REPO / "reference" / "coverage" / "CURRENT_M3").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            doc, rep = ln.split()
            out.append((REPO / "docs" / "coverage" / doc, paths.REFERENCE / "coverage" / rep))
    return out


def test_m3_coverage_worklists():
    """docs/coverage/<doc>.md of every M3 build is its gcov report (titled M3); every M3 variant is in a report, ran
    the_model_main once, printed the same %MON/cg2d lines as the plain oracle run, and is physics-active."""
    problems, seen = [], set()
    from tools.coverage import neutral_paths    # the page names $MJX_* locations, the report absolute ones
    for doc, md in _cov_current():
        if not doc.is_file() or doc.read_text() != neutral_paths(md.read_text()):
            problems.append(f"{doc} is not the report {md}")
        if not md.read_text().split("\n")[0].endswith("— M3 porting worklist"):
            problems.append(f"{md}: title")
        rep = json.load(gzip.open(str(md)[:-3] + ".json.gz", "rt"))
        exp = rep["meta"]["experiment"]
        known = {f for (e, f) in KNOWN_MAPPING if e == exp}
        hit = {p.split(":")[0] for p in rep["problems"]}
        problems += [f"{md.name}: mapping problem {p}" for p in rep["problems"] if p.split(":")[0] not in known]
        problems += [f"{md.name}: known mapping problem of {f} no longer reported (update KNOWN_MAPPING)"
                     for f in sorted(known - hit)]
        for v in rep["variants"]:
            seen.add((rep["meta"]["experiment"], v["input_dir"]))
            if v["compare_plain"]["equal"] is not True:
                problems.append(f"{md}: {v['input_dir']} gcov run vs plain {v['compare_plain']}")
            main = [r for r in v["routines"] if r["routine"] == "the_model_main"]
            if not main or main[0]["calls"] != 1:
                problems.append(f"{md}: {v['input_dir']} the_model_main not executed once")
            problems += [f"{md}: {v['input_dir']} inactive {c}" for c in v["physics"] if not c["ok"]]
    missing = {(e, i) for e, i, _ in rr.M3_VARIANTS} - seen
    assert not problems and not missing, "\n".join(problems + [f"not in a report: {m}" for m in sorted(missing)])
