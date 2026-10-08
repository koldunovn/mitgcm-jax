"""Branch coverage (plan Task 5): the gcov optfile, the CPP arm parser, the coverage tool on real gcov data with
planted errors, and the M1 coverage reports (docs/coverage/<experiment>.md) with their physics-active checks.

Coverage data: reference/coverage/CURRENT names the gcov binaries, the run id, the collect directory and the report
of the current coverage runs under $MJX_REFERENCE/coverage (reference/coverage/run_gcov.sbatch, report_gcov.sbatch).
Tests reading them fail, not skip, when the data is missing. Each check carries a negative control: the same check
applied to a planted error reports it.
"""

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from mitjax import paths
from mitjax.config import cpp_options
from mitjax.io import stdout as sio
from tools import coverage as cv
from tools import cpp_live

REPO = Path(__file__).resolve().parents[2]
M1 = [("tutorial_barotropic_gyre", "input"), ("tutorial_baroclinic_gyre", "input"), ("advect_xy", "input"),
      ("advect_xy", "input.ab3_c4"), ("advect_xz", "input"), ("advect_xz", "input.nlfs"), ("advect_xz", "input.pqm"),
      ("global_ocean.90x40x15", "input"), ("tutorial_global_oce_optim", "input_ad")]
# physics-active checks measured inactive, pending Nikolay's decision; strict: the test fails
# when one becomes active (update this table) or when any other check is inactive
KNOWN_INACTIVE = {
    ("tutorial_baroclinic_gyre", "input", "IVDC events (statically unstable points)"):
        "calc_ivdc.F:48 executed 0 times",
}


def current():
    f = REPO / "reference" / "coverage" / "CURRENT"
    out = {}
    for ln in f.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, v = ln.split(None, 1)
            out[k] = v.strip()
    return out


def run_top(exp, inp):
    return paths.REFERENCE / "coverage" / exp / inp / current()["runs"]


def gcov_build(exp, inp):
    code = "code_ad" if inp.startswith("input_ad") else "code"
    return paths.REFERENCE / "build" / f"{exp}-{code}-63cdc0b-{current()['binaries']}-gcov"


# ---------------------------------------------------------------------------------------------------------------

def optfile_problems(gcov_text, oracle_text):
    """The gcov optfile = line 1 of the oracle optfile + the MJX-GCOV header + the rest of it byte for byte + the
    MJX-COVERAGE block holding exactly the two --coverage lines."""
    first, rest = gcov_text.split("\n", 1)
    h0, h1 = "# MJX-GCOV-HEADER-BEGIN\n", "# MJX-GCOV-HEADER-END\n"
    if not rest.startswith(h0) or h1 not in rest or "#- MJX-COVERAGE" not in rest:
        return ["layout: header or coverage block markers missing"]
    body, block = rest.split(h1, 1)[1].split("#- MJX-COVERAGE", 1)
    probs = []
    if first + "\n" + body != oracle_text:
        probs.append("the copy differs from reference/optfile_levante_gfortran")
    flags = [ln for ln in block.splitlines()[1:] if ln.strip() and not ln.lstrip().startswith("#")]
    if flags != ['FFLAGS="$FFLAGS --coverage"', 'F90FLAGS="$F90FLAGS --coverage"']:
        probs.append(f"coverage block is {flags}")
    return probs


def test_gcov_optfile_is_oracle_plus_coverage():
    g = (REPO / "reference" / "coverage" / "optfile_levante_gfortran_gcov").read_text()
    o = (REPO / "reference" / "optfile_levante_gfortran").read_text()
    assert optfile_problems(g, o) == []
    # negative controls: an edited copy (-O0 -> -O2 in the IEEE branch, not in the header's prose) and an extra
    # flag are reported
    assert g.count("    FOPTIM='-O0'\n") == 1
    assert "differs" in optfile_problems(g.replace("    FOPTIM='-O0'\n", "    FOPTIM='-O2'\n"), o)[0]
    assert "coverage block" in optfile_problems(g.replace('--coverage"\n', '--coverage -O2"\n', 1), o)[0]


def test_arm_parser_and_cpp_decision():
    src = ["#ifdef HAVE_NETCDF", "a1", "# ifndef HAVE_NETCDF", "a2", "# else", "a3", "# endif", "#elif 1", "b1",
           "#else", "c1", "#endif", "  #ifdef NOT_A_DIRECTIVE_IN_TRADITIONAL_CPP", "x",
           "#if defined(WORDLENGTH) && WORDLENGTH == 4 \\", "    && !defined(MJX_UNDEFINED)", "w1", "#endif"]
    arms = cpp_live.parse_arms(src)
    assert [(a.line, a.kind, a.depth, a.parent, a.end, a.endif) for a in arms] == [
        (1, "ifdef", 0, None, 8, 12), (3, "ifndef", 1, 1, 5, 7), (5, "else", 1, 1, 7, 7), (8, "elif", 0, None, 10, 12),
        (10, "else", 0, None, 12, 12), (15, "if", 0, None, 18, 18)]
    assert arms[-1].last == 16                       # the backslash continuation belongs to the directive
    with pytest.raises(ValueError, match="never closed"):
        cpp_live.parse_arms(src[:6])
    with pytest.raises(ValueError, match="without #if"):
        cpp_live.parse_arms(["#endif"])
    # cpp decides (the oracle's DEFINES have -DHAVE_NETCDF -DWORDLENGTH=4); the indented #ifdef is plain text
    b = cpp_live.build_for("tutorial_barotropic_gyre", code="code")
    taken = {a.line: a.taken for a in cpp_live.evaluate_arms(b, "synthetic.F", src)}
    assert taken == {1: True, 3: False, 5: True, 8: False, 10: False, 15: True}
    live = cpp_live.live_mask(src, cpp_live.evaluate_arms(b, "synthetic.F", src))
    assert [i + 1 for i, v in enumerate(live) if v] == [2, 6, 13, 14, 17]
    # negative control: without -DHAVE_NETCDF on the command line the decision flips
    tc = dataclasses.replace(b.toolchain, defines=tuple(d for d in b.toolchain.defines if d != "-DHAVE_NETCDF"),
                             have_netcdf=False)
    flipped = {a.line: a.taken for a in cpp_live.evaluate_arms(dataclasses.replace(b, toolchain=tc), "s.F", src)}
    assert flipped != taken and flipped[1] is False and flipped[8] is True


def _barotropic_dynamics():
    exp, inp = "tutorial_barotropic_gyre", "input"
    cdir = run_top(exp, inp) / current()["collect"]
    assert (cdir / "COLLECTED").is_file(), f"no gcov collection {cdir} (reference/coverage/run_gcov.sbatch)"
    data = cv.load_collected(cdir)
    data["files"] = [d for d in data["files"] if d["f"] == "dynamics.f"]
    mapper = cv.Mapper(cpp_live.build_for(exp, code="code"), gcov_build(exp, inp) / "bld")
    return data, mapper


def test_planted_unexecuted_branch_reported():
    data, mapper = _barotropic_dynamics()
    rows = {r["routine"]: r for r in cv.routines(data, mapper)}
    dyn = rows["dynamics"]
    # known answers: DYNAMICS runs once per step (nTimeSteps=10, tutorial_barotropic_gyre/input/data); dynamics.F:250
    # `IF (debugMode) CALL DEBUG_ENTER( 'DYNAMICS', myThid )` runs, its CALL never (debugMode=.FALSE.)
    assert dyn["calls"] == 10 and dyn["file"] == "dynamics.F" and dyn["line"] == 21
    assert ["dynamics.F", 250, 1, 2] in dyn["never_taken"]
    # plant: an executed statement without branches gets count 0, an executed line with arcs gets an arc of 0
    _, lmap = mapper.line_map("dynamics.f")
    doc = data["files"][0]
    plain = next(e for e in doc["lines"] if e[1] > 0 and not e[3] and lmap[e[0] - 1][0] == "dynamics.F"
                 and lmap[e[0] - 1][1] > 250)
    branchy = next(e for e in doc["lines"] if e[1] > 0 and len(e[3]) == 2 and min(e[3]) > 0)
    p_loc, b_loc = lmap[plain[0] - 1], lmap[branchy[0] - 1]

    def covered(r, loc):
        return any(f == loc[0] and a <= loc[1] <= b for f, a, b in r["unexecuted"])

    assert not covered(dyn, p_loc) and [b_loc[0], b_loc[1], 1, 2] not in dyn["never_taken"]
    plain[1] = 0
    branchy[3][0] = 0
    dyn2 = {r["routine"]: r for r in cv.routines(data, mapper)}["dynamics"]
    assert covered(dyn2, p_loc), f"planted unexecuted {p_loc} not reported: {dyn2['unexecuted']}"
    assert [b_loc[0], b_loc[1], 1, 2] in dyn2["never_taken"], f"planted never-taken arc at {b_loc} not reported"
    assert dyn2["lines_run"] == dyn["lines_run"] - 1


def report_json(exp):
    f = paths.REFERENCE / "coverage" / exp / f"coverage-{current()['report']}.json.gz"
    assert f.is_file(), f"no coverage report {f} (reference/coverage/report_gcov.sbatch)"
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def test_m1_coverage_reports():
    problems = []
    for exp in sorted({e for e, _ in M1}):
        md = paths.REFERENCE / "coverage" / exp / f"coverage-{current()['report']}.md"
        doc = REPO / "docs" / "coverage" / f"{exp}.md"
        # the page is the report with the machine's locations as variables (cv.neutral_paths; reports written
        # before 2026-10-08 hold absolute paths), and it names no location of this machine
        if not doc.is_file() or doc.read_text() != cv.neutral_paths(md.read_text()):
            problems.append(f"{doc} is not the report {md}")
        elif any(str(v) in doc.read_text() for k, v in paths.ALL.items() if k != "MJX_PYTHON"):
            problems.append(f"{doc} names a machine path")
        rep = report_json(exp)
        problems += [f"{exp}: mapping problem {p}" for p in rep["problems"]]
        by = {v["input_dir"]: v for v in rep["variants"]}
        for e, inp in M1:
            if e != exp:
                continue
            v = by.get(inp)
            if v is None:
                problems.append(f"{exp}/{inp}: not in the report")
                continue
            if v["compare_plain"]["equal"] is not True:
                problems.append(f"{exp}/{inp}: gcov run vs plain oracle {v['compare_plain']}")
            main = [r for r in v["routines"] if r["routine"] == "the_model_main"]
            if not main or main[0]["calls"] != 1:
                problems.append(f"{exp}/{inp}: the_model_main not executed once")
            for c in v["physics"]:
                known = KNOWN_INACTIVE.get((exp, inp, c["check"]))
                if known is not None:
                    if c["ok"] or c["value"] != known:
                        problems.append(f"{exp}/{inp}: known-inactive check {c['check']!r} now {c}")
                elif not c["ok"]:
                    problems.append(f"{exp}/{inp}: physics-active check failed: {c}")
    assert not problems, "\n".join(problems)


def test_physics_and_comparison_negative_controls():
    """The physics-active and plain-oracle comparisons report planted errors (gcov run of the barotropic gyre)."""
    lines = sio.read_stdout(run_top("tutorial_barotropic_gyre", "input") / "rundir" / "output.txt")
    ok = cv.physics_checks("tutorial_barotropic_gyre", lines, None, [], None)
    assert all(c["ok"] for c in ok)
    # planted: eta frozen at a nonzero value (every dynstat_eta_sd equal to the last one) -> "eta" inactive;
    # (the first value is 0, which the nonzero test alone would catch)
    last = [ln.text for ln in lines if "%MON dynstat_eta_sd" in ln.text][-1]
    assert sio.fortran_number(last.split("=")[1]) != 0
    frozen = [dataclasses.replace(ln, text=last) if "%MON dynstat_eta_sd" in ln.text else ln for ln in lines]
    bad = {c["check"]: c["ok"] for c in cv.physics_checks("tutorial_barotropic_gyre", frozen, None, [], None)}
    assert bad == {"eta": False, "u": True, "v": True}
    # planted: one %MON digit changed -> the comparison with the plain oracle differs
    a = cv.compared_lines(lines)
    k = next(i for i, t in enumerate(a) if t.startswith("%MON dynstat_uvel_max"))
    b = list(a)
    b[k] = b[k][:-2] + ("1" if b[k][-2] != "1" else "2") + b[k][-1]
    assert a != b and cv.compared_lines(sio.read_stdout(cv.plain_oracle_output("tutorial_barotropic_gyre",
                                                                               "input"))) == a
    # the oracle records still agree on the preprocessor with the gcov builds present
    assert cpp_options.oracle_toolchain().have_netcdf
