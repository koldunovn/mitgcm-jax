"""tools/cpp_live.py (plan Task 5): the live code of a routine under an experiment's CPP options, and the map from the
compiled `.f` back to `file.F:line`.

Both tests run the oracle's preprocessor in lane D's link farms (seconds) and need the frozen oracle build records
(cpp_live.tool_toolchain = cpp_options.oracle_toolchain) and, for the `.f`, lane A's plain advect_xz build directory:
they fail, not skip, when those are missing. Each carries its negative control: the same check applied to a planted error must report it.
"""

import dataclasses
import re

import pytest

from mitjax import paths
from mitjax.config import cpp_options
from tools import cpp_live

UP7 = "63cdc0b"

# (routine file, directive line, directive text, experiment/code where the arm is taken, where it is not)
#   calc_r_star.F:40 `#ifdef NONLIN_FRSURF`: advect_xz/code/CPP_OPTIONS.h defines NONLIN_FRSURF, the barotropic gyre
#   (model/inc/CPP_OPTIONS.h) does not;
#   forward_step.F:4 `#ifdef ALLOW_AUTODIFF`: tutorial_global_oce_optim/code_ad compiles pkg/autodiff,
#   global_ocean.90x40x15/code does not.
PAIRS = [
    ("calc_r_star.F", 40, "#ifdef NONLIN_FRSURF", ("advect_xz", "code"), ("tutorial_barotropic_gyre", "code")),
    ("forward_step.F", 4, "#ifdef ALLOW_AUTODIFF", ("tutorial_global_oce_optim", "code_ad"),
     ("global_ocean.90x40x15", "code")),
]
# a line inside the NONLIN_FRSURF arm: live in advect_xz, absent in the barotropic gyre
LIVE_LINE = ("calc_r_star.F", 82, "rStarFacNm1C(i,j,bi,bj) = rStarFacC(i,j,bi,bj)")


def _arm(build, name, line):
    arms = {a.line: a for a in cpp_live.evaluate_arms(build, name)}
    return arms[line]


def pair_problems(build_on, build_off, name, line, text):
    """Problems of one pair: the directive at `line` must be `text`, taken in build_on and not in build_off."""
    probs = []
    for b, want in ((build_on, True), (build_off, False)):
        a = _arm(b, name, line)
        if a.text.strip() != text:
            probs.append(f"{b.experiment}: {name}:{line} is {a.text!r}, not {text!r}")
        if a.taken is not want:
            probs.append(f"{b.experiment}/{b.code}: {name}:{line} {text} taken={a.taken}, expected {want}")
    return probs


def test_viewer_differs_between_experiments():
    builds = {}
    for _, _, _, on, off in PAIRS:
        for exp, code in (on, off):
            builds[(exp, code)] = cpp_live.build_for(exp, code=code)
    problems = []
    for name, line, text, on, off in PAIRS:
        problems += pair_problems(builds[on], builds[off], name, line, text)
    # the live listing: the line inside the arm is shown with its .F number in one experiment only
    name, ln, code_text = LIVE_LINE
    xz = cpp_live.live_view(builds[("advect_xz", "code")], "calc_r_star")
    baro = cpp_live.live_view(builds[("tutorial_barotropic_gyre", "code")], "CALC_R_STAR")
    xz_lines = {k: t for k, t, _ in xz.lines}
    assert xz_lines.get(ln, "").strip() == code_text, f"advect_xz: {name}:{ln} not live or not {code_text!r}"
    assert ln not in {k for k, _, _ in baro.lines}, f"barotropic gyre: {name}:{ln} must not be live"
    text_xz = cpp_live.format_view(builds[("advect_xz", "code")], xz)
    assert f"{name}:{ln} " in text_xz and re.search(rf"^{name}:40\s+taken\s+#ifdef NONLIN_FRSURF", text_xz, re.M)
    assert re.search(rf"^{name}:40\s+NOT taken\s+#ifdef NONLIN_FRSURF",
                     cpp_live.format_view(builds[("tutorial_barotropic_gyre", "code")], baro), re.M)
    # cpp_text (used for the arm-marked copy) runs exactly cpp_options.preprocess's command
    b = builds[("advect_xz", "code")]
    assert cpp_live.cpp_text(b, (b.path / name).read_text(encoding="latin-1")) == \
        cpp_options.preprocess(b.farm, b.toolchain, name, line_markers=True)
    assert not problems, "\n".join(problems)

    # negative control: a viewer that preprocessed with another experiment's options (advect_xz's routine viewed in
    # the barotropic gyre's link farm) must be caught by the same check. (Adding -DNONLIN_FRSURF to the barotropic
    # gyre's command line is not an error that bites: model/inc/CPP_OPTIONS.h `#undef NONLIN_FRSURF` wins.)
    on, off = builds[("advect_xz", "code")], builds[("tutorial_barotropic_gyre", "code")]
    planted = dataclasses.replace(on, farm=off.farm)
    got = pair_problems(planted, off, "calc_r_star.F", 40, "#ifdef NONLIN_FRSURF")
    assert got and "advect_xz/code: calc_r_star.F:40 #ifdef NONLIN_FRSURF taken=False, expected True" in got[0], got


def known_line_problems(lmap, f_lines, src_lines):
    """The .f -> .F map on calc_r_star.f: the known statement maps to calc_r_star.F:82, the sNx line of SIZE.h to
    its SIZE.h line, and every .f line mapped to calc_r_star.F that has no macro text equals its .F line."""
    probs = []
    k = [i for i, t in enumerate(f_lines) if t.strip() == LIVE_LINE[2]]
    if len(k) != 1:
        return [f"expected one .f line {LIVE_LINE[2]!r}, found {len(k)}"]
    if lmap[k[0]] != (LIVE_LINE[0], LIVE_LINE[1]):
        probs.append(f".f line {k[0] + 1} maps to {lmap[k[0]]}, expected {LIVE_LINE[:2]}")
    size_hits = [i for i in range(len(f_lines)) if lmap[i] and lmap[i][0] == "SIZE.h"
                 and "sNx =" in f_lines[i]]
    if not size_hits:
        probs.append("no .f line with 'sNx =' maps to SIZE.h")
    for i in size_hits:
        f, ln = lmap[i]
        if "sNx =" not in src_lines["SIZE.h"][ln - 1]:
            probs.append(f".f line {i + 1} maps to SIZE.h:{ln} = {src_lines['SIZE.h'][ln - 1]!r}")
    same = diff = 0
    for i, loc in enumerate(lmap):
        if loc and loc[0] == "calc_r_star.F" and f_lines[i].strip():
            src = src_lines["calc_r_star.F"][loc[1] - 1]
            if re.search(r"\b_[A-Za-z]\w*|_d\b", src):
                continue                                # macro or _d constant: text changes by design
            same, diff = (same + 1, diff) if src == f_lines[i] else (same, diff + 1)
    if diff or same < 150:
        probs.append(f"{diff} mapped lines differ from their .F text ({same} equal)")
    return probs


def test_compiled_line_map_known_lines():
    recs = sorted(p for p in (paths.REFERENCE / "build").glob(f"advect_xz-code-{UP7}-*")
                  if re.fullmatch(rf"advect_xz-code-{UP7}-[0-9a-f]{{7}}", p.name))
    assert recs, f"no plain advect_xz build directory under {paths.REFERENCE / 'build'} (reference/jobs/build.sbatch)"
    f_text = (recs[-1] / "bld" / "calc_r_star.f").read_text(encoding="latin-1")
    b = cpp_live.build_for("advect_xz", code="code")
    lmap = cpp_live.compiled_line_map(b, "calc_r_star.F", f_text)
    f_lines = cpp_live.output_lines(f_text)
    assert len(lmap) == len(f_lines)
    src = {n: cpp_live.read_source(b, n) for n in ("calc_r_star.F", "SIZE.h")}
    assert known_line_problems(lmap, f_lines, src) == []

    # negative controls: a .f that is not what the farm produces is refused; a map shifted by one line is caught
    bad = f_text.replace("rStarFacNm1C(i,j,bi,bj) = rStarFacC", "rStarFacNm1C(i,j,bi,bj) = rStarFacW", 1)
    with pytest.raises(ValueError, match="differs from the build's .f"):
        cpp_live.compiled_line_map(b, "calc_r_star.F", bad)
    shifted = [None] + lmap[:-1]
    got = known_line_problems(shifted, f_lines, src)
    assert got and "expected ('calc_r_star.F', 82)" in got[0], got
