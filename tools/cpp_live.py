#!/usr/bin/env python3
"""The live code of a routine under an experiment's CPP options, with the `.F` line numbers kept (plan Task 5).

    cpp_live.py EXPERIMENT ROUTINE [--variant INPUT_DIR] [--code code|code_ad] [--expanded] [--includes]

ROUTINE is a subroutine/function name (`calc_r_star`, case-insensitive) or a file name (`calc_r_star.F`). The code
directory is --code, else the one testreport uses for --variant (input_ad[.v] -> code_ad, input[.v] -> code;
mitjax/config/namelists.code_dir_for), else `code`.

How (never by reading `#define` lines [E§2], [L-ORA-12]): the project's single CPP module, lane D's
mitjax/config/cpp_options.py, rebuilds the build's flat directory (link farm: the experiment's code directory, its
packages, eesupp, model, generated PACKAGES_CONFIG.h/AD_CONFIG.h) and runs the oracle build's own preprocessor
(`cpp -traditional`, DEFINES, INCLUDES from the frozen build record) the way genmake2's Makefile does
(`cat file | cpp ...`, MJX_UPSTREAM tools/genmake2:3380), here WITHOUT `-P`, so the output carries `# <line>
"<file>"` markers that map every output line back to `file.F:line`.

Liveness of each `#if/#ifdef/#ifndef/#elif/#else` arm is decided by cpp itself (library functions in
mitjax/config/cpp_options.py, shared with mitjax/params_io.fortran_default): a marker line `MJXARM_<n>` is put
after the directive at line n, the text is preprocessed the same way, and the arm is taken iff its marker survives.
A source line is live iff every arm enclosing it is taken; conditional directive lines themselves are not code.
Traditional cpp recognises a directive only with `#` in column 1 (`# ifdef` counts, `  #ifdef` does not: measured
with the build's cpp), and so does this parser.

Output: the live, non-blank lines of the routine prefixed with `file.F:line` (source text, or with --expanded the
text cpp produces: macros such as _RL expanded), `[+N lines of X.h]` where an #include brought lines in (--includes
lists them with `X.h:line`), then the list of conditional arms: taken / NOT taken / dead (inside an arm that is not taken).

Also used by tools/coverage.py: `compiled_line_map` maps every line of the compiled `.f` (genmake2 compiles the
`cpp -P` output piped through tools/set64bitConst.sh, Makefile `.F.f` rule) back to `file.F:line`, after checking
that the regenerated `.f` equals the build's own `.f` byte for byte.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mitjax import paths  # noqa: E402
from mitjax.config import cpp_options, namelists, packages  # noqa: E402
from mitjax.config.cpp_options import (Arm, arm_marked_text, live_mask, output_lines,  # noqa: E402,F401
                                       parse_arms)

_MARKER = cpp_options._LINE_MARKER
_ROUTINE = re.compile(r"^[ \t]+(?:[A-Za-z_*0-9()]+[ \t]+)*?(SUBROUTINE|FUNCTION)[ \t]+(\w+)", re.I)
STDIN = "<stdin>"


# ---------------------------------------------------------------------------------------------------------------
# the build: farm + toolchain

@dataclass(frozen=True)
class Build:
    experiment: str
    code: str
    farm: cpp_options.Farm          # the flat directory cpp runs in (lane D's link farm, or a genmake2 build dir)
    toolchain: cpp_options.Toolchain

    @property
    def path(self):
        return Path(self.farm.path)

    def source_of(self, name):
        """Upstream path of a farm file (relative to the clone when it is one), else the file itself."""
        target = self.farm.manifest.get("links", {}).get(name)
        if target is None:
            return str(self.path / name)
        root = self.farm.manifest.get("rootdir", "")
        return target[len(root) + 1:] if root and target.startswith(root + "/") else target


def tool_toolchain():
    """The Fortran oracle's preprocessor where its build records exist (the compiled `.f` files and gcov data this tool
    maps were made by it; cpp_options.oracle_toolchain), else the system toolchain the model uses (default_toolchain;
    docs plan 20261006 Task 4: both give every verification build the same text, test_toolchain_system.py)."""
    try:
        return cpp_options.oracle_toolchain()
    except FileNotFoundError:
        return cpp_options.default_toolchain()


def build_for(experiment, code=None, variant=None, upstream=None, toolchain=None):
    """The link farm and toolchain of EXPERIMENT/<code> (as mitjax/config/params.load makes them; the toolchain is
    tool_toolchain() unless given)."""
    upstream = Path(paths.UPSTREAM if upstream is None else upstream)
    code = code or namelists.code_dir_for(variant or "input")
    tc = toolchain or tool_toolchain()
    mods = [upstream / "verification" / experiment / code]
    if not mods[0].is_dir():
        raise FileNotFoundError(f"no {mods[0]}")
    pset = packages.package_set(upstream, mods, have_netcdf=tc.have_netcdf)
    farm = cpp_options.make_farm(f"{experiment}-{code}", upstream, mods, pset, tc)
    return Build(experiment, code, farm, tc)


def build_dir_view(build, bld_dir):
    """The same toolchain run in a genmake2 build directory (for template-generated sources the farm lacks)."""
    return Build(build.experiment, build.code, cpp_options.Farm(Path(bld_dir), {}), build.toolchain)


def cpp_text(build, text):
    """`cpp` with line markers of a text given on stdin in the build's flat directory:
    cpp_options.preprocess_text(..., line_markers=True), the command cpp_options.preprocess runs on a farm file.
    test_cpp_live checks both agree on an unmodified file."""
    return cpp_options.preprocess_text(build.farm, build.toolchain, text, line_markers=True)


# ---------------------------------------------------------------------------------------------------------------
# line markers

def source_map(text, main):
    """[(file, line, content)] for every non-marker line of cpp output with line markers; `<stdin>` -> main."""
    out, cur, n = [], None, 0
    for raw in output_lines(text):
        m = _MARKER.match(raw)
        if m:
            cur, n = m.group(2), int(m.group(1))
            continue
        if cur is None:
            raise ValueError(f"cpp output line before any line marker: {raw!r}")
        out.append((main if cur == STDIN else cur, n, raw))
        n += 1
    return out


# ---------------------------------------------------------------------------------------------------------------
# conditional arms

def evaluate_arms(build, name, src_lines=None):
    """Arms of farm file `name` with `taken` decided by cpp (cpp_options.evaluate_arms)."""
    return cpp_options.evaluate_arms(build.farm, build.toolchain, name, src_lines)


def read_source(build, name):
    return cpp_options.read_source(build.farm, name)


# ---------------------------------------------------------------------------------------------------------------
# routines

def find_routine(build, routine):
    """(file name, first line, last line) of a routine: the file `<routine>.F` (whole file), else the .F file of the
    farm whose SUBROUTINE/FUNCTION statement names it (lines up to the next such statement)."""
    if routine.endswith((".F", ".F90", ".h")):
        lines = read_source(build, routine)
        return routine, 1, len(lines)
    name = routine.lower()
    if (build.path / f"{name}.F").is_file():
        return f"{name}.F", 1, len(read_source(build, f"{name}.F"))
    hits = []
    for p in sorted(build.path.glob("*.F")):
        lines = output_lines(p.read_text(encoding="latin-1"))
        starts = [(i + 1, m.group(2).lower()) for i, ln in enumerate(lines) if (m := _ROUTINE.match(ln))]
        for k, (ln_no, rname) in enumerate(starts):
            if rname == name:
                end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(lines)
                hits.append((p.name, ln_no, end))
    if not hits:
        raise KeyError(f"no routine {routine!r} in {build.experiment}/{build.code} ({build.path})")
    if len(hits) > 1:
        raise KeyError(f"routine {routine!r} defined more than once: {hits}")
    return hits[0]


# ---------------------------------------------------------------------------------------------------------------
# the view

@dataclass(frozen=True)
class LiveView:
    name: str
    first: int
    last: int
    source: str
    lines: list          # [(line, source text, expanded text or None)] of the live lines in range
    includes: dict       # line of the #include -> [(file, line, text)] of non-blank lines it brought in
    arms: list           # arms in range


def include_lines(text):
    """{line of an #include in the main file: [(file, line, text)] of the non-blank lines it brought in}: header
    lines are collected until the marker that returns to `<stdin>` (flag 2) at line R, i.e. the #include at R-1."""
    out, pending, cur, n = {}, [], None, 0
    for raw in output_lines(text):
        m = _MARKER.match(raw)
        if m:
            cur, n = m.group(2), int(m.group(1))
            if cur == STDIN and "2" in m.group(3).split() and pending:
                out.setdefault(n - 1, []).extend(pending)
                pending = []
            continue
        if cur != STDIN and raw.strip() and not cur.startswith(("<", "/usr/include")):
            pending.append((cur, n, raw))
        n += 1
    return out


def live_view(build, routine):
    name, first, last = find_routine(build, routine)
    src = read_source(build, name)
    arms = evaluate_arms(build, name, src)
    live = live_mask(src, arms)
    out = cpp_options.preprocess(build.farm, build.toolchain, name, line_markers=True)
    expanded = {}
    for f, ln, text in source_map(out, name):
        if f == name and text.strip():
            if not live[ln - 1]:
                raise ValueError(f"{name}:{ln}: cpp emitted {text!r} for a line the arm analysis calls dead")
            expanded[ln] = text
    lines = [(k, src[k - 1], expanded.get(k)) for k in range(first, last + 1) if live[k - 1] and src[k - 1].strip()]
    inc = {k: v for k, v in include_lines(out).items() if first <= k <= last}
    return LiveView(name, first, last, build.source_of(name), lines, inc,
                    [a for a in arms if first <= a.line <= last])


def format_view(build, view, expanded=False, show_includes=False):
    w = len(f"{view.name}:{view.last}")
    out = [f"# cpp_live {build.experiment}/{build.code}: {view.name} lines {view.first}-{view.last} "
           f"(source {view.source} @ {build.farm.manifest.get('upstream_commit', '?')[:7]})",
           f"# preprocessor: {' '.join(a for a in build.toolchain.cpp if a != '-P')} DEFINES INCLUDES "
           f"(record {build.toolchain.record}); cwd {build.path}",
           f"# {len(view.lines)} live non-blank lines; {sum(a.taken for a in view.arms)} of {len(view.arms)} "
           f"conditional arms taken"]
    inc_after = view.includes
    for ln, text, exp in view.lines:
        shown = exp if (expanded and exp is not None) else text
        if expanded and exp is None:
            shown = text + "      ! [expands to blank]"
        out.append(f"{view.name + ':' + str(ln):<{w}}  {shown}")
        if ln in inc_after:
            files = sorted({f for f, _, _ in inc_after[ln]})
            if show_includes:
                out += [f"{'  ' + f + ':' + str(l):<{w}}  {t}" for f, l, t in inc_after[ln]]
            else:
                out.append(f"{'':<{w}}  [+{len(inc_after[ln])} lines of {', '.join(files)}]")
    out.append(f"# conditional arms of {view.name} lines {view.first}-{view.last}")
    by_line = {a.line: a for a in view.arms}
    for a in view.arms:
        outer_dead = a.parent is not None and a.parent in by_line and not by_line[a.parent].taken
        tag = "taken    " if a.taken else ("dead     " if outer_dead else "NOT taken")
        out.append(f"{view.name + ':' + str(a.line):<{w}}  {tag}  {'  ' * a.depth}{a.text.splitlines()[0]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# compiled .f -> .F

def set64bit(text, upstream=None):
    """tools/set64bitConst.sh (the Makefile's CPPCMD post-filter; line-preserving sed)."""
    script = Path(paths.UPSTREAM if upstream is None else upstream) / "tools" / "set64bitConst.sh"
    res = subprocess.run(["sh", str(script)], input=text.encode("latin-1"), capture_output=True, check=True)
    return res.stdout.decode("latin-1")


def compiled_line_map(build, name, f_text=None):
    """[(file, line) or None] for each line of the compiled `<stem>.f` of farm file `name`: the `cpp -P` output
    (cpp_options.preprocess) piped through set64bitConst.sh must equal `f_text` (the build's .f) byte for byte when
    given; its lines are aligned with the line-marker output, whose non-blank lines must be the same sequence."""
    plain = cpp_options.preprocess(build.farm, build.toolchain, name)
    if f_text is not None:
        regen = set64bit(plain, build.farm.manifest.get("rootdir"))
        if regen != f_text:
            raise ValueError(f"{name}: regenerated .f differs from the build's .f (farm {build.path})")
    smap = source_map(cpp_options.preprocess(build.farm, build.toolchain, name, line_markers=True), name)
    p_lines = output_lines(plain)
    out, j = [], 0
    for k, text in enumerate(p_lines, start=1):
        if text.strip():
            while j < len(smap) and not smap[j][2].strip():
                j += 1
            if j >= len(smap) or smap[j][2] != text:
                got = smap[j][2] if j < len(smap) else "<end>"
                raise ValueError(f"{name}: .f line {k} {text!r} does not align with marker output {got!r}")
            out.append((smap[j][0], smap[j][1]))
            j += 1
        else:
            out.append(None)
    rest = [s for s in smap[j:] if s[2].strip()]
    if rest:
        raise ValueError(f"{name}: {len(rest)} non-blank marker-output lines left after the .f, first {rest[0]}")
    return out


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment")
    ap.add_argument("routine")
    ap.add_argument("--variant", help="input directory (input, input.<v>, input_ad, ...): picks the code directory")
    ap.add_argument("--code", choices=("code", "code_ad"))
    ap.add_argument("--expanded", action="store_true", help="print cpp's text (macros expanded)")
    ap.add_argument("--includes", action="store_true", help="list the lines included headers bring in")
    a = ap.parse_args(argv)
    build = build_for(a.experiment, code=a.code, variant=a.variant)
    print(format_view(build, live_view(build, a.routine), expanded=a.expanded, show_includes=a.includes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
