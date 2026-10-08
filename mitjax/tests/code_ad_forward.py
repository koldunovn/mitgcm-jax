"""Forward differences between the code_ad build of tutorial_global_oce_optim and the same build without the AD
packages, in the routines the experiment executes (lane ctrl; docs/R5_CODE_AD_FORWARD.md).

    python mitjax/tests/code_ad_forward.py [--markdown]

"Plain" = the same code_ad directory with `autodiff cost ctrl grdchk` removed from packages.conf (genmake2 then
writes `#undef ALLOW_AUTODIFF`, `ALLOW_COST`, `ALLOW_CTRL`, `ALLOW_GRDCHK` into PACKAGES_CONFIG.h, and every option
header that keys on them -- AUTODIFF_OPTIONS.h, COST_OPTIONS.h, CTRL_OPTIONS.h -- defines nothing). Both builds are
link farms of mitjax/config/cpp_options (the project's one CPP module); each file's live lines come from the build's
own cpp (cpp_options.live_lines: arm markers, never by reading #define lines). The executed routines are the rows of
docs/coverage/tutorial_global_oce_optim.md (gcov of the code_ad build, job 27827383). For every executed routine of a
file both builds compile, the lines live in one build only are grouped into blocks (consecutive differing lines; dead
and blank lines between them do not split a block) and classified:
  taf      every non-blank line is a comment (C/c/*/!) -- CADJ store directives, TAF comments: no forward effect
  include  only #include lines (declarations; a header's own AD branches are listed separately, see --headers)
  code     anything else: a forward branch that differs (what R5 needs from the lane that owns the routine)
Routines only the code_ad build compiles (pkg/autodiff, cost, ctrl, grdchk, code_ad/cost_*.F) are listed by name.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mitjax import paths  # noqa: E402
from mitjax.config import cpp_options, packages  # noqa: E402
from tools import cpp_live  # noqa: E402

EXP = "tutorial_global_oce_optim"
AD_PKGS = ("autodiff", "cost", "ctrl", "grdchk")
COVERAGE = REPO / "docs" / "coverage" / f"{EXP}.md"
_ROW = re.compile(r"^\| `(\w+)` \| ([\w.]+):(\d+) \|")
_ROUTINE = re.compile(r"^[ \t]+(?:[A-Za-z_*0-9()]+[ \t]+)*?(SUBROUTINE|FUNCTION|PROGRAM)[ \t]+(\w+)", re.I)
_INCLUDE = re.compile(r"^#\s*include\b")


def executed_routines():
    """[(routine, file, first line)] of the coverage worklist's executed-routine tables."""
    out = []
    for ln in COVERAGE.read_text().splitlines():
        m = _ROW.match(ln)
        if m:
            out.append((m.group(1), m.group(2), int(m.group(3))))
    return out


def plain_mods():
    """A mods directory: links to every file of code_ad but packages.conf, which is written without AD_PKGS. Kept
    under $MJX_RUNS/ctrl (named by its content; never deleted)."""
    code = paths.UPSTREAM / "verification" / EXP / "code_ad"
    conf = (code / "packages.conf").read_text()
    lines = [ln for ln in conf.splitlines() if ln.split("#")[0].strip() not in AD_PKGS]
    text = "\n".join(lines) + "\n"
    import hashlib
    d = paths.RUNS / "ctrl" / f"plain_mods-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
    if not d.exists():
        d.mkdir(parents=True)
        for p in sorted(code.iterdir()):
            if p.name != "packages.conf":
                (d / p.name).symlink_to(p)
        (d / "packages.conf").write_text(text)
    return d


def builds():
    tc = cpp_options.default_toolchain()
    ad = cpp_live.build_for(EXP, code="code_ad", toolchain=tc)
    mods = [plain_mods()]
    pset = packages.package_set(paths.UPSTREAM, mods, have_netcdf=tc.have_netcdf)
    for p in AD_PKGS:
        if p in pset.packages:
            raise RuntimeError(f"plain build still compiles pkg/{p}")
    farm = cpp_options.make_farm(f"{EXP}-code_ad-noad", paths.UPSTREAM, mods, pset, tc)
    return ad, cpp_live.Build(EXP, "code_ad-noad", farm, tc)


def routine_ranges(src):
    starts = [(i + 1, m.group(2).lower()) for i, ln in enumerate(src) if (m := _ROUTINE.match(ln))
              and src[i][:1] not in "Cc*!"]
    out = {}
    for k, (n, name) in enumerate(starts):
        out[name] = (n, starts[k + 1][0] - 1 if k + 1 < len(starts) else len(src))
    return out


_DECL = re.compile(r"^\s+(INTEGER|LOGICAL|_RL|_RS|REAL\*8|CHARACTER\*?\S*)\s+\w", re.I)
_KEY = re.compile(r"^\s+(\w*key\w*|ilev_\d|act\d|max\d|iloop)\s*=", re.I)


def classify(texts):
    body = [t for t in texts if t.strip() and t[:1] not in "Cc*!"]
    if not body:
        return "taf"
    if all(_INCLUDE.match(t) for t in body):
        return "include"
    if all(_DECL.match(t) or _INCLUDE.match(t) for t in body):
        return "decl"
    if all(_KEY.match(t) or _DECL.match(t) for t in body):
        return "tafkey"
    return "code"


def diff_file(ad, plain, name, ranges):
    """[(routine, sign, first, last, kind, arms text, first code line)] for one file."""
    src, live_ad, arms_ad = cpp_options.live_lines(ad.farm, ad.toolchain, name)
    src_p, live_p, arms_p = cpp_options.live_lines(plain.farm, plain.toolchain, name)
    if src != src_p:
        raise RuntimeError(f"{name}: the two builds compile different files")
    out = []
    for routine, (lo, hi) in ranges:
        for sign, live, other, arms in (("+", live_ad, live_p, arms_ad), ("-", live_p, live_ad, arms_p)):
            block = []
            for n in range(lo, hi + 1):
                differs = live[n - 1] and not other[n - 1] and src[n - 1].strip() != ""
                if differs:
                    block.append(n)
                elif block and src[n - 1].strip() and (live[n - 1] or other[n - 1]):
                    out.append(_block(routine, sign, block, src, arms))
                    block = []
            if block:
                out.append(_block(routine, sign, block, src, arms))
    return out


def _block(routine, sign, block, src, arms):
    texts = [src[n - 1] for n in block]
    kind = classify(texts)
    encl = cpp_options.enclosing_arms(arms, block[0])
    arm = " / ".join(f":{a.line} {a.text.splitlines()[0].strip()}" for a in encl if a.taken)
    first = next((t.strip() for t in texts if t.strip() and t[:1] not in "Cc*!"), texts[0].strip())
    return (routine, sign, block[0], block[-1], kind, arm, first, len(block))


def analyse():
    ad, plain = builds()
    rows, only_ad = [], []
    by_file = {}
    for routine, fname, first in executed_routines():
        by_file.setdefault(fname, []).append((routine, first))
    for fname, routines in sorted(by_file.items()):
        if not (plain.path / fname).exists():
            only_ad += [(r, fname, f) for r, f in routines]
            continue
        if not (ad.path / fname).exists():
            raise FileNotFoundError(f"{fname} not in the code_ad farm")
        src = cpp_options.read_source(ad.farm, fname)
        rr = routine_ranges(src)
        ranges = []
        for r, f in routines:
            hit = [v for k, v in rr.items() if v[0] == f] or ([rr[r.lower()]] if r.lower() in rr else [])
            if not hit:
                raise KeyError(f"{r} at {fname}:{f}: no routine statement there")
            ranges.append((r, hit[0]))
        try:
            rows += [(fname,) + x for x in diff_file(ad, plain, fname, ranges)]
        except RuntimeError as e:          # code_ad/cost_*.F include cost.h: they need pkg/cost
            if "No such file" not in str(e):
                raise
            only_ad += [(r, fname, f) for r, f in routines]
    return ad, plain, rows, only_ad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kinds", default="taf,include,decl,tafkey,code")
    ap.add_argument("--text", action="store_true", help="print the live lines of each block")
    a = ap.parse_args(argv)
    ad, plain, rows, only_ad = analyse()
    kinds = set(a.kinds.split(","))
    print(f"# code_ad farm {ad.path}\n# plain farm   {plain.path}")
    for r in rows:
        fname, routine, sign, lo, hi, kind, arm, first, n = r
        if kind in kinds:
            print(f"{kind:7s} {sign} {fname}:{lo}-{hi} ({n}) [{routine}] {arm} || {first}")
            if a.text:
                b = ad if sign == "+" else plain
                src, live, _ = cpp_options.live_lines(b.farm, b.toolchain, fname)
                for k in range(lo, hi + 1):
                    if live[k - 1] and src[k - 1].strip() and src[k - 1][:1] not in "Cc*!":
                        print(f"        {k:5d} {src[k - 1]}")
    print(f"# only in code_ad: {len(only_ad)} routines: " + ", ".join(f"{r} ({f})" for r, f, _ in only_ad))
    from collections import Counter
    print("# blocks:", Counter(r[5] for r in rows))


if __name__ == "__main__":
    main()
