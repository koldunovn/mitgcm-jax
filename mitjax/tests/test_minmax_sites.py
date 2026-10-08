"""Fortran MAX/MIN in the physics code: every call states gfortran's per-site winner of ties and NaN (lane MINMAX).

The oracle is the site table of each M1 and M2 build, $MJX_REFERENCE/minmax_sites/<build>.json
(tools/fortran_minmax_sites.py: the oracle's own compilation of every REAL MAX/MIN; the winner is the source operand
of the emitted maxsd/minsd).
1. Tables: present for the six M1 and ten M2 builds, sha256 as recorded, objects byte-identical to the builds', no
   unresolved site, and every site has the same winner in every build that compiles it.
2. Audit of mitjax/model and mitjax/pkg (static, `ast`): every call of the helpers (`MAX`/`MIN` of
   mitjax.ops.fortran_minmax and fortran_minmax_host, `max_chain`/`min_chain`, and the statement-function ports of
   STATEMENT_FUNCTIONS) carries a literal `p=` and a `# :NN` / `# :NN-MM` citation of the Fortran statement (a comment
   on a line of the enclosing Python statement), the cited statement of the routine's `.F` (the enclosing function's
   docstring, else the module docstring, names it) holds a site of that operation and argument count, and `p` equals
   the table. A raw jnp/np maximum/minimum/fmax/fmin/clip or lax.max/min needs a `MINMAX-RAW: <reason>` comment, a
   builtin max/min (host integers) a `MINMAX-INT: <why>` comment.
   Negative controls: a flipped `p`, a missing citation, a raw jnp.maximum, each measured to fail.
3. Values: the helpers reproduce gfortran bit for bit on +0/-0 ties and NaN at probe sites of both winners
   (mitjax/tests/fortran_minmax/: mmsites.F, sfprobe.F with their outputs and the tool's extraction probe_sites.json,
   and the oracle's own mon_advcflw2.o run in situ, insitu_mon_advcflw2.F).
Seconds; numpy + jax on CPU. Fails (not skips) without the tables.
"""

import ast
import functools
import hashlib
import io
import json
import re
import struct
import tokenize
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mitjax import paths
from mitjax.ops import fortran_minmax_host as host
from mitjax.ops.fortran_minmax import MAX, MAX_CHAIN, MIN
from tools import fortran_minmax_sites as fms

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "fortran_minmax"
M1_BUILDS = ("advect_xy-code-63cdc0b-f48c062", "advect_xz-code-63cdc0b-f48c062",
             "global_ocean.90x40x15-code-63cdc0b-f48c062", "tutorial_baroclinic_gyre-code-63cdc0b-f48c062",
             "tutorial_barotropic_gyre-code-63cdc0b-f48c062", "tutorial_global_oce_optim-code_ad-63cdc0b-f48c062")
# M2 oracle builds (lane A sessions 6-8, jobs 27833815/27833918): the audit covers the code the M2 lanes port
M2_BUILDS = ("adjustment.cs-32x32x1-code-63cdc0b-06b4417", "adjustment.cs-32x32x1-code_min-63cdc0b-78ca390",
             "advect_cs-code-63cdc0b-06b4417", "global_ocean.90x40x15-code_ad-63cdc0b-06b4417",
             "global_ocean.cs32x15-code-63cdc0b-06b4417", "global_ocean.cs32x15-code_ad-63cdc0b-78ca390",
             "solid-body.cs-32x32x1-code-63cdc0b-06b4417", "tutorial_advection_in_gyre-code-63cdc0b-06b4417",
             "tutorial_global_oce_latlon-code-63cdc0b-06b4417", "tutorial_tracer_adjsens-code_ad-63cdc0b-06b4417")
# M3 (lane A session 9, reference/jobs/minmax_sites.sbatch job 27840408); global_ocean.90x40x15's input.dwnslp and
# input.idemix run the M1 code build, whose table is M1's
M3_BUILDS = ("MLAdjust-code-63cdc0b-463504e", "front_relax-code-63cdc0b-463504e", "ideal_2D_oce-code-63cdc0b-463504e",
             "tutorial_reentrant_channel-code-63cdc0b-463504e", "vermix-code-63cdc0b-463504e")
# M4 (lane A session 11, reference/jobs/minmax_sites.sbatch job 27856188): plain builds 704fd6b of the M4 codes;
# global_ocean.cs32x15's sea-ice variants run the M2 builds, whose tables are M2's
M4_BUILDS = ("1D_ocean_ice_column-code-63cdc0b-704fd6b", "1D_ocean_ice_column-code_ad-63cdc0b-704fd6b",
             "lab_sea-code-63cdc0b-704fd6b", "lab_sea-code_ad-63cdc0b-704fd6b",
             "offline_exf_seaice-code-63cdc0b-704fd6b", "offline_exf_seaice-code_ad-63cdc0b-704fd6b",
             "seaice_itd-code-63cdc0b-704fd6b")
BUILDS = M1_BUILDS + M2_BUILDS + M3_BUILDS + M4_BUILDS
# (compiled files, REAL MAX/MIN sites) of each M2 table as lane A's jobs measured them: a truncated table fails
M2_COUNTS = {"adjustment.cs-32x32x1-code-63cdc0b-06b4417": (657, 199),
             "adjustment.cs-32x32x1-code_min-63cdc0b-78ca390": (298, 8),
             "advect_cs-code-63cdc0b-06b4417": (749, 347), "global_ocean.90x40x15-code_ad-63cdc0b-06b4417": (766, 327),
             "global_ocean.cs32x15-code-63cdc0b-06b4417": (972, 588),
             "global_ocean.cs32x15-code_ad-63cdc0b-78ca390": (1074, 455),
             "solid-body.cs-32x32x1-code-63cdc0b-06b4417": (728, 321),
             "tutorial_advection_in_gyre-code-63cdc0b-06b4417": (675, 352),
             "tutorial_global_oce_latlon-code-63cdc0b-06b4417": (655, 352),
             "tutorial_tracer_adjsens-code_ad-63cdc0b-06b4417": (787, 389)}
M3_COUNTS = {"MLAdjust-code-63cdc0b-463504e": (815, 385), "front_relax-code-63cdc0b-463504e": (654, 363),
             "ideal_2D_oce-code-63cdc0b-463504e": (659, 383),
             "tutorial_reentrant_channel-code-63cdc0b-463504e": (672, 364), "vermix-code-63cdc0b-463504e": (703, 412)}
M4_COUNTS = {"1D_ocean_ice_column-code-63cdc0b-704fd6b": (732, 393),
             "1D_ocean_ice_column-code_ad-63cdc0b-704fd6b": (903, 376),
             "lab_sea-code-63cdc0b-704fd6b": (899, 474), "lab_sea-code_ad-63cdc0b-704fd6b": (1031, 443),
             "offline_exf_seaice-code-63cdc0b-704fd6b": (777, 461),
             "offline_exf_seaice-code_ad-63cdc0b-704fd6b": (963, 447), "seaice_itd-code-63cdc0b-704fd6b": (768, 415)}
COUNTS = {**M2_COUNTS, **M3_COUNTS, **M4_COUNTS}
# Sites whose winner differs between builds (measured, job 27856188; M1-M3 had none). Both sit in the
# SEAICE_areaLossFormula=3 branch of SEAICE_GROWTH (pkg/seaice/seaice_growth.F:1840-1848, the ELSE of the IF at :1833):
#   :1847  tmpscal2 = MAX(-tmpscal0,tmpscal1)    :1848  tmpscal3 = MIN(ZERO,tmpscal2)
# The same statements compile to a different source-operand winner in different builds (the emitted maxsd/minsd keeps
# the other operand), so p is a property of the build, not of the statement. Only lab_sea/input_ad and
# input_ad.noseaicedyn set SEAICE_areaLossFormula=3 (their data.seaice; default 1, seaice_readparms.F:507), both on
# lab_sea-code_ad (MAX p="a", MIN p="b"); 1D_ocean_ice_column (formula 1) and global_ocean.cs32x15 /
# offline_exf_seaice.thermo (formula 2) never execute these lines. Plan decision 15 (lane M4ADLAB session 3): a port
# of such a statement takes the winner of the build that executes it; the call carries `# MINMAX-BUILD: <build>[,
# <build>...]` (oracle build names, each compiling the site with the call's literal p), and its module a static table
# `MINMAX_BUILD = {"<F file>:<line>": {build: p}}` equal to the marker (mitjax.ops.fortran_minmax.build_winner refuses
# every other build at run time). A marker at a site whose winner is the same in every build, or a KNOWN_DIFFER call
# without one, fails the audit.
KNOWN_DIFFER = {
    ("pkg/seaice/seaice_growth.F", 1847, "MAX"): {
        "a": {"global_ocean.cs32x15-code-63cdc0b-06b4417", "global_ocean.cs32x15-code_ad-63cdc0b-78ca390",
              "offline_exf_seaice-code-63cdc0b-704fd6b", "seaice_itd-code-63cdc0b-704fd6b",
              "lab_sea-code-63cdc0b-704fd6b", "lab_sea-code_ad-63cdc0b-704fd6b"},
        "b": {"1D_ocean_ice_column-code-63cdc0b-704fd6b", "1D_ocean_ice_column-code_ad-63cdc0b-704fd6b"}},
    ("pkg/seaice/seaice_growth.F", 1848, "MIN"): {
        "a": {"global_ocean.cs32x15-code-63cdc0b-06b4417", "global_ocean.cs32x15-code_ad-63cdc0b-78ca390"},
        "b": {"1D_ocean_ice_column-code-63cdc0b-704fd6b", "1D_ocean_ice_column-code_ad-63cdc0b-704fd6b",
              "offline_exf_seaice-code-63cdc0b-704fd6b", "seaice_itd-code-63cdc0b-704fd6b",
              "lab_sea-code-63cdc0b-704fd6b", "lab_sea-code_ad-63cdc0b-704fd6b"}}}
PHYSICS = ("mitjax/model", "mitjax/pkg")
HELPERS = {"MAX": "MAX", "MIN": "MIN", "max_chain": "MAX", "min_chain": "MIN", "MAX_CHAIN": "MAX"}
# statement-function ports whose callers pass the winners of their own expansion: name -> (module, ordered sites)
STATEMENT_FUNCTIONS = {"Limiter": ("mitjax/pkg/generic_advdiff/gad_flux_limiter_h.py", ("MIN", "MIN", "MAX", "MAX"))}
RAW = {("jnp", "maximum"), ("jnp", "minimum"), ("jnp", "fmax"), ("jnp", "fmin"), ("jnp", "clip"),
       ("np", "maximum"), ("np", "minimum"), ("np", "fmax"), ("np", "fmin"), ("np", "clip"),
       ("lax", "max"), ("lax", "min")}
_CITE = re.compile(r"(?<![\w.]):(\d+)(?:-(\d+))?")
_BUILD_MARK = re.compile(r"MINMAX-BUILD:\s*([\w.\-]+(?:\s*,\s*[\w.\-]+)*)")
_FORTRAN = re.compile(r"\b((?:model|pkg|eesupp)/[\w/]+\.F)\b")


# ---------------------------------------------------------------------------------------------------------------
# tables

@functools.lru_cache(maxsize=None)
def tables():
    root = paths.REFERENCE / "minmax_sites"
    out = {}
    for b in BUILDS:
        path = root / f"{b}.json"
        text = path.read_text()
        want = (root / f"{b}.json.sha256").read_text().split()[0]
        assert hashlib.sha256(text.encode()).hexdigest() == want, path
        out[b] = json.loads(text)
    return out


@functools.lru_cache(maxsize=None)
def site_index():
    """{F_file: [(first, last .F line, op, nargs, p, build)]} over the M1, M2 and M3 tables, in compilation order."""
    idx = {}
    for b, t in tables().items():
        for s in t["sites"]:
            idx.setdefault(s["F_file"], []).append((s["F_stmt"][0], s["F_stmt"][1], s["op"], s["nargs"], s["p"], b))
    return idx


def test_tables_complete_and_consistent():
    """The M1-M4 tables: objects byte-identical, every step resolved at the machine level, the same winner per site in
    every build that compiles it (tools/fortran_minmax_sites.compare), except exactly the two measured build-dependent
    sites of KNOWN_DIFFER (planted: any other difference, or one of these resolved, fails)."""
    for b, t in tables().items():
        m = t["meta"]
        if b in COUNTS:
            assert m["objects_differ"] == [] and m["objects_identical"] == m["n_files"] == COUNTS[b][0], b
            assert m["problems"] == [] and m["n_sites"] == COUNTS[b][1], b
        else:
            assert m["objects_differ"] == [] and m["objects_identical"] == m["n_files"] > 500, b
            assert m["problems"] == [] and m["n_sites"] > 300, b
        assert all(st["winner"] in ("acc", "new") for s in t["sites"] for st in s["steps"]), b
        assert m["compiler_version"] == "GNU Fortran (Spack GCC) 11.2.0" and "-O0" in m["flags"], b
    root = paths.REFERENCE / "minmax_sites"
    differ, n, _ = fms.compare([root / f"{b}.json" for b in BUILDS])
    got = {(k[0], k[1], k[4]): {p: set(bs) for p, bs in v.items()} for k, v in differ.items()}
    assert got == KNOWN_DIFFER and n > 400, got


# ---------------------------------------------------------------------------------------------------------------
# audit

def _comments(text):
    """{line: comment text} of a Python source."""
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = tok.string
    return out


def _name(func):
    if isinstance(func, ast.Name):
        return None, func.id
    if isinstance(func, ast.Attribute):
        v = func.value
        base = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
        return base, func.attr
    return None, None


def _fortran_file(stack, module_doc):
    for node in reversed(stack):
        doc = ast.get_docstring(node) or ""
        m = _FORTRAN.search(doc)
        if m:
            return m.group(1)
    m = _FORTRAN.search(module_doc or "")
    return m.group(1) if m else None


def _citations(stmt, comments):
    cites = []
    for ln in range(stmt.lineno, stmt.end_lineno + 1):
        for a, b in _CITE.findall(comments.get(ln, "")):
            cites.append((int(a), int(b or a)))
    return cites


def _literal(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def _ordered_winner(call, stmt, hits, op, nargs):
    """Several sites of one op and argument count in ONE Fortran statement with different winners (e.g.
    ggl90_calc.F:248-250, MIN(halfRS,hFacC(km1)) + MIN(halfRS,hFacC(k)): 'a' then 'b'): the winner of `call` is the
    table's winner at the call's place among the same-op, same-nargs helper calls of its Python statement, in source
    order (line, column), which must equal the table's compilation order. Every build that compiles the statement must
    list the same winner sequence, and the statement must hold as many such calls as the table has sites (a nested
    same-op call is refused: its source order is not its evaluation order). Returns the winner, or a string reason
    (prefixed '!') when the calls cannot be paired. `hits`: the sites in the index's (compilation) order."""
    stm = {(f, l) for f, l, *_ in hits}
    if len(stm) != 1:
        return "!several Fortran statements cited"
    by_build = {}
    for f, l, o, n, pp, bld in hits:
        by_build.setdefault(bld, []).append(pp)
    seqs = {tuple(v) for v in by_build.values()}
    if len(seqs) != 1:
        return f"!the builds disagree on the winner sequence {sorted(seqs)}"
    (seq,) = seqs
    name = _name(call.func)[1]
    calls = [c for c in ast.walk(stmt) if isinstance(c, ast.Call) and _name(c.func)[1] == name
             and _name(c.func)[0] in (None, "host", "fortran_minmax", "fortran_minmax_host") and len(c.args) == nargs]
    for c in calls:
        for d in ast.walk(c):
            if d is not c and isinstance(d, ast.Call) and _name(d.func)[1] == name:
                return "!nested same-op calls"
    calls.sort(key=lambda c: (c.lineno, c.col_offset))
    if len(calls) != len(seq):
        return f"!{len(calls)} {op} calls in the Python statement for {len(seq)} sites"
    return seq[[id(c) for c in calls].index(id(call))]


def audit_source(text, relpath, index):
    """Errors of one physics module (list of strings)."""
    tree = ast.parse(text)
    comments = _comments(text)
    module_doc = ast.get_docstring(tree)
    errors = []
    build_table = None                     # the module's MINMAX_BUILD (plan decision 15), a literal dict
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "MINMAX_BUILD"
                                                for x in node.targets):
            build_table = _literal(node.value)
    sf_module = {v[0]: k for k, v in STATEMENT_FUNCTIONS.items()}

    def where(node):
        return f"{relpath}:{node.lineno}"

    def visit(node, stack, stmt):
        if isinstance(node, ast.stmt):
            stmt = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack = stack + [node]
        if isinstance(node, ast.Call):
            check_call(node, stack, stmt)
        for child in ast.iter_child_nodes(node):
            visit(child, stack, stmt)

    def check_call(call, stack, stmt):
        base, name = _name(call.func)
        if base is None and name in ("max", "min"):
            text_c = " ".join(comments.get(ln, "") for ln in range(stmt.lineno, stmt.end_lineno + 1))
            if not re.search(r"MINMAX-(?:INT|RAW):\s*\S", text_c):
                errors.append(f"{where(call)}: builtin {name} without a `MINMAX-INT: <why>` (integer MAX/MIN: no tie "
                              f"or NaN case) or `MINMAX-RAW: <reason>` comment")
            return
        if (base, name) in RAW:
            text_c = " ".join(comments.get(ln, "") for ln in range(stmt.lineno, stmt.end_lineno + 1))
            m = re.search(r"MINMAX-RAW:\s*(\S.*)", text_c)
            if not m:
                errors.append(f"{where(call)}: raw {base}.{name} without a `MINMAX-RAW: <reason>` comment")
            return
        if name in STATEMENT_FUNCTIONS and relpath != STATEMENT_FUNCTIONS[name][0]:
            sites = STATEMENT_FUNCTIONS[name][1]
            check_site_call(call, stack, stmt, ops=sites, nargs=None, statement_function=name)
            return
        if name not in HELPERS or base not in (None, "host", "fortran_minmax", "fortran_minmax_host"):
            return
        pkw = [k for k in call.keywords if k.arg == "p"]
        if not pkw:
            errors.append(f"{where(call)}: {name} without p=")
            return
        if _literal(pkw[0].value) is None:
            fn = stack[-1].name if stack else None
            if relpath in sf_module and fn == sf_module[relpath]:
                return              # the body of a statement-function port: its callers are checked
            errors.append(f"{where(call)}: {name} with a non-literal p")
            return
        nargs = 2 if name in ("max_chain", "min_chain", "MAX_CHAIN") else len(call.args)
        check_site_call(call, stack, stmt, ops=(HELPERS[name],), nargs=nargs, statement_function=None)

    def check_site_call(call, stack, stmt, ops, nargs, statement_function):
        label = statement_function or _name(call.func)[1]
        pkw = [k for k in call.keywords if k.arg == "p"]
        p = _literal(pkw[0].value) if pkw else None
        if p is None:
            errors.append(f"{where(call)}: {label} without a literal p=")
            return
        cites = _citations(stmt, comments)
        if not cites:
            errors.append(f"{where(call)}: {label} without a `# :NN` citation of the Fortran line")
            return
        ffile = _fortran_file(stack, module_doc)
        if ffile is None:
            errors.append(f"{where(call)}: no Fortran file named in the docstrings")
            return
        sites = index.get(ffile, [])
        if statement_function:
            # the expansion's sites at one cited statement, in compilation order
            for a, b in cites:
                stm = sorted({(f, l) for f, l, *_ in sites if f <= b and l >= a})
                for f, l in stm:
                    got = {}
                    for f2, l2, op, n, pp, bld in sites:
                        if (f2, l2) == (f, l):
                            got.setdefault(bld, []).append((op, pp))
                    seqs = {tuple(v) for v in got.values()}
                    want = [tuple(zip(ops, p))]
                    if len(seqs) != 1 or list(seqs) != want:
                        errors.append(f"{where(call)}: {label} p={p!r} vs the table at {ffile}:{f}-{l}: {seqs}")
                    return
            errors.append(f"{where(call)}: {label}: no MAX/MIN site at the cited {ffile} lines {cites}")
            return
        hits = {s for s in sites for a, b in cites if s[0] <= b and s[1] >= a and s[2] == ops[0] and s[3] == nargs}
        if not hits:
            errors.append(f"{where(call)}: {label}: no {ops[0]} site of {nargs} arguments at the cited {ffile} "
                          f"lines {cites}")
            return
        # plan decision 15: build-dependent sites (KNOWN_DIFFER) only with a MINMAX-BUILD marker and the table
        text_c = " ".join(comments.get(ln, "") for ln in range(stmt.lineno, stmt.end_lineno + 1))
        mark = _BUILD_MARK.search(text_c)
        kd = sorted(k for k in KNOWN_DIFFER if k[0] == ffile and k[2] == ops[0]
                    and any(s[0] <= k[1] <= s[1] for s in hits))
        if mark or kd:
            if not kd:
                errors.append(f"{where(call)}: {label}: MINMAX-BUILD marker at {ffile} lines {cites}, whose winner "
                              "is the same in every build")
                return
            if not mark:
                errors.append(f"{where(call)}: {label}: {ffile}:{kd[0][1]} has build-dependent winners "
                              f"{KNOWN_DIFFER[kd[0]]}: the call needs a `# MINMAX-BUILD: <build>` marker (plan "
                              "decision 15)")
                return
            named = [b.strip() for b in mark.group(1).split(",")]
            bad = [b for b in named if b not in BUILDS]
            if bad:
                errors.append(f"{where(call)}: {label}: MINMAX-BUILD names unknown builds {bad}")
                return
            hits = {s for s in hits if s[5] in named}
            if {s[5] for s in hits} != set(named):
                errors.append(f"{where(call)}: {label}: MINMAX-BUILD builds {named} do not all compile "
                              f"{ffile} lines {cites}")
                return
            for k in kd:
                want_t = {b: p for b in named}
                got_t = (build_table or {}).get(f"{ffile}:{k[1]}")
                if got_t != want_t:
                    errors.append(f"{where(call)}: {label}: the module's MINMAX_BUILD[{ffile}:{k[1]!r}] = {got_t} "
                                  f"is not the marker's {want_t}")
                    return
        winners = {s[4] for s in hits}
        if len(winners) != 1:
            want = _ordered_winner(call, stmt, [x for x in sites if x in hits], ops[0], nargs)
            if not want.startswith("!"):
                if p != want:
                    errors.append(f"{where(call)}: {label}(..., p={p!r}) but the oracle's {ffile} lines {cites} "
                                  f"keep p={want!r} for this call (its place among the statement's {ops[0]} calls "
                                  "in source order = the table's compilation order)")
            else:
                errors.append(f"{where(call)}: {label}: the cited {ffile} lines {cites} hold {ops[0]} sites with "
                              f"different winners {sorted(winners)}: cite the statement alone ({want})")
        elif p not in winners:
            errors.append(f"{where(call)}: {label}(..., p={p!r}) but the oracle's {ffile} lines {cites} keep "
                          f"p={winners.pop()!r}")

    visit(tree, [], None)
    return errors


def physics_files():
    return sorted(f for d in PHYSICS for f in (REPO / d).rglob("*.py"))


def test_physics_calls_cite_and_match_the_table():
    index = site_index()
    errors, n_calls = [], 0
    for f in physics_files():
        text = f.read_text()
        rel = str(f.relative_to(REPO))
        errors += audit_source(text, rel, index)
        n_calls += len(re.findall(r"\b(?:MAX|MIN|max_chain|min_chain|Limiter)\(", text))
    assert not errors, "\n".join(errors)
    assert n_calls > 80


def _mutate(rel, old, new):
    text = (REPO / rel).read_text()
    assert text.count(old) == 1, (rel, old)
    return audit_source(text.replace(old, new), rel, site_index())


def test_audit_negative_controls():
    """A flipped p, a missing citation, a raw jnp.maximum, a non-literal p, a wrong Limiter expansion and swapped
    winners of two same-op sites in one statement are caught."""
    rel = "mitjax/pkg/mom_common/mom_u_sidedrag.py"
    assert audit_source((REPO / rel).read_text(), rel, site_index()) == []
    line = 'A4tmp = MAX(A4tmp, viscA4GridMin*(rAw[i, j]**2)/deltaTMom, p="a")   # :76'
    errs = _mutate(rel, line, line.replace('p="a"', 'p="b"'))
    assert len(errs) == 1 and "keep p='a'" in errs[0], errs
    errs = _mutate(rel, line, line.replace("   # :76", ""))
    assert len(errs) == 1 and "citation" in errs[0], errs
    errs = _mutate(rel, line, line.replace('MAX(', 'jnp.maximum(').replace(', p="a")', ')'))
    assert len(errs) == 1 and "raw jnp.maximum" in errs[0], errs
    errs = _mutate(rel, line, line.replace('p="a"', 'p=P'))
    assert len(errs) == 1 and "non-literal" in errs[0], errs
    errs = _mutate(rel, line, line.replace(":76", ":40"))
    assert len(errs) == 1 and "no MAX site" in errs[0], errs
    # several same-op sites in one statement, paired by order (ggl90_calc.F:248-250: 'a' then 'b'): swapped fails
    rel = "mitjax/pkg/ggl90/ggl90_calc.py"
    assert audit_source((REPO / rel).read_text(), rel, site_index()) == []
    pair = ('hFac = (MIN(halfRS, hFacC_km1, p="a")', '+ MIN(halfRS, hFacC[i, j, k], p="b"))')
    text = (REPO / rel).read_text()
    assert all(text.count(x) == 1 for x in pair), pair
    swapped = text.replace(pair[0], pair[0].replace('p="a"', 'p="b"')).replace(pair[1], pair[1].replace('p="b"',
                                                                                                          'p="a"'))
    errs = audit_source(swapped, rel, site_index())
    assert len(errs) == 2 and all("for this call" in e for e in errs), errs
    errs = _mutate(rel, pair[1], pair[1].replace('p="b"', 'p="a"'))
    assert len(errs) == 1 and "keep p='b'" in errs[0], errs
    rel = "mitjax/pkg/generic_advdiff/gad_fluxlimit_impl_r.py"
    errs = _mutate(rel, 'Limiter(Cr, p=("b", "a", "b", "a"))', 'Limiter(Cr, p=("b", "a", "b", "b"))')
    assert len(errs) == 1 and "Limiter" in errs[0], errs
    errs = _mutate(rel, "km2 = max(1, k-2)                                               # :76-78; MINMAX-INT: integer (no tie or NaN case)",
                   "km2 = max(1, k-2)                                               # :76-78")
    assert len(errs) == 1 and "builtin max" in errs[0], errs


def test_audit_build_dependent_sites():
    """Plan decision 15 (lane M4ADLAB session 3): the KNOWN_DIFFER calls of seaice_growth.F:1847-1848 pass with their
    MINMAX-BUILD marker and the module's MINMAX_BUILD table; a planted wrong winner (in the call, or in the call and
    the table), a missing marker, a marker naming a build of the other winner, a table that disagrees with the marker,
    and a marker at a site with one winner in every build are caught."""
    rel = "mitjax/pkg/seaice/seaice_growth.py"
    assert audit_source((REPO / rel).read_text(), rel, site_index()) == []
    mx = 'tmpscal2 = MAX(-tmpscal0, tmpscal1, p="a")       # :1847; MINMAX-BUILD: lab_sea-code_ad-63cdc0b-704fd6b'
    mn = 'tmpscal3 = MIN(ZERO, tmpscal2, p="b")            # :1848; MINMAX-BUILD: lab_sea-code_ad-63cdc0b-704fd6b'
    tab = '"pkg/seaice/seaice_growth.F:1847": {"lab_sea-code_ad-63cdc0b-704fd6b": "a"}'
    errs = _mutate(rel, mx, mx.replace('p="a"', 'p="b"'))                      # the wrong winner in the call
    assert len(errs) == 1 and "MINMAX_BUILD" in errs[0], errs
    text = (REPO / rel).read_text()
    both = text.replace(mx, mx.replace('p="a"', 'p="b"')).replace(tab, tab.replace('"a"}', '"b"}'))
    errs = audit_source(both, rel, site_index())                               # call and table planted alike
    assert len(errs) == 1 and "keep p='a'" in errs[0], errs
    errs = _mutate(rel, mn, mn.replace('p="b"', 'p="a"'))
    assert len(errs) == 1 and "MINMAX_BUILD" in errs[0], errs
    errs = _mutate(rel, mx, mx.split("; MINMAX-BUILD")[0])                     # no marker
    assert len(errs) == 1 and "needs a `# MINMAX-BUILD" in errs[0], errs
    other = "1D_ocean_ice_column-code_ad-63cdc0b-704fd6b"                      # MAX "b" there
    errs = _mutate(rel, mx, mx.replace("lab_sea-code_ad-63cdc0b-704fd6b", other))
    assert len(errs) == 1 and "MINMAX_BUILD" in errs[0], errs
    errs = _mutate(rel, tab, tab.replace('"a"}', '"b"}'))                      # the table alone planted
    assert len(errs) == 1 and "MINMAX_BUILD" in errs[0], errs
    one = 'tmpscal4 = MAX(ZERO, a_QbyATM_open, p="b")                             # :1825'
    errs = _mutate(rel, one, one + "; MINMAX-BUILD: lab_sea-code_ad-63cdc0b-704fd6b")
    assert len(errs) == 1 and "same in every build" in errs[0], errs


def default_errors(defaults, table):
    """Plan decision 7 (docs plan 20261006 Task 4): a module's MINMAX_DEFAULT names exactly the sites of its
    MINMAX_BUILD, each a KNOWN_DIFFER site, with the winner of most of the measured builds that compile the site (a
    strict majority; a tie has no documented default and is an error)."""
    errs = []
    if set(defaults) != set(table):
        errs.append(f"MINMAX_DEFAULT sites {sorted(defaults)} != MINMAX_BUILD sites {sorted(table)}")
    for site, p in sorted(defaults.items()):
        f, line = site.rsplit(":", 1)
        kd = [v for k, v in KNOWN_DIFFER.items() if k[0] == f and k[1] == int(line)]
        if len(kd) != 1:
            errs.append(f"{site}: not one build-dependent site of KNOWN_DIFFER")
            continue
        counts = {w: len(bs) for w, bs in kd[0].items()}
        top = [w for w, c in counts.items() if c == max(counts.values())]
        if len(top) != 1 or top[0] != p:
            errs.append(f"{site}: MINMAX_DEFAULT {p!r}, the measured builds' majority is {top} ({counts})")
    return errs


def _modules_with_build_tables():
    """{module path: module} of every physics module that defines MINMAX_BUILD."""
    import importlib
    out = {}
    for d in PHYSICS:
        for f in sorted((REPO / d).rglob("*.py")):
            if re.search(r"^MINMAX_BUILD\s*=", f.read_text(), re.M):
                rel = f.relative_to(REPO).as_posix()
                out[rel] = importlib.import_module(rel[:-3].replace("/", "."))
    return out


def test_audit_default_winners():
    """Plan decision 7: every module with a MINMAX_BUILD table has its MINMAX_DEFAULT, the measured majority winner
    per site (seaice_growth.F:1847 MAX "a" in 6 of 8 builds, :1848 MIN "b" in 6 of 8); planted: a flipped default, a
    missing site, a default at a site with one winner everywhere are caught."""
    mods = _modules_with_build_tables()
    assert list(mods) == ["mitjax/pkg/seaice/seaice_growth.py"], list(mods)
    for rel, m in mods.items():
        assert default_errors(getattr(m, "MINMAX_DEFAULT", {}), m.MINMAX_BUILD) == [], rel
    G = mods["mitjax/pkg/seaice/seaice_growth.py"]
    flipped = {**G.MINMAX_DEFAULT, "pkg/seaice/seaice_growth.F:1847": "b"}
    errs = default_errors(flipped, G.MINMAX_BUILD)
    assert len(errs) == 1 and "majority is ['a']" in errs[0], errs
    errs = default_errors({"pkg/seaice/seaice_growth.F:1847": "a"}, G.MINMAX_BUILD)
    assert len(errs) == 1 and "!= MINMAX_BUILD sites" in errs[0], errs
    errs = default_errors({**G.MINMAX_DEFAULT, "pkg/seaice/seaice_growth.F:1825": "b"},
                          {**G.MINMAX_BUILD, "pkg/seaice/seaice_growth.F:1825": {}})
    assert len(errs) == 1 and "not one build-dependent site" in errs[0], errs


# ---------------------------------------------------------------------------------------------------------------
# values

def _f(h):
    return struct.unpack(">d", bytes.fromhex(h))[0]


def _bits(x):
    return np.asarray(x, np.float64).reshape(()).view(np.int64)


def _same(got, want):
    return int(_bits(got)) == int(_bits(np.float64(want)))


def _probe_sites(probe):
    d = json.loads((HERE / "probe_sites.json").read_text())
    out = {}
    for s in d["probes"][probe]:
        out.setdefault(s["f_line"], []).append(s)
    return out


# mmsites.F: site number -> (.F line, the Fortran arguments as a function of (a(1), b(1)); c(1) = -1)
_C = -1.0
MMSITES = {1: (15, lambda a, b: [a, b]), 2: (16, lambda a, b: [a, b]), 3: (19, lambda a, b: [a, b]),
           4: (22, lambda a, b: [a, b]), 5: (27, lambda a, b: [a, b]), 6: (33, lambda a, b: [a, b]),
           7: (37, lambda a, b: [a, b]), 8: (39, lambda a, b: [a, b]), 9: (40, lambda a, b: [b, a]),
           10: (42, lambda a, b: [a, b]), 11: (44, lambda a, b: [a, b, _C]), 12: (46, lambda a, b: [_C, a, b]),
           13: (48, lambda a, b: [a, 0.0]), 14: (49, lambda a, b: [0.0, a]), 15: (50, lambda a, b: [a, 0.0]),
           16: (51, lambda a, b: [0.0, a]), 17: (54, lambda a, b: [a, b]), 18: (55, lambda a, b: [b, a]),
           19: (59, lambda a, b: [b, a]), 20: (63, lambda a, b: [a, b]), 21: (65, lambda a, b: [a, b]),
           23: (69, lambda a, b: [abs(a), b])}


def test_helper_reproduces_gfortran_probe_sites():
    """Every probe site (+0/-0 ties and NaN in both positions), with p from the tool's extraction of the probe's own
    compilation: the helper equals gfortran bit for bit, eagerly and under jit; both winners occur; flipping p fails."""
    sites = _probe_sites("mmsites.F")
    rows = [ln.split() for ln in (HERE / "mmsites_gfortran11.out").read_text().splitlines()]
    seen, flipped_bites = set(), 0
    for r in rows:
        k = int(r[0][1:])
        if k == 22:
            continue                    # nested MAX(MAX(a,c),b): covered by the two-step sites below
        line, args = MMSITES[k]
        a, b, want = _f(r[1]), _f(r[2]), _f(r[3])
        (s,) = sites[line]
        op = MAX if s["op"] == "MAX" else MIN
        xs = [jnp.float64(x) for x in args(a, b)]
        got = op(*xs, p=s["p"])
        assert _same(got, want), (k, s, a, b, got, want)
        assert _same(jax.jit(functools.partial(op, p=s["p"]))(*xs), want), k
        flip = "".join("a" if c == "b" else "b" for c in s["p"])
        flipped_bites += not _same(op(*xs, p=flip), want)
        seen.add(s["p"])
    assert {"a", "b", "bb"} <= seen and flipped_bites > 20


def test_statement_function_probe():
    """sfprobe.F: MIN(2.D0,Cr) keeps 2 for a NaN Cr inside the statement function (p="a", the RTL source operand,
    although the GIMPLE order says otherwise) and NaN when written inline (p="b"); the helper reproduces both."""
    sites = _probe_sites("sfprobe.F")
    assert [s["p"] for s in sites[11]] == ["b", "a", "b", "b"] and [s["gimple_p"] for s in sites[11]] == ["b"] * 4
    assert [s["p"] for s in sites[15]] == ["b"] * 4
    rows = [ln.split() for ln in (HERE / "sfprobe_gfortran11.out").read_text().splitlines()]

    def limiter(cr, p):
        return MAX(0., MAX(MIN(1., 2.*cr, p=p[0]), MIN(2., cr, p=p[1]), p=p[2]), p=p[3])

    for r in rows:
        k, x, want = int(r[0][1:]), _f(r[1]), _f(r[2])
        cr = jnp.float64(x)
        got = {1: lambda: limiter(cr, [s["p"] for s in sites[11]]),
               2: lambda: limiter(cr, [s["p"] for s in sites[15]]),
               3: lambda: MIN(2., cr, p=sites[18][0]["p"]),
               4: lambda: MIN(cr, 2., p=sites[19][0]["p"])}[k]()
        assert _same(got, want), (k, x, got, want)
    assert rows[0][0] == "R1" and np.isnan(_f(rows[0][1])) and _f(rows[0][2]) == 2.0       # statement function
    assert rows[1][0] == "R2" and np.isnan(_f(rows[1][2]))                                   # inline


def test_insitu_oracle_object():
    """The oracle's own mon_advcflw2.o (insitu_mon_advcflw2.F): max(K term, K-1 term) keeps the K term on -0/+0 and
    NaN (the table's p="a"; the GIMPLE order would keep the K-1 term), and the chain keeps the new value."""
    rows = [ln.split() for ln in (HERE / "insitu_mon_advcflw2.out").read_text().splitlines()]
    assert [r[1] for r in rows] == ["8000000000000000", "FFF8000000000000"]
    idx = site_index()["pkg/monitor/mon_advcflw2.F"]
    assert {s[4] for s in idx if s[0] == 42} == {"a"} and {s[4] for s in idx if s[0] == 45} == {"b"}
    nan = _f("FFF8000000000000")        # 0/0 on x86-64 (the driver's z/z)
    for kterm, km1, want in ((-0.0, 0.0, rows[0][1]), (nan, 1.0, rows[1][1])):
        tmp = 1.0*1.0*MAX(jnp.float64(kterm), jnp.float64(km1), p="a")
        got = host.max_chain(0.0, np.array([0.0, float(tmp)]), p="b")
        assert _same(got, _f(want)), (kterm, km1, got)
        assert not _same(host.max_chain(0.0, np.array([0.0, float(MAX(kterm, km1, p="b"))]), p="b"), _f(want))


def test_derivative_is_jax_own_for_both_winners():
    a = jnp.asarray([1.0, 0.0, -2.0, 0.0])
    b = jnp.asarray([0.5, 0.0, 3.0, -0.0])
    for op, ref in ((MAX, jnp.maximum), (MIN, jnp.minimum)):
        for p in ("a", "b"):
            t = jax.jvp(lambda x, y: op(x, y, p=p), (a, b), (jnp.ones(4), 10*jnp.ones(4)))[1]
            r = jax.jvp(ref, (a, b), (jnp.ones(4), 10*jnp.ones(4)))[1]
            assert np.array_equal(np.asarray(t), np.asarray(r))


def test_bad_p_rejected():
    for bad in (None, "c", "ab", ("a",)):
        with pytest.raises((ValueError, TypeError)):
            MAX(1., 2., p=bad)
    with pytest.raises(ValueError):
        host.MAX(1., 2., p="c")


# ---------------------------------------------------------------------------------------------------------------
# running MAX chains (cg2d.F:111, ini_cg2d.F:110-111): MAX_CHAIN against chainprobe.F

def _chain_cases():
    rows = [ln.split() for ln in (HERE / "chainprobe_gfortran11.out").read_text().splitlines()]
    ins = {int(r[0][2:]): np.array([_f(h) for h in r[1:]]) for r in rows if r[0].startswith("IN")}
    outs = {int(r[0][3:]): [_f(h) for h in r[1:]] for r in rows if r[0].startswith("OUT")}
    # b(3,2,2,2) in Fortran order -> [tile = (bj-1)*2 + bi-1, j, i]
    tiles = {c: v.reshape(2, 2, 2, 3).reshape(4, 2, 3) for c, v in ins.items()}
    return tiles, outs


class _BlockEx:
    """ex.all_tiles of a TileBlocks layout (eesupp/global_sum.all_tiles), for MAX_CHAIN under shard_map."""
    def __init__(self, blocks, axis):
        self.blocks, self.axis = blocks, axis

    def all_tiles(self, per_tile):
        from mitjax.eesupp import global_sum as gs
        return gs.all_tiles(per_tile, self.blocks, self.axis)


def _sharded_chain(fn, b, P):
    from jax.sharding import PartitionSpec
    from mitjax.eesupp.shard import tile_mesh
    from mitjax.eesupp.sharded_exchange import AXIS, TileBlocks
    blocks = TileBlocks(b.shape[0], P)
    ex = _BlockEx(blocks, AXIS)
    f = jax.jit(jax.shard_map(lambda x: fn(x, ex), mesh=tile_mesh(P), in_specs=PartitionSpec(AXIS),
                              out_specs=PartitionSpec(), check_vma=False))
    return f(blocks.pad(np.asarray(b)))


def test_max_chain_matches_gfortran_chain_with_planted_nans():
    """chainprobe.F (4 tiles, NaN at the first/middle/last point of a tile, at the last point of the last tile, two
    NaNs, all NaN, -0 values, a NaN after the maximum): MAX_CHAIN equals the gfortran chain bit for bit for the cg2d
    form (A: the new value wins NaN), the dummy-argument form (B: the running value wins) and ini_cg2d's two statements
    per point (C), at P=1 and sharded at P=2, 3 (padding) and 4; the probe's statements have the winners of the oracle's
    cg2d.F:111 and ini_cg2d.F:110-111. Negative control: the former order-free maximum differs on the NaN cases."""
    sites = _probe_sites("chainprobe.F")
    pA, pB, pC = sites[26][0]["p"], sites[39][0]["p"], [s["p"] for s in sites[59] + sites[60]]
    idx = site_index()
    assert {s[4] for s in idx["model/src/cg2d.F"] if s[0] == 111} == {pA} == {"a"}
    assert {s[4] for s in idx["model/src/ini_cg2d.F"] if s[0] in (110, 111)} == set(pC) == {"a"}
    assert pB == "b"
    tiles, outs = _chain_cases()
    forms = {0: lambda b, ex: MAX_CHAIN(0.0, jnp.abs(b), p=pA, acc="b", ex=ex),
             1: lambda b, ex: MAX_CHAIN(0.0, jnp.abs(b), p=pB, acc="b", ex=ex),
             2: lambda b, ex: MAX_CHAIN(0.0, jnp.stack([jnp.abs(b), jnp.abs(0.5*b)], axis=-1), p="a", acc="b", ex=ex)}
    order_free_bites = 0
    for c, b in tiles.items():
        for k, fn in forms.items():
            want = outs[c][k]
            assert _same(fn(jnp.asarray(b), None), want), (c, k, fn(jnp.asarray(b), None), want)
            for P in (2, 3, 4):
                assert _same(_sharded_chain(fn, b, P), want), (c, k, P)
        order_free_bites += not _same(jnp.maximum(jnp.max(jnp.abs(jnp.asarray(b))), 0.0), outs[c][0])
    assert order_free_bites == 5                                     # cases 1, 2, 4, 5, 8: a NaN not at the end


def test_max_chain_derivative_is_jax_own_without_nan():
    """Without NaN the chain's derivative is that of the former order-free maximum (the selected element)."""
    b = jnp.asarray(_chain_cases()[0][0])
    g = jax.grad(lambda x: MAX_CHAIN(0.0, jnp.abs(x), p="a", acc="b"))(b)
    r = jax.grad(lambda x: jnp.maximum(jnp.max(jnp.abs(x)), 0.0))(b)
    assert np.array_equal(np.asarray(g), np.asarray(r)) and np.count_nonzero(np.asarray(g)) == 1
