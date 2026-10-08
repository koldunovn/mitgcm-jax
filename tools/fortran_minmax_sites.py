#!/usr/bin/env python3
"""Which argument gfortran's MAX/MIN returns on a tie (+0/-0) or a NaN, at every REAL MAX/MIN site of an oracle build
(lane MINMAX, 2026-10-01).

    fortran_minmax_sites.py run BUILD_DIR... [--out-root DIR] [--scratch-root DIR] [-j N]
    fortran_minmax_sites.py compare JSON...           (per routine: are the site tables identical across builds?)

BUILD_DIR is a genmake2 build of the oracle ($MJX_REFERENCE/build/<name>, $MJX_REFERENCE/replay*/<name>): `bld/` with
the Makefile, the preprocessed `*.f` and the objects `*.o`, and `provenance.txt` naming the compiler.

The rule (measured, docs/KERNEL_GUIDE.md §3; probe mitjax/tests/fortran_minmax/mmsites.F):
1. gfortran 11.2.0 (trans-intrinsic.c, gfc_conv_intrinsic_minmax) lowers MAX(a1,a2,...,an) to the chain
   `M = a1; M = MAX_EXPR <a2, M>; ...; M = MAX_EXPR <an, M>` (the NEW argument is the first operand); an argument that
   is not a variable is first evaluated into a temporary.
2. GCC's canonical order of commutative operands (fold-const.c tree_swap_operands_p, applied by fold when the tree is
   built and by fold_stmt when the gimplifier emits the statement) swaps the operands when the new argument is a
   constant (literal or PARAMETER) or an SSA temporary, i.e. a variable that is not a register: a local whose address
   is taken (passed to a subroutine or to I/O), a COMMON-block or module variable. Then the running value M comes first.
   A dummy-argument or array element, or an expression, is a temporary declared by the front end and stays first.
3. At -O0 on x86-64 the GIMPLE statement `M = MAX_EXPR <p, q>` becomes one `maxsd`/`minsd` whose SOURCE operand is p
   (RTL `(smax:DF op1 op2)` after register allocation, op2 = p); maxsd/minsd return the source operand when the
   operands are equal (+0 vs -0) or either is NaN (Intel SDM). So p, the first GIMPLE operand, wins ties and NaN:
   MAX_EXPR <p, q> = (q > p) ? q : p,  MIN_EXPR <p, q> = (q < p) ? q : p.

How: every `bld/*.f` is compiled again with the build's own compiler, FFLAGS and FOPTIM (bld/Makefile, `.f.o` rule),
from a symbolic link of the same name in a NEW scratch directory, adding `-fdump-tree-optimized-lineno` and
`-fdump-rtl-final` (both written to pipes, nothing large is stored); the object must equal the build's `bld/*.o` byte
for byte (dumps do not change code). For each REAL MAX/MIN chain (`M.<k>` temporaries of a floating type) the tool
records, per step, whether the winner is the running value (`acc`) or the new argument (`new`), from the GIMPLE operand
order (rule 3), and checks it against the machine-level fact: the source operand of the post-register-allocation
smax/smin insn (named by its `[orig: ...]`/memory attributes), resolved to a GIMPLE operand. A site where the two
disagree or the RTL operand cannot be resolved is reported (`rtl_check`). The `.f` line of each site is mapped to
`file.F:line` with the build's own preprocessor run without -P in the build directory (tools/cpp_live.compiled_line_map,
which checks the regenerated `.f` equals the build's byte for byte).

Output: <out-root>/<build>.json (+ .sha256): per site the routine, `.F` file (upstream-relative when the build links
the upstream snapshot) and lines of the statement, MAX/MIN, the number of arguments, the per-step winners and the
helper's `p` (two arguments: "a" = first Fortran argument wins ties/NaN, "b" = second; n arguments: one letter per
step of the left fold MAX(MAX(a1,a2),a3)..., "a" = the running value, "b" = the new argument). Default out-root
$MJX_REFERENCE/minmax_sites; scratch under $MJX_RUNS/minmax_sites/<build>-<stamp>-<pid> (left in place: symlinks and
objects only). An existing table is never overwritten: identical content is accepted, different content is an error.
"""

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mitjax import paths  # noqa: E402
from mitjax.config import cpp_options  # noqa: E402
from tools import cpp_live  # noqa: E402

FLOAT_TYPES = ("real(kind=8)", "real(kind=4)", "real(kind=16)", "real(kind=10)")
_FUNC = re.compile(r"^;; Function (\S+) \(")
_DECL = re.compile(r"^\s+(real\(kind=\d+\)|integer\(kind=\d+\)|logical\(kind=\d+\)|character\(kind=\d+\)\S*)\s+"
                   r"(M\.\d+);$")
_MINMAX = re.compile(r"^\s+\[([^:\]]+):(\d+):(\d+)\] (M\.\d+)_(\d+) = (MAX|MIN)_EXPR <([^,>]+), ([^>]+)>;$")
_COPY = re.compile(r"^\s+(?:\[[^\]]*\] )?([\w.]+) = ([\w.()]+);$")
_RTL_OP = re.compile(r"\((smax|smin):(DF|SF|XF|TF) ")
_RTL_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')


# ---------------------------------------------------------------------------------------------------------------
# build record

def makefile_var(bld, name):
    return cpp_options._makefile_var(Path(bld) / "Makefile", name) or ""


def compiler(build_dir):
    """(argv prefix, version line): the absolute compiler of provenance.txt (`gfortran <path>: <version>`), checked
    against the Makefile's FC."""
    fc = makefile_var(Path(build_dir) / "bld", "FC").split()
    prov = (Path(build_dir) / "provenance.txt").read_text()
    m = re.search(r"^gfortran\s+(\S+?):\s*(.*)$", prov, re.M)
    if not m or Path(m.group(1)).name != fc[0]:
        raise ValueError(f"{build_dir}/provenance.txt: no absolute path for FC {fc!r}")
    return [m.group(1)] + fc[1:], m.group(2).strip()


def toolchain(build_dir):
    """The build's preprocessor (cpp_options.toolchain_from_record, reading bld/Makefile and provenance.txt)."""
    d, bld = Path(build_dir), Path(build_dir) / "bld"
    cppcmd = makefile_var(bld, "CPPCMD")
    m = re.match(r"cat \$< \|\s*(.*?) \$\(DEFINES\) \$\(INCLUDES\)", cppcmd)
    if not m:
        raise ValueError(f"{bld}/Makefile: cannot read the preprocessor from CPPCMD = {cppcmd!r}")
    argv = m.group(1).split()
    pm = re.search(r"^cpp\s+(\S+?):", (d / "provenance.txt").read_text(), re.M)
    if pm and Path(pm.group(1)).name == argv[0]:
        argv[0] = pm.group(1)
    else:   # replay harnesses record only the compiler: its own cpp (same installation, same module)
        sibling = Path(compiler(build_dir)[0][0]).parent / argv[0]
        if not sibling.is_file():
            raise ValueError(f"{d}/provenance.txt: no absolute path for {argv[0]!r} and no {sibling}")
        argv[0] = str(sibling)
    return cpp_options.Toolchain(tuple(argv), tuple(makefile_var(bld, "DEFINES").split()),
                                 tuple(makefile_var(bld, "INCLUDES").split()), str(d))


# ---------------------------------------------------------------------------------------------------------------
# compile one file with the dumps on pipes

def compile_one(fc, flags, src, wdir):
    """Compile wdir/<name>.f (a link to src) as the Makefile does; returns (tree dump, rtl dump, object bytes, stderr)."""
    name = src.name
    link = wdir / name
    if not link.is_symlink():
        link.symlink_to(src)
    obj = wdir / (src.stem + ".o")
    r_fd, w_fd = os.pipe()
    chunks = []

    def reader():
        with os.fdopen(r_fd, "rb") as fh:
            chunks.append(fh.read())

    th = threading.Thread(target=reader)
    th.start()
    try:
        argv = list(fc) + list(flags) + ["-c", name, "-o", obj.name, "-fdump-tree-optimized-lineno=stdout",
                                          f"-fdump-rtl-final=/dev/fd/{w_fd}"]
        p = subprocess.Popen(argv, cwd=wdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(w_fd,))
    finally:
        os.close(w_fd)
    out, err = p.communicate()
    th.join()
    if p.returncode != 0:
        raise RuntimeError(f"{name}: compiler failed ({p.returncode}): {err.decode(errors='replace')[-2000:]}")
    return out.decode("latin-1"), chunks[0].decode("latin-1"), obj.read_bytes(), err.decode("latin-1")


# ---------------------------------------------------------------------------------------------------------------
# parse the dumps

def _sections(text, kind):
    """{function: [lines]} of a dump (sections start with `;; Function name (`)."""
    out, cur = {}, None
    for ln in text.split("\n"):
        m = _FUNC.match(ln)
        if m:
            cur = m.group(1)
            if cur in out:
                raise ValueError(f"{kind} dump: function {cur} twice")
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(ln)
    return out


def _base(name):
    """SSA name -> its variable (`psip_237` -> `psip`, `M.0_19` -> `M.0`, `_21` -> None, `y2.27_3` -> `y2.27`)."""
    m = re.match(r"^(.*?)_(\d+)(?:\(D\))?$", name)
    if not m:
        return name
    return m.group(1) or None


def gimple_sites(lines):
    """[(stmt dict)] of the floating MAX/MIN chain steps of one function's optimized dump, plus the copy map."""
    types, steps, parent = {}, [], {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for ln in lines:
        m = _DECL.match(ln)
        if m:
            types[m.group(2)] = m.group(1)
            continue
        m = _MINMAX.match(ln)
        if m:
            f, line, col, mvar, ver, op, o0, o1 = m.groups()
            steps.append({"f_file": f, "f_line": int(line), "col": int(col), "mvar": mvar, "lhs": f"{mvar}_{ver}",
                          "op": op, "ops": [o0.strip(), o1.strip()], "gimple": ln.strip()})
            continue
        m = _COPY.match(ln)
        if m:
            a, b = m.group(1), m.group(2)
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    out = []
    for s in steps:
        t = types.get(s["mvar"])
        if t is None:
            raise ValueError(f"no declaration of {s['mvar']} ({s['gimple']})")
        if t not in FLOAT_TYPES:
            continue
        s["type"] = t
        acc = [i for i, o in enumerate(s["ops"]) if _base(o) == s["mvar"]]
        if len(acc) != 1:
            raise ValueError(f"cannot tell the running value of {s['gimple']}")
        s["acc_index"] = acc[0]
        s["gimple_winner"] = "acc" if acc[0] == 0 else "new"      # the first GIMPLE operand
        out.append(s)
    return out, find


def _balanced(text, i):
    """End index (exclusive) of the parenthesised RTX starting at text[i] == '('."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced RTL")


def _rtl_name(rtx):
    """The source-level name of an RTL operand: a register's `[orig:N name ]`, a memory reference's expression
    (`[1 psip+0 S8 A64]` -> `psip`), else None (an unnamed pseudo or a constant-pool load)."""
    m = re.search(r"\[orig:\d+ ([^\]]+?) \]", rtx)
    if m:
        return m.group(1)
    if rtx.startswith("(mem"):
        m = re.search(r"\[\d+ (.+?)\+\d+ S\d+ A\d+\]\)?$", rtx.strip())
        if m:
            return m.group(1)
        return None
    m = re.search(r"\(reg:\w+ \d+ \w+ \[ ?([^\]\d][^\]]*?) ?\]", rtx)
    return m.group(1) if m else None


def rtl_sites(lines):
    """[{op, f_line, dest, op1, op2}] of the floating smax/smin insns of one function's final RTL dump, in order."""
    text = "\n".join(lines)
    out = []
    for m in _RTL_OP.finditer(text):
        end = _balanced(text, m.start())
        body = text[m.end():end - 1].strip()
        k1 = _balanced(body, 0)
        op1 = body[:k1].strip()
        rest = body[k1:].strip()
        op2 = rest[:_balanced(rest, 0)].strip()
        s0 = text.rfind("(set ", 0, m.start())
        dest = text[s0 + 5:_balanced(text, s0 + 5)] if s0 >= 0 else ""
        loc = _RTL_LOC.search(text, end, end + 400)
        out.append({"op": "MAX" if m.group(1) == "smax" else "MIN", "mode": m.group(2),
                    "f_line": int(loc.group(2)) if loc else None, "dest": " ".join(dest.split()),
                    "op1": " ".join(op1.split()), "op2": " ".join(op2.split())})
    return out


_HARD = re.compile(r"^\(reg:\w+ (\d+) ")


def _matches(rname, gop, find):
    """Does the RTL operand name `rname` denote the GIMPLE operand `gop`? Same SSA name, SSA names joined by a copy
    (`M.0_19 = _18;`), or a memory operand naming the variable of an SSA name (`s` for `s_36`)."""
    if rname is None:
        return False
    if rname == gop or find(rname) == find(gop):
        return True
    gb = _base(gop)
    return gb is not None and rname == gb


def rtl_check(step, rins, find):
    """The machine-level winner of one step: the insn is `maxsd/minsd op2, op1` with op1 the destination register
    (the SSE pattern ties operand 1 to the output), and maxsd/minsd return op2, the source, when the operands are
    equal or either is NaN (Intel SDM). op2 (else, by elimination, op1) is resolved to a GIMPLE operand by name."""
    g = step["ops"]
    n1, n2 = _rtl_name(rins["op1"]), _rtl_name(rins["op2"])
    hd, h1 = _HARD.match(rins["dest"]), _HARD.match(rins["op1"])
    info = {"op1": n1, "op2": n2, "insn": f"(set {rins['dest']} ({'smax' if rins['op'] == 'MAX' else 'smin'}:"
            f"{rins['mode']} {rins['op1']} {rins['op2']}))"}
    if not (hd and h1 and hd.group(1) == h1.group(1)):
        return {**info, "status": "op1 is not the destination register"}
    src = None
    for i in (0, 1):
        if _matches(n2, g[i], find):
            src = i
            break
    if src is None:
        for i in (0, 1):
            if _matches(n1, g[i], find):
                src = 1 - i
                break
    if src is None:
        return {**info, "status": "unresolved"}
    return {**info, "status": "resolved", "src": g[src], "src_index": src}


def file_sites(tree, rtl):
    """Chains (sites) of one compiled file: [{routine, op, type, steps:[...], f_line...}], plus problems."""
    tsec, rsec = _sections(tree, "tree"), _sections(rtl, "rtl")
    sites, problems = [], []
    for fn, lines in tsec.items():
        steps, find = gimple_sites(lines)
        rins = rtl_sites(rsec.get(fn, []))
        if len(steps) != len(rins) or any(s["op"] != r["op"] or s["f_line"] != r["f_line"]
                                          for s, r in zip(steps, rins)):
            problems.append(f"{fn}: {len(steps)} GIMPLE steps vs {len(rins)} RTL insns do not pair "
                            f"({[(s['op'], s['f_line']) for s in steps]} vs {[(r['op'], r['f_line']) for r in rins]})")
            rins = [None] * len(steps)
        chains = {}
        for s, r in zip(steps, rins):
            s["rtl"] = rtl_check(s, r, find) if r is not None else {"status": "unpaired"}
            if s["rtl"]["status"] != "resolved":
                problems.append(f"{fn}: {s['gimple']}: RTL {s['rtl']['status']} ({s['rtl'].get('insn', '')[:300]})")
                s["winner"] = None
            else:   # the machine-level winner (rule 3 is the GIMPLE prediction, kept for the record)
                s["winner"] = "acc" if s["rtl"]["src_index"] == s["acc_index"] else "new"
            chains.setdefault(s["mvar"], []).append(s)
        for mvar, ch in chains.items():
            ops = {s["op"] for s in ch}
            if len(ops) != 1:
                problems.append(f"{fn}: chain {mvar} mixes MAX and MIN")
            sites.append({"routine": fn, "op": ch[0]["op"], "type": ch[0]["type"], "nargs": len(ch) + 1,
                          "f_file": ch[0]["f_file"], "f_line": ch[0]["f_line"],
                          "p": "".join({"acc": "a", "new": "b"}.get(s["winner"], "?") for s in ch),
                          "steps": [{"winner": s["winner"], "gimple_winner": s["gimple_winner"],
                                     "gimple": s["gimple"], "rtl": s["rtl"], "f_line": s["f_line"]} for s in ch]})
    return sites, problems


# ---------------------------------------------------------------------------------------------------------------
# .f -> .F

def is_continuation(line):
    """Fixed form: a non-comment line with a character other than blank or '0' in column 6 and columns 1-5 blank."""
    if not line or line[0] in "Cc*!":
        return False
    return len(line) > 5 and line[5] not in " 0" and line[:5].strip() == ""


def statement_range(f_lines, k):
    """1-based first and last .f line of the statement containing line k."""
    a = k
    while a > 1 and is_continuation(f_lines[a - 1]):
        a -= 1
    b = k
    while b < len(f_lines) and is_continuation(f_lines[b]):
        b += 1
    return a, b


class LineMapper:
    def __init__(self, build_dir):
        self.bld = Path(build_dir) / "bld"
        self.view = cpp_live.Build("?", "?", cpp_options.Farm(self.bld, {}), toolchain(build_dir))
        self.cache = {}

    def source_name(self, fname):
        stem = fname.rsplit(".", 1)[0]
        for ext in (".F", ".F90"):
            if (self.bld / (stem + ext)).exists():
                return stem + ext
        raise FileNotFoundError(f"no source of {fname} in {self.bld}")

    def where(self, name):
        """Upstream-relative path of a file of the build directory (a link into the build's MITgcm snapshot)."""
        p = (self.bld / name)
        r = str(p.resolve()) if p.exists() else name
        return r.rsplit("/MITgcm/", 1)[1] if "/MITgcm/" in r else r

    def map(self, fname):
        if fname not in self.cache:
            f_text = (self.bld / fname).read_text(encoding="latin-1")
            name = self.source_name(fname)
            m = cpp_live.compiled_line_map(self.view, name, f_text)
            self.cache[fname] = (m, cpp_options.output_lines(f_text))
        return self.cache[fname]

    def annotate(self, site):
        m, f_lines = self.map(site["f_file"])
        a, b = statement_range(f_lines, site["f_line"])
        locs = [m[i - 1] for i in range(a, b + 1) if m[i - 1] is not None]
        at = m[site["f_line"] - 1]
        if at is None or not locs:
            raise ValueError(f"{site['f_file']}:{site['f_line']} maps to no source line")
        files = {f for f, _ in locs}
        if len(files) != 1:
            raise ValueError(f"{site['f_file']}:{a}-{b} spans several files {files}")
        fname = at[0] if at[0] != cpp_live.STDIN else self.source_name(site["f_file"])
        site["F_file"] = self.where(fname)
        site["F_line"] = at[1]
        site["F_stmt"] = [min(ln for _, ln in locs), max(ln for _, ln in locs)]
        site["f_stmt"] = [a, b]
        site["text"] = " ".join(f_lines[i - 1].strip() for i in range(a, b + 1))
        return site


# ---------------------------------------------------------------------------------------------------------------
# one build

def table_name(build_dir):
    """`<build>` for $MJX_REFERENCE/build/<build>, `<family>__<build>` for $MJX_REFERENCE/<family>/<build>."""
    build_dir = Path(build_dir)
    return build_dir.name if build_dir.parent.name == "build" else f"{build_dir.parent.name}__{build_dir.name}"


def run_build(build_dir, out_root, scratch_root, jobs):
    build_dir = Path(build_dir).resolve()
    name = table_name(build_dir)
    bld = build_dir / "bld"
    fc, version = compiler(build_dir)
    fflags = makefile_var(bld, "FFLAGS").split() + makefile_var(bld, "FOPTIM").split()
    noopt = set(makefile_var(bld, "NOOPTFILES").split())
    if noopt:
        raise NotImplementedError(f"{bld}: NOOPTFILES {sorted(noopt)} (not handled)")
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    wdir = Path(scratch_root) / f"{name}-{stamp}-{os.getpid()}"
    wdir.mkdir(parents=True, exist_ok=False)
    srcs = sorted(bld.glob("*.f"))
    # Fortran modules (e.g. ptracers_dyn_state_mod of tutorial_advection_in_gyre, lane A session 8): the files are
    # compiled in parallel in the scratch directory, so a `use` must find the build's own .mod files: -I<bld>
    # (module search path only; the object is still compared with the build's byte for byte)
    mod_dirs = [f"-I{bld}"] if any(bld.glob("*.mod")) else []

    def work(src):
        tree, rtl, obj, err = compile_one(fc, fflags + mod_dirs, src, wdir)
        ref = (bld / (src.stem + ".o")).read_bytes()
        sites, problems = file_sites(tree, rtl)
        return src.name, obj == ref, sites, problems

    results = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(work, srcs):
            results.append(r)
    not_identical = [n for n, same, _, _ in results if not same]
    sites = [s for _, _, ss, _ in results for s in ss]
    problems = [p for _, _, _, ps in results for p in ps]
    mapper = LineMapper(build_dir)
    files = sorted({s["f_file"] for s in sites})
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(mapper.map, files))
    for s in sites:
        mapper.annotate(s)
    sites.sort(key=lambda s: (s["F_file"], s["F_line"], s["routine"], s["f_line"]))
    rtl = {}
    for s in sites:
        for st in s["steps"]:
            k = st["rtl"]["status"] + ("" if st["rtl"]["status"] != "resolved" else
                                       ", = first GIMPLE operand" if st["winner"] == st["gimple_winner"] else
                                       ", = second GIMPLE operand")
            rtl[k] = rtl.get(k, 0) + 1
    meta = {"build": name, "build_dir": str(build_dir), "compiler": fc, "compiler_version": version,
            "flags": fflags, **({"module_search": mod_dirs} if mod_dirs else {}), "n_files": len(srcs), "objects_identical": len(srcs) - len(not_identical),
            "objects_differ": not_identical, "n_sites": len(sites),
            "n_steps": sum(len(s["steps"]) for s in sites), "rtl_check": rtl, "problems": problems,
            "scratch": str(wdir), "tool": "tools/fortran_minmax_sites.py",
            "commit": ((REPO / "COMMIT").read_text().strip() if (REPO / "COMMIT").is_file() else
                       subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()),
            "date": datetime.datetime.now().isoformat(timespec="seconds")}
    write_table(Path(out_root), name, {"meta": meta, "sites": sites})
    return meta


def _payload(doc):
    """The content that must not change between runs (meta minus date, scratch and commit)."""
    m = {k: v for k, v in doc["meta"].items() if k not in ("date", "scratch", "commit")}
    return json.dumps({"meta": m, "sites": doc["sites"]}, sort_keys=True)


def write_table(out_root, name, doc):
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{name}.json"
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists():
        old = json.loads(path.read_text())
        if _payload(old) != _payload(doc):
            raise RuntimeError(f"{path} exists with different content; refusing to overwrite (use a new --out-root)")
        return path
    path.write_text(text)
    (out_root / f"{name}.json.sha256").write_text(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}.json\n")
    return path


# ---------------------------------------------------------------------------------------------------------------
# comparison across builds

def site_ids(sites):
    """{(F_file, first, last .F line of the statement, routine, op, ordinal): p} of one table (the ordinal counts the
    sites of one operation in one statement in compilation order)."""
    out, seen = {}, {}
    for s in sites:
        k = (s["F_file"], s["F_stmt"][0], s["F_stmt"][1], s["routine"], s["op"])
        n = seen.get(k, 0)
        seen[k] = n + 1
        out[k + (n,)] = s["p"]
    return out


def load(path):
    return json.loads(Path(path).read_text())


def compare(paths_):
    """Sites compiled in several builds: is `p` the same in every build that compiles the site?
    Returns ({site: {p: [builds]}} of the sites that differ, number of sites, per-build site counts)."""
    per, counts = {}, {}
    for path in paths_:
        t = load(path)
        b = t["meta"]["build"]
        ids = site_ids(t["sites"])
        counts[b] = len(ids)
        for k, p in ids.items():
            per.setdefault(k, {}).setdefault(p, []).append(b)
    differ = {k: v for k, v in per.items() if len(v) > 1}
    return differ, len(per), counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("build_dirs", nargs="+")
    r.add_argument("--out-root", default=str(paths.REFERENCE / "minmax_sites"))
    r.add_argument("--scratch-root", default=str(paths.RUNS / "minmax_sites"))
    r.add_argument("-j", "--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    c = sub.add_parser("compare")
    c.add_argument("tables", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        bad = 0
        for b in a.build_dirs:
            if not (Path(b) / "bld" / "Makefile").is_file():
                print(f"SKIP {b}: not a genmake2 build (no bld/Makefile)")
                continue
            meta = run_build(b, a.out_root, a.scratch_root, a.jobs)
            ok = not meta["objects_differ"] and not meta["problems"]
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {meta['build']}: {meta['n_files']} files, objects identical "
                  f"{meta['objects_identical']}/{meta['n_files']}, {meta['n_sites']} REAL MAX/MIN sites "
                  f"({meta['n_steps']} steps), rtl check {meta['rtl_check']}, problems {len(meta['problems'])}")
            for p in meta["problems"][:20]:
                print("   ", p)
        return 1 if bad else 0
    differ, n, counts = compare(a.tables)
    print(f"{n} distinct sites in {len(a.tables)} tables ({counts}); {len(differ)} with a different p between builds")
    for k, v in sorted(differ.items()):
        print(f"  {k}: {v}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
