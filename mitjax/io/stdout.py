"""Parsers for MITgcm standard output (STDOUT.0000, verification results/output*.txt): monitor blocks, cg2d residuals,
SBO values, gradient-check (grdchk) points with their ADM lines, and the parameter printout.

Formats are those written by MITgcm @63cdc0b and by the older checkpoints (58u..69i) of the M1 reference files;
anything else raises ValueError naming the file line, never a silent skip. Writers (MJX_UPSTREAM):
    %MON name = value          pkg/monitor/monitor.F:82,187 (Begin/End), mon_out*.F; AD_MONITOR: monitor_ad.F:97,228;
                               ptracers: pkg/ptracers/ptracers_monitor.F:86,126
    cg2d_init_res = v          model/src/solve_for_pressure.F:336 '(A20,1PE23.14)'; cg2d_iters(min,last), cg2d_last_res
    %SBO name = value          pkg/sbo/sbo_output.F:111-120
    grdchk / ADM lines         pkg/grdchk/grdchk_main.F, grdchk_print.F
    name = /* comment */       eesupp/src/write_utils.F (WRITE_0D_*, WRITE_1D_*: header, values, '    ;')
    name = value ; /* ... */   eesupp/src/eewrite_eeenv.F (execution environment: SIZE.h, eedata)
Stdlib only.
"""

import gzip
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_PREFIX = re.compile(r"^\(PID\.TID (\d{4}\.\d{4})\) ?")


@dataclass(frozen=True)
class Line:
    number: int          # 1-based line number in the file
    pid: str | None      # "0000.0001" for "(PID.TID 0000.0001) " lines, None for unprefixed lines
    text: str            # the line without that prefix


def split_lines(text):
    """Lines of an output text, with the "(PID.TID xxxx.xxxx) " prefix split off."""
    out = []
    for n, raw in enumerate(text.split("\n"), start=1):
        m = _PREFIX.match(raw)
        out.append(Line(n, m[1], raw[m.end():]) if m else Line(n, None, raw))
    if out and out[-1].text == "" and out[-1].pid is None:
        out.pop()
    return out


def read_stdout(path):
    """Lines of an output file (a .gz file is decompressed); bytes are kept 1:1 (latin-1)."""
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return split_lines(data.decode("latin-1"))


def _err(line, what):
    return ValueError(f"line {line.number}: {what}: {line.text!r}")


_INT = re.compile(r"[+-]?\d+")
_REAL = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([EeDd][+-]?\d+)?")
_REAL_E3 = re.compile(r"([+-]?\d+\.\d*)([+-]\d{3})")     # 1PE format with a 3-digit exponent drops the 'E'
_SPECIAL = {"nan": math.nan, "infinity": math.inf, "+infinity": math.inf, "-infinity": -math.inf,
            "inf": math.inf, "+inf": math.inf, "-inf": -math.inf}


def fortran_number(text):
    """A number as Fortran prints it: integer -> int; E/D real, 3-digit-exponent real (`2.0962566239063-116`),
    gfortran's NaN / Infinity -> float. Anything else (e.g. `*****`) raises ValueError."""
    t = text.strip()
    if _INT.fullmatch(t):
        return int(t)
    if _REAL.fullmatch(t):
        return float(t.replace("D", "E").replace("d", "e"))
    m = _REAL_E3.fullmatch(t)
    if m:
        return float(f"{m[1]}E{m[2]}")
    if t.lower() in _SPECIAL:
        return _SPECIAL[t.lower()]
    raise ValueError(f"not a Fortran number: {text!r}")


# ---------------------------------------------------------------------------------------------------------------
# %MON blocks

# End markers of older checkpoints that differ from their Begin marker (the reference files of
# global_ocean.90x40x15/input.dwnslp and others; @63cdc0b ptracers_monitor.F:86,126 print "ptracer" in both).
_END_ALIASES = {
    "MONITOR ptracer field statistics": {"MONITOR ptracers field statistics"},
    "MONITOR adptracer field statistics": {"MONITOR ptracers field statistics"},
}
_BEGIN_END = re.compile(r"^// (Begin|End) (.*MONITOR.*)$")
_MON = re.compile(r"^%MON\s+(\S+)\s*=\s*(\S+)\s*$")

FORWARD_DYNAMICS = "MONITOR dynamic field statistics"
ADJOINT_DYNAMICS = "AD_MONITOR dynamic field statistics"


@dataclass
class MonBlock:
    kind: str | None          # Begin marker text ("MONITOR dynamic field statistics", ...); None: %MON lines
                              # outside any Begin/End block (the grid statistics after INI_VERTICAL_GRID)
    first: int                # line number of the Begin marker (or of the first %MON line)
    last: int = 0             # line number of the End marker (or of the last %MON line)
    text: dict = field(default_factory=dict)     # name -> value text, in file order
    lines: dict = field(default_factory=dict)    # name -> line number

    @property
    def values(self):
        return {k: fortran_number(v) for k, v in self.text.items()}

    def __getitem__(self, name):
        return fortran_number(self.text[name])


def monitor_blocks(lines):
    """All %MON blocks in file order. Inside a "// Begin ...MONITOR..." / "// End ..." pair every %MON line belongs to
    the block; consecutive %MON lines outside any pair form one block of kind None. Raises on nested or unmatched
    markers, a %MON line not of the form `%MON name = value`, or a name repeated within a block."""
    blocks, cur, loose = [], None, None
    for ln in lines:
        m = _BEGIN_END.match(ln.text)
        if m:
            loose = None
            if m[1] == "Begin":
                if cur is not None:
                    raise _err(ln, f"Begin inside the block opened at line {cur.first}")
                cur = MonBlock(m[2], ln.number)
            else:
                if cur is None:
                    raise _err(ln, "End without Begin")
                if m[2] != cur.kind and m[2] not in _END_ALIASES.get(cur.kind, ()):
                    raise _err(ln, f"End does not match Begin {cur.kind!r} at line {cur.first}")
                cur.last = ln.number
                blocks.append(cur)
                cur = None
            continue
        if not ln.text.startswith("%MON"):
            if cur is None:
                loose = None
            continue
        mm = _MON.match(ln.text)
        if mm is None:
            raise _err(ln, "unknown %MON format")
        if cur is None and loose is None:
            loose = MonBlock(None, ln.number)
            blocks.append(loose)
        blk = cur if cur is not None else loose
        if mm[1] in blk.text:
            raise _err(ln, f"{mm[1]} twice in the block starting at line {blk.first}")
        blk.text[mm[1]] = mm[2]
        blk.lines[mm[1]] = ln.number
        blk.last = ln.number
    if cur is not None:
        raise ValueError(f"block {cur.kind!r} opened at line {cur.first} is never closed")
    return blocks


def monitor_series(blocks, name, kind=FORWARD_DYNAMICS):
    """Values of %MON `name` in every block of `kind`, in file order."""
    return [b[name] for b in blocks if b.kind == kind and name in b.text]


# ---------------------------------------------------------------------------------------------------------------
# cg2d

_CG2D = re.compile(r"^\s*(CG2D_MAD: )?cg2d_(init_res|iters\(min,last\)|last_res)\s*=\s*(.*?)\s*$")


@dataclass
class Cg2dRecord:
    line: int                 # line of cg2d_init_res (testreport's `PS`)
    init_res: float
    mad: bool = False         # printed by CG2D_MAD (the adjoint solve)
    iters: tuple | None = None        # (min, last) from cg2d_iters(min,last)
    last_res: float | None = None


def cg2d_records(lines):
    """One record per solver call that printed cg2d_init_res, with the iteration counts and final residual printed
    after it (solve_for_pressure.F:336-345). Raises on an iters/last_res line with no open record or printed twice."""
    recs = []
    for ln in lines:
        m = _CG2D.match(ln.text)
        if m is None:
            continue
        what, val = m[2], m[3]
        if what == "init_res":
            recs.append(Cg2dRecord(ln.number, fortran_number(val), bool(m[1])))
            continue
        if not recs:
            raise _err(ln, "cg2d line before any cg2d_init_res")
        r = recs[-1]
        if what == "last_res":
            if r.last_res is not None:
                raise _err(ln, f"second cg2d_last_res for cg2d_init_res at line {r.line}")
            r.last_res = fortran_number(val)
        else:
            if r.iters is not None:
                raise _err(ln, f"second cg2d_iters for cg2d_init_res at line {r.line}")
            parts = val.split()
            if len(parts) != 2:
                raise _err(ln, "cg2d_iters(min,last) needs two integers")
            r.iters = (int(parts[0]), int(parts[1]))
    return recs


# ---------------------------------------------------------------------------------------------------------------
# SBO

_SBO = re.compile(r"^%SBO\s+(\S+)\s*=\s*(\S+)\s*$")


@dataclass
class SboRecord:
    line: int                                    # line of the first %SBO line of the group
    text: dict = field(default_factory=dict)     # name -> value text

    def __getitem__(self, name):
        return fortran_number(self.text[name])


def sbo_records(lines):
    """Consecutive %SBO lines (sbo_output.F:111-120) grouped into one record each, in file order."""
    recs, prev = [], None
    for ln in lines:
        if not ln.text.startswith("%SBO"):
            continue
        m = _SBO.match(ln.text)
        if m is None:
            raise _err(ln, "unknown %SBO format")
        if prev is None or ln.number != prev + 1 or m[1] in recs[-1].text:
            recs.append(SboRecord(ln.number))
        recs[-1].text[m[1]] = m[2]
        prev = ln.number
    return recs


# ---------------------------------------------------------------------------------------------------------------
# grdchk

_FCREF = re.compile(r"^grdchk reference fc: fcref\s*=\s*(\S+)\s*$")
_POS = re.compile(r"^grdchk pos: i,j,k=\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*;\s*bi,bj=\s*(-?\d+)\s+(-?\d+)\s*;"
                  r"\s*iobc=\s*(-?\d+)\s*;\s*rec=\s*(-?\d+)\s*$")
_PERT = re.compile(r"^grdchk perturb\(([+-])\)fc: fcpert(plus|minus)\s*=\s*(\S+)\s*$")
_ADM = re.compile(r"^\s*ADM  (ref_cost_function|adjoint_gradient|finite-diff_grad)\s*=\s*(\S+)\s*$")
_TABLE = re.compile(r"^grdchk output \(([pcg])\):\s*(.*?)\s*$")
_ADM_ORDER = ("ref_cost_function", "adjoint_gradient", "finite-diff_grad")


@dataclass
class GrdchkPoint:
    line: int                 # line of "grdchk pos:"
    i: int
    j: int
    k: int
    bi: int
    bj: int
    iobc: int
    rec: int
    fcpertplus: float | None = None
    fcpertminus: float | None = None
    adm: dict = field(default_factory=dict)       # ref_cost_function / adjoint_gradient / finite-diff_grad -> value
    adm_lines: dict = field(default_factory=dict)


@dataclass
class Grdchk:
    fcref: float | None
    points: list


def grdchk(lines):
    """Gradient-check points in file order: position, perturbed costs and the three ADM lines each (testreport's
    admCst / admGrd / admFwd). Raises when an ADM line has no open point, comes out of order or twice, a point lacks
    any of the three, or the summary table ("grdchk output (p):") disagrees with the positions."""
    fcref, pts, table = None, [], []
    for ln in lines:
        t = ln.text
        if m := _FCREF.match(t):
            fcref = fortran_number(m[1])
        elif m := _POS.match(t):
            pts.append(GrdchkPoint(ln.number, *(int(x) for x in m.groups())))
        elif t.startswith("grdchk pos"):
            raise _err(ln, "unknown grdchk pos format")
        elif m := _PERT.match(t):
            if not pts:
                raise _err(ln, "perturbed cost before any grdchk pos")
            setattr(pts[-1], "fcpert" + m[2], fortran_number(m[3]))
        elif m := _ADM.match(t):
            if not pts:
                raise _err(ln, "ADM line before any grdchk pos")
            p = pts[-1]
            want = _ADM_ORDER[len(p.adm)] if len(p.adm) < 3 else None
            if m[1] != want:
                raise _err(ln, f"ADM line out of order for the point at line {p.line} (expected {want})")
            p.adm[m[1]] = fortran_number(m[2])
            p.adm_lines[m[1]] = ln.number
        elif re.match(r"^\s*ADM ", t):
            raise _err(ln, "unknown ADM line format")
        elif (m := _TABLE.match(t)) and m[1] == "p":
            table.append((ln, m[2].split()))
    for p in pts:
        if len(p.adm) != 3:
            raise ValueError(f"line {p.line}: grdchk point without its three ADM lines (got {sorted(p.adm)})")
    if table:
        if len(table) != len(pts):
            raise ValueError(f"{len(table)} 'grdchk output (p)' rows for {len(pts)} grdchk points")
        for (ln, f), p in zip(table, pts):
            if [int(x) for x in f[1:6]] != [p.i, p.j, p.k, p.bi, p.bj]:
                raise _err(ln, f"table row disagrees with grdchk pos at line {p.line}")
    return Grdchk(fcref, pts)


# ---------------------------------------------------------------------------------------------------------------
# parameter printout

_HEADER = re.compile(r"^\s*([^\s=]+)\s*=\s*(/\*.*\*/)\s*$")
_INLINE = re.compile(r"^\s*([^\s=]+)\s*=\s*(.*?)\s*;\s*(/\*.*\*/)\s*$")
_REPEAT = re.compile(r"^(\d+)\s+@\s+(.*)$")


@dataclass
class Param:
    name: str
    text: str                 # value lines as printed (stripped), joined by "\n"
    line: int                 # line of the header
    comment: str              # the /* ... */ description
    form: str                 # "block" (write_utils.F) or "inline" (eewrite_eeenv.F)

    def values(self):
        """The printed values, one string per element: `n @ v` repeats expanded, `/* ... */` annotations and the
        separating commas dropped. A list printed in part (". . ." lines) raises ValueError."""
        out = []
        for raw in self.text.split("\n"):
            v = re.sub(r"/\*.*?\*/", "", raw).strip()
            if v == ". . .":
                raise ValueError(f"{self.name} (line {self.line}): list printed in part ('. . .')")
            if v.endswith(","):
                v = v[:-1].rstrip()
            m = _REPEAT.match(v)
            out += [m[2].strip()] * int(m[1]) if m else [v]
        return out


def parameters(lines):
    """The parameter printout in file order: blocks `name = /* comment */`, value lines, `;` (WRITE_0D_*/WRITE_1D_*)
    and the one-line form `name = value ; /* comment */` (execution environment). Raises on a header inside an open
    block or a block never closed."""
    params, cur, vals = [], None, []
    for ln in lines:
        t = ln.text
        if cur is not None:
            if t.strip() == ";":
                params.append(Param(cur[0], "\n".join(vals), cur[2], cur[1], "block"))
                cur, vals = None, []
            elif _HEADER.match(t) or _INLINE.match(t):
                raise _err(ln, f"parameter header inside the block of {cur[0]!r} (line {cur[2]})")
            else:
                vals.append(t.strip())
            continue
        if m := _HEADER.match(t):
            cur = (m[1], m[2], ln.number)
        elif m := _INLINE.match(t):
            params.append(Param(m[1], m[2], ln.number, m[3], "inline"))
    if cur is not None:
        raise ValueError(f"line {cur[2]}: parameter block {cur[0]!r} is never closed by ';'")
    return params


def parameter_dict(params):
    """name -> Param (a repeated printout with the same value is kept once). A name printed with different values
    under different comments, such as the grid lines `dxF = /* dxF(:,1,:,1) ... */` and `dxF = /* dxF(1,:,1,:) ... */`
    (write_utils.F WRITE_XY_XLINE_RS / _YLINE_RS), is keyed "name /* comment */" for each printout instead; a
    conflict that remains raises ValueError."""
    texts = {}
    for p in params:
        texts.setdefault(p.name, set()).add(p.text)
    out = {}
    for p in params:
        key = p.name if len(texts[p.name]) == 1 else f"{p.name} {p.comment}"
        q = out.get(key)
        if q is not None and q.text != p.text:
            raise ValueError(f"{key} printed with different values at lines {q.line} and {p.line}")
        out.setdefault(key, p)
    return out
