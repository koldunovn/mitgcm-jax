"""Just enough fixed-form Fortran to read declarations, PARAMETER and NAMELIST statements from preprocessed MITgcm
sources (the output of mitjax/config/cpp_options.preprocess, i.e. what the compiler sees). Used by size.py (SIZE.h)
and namelists.py (namelist groups and the types and shapes of their variables).

Fixed form (gfortran -ffixed-form): a line with C, c, *, or ! in column 1 is a comment; a line whose column 6 is
neither blank nor '0' (columns 1-5 blank) continues the previous statement; statement text starts in column 7; an
`!` outside a character constant starts a comment. Anything not understood in a statement this module is asked to
read raises ValueError; statements of other kinds are ignored. Stdlib only.

Also `real4`: the binary32 value of a default-kind (REAL*4) real, which is what a Fortran real literal without a `D`
exponent or `_d` is in this build (no -fdefault-real-8: reference/optfile_levante_gfortran:87-88, the line is
commented out), and what a REAL*4 variable stores.
"""

import ast
import re
import struct
from dataclasses import dataclass
from fractions import Fraction


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def real4(x):
    """The REAL*4 (binary32) value, returned as the Python float it widens to exactly.

    x a str: the decimal text of a real literal or namelist value (`0.1`, `1.e-6`, `1.0E-12`, `2.`; a `D` exponent is
    read as E), rounded ONCE to the nearest binary32, ties to even, from the exact decimal value: gfortran converts a
    default-kind literal at compile time with that rounding (MPFR), and the namelist READ of a REAL*4 variable gave
    the same value for every string of up to 12 significant digits tried (probe compiled with the oracle flags,
    $MJX_RUNS/hardening_scratch/r4probe_*). Going through float64 first (`np.float32(float(text))`) rounds twice and
    differs from gfortran on rare inputs (e.g. 1 + 2**-24 + 2**-60 written out: gfortran 1 + 2**-23, twice-rounded 1.0).
    x a float or int: the value converted to REAL*4 (one rounding from the double, as a REAL*8 -> REAL*4 assignment)."""
    if isinstance(x, str):
        t = x.strip().replace("D", "E").replace("d", "e")
        if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([Ee][+-]?\d+)?", t):
            raise ValueError(f"{x!r} is not a real literal")
        fr = Fraction(t)
        mag = abs(fr)
        b = _f32_bits(float(mag))                     # within one binary32 ulp of the exact value
        cands = [c for c in (b - 1, b, b + 1) if 0 <= c < 0x7F800000]
        best = min(cands, key=lambda c: (abs(Fraction(_f32_value(c)) - mag), c & 1))
        if best == 0x7F7FFFFF and Fraction(_f32_value(best)) < mag:
            raise OverflowError(f"{x!r} overflows REAL*4")
        v = _f32_value(best)
        return -v if fr < 0 else v
    return _f32_value(_f32_bits(float(x)))


def real_bytes(ftype):
    """Storage size of a REAL declaration as written after preprocessing: `Real*8`/`DOUBLE PRECISION` 8, `Real*4` and
    plain `REAL` (default kind, no -fdefault-real-8) 4; None for a non-REAL type."""
    t = ftype.lower().replace(" ", "")
    if t.startswith("doubleprecision"):
        return 8
    if t.startswith("real"):
        m = re.fullmatch(r"real(?:\*(\d+))?", t)
        if not m:
            raise ValueError(f"cannot read the kind of {ftype!r}")
        return int(m.group(1)) if m.group(1) else 4
    return None


def statements(text, source="<text>"):
    """[(first line number, statement text)] of a fixed-form source, continuations joined."""
    out = []
    for n, raw in enumerate(text.split("\n"), start=1):
        if raw[:1] in ("C", "c", "*", "!") or not raw.strip():
            continue
        if raw.lstrip().startswith("!"):
            continue
        body = _strip_bang(raw[6:] if len(raw) > 6 else "")
        if len(raw) > 5 and raw[5] not in " 0" and raw[:5].strip() == "":
            if not out:
                raise ValueError(f"{source}:{n}: continuation line without a statement")
            out[-1] = (out[-1][0], out[-1][1] + body)
        else:
            if raw[:5].strip() and not raw[:5].strip().isdigit():
                raise ValueError(f"{source}:{n}: not fixed-form Fortran: {raw!r}")
            out.append((n, body))
    return out


def _strip_bang(s):
    q = None
    for i, ch in enumerate(s):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "!":
            return s[:i]
    return s


def split_top(s, sep=","):
    """Split at `sep` outside parentheses and quotes."""
    parts, depth, q, cur = [], 0, None, []
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def int_expr(expr, env, where="?"):
    """Value of a Fortran integer constant expression (+ - * / ** and parentheses; integer division truncates
    toward zero) with names from env (case-insensitive)."""
    e = expr.strip()
    try:
        tree = ast.parse(e.lower(), mode="eval")
    except SyntaxError:
        raise ValueError(f"{where}: cannot read integer expression {expr!r}") from None
    envl = {k.lower(): v for k, v in env.items()}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in envl:
                raise KeyError(f"{where}: {node.id} in {expr!r} has no known integer value")
            return envl[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                q = abs(a) // abs(b)
                return q if (a >= 0) == (b >= 0) else -q
            if isinstance(node.op, ast.Pow):
                return a ** b
        raise ValueError(f"{where}: unsupported integer expression {expr!r}")
    return ev(tree)


def parameters(stmts, env=None, source="?"):
    """Integer PARAMETER values in statement order ({name: int}); non-integer parameters are skipped (they are not
    array bounds). env holds values already known (e.g. SIZE.h)."""
    env = dict(env or {})
    for n, s in stmts:
        m = re.match(r"\s*parameter\s*\((.*)\)\s*$", s, re.I | re.S)
        if not m:
            continue
        for part in split_top(m.group(1)):
            k, _, v = part.partition("=")
            try:
                env[k.strip()] = int_expr(v, env, f"{source}:{n}")
            except (ValueError, KeyError):
                continue
    return env


_TYPE = re.compile(r"\s*(real\s*\*\s*8|real\s*\*\s*4|real|double\s+precision|integer\s*\*\s*\d+|integer|logical|"
                   r"character\s*\*\s*\(\s*[^)]*\)|character\s*\*\s*\d+|character\s*\(\s*(?:len\s*=\s*)?[^)]*\)|"
                   r"character)\s*(.*)$", re.I | re.S)


@dataclass(frozen=True)
class Decl:
    name: str          # as written
    kind: str          # "float", "int", "bool", "str"
    ftype: str         # the type spec as written (after preprocessing), e.g. "Real*8"
    dims: tuple        # ((lower expr, upper expr), ...) as written; () for a scalar
    line: int


def _kind(ftype):
    t = ftype.lower().replace(" ", "")
    if t.startswith(("real", "doubleprecision")):
        return "float"
    if t.startswith("integer"):
        return "int"
    if t.startswith("logical"):
        return "bool"
    if t.startswith("character"):
        return "str"
    raise ValueError(ftype)


def declarations(stmts):
    """{lower-case name: Decl} of the type declaration statements."""
    out = {}
    for n, s in stmts:
        m = _TYPE.match(s)
        if not m or re.match(r"\s*(function|subroutine)\b", m.group(2), re.I):
            continue
        if m.group(2).lstrip().lower().startswith("function"):
            continue
        ftype, rest = m.group(1), m.group(2)
        for ent in split_top(rest):
            em = re.fullmatch(r"([A-Za-z]\w*)\s*(?:\*\s*(?:\d+|\([^)]*\)))?\s*(?:\((.*)\))?", ent.strip(), re.S)
            if not em:
                break                                       # not a declaration after all (e.g. `real function`)
            dims = ()
            if em.group(2) is not None:
                dims = tuple((d.split(":")[0].strip(), d.split(":")[1].strip()) if ":" in d else ("1", d.strip())
                             for d in split_top(em.group(2)))
            out[em.group(1).lower()] = Decl(em.group(1), _kind(ftype), ftype.strip(), dims, n)
    return out


def shape(decl, env, where="?"):
    """Extents of a declared array (int tuple), () for a scalar."""
    return tuple(int_expr(hi, env, where) - int_expr(lo, env, where) + 1 for lo, hi in decl.dims)


def namelists(stmts):
    """{lower-case group: (line, [variable names])} of the NAMELIST statements."""
    out = {}
    for n, s in stmts:
        m = re.match(r"\s*namelist\s*/\s*(\w+)\s*/(.*)$", s, re.I | re.S)
        if not m:
            continue
        if re.search(r"/\s*\w+\s*/", m.group(2)):
            raise ValueError(f"line {n}: several groups in one NAMELIST statement")
        g = m.group(1).lower()
        if g in out:
            raise ValueError(f"line {n}: NAMELIST /{g}/ declared twice")
        out[g] = (n, [v.strip() for v in split_top(m.group(2)) if v.strip()])
    return out
