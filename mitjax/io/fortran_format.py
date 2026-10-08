"""Fortran edit descriptors as gfortran 11 writes them: the text of a formatted WRITE (plan Task 11, lane MON).

    fortran_write('(1X,1P1E21.13)', x)   -> '   9.6840090317157E-01'   (mon_out.F:197)
    fortran_write('(1X,I21)', 36000)     -> '                 36000'   (mon_out.F:195)
    format_E(x, 21, 13, scale=1)         -> the field alone

The oracle is gfortran itself: mitjax/tests/fortran_format/probe.F (written by make_probe.py there) prints a table of
values (signed zeros, subnormals, huge and tiny numbers, 3-digit exponents, exact rounding ties, NaN, +-Infinity) with
every descriptor below, compiled with the oracle's compiler and flags (gfortran 11.2.0, -O0 -ffp-contract=off);
mitjax/tests/test_fortran_format.py compares this module with that table character for character.

Rules (measured on that table; F2008 10.7.2 where gfortran follows it):
* E/D editing `[kP]Ew.d[Ee]`: k <= 0 prints `0.` then -k zeros and d+k significant digits; 0 < k < d+2 prints k digits,
  the point and d-k+1 digits (d+1 significant digits). The exponent is the decimal exponent minus k. Without `Ee` it is
  `E+dd` for |exp| <= 99 and `+ddd` (no letter) for 99 < |exp| <= 999; with `Ee` it is `E+` and e digits; an exponent
  that does not fit fills the field with asterisks. The leading `0` of a k <= 0 mantissa is dropped only when the field
  is otherwise too narrow.
* ES editing `ESw.d[Ee]`: one nonzero digit (0 for zero), the point, d digits; kP has no effect.
* F editing `[kP]Fw.d`: the value times 10**k with d decimals; a leading `0.` is kept when it fits.
* Digits are the correctly rounded decimal value of the exact binary number, ties to even (glibc printf, which
  libgfortran calls for the default rounding mode; Python's own float formatting is also correctly rounded with ties
  to even, measured equal on every row of the table, including exact ties such as 1.0625 and 100000000000005).
* Signs: '-' for negative numbers including -0.0 and values that round to zero; no '+' (SS, the default).
* NaN prints `NaN`; infinities `Infinity` / `-Infinity` when they fit (`Inf` / `-Inf` otherwise), right-justified.
* A field longer than w is w asterisks; a shorter one is right-justified with blanks.
* I editing `Iw[.m]`: at least m digits (zero-filled), '-' for negative values, asterisks when it does not fit.
* `nX` writes n blanks, `A` the string, `Aw` the string left-justified in (or truncated to) w characters. kP applies to
  every following E/F/D descriptor of the same format until changed (F2008 10.8.5). Repeat counts and parenthesised
  groups `r(...)` are expanded; format reversion (more items than descriptors) is not supported and raises.

Host-side Python on concrete values; stdlib only.
"""

import math
import re

__all__ = ["format_E", "format_ES", "format_F", "format_I", "format_A", "fortran_write", "FortranFormatError"]


class FortranFormatError(ValueError):
    pass


def _stars(w):
    return "*" * w


def _special(x, w):
    """NaN / Infinity as gfortran writes them in a field of width w (libgfortran write.c, build_infnan_string)."""
    if math.isnan(x):
        s = "NaN"
    elif x > 0:
        s = "Infinity" if w >= 8 else "Inf"
    else:
        s = "-Infinity" if w >= 9 else "-Inf"
    return s.rjust(w) if len(s) <= w else _stars(w)


def _digits(x, nsig):
    """(digit string of nsig significant digits, decimal exponent e10 of the first digit) of |x| > 0, correctly
    rounded (ties to even): |x| ~ 0.D1D2... * 10**(e10+1)."""
    s = f"{abs(x):.{nsig - 1}e}"
    mant, exp = s.split("e")
    return mant.replace(".", ""), int(exp)


def _exponent(q, e, letter):
    """Exponent part for exponent value q: `E+dd` / `+ddd` without Ee, `E+` and e digits with it; None if it does not
    fit."""
    sign = "-" if q < 0 else "+"
    a = abs(q)
    if e is None:
        if a <= 99:
            return f"{letter}{sign}{a:02d}"
        if a <= 999:
            return f"{sign}{a:03d}"
        return None
    if a >= 10 ** e:
        return None
    return f"{letter}{sign}{a:0{e}d}"


def format_E(x, w, d, e=None, scale=0, letter="E"):
    """`[kP]Ew.d[Ee]` of a float (k = scale)."""
    x = float(x)
    if not math.isfinite(x):
        return _special(x, w)
    neg = math.copysign(1.0, x) < 0
    k = scale
    if not (-d < k < d + 2):
        raise FortranFormatError(f"scale factor {k}P not allowed with E{w}.{d} (needs -d < k < d+2)")
    nsig = d + k if k <= 0 else d + 1
    if x == 0.0:
        digs, q = "0" * nsig, 0
    else:
        digs, e10 = _digits(x, nsig)
        q = e10 + 1 - k
    if k <= 0:
        mant_body = "." + "0" * (-k) + digs
        lead = "0"
    else:
        mant_body = digs[:k] + "." + digs[k:]
        lead = ""
    ex = _exponent(q, e, letter)
    if ex is None:
        return _stars(w)
    sign = "-" if neg else ""
    s = sign + lead + mant_body + ex
    if len(s) > w and lead:
        s = sign + mant_body + ex
    return s.rjust(w) if len(s) <= w else _stars(w)


def format_ES(x, w, d, e=None):
    """`ESw.d[Ee]` of a float: scientific (one nonzero digit before the point)."""
    x = float(x)
    if not math.isfinite(x):
        return _special(x, w)
    neg = math.copysign(1.0, x) < 0
    if x == 0.0:
        digs, q = "0" * (d + 1), 0
    else:
        digs, q = _digits(x, d + 1)
    ex = _exponent(q, e, "E")
    if ex is None:
        return _stars(w)
    s = ("-" if neg else "") + digs[0] + "." + digs[1:] + ex
    return s.rjust(w) if len(s) <= w else _stars(w)


def format_F(x, w, d, scale=0):
    """`[kP]Fw.d` of a float."""
    x = float(x)
    if not math.isfinite(x):
        return _special(x, w)
    if scale:
        raise FortranFormatError(f"{scale}P with F{w}.{d} is not supported (no MITgcm printout uses it)")
    neg = math.copysign(1.0, x) < 0
    s = f"{abs(x):.{d}f}"
    if s.startswith("0") and len(s) > 1 and s[1] == ".":
        body_short = s[1:]
    else:
        body_short = None
    sign = "-" if neg else ""
    out = sign + s
    if len(out) > w and body_short is not None:
        out = sign + body_short
    return out.rjust(w) if len(out) <= w else _stars(w)


def format_I(n, w, m=None):
    """`Iw[.m]` of an integer."""
    n = int(n)
    digits = str(abs(n))
    if m is not None:
        digits = digits.rjust(m, "0") if m > 0 else ("" if n == 0 else digits)
    s = ("-" if n < 0 else "") + digits
    return s.rjust(w) if len(s) <= w else _stars(w)


def format_A(s, w=None):
    """`A` / `Aw` of a string (F2008 10.7.4: w > len gives leading blanks, w < len the leftmost w characters)."""
    s = str(s)
    if w is None:
        return s
    return s.rjust(w) if len(s) <= w else s[:w]


# ---------------------------------------------------------------------------------------------------------------
# format strings

_TOKEN = re.compile(r"""\s*(?:
      (?P<rep>\d+)?\(                                   # group with optional repeat
    | (?P<close>\))
    | (?P<x>\d+)X                                       # nX
    | (?P<p>[+-]?\d+)P                                  # kP (may be followed directly by a descriptor)
    | (?P<rr>\d+)?(?P<ed>ES|EN|E|D|F|I|A)(?P<w>\d+)?(?:\.(?P<d>\d+))?(?:E(?P<e>\d+))?
    | (?P<comma>,)
    )""", re.X | re.I)


def _parse(fmt):
    """Format string -> nested list of descriptors: ('X', n), ('P', k), (ed, w, d, e), ('G', rep, [items])."""
    t = fmt.strip()
    if not (t.startswith("(") and t.endswith(")")):
        raise FortranFormatError(f"format must be parenthesised: {fmt!r}")
    pos = 0
    stack = []
    reps = []
    while pos < len(t):
        m = _TOKEN.match(t, pos)
        if m is None or m.end() == pos:
            if t[pos:].strip() == "":
                break
            raise FortranFormatError(f"unsupported format item at {t[pos:]!r} in {fmt!r}")
        pos = m.end()
        if m["close"]:
            items = stack.pop()
            r = reps.pop()
            if not stack:
                if t[pos:].strip():
                    raise FortranFormatError(f"text after the closing parenthesis in {fmt!r}")
                return items
            stack[-1].append(("G", r, items))
        elif m.group(0).strip().endswith("("):
            stack.append([])
            reps.append(int(m["rep"]) if m["rep"] else 1)
        elif not stack:
            raise FortranFormatError(f"item outside the parentheses in {fmt!r}")
        elif m["x"]:
            stack[-1].append(("X", int(m["x"])))
        elif m["p"]:
            stack[-1].append(("P", int(m["p"])))
        elif m["ed"]:
            ed = m["ed"].upper()
            w = int(m["w"]) if m["w"] else None
            d = int(m["d"]) if m["d"] is not None else None
            e = int(m["e"]) if m["e"] else None
            if ed in ("E", "ES", "D", "F") and (w is None or d is None):
                raise FortranFormatError(f"{ed} needs w.d in {fmt!r}")
            if ed == "I" and w is None:
                raise FortranFormatError(f"I needs a width in {fmt!r}")
            if ed == "EN":
                raise FortranFormatError(f"EN editing is not supported ({fmt!r})")
            for _ in range(int(m["rr"]) if m["rr"] else 1):
                stack[-1].append((ed, w, d, e))
    raise FortranFormatError(f"unbalanced parentheses in {fmt!r}")


def _flatten(items):
    out = []
    for it in items:
        if it[0] == "G":
            for _ in range(it[1]):
                out += _flatten(it[2])
        else:
            out.append(it)
    return out


def fortran_write(fmt, *values):
    """The record a formatted WRITE(unit, fmt) values... produces (one record, no newline)."""
    descs = _flatten(_parse(fmt))
    vals = list(values)
    out = []
    scale = 0
    for dsc in descs:
        kind = dsc[0]
        if kind == "X":
            out.append(" " * dsc[1])
            continue
        if kind == "P":
            scale = dsc[1]
            continue
        if not vals:
            break
        v = vals.pop(0)
        _, w, d, e = dsc
        if kind in ("E", "D"):
            if isinstance(v, (str, bool)):
                raise FortranFormatError(f"{kind} edit descriptor for {v!r}")
            out.append(format_E(v, w, d, e, scale, "E" if kind == "E" else "D"))
        elif kind == "ES":
            out.append(format_ES(v, w, d, e))
        elif kind == "F":
            out.append(format_F(v, w, d, scale))
        elif kind == "I":
            if not isinstance(v, int) or isinstance(v, bool):
                raise FortranFormatError(f"I edit descriptor for non-integer {v!r}")
            out.append(format_I(v, w, d))
        elif kind == "A":
            if not isinstance(v, str):
                raise FortranFormatError(f"A edit descriptor for non-string {v!r}")
            out.append(format_A(v, w))
    if vals:
        raise FortranFormatError(f"{len(vals)} values left after format {fmt!r} (format reversion not supported)")
    return "".join(out)
