"""Reader for MITgcm run-time parameter files (`data`, `data.<pkg>`, `data.diagnostics`, `eedata`) as master @63cdc0b
reads them (plan Task 6; from the ECCO port's `io/namelist.py`, rewritten for the literal pre-processing below).

What the model does before the Fortran namelist READ (eesupp/src/open_copy_data_file.F @63cdc0b):
  * each record is read with FMT='(A)' (open_copy_data_file.F:135) into `CHARACTER*(MAX_LEN_PREC) record` (:46;
    MAX_LEN_PREC = 200, eesupp/inc/EEPARAMS.h:27): a longer line would be cut silently, so it is an error here;
  * a record whose column 1 is `commentCharacter` = '#' (EEPARAMS.h:125; open_copy_data_file.F:137) is dropped; a
    `#` in any other column is not a comment and is passed to the namelist READ;
  * NML_CHANGE_SYNTAX (called at open_copy_data_file.F:139; eesupp/src/nml_change_syntax.F:72-77): a record that is
    exactly ' &' (two characters after trailing blanks are removed) becomes the terminator nmlEnd. NML_EXTENDED_F77
    (the index-colon rewriting, `#ifdef NML_EXTENDED_F77` at :79) is not defined by the oracle's DEFINES; a colon
    index is standard Fortran-90 namelist syntax anyway.
Then gfortran's namelist READ (Fortran 2008 10.11.3): `&group` (or `$group`) opens a group; `name = values`,
`name(i) = ...`, `name(i:j,k) = ...` assign; values are separated by commas and/or blanks; `r*c` repeats c r times;
a null value (`=,`, `,,`, `r*`) leaves the element unchanged; `/`, `&end` or `$end` ends the group; `!` starts a
comment; logical values are an optional `.` then T or F then any characters; reals accept E and D exponents.

`read_namelist(path)` / `parse_namelist(text)` return a `Namelist`: the groups in file order, each a list of
`Assignment`s in file order (repeat counts expanded; `NULL` for a null value). Values are typed from their literal
form (str for quoted, bool, int, float); the declared Fortran type of the variable is applied by the caller
(mitjax/config/namelists.py), which also applies the assignments elementwise (`resolve`).

Every assignment is counted twice: by the parser and by an independent scan of `name =` occurrences outside quotes
in the comment-stripped text; a mismatch raises (a group silently lost to an unknown terminator, lesson L-LIT-11).
Anything this reader does not understand raises ValueError with the file line; nothing is skipped silently.
Stdlib only.
"""

import re
from dataclasses import dataclass
from pathlib import Path

MAX_LEN_PREC = 200          # eesupp/inc/EEPARAMS.h:27  PARAMETER ( MAX_LEN_PREC = 200 )
COMMENT_CHARACTER = "#"     # eesupp/inc/EEPARAMS.h:125 PARAMETER ( commentCharacter = '#' )


class _Null:
    """A null value (Fortran 2008 10.11.3.4): the element keeps its previous value."""
    __slots__ = ()

    def __repr__(self):
        return "NULL"

    def __reduce__(self):
        return "NULL"


NULL = _Null()


@dataclass(frozen=True)
class Assignment:
    group: str           # group name, lower case
    key: str             # variable name, lower case
    index: tuple | None  # None (no subscript) or ((lo, hi), ...) per dimension, 1-based inclusive
    values: tuple        # repeat counts expanded; NULL for null values
    line: int            # 1-based line of `name =` in the file
    text: str            # the name with its subscript as written

    @property
    def name(self):
        """`key` plus the subscript as written without blanks, e.g. 'fields(1:5,2)'."""
        i = self.text.find("(")
        return self.key + (re.sub(r"\s+", "", self.text[i:]) if i >= 0 else "")


@dataclass(frozen=True)
class Group:
    name: str            # lower case
    line: int            # line of `&name`
    assignments: tuple


@dataclass(frozen=True)
class Namelist:
    source: str
    groups: tuple

    def group(self, name):
        """The group `name` (case-insensitive); KeyError if absent. A group present twice raises at parse time."""
        for g in self.groups:
            if g.name == name.lower():
                return g
        raise KeyError(f"{self.source}: no namelist group &{name}")

    def has_group(self, name):
        return any(g.name == name.lower() for g in self.groups)

    def assignments(self):
        return [a for g in self.groups for a in g.assignments]


def preprocess(text, source="<text>"):
    """The records the namelist READ sees, as (line number, text) pairs (open_copy_data_file.F:135-140,
    nml_change_syntax.F:72-77). Raises on a record longer than MAX_LEN_PREC."""
    out = []
    for n, raw in enumerate(text.split("\n"), start=1):
        rec = raw.rstrip("\r")
        il = len(rec.rstrip(" "))
        if il > MAX_LEN_PREC:
            raise ValueError(f"{source}:{n}: record of {il} characters > MAX_LEN_PREC={MAX_LEN_PREC} "
                             "(the Fortran would cut it, open_copy_data_file.F:135)")
        if rec[:1] == COMMENT_CHARACTER:
            continue
        if max(il, 1) == 2 and rec[:2] == " &":
            rec = " /"                              # nmlEnd (any terminator ends the group the same way)
        out.append((n, rec))
    return out


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_INT = re.compile(r"[+-]?\d+")
_REAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([EeDd][+-]?\d+)?")
_SEP = " \t,/!"


class RealText(float):
    """A real namelist value: its float64 value, plus the decimal text as written (`.text`), from which the READ of a
    REAL*4 variable rounds once to binary32 (mitjax/config/namelists.py, mitjax/config/fortran.real4)."""

    def __new__(cls, text):
        obj = super().__new__(cls, float(text.replace("D", "E").replace("d", "e")))
        obj.text = text
        return obj

    def __reduce__(self):
        return (RealText, (self.text,))


def _constant(tok, where):
    """A non-null literal constant -> int / RealText (a float) / bool. Quoted strings are handled by the scanner."""
    if _INT.fullmatch(tok):
        return int(tok)
    if _REAL.fullmatch(tok):
        return RealText(tok)
    m = re.fullmatch(r"\.?([TtFf])[^\s,/=()]*", tok)
    if m:
        return m.group(1) in "Tt"
    raise ValueError(f"{where}: cannot read namelist value {tok!r}")


class _Scanner:
    """Character stream over the pre-processed records with line tracking."""

    def __init__(self, records, source):
        self.src = source
        self.text = "\n".join(r for _, r in records)
        self.lines = []
        for n, r in records:
            self.lines += [n] * (len(r) + 1)
        self.pos = 0

    def line(self, pos=None):
        p = self.pos if pos is None else pos
        return self.lines[min(p, len(self.lines) - 1)] if self.lines else 0

    def err(self, what, pos=None):
        return ValueError(f"{self.src}:{self.line(pos)}: {what}")

    def peek(self):
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_blanks(self, newlines=True):
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c in " \t" or (newlines and c == "\n"):
                self.pos += 1
            elif c == "!":                                   # comment to end of record
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            else:
                break

    def quoted(self):
        q = self.text[self.pos]
        i, out = self.pos + 1, []
        while True:
            j = self.text.find(q, i)
            if j < 0:
                raise self.err(f"unterminated string starting with {q}")
            out.append(self.text[i:j])
            if self.text[j + 1:j + 2] == q:              # doubled quote inside the string
                out.append(q)
                i = j + 2
                continue
            self.pos = j + 1
            return "".join(out).replace("\n", "")

    def name_assign(self):
        """If a `name [(subscript)] =` starts here, return (name, subscript text, text as written, end position)."""
        m = _NAME.match(self.text, self.pos)
        if not m:
            return None
        p = m.end()
        while p < len(self.text) and self.text[p] in " \t":
            p += 1
        sub = None
        if self.text[p:p + 1] == "(":
            q = self.text.find(")", p)
            if q < 0:
                return None
            sub = self.text[p + 1:q]
            p = q + 1
            while p < len(self.text) and self.text[p] in " \t":
                p += 1
        if self.text[p:p + 1] != "=":
            return None
        return m.group(0), sub, self.text[self.pos:p].rstrip(), p + 1


def _subscript(sub, sc, pos):
    """'1:5,2' -> ((1, 5), (2, 2)). Only integer subscripts and ranges with both bounds (what MITgcm inputs use)."""
    dims = []
    for part in sub.split(","):
        part = part.strip()
        m = re.fullmatch(r"([+-]?\d+)\s*(?::\s*([+-]?\d+))?", part)
        if not m:
            raise sc.err(f"unsupported subscript ({sub})", pos)
        lo = int(m.group(1))
        dims.append((lo, int(m.group(2)) if m.group(2) is not None else lo))
    return tuple(dims)


def _values(sc):
    """Values after `=` up to the next `name =`, a terminator, or the end. Returns (values, terminated)."""
    vals = []
    need_value = True                     # at the start and after each comma: a comma here is a null value
    while True:
        sc.skip_blanks()
        c = sc.peek()
        if c == "":
            return vals, False
        if c == "/":
            sc.pos += 1
            return vals, True
        if c in "&$":
            m = re.match(r"[&$]end\b", sc.text[sc.pos:], re.I)
            if m:
                sc.pos += m.end()
                return vals, True
            raise sc.err(f"unexpected {sc.text[sc.pos:sc.pos + 12]!r} inside a group")
        if c == ",":
            if need_value:
                vals.append(NULL)
            sc.pos += 1
            need_value = True
            continue
        if sc.name_assign():
            return vals, False
        start = sc.pos
        where = f"{sc.src}:{sc.line(start)}"
        r = 1
        m = re.match(r"(\d+)\*", sc.text[sc.pos:])
        if m:                                             # repeat count r*c, or r* (r null values)
            r = int(m.group(1))
            sc.pos += m.end()
        c = sc.peek()
        if c in "'\"":
            vals.extend([sc.quoted()] * r)
        elif m and (c == "" or c in " \t\n,/!"):
            vals.extend([NULL] * r)
        else:
            j = sc.pos
            while j < len(sc.text) and sc.text[j] not in " \t\n,/!":
                j += 1
            tok = sc.text[sc.pos:j]
            sc.pos = j
            vals.extend([_constant(tok, where)] * r)
        need_value = False
        sc.skip_blanks()
        if sc.peek() == ",":
            sc.pos += 1
            need_value = True


def _count_assignments(records):
    """Independent count of `name =` / `name(...) =` outside quoted strings and `!` comments (the L-LIT-11 check)."""
    n = 0
    for _, rec in records:
        s, out, q = rec, [], None
        for ch in s:
            if q:
                if ch == q:
                    q = None
                out.append(" ")
            elif ch in "'\"":
                q = ch
                out.append(" ")
            elif ch == "!":
                break
            else:
                out.append(ch)
        n += len(re.findall(r"(?:^|[\s,])[A-Za-z]\w*\s*(?:\([^()=]*\))?\s*=", "".join(out)))
    return n


def parse_namelist(text, source="<text>"):
    records = preprocess(text, source)
    sc = _Scanner(records, source)
    groups, seen = [], set()
    while True:
        sc.skip_blanks()
        c = sc.peek()
        if c == "":
            break
        m = re.match(r"[&$]([A-Za-z]\w*)", sc.text[sc.pos:])
        if not m or m.group(1).lower() == "end":
            raise sc.err(f"text outside a namelist group: {sc.text[sc.pos:sc.pos + 30]!r}")
        gname, gline = m.group(1).lower(), sc.line()
        if gname in seen:
            raise sc.err(f"group &{gname} appears twice (the Fortran READ would see only the first)")
        seen.add(gname)
        sc.pos += m.end()
        assigns, terminated = [], False
        while not terminated:
            sc.skip_blanks()
            c = sc.peek()
            if c == "":
                raise sc.err(f"group &{gname} (line {gline}) is never terminated")
            if c == "/":
                sc.pos += 1
                break
            mm = re.match(r"[&$]end\b", sc.text[sc.pos:], re.I)
            if mm:
                sc.pos += mm.end()
                break
            na = sc.name_assign()
            if na is None:
                raise sc.err(f"expected `name =` in group &{gname}: {sc.text[sc.pos:sc.pos + 30]!r}")
            name, sub, written, end = na
            line = sc.line()
            index = _subscript(sub, sc, sc.pos) if sub is not None else None
            sc.pos = end
            vals, terminated = _values(sc)
            if not vals:
                raise sc.err(f"{written} has no value (not even a null)", end)
            assigns.append(Assignment(gname, name.lower(), index, tuple(vals), line, written))
        groups.append(Group(gname, gline, tuple(assigns)))
    nl = Namelist(source, tuple(groups))
    want = _count_assignments(records)
    got = len(nl.assignments())
    if got != want:
        raise ValueError(f"{source}: parsed {got} assignments but the text has {want} `name =` occurrences "
                         "(lost group or terminator?)")
    return nl


def read_namelist(path):
    return parse_namelist(Path(path).read_text(encoding="latin-1"), str(path))


def resolve(assignments, shape=None):
    """Apply assignments to one variable in order, as the Fortran READ does. shape None (or ()) = scalar: returns the
    last non-null value, or NULL if every value was null. shape (n1, ...) = array: returns {(i1, ...): value} for
    the elements assigned (1-based); an unsubscripted assignment fills from the first element, a subscripted one
    its section, both in column-major order; a null leaves an element unchanged; too many values or an index out
    of bounds raises."""
    if not shape:
        val = NULL
        for a in assignments:
            if a.index is not None and any(lo != 1 or hi != 1 for lo, hi in a.index):
                raise ValueError(f"line {a.line}: {a.text} subscripts a scalar")
            if len(a.values) != 1:
                raise ValueError(f"line {a.line}: {len(a.values)} values for scalar {a.text}")
            if a.values[0] is not NULL:
                val = a.values[0]
        return val
    elems = {}
    for a in assignments:
        if a.index is None:
            section = [(1, n) for n in shape]
        else:
            if len(a.index) != len(shape):
                raise ValueError(f"line {a.line}: {a.text} has {len(a.index)} subscripts, the array {len(shape)}")
            for (lo, hi), n in zip(a.index, shape):
                if not (1 <= lo <= hi <= n):
                    raise ValueError(f"line {a.line}: {a.text} outside the declared shape {shape}")
            section = list(a.index)
        order = list(_column_major(section))
        if len(a.values) > len(order):
            raise ValueError(f"line {a.line}: {len(a.values)} values for the {len(order)} elements of {a.text}")
        for ix, v in zip(order, a.values):
            if v is not NULL:
                elems[ix] = v
    return elems


def _column_major(section):
    """Indices of a section [(lo, hi), ...] in Fortran storage order (first index fastest)."""
    if not section:
        yield ()
        return
    (lo, hi), rest = section[0], section[1:]
    for tail in _column_major(rest):
        for i in range(lo, hi + 1):
            yield (i,) + tail
