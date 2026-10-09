"""Parameters with no Python-side defaults (plan Task 6; from the ECCO port's `params_io.py`).

Every parameter comes from the run's parameter files (mitjax/config/namelists.py), else from the Fortran default,
which the caller names by its citation: the value is read from that line of the upstream source @63cdc0b, never typed
in Python [E§3, L-LIT-8]:

    viscAh = rp.get("data", "PARM01", "viscAh",
                    default=fortran_default("model/src/set_defaults.F:111", "viscAh", exp))

`fortran_default` accepts a citation only if the cited line is what THIS experiment's build executes (REVIEW_M0 #3):
  * the build compiles the cited file (its flat build directory links that name to the cited upstream file, not to an
    experiment's code/ override);
  * the line survives the build's preprocessing (the arms of every enclosing #if/#ifdef/#else are taken, decided by
    the build's own cpp: mitjax/config/cpp_options.live_lines, the machinery behind tools/cpp_live.py);
  * the line is not inside a run-time `IF ... THEN` block of its routine, unless the caller states that block's
    condition (`condition=`, the text the error prints) after checking it holds for the experiment;
  * no later live statement of the same routine assigns the name again before its namelist READ (the default is the
    value the READ starts from; code after the READ is ported as code);
  * the line reads `name = <literal constant>` (a default that needs code, an expression or another variable, is
    ported as code, not as a default), and the name is declared in the routine (after its includes) with a type the
    literal fits.
The value is the one the Fortran stores (REVIEW_M0 #4): a real literal without a `D` exponent or `_d` is a REAL*4
constant (no -fdefault-real-8 in the oracle build), i.e. its binary32 value (mitjax/config/fortran.real4); a REAL*4
variable stores binary32 whatever the literal. Without a default, a parameter that the files do not set raises
KeyError.

`params_pytree` registers a frozen dataclass as a pytree whose `float` fields are leaves (traced when the dataclass is
a jit argument) and whose other fields are static, so closed-over floats never get folded by XLA [L-XLA-4].
"""

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mitjax import paths
from mitjax.config import cpp_options, fortran
from mitjax.io.namelist import NULL

PINNED = "63cdc0b9602b46bda69df37c5eb9e17396116f67"


@lru_cache(maxsize=None)
def _check_upstream():
    """MJX_UPSTREAM is a git checkout (its own top level) at PINNED. The errors say how to get one: a user with an
    MITgcm clone at another commit adds a worktree of PINNED next to it (a user's report, 2026-10-09)."""
    up, pin = paths.UPSTREAM, PINNED[:7]
    why = f"MITgcm {pin} (the defaults and namelists it reads from the Fortran are cited by line at that commit)"
    try:
        r = subprocess.run(["git", "-C", str(up), "rev-parse", "--show-toplevel", "HEAD"], capture_output=True,
                           text=True)
        out, err = r.stdout.splitlines(), r.stderr.strip().splitlines()
        problem = None if r.returncode == 0 else (err[0] if err else f"git exit {r.returncode}")
    except FileNotFoundError:
        out, problem = [], "git is not installed"
    if problem is None and (len(out) != 2 or Path(out[0]).resolve() != Path(up).resolve()):
        problem = f"its git top level is {out[0] if out else '?'}"
    if problem is not None:
        raise RuntimeError(f"MJX_UPSTREAM={up} is not a git checkout of MITgcm ({problem}), but mitjax is a port of "
                           f"{why} and checks the commit. Clone MITgcm at {pin}:\n"
                           f"    git clone https://github.com/MITgcm/MITgcm && git -C MITgcm checkout {pin}\n"
                           f"    export MJX_UPSTREAM=$PWD/MITgcm")
    head = out[1]
    if head != PINNED:
        raise RuntimeError(f"MJX_UPSTREAM={up} is at {head[:7]}, but mitjax is a port of {why}. Keep your clone and "
                           f"add a checkout of {pin} next to it:\n"
                           f"    git -C {up} worktree add {up}-{pin} {pin}\n"
                           f"    export MJX_UPSTREAM={up}-{pin}")
    return True


@lru_cache(maxsize=None)
def _source_lines(relpath):
    _check_upstream()
    return (paths.UPSTREAM / relpath).read_text(encoding="latin-1").split("\n")


# `_d` is not a CPP macro: the Makefile pipes every preprocessed file through tools/set64bitConst.sh (CPPCMD,
# genmake2:3380), whose live line is `sed s'/ * _d  */'D'/g'` (set64bitConst.sh:7-8): blanks, ` _d`, a blank, blanks
# become `D`, so `1. _d 0` is the double 1.D0.
_SET64 = re.compile(r" * _d  *")


def fortran_literal(text, where="?"):
    """A Fortran literal constant as written in MITgcm source: `.TRUE.`/`.FALSE.`, integers, reals, quoted strings.
    A real with a `D` exponent or the `_d` marker is REAL*8 (float); a real without (`0.1`, `1.e-6`, `2.`) is a
    default-kind REAL*4 constant: its binary32 value (fortran.real4; e.g. `1.0e-12` -> 9.999999960041972e-13)."""
    t = text.strip()
    if re.fullmatch(r"\.(true|false)\.", t, re.I):
        return t.lower() == ".true."
    if re.fullmatch(r"'[^']*'", t) or re.fullmatch(r'"[^"]*"', t):
        return t[1:-1]
    t2 = _SET64.sub("D", t)
    if re.fullmatch(r"[+-]?\d+", t2):
        return int(t2)
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)[Dd][+-]?\d+", t2):
        return float(t2.replace("D", "E").replace("d", "e"))
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([Ee][+-]?\d+)?", t2):
        return fortran.real4(t2)
    raise ValueError(f"{where}: {text!r} is not a literal constant (port the code that sets it)")


@dataclass(frozen=True)
class CitedDefault:
    name: str
    value: object
    citation: str          # "path:line" at the pinned commit
    condition: str = ""    # run-time IF block(s) around the line, as stated by the caller ("" = none)


class DeadCitation(ValueError):
    """The cited line is not what this experiment's build executes."""


_ROUTINE = re.compile(r"^[ \t]+(?:[A-Za-z_*0-9()]+[ \t]+)*?(SUBROUTINE|FUNCTION)[ \t]+(\w+)", re.I)
_END = re.compile(r"^\s*END\s*$", re.I)
_IF_THEN = re.compile(r"^\s*IF\s*\(", re.I)
_ELSE_IF = re.compile(r"^\s*ELSE\s*IF\s*\(", re.I)
_ELSE = re.compile(r"^\s*ELSE\s*$", re.I)
_END_IF = re.compile(r"^\s*END\s*IF\s*$", re.I)


def _balanced_tail(stmt, start):
    """Text after the parenthesis group opening at stmt[start] ('(')."""
    depth, q = 0, None
    for k in range(start, len(stmt)):
        ch = stmt[k]
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return stmt[k + 1:]
    raise ValueError(f"unbalanced parentheses in {stmt!r}")


def _norm(s):
    return re.sub(r"\s+", "", s).upper()


def routine_context(src_lines, live, n):
    """(routine first line, routine END line, enclosing run-time IF blocks, live statements) for source line n.
    IF blocks: [[arm-opening statement text, ...]] outermost first (for an ELSE / ELSE IF arm the IF and every
    earlier arm of the block are listed). Statements are read from the live lines only (dead CPP arms and directive
    lines blanked; fixed-form continuations joined: mitjax/config/fortran.statements)."""
    start = None
    for k in range(n, 0, -1):
        if live[k - 1] and src_lines[k - 1][:1] not in "Cc*!#" and _ROUTINE.match(src_lines[k - 1]):
            start = k
            break
    if start is None:
        raise ValueError(f"line {n}: no SUBROUTINE/FUNCTION statement above it")
    text = "\n".join(ln if live[k] and not ln.startswith("#") else "" for k, ln in enumerate(src_lines))
    stmts = [(ln, s) for ln, s in fortran.statements(text) if ln >= start]
    end = next((ln for ln, s in stmts if ln > n and _END.match(s)), len(src_lines))
    blocks = []
    for ln, s in stmts:
        if ln >= n:
            break
        if _IF_THEN.match(s) and re.fullmatch(r"\s*THEN\s*", _balanced_tail(s, s.index("("))):
            blocks.append([s.strip()])
        elif _ELSE_IF.match(s) or _ELSE.match(s):
            if not blocks:
                raise ValueError(f"line {ln}: {s.strip()!r} outside an IF block")
            blocks[-1].append(s.strip())
        elif _END_IF.match(s):
            if not blocks:
                raise ValueError(f"line {ln}: {s.strip()!r} outside an IF block")
            blocks.pop()
    return start, end, blocks, [(ln, s) for ln, s in stmts if ln <= end]


def later_assignments(stmts, n, end, name):
    """Live statements after line n that assign `name` (`name = ...` or a logical `IF (...) name = ...`) before the
    routine's next READ statement (the namelist READ; what follows it is post-processing, ported as code) or END."""
    out, assign = [], re.compile(rf"\s*{re.escape(name)}\s*=(?!=)", re.I)
    for ln, s in stmts:
        if ln <= n or ln > end:
            continue
        if re.match(r"\s*READ\s*\(", s, re.I):
            break
        if assign.match(s) or (_IF_THEN.match(s) and assign.match(_balanced_tail(s, s.index("(")))):
            out.append(ln)
    return out


def condition_text(blocks):
    """The condition a caller states for a line inside run-time IF blocks: the arm chains joined by ' / '."""
    return " / ".join(" ... ".join(arms) for arms in blocks)


@lru_cache(maxsize=None)
def _live(farm_path, toolchain, name):
    farm = cpp_options.Farm(Path(farm_path), {})
    src, live, arms = cpp_options.live_lines(farm, toolchain, name)
    return src, live, arms


@lru_cache(maxsize=None)
def _declarations(farm_path, toolchain, name):
    farm = cpp_options.Farm(Path(farm_path), {})
    return fortran.declarations(fortran.statements(cpp_options.preprocess(farm, toolchain, name), name))


def fortran_default(citation, name, build, condition=None):
    """The default `name` gets at `citation` ("model/src/set_defaults.F:111") in the experiment's build `build` (any
    object with `.farm` (cpp_options.Farm) and `.toolchain`, e.g. mitjax.config.params.Experiment or
    tools/cpp_live.Build): the checks of the module docstring, then the value the Fortran stores."""
    path, _, line = citation.rpartition(":")
    lines = _source_lines(path)
    n = int(line)
    if not 1 <= n <= len(lines):
        raise ValueError(f"{citation}: no such line")
    src = lines[n - 1]
    if src[:1] in "Cc*!":
        raise ValueError(f"{citation}: a comment line: {src.strip()!r}")
    m = re.fullmatch(r"\s+(\w+)\s*=\s*(.*?)\s*", src)
    if not m or m.group(1).lower() != name.lower():
        raise ValueError(f"{citation}: {src.strip()!r} does not assign {name}")
    nxt = lines[n] if n < len(lines) else ""
    if len(nxt) > 5 and nxt[:1] not in "Cc*!#" and nxt[:5].strip() == "" and nxt[5] not in " 0":
        raise ValueError(f"{citation}: the statement continues on the next line (port it as code)")
    # 1. the build compiles this very file
    farm, tc = build.farm, build.toolchain
    fname = Path(path).name
    target = farm.manifest.get("links", {}).get(fname)
    want = str((paths.UPSTREAM / path).resolve())
    if target != want:
        raise DeadCitation(f"{citation}: this build does not compile {path} ({fname} in its build directory is "
                           f"{target or 'absent'}): cite the file the build uses")
    # 2. the line survives the build's preprocessing
    src_lines, live, arms = _live(str(farm.path), tc, fname)
    if src_lines[n - 1] != src:
        raise RuntimeError(f"{citation}: the build directory's {fname} differs from {paths.UPSTREAM / path}")
    if not live[n - 1]:
        why = "; ".join(f"{a.text.splitlines()[0].strip()} at :{a.line} {'taken' if a.taken else 'NOT taken'}"
                        for a in cpp_options.enclosing_arms(arms, n))
        raise DeadCitation(f"{citation}: {src.strip()!r} is not live in this build ({why}): cite the live line")
    # 3. run-time IF blocks, 4. a later assignment in the same routine
    start, end, blocks, stmts = routine_context(src_lines, live, n)
    if blocks:
        need = condition_text(blocks)
        if condition is None or _norm(condition) != _norm(need):
            raise DeadCitation(f"{citation}: inside a run-time IF block of the routine at :{start}; check that it "
                               f"holds for this experiment and state it: condition={need!r}"
                               + ("" if condition is None else f" (stated: {condition!r})"))
    elif condition:
        raise ValueError(f"{citation}: condition {condition!r} stated, but the line is in no run-time IF block")
    later = later_assignments(stmts, n, end, name)
    if later:
        raise DeadCitation(f"{citation}: {name} is assigned again later in the same routine, before its namelist "
                           f"READ (live lines {later}): the default is not this line's value")
    # 5. the value the Fortran stores
    value = fortran_literal(m.group(2), citation)
    decl = _declarations(str(farm.path), tc, fname).get(name.lower())
    if decl is None:
        raise ValueError(f"{citation}: no declaration of {name} in {fname} (after its includes)")
    if decl.dims:
        raise ValueError(f"{citation}: {name} is an array ({decl.ftype} {name}{decl.dims}): not a scalar default")
    if decl.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{citation}: {value!r} for REAL {name}")
        value = fortran.real4(value) if fortran.real_bytes(decl.ftype) == 4 else float(value)
    elif decl.kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{citation}: {value!r} for INTEGER {name}")
    elif decl.kind == "bool" and not isinstance(value, bool):
        raise ValueError(f"{citation}: {value!r} for LOGICAL {name}")
    elif decl.kind == "str" and not isinstance(value, str):
        raise ValueError(f"{citation}: {value!r} for CHARACTER {name}")
    return CitedDefault(name, value, citation, condition or "")


class RunParams:
    """Getter over mitjax.config.namelists.RunNamelists with cited defaults only."""

    def __init__(self, run_namelists):
        self.nml = run_namelists

    def has(self, fname, group, key):
        return self.nml.is_set(fname, group, key)

    def get(self, fname, group, key, default=None):
        """A scalar parameter: the file's value, else the cited default; KeyError without one."""
        v = self.nml.var(fname, group, key)
        if v is not None and v.shape:
            raise TypeError(f"{fname}:{group}:{key} is an array of shape {v.shape}; use get_array")
        if v is not None and v.value is not NULL:
            return v.value
        if default is None:
            raise KeyError(f"{fname}:{group}:{key} not set and no Fortran default given")
        if not isinstance(default, CitedDefault):
            raise TypeError(f"{fname}:{group}:{key}: a default must be a CitedDefault (fortran_default), "
                            f"not {default!r}")
        if default.name.lower() != key.lower():
            raise ValueError(f"default {default.citation} sets {default.name}, not {key}")
        return default.value

    def get_array(self, fname, group, key, default=None):
        """{index: value} of the elements the file sets, every other element from the cited default (as a
        (default, citation) pair the caller expands to the declared shape); KeyError if neither covers it."""
        v = self.nml.var(fname, group, key)
        elems = dict(v.value) if v is not None else {}
        if default is None:
            return elems, None
        if not isinstance(default, CitedDefault):
            raise TypeError(f"{fname}:{group}:{key}: a default must be a CitedDefault")
        return elems, default


def params_pytree(cls):
    """Register a frozen parameter dataclass as a pytree: fields annotated `float` (and `dict` of floats, for
    the namelist float table) become leaves (traced when the dataclass is passed as a jit argument), every other
    field (int, bool, str, tuple, ...) is static metadata.

    Why: XLA's algebraic simplifier folds `(x*c1)*c2` into `x*(c1*c2)` when c1, c2 are compile-time constants
    (closed-over Python floats), changing the rounding by up to 1 ulp versus the Fortran order (ECCO port: 42% of
    1e5 values differed; with c1, c2 traced, 0). Leaves should be strongly typed (numpy float64, not Python float:
    a weak type recompiles, L-ARCH-7)."""
    import dataclasses

    import jax

    fields = dataclasses.fields(cls)
    data = [f.name for f in fields if f.type in (float, "float", dict, "dict")]
    meta = [f.name for f in fields if f.name not in data]
    return jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
