"""The run-time parameter files of one experiment variant, read as the model reads them (plan Task 6).

Which files: the run directory testreport makes (mitjax/make_rundir.py, plan Task 3a): `linkdata` of the variant's
input directories, `input.<v>` (or `input_ad.<v>`) shadowing `input` (`input_ad`) — the directory list and the
directory listing are lane A's own functions (`input_dirs`, `ls1`), and `link_sources` below applies linkdata's rule
(first directory providing a name wins, directories skipped, a repeated directory skipped) without making links.
The variant's `prepare_run` may only bring in data files; one that names a parameter file is refused.

Which namelist groups and variables: for every parameter file of the run directory, the compiled Fortran routines
that name it in quotes (`'<file name>'` on a non-comment line of a compiled source of the build's flat directory; the
openers are among them, e.g. ini_parms.F:395 `OPEN_COPY_DATA_FILE( 'data', ...)`, packages_boot.F:106-107,
eeset_parms.F:171) are preprocessed with the build's
cpp (mitjax/config/cpp_options.py), and their live NAMELIST statements and the declarations of the variables in them
(from the routine and every header it includes) give each group's variables with type and shape. Then, as the
Fortran READ does:
  * a variable not in its group's NAMELIST list is an error (`UnknownNamelistVariable`);
  * a value of the wrong type is an error (an integer literal is accepted for a REAL variable, as the READ does);
  * a string longer than the declared CHARACTER length is an error (the READ would cut it);
  * assignments apply elementwise in file order (mitjax/io/namelist.resolve): a duplicated key keeps the last value.
A group in a file that no compiled routine declares is not read by the model; it is listed in `unread_groups`, and a
file no compiled routine opens (e.g. data.optim, read by the offline optimizer; data.exch2.mpi and eedata.mth under
MPI=0, single-threaded) is listed in `unread_files`.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from mitjax import make_rundir, paths
from mitjax.config import cpp_options, fortran
from mitjax.io import namelist as nlio
from mitjax.io.namelist import NULL



class UnknownNamelistVariable(KeyError):
    pass


def code_dir_for(input_dir):
    """testreport: input_ad[.<v>] runs with code_ad (testreport:1323-1332), input[.<v>] with code. Lane B (Task 25):
    `input_min` is adjustment.cs-32x32x1's "minimal" case, built from code_min (its README; testreport does not run
    it; lane A's registry reference/reference_runs.M2_MIN pairs the two)."""
    if input_dir == "input_min":
        return "code_min"
    return "code_ad" if input_dir.split(".")[0] == "input_ad" else "code"


def link_sources(exp_dir, input_dir):
    """{file name: input directory} of the run directory: testreport's linkdata (MULTI_THREAD=f, MPI=0) as lane A's
    mitjax/make_rundir.linkdata_plan plans it (the one implementation of the rule)."""
    dirs, _ = make_rundir.input_dirs(input_dir)
    out = {}
    for name, _, ldir in make_rundir.linkdata_plan(exp_dir, dirs):
        if name in out:
            raise ValueError(f"linkdata plan of {exp_dir}/{input_dir} links {name} twice")
        out[name] = ldir
    return out


def is_parameter_file(name):
    return name == "eedata" or name.startswith("eedata.") or name == "data" or name.startswith("data.")


@dataclass(frozen=True)
class Var:
    file: str
    group: str
    name: str           # as declared in Fortran
    kind: str           # float / int / bool / str
    shape: tuple        # () for a scalar
    char_len: int | None
    value: object       # scalar value, or {index tuple: value} for arrays; NULL when only null values were given
    assignments: tuple  # the Assignments, file order


@dataclass(frozen=True)
class Reader:
    routine: str        # file name of the Fortran routine, e.g. ini_parms.F
    source: str         # its path
    open_line: int      # line of the quoted file name in the routine
    groups: dict        # group -> (NAMELIST line in the preprocessed text, [variables])


@dataclass
class RunNamelists:
    experiment: str
    input_dir: str
    files: dict                     # name -> (input dir, path)
    parsed: dict                    # name -> mitjax.io.namelist.Namelist (parameter files)
    readers: dict                   # name -> [Reader]
    vars: dict                      # (file, group, lower name) -> Var
    unread_files: tuple
    unread_groups: tuple            # ((file, group), ...)
    prepare_run: str | None = None
    notes: list = field(default_factory=list)

    def var(self, fname, group, key):
        return self.vars.get((fname, group.lower(), key.lower()))

    def is_set(self, fname, group, key):
        v = self.var(fname, group, key)
        return v is not None and not (v.value is NULL or v.value == {})


_TEXT_CACHE = {}


def _readers_of(farm, fname):
    """Compiled sources that name `fname` in quotes on a non-comment line: [(routine, first such line)]."""
    pat = re.compile(r"""['"]""" + re.escape(fname) + r"""['"]""")
    key = str(farm.path)
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = {name: Path(src).read_text(encoding="latin-1")
                            for name, src in sorted(farm.manifest["links"].items()) if name.endswith(".F")}
    out = []
    for name, text in _TEXT_CACHE[key].items():
        if pat.search(text):
            for n, ln in enumerate(text.split("\n"), start=1):
                if pat.search(ln) and ln[:1] not in "Cc*!" and not ln.lstrip().startswith("#"):
                    out.append((name, n))
                    break
    return out


_DECL_CACHE = {}


def reader_info(farm, toolchain, routine, size_env):
    key = (str(farm.path), routine)
    if key not in _DECL_CACHE:
        text = cpp_options.preprocess(farm, toolchain, routine)
        st = fortran.statements(text, routine)
        env = fortran.parameters(st, size_env, source=routine)
        _DECL_CACHE[key] = (fortran.namelists(st), fortran.declarations(st), env)
    return _DECL_CACHE[key]


# Significant digits up to which the REAL*4 namelist READ was measured to round the decimal text once to binary32
# (probe compiled with the oracle flags, $MJX_RUNS/hardening_scratch/r4probe_*: every string of <= 12 significant
# digits tried; with 17 digits a single-group file gave 1.0 for 1.0000000596046448, which is not the rounding of
# the text): a longer value for a REAL*4 variable is not ported.
REAL4_READ_DIGITS = 12


def _coerce(v, kind, char_len, where, real4=False):
    if v is NULL:
        return v
    if kind == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}: {v!r} for a REAL variable")
        if real4:                                   # a REAL*4 variable stores what the READ rounds to binary32
            text = getattr(v, "text", None)
            if text is None:
                return fortran.real4(v)             # an integer value: exact up to 2**24, rounded beyond
            digits = re.sub(r"[EeDd].*$", "", text).lstrip("+-").replace(".", "").lstrip("0")
            if len(digits) > REAL4_READ_DIGITS:
                raise NotImplementedError(f"{where}: {text!r} for a REAL*4 variable has {len(digits)} significant "
                                          f"digits; the READ's rounding was measured only up to {REAL4_READ_DIGITS}")
            return fortran.real4(text)
        return float(v)
    if kind == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{where}: {v!r} for an INTEGER variable")
        return v
    if kind == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{where}: {v!r} for a LOGICAL variable")
        return v
    if not isinstance(v, str):
        raise ValueError(f"{where}: {v!r} for a CHARACTER variable")
    if char_len is not None and len(v.rstrip()) > char_len:
        raise ValueError(f"{where}: {v!r} longer than CHARACTER*{char_len}")
    return v


def resolve_file(fname, path, nl, rds):
    """Check one parsed parameter file against its reading routines and apply its assignments.
    rds: [(Reader, declarations, PARAMETER env)]. Returns ({(file, group, key): Var}, [(file, group) unread])."""
    vars_, unread_groups = {}, []
    for g in nl.groups:
        owner = [(r, decls, env) for r, decls, env in rds if g.name in r.groups]
        if not owner:
            unread_groups.append((fname, g.name))
            continue
        if len(owner) > 1:
            raise ValueError(f"{fname}: group &{g.name} declared by several routines "
                             f"({[r.routine for r, _, _ in owner]})")
        r, decls, env = owner[0]
        allowed = {v.lower(): v for v in r.groups[g.name][1]}
        by_key = {}
        for a in g.assignments:
            if a.key not in allowed:
                raise UnknownNamelistVariable(
                    f"{path}:{a.line}: {a.text} is not in NAMELIST /{g.name.upper()}/ of "
                    f"{r.routine} (the Fortran READ stops on it)")
            by_key.setdefault(a.key, []).append(a)
        for key, assigns in by_key.items():
            d = decls.get(key)
            if d is None:
                raise ValueError(f"{r.routine}: no declaration for namelist variable {allowed[key]}")
            shp = fortran.shape(d, env, r.routine)
            clen = None
            if d.kind == "str":
                m = re.search(r"\*\s*\(?\s*([^)]*?)\s*\)?$", d.ftype)
                clen = fortran.int_expr(m.group(1), env, r.routine) if m else 1
            where = f"{path}:{assigns[0].line}"
            coerced = [nlio.Assignment(a.group, a.key, a.index,
                                       tuple(_coerce(v, d.kind, clen, f"{path}:{a.line}",
                                                     d.kind == "float" and fortran.real_bytes(d.ftype) == 4)
                                             for v in a.values), a.line, a.text) for a in assigns]
            try:
                val = nlio.resolve(coerced, shp)
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            vars_[(fname, g.name, key)] = Var(fname, g.name, d.name, d.kind, shp, clen, val, tuple(coerced))
    return vars_, unread_groups


def file_readers(farm, toolchain, fname, size_env):
    """[(Reader, declarations, PARAMETER env)] of the compiled routines that name `fname`."""
    rds = []
    for routine, line in _readers_of(farm, fname):
        nml, decls, env = reader_info(farm, toolchain, routine, size_env)
        rds.append((Reader(routine, farm.manifest["links"][routine], line,
                           {g: (n, vs) for g, (n, vs) in nml.items()}), decls, env))
    return rds


def read_run_namelists(experiment, input_dir, farm, toolchain, size_env, upstream=None, exp_dir=None):
    """The parameter files of `<exp_dir>/<input_dir>` (exp_dir default: the verification experiment `experiment` of
    `upstream`, default MJX_UPSTREAM)."""
    if exp_dir is None:
        upstream = Path(upstream) if upstream is not None else paths.UPSTREAM
        exp_dir = upstream / "verification" / experiment
    exp_dir = Path(exp_dir)
    srcs = link_sources(exp_dir, input_dir)
    files = {n: (d, exp_dir / d / n) for n, d in srcs.items()}
    prep = None
    if "prepare_run" in files:
        prep = files["prepare_run"][1].read_text()
        body = "\n".join(ln for ln in prep.splitlines() if not ln.lstrip().startswith("#"))
        if re.search(r"(?<![\w.$])(eedata|data)(\.[\w.]+)?(?![\w.])", body):
            raise NotImplementedError(f"{files['prepare_run'][1]} may write a parameter file; not supported")
    parsed, readers, vars_, unread_files, unread_groups = {}, {}, {}, [], []
    for fname in sorted(n for n in files if is_parameter_file(n)):
        rds = file_readers(farm, toolchain, fname, size_env)
        readers[fname] = [r for r, _, _ in rds]
        if not rds:
            # no compiled routine opens this file: the model never reads it, and it need not be a namelist at all
            # (1D_ocean_ice_column/input/data.err is a table of numbers for pkg/ecco, not compiled in its code/)
            unread_files.append(fname)
            try:
                parsed[fname] = nlio.read_namelist(files[fname][1])
            except ValueError:
                pass
            continue
        nl = nlio.read_namelist(files[fname][1])
        parsed[fname] = nl
        fv, fu = resolve_file(fname, str(files[fname][1]), nl, rds)
        vars_.update(fv)
        unread_groups += fu
    return RunNamelists(experiment, input_dir, {n: (d, str(p)) for n, (d, p) in files.items()}, parsed, readers,
                        vars_, tuple(unread_files), tuple(unread_groups), prep)
