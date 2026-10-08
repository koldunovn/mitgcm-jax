"""The project's single C-preprocessor module (plan Task 6; used by Task 5's tools/cpp_live.py): the CPP options of an
experiment's build, obtained by running the build's own `cpp` the way genmake2's Makefile runs it, never by reading
`#define` lines [E§2].

How the oracle build preprocesses (MJX_UPSTREAM tools/genmake2 @63cdc0b):
  * every source and header of the -mods directories, the compiled packages, eesupp/src, model/src, then the include
    directories (-mods, packages, eesupp/inc, model/inc) is linked flat into the build directory, the first directory
    providing a name winning (genmake2:2946-3006: `alldirs="$SOURCEDIRS $INCLUDEDIRS ."`, files `*.[h,c,F] *.flow
    *.F90`, a name already linked is skipped; SOURCEDIRS = -mods (genmake2:2295), packages in list order
    (genmake2:2706-2713), then STANDARDDIRS src (genmake2:2765-2775); INCLUDEDIRS likewise with inc, genmake2:2776-2785);
  * PACKAGES_CONFIG.h and AD_CONFIG.h are generated in the build directory (mitjax/config/packages.py);
  * each file is preprocessed in that directory by `CPPCMD = cat $< | <cpp> $(DEFINES) $(INCLUDES) | ...`
    (genmake2:3380), so a quoted #include resolves to the flat directory.
This module rebuilds that flat directory as a link farm (symbolic links to the read-only upstream clone, plus the two
generated headers) and runs the build's cpp in it. The toolchain (cpp program, DEFINES, INCLUDES) is machine- and
compiler-specific (DEFINES holds genmake2's compiler-test results such as -DHAVE_NETCDF): default_toolchain() finds the
system's cpp as genmake2 does and uses the declared DEFINES of the verification builds (system_toolchain; overrides
MJX_CPP, MJX_CPP_DEFINES, MJX_CPP_INCLUDES, MJX_HAVE_NETCDF); where the Fortran oracle exists, oracle_toolchain() reads
a frozen oracle build record ($MJX_REFERENCE/bin/<build>/{Makefile, provenance.txt}), and
mitjax/tests/test_toolchain_system.py checks that both give every verification build the same configuration.
Everything experiment-specific (package list, link precedence, generated headers, which headers exist) is derived
here from the experiment directory and checked against the records (mitjax/tests/test_config.py).

The farm is content-addressed: $MJX_CACHE/config_cpp/<label>-<sha>/, created once (atomically, by a rename) and
reused after its MANIFEST.json is checked; nothing is ever deleted or overwritten.

`CppOptions` is the static, hashable result: the macros in effect after each *OPTIONS.h (as `cpp -dM` prints them,
without the compiler's own predefined macros), with `cfg.cpp.flag(NAME, header)` and attribute access
`cfg.cpp.NAME` (True when defined). A name that no compiled source tests in an `#if`/`#ifdef`/`#ifndef`/`#elif` is
not an option of this build and raises UnknownCppOption, as does a retired option (RETIRED).
"""

import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mitjax import paths
from mitjax.config import packages as pkgs

# Options master retired; testing them is an error (cited where master says so).
RETIRED = {
    "EXACT_CONSERV": "retired in master: model/src/config_summary.F:375-377 ('should remove retired EXACT_CONSERV "
                     "from CPP_OPTIONS.h'), model/src/config_check.F:293-301; exactConserv decides at run time",
}

_SRC_GLOBS = ("*.h", "*.c", "*.F", "*.flow", "*.F90")       # genmake2:2950-2951 `*.[h,c,F] *.flow` + `*.F90`
_GENERATED = ("PACKAGES_CONFIG.h", "AD_CONFIG.h", "FC_NAMEMANGLE.h", "BUILD_INFO.h", "EMBEDDED_FILES.h")  # :2963-2977
_IF_TEST = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif)\b(.*)$", re.M)
_IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")


class UnknownCppOption(KeyError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# toolchain

@dataclass(frozen=True)
class Toolchain:
    cpp: tuple          # e.g. ("/usr/bin/cpp", "-traditional", "-P")
    defines: tuple      # Makefile DEFINES
    includes: tuple     # Makefile INCLUDES
    record: str         # where it came from (an oracle build record, or the system detection and its overrides)
    # genmake2's HAVE_NETCDF (genmake2:2242-2244 adds -DHAVE_NETCDF to DEFINES exactly when it is set; without it
    # pkg/mnc, profiles and obsfit leave the package list, genmake2:2531-2560): None = read from `defines`
    have_netcdf: bool = None

    def __post_init__(self):
        has = "-DHAVE_NETCDF" in self.defines
        if self.have_netcdf is None:
            object.__setattr__(self, "have_netcdf", has)
        elif bool(self.have_netcdf) != has:
            raise ValueError(f"toolchain {self.record}: have_netcdf={self.have_netcdf} but DEFINES "
                             f"{'has' if has else 'lacks'} -DHAVE_NETCDF (genmake2:2242-2244 sets both together)")

    def key(self):
        return (self.cpp, self.defines, self.includes)


def _makefile_var(makefile, name):
    """`NAME = value` of a genmake2 Makefile (backslash continuations joined); None if absent."""
    lines = Path(makefile).read_text().splitlines()
    for i, ln in enumerate(lines):
        m = re.match(rf"^{name}\s*=(.*)$", ln)
        if m:
            val = m.group(1)
            while val.endswith("\\") and i + 1 < len(lines):
                i += 1
                val = val[:-1] + " " + lines[i]
            return val.strip()
    return None


def toolchain_from_record(record_dir):
    """The preprocessor of a frozen oracle build: CPPCMD's program and flags (genmake2:3380), with the program
    resolved to the absolute path the build used (provenance.txt `cpp` line, written by reference/build.sh)."""
    d = Path(record_dir)
    cppcmd = _makefile_var(d / "Makefile", "CPPCMD") or ""
    m = re.match(r"cat \$< \|\s*(.*?) \$\(DEFINES\) \$\(INCLUDES\)", cppcmd)
    if not m:
        raise ValueError(f"{d}/Makefile: cannot read the preprocessor from CPPCMD = {cppcmd!r}")
    argv = m.group(1).split()
    prov = (d / "provenance.txt").read_text()
    pm = re.search(r"^cpp\s+(\S+?):", prov, re.M)
    if not pm or Path(pm.group(1)).name != argv[0]:
        raise ValueError(f"{d}/provenance.txt: no absolute path for {argv[0]!r}")
    argv[0] = pm.group(1)
    return Toolchain(tuple(argv), tuple((_makefile_var(d / "Makefile", "DEFINES") or "").split()),
                     tuple((_makefile_var(d / "Makefile", "INCLUDES") or "").split()), str(d))


@functools.lru_cache(maxsize=None)
def oracle_toolchain():
    """The toolchain of the frozen oracle builds in $MJX_REFERENCE/bin (an optional source: only where the Fortran
    oracle exists; the equality gate mitjax/tests/test_toolchain_system.py and the oracle-side tools use it): they must
    all agree (one machine, one optfile); a disagreement or no record raises."""
    ref = getattr(paths, "REFERENCE", None)             # optional (mitjax/paths.py): unset without the oracle
    if ref is None:
        raise FileNotFoundError("MJX_REFERENCE is not set: no frozen oracle build records")
    recs = sorted(p for p in (ref / "bin").glob("*") if (p / "Makefile").is_file())
    if not recs:
        raise FileNotFoundError(f"no frozen oracle build record under {ref / 'bin'}")
    tcs = [toolchain_from_record(r) for r in recs]
    keys = {t.key() for t in tcs}
    if len(keys) != 1:
        raise ValueError(f"the oracle build records disagree on the preprocessor: {sorted(keys)}")
    return tcs[0]


# The system toolchain (docs plan 20261006 Task 4, R2 of docs/PORTABILITY.md): what genmake2 @63cdc0b sets up for a
# serial gfortran build on Linux with the optfile tools/build_options/linux_amd64_gfortran (the oracle builds use its
# Levante copy reference/optfile_levante_gfortran, the same DEFINES, CPP and INCLUDES lines). Declared here, not
# probed: genmake2 finds the DEFINES below by compiling and linking test programs with the Fortran compiler, which a
# user of mitjax need not have. They are the values of the verification builds (every build record in
# $MJX_REFERENCE/bin has exactly this DEFINES line), so the default reproduces those builds' CPP options.
DEFAULT_CPP = "cpp -traditional -P"                 # genmake2:1942-1943 (the optfile sets no CPP)
OPTFILE_DEFINES = ("-DWORDLENGTH=4", "-DNML_TERMINATOR")     # linux_amd64_gfortran:43 `DEFINES='-DWORDLENGTH=4 ...'`
COMPILER_TEST_DEFINES = (                           # genmake2's tests with gfortran, in its order:
    "-DHAVE_SYSTEM",                                # genmake2:2105-2121 (call system)
    "-DHAVE_FDATE",                                 # :2123-2142 (fdate)
    "-DHAVE_ETIME_SBR",                             # :2144-2194 (etime as a subroutine; HAVE_ETIME_FCT :2158 not)
    "-DHAVE_CLOC",                                  # :2196-2208 (C routine cloc)
    "-DHAVE_SETRLSTK",                              # :2211-2218 (setrlstk)
    "-DHAVE_SIGREG",                                # :2221-2228 (sigreg)
    "-DHAVE_STAT",                                  # :2231-2238 (stat via C)
)
NETCDF_DEFINE = "-DHAVE_NETCDF"                     # genmake2:2241-2248 (check_netcdf_libs)
FLUSH_DEFINES = ("-DHAVE_FLUSH",)                   # genmake2:2268-2275 (FLUSH intrinsic); -DHAVE_LAPACK
#                                                     (:2262) only with -lapack, which testreport does not pass
DEFAULT_HAVE_NETCDF = True                          # the verification builds: NetCDF found (records: -DHAVE_NETCDF)
DEFAULT_INCLUDES = ()                               # linux_amd64_gfortran:106-148 puts only NetCDF's include
#   directory here; it serves `#include "netcdf.inc"` of pkg/mnc and pkg/profiles sources, none of which mitjax
#   preprocesses (the gate preprocesses every farm file of every verification build with both toolchains)
_CPP_TEST = "#define A a\n"                         # genmake2:1950, 1954 `echo "#define A a" | $CPP`


def default_defines(have_netcdf=DEFAULT_HAVE_NETCDF):
    """genmake2's DEFINES for the declared defaults, in the order genmake2 appends them (genmake2:2105-2277)."""
    return OPTFILE_DEFINES + COMPILER_TEST_DEFINES + ((NETCDF_DEFINE,) if have_netcdf else ()) + FLUSH_DEFINES


class CppNotFound(RuntimeError):
    pass


def _cpp_works(argv):
    try:
        res = subprocess.run(argv, input=_CPP_TEST, capture_output=True, text=True)
    except OSError:
        return False
    return res.returncode == 0


def find_cpp(command=DEFAULT_CPP):
    """genmake2:1946-1970: the first of `$CPP` and `/lib/$CPP` that preprocesses `#define A a` (in genmake2 a working
    `$CPP` makes the `/lib/` try fail: CPP has become " cpp ..." and `/lib/ cpp` runs a directory), else an error.
    The program is returned as an absolute path (the build records keep it so, reference/build.sh)."""
    words = shlex.split(command)
    if not words:
        raise CppNotFound("empty C preprocessor command")
    for prefix in ("", "/lib/"):
        argv = [prefix + words[0]] + words[1:]
        prog = shutil.which(argv[0])
        if prog and _cpp_works([prog] + argv[1:]):
            return (str(Path(prog).absolute()),) + tuple(argv[1:])
    raise CppNotFound(f'C pre-processor "{command}" failed the test case (genmake2:1957-1965): install a C '
                      "preprocessor (GNU cpp or clang's cpp) or name one with MJX_CPP, e.g. "
                      "MJX_CPP='/usr/bin/cpp -traditional -P'")


def system_toolchain(environ=None):
    """The toolchain of this machine: the C preprocessor found as genmake2 finds it (find_cpp) and the declared
    DEFINES, with the overrides
        MJX_CPP            the preprocessor command (genmake2's CPP), default "cpp -traditional -P"
        MJX_CPP_DEFINES    the whole DEFINES line (shell words), default default_defines(have_netcdf)
        MJX_CPP_INCLUDES   the INCLUDES line, default empty
        MJX_HAVE_NETCDF    1/0 (also true/false, yes/no): genmake2's HAVE_NETCDF, default 1; with MJX_CPP_DEFINES it
                           must agree with -DHAVE_NETCDF in that line."""
    env = os.environ if environ is None else environ
    get = lambda k: (env.get(k) or "").strip()                                     # noqa: E731
    nc = get("MJX_HAVE_NETCDF").lower()
    if nc and nc not in ("1", "0", "true", "false", "yes", "no"):
        raise ValueError(f"MJX_HAVE_NETCDF={nc!r}: expected 1 or 0")
    have_netcdf = DEFAULT_HAVE_NETCDF if not nc else nc in ("1", "true", "yes")
    cpp = find_cpp(get("MJX_CPP") or DEFAULT_CPP)
    defines = tuple(shlex.split(get("MJX_CPP_DEFINES"))) if get("MJX_CPP_DEFINES") else default_defines(have_netcdf)
    includes = tuple(shlex.split(get("MJX_CPP_INCLUDES")))
    over = [k for k in ("MJX_CPP", "MJX_CPP_DEFINES", "MJX_CPP_INCLUDES", "MJX_HAVE_NETCDF") if get(k)]
    record = "system (genmake2 defaults" + (", overrides " + " ".join(over) if over else "") + ")"
    return Toolchain(cpp, defines, includes, record, have_netcdf if (nc or not get("MJX_CPP_DEFINES")) else None)


@functools.lru_cache(maxsize=None)
def default_toolchain():
    """The toolchain mitjax preprocesses with: system_toolchain() of the environment at the first call (cached for the
    process). The oracle build records are not read (oracle_toolchain() for those)."""
    return system_toolchain()


# ---------------------------------------------------------------------------------------------------------------
# the flat build directory (link farm)

def upstream_commit(rootdir):
    return subprocess.run(["git", "-C", str(rootdir), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def tree_generated(rootdir, pset, toolchain):
    """Headers genmake2 generates inside the source tree before linking ({package dir: {name: text}}): with NetCDF,
    `make templates` in pkg/mnc (genmake2:2561-2563) makes MNC_ID_HEADER.h by `./parse_local_info > $@`
    (pkg/mnc/Makefile:73-74), which only prints; it is run here in the read-only clone and its output goes to the
    farm. The template-generated .F files of eesupp/src, pkg/exch2, pkg/regrid and pkg/mnc (genmake2:2375-2398) are
    not made: no header and no parameter reader comes from them (preprocess() of one raises FileNotFoundError)."""
    out = {}
    if "mnc" in pset.packages and toolchain.have_netcdf:
        d = Path(rootdir) / "pkg" / "mnc"
        res = subprocess.run(["sh", "./parse_local_info"], cwd=d, capture_output=True, text=True, check=True)
        out[str(d.resolve())] = {"MNC_ID_HEADER.h": res.stdout}
    return out


def link_plan(rootdir, mods, pset, tree_gen=None):
    """{name: source path, or ("generated", text)} of the flat build directory (genmake2:2946-3006), with the
    build-directory headers (PACKAGES_CONFIG.h, AD_CONFIG.h, ...) excluded."""
    rootdir = Path(rootdir)
    tree_gen = tree_gen or {}
    srcdirs = [Path(m) for m in mods] + [rootdir / "pkg" / p for p in pset.packages] + \
              [rootdir / d / "src" for d in pkgs.STANDARDDIRS]
    incdirs = [Path(m) for m in mods] + [rootdir / "pkg" / p for p in pset.packages] + \
              [rootdir / d / "inc" for d in pkgs.STANDARDDIRS]
    plan = {}
    for d in srcdirs + incdirs:
        gen = tree_gen.get(str(d.resolve()), {})
        names = set(gen)
        for g in _SRC_GLOBS:
            names.update(p.name for p in d.glob(g))
        for name in sorted(names):
            if name in plan or not ((d / name).is_file() or name in gen):
                continue
            if name in _GENERATED:
                raise ValueError(f"{d / name}: a source file named like a generated header")
            plan[name] = ("generated", gen[name]) if name in gen else str((d / name).resolve())
    return plan


@dataclass(frozen=True)
class Farm:
    path: Path
    manifest: dict


@dataclass(frozen=True)
class FarmBuild:
    """A flat build directory and the preprocessor that runs in it (what mitjax/params_io.fortran_default needs)."""
    farm: Farm
    toolchain: Toolchain


def make_farm(label, rootdir, mods, pset, toolchain, root=None):
    """Create (or reuse after checking) the link farm for one experiment build; returns a Farm."""
    rootdir = Path(rootdir)
    plan = link_plan(rootdir, mods, pset, tree_generated(rootdir, pset, toolchain))
    generated = {"PACKAGES_CONFIG.h": pkgs.packages_config_h(rootdir, pset), "AD_CONFIG.h": pkgs.ad_config_h(rootdir)}
    generated.update({n: v[1] for n, v in plan.items() if isinstance(v, tuple)})
    plan = {n: v for n, v in plan.items() if not isinstance(v, tuple)}
    manifest = {"label": label, "rootdir": str(rootdir.resolve()), "upstream_commit": upstream_commit(rootdir),
                "mods": [str(Path(m).resolve()) for m in mods], "packages": list(pset.packages),
                "links": plan, "generated_sha256": {k: hashlib.sha256(v.encode()).hexdigest()
                                                    for k, v in generated.items()}}
    blob = json.dumps(manifest, sort_keys=True).encode()
    key = hashlib.sha256(blob).hexdigest()[:16]
    root = Path(root) if root is not None else paths.CACHE / "config_cpp"
    final = root / f"{label}-{key}"
    if not final.exists():
        root.mkdir(parents=True, exist_ok=True)
        part = root / f"{label}-{key}.part-{os.getpid()}"
        part.mkdir()
        for name, src in plan.items():
            (part / name).symlink_to(src)
        for name, text in generated.items():
            (part / name).write_text(text)
        (part / "MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
        try:
            os.rename(part, final)
        except OSError:
            if not final.exists():
                raise
            # another process made the same farm first: `part` stays (never deleted; same content)
    have = json.loads((final / "MANIFEST.json").read_text())
    if have != manifest:
        raise ValueError(f"{final}: MANIFEST.json differs from the expected content (refusing to use it)")
    for name, text in generated.items():
        if (final / name).read_text() != text:
            raise ValueError(f"{final / name} differs from the generated content")
    return Farm(final, manifest)


# ---------------------------------------------------------------------------------------------------------------
# running cpp

def run_cpp_dM(farm, toolchain, header):
    """`cpp -dM` of one header in the farm, as reference/check_build.py `options` runs it in the build directory:
    <cpp> -dM DEFINES INCLUDES <header>, cwd = the flat directory; returns the sorted lines (the record's format)."""
    res = subprocess.run(list(toolchain.cpp) + ["-dM"] + list(toolchain.defines) + list(toolchain.includes) + [header],
                         cwd=farm.path, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"cpp -dM {header} in {farm.path} failed: {res.stderr.strip()}")
    return "".join(sorted(res.stdout.splitlines(keepends=True)))


def preprocess_text(farm, toolchain, text, line_markers=False):
    """The text the compiler sees for `text` given on stdin in the flat directory: `cat <file> | <cpp> DEFINES
    INCLUDES` (genmake2:3380; the set64bitConst.sh post-filter only rewrites `_d` and is not applied).
    line_markers=True drops `-P` so the output carries `# <line> "<file>"` markers (`<stdin>` for the text itself)."""
    argv = [a for a in toolchain.cpp if not (line_markers and a == "-P")]
    res = subprocess.run(argv + list(toolchain.defines) + list(toolchain.includes), input=text.encode("latin-1"),
                         cwd=farm.path, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"cpp of stdin in {farm.path} failed: {res.stderr.decode(errors='replace').strip()}")
    return res.stdout.decode("latin-1")


def preprocess(farm, toolchain, name, line_markers=False):
    """preprocess_text of the farm file `name` (the file's bytes on stdin, as the Makefile's `cat $< | cpp`)."""
    return preprocess_text(farm, toolchain, (farm.path / name).read_text(encoding="latin-1"), line_markers)


# ---------------------------------------------------------------------------------------------------------------
# live lines: which conditional arms of a file the build's cpp takes (used by tools/cpp_live.py and
# mitjax/params_io.fortran_default)
#
# Liveness of each `#if/#ifdef/#ifndef/#elif/#else` arm is decided by cpp itself: a marker line `MJXARM_<n>` is put
# after the directive at line n, the text is preprocessed the same way, and the arm is taken iff its marker survives.
# A source line is live iff every arm enclosing it is taken; conditional directive lines themselves are not code.
# Traditional cpp recognises a directive only with `#` in column 1 (`# ifdef` counts, `  #ifdef` does not: measured
# with the build's cpp), and so does this parser.

_LINE_MARKER = re.compile(r'^# (\d+) "((?:[^"\\]|\\.)*)"((?: \d+)*)$')      # GNU cpp line marker
_COND = re.compile(r"^#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")      # '#' in column 1 only (traditional cpp)
_ARM_TAG = re.compile(r"\bMJXARM_(\d+)\b")


def output_lines(text):
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


@dataclass(frozen=True)
class Arm:
    line: int           # line of the directive that opens the arm (#if/#ifdef/#ifndef/#elif/#else)
    last: int           # last line of the directive (backslash continuations)
    end: int            # line of the directive that closes the arm (next #elif/#else, or #endif)
    endif: int          # line of the group's #endif
    group: int          # line of the group's opening #if
    parent: int | None  # line of the enclosing arm (None: top level)
    depth: int
    kind: str
    text: str
    taken: bool | None = None


def _replace(arm, **kw):
    d = dict(arm.__dict__)
    d.update(kw)
    return Arm(**d)


def parse_arms(src_lines):
    """Conditional arms of a source (list of lines, 1-based line numbers); raises on unbalanced directives."""
    arms, stack = [], []            # stack entries: [group line, [arm indices], parent arm line]
    i, n = 0, len(src_lines)
    while i < n:
        line_no, last = i + 1, i
        m = _COND.match(src_lines[i])
        if m:
            while src_lines[last].endswith("\\") and last + 1 < n:      # directive continued by backslash
                last += 1
            kind = m.group(1)
            if kind in ("if", "ifdef", "ifndef"):
                stack.append([line_no, [], arms[stack[-1][1][-1]].line if stack else None])
            elif not stack:
                raise ValueError(f"line {line_no}: #{kind} without #if")
            else:
                k = stack[-1][1][-1]
                arms[k] = _replace(arms[k], end=line_no)
            if kind == "endif":
                _, ks, _ = stack.pop()
                for k in ks:
                    arms[k] = _replace(arms[k], endif=line_no)
            else:
                g, ks, parent = stack[-1]
                arms.append(Arm(line_no, last + 1, 0, 0, g, parent, len(stack) - 1, kind,
                                "\n".join(src_lines[i:last + 1])))
                ks.append(len(arms) - 1)
        i = last + 1
    if stack:
        raise ValueError(f"#if at line {stack[-1][0]} never closed")
    return arms


def arm_marked_text(src_lines, arms):
    """The source with a line `MJXARM_<n>` after the directive (and its continuations) opening each arm."""
    after = {a.last: a.line for a in arms}
    out = []
    for i, ln in enumerate(src_lines, start=1):
        out.append(ln)
        if i in after:
            out.append(f"MJXARM_{after[i]}")
    return "\n".join(out) + "\n"


def read_source(farm, name):
    p = Path(farm.path) / name
    if not p.is_file():
        raise FileNotFoundError(f"{name} is not a file of the build directory {farm.path}")
    return output_lines(p.read_text(encoding="latin-1"))


def evaluate_arms(farm, toolchain, name, src_lines=None):
    """Arms of farm file `name` with `taken` decided by cpp (see above); consistency-checked."""
    if src_lines is None:
        src_lines = read_source(farm, name)
    arms = parse_arms(src_lines)
    out = preprocess_text(farm, toolchain, arm_marked_text(src_lines, arms), line_markers=True)
    seen = set()
    for raw in output_lines(out):
        if not _LINE_MARKER.match(raw):
            seen.update(int(x) for x in _ARM_TAG.findall(raw))
    arms = [_replace(a, taken=a.line in seen) for a in arms]
    by_line = {a.line: a for a in arms}
    groups = {}
    for a in arms:
        groups.setdefault(a.group, []).append(a)
        if a.taken and a.parent is not None and not by_line[a.parent].taken:
            raise ValueError(f"{name}:{a.line}: arm taken inside the arm at line {a.parent}, which is not taken")
    for g, ga in groups.items():
        if sum(a.taken for a in ga) > 1:
            raise ValueError(f"{name}:{g}: more than one arm of the group taken: {[a.line for a in ga if a.taken]}")
    return arms


def live_mask(src_lines, arms):
    """[bool] per source line: inside taken arms only, and not a conditional directive line."""
    n = len(src_lines)
    live = [True] * (n + 1)                             # index = 1-based line
    for a in arms:
        for k in range(a.line, a.last + 1):             # the opening directive (and its continuations)
            live[k] = False
        if not a.taken:
            for k in range(a.last + 1, a.end):          # the arm's body, nested arms included
                live[k] = False
        live[a.endif] = False
    return live[1:]


def live_lines(farm, toolchain, name):
    """(source lines, [bool] live per line, arms) of farm file `name` under this build's CPP options."""
    src = read_source(farm, name)
    arms = evaluate_arms(farm, toolchain, name, src)
    return src, live_mask(src, arms), arms


def enclosing_arms(arms, line):
    """The conditional arms that contain source line `line` (outermost first)."""
    return sorted((a for a in arms if a.last < line < a.end), key=lambda a: a.depth)


MODEL_VIEW = "PACKAGES_CONFIG.h+CPP_OPTIONS.h"
# The prologue of the model/src routines (e.g. model/src/ini_parms.F:1-2, packages_boot.F:1-2): what `cfg.cpp.NAME`
# answers. An experiment's CPP_OPTIONS.h need not include PACKAGES_CONFIG.h itself (tutorial_global_oce_optim's
# code_ad/CPP_OPTIONS.h does not), so CPP_OPTIONS.h alone does not show ALLOW_<PKG>.
_MODEL_PROLOGUE = '#include "PACKAGES_CONFIG.h"\n#include "CPP_OPTIONS.h"\n'


def run_cpp_dM_text(farm, toolchain, text):
    """`cpp -dM` of a text given on stdin in the farm (quoted includes resolve to the flat directory)."""
    res = subprocess.run(list(toolchain.cpp) + ["-dM"] + list(toolchain.defines) + list(toolchain.includes),
                         input=text, cwd=farm.path, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cpp -dM of stdin in {farm.path} failed: {res.stderr.strip()}")
    return "".join(sorted(res.stdout.splitlines(keepends=True)))


def routine_macros(farm, toolchain, routine):
    """Macros in effect at the end of a routine as the build preprocesses it (`cat routine | cpp -dM ...`)."""
    return _dM_dict(run_cpp_dM_text(farm, toolchain, (farm.path / routine).read_text(encoding="latin-1")))


def _dM_dict(text):
    out = {}
    for ln in text.splitlines():
        m = re.match(r"#define (\w+)(\([^)]*\))?(?: (.*))?$", ln)
        if not m:
            raise ValueError(f"unexpected cpp -dM line {ln!r}")
        out[m.group(1)] = (m.group(2) or "") + ((" " if m.group(2) else "") + m.group(3) if m.group(3) else "")
    return out


def predefined(toolchain):
    """Macros the compiler defines by itself (no DEFINES): `cpp -dM` of an empty input."""
    res = subprocess.run(list(toolchain.cpp) + ["-dM", "-"], input="", capture_output=True, text=True, check=True)
    return _dM_dict("".join(sorted(res.stdout.splitlines(keepends=True))))


def tested_names(farm):
    """Every identifier tested by an #if/#ifdef/#ifndef/#elif line in a file of the farm (the build's options)."""
    names = set()
    for name in list(farm.manifest["links"]) + list(farm.manifest["generated_sha256"]):
        text = (farm.path / name).read_text(encoding="latin-1")
        for m in _IF_TEST.finditer(text):
            names.update(x for x in _IDENT.findall(m.group(1)) if x != "defined")
    return names


# ---------------------------------------------------------------------------------------------------------------
# the static result

@dataclass(frozen=True)
class CppOptions:
    headers: tuple          # ((header, ((name, value), ...)), ...): macros after each *OPTIONS.h, predefined removed
    known: frozenset        # names tested by the compiled sources
    conflicts: tuple        # names whose definition in CPP_OPTIONS.h a header that includes it changes

    def macros(self, header):
        for h, items in self.headers:
            if h == header:
                return dict(items)
        raise KeyError(f"no header {header!r} in this build ({[h for h, _ in self.headers]})")

    def _check(self, name):
        if name in RETIRED:
            raise UnknownCppOption(f"{name}: {RETIRED[name]}")
        if name not in self.known:
            raise UnknownCppOption(f"{name} is not tested by any source of this build: not a CPP option here")

    def flag(self, name, header=MODEL_VIEW):
        """#ifdef NAME as seen by a routine that includes `header` (default: the model/src prologue)."""
        self._check(name)
        return name in self.macros(header)

    def value(self, name, header=MODEL_VIEW):
        self._check(name)
        return self.macros(header)[name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._check(name)
        if name in self.conflicts:
            raise UnknownCppOption(f"{name}: headers disagree; use flag({name!r}, header)")
        return any(name in dict(items) for _, items in self.headers)


def option_headers(farm):
    """The *OPTIONS.h of the build, as reference/check_build.py `options` lists them (glob in the build dir)."""
    return sorted(n for n in list(farm.manifest["links"]) + list(farm.manifest["generated_sha256"])
                  if n.endswith("OPTIONS.h"))


def read_cpp_options(farm, toolchain):
    pre = predefined(toolchain)
    heads = []
    for h in option_headers(farm):
        d = _dM_dict(run_cpp_dM(farm, toolchain, h))
        heads.append((h, tuple(sorted((k, v) for k, v in d.items() if not (k in pre and pre[k] == v)))))
    d = _dM_dict(run_cpp_dM_text(farm, toolchain, _MODEL_PROLOGUE))
    heads.append((MODEL_VIEW, tuple(sorted((k, v) for k, v in d.items() if not (k in pre and pre[k] == v)))))
    base = dict(dict(heads).get("CPP_OPTIONS.h", ()))
    conflicts = set()
    for h, items in heads:
        d = dict(items)
        if h != "CPP_OPTIONS.h" and "CPP_OPTIONS_H" in d:
            conflicts.update(k for k, v in base.items() if d.get(k) != v)
    return CppOptions(tuple(heads), frozenset(tested_names(farm)), tuple(sorted(conflicts)))
