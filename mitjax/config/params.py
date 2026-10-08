"""One experiment variant's configuration (plan Task 6): the static `ExperimentConfig` (CPP options, SIZE.h, compiled
packages, package switches, integer/logical/string parameters: hashable, closed over by drivers) and the traced
`NamelistParams` pytree (the REAL parameters of the parameter files, numpy float64 leaves), plus

  * `use_flags`: the package switches as packages_boot.F sets them (defaults, AUTODIFF adjustment, data.pkg,
    implied switches);
  * `require_ported(cfg, ported)`: the set-up check that a switched-on package is ported (unported = hard error;
    compiled-but-off packages are fine); `require_supported(exp, supported)`: the same for the run-time selectors of
    code paths (SELECTORS, a table with citations);
  * `stdout_crosscheck(run, lines)`: every parameter set in a parameter file that the oracle's STDOUT parameter
    printout shows, compared with the printed value in the printout's own format, plus the package summary.

    cfg, params, run = load("tutorial_barotropic_gyre", "input")
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mitjax import paths
from mitjax.config import cpp_options, namelists, own_code, packages, size
from mitjax.io import stdout as stdout_io
from mitjax.io.namelist import NULL
from mitjax.params_io import RunParams, fortran_default, params_pytree


class UnportedPackage(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    input_dir: str
    code_dir: str
    upstream_commit: str
    packages: tuple                 # compiled packages, genmake2 order
    cpp: cpp_options.CppOptions
    size: size.Size
    use: tuple                      # ((useX, bool), ...) as packages_boot.F leaves them
    static: tuple                   # (((file, group, key), value), ...) for int/bool/str parameters set in files
    # the experiment directory (absolute; anywhere on disk, docs plan 20261006 Task 2): it holds the code directories
    # (genmake2's -mods) and the input directories; the MITgcm sources come from MJX_UPSTREAM
    exp_dir: str = ""

    def use_flag(self, name):
        for k, v in self.use:
            if k.lower() == name.lower():
                return v
        raise KeyError(f"no package switch {name}")


@params_pytree
@dataclass(frozen=True)
class NamelistParams:
    values: dict                    # "file:group:key" (or "file:group:key(i,...)") -> np.float64

    def __getitem__(self, k):
        return self.values[k]


@dataclass
class Experiment:
    cfg: ExperimentConfig
    params: NamelistParams
    run: namelists.RunNamelists
    farm: cpp_options.Farm
    toolchain: cpp_options.Toolchain
    pset: packages.PackageSet
    notes: list = field(default_factory=list)


# packages_boot.F @63cdc0b
_PB = "model/src/packages_boot.F"
_PB_DEFAULTS = (113, 164)            # `useX =.FALSE.` block (packages_boot.F:111-164; useGAD commented at :112)
_PB_AUTODIFF = {"useAUTODIFF": ("ALLOW_AUTODIFF", 168), "useECCO": ("ALLOW_ECCO", 170),
                "useCTRL": ("ALLOW_CTRL", 173)}   # packages_boot.F:167-175, all under #ifdef ALLOW_AUTODIFF
_USE_TO_PKG = {"useGAD": "generic_advdiff", "useBulkForce": "bulk_force", "useCheapAML": "cheapaml", "useOffLine": "offline",
               "useShelfIce": "shelfice", "useStreamIce": "streamice", "useThSIce": "thsice",
               "useAtm_Phys": "atm_phys", "useAIM": "aim_v23", "useATM2d": "atm2d", "useMYPACKAGE": "mypackage"}


def use_flags(macros, rp, build):
    """The package switches after PACKAGES_BOOT (packages_boot.F @63cdc0b): defaults :113-164, the AUTODIFF
    adjustment :167-175 (cited instead of the default line it overrides), the data.pkg READ :178, implied switches
    :225-233 (ALLOW_CAL: usePROFILES/useOBSFIT/useECCO imply useCAL; ALLOW_CTRL: useGrdchk implies useCTRL) and
    useGAD :236. The HAVE_NETCDF resets :192-223 do not apply (the oracle defines HAVE_NETCDF; a build without it
    raises). `macros`: the macros of packages_boot.F itself (cpp_options.routine_macros), i.e. after its own includes;
    `build`: the experiment's build (cpp_options.FarmBuild), against which every cited line is checked to be live."""
    override = {}
    if "ALLOW_AUTODIFF" in macros:
        for name, (macro, n) in _PB_AUTODIFF.items():
            if name == "useAUTODIFF" or macro in macros:
                override[name] = n
    flags = {}
    for n in range(_PB_DEFAULTS[0], _PB_DEFAULTS[1] + 1):
        line = _src_line(_PB, n)
        m = re.fullmatch(r"\s+(use\w+)\s*=.*", line)
        if not m:
            raise ValueError(f"{_PB}:{n}: expected a `useX =` default, got {line!r}")
        name = m.group(1)
        d = fortran_default(f"{_PB}:{override.get(name, n)}", name, build)
        flags[d.name] = d.value
    if set(override) - set(flags):
        raise ValueError(f"{_PB}: AUTODIFF switches without a default line: {sorted(set(override) - set(flags))}")
    if "HAVE_NETCDF" not in macros:
        raise NotImplementedError("packages_boot.F:192-223 (no HAVE_NETCDF) is not ported")
    for name in list(flags):
        if rp.has("data.pkg", "PACKAGES", name):
            flags[name] = rp.get("data.pkg", "PACKAGES", name)
    if "ALLOW_CAL" in macros:
        if flags["usePROFILES"] or flags["useOBSFIT"] or flags["useECCO"]:
            flags["useCAL"] = True
    if "ALLOW_CTRL" in macros and flags["useGrdchk"]:
        flags["useCTRL"] = True
    temp = rp.get("data", "PARM01", "tempStepping", default=fortran_default("model/src/set_defaults.F:194",
                                                                            "tempStepping", build))
    salt = rp.get("data", "PARM01", "saltStepping", default=fortran_default("model/src/set_defaults.F:198",
                                                                            "saltStepping", build))
    flags["useGAD"] = temp or salt or flags["usePTRACERS"]              # packages_boot.F:236
    return flags


def _src_line(path, n):
    from mitjax.params_io import _source_lines
    return _source_lines(path)[n - 1]


def pkg_of_switch(name):
    """Package directory of a `useX` switch: `useGMRedi` -> gmredi (lower case without `use`), or the table above."""
    return _USE_TO_PKG.get(name, name[3:].lower())


def require_ported(cfg, ported):
    """Set-up check: a package switched on at run time that is not in `ported` is a hard error (PORTING_RULES 1).
    Compiled-but-off packages are fine. `ported` is a set of package directory names."""
    on = [(k, pkg_of_switch(k)) for k, v in cfg.use if v]
    missing = [(k, p) for k, p in on if p not in ported]
    if missing:
        raise UnportedPackage(f"{cfg.experiment}/{cfg.input_dir}: switched on but not ported: "
                              + ", ".join(f"{k}=.TRUE. (pkg/{p})" for k, p in missing))
    return [p for _, p in on]


class UnsupportedOption(RuntimeError):
    pass


_SD = "model/src/set_defaults.F"
# Run-time selectors of code paths (REVIEW_M0 #9): a parameter of `data` whose value chooses which code runs, with
# its default line @63cdc0b and one site of the branch it selects. require_supported() refuses an experiment whose
# value is not declared supported by the ported routines. (taveFreq is not listed: pkg/timeave is gone from master,
# ini_parms.F:190 keeps only a local of that name.) Values are the ones the READ leaves (before ini_parms.F's
# post-processing, e.g. selectCoriMap = -1 or monitorFreq = -1 mean "derived later").
SELECTORS = {
    "vectorInvariantMomentum": (f"{_SD}:193", "pkg/mom_vecinv instead of pkg/mom_fluxform (dynamics.F:515)"),
    "nonHydrostatic": (f"{_SD}:215", "non-hydrostatic dynamics (dynamics.F:181)"),
    "quasiHydrostatic": (f"{_SD}:216", "quasi-hydrostatic terms (calc_phi_hyd.F:181)"),
    "staggerTimeStep": (f"{_SD}:183", "staggered time stepping (forward_step.F:438)"),
    "implicitDiffusion": (f"{_SD}:209", "implicit vertical diffusion (temp_integrate.F:481)"),
    "implicitViscosity": (f"{_SD}:210", "implicit vertical viscosity (dynamics.F:572)"),
    "momImplVertAdv": (f"{_SD}:212", "implicit vertical advection of momentum (dynamics.F:572)"),
    "tempImplVertAdv": (f"{_SD}:213", "implicit vertical advection of theta (temp_integrate.F:264)"),
    "saltImplVertAdv": (f"{_SD}:214", "implicit vertical advection of salt (salt_integrate.F:262)"),
    "useCDscheme": (f"{_SD}:230", "C-D scheme for the Coriolis terms (dynamics.F:615)"),
    "useSmag3D": (f"{_SD}:205", "3-D Smagorinsky viscosity (dynamics.F:394)"),
    "implicitFreeSurface": (f"{_SD}:250", "implicit free surface (ini_parms.F:480, freeSurfFac)"),
    "rigidLid": (f"{_SD}:251", "rigid lid (config_check.F:680)"),
    "exactConserv": (f"{_SD}:254", "exact volume conservation (integr_continuity.F:90)"),
    "nonlinFreeSurf": (f"{_SD}:257", "non-linear free surface (forward_step.F:952)"),
    "select_rStar": (f"{_SD}:260", "r* coordinate (calc_grad_phi_hyd.F:63)"),
    "selectCoriMap": (f"{_SD}:92", "Coriolis map (ini_cori.F:55)"),
    "usingCurvilinearGrid": (f"{_SD}:86", "curvilinear grid (ini_grid.F:134)"),
    "usingCylindricalGrid": (f"{_SD}:90", "cylindrical grid (ini_grid.F:136)"),
    "monitorFreq": (f"{_SD}:351", "the monitor (forward_step.F:1151, CALL MONITOR)"),
}


def selector_values(exp):
    """{selector: (value, source)}: the value from the experiment's `data` (any group), else its cited default
    (fortran_default: the line must be live in this build)."""
    out = {}
    for name, (cite, _) in SELECTORS.items():
        hits = [v for (f, _, k), v in exp.run.vars.items() if f == "data" and k == name.lower() and v.value is not NULL]
        if len(hits) > 1:
            raise ValueError(f"{name} set in several groups of data")
        if hits:
            out[name] = (hits[0].value, f"data:{hits[0].group}")
        else:
            out[name] = (fortran_default(cite, name, exp).value, cite)
    return out


def require_supported(exp, supported):
    """Set-up check beside require_ported (PORTING_RULES §1: an unported option value is a hard error): every
    selector of SELECTORS must have a value the ported routines declare in `supported` ({name: set of values, or a
    predicate}); a selector missing from `supported` supports nothing. Returns selector_values(exp)."""
    vals = selector_values(exp)
    bad = []
    for name, (value, source) in vals.items():
        allowed = supported.get(name)
        ok = allowed is not None and (allowed(value) if callable(allowed) else value in allowed)
        if not ok:
            bad.append(f"{name}={value!r} ({source}; selects {SELECTORS[name][1]})")
    if bad:
        raise UnsupportedOption(f"{exp.cfg.experiment}/{exp.cfg.input_dir}: option values not ported: "
                                + "; ".join(bad))
    return vals


def code_path(cfg):
    """The build's code directory, genmake2's -mods directory: <experiment directory>/<code dir> (the experiment
    may live anywhere; docs plan 20261006 Task 2)."""
    if not cfg.exp_dir:
        raise ValueError(f"{cfg.experiment}/{cfg.code_dir}: the configuration carries no experiment directory")
    return Path(cfg.exp_dir) / cfg.code_dir


def check_input_dir(exp_dir, input_dir):
    """The input directory itself must exist. testreport runs only input directories that exist (the variants are
    `ls -d $inputdir.*`, verification/testreport:1709), while its linkdata skips a missing directory silently
    (testreport:814 `if test -d "../"$ldir ...`; make_rundir.linkdata_plan), so a variant name without its own
    directory would read the base input's files (lane API session 2)."""
    d = Path(exp_dir) / input_dir
    if not d.is_dir():
        have = sorted(p.name for p in Path(exp_dir).glob("input*") if p.is_dir())
        raise FileNotFoundError(f"no input dir {d} (variants here: {have})")


def load(experiment, input_dir, upstream=None, toolchain=None, exp_dir=None):
    """The configuration of `<exp_dir>/<input_dir>` built from `<exp_dir>/<code dir>` and the MITgcm sources of
    `upstream` (default MJX_UPSTREAM). exp_dir defaults to the verification experiment `experiment` of `upstream`;
    given, `experiment` must be its directory name. Every code/*.F must be a ported routine (decision 11,
    mitjax/config/own_code.py)."""
    upstream = Path(paths.UPSTREAM if upstream is None else upstream)
    exp_dir = (upstream / "verification" / experiment) if exp_dir is None else Path(exp_dir)
    exp_dir = exp_dir.resolve()
    if exp_dir.name != experiment:
        raise ValueError(f"{exp_dir}: the experiment directory's name is not {experiment!r}")
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"{exp_dir}: no experiment directory")
    check_input_dir(exp_dir, input_dir)
    tc = toolchain or cpp_options.default_toolchain()
    code = namelists.code_dir_for(input_dir)
    if not (exp_dir / code).is_dir():
        raise FileNotFoundError(f"{exp_dir / code}: no code directory (the build of {input_dir})")
    own_code.check_own_routines(exp_dir, code, upstream)
    mods = [exp_dir / code]
    pset = packages.package_set(upstream, mods, have_netcdf=tc.have_netcdf)
    farm = cpp_options.make_farm(f"{experiment}-{code}", upstream, mods, pset, tc)
    cpp = cpp_options.read_cpp_options(farm, tc)
    sz = size.read_size(farm, tc)
    run = namelists.read_run_namelists(experiment, input_dir, farm, tc, sz.env(), upstream, exp_dir=exp_dir)
    rp = RunParams(run)
    flags = use_flags(cpp_options.routine_macros(farm, tc, "packages_boot.F"), rp, cpp_options.FarmBuild(farm, tc))
    static, floats = [], {}
    for (f, g, k), v in sorted(run.vars.items()):
        if v.value is NULL:
            continue
        if v.kind == "float":
            if v.shape:
                for ix, x in sorted(v.value.items()):
                    floats[f"{f}:{g}:{k}({','.join(map(str, ix))})"] = np.float64(x)
            else:
                floats[f"{f}:{g}:{k}"] = np.float64(v.value)
        else:
            static.append(((f, g, k), tuple(sorted(v.value.items())) if v.shape else v.value))
    cfg = ExperimentConfig(experiment, input_dir, code, farm.manifest["upstream_commit"], pset.packages, cpp, sz,
                           tuple(sorted(flags.items())), tuple(static), str(exp_dir))
    return Experiment(cfg, NamelistParams(floats), run, farm, tc, pset)


# ---------------------------------------------------------------------------------------------------------------
# STDOUT cross-check

def fortran_e15(x):
    """`1PE23.15` (eesupp/src/print.F:663) without the padding: d.ddddddddddddddddE+xx; a three-digit exponent drops
    the E as gfortran does."""
    s = f"{x:.15E}"
    m = re.fullmatch(r"(-?\d\.\d+)E([+-])(\d+)", s)
    mant, sign, exp = m.groups()
    exp = exp.lstrip("0").rjust(2, "0")
    return f"{mant}E{sign}{exp}" if len(exp) <= 2 else f"{mant}{sign}{exp}"


def fortran_text(v, kind):
    if kind == "float":
        return fortran_e15(v)
    if kind == "bool":
        return "T" if v else "F"
    if kind == "int":
        return str(v)
    return "'" + v.rstrip() + "'"


def _same(ours, got, kind):
    """Printed text equality; character values compare as Fortran does (trailing blanks ignored: WRITE_0D_C trims,
    pkg printouts such as gmredi_readparms' print the full CHARACTER*(40) inside the quotes)."""
    if kind == "str" and got is not None and len(got) >= 2 and got[0] == got[-1] == "'":
        return ours[1:-1].rstrip() == got[1:-1].rstrip()
    return ours == got


_SUMMARY = re.compile(r"^ pkg/(\w+)\s+compiled\s+(?:and\s+used|but not used)\s*(?:\(\s*([+&]?\s*[\w ]+?)\s*=\s*([TF])"
                      r"\s*\))?\s*$")


@dataclass
class CrossCheck:
    compared: list = field(default_factory=list)    # (name, ours, printed)
    mismatched: list = field(default_factory=list)  # (name, ours, printed, line)
    not_printed: list = field(default_factory=list)  # parameters set in files that the printout does not show
    packages: list = field(default_factory=list)    # (pkg, switch, ours, printed)

    @property
    def ok(self):
        return not self.mismatched and not [p for p in self.packages if p[2] != p[3]]


def stdout_crosscheck(exp, lines, defaults=()):
    """Compare every parameter set in the experiment's parameter files with the oracle's printout of it (same name,
    case-insensitive; scalars and the set elements of 1-D arrays, in the printout's own format), and the use-flags
    with the PACKAGES_BOOT summary lines. Parameters whose printed name differs, or that the printout does not show,
    are listed in `not_printed` (advisory). `defaults`: CitedDefaults of parameters the files do not set, each
    compared with its printout the same way (a parameter the files set is refused here)."""
    printed = stdout_io.parameter_dict(stdout_io.parameters(lines))
    by_lower = {}
    for k, p in printed.items():
        by_lower.setdefault(k.split(" ")[0].lower(), []).append(p)
    cc = CrossCheck()
    for (f, g, k), v in sorted(exp.run.vars.items()):
        if v.value is NULL or v.value == {}:
            continue
        ps = by_lower.get(k)
        if not ps or len(ps) != 1 or (v.shape and len(v.shape) != 1):
            cc.not_printed.append(f"{f}:{g}:{v.name}")
            continue
        p = ps[0]
        vals = p.values()
        if v.shape:
            for (i,), x in sorted(v.value.items()):
                ours = fortran_text(x, v.kind)
                got = vals[i - 1] if i - 1 < len(vals) else None
                same = _same(ours, got, v.kind)
                (cc.compared if same else cc.mismatched).append(
                    (f"{v.name}({i})", ours, got) + (() if same else (p.line,)))
        else:
            ours = fortran_text(v.value, v.kind)
            got = vals[0] if vals else None
            same = _same(ours, got, v.kind)
            (cc.compared if same else cc.mismatched).append(
                (v.name, ours, got) + (() if same else (p.line,)))
    for d in defaults:
        if any(k.lower() == d.name.lower() and v.value is not NULL for (_, _, k), v in exp.run.vars.items()):
            raise ValueError(f"{d.name} is set in a parameter file; its default ({d.citation}) is not in effect")
        ps = by_lower.get(d.name.lower())
        if not ps or len(ps) != 1:
            raise KeyError(f"{d.name} ({d.citation}) is not in the printout")
        kind = {bool: "bool", int: "int", float: "float", str: "str"}[type(d.value)]
        ours, got = fortran_text(d.value, kind), ps[0].values()[0]
        same = _same(ours, got, kind)
        (cc.compared if same else cc.mismatched).append(
            (f"{d.name} [{d.citation}]", ours, got) + (() if same else (ps[0].line,)))
    flags = dict(exp.cfg.use)
    for ln in lines:
        m = _SUMMARY.match(ln.text)
        if not m or not m.group(2):
            continue
        sw = m.group(2).replace(" ", "")
        if sw in flags:
            cc.packages.append((m.group(1), sw, flags[sw], m.group(3) == "T"))
    return cc
