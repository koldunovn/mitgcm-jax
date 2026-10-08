"""An experiment's own Fortran routines (docs plan 20261006, decision 11): every `<code dir>/*.F` must be ported.

genmake2 links the experiment's code directory (-mods) first into the flat build directory, so a routine there
replaces the model's or a package's routine of the same file name, or adds a new one (tools/genmake2:2946-3006, the
first directory providing a name wins; mitjax/config/cpp_options.link_plan). The port reads only the CPP options,
SIZE.h and other headers of that directory; an own `.F` runs only where a Python port of THAT version exists.
`check_own_routines` accepts a file `<code dir>/<name>.F` when

  (a) its bytes equal a ported experiment version: an entry of PORTED below (verification/<exp>/<code dir>/<name>.F
      @63cdc0b, read from MJX_UPSTREAM) -- the dispatch to that port is by experiment name at the routine's call
      site (mitjax/model/src/ini_fields.py, pkg/ptracers/ptracers_forcing_surf.py, pkg/cost/cost_h.require_code_dir),
      which refuses a renamed copy by name; or
  (b) its bytes equal the MITgcm file it shadows (the same name in model/src, eesupp/src or pkg/*/ of MJX_UPSTREAM):
      an unchanged copy builds the same model.

Anything else stops with `UnportedRoutine` naming the routine(s) of the file (its SUBROUTINE / FUNCTION / PROGRAM
statements). docs/configurations.md (Task 10) explains how to add a port: a Python port of the routine plus an entry
here. `.F90` and `.c` files are refused the same way (no port reads them).
"""

import hashlib
import re
from pathlib import Path

# (experiment, code dir, file) -> the Python port of that experiment's own version (module[:function])
PORTED = {
    ("advect_xy", "code", "ini_salt.F"): "mitjax.verification.advect_xy.code.ini_salt",
    ("advect_xy", "code", "ini_theta.F"): "mitjax.verification.advect_xy.code.ini_theta",
    ("advect_xy", "code", "ini_vel.F"): "mitjax.verification.advect_xy.code.ini_vel",
    ("advect_cs", "code", "ini_vel.F"): "mitjax.verification.advect_cs.code.ini_vel",
    ("solid-body.cs-32x32x1", "code", "ini_psurf.F"): "mitjax.verification.solid_body_cs_32x32x1.code.ini_psurf",
    ("solid-body.cs-32x32x1", "code", "ini_vel.F"): "mitjax.verification.solid_body_cs_32x32x1.code.ini_vel",
    ("tutorial_global_oce_latlon", "code", "ptracers_apply_forcing.F"):
        "mitjax.verification.tutorial_global_oce_latlon.code.ptracers_apply_forcing",
    ("tutorial_global_oce_latlon", "code", "ptracers_forcing_surf.F"):
        "mitjax.verification.tutorial_global_oce_latlon.code.ptracers_forcing_surf",
    ("tutorial_tracer_adjsens", "code_ad", "ptracers_forcing_surf.F"):
        "mitjax.verification.tutorial_tracer_adjsens.code_ad.ptracers_forcing_surf",
    ("tutorial_global_oce_optim", "code_ad", "cost_hflux.F"): "mitjax.pkg.cost.cost_hflux",
    ("tutorial_global_oce_optim", "code_ad", "cost_temp.F"): "mitjax.pkg.cost.cost_temp",
    ("tutorial_global_oce_optim", "code_ad", "cost_weights.F"): "mitjax.pkg.cost.cost_weights",
    ("global_ocean.cs32x15", "code_ad", "cost_test.F"): "mitjax.pkg.cost.cost_test",
    # code_min/main.F skips THE_MODEL_MAIN (main.F:184-187 of that file): EEBOOT and the W2 set-up only, which
    # mitjax/pkg/exch2/w2_eeboot.py ports; drivers.model.Model refuses the build (MAIN_ONLY)
    ("adjustment.cs-32x32x1", "code_min", "main.F"): "mitjax.pkg.exch2.w2_eeboot:w2_eeboot",
}
# own files after which no model runs (drivers.model.Model refuses such a build)
MAIN_ONLY = ("main.F",)

_SOURCE_GLOBS = ("*.F", "*.F90", "*.c")
_UNIT = re.compile(r"^[ \t]+(?:[\w*() ]+\s)?(?:SUBROUTINE|FUNCTION|PROGRAM)\s+(\w+)", re.I | re.M)


class UnportedRoutine(NotImplementedError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def routine_names(path):
    """The program units a Fortran file defines (SUBROUTINE / FUNCTION / PROGRAM statements outside comments)."""
    text = "\n".join(ln for ln in Path(path).read_text(errors="replace").splitlines()
                     if ln[:1] not in ("C", "c", "*", "!") and not ln.lstrip().startswith("!"))
    return [m.group(1).upper() for m in _UNIT.finditer(text)]


def _shadowed(upstream, name):
    """The MITgcm files of that name a build could compile in its place (model/src, eesupp/src, pkg/*)."""
    up = Path(upstream)
    cands = [up / "model" / "src" / name, up / "eesupp" / "src" / name] + sorted(up.glob(f"pkg/*/{name}"))
    return [c for c in cands if c.is_file()]


def classify(exp_dir, code_dir, upstream):
    """{file name: (verdict, detail)} for every source file of `<exp_dir>/<code_dir>`; verdict 'ported' (detail: the
    Python port), 'unchanged' (detail: the identical MITgcm file) or 'unported' (detail: its routines)."""
    d = Path(exp_dir) / code_dir
    out = {}
    names = sorted({p.name for g in _SOURCE_GLOBS for p in d.glob(g) if p.is_file()})
    for name in names:
        f = d / name
        sha = _sha(f)
        hit = None
        for (e, c, n), port in PORTED.items():
            ref = Path(upstream) / "verification" / e / c / n
            if n == name and ref.is_file() and _sha(ref) == sha:
                hit = ("ported", f"{port} (= verification/{e}/{c}/{n})")
                break
        if hit is None and name.endswith(".F"):
            same = [s for s in _shadowed(upstream, name) if _sha(s) == sha]
            if same:
                hit = ("unchanged", str(same[0].relative_to(upstream)))
        out[name] = hit or ("unported", ", ".join(routine_names(f)) or "(no program unit found)")
    return out


def check_own_routines(exp_dir, code_dir, upstream):
    """Raise UnportedRoutine naming every unported routine of `<exp_dir>/<code_dir>`; returns classify()."""
    res = classify(exp_dir, code_dir, upstream)
    bad = {n: r for n, (v, r) in res.items() if v == "unported"}
    if bad:
        raise UnportedRoutine(
            f"{Path(exp_dir) / code_dir}: own source files without a Python port (decision 11): "
            + "; ".join(f"{n} (routine {r})" for n, r in bad.items())
            + ". Each must equal a ported version (mitjax/config/own_code.py PORTED) or the MITgcm file it replaces.")
    return res
