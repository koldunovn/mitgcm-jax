#!/usr/bin/env python3
"""Branch coverage of the gcov oracle builds (plan Task 5): per experiment and variant, the executed routines with
call counts, the unexecuted lines and never-taken branches inside them, all at `file.F:line` (upstream 63cdc0b),
the derived run-time switches from STDOUT and the physics-active check of the rung -> docs/coverage/<experiment>.md,
the porting worklist of the rung's milestone (RUNG_MILESTONE: M1 Tasks 10-16, M2 Tasks 22-27, M3, M4).

    coverage.py collect RUN_TOP --build BUILD_DIR --out DIR
    coverage.py report EXPERIMENT --variant INPUT_DIR RUN_TOP COLLECT_DIR [--variant ...] --build BUILD_DIR
                       --md FILE --json FILE

collect (needs gcov of the compiler that built the binary, e.g. `module load gcc/11.2.0-gcc-11.2.0`): the run was
made with GCOV_PREFIX=RUN_TOP/gcda (reference/coverage/run_gcov.sbatch), so its .gcda files sit under
RUN_TOP/gcda/<BUILD_DIR/bld>; they are linked with the build's .gcno and .f into DIR/obj (links only) and
`gcov --json-format --stdout --branch-probabilities` is run there; the line counts, branch-arc counts and function
execution counts of every compiled file go to DIR/gcov.json.gz. (R's tools/branch_coverage.py did the linking the
same way but read routine-level percentages only.)

report: genmake2 compiles the `cpp -P` output (Makefile `.F.f` rule), so gcov's line numbers refer to the `.f`;
tools/cpp_live.compiled_line_map regenerates each `.f` with the project's CPP module (lane D's link farm), checks it
equals the build's `.f` byte for byte, and maps every `.f` line to `file.F:line` through the line markers of the
same preprocessing without `-P` [E§2]. Sources the farm lacks (genmake2's template-generated exch routines) are
mapped in the build directory itself with the same preprocessor.
A routine is executed when gcov's function execution count is > 0 (the call count). Inside an executed routine,
"unexecuted" = instrumented lines with count 0 (grouped into ranges of consecutive instrumented lines), and a
"never-taken branch" = an executed line with a branch arc of count 0 (e.g. an `IF` whose THEN never ran, `gcov -b`).
Physics-active check [P§1]: the rung's key fields are non-trivial in the run (from %MON / %SBO / cost lines via
mitjax.io.stdout, and from execution counts: e.g. IVDC events = executions of calc_ivdc.F:48). The %MON, cg2d,
%SBO and cost lines of the gcov run are compared with those of the plain oracle run of the same variant.
"""

import argparse
import datetime
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mitjax import paths  # noqa: E402
from mitjax.config import cpp_options as _co  # noqa: E402
from mitjax.io import stdout as sio  # noqa: E402
from tools import cpp_live  # noqa: E402

GCOV_BATCH = 100
# headers genmake2/make write into the build directory that lane D's farm does not reproduce (BUILD_INFO.h holds the
# build date and user; FC_NAMEMANGLE.h, EMBEDDED_FILES.h); a source including one is mapped in the build directory
GENERATED_AT_MAKE = set(_co._GENERATED) - {"PACKAGES_CONFIG.h", "AD_CONFIG.h"}

# ---------------------------------------------------------------------------------------------------------------
# collect


def gcov_version():
    res = subprocess.run(["gcov", "--version"], capture_output=True, text=True, check=True)
    return res.stdout.splitlines()[0]


def compiler_version(build_dir):
    prov = (Path(build_dir) / "provenance.txt").read_text()
    m = re.search(r"^gfortran\s+\S+: (.*)$", prov, re.M)
    if not m:
        raise ValueError(f"{build_dir}/provenance.txt: no gfortran line")
    return m.group(1)


def _version_number(text):
    m = re.search(r"(\d+\.\d+\.\d+)", text)
    return m.group(1) if m else None


def parse_json_stream(text):
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return out
        obj, i = dec.raw_decode(text, i)
        out.append(obj)


def compact(doc):
    """One gcov JSON document (one .f) -> {"f", "functions": [[name, start, end, calls, blocks, blocks_run]],
    "lines": [[.f line, count, function index, [arc counts]]]}."""
    if len(doc["files"]) != 1:
        raise ValueError(f"{doc.get('data_file')}: expected one source file, got {[f['file'] for f in doc['files']]}")
    f = doc["files"][0]
    funcs = [[fn["name"], fn["start_line"], fn["end_line"], fn["execution_count"], fn["blocks"],
              fn["blocks_executed"]] for fn in f["functions"]]
    index = {fn[0]: k for k, fn in enumerate(funcs)}
    lines = [[ln["line_number"], ln["count"], index.get(ln.get("function_name"), -1),
              [b["count"] for b in ln["branches"]]] for ln in f["lines"]]
    return {"f": f["file"], "functions": funcs, "lines": lines}


def collect(run_top, build_dir, out):
    run_top, build_dir, out = Path(run_top).resolve(), Path(build_dir).resolve(), Path(out)
    bld = build_dir / "bld"
    gdir = run_top / "gcda" / str(bld).lstrip("/")
    gcdas = sorted(gdir.glob("*.gcda"))
    if not gcdas:
        raise SystemExit(f"no .gcda in {gdir}: was the run made with GCOV_PREFIX={run_top / 'gcda'}?")
    stray = [p for p in (run_top / "gcda").rglob("*.gcda") if p.parent != gdir]
    if stray:
        raise SystemExit(f"{len(stray)} .gcda files not from {bld}, first {stray[0]}")
    gv, cv = gcov_version(), compiler_version(build_dir)
    if _version_number(gv) != _version_number(cv):
        raise SystemExit(f"gcov '{gv}' does not match the compiler '{cv}' of the build")
    out.mkdir(parents=True)                         # refuses an existing directory
    obj = out / "obj"
    obj.mkdir()
    sources = []
    for g in gcdas:
        stem = g.name[:-len(".gcda")]
        src = next((bld / (stem + e) for e in (".f", ".f90") if (bld / (stem + e)).is_file()), None)
        gcno = bld / (stem + ".gcno")
        if src is None or not gcno.is_file():
            raise SystemExit(f"{g}: no {stem}.f/.f90 or {stem}.gcno in {bld}")
        for p in (g, gcno, src):
            (obj / p.name).symlink_to(p)
        sources.append(src.name)
    docs, errors = {}, []
    for k in range(0, len(sources), GCOV_BATCH):
        chunk = sources[k:k + GCOV_BATCH]
        res = subprocess.run(["gcov", "--json-format", "--stdout", "--branch-probabilities", "-o", "."] + chunk,
                             cwd=obj, capture_output=True, text=True)
        if res.returncode != 0 or res.stderr.strip():
            errors.append(f"gcov rc={res.returncode}: {res.stderr.strip()[:2000]}")
        for d in parse_json_stream(res.stdout):
            c = compact(d)
            docs[c["f"]] = c
    missing = sorted(set(sources) - set(docs))
    if errors or missing:
        raise SystemExit("gcov failed:\n" + "\n".join(errors) + (f"\nno output for {missing[:5]}" if missing else ""))
    data = {"run_top": str(run_top), "build_dir": str(build_dir), "gcov": gv, "compiler": cv,
            "made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "files": [docs[s] for s in sources]}
    with gzip.open(out / "gcov.json.gz", "wt") as fh:
        json.dump(data, fh)
    (out / "COLLECTED").write_text(f"{len(sources)} files\n")
    print(f"COLLECT OK {out}: {len(sources)} files, "
          f"{sum(1 for d in docs.values() for fn in d['functions'] if fn[3] > 0)} executed functions")
    return data


def load_collected(cdir):
    with gzip.open(Path(cdir) / "gcov.json.gz", "rt") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------------------------
# mapping .f -> .F

class Mapper:
    """compiled_line_map per compiled file, from the farm (byte check against the build's .f) or, for sources the
    farm does not hold, from the build directory itself."""

    def __init__(self, build, bld):
        self.build, self.bld = build, Path(bld)
        self.bview = cpp_live.build_dir_view(build, self.bld)
        self.cache, self.where, self.problems = {}, {}, []

    def source(self, fname):
        stem = fname.rsplit(".", 1)[0]
        for ext in (".F", ".F90"):
            if (self.build.path / (stem + ext)).is_file():
                return stem + ext, self.build, "farm"
        for ext in (".F", ".F90"):
            if (self.bld / (stem + ext)).exists():
                return stem + ext, self.bview, "build dir"
        raise FileNotFoundError(f"no source of {fname} in the farm or in {self.bld}")

    def line_map(self, fname):
        if fname in self.cache:
            return self.cache[fname]
        name, view, where = self.source(fname)
        f_text = (self.bld / fname).read_text(encoding="latin-1")
        try:
            m = cpp_live.compiled_line_map(view, name, f_text)
        except (ValueError, RuntimeError) as e:
            if view is self.bview:
                raise
            missing = re.search(r"fatal error: (\S+): No such file", str(e))
            if missing and missing.group(1) in GENERATED_AT_MAKE:
                # the farm leaves out headers genmake2/make generate per build (cpp_options._GENERATED)
                where = f"build dir (includes {missing.group(1)})"
            else:
                self.problems.append(f"{name}: farm mapping failed ({str(e)[:300]}); mapped in the build directory")
                where = "build dir (farm mismatch)"
            view = self.bview
            m = cpp_live.compiled_line_map(view, name, f_text)
        self.cache[fname] = (name, m)
        if view is self.build:
            src = view.source_of(name)
        else:                       # genmake2 writes generated sources into the build's upstream snapshot
            src = str(Path(self.bld / name).resolve())
            src = src.rsplit("/MITgcm/", 1)[1] if "/MITgcm/" in src else src
        self.where[name] = (where, src)
        return self.cache[fname]


def _ranges(items):
    """[(file, line)] in .f order -> 'a-b' strings of runs within one file (consecutive entries of the input)."""
    out, cur = [], None
    for f, ln in items:
        if cur and cur[0] == f and ln >= cur[2]:
            cur[2] = ln
        else:
            if cur:
                out.append(cur)
            cur = [f, ln, ln]
    if cur:
        out.append(cur)
    return out


def routines(data, mapper):
    """Per gcov function: routine, .F file and line, calls, line counts, unexecuted ranges, never-taken arcs."""
    out = []
    for doc in data["files"]:
        try:
            name, lmap = mapper.line_map(doc["f"])
        except Exception as e:                          # listed under "Mapping problems" in the report
            mapper.problems.append(f"{doc['f']}: not mapped ({type(e).__name__}: {str(e)[:300]}); its "
                                   f"{sum(1 for fn in doc['functions'] if fn[3] > 0)} executed functions are "
                                   f"missing from the tables")
            continue

        def where(n):
            if n - 1 >= len(lmap):
                raise ValueError(f"{doc['f']}: gcov line {n} beyond the {len(lmap)} lines of the .f")
            return lmap[n - 1]

        per = {k: [] for k in range(len(doc["functions"]))}
        for n, count, fi, arcs in doc["lines"]:
            per.setdefault(fi, []).append((n, count, arcs))
        for k, (fname, start, end, calls, blocks, bexec) in enumerate(doc["functions"]):
            src = where(start)
            lines = sorted(per.get(k, []))
            runs, zero, partial = 0, [], []
            for n, count, arcs in lines:
                loc = where(n)
                if loc is None:
                    raise ValueError(f"{doc['f']}:{n}: instrumented line maps to no source line")
                if count > 0:
                    runs += 1
                    nz = sum(1 for a in arcs if a == 0)
                    if nz:
                        partial.append((loc[0], loc[1], nz, len(arcs)))
                    if zero and zero[-1] is not None:
                        zero.append(None)
                else:
                    zero.append(loc)
            groups, cur = [], []
            for z in zero + [None]:
                if z is None:
                    if cur:
                        groups += _ranges(cur)
                    cur = []
                else:
                    cur.append(z)
            out.append({"function": fname, "routine": fname[:-1] if fname.endswith("_") else fname,
                        "file": name, "f": doc["f"], "line": src[1] if src else None,
                        "src_file": src[0] if src else None, "calls": calls, "lines": len(lines),
                        "lines_run": runs, "blocks": blocks, "blocks_run": bexec,
                        "unexecuted": [[f, a, b] for f, a, b in groups],
                        "never_taken": [list(p) for p in partial]})
    return out


def line_count(data, mapper, src_name, src_line):
    """Executions of the instrumented .f lines that map to src_name:src_line (None: no such instrumented line)."""
    total, found = 0, False
    for doc in data["files"]:
        if doc["f"] not in mapper.cache:
            continue                                    # routines() mapped every file it could
        name, lmap = mapper.line_map(doc["f"])
        for n, count, _, _ in doc["lines"]:
            loc = lmap[n - 1]
            if loc and loc[0] == src_name and loc[1] == src_line:
                total, found = total + count, True
    return total if found else None


# ---------------------------------------------------------------------------------------------------------------
# STDOUT: physics-active checks, switches, comparison with the plain oracle

_COST = re.compile(r"^\s*global fc =\s*(\S+)\s*$")
_OBJF = re.compile(r"^\s*--> (objf_\w+)\s*\(bi,bj=\s*(\d+),\s*(\d+)\)\s*=\s*(\S+)\s+(\S+)\s*$")

# rung key fields (plan Task 5 [P§1]; dispatch 2026-10-01): (kind, argument, label)
#   mon       %MON forward-dynamics series: nonzero somewhere and changing between monitor blocks
#   line      executions of an instrumented .F line > 0
#   routines  at least one executed routine whose name matches the regex
#   sbo       every %SBO value of the last record nonzero, and the records changing
#   cost      `global fc` (cost_final.F) nonzero; the objf_* terms listed
RUNG_CHECKS = {
    "tutorial_barotropic_gyre": [("mon", "dynstat_eta_sd", "eta"), ("mon", "dynstat_uvel_sd", "u"),
                                 ("mon", "dynstat_vvel_sd", "v")],
    "tutorial_baroclinic_gyre": [("mon", "dynstat_theta_sd", "theta stratification"),
                                 ("mon", "dynstat_uvel_sd", "u"), ("mon", "dynstat_eta_sd", "eta"),
                                 ("line", ("calc_ivdc.F", 48), "IVDC events (statically unstable points)")],
    "advect_xy": [("mon", "dynstat_theta_del2", "theta moved"), ("mon", "dynstat_salt_del2", "salt moved"),
                  ("routines", r"^gad_\w+_adv_[xy]$", "horizontal advection routine")],
    "advect_xz": [("mon", "dynstat_theta_del2", "theta moved"), ("mon", "dynstat_salt_del2", "salt moved"),
                  ("routines", r"^gad_\w+_(adv|impl)_r$", "vertical advection routine (explicit or implicit)")],
    "global_ocean.90x40x15": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                              ("mon", "dynstat_uvel_sd", "u"),
                              ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                              ("routines", r"^gmredi_slope_limit$", "GM/Redi slope limit"),
                              ("sbo", None, "SBO values")],
    "tutorial_global_oce_optim": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_uvel_sd", "u"),
                                  ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                  ("cost", None, "cost function")],
    # M2 (added by lane A, session 6: the M2 worklists). global_ocean.90x40x15 above also serves its code_ad
    # variants (input_ad*), whose report is a separate build/document (docs/coverage/global_ocean.90x40x15-code_ad.md)
    "adjustment.cs-32x32x1": [("mon", "dynstat_eta_sd", "eta"), ("mon", "dynstat_uvel_sd", "u"),
                              ("routines", r"^exch2_", "exch2 cube exchange")],
    "solid-body.cs-32x32x1": [("mon", "dynstat_uvel_sd", "u"), ("mon", "dynstat_eta_sd", "eta"),
                              ("routines", r"^mom_vecinv$", "vector-invariant momentum")],
    "advect_cs": [("mon", "dynstat_theta_del2", "theta moved"),
                  ("routines", r"^gad_\w+_adv_[xy]$", "horizontal advection routine"),
                  ("routines", r"^exch2_", "exch2 cube exchange")],
    "global_ocean.cs32x15": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                             ("mon", "dynstat_uvel_sd", "u"),
                             ("routines", r"^mom_vecinv$", "vector-invariant momentum"),
                             ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor")],
    "tutorial_global_oce_latlon": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_uvel_sd", "u"),
                                   ("routines", r"^ptracers_integrate$", "passive tracer step"),
                                   ("routines", r"^gad_calc_rhs$", "tracer tendencies"),
                                   ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor")],
    "tutorial_advection_in_gyre": [("mon", "dynstat_uvel_sd", "u"),
                                   ("routines", r"^ptracers_integrate$", "passive tracer step"),
                                   ("routines", r"^gad_\w+_adv_[xy]$", "horizontal advection routine")],
    "tutorial_tracer_adjsens": [("mon", "dynstat_theta_sd", "theta"),
                                ("routines", r"^ptracers_integrate$", "passive tracer step"),
                                ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                ("cost", None, "cost function")],
    # the forward-only code_ad build of global_ocean.90x40x15 (input_ad*): sbo compiled but not used (no useSBO)
    "global_ocean.90x40x15/code_ad": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                                      ("mon", "dynstat_uvel_sd", "u"),
                                      ("routines", r"^mom_vecinv$", "vector-invariant momentum"),
                                      ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                      ("cost", None, "cost function")],
    # input.in_p: no GM/Redi (useGMRedi commented out in its data.pkg); sea ice, EXF and GGL90 on
    "global_ocean.cs32x15/input.in_p": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                                        ("mon", "dynstat_uvel_sd", "u"),
                                        ("routines", r"^mom_vecinv$", "vector-invariant momentum"),
                                        ("routines", r"^seaice_growth$", "sea-ice thermodynamics"),
                                        ("routines", r"^ggl90_calc$", "GGL90 TKE")],
    # lane A session 7: the forward-only code_ad build of global_ocean.cs32x15 (input_ad: seaice/thsice/exf compiled,
    # all three off in its data.pkg) and the "minimal" code_min build of adjustment.cs-32x32x1 (eesupp + exch2 +
    # debug; its main.F skips THE_MODEL_MAIN: only the execution environment and the W2 topology are set up)
    "global_ocean.cs32x15/code_ad": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                                     ("mon", "dynstat_uvel_sd", "u"),
                                     ("routines", r"^mom_vecinv$", "vector-invariant momentum"),
                                     ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                     ("cost", None, "cost function")],
    "adjustment.cs-32x32x1/input_min": [("routines", r"^w2_eeboot$", "W2 topology set-up (EEBOOT)"),
                                        ("routines", r"^w2_e2setup$", "exch2 tile connectivity")],
    # M3 (lane A session 9, plan Task 28): one rung per variant where the mixing scheme differs
    "vermix": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^kpp_calc$", "KPP")],
    "vermix/input.dd": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^kpp_calc$", "KPP"),
                        ("routines", r"^kpp_doublediff$", "KPP double diffusion (KPPuseDoubleDiff)")],
    "vermix/input.ggl90": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^ggl90_calc$", "GGL90 TKE")],
    "vermix/input.gglLC": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^ggl90_calc$", "GGL90 TKE"),
                           ("line", ("ggl90_calc.F", 320), "Langmuir circulation branch (useLANGMUIR)")],
    "vermix/input.my82": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^my82_calc$", "MY82")],
    "vermix/input.opps": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^opps_calc$", "OPPS convection")],
    "vermix/input.pp81": [("mon", "dynstat_theta_sd", "theta"), ("routines", r"^pp81_calc$", "PP81")],
    "front_relax": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_uvel_sd", "u"),
                    ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor")],
    **{f"front_relax/input.{v}": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_uvel_sd", "u"),
                                  ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                  ("routines", r"^convective_adjustment$", "convective adjustment (cAdjFreq)")]
       for v in ("bvp", "mxl", "top")},
    "ideal_2D_oce": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_vvel_sd", "v"),
                     ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor")],
    "tutorial_reentrant_channel": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_uvel_sd", "u"),
                                   ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor"),
                                   ("routines", r"^rbcs_add_tendency$", "RBCS relaxation"),
                                   ("routines", r"^layers_calc$", "layers (output only)")],
    "MLAdjust": [("mon", "dynstat_uvel_sd", "u"), ("mon", "dynstat_vvel_sd", "v"),
                 ("routines", r"^mom_calc_visc$|^mom_(u|v)_(del2|biharm)", "horizontal viscosity")],
    "global_ocean.90x40x15/input.dwnslp": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                                           ("routines", r"^dwnslp_calc_flow$", "down-slope flow"),
                                           ("routines", r"^gmredi_calc_tensor$", "GM/Redi tensor")],
    "global_ocean.90x40x15/input.idemix": [("mon", "dynstat_theta_sd", "theta"), ("mon", "dynstat_eta_sd", "eta"),
                                           ("routines", r"^ggl90_calc$", "GGL90 TKE"),
                                           ("routines", r"^ggl90_idemix$", "IDEMIX internal-wave energy")],
}

# M4 (lane A session 11): sea ice, EXF, thsice, salt plume. Checks chosen from the measured runs: routine call counts of
# the gcov runs (job 27856010; cs32x15 jobs 27855995/27855996) and the %MON series that change in the plain runs
_SEAICE_MON = "MONITOR SEAICE statistics"
_HEFF = ("mon", ("seaice_heff_mean", _SEAICE_MON), "sea-ice volume changes")
_UICE = ("mon", ("seaice_uice_mean", _SEAICE_MON), "sea ice moves")
_THETA = ("mon", "dynstat_theta_sd", "theta")
_R = {k: ("routines", rf"^{k}$", w) for k, w in (
    ("seaice_growth", "sea-ice thermodynamics"), ("seaice_growth_adx", "sea-ice thermodynamics (GROWTH_ADX)"),
    ("seaice_solve4temp", "ice surface temperature"), ("seaice_lsr", "LSR momentum solver"),
    ("seaice_evp", "EVP momentum solver"), ("seaice_jfnk", "JFNK momentum solver"),
    ("seaice_krylov", "Krylov momentum solver"), ("seaice_freedrift", "free drift"),
    ("seaice_advdiff", "sea-ice advection"), ("seaice_itd_remap", "ITD remapping"),
    ("seaice_itd_redist", "ITD redistribution"), ("dynsolver", "B-grid DYNSOLVER"),
    ("ostres", "B-grid ocean stress under ice"), ("exf_bulkformulae", "EXF bulk formulae"),
    ("thsice_main", "thermodynamic sea ice (pkg/thsice)"), ("kpp_calc", "KPP"),
    ("gmredi_calc_tensor", "GM/Redi tensor"), ("salt_plume_calc_depth", "salt plume"),
    ("ctrl_map_gentim2d", "time-varying forcing controls"))}
_COST_CHECK = ("cost", None, "cost function")
RUNG_CHECKS.update({
    "1D_ocean_ice_column": [_HEFF, _THETA, _R["seaice_growth"], _R["seaice_solve4temp"], _R["exf_bulkformulae"],
                            _R["dynsolver"], _R["ostres"], _R["kpp_calc"]],
    "1D_ocean_ice_column/code_ad": [_HEFF, _THETA, _R["seaice_growth"], _R["exf_bulkformulae"], _R["kpp_calc"],
                                    _R["ctrl_map_gentim2d"], _COST_CHECK],
    "offline_exf_seaice": [_UICE, _R["seaice_lsr"], _R["thsice_main"], _R["exf_bulkformulae"]],
    "offline_exf_seaice/input.thermo": [_HEFF, _R["seaice_growth"], _R["seaice_solve4temp"], _R["exf_bulkformulae"]],
    **{f"offline_exf_seaice/input.dyn_{v}": [_UICE, _R["seaice_lsr"], _R["seaice_advdiff"]]
       for v in ("lsr", "ellnnfr", "mce")},
    "offline_exf_seaice/input.dyn_teardrop": [_UICE, _R["seaice_lsr"], _R["thsice_main"]],
    "offline_exf_seaice/input.dyn_jfnk": [_UICE, _R["seaice_jfnk"], _R["thsice_main"]],
    "offline_exf_seaice/input.dyn_paralens": [_UICE, _R["seaice_krylov"], _R["thsice_main"]],
    "offline_exf_seaice/input.thsice": [_R["thsice_main"], _R["exf_bulkformulae"]],
    "offline_exf_seaice/input_ad": [_HEFF, _R["seaice_growth_adx"], _R["ctrl_map_gentim2d"], _COST_CHECK],
    "offline_exf_seaice/input_ad.thsice": [_R["thsice_main"], _R["ctrl_map_gentim2d"], _COST_CHECK],
    "lab_sea": [_HEFF, _UICE, _R["seaice_growth"], _R["seaice_lsr"], _R["exf_bulkformulae"], _R["kpp_calc"],
                _R["gmredi_calc_tensor"]],
    "lab_sea/input.fd": [_HEFF, _R["seaice_freedrift"], _R["seaice_growth"]],
    "lab_sea/input.hb87": [_HEFF, _R["seaice_evp"], _R["seaice_growth"]],
    "lab_sea/input.salt_plume": [_HEFF, _R["seaice_growth"], _R["seaice_lsr"], _R["salt_plume_calc_depth"]],
    # input.longstep, input.natl_box run neither sea ice nor EXF (no seaice_*/exf_* routine executed)
    "lab_sea/input.longstep": [_THETA, _R["kpp_calc"], _R["gmredi_calc_tensor"]],
    "lab_sea/input.natl_box": [_THETA, _R["kpp_calc"]],
    "lab_sea/code_ad": [_HEFF, _UICE, _R["seaice_growth"], _R["seaice_lsr"], _R["ctrl_map_gentim2d"], _COST_CHECK],
    "lab_sea/input_ad.noseaice": [_THETA, _R["exf_bulkformulae"], _R["kpp_calc"], _R["ctrl_map_gentim2d"],
                                  _COST_CHECK],
    "lab_sea/input_ad.noseaicedyn": [_HEFF, _R["seaice_growth"], _R["salt_plume_calc_depth"],
                                     _R["ctrl_map_gentim2d"], _COST_CHECK],
    "seaice_itd": [_HEFF, _UICE, _R["seaice_growth"], _R["seaice_itd_remap"], _R["seaice_lsr"]],
    "seaice_itd/input.lipscomb07": [_UICE, _R["seaice_itd_redist"], _R["seaice_lsr"]],
    "seaice_itd/input.thermo": [_HEFF, _R["seaice_growth"], _R["seaice_itd_redist"]],
    "global_ocean.cs32x15/input.seaice": [_THETA, _R["seaice_growth"], _R["seaice_lsr"], _R["exf_bulkformulae"]],
    "global_ocean.cs32x15/input.icedyn": [_UICE, _R["seaice_lsr"], _R["thsice_main"]],
    "global_ocean.cs32x15/input.thsice": [_THETA, _R["thsice_main"]],
    **{f"global_ocean.cs32x15/input_ad.{v}": [_R["seaice_growth"], _R["seaice_lsr"], _R["ctrl_map_gentim2d"],
                                              _COST_CHECK] for v in ("seaice", "seaice_dynmix")},
    "global_ocean.cs32x15/input_ad.thsice": [_R["thsice_main"], _R["ctrl_map_gentim2d"], _COST_CHECK],
})
M4_RUNGS = [k for k in RUNG_CHECKS if k.split("/")[0] in ("1D_ocean_ice_column", "offline_exf_seaice", "lab_sea",
                                                            "seaice_itd")
            or k.startswith(("global_ocean.cs32x15/input.seaice", "global_ocean.cs32x15/input.icedyn",
                             "global_ocean.cs32x15/input.thsice", "global_ocean.cs32x15/input_ad."))]

# milestone of each rung (the project plan: M1 Tasks 10-16, M2 Tasks 22-27, M3 28-31); the
# report title names it (one report covers the variants of one build)
RUNG_MILESTONE = {k: "M1" for k in ("tutorial_barotropic_gyre", "tutorial_baroclinic_gyre", "advect_xy", "advect_xz",
                                    "global_ocean.90x40x15", "tutorial_global_oce_optim")}
RUNG_MILESTONE.update({k: "M3" for k in RUNG_CHECKS if k.split("/")[0] in (
    "vermix", "front_relax", "ideal_2D_oce", "tutorial_reentrant_channel", "MLAdjust")
    or k in ("global_ocean.90x40x15/input.dwnslp", "global_ocean.90x40x15/input.idemix")})
RUNG_MILESTONE.update({k: "M4" for k in M4_RUNGS})
RUNG_MILESTONE.update({k: "M2" for k in RUNG_CHECKS if k not in RUNG_MILESTONE})


def milestone(experiment, input_dir):
    """Milestone of a variant's rung (RUNG_MILESTONE of rung_key)."""
    return RUNG_MILESTONE[rung_key(experiment, input_dir)]


def report_title(experiment, code, input_dirs):
    """Title line of a coverage report: the milestones of its variants' rungs."""
    ms = sorted({milestone(experiment, i) for i in input_dirs})
    return f"# Coverage: {experiment} ({code}) — {'/'.join(ms)} porting worklist"


def rung_key(experiment, input_dir):
    """RUNG_CHECKS key of a variant: `<experiment>/<input dir>` when listed, else `<experiment>/code_ad` for an
    input_ad* variant when listed, else the experiment (RUNG_CHECKS must list it)."""
    for key in (f"{experiment}/{input_dir}",
                f"{experiment}/code_ad" if input_dir.split(".", 1)[0] == "input_ad" else None):
        if key in RUNG_CHECKS:
            return key
    return experiment


def physics_checks(experiment, lines, data, rows, mapper):
    out = []
    blocks = sio.monitor_blocks(lines)
    for kind, arg, label in RUNG_CHECKS[experiment]:
        if kind == "mon":
            # arg: a %MON name of the dynamics monitor, or (name, block kind) for another monitor (M4: the SEAICE
            # statistics block, `// Begin MONITOR SEAICE statistics`)
            s = sio.monitor_series(blocks, *arg) if isinstance(arg, tuple) else sio.monitor_series(blocks, arg)
            ok = len(s) >= 2 and any(v != 0 for v in s) and max(s) != min(s)
            val = f"{len(s)} blocks, first {s[0]:.6e}, last {s[-1]:.6e}, min {min(s):.6e}, max {max(s):.6e}" \
                if s else "no such %MON series"
        elif kind == "line":
            n = line_count(data, mapper, *arg)
            ok = bool(n)
            val = f"{arg[0]}:{arg[1]} executed {n} times" if n is not None else f"{arg[0]}:{arg[1]} not compiled"
        elif kind == "routines":
            hit = sorted({(r["routine"], r["calls"]) for r in rows if re.match(arg, r["routine"]) and r["calls"] > 0})
            ok = bool(hit)
            val = ", ".join(f"{n} ({c} calls)" for n, c in hit) or f"no executed routine matches {arg}"
        elif kind == "sbo":
            recs = sio.sbo_records(lines)
            last = recs[-1].text if recs else {}
            ok = bool(recs) and all(sio.fortran_number(v) != 0 for v in last.values()) and len(recs) >= 2 and \
                recs[0].text != recs[-1].text
            val = f"{len(recs)} records; last: " + ", ".join(f"{k} {v}" for k, v in last.items())
        elif kind == "cost":
            fc = [sio.fortran_number(m[1]) for ln in lines if (m := _COST.match(ln.text))]
            terms = {}
            for ln in lines:
                if m := _OBJF.match(ln.text):
                    terms.setdefault(m[1], []).append(sio.fortran_number(m[4]))
            ok = bool(fc) and fc[-1] != 0
            val = (f"global fc = {fc[-1]!r}" if fc else "no `global fc` line") + "; terms: " + ", ".join(
                f"{k} sum {sum(v):.6e}{' (ZERO)' if all(x == 0 for x in v) else ''}" for k, v in terms.items())
        else:
            raise ValueError(kind)
        out.append({"check": label, "kind": kind, "arg": arg if not isinstance(arg, tuple) else list(arg),
                    "ok": ok, "value": val})
    return out


_COMPARE = re.compile(r"^(%MON |%SBO |\s*cg2d_init_res|\s*cg2d_iters|\s*cg2d_last_res|\s*global fc|\s*--> objf)")


def compared_lines(lines):
    return [ln.text for ln in lines if _COMPARE.match(ln.text)]


def plain_oracle_output(experiment, input_dir):
    """output.txt of the newest plain oracle run (run id job<N> or job<N>-plain) of the variant, or None."""
    runs = []
    for p in (paths.REFERENCE_RUNS / experiment / input_dir).glob("job*"):
        m = re.fullmatch(r"job(\d+)(-plain)?", p.name)
        if m and (p / "READY").exists() and (p / "rundir" / "output.txt").exists():
            runs.append((int(m[1]), p / "rundir" / "output.txt"))
    return max(runs)[1] if runs else None


def switches(lines, run_keys):
    """Logical switches and integer selectors of the STDOUT parameter printout not set in the run's data files."""
    on, off, sel = [], [], []
    for p in sio.parameters(lines):
        try:
            vals = p.values()
        except ValueError:
            continue
        if len(vals) != 1 or p.name.lower() in run_keys:
            continue
        v = vals[0]
        if v in ("T", "F"):
            (on if v == "T" else off).append(p.name)
        elif re.fullmatch(r"[+-]?\d+", v) and re.search(r"(?i)select|scheme|order|method|type", p.name):
            sel.append(f"{p.name}={v}")
    return sorted(set(on), key=str.lower), sorted(set(off), key=str.lower), sorted(set(sel), key=str.lower)


def run_param_keys(experiment, input_dir):
    """Lower-case names of every parameter set in the variant's parameter files (lane D's reader), or None."""
    try:
        from mitjax.config import params
        e = params.load(experiment, input_dir)
    except Exception as exc:                                # reported, not fatal: the switch list is informative
        return None, f"{type(exc).__name__}: {exc}"
    return {k.lower() for (_, _, k) in e.run.vars}, None


# ---------------------------------------------------------------------------------------------------------------
# report

def variant_report(experiment, input_dir, run_top, cdir, mapper):
    data = load_collected(cdir)
    rows = routines(data, mapper)
    out_txt = Path(run_top) / "rundir" / "output.txt"
    lines = sio.read_stdout(out_txt)
    phys = physics_checks(rung_key(experiment, input_dir), lines, data, rows, mapper)
    plain = plain_oracle_output(experiment, input_dir)
    if plain is None:
        cmp = {"plain": None, "equal": None, "note": "no plain oracle run found"}
    else:
        a, b = compared_lines(lines), compared_lines(sio.read_stdout(plain))
        diff = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
        cmp = {"plain": str(plain), "equal": a == b, "lines": len(a), "plain_lines": len(b),
               "first_difference": diff if diff is not None else (min(len(a), len(b)) if len(a) != len(b) else None)}
    keys, kerr = run_param_keys(experiment, input_dir)
    on, off, sel = switches(lines, keys or set())
    prov = (Path(run_top) / "run_provenance.txt").read_text() if (Path(run_top) / "run_provenance.txt").exists() else ""
    m = re.search(r"exit (\d+) normal_end (\d+)", prov)
    return {"input_dir": input_dir, "run_top": str(run_top), "collect": str(cdir), "gcov": data["gcov"],
            "exit": int(m[1]) if m else None, "normal_end": int(m[2]) if m else None,
            "mon_blocks": len(sio.monitor_series(sio.monitor_blocks(lines), "time_tsnumber")),
            "routines": rows, "physics": phys, "compare_plain": cmp,
            "switches": {"on": on, "off": off, "selectors": sel, "keys_error": kerr}}


def _fmt_ranges(rs, home):
    parts = []
    for f, a, b in rs:
        s = f"{a}" if a == b else f"{a}-{b}"
        parts.append(s if f == home else f"{f}:{s}")
    return ", ".join(parts)


def _group_dir(src):
    parts = src.split("/")
    if parts[0] == "pkg" and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[0] in ("model", "eesupp") and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[0] == "verification" and len(parts) > 3:
        return "/".join(parts[:3])
    return "/".join(parts[:-1]) or src


def neutral_paths(text):
    """`text` with this machine's locations written as their variables, longest first ($MJX_REFERENCE_RUNS before
    $MJX_REFERENCE, ...; mitjax/paths.py): the pages are published, so they name `$MJX_REFERENCE/coverage/...`, not the
    absolute run directory (Nikolay 2026-10-08). Idempotent; the JSON report keeps the absolute paths."""
    subs = sorted(((str(v), f"${k}") for k, v in paths.ALL.items() if k != "MJX_PYTHON"), key=lambda kv: -len(kv[0]))
    for a, b in subs:
        text = text.replace(a, b)
    return text


def markdown(experiment, code, build_name, variants, mapper, meta):
    L = [report_title(experiment, code, [v["input_dir"] for v in variants]), "",
         f"Written by `tools/coverage.py` (mitjax {meta['commit']}) from the gcov build `{build_name}` "
         f"(oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are "
         f"`file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and "
         f"matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): "
         f"{sum(1 for w, _ in mapper.where.values() if w == 'farm')} files from the link farm, "
         f"{sum(1 for w, _ in mapper.where.values() if w != 'farm')} from the build directory "
         f"(genmake2-generated sources the farm does not hold).", ""]
    if mapper.problems:
        L += ["**Mapping problems:**", ""] + [f"- {p}" for p in mapper.problems] + [""]
    L += ["Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented "
          "lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of "
          "the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken "
          "(`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).", "",
          "| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |",
          "|---|---|---|---|---|---|---|"]
    for v in variants:
        ex = sum(1 for r in v["routines"] if r["calls"] > 0)
        ok = all(c["ok"] for c in v["physics"])
        c = v["compare_plain"]
        cmpt = "no plain run" if c["equal"] is None else (
            f"identical ({c['lines']} lines)" if c["equal"] else
            f"DIFFERENT ({c['lines']} vs {c['plain_lines']} lines, first at {c['first_difference']})")
        L.append(f"| `{v['input_dir']}` | `{Path(v['run_top']).name}` | {v['mon_blocks']} | {v['exit']} / "
                 f"{v['normal_end']} | {ex} | {'yes' if ok else '**NO**'} | {cmpt} |")
    L.append("")
    for v in variants:
        L += [f"## {experiment}/{v['input_dir']}", "",
              f"Run `{v['run_top']}` (STDOUT `rundir/output.txt`); counts `{v['collect']}/gcov.json.gz`.", ""]
        if not v["normal_end"]:
            L += ["The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the "
                  "code executed up to the stop (see the run's output.txt).", ""]
        L += ["### Physics-active check", "", "| check | measured | active |", "|---|---|---|"]
        for c in v["physics"]:
            L.append(f"| {c['check']} | {c['value']} | {'yes' if c['ok'] else '**NO**'} |")
        s = v["switches"]
        L += ["", "### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)", ""]
        if s["keys_error"]:
            L += [f"(parameter files could not be read by mitjax.config: {s['keys_error']}; all printed switches "
                  "are listed)", ""]
        L += [f"- `.TRUE.`: {', '.join(s['on']) or '-'}", f"- `.FALSE.`: {', '.join(s['off']) or '-'}",
              f"- selectors: {', '.join(s['selectors']) or '-'}", ""]
        rows = [r for r in v["routines"] if r["calls"] > 0]
        by = {}
        for r in rows:
            by.setdefault(_group_dir(mapper.where[r["file"]][1]), []).append(r)
        L += [f"### Executed routines ({len(rows)})", ""]
        for g in sorted(by):
            L += [f"**{g}** ({len(by[g])})", "", "| routine | file:line | calls | lines run | unexecuted | never taken |",
                  "|---|---|---|---|---|---|"]
            for r in sorted(by[g], key=lambda r: r["routine"]):
                nt = ", ".join(f"{ln}:{z}/{n}" if f == r["file"] else f"{f}:{ln}:{z}/{n}"
                               for f, ln, z, n in r["never_taken"])
                L.append(f"| `{r['routine']}` | {r['file']}:{r['line']} | {r['calls']} | {r['lines_run']}/{r['lines']} "
                         f"| {_fmt_ranges(r['unexecuted'], r['file']) or '-'} | {nt or '-'} |")
            L.append("")
        dead = [r for r in v["routines"] if r["calls"] == 0]
        dby = {}
        for r in dead:
            dby.setdefault(_group_dir(mapper.where[r["file"]][1]), []).append(r["routine"])
        L += [f"### Compiled but not executed ({len(dead)} routines)", ""]
        L += [f"- {g}: {', '.join(sorted(dby[g]))}" for g in sorted(dby)] + [""]
    return neutral_paths("\n".join(L) + "\n")


def report(experiment, variants, build_dir, md, js, commit):
    build_dir = Path(build_dir)
    name = build_dir.name
    m = re.search(r"-(code(?:_\w+?)?)-[0-9a-f]{7}-[0-9a-f]{7}", name)
    code = m[1] if m else "code"
    build = cpp_live.build_for(experiment, code=code)
    mapper = Mapper(build, build_dir / "bld")
    vs = [variant_report(experiment, inp, top, cdir, mapper) for inp, top, cdir in variants]
    meta = {"commit": commit, "build": name, "experiment": experiment, "code": code}
    text = markdown(experiment, code, name, vs, mapper, meta)
    Path(md).write_text(text)
    with gzip.open(js, "wt") if str(js).endswith(".gz") else open(js, "w") as fh:
        json.dump({"meta": meta, "where": mapper.where, "problems": mapper.problems, "variants": vs}, fh)
    for v in vs:
        ok = all(c["ok"] for c in v["physics"])
        print(f"REPORT {experiment}/{v['input_dir']}: {sum(1 for r in v['routines'] if r['calls'] > 0)} executed "
              f"routines; physics-active {'yes' if ok else 'NO'}; vs plain {v['compare_plain'].get('equal')}")
    print(f"REPORT OK {md}")
    return vs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("run_top")
    c.add_argument("--build", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("experiment")
    r.add_argument("--variant", nargs=3, action="append", required=True, metavar=("INPUT_DIR", "RUN_TOP", "COLLECT"))
    r.add_argument("--build", required=True)
    r.add_argument("--md", required=True)
    r.add_argument("--json", required=True)
    r.add_argument("--commit", default="?")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.run_top, a.build, a.out)
    else:
        report(a.experiment, a.variant, a.build, a.md, a.json, a.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
