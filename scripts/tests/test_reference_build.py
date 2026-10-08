"""Fortran oracle infrastructure (plan Task 3a): the optfile is master's gfortran optfile plus the audited Levante
block, the frozen M1 binaries exist with matching sha256 and passed their build checks, make_rundir reproduces
testreport's run directories, and the build checks catch what they are meant to catch (negative controls).

Oracle-dependent: test_m1_binaries fails (does not skip) while a binary is missing."""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mitjax import paths

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = paths.UPSTREAM
VERIF = UPSTREAM / "verification"
UP7 = "63cdc0b"   # branch `pinned` of the clone (plan: MITgcm master 63cdc0b)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_mjx_ref_{name}", REPO / "reference" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_build = _load("check_build")

# The M1 builds (plan Task 3a; Context "M1 experiment facts"). Variants share their experiment's build, as in
# testreport (one build per code directory, every input.<v> run with it): advect_xy/code defines
# ALLOW_ADAMSBASHFORTH_3 for ab3_c4, advect_xz/code defines NONLIN_FRSURF for nlfs.
M1_BUILDS = [("tutorial_barotropic_gyre", "code"), ("tutorial_baroclinic_gyre", "code"), ("advect_xy", "code"),
             ("advect_xz", "code"), ("global_ocean.90x40x15", "code"), ("tutorial_global_oce_optim", "code_ad")]

# Package sets derived by hand from pkg/pkg_groups ("gfd : mom_common mom_fluxform mom_vecinv generic_advdiff debug
# mdsio rw monitor", "oceanic : gfd gmredi kpp", "adjoint : autodiff cost ctrl grdchk") and each packages.conf.
GFD = {"mom_common", "mom_fluxform", "mom_vecinv", "generic_advdiff", "debug", "mdsio", "rw", "monitor"}
M1_REQUESTED = {
    "tutorial_barotropic_gyre": GFD,                                        # no packages.conf: default_pkg_list
    "tutorial_baroclinic_gyre": GFD | {"diagnostics", "mnc"},
    "advect_xy": GFD,                                                       # no packages.conf
    "advect_xz": (GFD - {"mom_common", "mom_fluxform", "mom_vecinv"}) | {"diagnostics"},
    "global_ocean.90x40x15": GFD | {"exch2", "gmredi", "cd_code", "down_slope", "ggl90", "ptracers", "sbo",
                                    "diagnostics"},                         # oceanic minus kpp
    "tutorial_global_oce_optim": GFD | {"cd_code", "gmredi", "autodiff", "cost", "ctrl", "grdchk"},
}


# M2 (lane A session 6): every code dir of the M2 variants; plain build 06b4417 (job 27832154), jaxdump build
# 8316e95-jaxdump (job 27832225), gcov build 8316e95-gcov (job 27832234). Package sets by hand as above.
M2_BUILDS = [("adjustment.cs-32x32x1", "code"), ("solid-body.cs-32x32x1", "code"), ("advect_cs", "code"),
             ("global_ocean.cs32x15", "code"), ("global_ocean.90x40x15", "code_ad"),
             ("tutorial_global_oce_latlon", "code"), ("tutorial_advection_in_gyre", "code"),
             ("tutorial_tracer_adjsens", "code_ad")]
M2_TAGS = ("06b4417", "8316e95-jaxdump", "8316e95-gcov")
ADJOINT = {"autodiff", "cost", "ctrl", "grdchk"}
M2_REQUESTED = {
    "adjustment.cs-32x32x1": (GFD - {"generic_advdiff"}) | {"exch2", "diagnostics"},
    "solid-body.cs-32x32x1": (GFD - {"mom_fluxform"}) | {"exch2", "diagnostics"},
    "advect_cs": GFD | {"exch2", "diagnostics"},
    "global_ocean.cs32x15": GFD | {"exch2", "gmredi", "ggl90", "bulk_force", "exf", "seaice", "thsice", "diagnostics",
                                   "mnc"},                                  # "-cal" removes cal
    "global_ocean.90x40x15": GFD | {"cd_code", "gmredi", "sbo", "mnc"} | ADJOINT,      # code_ad
    "tutorial_global_oce_latlon": GFD | {"cd_code", "gmredi", "ptracers", "mnc"},
    "tutorial_advection_in_gyre": GFD | {"ptracers", "diagnostics", "mnc"},
    "tutorial_tracer_adjsens": GFD | {"cd_code", "gmredi", "kpp", "ptracers"} | ADJOINT,
}
# M3 (lane A session 9, plan Task 28): plain, jaxdump and gcov binaries 463504e of the five M3 codes (the hand-derived
# package sets from their packages.conf); global_ocean.90x40x15/code has the M1 plain and gcov binaries and a new
# jaxdump binary with the column-mixing stages
M3_BUILDS = {"vermix": GFD | {"kpp", "pp81", "my82", "ggl90", "opps", "diagnostics", "mnc"},
             "front_relax": GFD | {"gmredi", "diagnostics"},
             "ideal_2D_oce": GFD | {"cd_code", "gmredi", "diagnostics"},
             "tutorial_reentrant_channel": GFD | {"gmredi", "rbcs", "layers", "diagnostics"},
             "MLAdjust": GFD | {"exch2", "gmredi", "flt", "diagnostics", "mnc"}}
M3_TAGS = ("463504e", "463504e-jaxdump", "463504e-gcov")
# lane A session 7: (exp, code) -> (binary tags, hand-derived requested set, extra genmake2 options)
M2B_BUILDS = {
    ("global_ocean.cs32x15", "code_ad"): (("78ca390", "78ca390-jaxdump", "78ca390-gcov"),
                                          (GFD - {"mom_fluxform"}) | {"exch2", "gmredi", "exf", "seaice", "thsice",
                                                                      "diagnostics"} | ADJOINT, ""),
    # README of adjustment.cs-32x32x1 ("minimal" test case): genmake2 -standarddirs eesupp, packages exch2 + debug
    ("adjustment.cs-32x32x1", "code_min"): (("78ca390", "78ca390-gcov"), {"exch2", "debug"}, "-standarddirs eesupp"),
}


def _git_show(path):
    return subprocess.run(["git", "-C", str(UPSTREAM), "show", f"pinned:{path}"], check=True,
                          capture_output=True).stdout


def master_block(text):
    m = re.search(r"^#- MJX-BEGIN-MASTER[^\n]*\n(.*?)^#- MJX-END-MASTER\n", text, re.S | re.M)
    return m.group(1).encode() if m else None


def test_optfile_is_master_plus_levante_block():
    text = (REPO / "reference" / "optfile_levante_gfortran").read_text()
    master = _git_show("tools/build_options/linux_amd64_gfortran")
    assert master_block(text) == master
    tail = text.split("#- MJX-END-MASTER\n", 1)[1]
    code = [ln.strip() for ln in tail.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # the only additions: no-FMA flags and the NetCDF paths (README "Optfile audit")
    assert code[:2] == ['FFLAGS="$FFLAGS -ffp-contract=off"', 'F90FLAGS="$F90FLAGS -ffp-contract=off"']
    assert all(not re.search(r"FOPTIM|NOOPT|-O[0-9s]|fast|march|DEFINES", ln) for ln in code), code
    # master's optfile itself: -fconvert=big-endian, -O0 under IEEE, no fast-math (what the build relies on)
    ms = master.decode()
    assert 'FFLAGS="$FFLAGS -fconvert=big-endian -fimplicit-none"' in ms and "FOPTIM='-O0'" in ms
    assert "fast-math" not in ms
    # negative control: one changed character in the master block is caught
    planted = text.replace("-fconvert=big-endian", "-fconvert=little-endian", 1)
    assert master_block(planted) != master


def bin_problems(d):
    """Problems of a frozen binary directory (empty list = good)."""
    d = Path(d)
    exe, sha = d / "mitgcmuv", d / "sha256"
    if not exe.is_file() or not sha.is_file():
        return [f"{d}: mitgcmuv or sha256 missing"]
    out = []
    want = sha.read_text().split()
    if len(want) != 2 or want[1] != "mitgcmuv":
        out.append(f"{sha}: not a `sha256sum mitgcmuv` line")
    elif hashlib.sha256(exe.read_bytes()).hexdigest() != want[0]:
        out.append(f"{exe}: sha256 differs from {sha}")
    for f, verdict in (("packages_check_after_make.txt", "PACKAGES OK"), ("flags_check.txt", "FLAGS OK"),
                       ("forward_check.txt", "FORWARD OK")):
        if not (d / f).is_file() or verdict not in (d / f).read_text():
            out.append(f"{d / f}: no '{verdict}'")
    for f in ("build.log", "genmake2_command.txt", "genmake_state", "provenance.txt", "PACKAGES_CONFIG.h",
              "AD_CONFIG.h", "logs/make.log", "options/SOURCES.txt", "options/CPP_OPTIONS.h.dM"):
        if not (d / f).is_file():
            out.append(f"{d / f} missing")
    return out


def test_m1_binaries():
    """Every M1 build has a frozen binary at upstream 63cdc0b with a matching sha256, passed checks, and the
    expected package set compiled (fails while a binary is missing: reference/jobs/build.sbatch)."""
    problems = []
    for exp, code in M1_BUILDS:
        found = sorted((paths.REFERENCE / "bin").glob(f"{exp}-{code}-{UP7}-*"))
        if not found:
            problems.append(f"no binary for {exp}/{code} in {paths.REFERENCE / 'bin'}")
            continue
        for d in found:
            problems += bin_problems(d)
            enabled = check_build.enabled_in_config_h(d / "PACKAGES_CONFIG.h")
            missing = {p.upper() for p in M1_REQUESTED[exp]} - enabled
            if missing:
                problems.append(f"{d.name}: requested packages not compiled: {sorted(missing)}")
            if "-ieee" not in (d / "genmake2_command.txt").read_text():
                problems.append(f"{d.name}: not built with -ieee (testreport default)")
    assert not problems, "\n".join(problems)


def test_m2_binaries():
    """Every M2 code has its plain, jaxdump and gcov binary at upstream 63cdc0b with a matching sha256, passed
    checks, -ieee and the hand-derived package set compiled (lane A session 6; fails while a binary is missing)."""
    problems = []
    for exp, code in M2_BUILDS:
        for tag in M2_TAGS:
            d = paths.REFERENCE / "bin" / f"{exp}-{code}-{UP7}-{tag}"
            if not d.is_dir():
                problems.append(f"missing {d}")
                continue
            problems += bin_problems(d)
            enabled = check_build.enabled_in_config_h(d / "PACKAGES_CONFIG.h")
            missing = {p.upper() for p in M2_REQUESTED[exp]} - enabled
            if missing:
                problems.append(f"{d.name}: requested packages not compiled: {sorted(missing)}")
            if "-ieee" not in (d / "genmake2_command.txt").read_text():
                problems.append(f"{d.name}: not built with -ieee (testreport default)")
            if tag.endswith("-gcov") and "COVERAGE OK" not in (d / "COVERAGE.txt").read_text():
                problems.append(f"{d.name}: COVERAGE.txt does not say COVERAGE OK")
    for (exp, code), (tags, requested, extra) in M2B_BUILDS.items():
        for tag in tags:
            d = paths.REFERENCE / "bin" / f"{exp}-{code}-{UP7}-{tag}"
            if not d.is_dir():
                problems.append(f"missing {d}")
                continue
            problems += bin_problems(d)
            missing = {q.upper() for q in requested} - check_build.enabled_in_config_h(d / "PACKAGES_CONFIG.h")
            if missing:
                problems.append(f"{d.name}: requested packages not compiled: {sorted(missing)}")
            cmd = (d / "genmake2_command.txt").read_text()
            if "-ieee" not in cmd or not cmd.rstrip().endswith(("-ieee " + extra).rstrip()):
                problems.append(f"{d.name}: genmake2 command does not end with '-ieee {extra}'")
            if tag.endswith("-gcov") and "COVERAGE OK" not in (d / "COVERAGE.txt").read_text():
                problems.append(f"{d.name}: COVERAGE.txt does not say COVERAGE OK")
    assert not problems, "\n".join(problems)


def test_m3_binaries_and_package_sets():
    """Every M3 code has its plain, jaxdump and gcov binary (463504e) with a matching sha256, passed checks, -ieee and
    the hand-derived package set, which equals the checker's expansion of its packages.conf; the jaxdump binary of
    global_ocean.90x40x15/code (input.dwnslp, input.idemix) as well. Negative control: a planted change of one
    hand-derived set is detected."""
    problems = []
    groups = check_build.read_groups(UPSTREAM / "pkg" / "pkg_groups")
    dirs = [(exp, tag, want) for exp, want in M3_BUILDS.items() for tag in M3_TAGS]
    dirs.append(("global_ocean.90x40x15", "463504e-jaxdump", M1_REQUESTED["global_ocean.90x40x15"]))
    for exp, tag, want in dirs:
        d = paths.REFERENCE / "bin" / f"{exp}-code-{UP7}-{tag}"
        if not d.is_dir():
            problems.append(f"missing {d}")
            continue
        problems += bin_problems(d)
        missing = {q.upper() for q in want} - check_build.enabled_in_config_h(d / "PACKAGES_CONFIG.h")
        if missing:
            problems.append(f"{d.name}: requested packages not compiled: {sorted(missing)}")
        if "-ieee" not in (d / "genmake2_command.txt").read_text():
            problems.append(f"{d.name}: not built with -ieee")
        if tag.endswith("-gcov") and "COVERAGE OK" not in (d / "COVERAGE.txt").read_text():
            problems.append(f"{d.name}: COVERAGE.txt does not say COVERAGE OK")
    for exp, want in M3_BUILDS.items():
        conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / exp / "code"])
        requested, _, _ = check_build.requested_packages(conf, groups)
        if requested != want:
            problems.append(f"{exp}: packages.conf expands to {sorted(requested ^ want)} beyond the hand-derived set")
    assert not problems, "\n".join(problems)
    conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / "vermix" / "code"])
    assert check_build.requested_packages(conf, groups)[0] != M3_BUILDS["vermix"] - {"opps"}       # planted


def test_m2_requested_package_sets():
    """The checker's expansion of every M2 packages.conf equals the hand-derived set; a planted change of one
    hand-derived set is detected (negative control)."""
    groups = check_build.read_groups(UPSTREAM / "pkg" / "pkg_groups")
    for exp, code in M2_BUILDS:
        conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / exp / code])
        requested, _, _ = check_build.requested_packages(conf, groups)
        assert requested == M2_REQUESTED[exp], (exp, sorted(requested ^ M2_REQUESTED[exp]))
    for (exp, code), (_, want, _) in M2B_BUILDS.items():
        conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / exp / code])
        requested, _, _ = check_build.requested_packages(conf, groups)
        assert requested == want, (exp, code, sorted(requested ^ want))
    conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / "global_ocean.cs32x15" / "code"])
    requested, _, _ = check_build.requested_packages(conf, groups)
    assert "cal" not in requested and requested != M2_REQUESTED["global_ocean.cs32x15"] | {"cal"}


def test_bin_problems_negative_controls(tmp_path):
    d = tmp_path / "x-code-63cdc0b-0000000"
    (d / "logs").mkdir(parents=True)
    (d / "options").mkdir()
    (d / "mitgcmuv").write_bytes(b"binary")
    (d / "sha256").write_text(hashlib.sha256(b"binary").hexdigest() + "  mitgcmuv\n")
    for f, v in (("packages_check_after_make.txt", "PACKAGES OK"), ("flags_check.txt", "FLAGS OK"),
                 ("forward_check.txt", "FORWARD OK")):
        (d / f).write_text(v + "\n")
    for f in ("build.log", "genmake2_command.txt", "genmake_state", "provenance.txt", "PACKAGES_CONFIG.h",
              "AD_CONFIG.h", "logs/make.log", "options/SOURCES.txt", "options/CPP_OPTIONS.h.dM"):
        (d / f).write_text("x\n")
    assert bin_problems(d) == []
    (d / "sha256").write_text("0" + hashlib.sha256(b"binary").hexdigest()[1:] + "  mitgcmuv\n")
    assert any("sha256 differs" in p for p in bin_problems(d))
    (d / "sha256").write_text(hashlib.sha256(b"binary").hexdigest() + "  mitgcmuv\n")
    (d / "forward_check.txt").write_text("FAIL: AD_CONFIG.h defines ALLOW_ADJOINT_RUN\nFORWARD BAD\n")
    assert any("FORWARD OK" in p for p in bin_problems(d))


# --- make_rundir vs testreport -------------------------------------------------------------------------------

def make_rundir(*args, upstream=UPSTREAM, runs):
    return subprocess.run([sys.executable, str(REPO / "reference" / "make_rundir.py"), *args,
                           "--upstream", str(upstream), "--runs", str(runs)], capture_output=True, text=True)


def links_of(rundir):
    return {n: os.readlink(rundir / n) for n in os.listdir(rundir) if (rundir / n).is_symlink()}


def run_testreport_linkdata(tmp, experiment, dirs, run_name, mpi=0):
    """Run testreport's own linkdata function (verification/testreport:771-834, cut from the file at `pinned`; it
    ends with ./prepare_run) with testreport's defaults MPI=0 (or MPI=`mpi`), MULTI_THREAD=f, in a mirror of
    verification/; return {name: link target}."""
    text = _git_show("verification/testreport").decode()
    func = re.search(r"^linkdata\(\)\n\{\n.*?^\}\n", text, re.S | re.M).group(0)
    mirror = tmp / "verification"
    mirror.mkdir(parents=True)
    for sib in os.listdir(VERIF):
        if sib != experiment and (VERIF / sib).is_dir():
            (mirror / sib).symlink_to(VERIF / sib)
    (mirror / experiment).mkdir()
    for d in dirs:
        (mirror / experiment / d).symlink_to(VERIF / experiment / d)
    rundir = mirror / experiment / run_name
    rundir.mkdir()
    script = f"MPI={mpi}\nMULTI_THREAD=f\n{func}\nlinkdata {rundir} {' '.join(dirs)}\n"
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    return links_of(rundir)


# literal expectations, derived by hand from testreport's rules and the upstream directories at 63cdc0b
NLFS = {**{n: f"../input.nlfs/{n}" for n in ("check_conserve_TS.txt", "data", "data.diagnostics", "data.pkg",
                                              "eedata", "eedata.mth", "grph_StD_AB.m")},
        **{n: f"../input/{n}" for n in ("bathy_slope.bin", "gendata.m", "Tini_G.bin", "tr_checklist", "Udiv.bin",
                                        "Uvel.bin")}}
LATLON_BINS = ("bathymetry.bin", "lev_s.bin", "lev_sss.bin", "lev_sst.bin", "lev_t.bin", "ncep_emp.bin",
               "ncep_qnet.bin", "trenberth_taux.bin", "trenberth_tauy.bin")
OPTIM = {**{n: f"../input_ad/{n}" for n in ("cycsh", "data", "data.autodiff", "data.cost", "data.ctrl",
                                             "data.gmredi", "data.grdchk", "data.optim", "data.pkg", "eedata",
                                             "Err_hflux.bin", "Err_levitus_15layer.bin", "lev_t_an.bin",
                                             "prepare_run")},
         **{n: f"../../tutorial_global_oce_latlon/input/{n}" for n in LATLON_BINS},
         "ones_64b.bin": "../../isomip/input_ad/ones_64b.bin"}


@pytest.mark.parametrize("experiment,input_dir,dirs,run_name,literal", [
    ("advect_xz", "input.nlfs", ["input.nlfs", "input"], "tr_run.nlfs", NLFS),
    ("tutorial_global_oce_optim", "input_ad", ["input_ad"], "run", OPTIM),
])
def test_make_rundir_matches_testreport(tmp_path, experiment, input_dir, dirs, run_name, literal):
    res = make_rundir(experiment, input_dir, "--run-id", "t", runs=tmp_path / "runs")
    assert res.returncode == 0, res.stdout + res.stderr
    top = tmp_path / "runs" / experiment / input_dir / "t"
    rundir = top / "rundir"
    assert (top / "READY").is_file() and rundir.resolve() == (top / "verification" / experiment / run_name)
    ours = links_of(rundir)
    assert ours == literal
    # testreport's linkdata itself ends with ./prepare_run (testreport:829-831)
    assert ours == run_testreport_linkdata(tmp_path / "tr", experiment, dirs, run_name)
    assert all((rundir / n).resolve().is_file() for n in ours)
    man = json.loads((top / "MANIFEST.json").read_text())
    assert set(man["entries"]) == set(literal) and man["linkdata_dirs"] == dirs
    # an existing run directory is refused, nothing overwritten
    again = make_rundir(experiment, input_dir, "--run-id", "t", runs=tmp_path / "runs")
    assert again.returncode != 0 and "exists" in again.stderr


def test_make_rundir_mpi_matches_testreport(tmp_path):
    """--mpi (plan Task 3b: second tiling of global_ocean with its SIZE.h_mpi tile shape in one process) gives the run
    directory of testreport's linkdata with MPI=1: data.exch2.mpi linked as data.exch2 and under its own name."""
    res = make_rundir("global_ocean.90x40x15", "input", "--run-id", "t", "--mpi", runs=tmp_path / "runs")
    assert res.returncode == 0, res.stdout + res.stderr
    top = tmp_path / "runs" / "global_ocean.90x40x15" / "input" / "t"
    ours = links_of(top / "rundir")
    assert ours["data.exch2"] == "../input/data.exch2.mpi" and ours["data.exch2.mpi"] == "../input/data.exch2.mpi"
    assert ours == run_testreport_linkdata(tmp_path / "tr", "global_ocean.90x40x15", ["input"], "run", mpi=1)
    # negative control: without --mpi there is no data.exch2 (testreport's MPI=0 run directory)
    res = make_rundir("global_ocean.90x40x15", "input", "--run-id", "u", runs=tmp_path / "runs")
    plain = links_of(tmp_path / "runs" / "global_ocean.90x40x15" / "input" / "u" / "rundir")
    assert "data.exch2" not in plain and plain != ours
    assert plain == run_testreport_linkdata(tmp_path / "tr0", "global_ocean.90x40x15", ["input"], "run", mpi=0)
    assert json.loads((top / "MANIFEST.json").read_text())["mpi_linkdata"] is True


def test_linkdata_plan_is_the_links_made(tmp_path):
    """linkdata_plan (pure, for mitjax/config) lists exactly the links make_rundir makes, in both linkdata modes."""
    mr = _load("make_rundir")
    for exp, inp, mpi in (("advect_xz", "input.nlfs", False), ("global_ocean.90x40x15", "input", True),
                          ("tutorial_global_oce_optim", "input_ad", False)):
        dirs, _ = mr.input_dirs(inp)
        plan = mr.linkdata_plan(VERIF / exp, dirs, mpi)
        args = (exp, inp, "--run-id", "p") + (("--mpi",) if mpi else ())
        res = make_rundir(*args, runs=tmp_path / exp)
        assert res.returncode == 0, res.stdout + res.stderr
        made = links_of(tmp_path / exp / exp / inp / "p" / "rundir")
        planned = {n: t for n, t, _ in plan}
        # prepare_run adds links of its own (optim, global_ocean: *.bin of other experiments); every planned link
        # is there with its target, and nothing else comes from the input directories
        assert all(made[n] == t for n, t in planned.items()) and len(planned) == len(plan)
        extra = set(made) - set(planned)
        assert all(made[n].startswith("../../") for n in extra)
        assert (extra == set()) == (exp == "advect_xz")


def test_make_rundir_overlay(tmp_path):
    """--set writes the overlaid namelist file instead of the link; refuses ambiguous or impossible overlays."""
    res = make_rundir("tutorial_baroclinic_gyre", "input", "--run-id", "o", "--set",
                      "data.pkg:PACKAGES:useMNC=.FALSE.", "--set", "eedata:EEPARMS:debugMode=.TRUE.",
                      runs=tmp_path / "runs")
    assert res.returncode == 0, res.stdout + res.stderr
    top = tmp_path / "runs" / "tutorial_baroclinic_gyre" / "input" / "o"
    rundir = top / "rundir"
    assert not (rundir / "data.pkg").is_symlink() and not (rundir / "eedata").is_symlink()
    pkg = (rundir / "data.pkg").read_text()
    src = (VERIF / "tutorial_baroclinic_gyre" / "input" / "data.pkg").read_text()
    assert pkg == src.replace(" useMNC=.TRUE.,", " useMNC=.FALSE.,") and pkg != src
    ee = (rundir / "eedata").read_text().split("\n")
    i = next(k for k, ln in enumerate(ee) if ln.strip() == "&EEPARMS")
    assert ee[i + 1] == " debugMode=.TRUE.,"
    man = json.loads((top / "MANIFEST.json").read_text())
    assert [o["key"] for o in man["overlays"]] == ["useMNC", "debugMode"]
    assert man["entries"]["data.pkg"]["linked_by"] == "overlay input"
    assert "OVERLAY.txt" in os.listdir(top)
    # refusals: a group the file does not have; a file linkdata does not link
    bad = make_rundir("tutorial_baroclinic_gyre", "input", "--run-id", "x", "--set", "data.pkg:NOPE:a=1",
                      runs=tmp_path / "runs")
    assert bad.returncode != 0 and "NOPE" in bad.stderr
    bad = make_rundir("tutorial_baroclinic_gyre", "input", "--run-id", "y", "--set", "data.nofile:G:a=1",
                      runs=tmp_path / "runs")
    assert bad.returncode != 0 and "not linked" in bad.stderr
    mr = _load("make_rundir")
    with pytest.raises(SystemExit, match="assigns more than"):
        mr.overlay_namelist(" &G\n a=1, b=2,\n &\n", "G", "a", "3")
    assert mr.overlay_namelist(" &G\n a=1,\n#a=5,\n &\n", "G", "a", "3")[0] == " &G\n a=3,\n#a=5,\n &\n"


def _fake_upstream(tmp, experiment, drop=(), skip_dirs=()):
    """A verification tree of symlinks to the real one, with some input files or whole experiments left out."""
    up = tmp / "up"
    for sib in os.listdir(VERIF):
        if sib in skip_dirs or not (VERIF / sib).is_dir():
            continue
        if sib != experiment:
            (up / "verification").mkdir(parents=True, exist_ok=True)
            (up / "verification" / sib).symlink_to(VERIF / sib)
            continue
        for d in os.listdir(VERIF / sib):
            if d.startswith("input"):
                (up / "verification" / sib / d).mkdir(parents=True)
                for f in os.listdir(VERIF / sib / d):
                    if f not in drop:
                        (up / "verification" / sib / d / f).symlink_to(VERIF / sib / d / f)
    return up


def test_make_rundir_refuses_missing_input(tmp_path):
    # positive: the full tree gives a READY directory
    up = _fake_upstream(tmp_path / "a", "tutorial_barotropic_gyre")
    res = make_rundir("tutorial_barotropic_gyre", "input", "--run-id", "t", upstream=up, runs=tmp_path / "ra")
    assert res.returncode == 0, res.stdout + res.stderr
    # planted: bathy.bin (bathyFile in data) missing from input/
    up = _fake_upstream(tmp_path / "b", "tutorial_barotropic_gyre", drop=("bathy.bin",))
    res = make_rundir("tutorial_barotropic_gyre", "input", "--run-id", "t", upstream=up, runs=tmp_path / "rb")
    top = tmp_path / "rb" / "tutorial_barotropic_gyre" / "input" / "t"
    assert res.returncode == 1 and "missing input bathy.bin" in res.stdout
    assert (top / "REFUSED.txt").is_file() and not (top / "READY").exists()
    # planted: prepare_run's source experiment absent (it only prints an error and exits 0): bathymetry.bin,
    # ones_64b.bin etc. never arrive, the namelist check refuses the run
    # (since lane A session 6 refused already on prepare_run's own `Error:` line, before the namelist check)
    up = _fake_upstream(tmp_path / "c", "tutorial_global_oce_optim", skip_dirs=("tutorial_global_oce_latlon",))
    res = make_rundir("tutorial_global_oce_optim", "input_ad", "--run-id", "t", upstream=up, runs=tmp_path / "rc")
    assert res.returncode == 1 and "prepare_run reported" in res.stdout and "not a directory" in res.stdout
    # the pickup of a nIter0 > 0 start is required too (global_ocean.90x40x15: nIter0=36000)
    up = _fake_upstream(tmp_path / "d", "global_ocean.90x40x15", drop=("pickup.0000036000",))
    res = make_rundir("global_ocean.90x40x15", "input", "--run-id", "t", upstream=up, runs=tmp_path / "rd")
    assert res.returncode == 1 and "missing input pickup.0000036000" in res.stdout


def _tiny_upstream(tmp, prepare, extra=None):
    """verification/x/{input,input.v,input.y}: input.v/prepare_run = `prepare`; input/a.gz = gzip of b"hello";
    input.y/z (a sibling directory a prepare_run may read)."""
    import gzip
    up = tmp / "up"
    for d in ("input", "input.v", "input.y"):
        (up / "verification" / "x" / d).mkdir(parents=True)
    (up / "verification" / "x" / "input" / "data").write_text(" &PARM01\n &\n")
    (up / "verification" / "x" / "input" / "a.gz").write_bytes(gzip.compress(b"hello"))
    (up / "verification" / "x" / "input.y" / "z").write_text("z\n")
    pr = up / "verification" / "x" / "input.v" / "prepare_run"
    pr.write_text("#!/bin/sh\n" + prepare + "\n")
    pr.chmod(0o755)
    return up


def test_make_rundir_never_removes(tmp_path):
    """prepare_run never removes anything: gunzip runs with --keep (shim; the .gz link stays, the decompressed file is
    the same), rm/mv are refused before the script runs, an `Error:` line refuses the directory; sibling input
    directories of the experiment are reachable (global_ocean.cs32x15/input.in_p reads ../input.seaice)."""
    up = _tiny_upstream(tmp_path / "a", "gunzip -f a.gz\nln -sf ../input.y/z z")
    res = make_rundir("x", "input.v", "--run-id", "t", upstream=up, runs=tmp_path / "ra")
    top = tmp_path / "ra" / "x" / "input.v" / "t"
    assert res.returncode == 0, res.stdout + res.stderr
    rd = top / "rundir"
    assert (rd / "a").read_bytes() == b"hello" and (rd / "a.gz").is_symlink() and (rd / "z").read_text() == "z\n"
    man = json.loads((top / "MANIFEST.json").read_text())
    assert man["prepare_run"]["shims"] and "--keep" in man["prepare_run"]["shims"][0]
    # planted: an rm in prepare_run -> refused before it runs (the link it would remove is still there)
    up = _tiny_upstream(tmp_path / "b", "rm -f a.gz")
    res = make_rundir("x", "input.v", "--run-id", "t", upstream=up, runs=tmp_path / "rb")
    top = tmp_path / "rb" / "x" / "input.v" / "t"
    assert res.returncode == 1 and "removing commands" in res.stdout and (top / "rundir" / "a.gz").is_symlink()
    assert not (top / "prepare_run.log").exists() and not (top / "READY").exists()
    # planted: a prepare_run that reports a missing source directory and exits 0
    up = _tiny_upstream(tmp_path / "c", "echo ' Error: ../input.q not a directory'")
    res = make_rundir("x", "input.v", "--run-id", "t", upstream=up, runs=tmp_path / "rc")
    assert res.returncode == 1 and "prepare_run reported" in res.stdout
    # the after-the-fact check (a linkdata entry missing or replaced after prepare_run) is planted as a pure function
    # in test_make_rundir_removal_check_pure: no file is removed to test it


def test_make_rundir_removal_check_pure():
    """make_rundir.removed_by_prepare_run on fabricated before/after listings (nothing on disk, nothing removed):
    an unchanged listing and added entries pass; a missing entry, a link with another target, a link that became a
    regular file and a regular file that became a link are each reported."""
    mr = _load("make_rundir")
    before = {"data": "../input/data", "eedata": "../input_ad/eedata", "data.pkg": None, "bathy.bin": "../input/b.bin"}
    assert mr.removed_by_prepare_run(before, dict(before)) == []
    assert mr.removed_by_prepare_run(before, dict(before, extra="../../x/y", **{"grid.face001.bin": "../g"})) == []
    for planted, want in (
            ({k: v for k, v in before.items() if k != "data"}, ["data"]),              # removed
            (dict(before, eedata="../input/eedata"), ["eedata"]),                        # relinked elsewhere
            (dict(before, **{"bathy.bin": None}), ["bathy.bin"]),                        # link -> regular file
            (dict(before, **{"data.pkg": "../input/data.pkg"}), ["data.pkg"]),          # file (overlay) -> link
            ({}, sorted(before))):
        assert mr.removed_by_prepare_run(before, planted) == want, planted
    assert mr.removed_by_prepare_run({}, {"a": None}) == []


def test_make_rundir_accepts_per_tile_pickup(tmp_path):
    """A restart from a pickup written with globalFiles=F (per-tile files <name>.001.001.data, MDS_READ_FIELD's
    globalFile=F branch, mdsio_read_field.F:446-454) is accepted; a malformed tile suffix, no pickup at all, or a
    per-tile copy of a *File input (read only as a global file) is refused."""
    blob = tmp_path / "blob"
    blob.write_bytes(b"\0" * 16)
    base = ("tutorial_barotropic_gyre", "input", "--set", "data:PARM03:nIter0=5")
    res = make_rundir(*base, "--run-id", "ok", "--copy", f"{blob}:pickup.0000000005.001.001.data",
                      runs=tmp_path / "runs")
    assert res.returncode == 0, res.stdout + res.stderr
    for rid, extra in (("none", ()), ("bad", ("--copy", f"{blob}:pickup.0000000005.001.data"))):
        res = make_rundir(*base, "--run-id", rid, *extra, runs=tmp_path / "runs")
        assert res.returncode == 1 and "missing input pickup.0000000005" in res.stdout, (rid, res.stdout)
    up = _fake_upstream(tmp_path / "f", "tutorial_barotropic_gyre", drop=("bathy.bin",))
    res = make_rundir("tutorial_barotropic_gyre", "input", "--run-id", "t", "--copy",
                      f"{blob}:bathy.bin.001.001.data", upstream=up, runs=tmp_path / "rf")
    assert res.returncode == 1 and "missing input bathy.bin" in res.stdout



def test_make_rundir_copy_into_ctrl_dir(tmp_path):
    """--copy NAME may be <dir>/<file> (lab_sea/input_ad*: ctrlDir='./ctrl_variables'); the directory is made and the
    MANIFEST lists the copy under that name; deeper paths, '..' and '.' components are refused."""
    blob = tmp_path / "blob"
    blob.write_bytes(b"\1" * 16)
    res = make_rundir("tutorial_barotropic_gyre", "input", "--run-id", "ok", "--copy",
                      f"{blob}:ctrl_variables/adxx_atemp.0000000000.001.001.data", runs=tmp_path / "runs")
    assert res.returncode == 0, res.stdout + res.stderr
    top = tmp_path / "runs" / "tutorial_barotropic_gyre" / "input" / "ok"
    man = json.loads((top / "MANIFEST.json").read_text())
    assert [c["name"] for c in man["copies"]] == ["ctrl_variables/adxx_atemp.0000000000.001.001.data"]
    f = (top / "rundir").resolve() / "ctrl_variables" / "adxx_atemp.0000000000.001.001.data"
    assert f.is_file() and not f.is_symlink() and f.read_bytes() == blob.read_bytes()
    for k, bad in enumerate(("a/b/c.data", "../x.data", "./x.data", "d/", "/abs.data")):
        res = make_rundir("tutorial_barotropic_gyre", "input", "--run-id", f"bad{k}", "--copy", f"{blob}:{bad}",
                          runs=tmp_path / "runs")
        assert res.returncode != 0 and "bad --copy" in res.stdout + res.stderr, (bad, res.stdout, res.stderr)


# --- build checks ------------------------------------------------------------------------------------------

def _fake_build(tmp, conf, enabled, ad_define=False, fflags="-fconvert=big-endian -ffp-contract=off", foptim="-O0"):
    b = tmp / "bld"
    b.mkdir(parents=True)
    if conf is not None:
        (b / "packages.conf").write_text(conf)
    up = " ".join(f"-DALLOW_{p.upper()}" for p in sorted(enabled))
    (b / "Makefile").write_text(
        f"ENABLED_PACKAGES = {up}\nDISABLED_PACKAGES = -UALLOW_KPP\nFFLAGS = {fflags}\nFOPTIM = {foptim} \n"
        f"F90FLAGS = {fflags}\nF90OPTIM = {foptim}\nCFLAGS = -O0\nDEFINES = -DWORDLENGTH=4 -DHAVE_NETCDF\n"
        "CPPCMD = cat $< | cpp -traditional -P $(DEFINES) $(INCLUDES) | sed 's/x/y/'\n")
    (b / "PACKAGES_CONFIG.h").write_text("".join(f"#define ALLOW_{p.upper()}\n" for p in sorted(enabled)))
    (b / "AD_CONFIG.h").write_text("#define ALLOW_ADJOINT_RUN\n" if ad_define else
                                   "#undef ALLOW_ADJOINT_RUN\n#undef ALLOW_TANGENTLINEAR_RUN\n")
    return b


def test_requested_package_sets():
    """The checker's expansion of every M1 package list equals the hand-derived set."""
    groups = check_build.read_groups(UPSTREAM / "pkg" / "pkg_groups")
    for exp, code in M1_BUILDS:
        conf = check_build.find_packages_conf(Path("/nonexistent"), [VERIF / exp / code])
        requested, _, _ = check_build.requested_packages(conf, groups)
        assert requested == M1_REQUESTED[exp], exp


def test_build_checks_negative_controls(tmp_path, capsys):
    root = UPSTREAM
    conf = (VERIF / "tutorial_baroclinic_gyre" / "code" / "packages.conf").read_text()
    full = M1_REQUESTED["tutorial_baroclinic_gyre"]
    assert check_build.check_packages(_fake_build(tmp_path / "ok", conf, full), root, [])
    # planted: genmake2 dropped mnc (what it does when its NetCDF test fails)
    assert not check_build.check_packages(_fake_build(tmp_path / "mnc", conf, full - {"mnc"}), root, [])
    assert "requested but not in ENABLED_PACKAGES: MNC" in capsys.readouterr().out
    # dependency rules may add packages: not a failure
    assert check_build.check_packages(_fake_build(tmp_path / "dep", conf, full | {"timeave"}), root, [])
    # planted: the default package list (no packages.conf) with one package missing
    assert not check_build.check_packages(_fake_build(tmp_path / "def", None, GFD - {"monitor"}), root, [])
    # flags: good, fast-math, FMA contraction allowed, optimised IEEE build
    assert check_build.check_flags(_fake_build(tmp_path / "f0", conf, full), ieee=True)
    assert not check_build.check_flags(_fake_build(tmp_path / "f1", conf, full,
                                                   fflags="-fconvert=big-endian -ffp-contract=off -ffast-math"), True)
    assert not check_build.check_flags(_fake_build(tmp_path / "f2", conf, full, fflags="-fconvert=big-endian"), True)
    assert not check_build.check_flags(_fake_build(tmp_path / "f3", conf, full, foptim="-O3 -funroll-loops"), True)
    # forward-only: an adjoint AD_CONFIG.h is caught
    assert check_build.check_forward(_fake_build(tmp_path / "w0", conf, full))
    assert not check_build.check_forward(_fake_build(tmp_path / "w1", conf, full, ad_define=True))
