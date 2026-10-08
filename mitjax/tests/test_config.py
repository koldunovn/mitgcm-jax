"""Plan Task 6: the experiment configuration reader (mitjax/config, mitjax/io/namelist.py, mitjax/params_io.py).

Gates: every M1 variant parses; PACKAGES_CONFIG.h, AD_CONFIG.h and every *OPTIONS.h's `cpp -dM` equal the frozen
oracle build records byte for byte; the parameters set in the files and a set of cited defaults equal the oracle's
STDOUT printout wherever an oracle run exists (STDOUT_PENDING lists the variants still without one, strictly);
the namelist cases of the M1 inputs. Negative controls: a planted wrong default, a planted wrong value, a planted
wrong package list, a planted extra macro, an unknown namelist variable, an unknown or retired CPP option, an
unported package switched on, a parser that loses an assignment.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.config import cpp_options, namelists, packages, params
from mitjax.io import namelist as nlio
from mitjax.io import stdout as stdout_io
from mitjax.params_io import CitedDefault, DeadCitation, RunParams, fortran_default, fortran_literal

U = paths.UPSTREAM
BUILDS = [("tutorial_barotropic_gyre", "code"), ("tutorial_baroclinic_gyre", "code"), ("advect_xy", "code"),
          ("advect_xz", "code"), ("global_ocean.90x40x15", "code"), ("tutorial_global_oce_optim", "code_ad")]
VARIANTS = [("tutorial_barotropic_gyre", "input"), ("tutorial_baroclinic_gyre", "input"), ("advect_xy", "input"),
            ("advect_xy", "input.ab3_c4"), ("advect_xz", "input"), ("advect_xz", "input.nlfs"),
            ("advect_xz", "input.pqm"), ("global_ocean.90x40x15", "input"), ("tutorial_global_oce_optim", "input_ad")]
RECORD = "{exp}-{code}-63cdc0b-f48c062"          # lane A's frozen builds (build job 27826031)
# Variants whose oracle STDOUT does not exist yet. Strict: a variant listed here that has a run fails the test, so
# the list is kept current. Empty since lane D's run job 27826871 (plain binaries of build job 27826031).
STDOUT_PENDING = set()
# Cited defaults of parameters the barotropic gyre does not set, checked against its printout.
DEFAULTS = [("model/src/set_defaults.F:101", "gravity"), ("model/src/set_defaults.F:104", "rhoNil"),
            ("model/src/set_defaults.F:132", "no_slip_sides"), ("model/src/set_defaults.F:135", "sideDragFactor"),
            ("model/src/set_defaults.F:140", "viscA4"), ("model/src/set_defaults.F:174", "HeatCapacity_Cp"),
            ("model/src/set_defaults.F:180", "hFacMin"), ("model/src/set_defaults.F:252", "implicSurfPress")]

_CACHE = {}


def experiment(exp, inp):
    if (exp, inp) not in _CACHE:
        _CACHE[(exp, inp)] = params.load(exp, inp)
    return _CACHE[(exp, inp)]


def oracle_stdout(exp, inp):
    """output.txt of an oracle run of the plain binary (run ids job<N> or job<N>-plain only, newest first: the -jd*,
    -debug, -retile and -restartA/B runs change the build or the namelists) whose parameter
    printout is complete: READY and "// CONFIG_CHECK : Normal End" (config_check.F, printed after the parameter
    summary). The run need not end normally: the forward-only tutorial_global_oce_optim/code_ad build stops in
    GRDCHK_MAIN (MDS_READ_FIELD of adxx_qnet, which only an adjoint build writes; run jobs 27826794, 27826871),
    after the printout."""
    runs = sorted((p for p in (paths.REFERENCE_RUNS / exp / inp).glob("job*/rundir/output.txt")
                   if re.fullmatch(r"job\d+(-plain)?", p.parent.parent.name)), reverse=True)
    for out in runs:
        if (out.parent.parent / "READY").exists() and "// CONFIG_CHECK : Normal End" in out.read_text(
                encoding="latin-1"):
            return out
    return None


# ---------------------------------------------------------------------------------------------------------------
# namelist parser

def test_namelist_terminators_and_literals():
    text = ("# comment in column 1\n &G1\n a=1, b = 2.5,\n /\n"          # `/`
            " &G2\n c = 1.d-4, d=.TRUE., e=T,\n &\n"                        # ` &` (nml_change_syntax.F:72-77)
            " &G3\n f = 'x''y', g = \"dq\",\n &end\n"                       # `&end`
            " $G4\n h = 3*7, i = 2*, j = 4\n $end\n")
    nl = nlio.parse_namelist(text)
    assert [g.name for g in nl.groups] == ["g1", "g2", "g3", "g4"]
    v = {a.key: a.values for a in nl.assignments()}
    assert v["a"] == (1,) and v["b"] == (2.5,) and v["c"] == (1e-4,) and v["d"] == (True,) and v["e"] == (True,)
    assert v["f"] == ("x'y",) and v["g"] == ("dq",) and v["h"] == (7, 7, 7) and v["i"] == (nlio.NULL, nlio.NULL)
    # a column-1 '#' is a comment, an indented one is not (open_copy_data_file.F:137): the READ would fail
    with pytest.raises(ValueError):
        nlio.parse_namelist(" &G\n a=1,\n  # not a comment\n /\n")
    # '  &' (three characters) is not converted to a terminator
    with pytest.raises(ValueError):
        nlio.parse_namelist(" &G\n a=1,\n  &\n")
    with pytest.raises(ValueError, match="MAX_LEN_PREC"):
        nlio.parse_namelist(" &G\n a=" + "1," * 120 + "\n /\n")


def test_namelist_key_count_crosscheck_bites(monkeypatch):
    """L-LIT-11: the independent `name =` count catches a parser that loses assignments (planted: a value reader
    that swallows the rest of a group, as the first ECCO parser did with `&`)."""
    text = " &EEPARMS\n nTx=1,\n nTy=1,\n &\n"
    assert len(nlio.parse_namelist(text).assignments()) == 2
    real = nlio._values

    def lossy(sc):
        vals, term = real(sc)
        if not term:
            j = sc.text.find("/", sc.pos)
            sc.pos = j + 1
            return vals, True
        return vals, term
    monkeypatch.setattr(nlio, "_values", lossy)
    with pytest.raises(ValueError, match="parsed 1 assignments but the text has 2"):
        nlio.parse_namelist(text)


def test_namelist_cases_from_m1_inputs():
    e = experiment("tutorial_barotropic_gyre", "input")
    dx = e.run.var("data", "PARM04", "delX")                  # data:40 delX=62*20.E3
    assert dx.shape == (62,) and dx.value == {(i,): 20.e3 for i in range(1, 63)}
    assert e.run.var("data", "PARM05", "meridWindFile").value is nlio.NULL     # data:52 meridWindFile=,
    with pytest.raises(KeyError):
        RunParams(e.run).get("data", "PARM05", "meridWindFile")
    g = experiment("global_ocean.90x40x15", "input")
    tref = g.run.var("data", "PARM01", "tRef")                # data:7 tRef = 15*20.
    assert tref.value == {(k,): 20.0 for k in range(1, 16)}
    df = g.run.var("data", "PARM03", "dumpFreq")              # data:62-63, duplicated: last wins
    assert [a.line for a in df.assignments] == [62, 63] and df.value == 864000.0
    b = experiment("tutorial_baroclinic_gyre", "input")
    f = b.run.var("data.diagnostics", "DIAGNOSTICS_LIST", "fields")      # data.diagnostics:21 fields(1:3,1)
    assert f.value[(1, 1)].strip() == "ETAN" and f.value[(3, 1)].strip() == "MXLDEPTH" and (4, 1) not in f.value
    o = experiment("tutorial_global_oce_optim", "input_ad")
    xf = o.run.var("data.ctrl", "CTRL_NML_GENARR", "xx_gentim2d_file")   # data.ctrl:19 xx_gentim2d_file(1)
    assert xf.value[(1,)] == "xx_qnet" and len(xf.value) == 1
    assert o.run.var("data.grdchk", "GRDCHK_NML", "grdchkvarname").value == "xx_qnet"   # double-quoted
    # key counts: every `name =` of every file was parsed (the check is inside parse_namelist)
    for exp, inp in VARIANTS:
        for name, nl in experiment(exp, inp).run.parsed.items():
            assert nl.assignments() or name in ("data.pkg", "data.sbo", "data.autodiff", "data.ctrl"), name


def test_unknown_namelist_variable_and_wrong_type_raise():
    e = experiment("tutorial_barotropic_gyre", "input")
    rds = namelists.file_readers(e.farm, e.toolchain, "data", e.cfg.size.env())
    ok = nlio.parse_namelist(" &PARM01\n viscAh=4.E2,\n &\n")
    vs, _ = namelists.resolve_file("data", "planted", ok, rds)
    assert vs[("data", "parm01", "viscah")].value == 400.0
    with pytest.raises(namelists.UnknownNamelistVariable, match="viscAhh"):
        namelists.resolve_file("data", "planted", nlio.parse_namelist(" &PARM01\n viscAhh=4.E2,\n &\n"), rds)
    with pytest.raises(ValueError, match="INTEGER"):
        namelists.resolve_file("data", "planted", nlio.parse_namelist(" &PARM03\n nIter0=1.5,\n &\n"), rds)
    # an integer literal for a REAL is fine (the READ converts it)
    vs, _ = namelists.resolve_file("data", "planted", nlio.parse_namelist(" &PARM01\n rhoConst=1000,\n &\n"), rds)
    assert vs[("data", "parm01", "rhoconst")].value == 1000.0 and isinstance(vs[("data", "parm01", "rhoconst")].value,
                                                                              float)


# ---------------------------------------------------------------------------------------------------------------
# packages and CPP options against the frozen build records

def test_packages_config_equals_build_records():
    for exp, code in BUILDS:
        rec = paths.REFERENCE / "bin" / RECORD.format(exp=exp, code=code)
        ps = packages.package_set(U, [U / "verification" / exp / code])
        assert packages.packages_config_h(U, ps) == (rec / "PACKAGES_CONFIG.h").read_text(), exp
        assert packages.ad_config_h(U) == (rec / "AD_CONFIG.h").read_text(), exp
    # planted: one package fewer (gmredi dropped from global_ocean) must differ
    ps = packages.package_set(U, [U / "verification" / "global_ocean.90x40x15" / "code"])
    planted = packages.PackageSet(ps.source, ps.words, ps.expanded, ps.disable,
                                  tuple(p for p in ps.packages if p != "gmredi"), ps.added_by_rules, ps.all_dirs)
    rec = paths.REFERENCE / "bin" / RECORD.format(exp="global_ocean.90x40x15", code="code")
    assert packages.packages_config_h(U, planted) != (rec / "PACKAGES_CONFIG.h").read_text()


def test_pkg_groups_and_depend_literal():
    groups = packages.read_pkg_groups(U / "pkg" / "pkg_groups")
    words, hit = packages.expand_pkg_groups(["default_pkg_list"], groups)
    assert hit and words == ["gfd"]
    words, hit = packages.expand_pkg_groups(words, groups)
    assert words == ["mom_common", "mom_fluxform", "mom_vecinv", "generic_advdiff", "debug", "mdsio", "rw", "monitor"]
    rules = packages.read_pkg_depend(U / "pkg" / "pkg_depend")
    assert ("model", "+rw") in rules and ("grdchk", "+ctrl") in rules


def test_cpp_options_equal_build_records():
    tc = cpp_options.oracle_toolchain()        # the records hold the oracle compiler's own predefined macros too
    for exp, code in BUILDS:
        rec = paths.REFERENCE / "bin" / RECORD.format(exp=exp, code=code) / "options"
        e = experiment(exp, [v for x, v in VARIANTS if x == exp][0])
        assert cpp_options.option_headers(e.farm) == sorted(p.name[:-3] for p in rec.glob("*.dM")), exp
        for h in cpp_options.option_headers(e.farm):
            assert cpp_options.run_cpp_dM(e.farm, tc, h) == (rec / f"{h}.dM").read_text(), (exp, h)
    # planted: one extra macro on the command line changes the dump (the comparison bites)
    e = experiment("tutorial_barotropic_gyre", "input")
    bad = cpp_options.Toolchain(tc.cpp, tc.defines + ("-DALLOW_AUTODIFF",), tc.includes, "planted")
    rec = paths.REFERENCE / "bin" / RECORD.format(exp="tutorial_barotropic_gyre", code="code") / "options"
    assert cpp_options.run_cpp_dM(e.farm, bad, "CPP_OPTIONS.h") != (rec / "CPP_OPTIONS.h.dM").read_text()


def test_cpp_flags_unknown_and_retired_options_raise():
    b = experiment("tutorial_barotropic_gyre", "input").cfg
    o = experiment("tutorial_global_oce_optim", "input_ad").cfg
    assert b.cpp.ALLOW_AUTODIFF is False and o.cpp.ALLOW_AUTODIFF is True
    assert b.cpp.flag("ALLOW_SRCG") is True                    # master's default CPP_OPTIONS.h:126
    # optim's code_ad/CPP_OPTIONS.h does not include PACKAGES_CONFIG.h: the model/src prologue view has ALLOW_AUTODIFF
    assert o.cpp.flag("ALLOW_AUTODIFF", "CPP_OPTIONS.h") is False and o.cpp.flag("ALLOW_AUTODIFF") is True
    assert b.cpp.flag("ALLOW_GENERIC_ADVDIFF", "GAD_OPTIONS.h") is True
    with pytest.raises(cpp_options.UnknownCppOption):
        b.cpp.ALLOW_AUTODIF                                     # typo: tested by no source
    with pytest.raises(cpp_options.UnknownCppOption, match="retired"):
        b.cpp.EXACT_CONSERV
    hash(b)                                                     # static configuration is hashable


# ---------------------------------------------------------------------------------------------------------------
# all variants, SIZE.h, package switches, params pytree

def test_all_m1_variants_parse():
    want = {"tutorial_barotropic_gyre": (62, 62, 2, 2, 1, 1, 1), "tutorial_baroclinic_gyre": (31, 31, 2, 2, 2, 2, 15),
            "global_ocean.90x40x15": (10, 10, 3, 3, 9, 4, 15), "tutorial_global_oce_optim": (45, 20, 2, 2, 2, 2, 15)}
    for exp, inp in VARIANTS:
        e = experiment(exp, inp)
        s = e.cfg.size
        if exp in want:
            assert (s.sNx, s.sNy, s.OLx, s.OLy, s.nSx, s.nSy, s.Nr) == want[exp], exp
        assert s.Nx == s.sNx * s.nSx * s.nPx and s.Ny == s.sNy * s.nSy * s.nPy
        assert e.run.vars and not e.run.unread_groups, (exp, inp, e.run.unread_groups)
        assert set(e.run.unread_files) <= {"eedata.mth", "data.exch2.mpi"}, (exp, inp, e.run.unread_files)
    assert experiment("advect_xz", "input").cfg.size.sNy == 1        # sNy=1 < OLy


def test_link_sources_equal_make_rundir():
    """The variant overlay equals the run directory lane A's make_rundir.py made (MANIFEST.json, job 27826032)."""
    man = json.loads((paths.REFERENCE_RUNS / "tutorial_barotropic_gyre" / "input" / "job27826032" /
                      "MANIFEST.json").read_text())
    made = {n: e["linked_by"].split()[-1] for n, e in man["entries"].items() if e["linked_by"].startswith("linkdata")}
    assert namelists.link_sources(U / "verification" / "tutorial_barotropic_gyre", "input") == made
    v = namelists.link_sources(U / "verification" / "advect_xz", "input.pqm")
    assert v["data"] == "input.pqm" and v["data.pkg"] == "input"     # input.pqm has no data.pkg


def test_use_flags_and_require_ported():
    g = experiment("global_ocean.90x40x15", "input").cfg
    on = {k for k, v in g.use if v}
    assert on == {"useGAD", "useGMRedi", "useSBO", "useDiagnostics"}, on
    assert g.use_flag("useGGL90") is False and "ggl90" in g.packages     # compiled but off: fine
    params.require_ported(g, {"generic_advdiff", "gmredi", "sbo", "diagnostics"})
    with pytest.raises(params.UnportedPackage, match="useGMRedi"):
        params.require_ported(g, {"generic_advdiff", "sbo", "diagnostics"})
    o = experiment("tutorial_global_oce_optim", "input_ad").cfg
    assert o.use_flag("useAUTODIFF") and o.use_flag("useCTRL") and o.use_flag("useGrdchk")   # packages_boot.F:168, :173, :232
    params.require_ported(experiment("tutorial_barotropic_gyre", "input").cfg, set())
    # run-time selectors (REVIEW_M0 #9): the barotropic gyre's values pass when declared; a planted unsupported value
    # (advect_xz/input.nlfs sets nonlinFreeSurf=4, select_rStar=2) raises, naming them; an undeclared selector raises
    b = experiment("tutorial_barotropic_gyre", "input")
    vals = params.selector_values(b)
    assert vals["vectorInvariantMomentum"] == (False, "model/src/set_defaults.F:193")
    supported = {k: {v} for k, (v, _) in vals.items()}
    params.require_supported(b, supported)
    with pytest.raises(params.UnsupportedOption, match=r"nonlinFreeSurf=4 .*select_rStar=2"):
        params.require_supported(experiment("advect_xz", "input.nlfs"), supported)
    with pytest.raises(params.UnsupportedOption, match="monitorFreq"):
        params.require_supported(b, {k: v for k, v in supported.items() if k != "monitorFreq"})


def test_params_pytree_floats_traced_static_rest():
    e = experiment("tutorial_barotropic_gyre", "input")
    p = e.params
    assert p["data:parm01:viscah"] == np.float64(400.0) and p["data:parm01:viscah"].dtype == np.float64
    seen = {}

    def f(q):
        x = q["data:parm01:viscah"]
        seen["tracer"] = isinstance(x, jax.core.Tracer)
        seen["weak"] = x.aval.weak_type
        return x * 2.0
    assert float(jax.jit(f)(p)) == 800.0
    assert seen == {"tracer": True, "weak": False}
    assert all(not isinstance(v, float) or isinstance(v, np.float64) for v in p.values.values())
    assert ("data", "parm03", "niter0") in dict(e.cfg.static)
    assert jnp.asarray(p["data:parm04:delx(1)"]).dtype == jnp.float64


# ---------------------------------------------------------------------------------------------------------------
# STDOUT cross-check and defaults

def test_fortran_default_reads_the_cited_line():
    b = experiment("tutorial_barotropic_gyre", "input")
    assert fortran_default("model/src/set_defaults.F:101", "gravity", b).value == 9.81
    assert fortran_default("model/src/set_defaults.F:140", "viscA4", b).value == 0.0       # `0. _d 11`
    with pytest.raises(ValueError, match="does not assign"):                              # planted wrong citation
        fortran_default("model/src/set_defaults.F:104", "gravity", b)
    with pytest.raises(ValueError, match="not a literal"):                                # needs code, not a default
        fortran_default("model/src/set_defaults.F:290", "cg2dUseMinResSol", b)
    rp = RunParams(b.run)
    with pytest.raises(KeyError):
        rp.get("data", "PARM01", "viscA4")                                                # no Python default
    with pytest.raises(TypeError):
        rp.get("data", "PARM01", "viscA4", default=0.0)                                    # uncited default refused
    assert rp.get("data", "PARM01", "viscA4",
                  default=fortran_default("model/src/set_defaults.F:140", "viscA4", b)) == 0.0


def test_fortran_default_refuses_lines_this_build_does_not_execute():
    """REVIEW_M0 #3, #4. set_defaults.F:318-326: `#ifdef ALLOW_ADAMSBASHFORTH_3` (advect_xy/code/CPP_OPTIONS.h:100
    defines it, model/inc/CPP_OPTIONS.h:100 undefines it): each arm gives the live value in one build and is refused in
    the other. A line in a run-time IF block needs its condition stated; a default overridden before the READ is
    refused; a REAL*4 literal is its binary32 value."""
    ab3, baro = experiment("advect_xy", "input.ab3_c4"), experiment("tutorial_barotropic_gyre", "input")
    sd = "model/src/set_defaults.F"
    assert fortran_default(f"{sd}:321", "startFromPickupAB2", ab3).value is False
    assert fortran_default(f"{sd}:325", "startFromPickupAB2", baro).value is True
    assert fortran_default(f"{sd}:319", "alph_AB", ab3).value == 0.5
    with pytest.raises(DeadCitation, match=r"not live in this build \(#ifdef ALLOW_ADAMSBASHFORTH_3 at :318 NOT"):
        fortran_default(f"{sd}:321", "startFromPickupAB2", baro)
    with pytest.raises(DeadCitation, match=r"not live in this build \(#else at :322 NOT taken"):
        fortran_default(f"{sd}:325", "startFromPickupAB2", ab3)
    with pytest.raises(DeadCitation, match="not live"):
        fortran_default(f"{sd}:319", "alph_AB", baro)
    # run-time IF: w2_readparms.F:66-70 `IF ( useCubedSphereExchange ) THEN / preDefTopol = 3 / ELSE / = 1`
    g = experiment("global_ocean.90x40x15", "input")
    cond = "IF ( useCubedSphereExchange ) THEN ... ELSE"
    with pytest.raises(DeadCitation, match="inside a run-time IF block"):
        fortran_default("pkg/exch2/w2_readparms.F:69", "preDefTopol", g)
    with pytest.raises(DeadCitation, match="inside a run-time IF block"):
        fortran_default("pkg/exch2/w2_readparms.F:69", "preDefTopol", g, condition="IF ( useCubedSphereExchange ) THEN")
    d = fortran_default("pkg/exch2/w2_readparms.F:69", "preDefTopol", g, condition=cond)
    assert (d.value, d.condition) == (1, cond)
    # overridden before the READ: packages_boot.F:130 useAUTODIFF=.FALSE., :168 `useAUTODIFF=.TRUE.` (ALLOW_AUTODIFF)
    o = experiment("tutorial_global_oce_optim", "input_ad")
    with pytest.raises(DeadCitation, match=r"assigned again later .* \[168\]"):
        fortran_default("model/src/packages_boot.F:130", "useAUTODIFF", o)
    assert fortran_default("model/src/packages_boot.F:130", "useAUTODIFF", baro).value is False
    # REAL*4 literal (no -fdefault-real-8): optim_readparms.F:87 `epsx = 1.e-6` is float32(1e-6), gfortran-checked
    assert fortran_default("pkg/ctrl/optim_readparms.F:87", "epsx", o).value == 9.999999974752427e-07
    assert fortran_literal("1.0e-12") == 9.999999960041972e-13 and fortran_literal("1.0 _d -12") == 1e-12


def test_stdout_crosscheck_where_oracle_runs_exist():
    checked = []
    for exp, inp in VARIANTS:
        out = oracle_stdout(exp, inp)
        if (exp, inp) in STDOUT_PENDING:
            assert out is None, f"{exp}/{inp} has an oracle run now ({out}): remove it from STDOUT_PENDING"
            continue
        assert out is not None, f"no oracle STDOUT for {exp}/{inp}"
        e = experiment(exp, inp)
        defaults = [fortran_default(c, n, e) for c, n in DEFAULTS] if exp == "tutorial_barotropic_gyre" else []
        cc = params.stdout_crosscheck(e, stdout_io.read_stdout(out), defaults)
        assert cc.ok and len(cc.compared) >= 20, (exp, inp, cc.mismatched, cc.packages)
        checked.append((exp, inp))
    assert ("tutorial_barotropic_gyre", "input") in checked


def test_stdout_crosscheck_negative_controls():
    e = experiment("tutorial_barotropic_gyre", "input")
    lines = stdout_io.read_stdout(oracle_stdout("tutorial_barotropic_gyre", "input"))
    # planted wrong default (gravity 9.8 instead of set_defaults.F:101's 9.81)
    cc = params.stdout_crosscheck(e, lines, [CitedDefault("gravity", 9.8, "planted")])
    assert [m[0] for m in cc.mismatched] == ["gravity [planted]"]
    # planted wrong value read from a file (viscAh 4.E2 -> 4.01E2)
    key = ("data", "parm01", "viscah")
    old = e.run.vars[key]
    e.run.vars[key] = namelists.Var(old.file, old.group, old.name, old.kind, old.shape, old.char_len, 401.0,
                                    old.assignments)
    try:
        cc = params.stdout_crosscheck(e, lines)
        assert [m[0] for m in cc.mismatched] == ["viscAh"]
    finally:
        e.run.vars[key] = old
    # a default for a parameter the file sets is refused
    with pytest.raises(ValueError, match="is set in a parameter file"):
        params.stdout_crosscheck(e, lines, [CitedDefault("viscAh", 0.0, "planted")])
