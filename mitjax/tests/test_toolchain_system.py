"""The system toolchain (docs plan 20261006 Task 4, R2 of docs/PORTABILITY.md): the C preprocessor found as genmake2
finds it, with the declared DEFINES (mitjax/config/cpp_options.py system_toolchain), gives every verification build
of the registry (reference/reference_runs.py: the M1-M4 variants and the code_min case) the same configuration as the
Fortran oracle's own preprocessor read from its build records (cpp_options.oracle_toolchain).

Per build, with both toolchains: the loaded configuration (ExperimentConfig: CPP options of every *OPTIONS.h and the
model/src prologue, SIZE.h, package switches, namelist values) and the float parameters equal; packages_boot.F's
macros equal; EVERY source and header of the flat build directory preprocessed byte for byte equal (what SIZE.h, the
namelist readers and params_io.fortran_default read), and every conditional arm of every .F taken alike (what
cpp_options.live_lines decides). The one allowed difference: the default INCLUDES are empty (DEFAULT_INCLUDES), so a
file with `#include "netcdf.inc"` (pkg/mnc, pkg/profiles; never preprocessed by mitjax) fails with the default
toolchain and must fail for exactly that reason.

The default toolchain is the environment's: run with MJX_CPP set (e.g. clang's cpp) the same gate measures that
preprocessor (decision 10, scripts/port_gate_s2.sbatch).
Negative controls: a planted DEFINE changes the configuration and the preprocessed text; no working cpp is a clear
error naming MJX_CPP; inconsistent overrides are refused.
Costs: about 30 s per build on a CPU node (two configuration loads, ~2x1100 cpp runs on 32 threads), 28 builds.
"""

import functools
import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mitjax import paths
from mitjax.config import cpp_options as C
from mitjax.config import params


@functools.lru_cache(maxsize=None)
def _rr():
    spec = importlib.util.spec_from_file_location("_mjx_reference_runs_tc", paths.REPO / "reference" /
                                                  "reference_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def registry_builds():
    """{(experiment, code dir): first input dir} over every variant of the registry."""
    rr = _rr()
    out = {}
    for e, i, c in rr.VARIANTS + rr.M2_VARIANTS + rr.M2_MIN + rr.M3_VARIANTS + rr.M4_VARIANTS:
        out.setdefault((e, c), i)
    return out


BUILDS = sorted(registry_builds())
_SRC = (".F", ".F90", ".h")          # the .c files go to the C compiler, never through CPPCMD (genmake2:3380)
_NETCDF_INC = re.compile(r'^#\s*include\s+"netcdf\.inc"', re.M)


def _preprocess(farm, tc, name):
    try:
        return C.preprocess(farm, tc, name)
    except RuntimeError as x:
        return f"ERROR {x}"


def _arms(farm, tc, name):
    try:
        return tuple((a.line, a.taken) for a in C.evaluate_arms(farm, tc, name))
    except (RuntimeError, ValueError) as x:
        return f"ERROR {x}"


def build_differences(exp, inp, ref, tc, threads=32):
    """The differences between the configurations and preprocessed sources of one build under toolchains `ref` and
    `tc` ([] = identical, the netcdf.inc case aside), and the names that differ only by the missing netcdf.inc."""
    ea = params.load(exp, inp, toolchain=ref)
    eb = params.load(exp, inp, toolchain=tc)
    diffs = []
    if ea.cfg != eb.cfg:
        diffs.append("ExperimentConfig")
    if {k: float(v).hex() for k, v in ea.params.values.items()} != \
            {k: float(v).hex() for k, v in eb.params.values.items()}:
        diffs.append("NamelistParams")
    if ea.farm.path != eb.farm.path:
        diffs.append("farm")
    own = lambda farm, t: {k: v for k, v in C.routine_macros(farm, t, "packages_boot.F").items()     # noqa: E731
                           if C.predefined(t).get(k, None) != v}      # the compiler's own macros differ by version
    if own(ea.farm, ref) != own(eb.farm, tc):
        diffs.append("packages_boot.F macros")
    names = sorted(n for n in list(ea.farm.manifest["links"]) + list(ea.farm.manifest["generated_sha256"])
                   if n.endswith(_SRC))

    def one(n):
        a, b = _preprocess(ea.farm, ref, n), _preprocess(eb.farm, tc, n)
        arms = n.endswith(".F") and _arms(ea.farm, ref, n) != _arms(eb.farm, tc, n)
        return n, a, b, arms

    netcdf_only = []
    with ThreadPoolExecutor(threads) as ex:
        for n, a, b, arms in ex.map(one, names):
            if a == b and not arms:
                continue
            text = (ea.farm.path / n).read_text(encoding="latin-1")
            if (not tc.includes and b.startswith("ERROR") and "netcdf.inc" in b and not a.startswith("ERROR")
                    and _NETCDF_INC.search(text)):
                netcdf_only.append(n)
                continue
            diffs.append(f"{n}: {'text' if a != b else 'arms'}")
    return diffs, netcdf_only


@functools.lru_cache(maxsize=None)
def _toolchains():
    return C.oracle_toolchain(), C.default_toolchain()


@pytest.mark.parametrize("exp,code", BUILDS, ids=[f"{e}-{c}" for e, c in BUILDS])
def test_system_toolchain_equals_oracle(exp, code):
    ref, tc = _toolchains()
    diffs, netcdf_only = build_differences(exp, registry_builds()[(exp, code)], ref, tc)
    assert diffs == [], (exp, code, tc, diffs[:20])
    assert all(n.startswith(("mnc_", "profiles_")) for n in netcdf_only), netcdf_only


def test_declared_defaults_are_the_oracle_records():
    """The declared DEFINES are the build records' DEFINES line, in genmake2's order; the oracle's cpp is GNU cpp in
    traditional mode with -P, as DEFAULT_CPP; the default toolchain finds a working cpp by absolute path."""
    ref = C.oracle_toolchain()
    assert C.default_defines() == ref.defines and ref.have_netcdf
    assert ref.cpp[1:] == tuple(C.DEFAULT_CPP.split()[1:]) and Path(ref.cpp[0]).name == C.DEFAULT_CPP.split()[0]
    tc = C.system_toolchain({})
    assert Path(tc.cpp[0]).is_absolute() and tc.cpp[1:] == ("-traditional", "-P"), tc
    assert tc.defines == C.default_defines() and tc.includes == () and tc.have_netcdf


def test_planted_define_changes_the_set():
    """Negative control: one DEFINE added to the default toolchain changes the configuration (CPP options) and the
    preprocessed text of the sources that test it."""
    ref, tc = _toolchains()
    planted = C.Toolchain(tc.cpp, tc.defines + ("-DALLOW_AUTODIFF",), tc.includes, "planted")
    diffs, _ = build_differences("tutorial_barotropic_gyre", "input", ref, planted)
    assert diffs == ["ExperimentConfig", "the_model_main.F: text"], diffs[:10]
    a = params.load("tutorial_barotropic_gyre", "input", toolchain=ref).cfg
    b = params.load("tutorial_barotropic_gyre", "input", toolchain=planted).cfg
    assert a.cpp.flag("ALLOW_AUTODIFF", "CPP_OPTIONS.h") is False
    assert b.cpp.flag("ALLOW_AUTODIFF", "CPP_OPTIONS.h") is True


def test_no_cpp_is_a_clear_error(tmp_path, monkeypatch):
    """genmake2:1946-1965: neither `$CPP` nor `/lib/$CPP` works -> an error that names MJX_CPP; a command that runs
    but fails the `#define A a` test counts as not working; the /lib/ fallback is taken when PATH has no cpp."""
    with pytest.raises(C.CppNotFound, match="MJX_CPP"):
        C.system_toolchain({"MJX_CPP": "mjx-no-such-cpp -traditional -P"})
    with pytest.raises(C.CppNotFound, match="failed the test case"):
        C.find_cpp("false")
    if Path("/lib/cpp").is_file():
        monkeypatch.setenv("PATH", str(tmp_path))
        assert C.find_cpp()[0] == "/lib/cpp"
        with pytest.raises(C.CppNotFound):
            C.find_cpp("mjx-no-such-cpp")


def test_overrides():
    """MJX_CPP_DEFINES, MJX_CPP_INCLUDES, MJX_HAVE_NETCDF: the explicit HAVE_NETCDF drops -DHAVE_NETCDF from the
    declared set (and pkg/mnc from the package list, genmake2:2531-2560); a DEFINES line that disagrees with it is
    refused; a bad value is refused."""
    off = C.system_toolchain({"MJX_HAVE_NETCDF": "0"})
    assert not off.have_netcdf and "-DHAVE_NETCDF" not in off.defines
    assert off.defines == C.default_defines(False)
    own = C.system_toolchain({"MJX_CPP_DEFINES": "-DWORDLENGTH=4 -DHAVE_NETCDF", "MJX_CPP_INCLUDES": "-I/x"})
    assert own.defines == ("-DWORDLENGTH=4", "-DHAVE_NETCDF") and own.have_netcdf and own.includes == ("-I/x",)
    with pytest.raises(ValueError, match="have_netcdf"):
        C.system_toolchain({"MJX_CPP_DEFINES": "-DWORDLENGTH=4", "MJX_HAVE_NETCDF": "1"})
    with pytest.raises(ValueError, match="MJX_HAVE_NETCDF"):
        C.system_toolchain({"MJX_HAVE_NETCDF": "maybe"})
    from mitjax.config import packages
    u = paths.UPSTREAM
    mods = [u / "verification" / "tutorial_baroclinic_gyre" / "code"]
    assert "mnc" in packages.package_set(u, mods, have_netcdf=True).packages
    assert "mnc" not in packages.package_set(u, mods, have_netcdf=off.have_netcdf).packages
