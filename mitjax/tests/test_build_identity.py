"""Build identity and the MAX/MIN default (docs plan 20261006 Task 4: R6 of docs/PORTABILITY.md, decision 7).

1. Every verification build of the registry (reference/reference_runs.py: M1-M4 variants and code_min) has the
   identity mitjax/config/build_identity.KNOWN names it by, and KNOWN holds nothing else.
2. A renamed copy of lab_sea (code_ad unchanged) is the build lab_sea-code_ad-63cdc0b: the measured winners of
   seaice_growth.F:1847-1848 (MINMAX_BUILD), no warning -- the known build stays bitwise.
3. Negative controls: the same copy with one planted #define in code_ad/SEAICE_OPTIONS.h, or another SIZE.h tiling,
   is a new build: build_winner gives the documented default (MINMAX_DEFAULT) with exactly one MinMaxDefaultWarning
   per site naming it (and raises without a default); the old name-based lookup would have taken the edited copy for
   the known build.
4. A known build without a table entry (lab_sea/code, where formula 3 was never measured; plan decision 7 revised
   2026-10-07): refused under MJX_MINMAX_STRICT=1, which conftest.py sets for every test; for a user (0 or unset) the
   documented default with one warning per site naming the build; any other value of the switch is an error.
Costs: about 2 min on a CPU node (29 configuration loads with cpp, no model).
"""

import dataclasses
import importlib.util
import os
import shutil
import time
import warnings

import pytest

from mitjax import paths
from mitjax.config import build_identity as bi
from mitjax.config import params
from mitjax.ops import fortran_minmax as fm

SITES = ("pkg/seaice/seaice_growth.F:1847", "pkg/seaice/seaice_growth.F:1848")


def _rr():
    spec = importlib.util.spec_from_file_location("_mjx_reference_runs_bi", paths.REPO / "reference" /
                                                  "reference_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _new_dir(tag):
    d = paths.RUNS / "tests_port" / f"{tag}-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    return d


def _copy_lab_sea(name, edit=None):
    """<new dir>/<name> holding lab_sea's code_ad and input_ad (symlinks dereferenced); `edit(exp_dir)` plants."""
    src = paths.UPSTREAM / "verification" / "lab_sea"
    dst = _new_dir("build-identity") / name
    for d in ("code_ad", "input_ad"):
        shutil.copytree(src / d, dst / d, symlinks=False)
    if edit:
        edit(dst)
    return dst


def _winners(cfg):
    from mitjax.pkg.seaice import seaice_growth as G
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = tuple(fm.build_winner(cfg, G.MINMAX_BUILD, s, G.MINMAX_DEFAULT) for s in SITES)
    return got, [x for x in w if issubclass(x.category, fm.MinMaxDefaultWarning)]


def test_known_identities_are_the_registry_builds():
    rr = _rr()
    builds = {}
    for e, i, c in rr.VARIANTS + rr.M2_VARIANTS + rr.M2_MIN + rr.M3_VARIANTS + rr.M4_VARIANTS:
        builds.setdefault((e, c), i)
    got = {}
    for (e, c), i in sorted(builds.items()):
        cfg = params.load(e, i).cfg
        got[bi.identity(cfg)] = f"{e}-{c}-{cfg.upstream_commit[:7]}"
    assert len(got) == len(builds) == 28                       # no two builds share an identity
    assert got == bi.KNOWN


def test_renamed_copy_is_the_known_build(monkeypatch):
    monkeypatch.setattr(fm, "_WARNED", set())
    cfg = params.load("my_lab_sea", "input_ad", exp_dir=_copy_lab_sea("my_lab_sea")).cfg
    assert bi.label(cfg) == "lab_sea-code_ad-63cdc0b"
    got, warned = _winners(cfg)
    assert got == ("a", "b") and warned == []


def _plant_option(exp_dir):
    h = exp_dir / "code_ad" / "SEAICE_OPTIONS.h"
    h.write_text(h.read_text() + "\n#define MJX_PLANTED_OPTION\n")


def _plant_tiling(exp_dir):
    h = exp_dir / "code_ad" / "SIZE.h"
    t = h.read_text()
    assert "sNx =  10," in t and "nSx =   2," in t, "lab_sea code_ad SIZE.h changed"
    h.write_text(t.replace("sNx =  10,", "sNx =  20,").replace("nSx =   2,", "nSx =   1,"))     # 2 x 10 -> 1 x 20


@pytest.mark.parametrize("plant", [_plant_option, _plant_tiling], ids=["option", "tiling"])
def test_edited_copy_gets_the_default_with_one_warning(plant, monkeypatch):
    monkeypatch.setattr(fm, "_WARNED", set())
    cfg = params.load("lab_sea", "input_ad", exp_dir=_copy_lab_sea("lab_sea", plant)).cfg
    assert bi.label(cfg) is None
    got, warned = _winners(cfg)
    assert got == ("a", "b")                                   # MINMAX_DEFAULT
    assert [str(w.message).split(":")[0] + ":" + str(w.message).split(":")[1] for w in warned] == list(SITES)
    assert all("not a build the Fortran oracle measured" in str(w.message) for w in warned)
    got, warned = _winners(cfg)                                 # one warning per site and process
    assert got == ("a", "b") and warned == []
    from mitjax.pkg.seaice import seaice_growth as G
    with pytest.raises(NotImplementedError, match="no documented default"):
        fm.build_winner(cfg, G.MINMAX_BUILD, SITES[0])
    # the name-based lookup of 1b02ac0 (`<experiment>-<code dir>-<commit>`) took this edited build for the known one
    old = f"{cfg.experiment}-{cfg.code_dir}-{cfg.upstream_commit[:7]}"
    assert old == "lab_sea-code_ad-63cdc0b" and old in bi.KNOWN.values()


def test_known_build_without_entry_strict_in_tests_default_for_users(monkeypatch):
    monkeypatch.setattr(fm, "_WARNED", set())
    cfg = params.load("lab_sea", "input").cfg                 # lab_sea/code: formula 3 never measured there
    assert bi.label(cfg) == "lab_sea-code-63cdc0b"
    from mitjax.pkg.seaice import seaice_growth as G
    # our tests run strict (conftest.py): the known build without an entry is refused
    assert os.environ.get(fm.STRICT_ENV) == "1"
    with pytest.raises(NotImplementedError, match="MJX_MINMAX_STRICT=1"):
        fm.build_winner(cfg, G.MINMAX_BUILD, SITES[0], G.MINMAX_DEFAULT)
    # a user (switch 0 or unset): the documented default, one warning per site naming the build
    for value in ("0", None):
        monkeypatch.setattr(fm, "_WARNED", set())
        if value is None:
            monkeypatch.delenv(fm.STRICT_ENV)
        else:
            monkeypatch.setenv(fm.STRICT_ENV, value)
        got, warned = _winners(cfg)
        assert got == tuple(G.MINMAX_DEFAULT[s] for s in SITES) == ("a", "b")
        assert [str(w.message).split(": the MAX/MIN")[0] for w in warned] == list(SITES)
        assert all("lab_sea-code-63cdc0b" in str(w.message) and "never measured" in str(w.message) for w in warned)
        got, warned = _winners(cfg)                             # one warning per site and process
        assert got == ("a", "b") and warned == []
        with pytest.raises(NotImplementedError, match="no documented default"):
            fm.build_winner(cfg, G.MINMAX_BUILD, SITES[0])
    monkeypatch.setenv(fm.STRICT_ENV, "yes")
    with pytest.raises(ValueError, match="MJX_MINMAX_STRICT"):
        fm.build_winner(cfg, G.MINMAX_BUILD, SITES[0], G.MINMAX_DEFAULT)
    # a stand-in configuration that only renames the code directory is still the same build
    assert bi.label(dataclasses.replace(cfg, code_dir="code_ad")) == "lab_sea-code-63cdc0b"
