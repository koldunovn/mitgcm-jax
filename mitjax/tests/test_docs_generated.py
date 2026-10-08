"""The generated reference pages are fresh and their citations are true (docs plan 20261006 Task 7, Task 10 adds
its pages to tools/gen_docs.PAGES).

* every page of tools/gen_docs.PAGES regenerated in memory equals the committed file byte for byte;
* every port citation the map is built from names a file and lines of MITgcm at 63cdc0b ($MJX_UPSTREAM); the
  generator stops on one that does not (tools/gen_docs.CitationError);
* the experiment ports registered in mitjax/config/own_code.PORTED are on the map, each next to its Python module;
* no machine path on a page (it is published).

Negative controls, measured on the real page and sources: a committed page with one routine span changed is stale;
a planted source file with a new port citation makes the committed page stale; a planted citation of a missing
Fortran file, of a line past the end of a file, and a machine path on a page each stop the generator.
Needs $MJX_UPSTREAM only (never $MJX_REFERENCE); stdlib generator, seconds.
"""

import importlib.util
import re

import pytest

from mitjax import paths

REPO = paths.REPO


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("_mjx_gen_docs_test", "tools/gen_docs.py")


@pytest.fixture(scope="module")
def up():
    return G.Upstream(paths.UPSTREAM)


@pytest.fixture(scope="module")
def sources():
    return G.read_sources()


@pytest.fixture(scope="module")
def pages(up, sources):
    return {name: G.render(name, up, sources) for name in G.PAGES}


def _committed(name):
    return (REPO / G.PAGES[name][0]).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(G.PAGES))
def test_page_is_fresh(name, pages):
    assert pages[name] == _committed(name), (
        f"{G.PAGES[name][0]} is stale: regenerate with `{G.REGENERATE}` (and commit it)")


def test_control_stale_page_is_caught(pages):
    """A committed map with one routine span off by one line differs from the regeneration."""
    old = _committed("fortran_map")
    planted, n = re.subn(r"SEAICE_LSR `:24-1167`", "SEAICE_LSR `:24-1166`", old)
    assert n == 1, "the control's anchor moved: pick another span"
    assert planted != pages["fortran_map"]


def test_control_new_citation_makes_page_stale(up, sources, pages):
    """A source change the page has not seen (a planted module porting a routine) makes the committed page stale."""
    planted = dict(sources)
    planted["mitjax/pkg/seaice/zz_planted.py"] = (
        '"""SEAICE_EVP: pkg/seaice/seaice_evp.F @63cdc0b."""\n\n\n'
        'def seaice_evp():\n    """SEAICE_EVP( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_evp.F:20-60"""\n')
    text = G.render("fortran_map", up, planted)
    assert text != pages["fortran_map"]
    assert "zz_planted.py" in text and "SEAICE_EVP" in text


@pytest.mark.parametrize("cite, why", [
    ("pkg/seaice/seaice_no_such_file.F:10-20", "is not a file of MITgcm"),
    ("pkg/seaice/seaice_lsr.F:24-99999", "is outside pkg/seaice/seaice_lsr.F"),
])
def test_control_bad_citation_stops_generator(up, sources, cite, why):
    planted = dict(sources)
    planted["mitjax/pkg/seaice/zz_planted.py"] = f'def f():\n    """F( x )   @63cdc0b {cite}"""\n'
    with pytest.raises(G.CitationError, match=why):
        G.collect(up, sources=planted)


def test_control_machine_path_refused():
    with pytest.raises(G.CitationError, match="machine-specific"):
        G.check_neutral("fortran_map", "see /home/someone/MITgcm/model/src/cg2d.F")


def test_own_code_ports_are_mapped(up, sources):
    """Every experiment routine registered in own_code.PORTED is on the map next to the module that ports it (the
    MAIN_ONLY files excepted: a code_min/main.F stops before the model, own_code.MAIN_ONLY)."""
    own = _load("_mjx_own_code_test", "mitjax/config/own_code.py")
    by_file = {}
    for e in G.collect(up, sources=sources):
        by_file.setdefault(e["fortran"], set()).add(e["py"])
    missing = []
    for (exp, code_dir, fname), target in own.PORTED.items():
        if fname in own.MAIN_ONLY:
            continue
        py = target.split(":")[0].replace(".", "/") + ".py"
        if py not in by_file.get(f"verification/{exp}/{code_dir}/{fname}", set()):
            missing.append((exp, code_dir, fname, py))
    assert not missing, f"own_code.PORTED entries without a port citation on the map: {missing}"


# ------------------------------------------------------------------------------------------------- Task 10 pages


def test_control_stale_status_page_is_caught(up, sources, pages):
    """status: a committed page with one test dropped from a cell is stale; a test renamed in the test code (a planted
    source) changes the page, so the page follows the tests."""
    old = _committed("status")
    assert old == pages["status"]
    row = next(ln for ln in old.splitlines() if ln.startswith("| M4 | `lab_sea` | `input` |"))
    assert "test_m4lab_run" in row, row
    planted = old.replace(row, row.replace("[test_m4lab_run](../mitjax/tests/test_m4lab_run.py)", "", 1))
    assert planted != pages["status"]
    rel = "mitjax/tests/test_m4lab_run.py"
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "def test_whole_run_monitor_solver_digits_pickups" in text
    renamed = dict(sources)
    renamed[rel] = text.replace("def test_whole_run_monitor_solver_digits_pickups", "def test_renamed_planted")
    assert G.render("status", up, renamed) != pages["status"]


def test_control_stale_options_block_is_caught(up, sources, pages):
    """configurations: the generated block is fresh; a planted refusal in the code changes it; an edit inside the
    block is stale; the hand-written part outside the markers is free text."""
    old = _committed("configurations")
    assert old == pages["configurations"]
    planted_src = dict(sources)
    planted_src["mitjax/pkg/seaice/zz_planted.py"] = (
        "def zz_planted(cfg):\n    if cfg.cpp.SEAICE_ALLOW_EVP:\n        raise NotImplementedError('planted')\n")
    text = G.render("configurations", up, planted_src)
    assert text != old and "zz_planted.py" in text
    a = old.index(G.BEGIN_OPTIONS)
    edited_block = old[:a] + old[a:].replace("| `pkg/seaice` | ported |", "| `pkg/seaice` | refused |", 1)
    assert edited_block != old and edited_block != pages["configurations"]


def test_hand_part_of_configurations_is_kept(up, sources, pages, tmp_path, monkeypatch):
    """Regeneration keeps the hand-written text around the GENERATED block as committed (an edited title stays)."""
    old = _committed("configurations")
    edited = old.replace("# Configurations: your own experiment", "# Configurations (edited by hand)", 1)
    assert edited != old
    fake = tmp_path / "repo"
    (fake / "docs").mkdir(parents=True)
    (fake / "docs" / "configurations.md").write_text(edited, encoding="utf-8")
    monkeypatch.setattr(G, "REPO", fake)
    assert G.render_configurations(up, sources) == edited


def test_control_stale_readme_summary_is_caught(pages):
    old = _committed("readme")
    assert old == pages["readme"]
    planted = old.replace("| `lab_sea` | 9 |", "| `lab_sea` | 8 |", 1)
    assert planted != old and planted != pages["readme"]


def test_generated_blocks_need_their_markers(up, sources, tmp_path, monkeypatch):
    """A page whose GENERATED markers were deleted is refused, not silently regenerated without its block."""
    fake = tmp_path / "repo"
    (fake / "docs").mkdir(parents=True)
    (fake / "docs" / "configurations.md").write_text("# Configurations\n\nno markers\n", encoding="utf-8")
    monkeypatch.setattr(G, "REPO", fake)
    with pytest.raises(G.CitationError, match="GENERATED markers"):
        G.render_configurations(up, sources)
