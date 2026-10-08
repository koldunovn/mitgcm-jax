"""The W2 (exch2) topology set-up of global_ocean.90x40x15 (mitjax/pkg/exch2, w2_eeboot.F chain @63cdc0b), tier1x.

Gates (oracle: the registered dumps-on run of global_ocean.90x40x15/input and the measured exchange maps;
helpers in mitjax/tests/w2_gate.py):
* print-out: our `w2_tile_topology.0000.log` (850 records) and the W2 lines of STDOUT (9 records) equal the oracle's
  character for character;
* topology: exch2_tBasex/y and W2_myTileList on every tile equal the dump headers', exch2_mydNx/y the facet size;
  INI_PARMS's grid parameters are the same with our topology as with the header topology; the scalar exchange the
  topology implies (EXCH2_3D_RL replayed on exch2_neighbourId/opposingSend/pij/oi/oj/iLo..jHi through
  EXCH2_GET_SCAL_BOUNDS) equals the measured EXCH_XY_RL and EXCH_3D_RL maps on every point;
* the global_ocean grid built with our topology equals the oracle on every GRID.h field and point.
Negative controls (measured to bite, 2026-10-01): a tile offset +1 (exch2_tBasex(2)) and a facet size of 80
(exch2_mydNx) in the result fail the topology gates (headers, maps: 318 points; grid parameters: delX);
W2_myTileList(1,1)/(2,1) swapped after W2_MAP_PROCS fails the print-out gate; planted inside the chain, a tile offset
+1 after W2_SET_MAP_TILES and a facet size of 80 in W2_SET_SINGLE_FACET are stopped by the set-up's own checks
(W2_SET_TILE2TILES "Dbl/No connection", W2_SET_MAP_TILES tile count).
"""

import pytest

from mitjax.eesupp.print import MessageUnits
from mitjax.pkg.exch2 import w2_e2setup as E2, w2_eeboot as EB, w2_readparms as RP
from mitjax.pkg.exch2.w2_exch2_h import idiv, imod, int_, nint, r4
from mitjax.tests import grid_gate as gg
from mitjax.tests import w2_gate as G


def test_w2_printout_equals_oracle():
    e, w2, io = G.setup()
    bad = G.printout_mismatches(io, w2)
    assert bad == {"log": 0, "stdout": 0, "log_records": 850, "stdout_records": 9}, bad


def _run_planted(monkeypatch, module, name, after):
    """W2_EEBOOT with `after(w2)` applied right after the routine `module.name` returns (a planted error in the
    chain)."""
    orig = getattr(module, name)

    def wrapped(w2, **kw):
        out = orig(w2, **kw)
        after(w2)
        return out
    monkeypatch.setattr(module, name, wrapped)
    e, _, _ = G.setup()
    return EB.w2_eeboot(e, MessageUnits())


def test_w2_printout_negative_controls(monkeypatch):
    def offset(w2):
        w2.exch2_tBasex[2] = w2.exch2_tBasex[2] + 1
    with pytest.raises(RuntimeError, match="Dbl/No connection"):      # W2_SET_TILE2TILES's own check stops
        _run_planted(monkeypatch, E2, "w2_set_map_tiles", offset)
    monkeypatch.undo()

    def swap(w2):
        a, b = w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1]
        w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1] = b, a
    w2, io = _run_planted(monkeypatch, EB, "w2_map_procs", swap)
    assert G.printout_mismatches(io, w2)["log"] > 0
    assert "W2_myTileList" in G.header_mismatches(EB.exch2_topology(w2))
    monkeypatch.undo()

    def facet_size(w2):
        w2.facet_dims[1] = 80
    with pytest.raises(RuntimeError, match="does not match"):
        _run_planted(monkeypatch, E2, "w2_set_single_facet", facet_size)


def test_w2_topology_equals_headers_maps_and_grid_params():
    e, w2, io = G.setup()
    ex2 = EB.exch2_topology(w2)
    assert w2.exch2_nTiles == 36 and len(ex2.exch2_tBasex) == 36
    assert G.header_mismatches(ex2) == []
    assert G.gridparams_mismatches(ex2) == []
    assert G.map_mismatches(w2) == {"XY": 0, "3D": 0}
    # negative controls on the result
    w = G.planted(lambda w: w.exch2_tBasex.__setitem__(2, w.exch2_tBasex[2] + 1))
    assert G.header_mismatches(EB.exch2_topology(w)) == ["exch2_tBasex"]
    m = G.map_mismatches(w)
    assert m["XY"] > 0 and m["3D"] > 0, m

    def facet80(w):
        for t in range(1, w.exch2_nTiles + 1):
            w.exch2_mydNx[t] = 80
    w = G.planted(facet80)
    assert G.header_mismatches(EB.exch2_topology(w)) == ["exch2_mydNx"]
    assert "delX" in G.gridparams_mismatches(EB.exch2_topology(w))


def test_global_ocean_grid_from_w2_topology():
    """The grid gate of global_ocean with the topology from the W2 set-up instead of the dump headers."""
    e, w2, io = G.setup()
    params = gg.ini_parms_grid(e, EB.exch2_topology(w2))
    grid = gg.build_grid(G.EXP, G.INP, params=params)
    result, exempt = gg.compare(grid, G.EXP, G.INP)
    bad = gg.failures(result)
    assert not bad, bad
    assert len(result) >= 62


def test_w2_unported_options_raise(monkeypatch):
    e, _, _ = G.setup()
    # cubed sphere forced on the single-facet layout (36 tiles of 10x10: 3600 points are not 6*n^2): the cube set-up is
    # ported (test_cube.py), so the Fortran's own stop of W2_SET_CS6_FACETS (w2_set_cs6_facets.F:111-121) fires
    monkeypatch.setattr(RP, "use_cubed_sphere_exchange", lambda exp: True)
    monkeypatch.setattr(EB, "use_cubed_sphere_exchange", lambda exp: True)
    with pytest.raises(RuntimeError, match="attempt to fit single dim FAIL"):
        EB.w2_eeboot(e, MessageUnits())
    monkeypatch.undo()
    # (reading data.exch2 is ported and gated on solid-body.cs-32x32x1 in test_cube.py)
    # more than one process
    with pytest.raises(NotImplementedError, match="process"):
        EB.w2_eeboot(e, MessageUnits(myProcId=1))
    # Fortran integer / REAL*4 semantics the set-up relies on
    assert (idiv(-1, 10), idiv(-11, 10), idiv(11, 10), imod(-1, 10), imod(13, 10)) == (0, -1, 1, -1, 3)
    assert [imod(nint(r4(x) * r4(10.0)), 10) for x in (1.1, 1.2, 1.3, 1.4)] == [1, 2, 3, 4]
    assert [int_(r4(x)) for x in (1.1, 1.4, 2.3)] == [1, 1, 2]
    assert (nint(2.5), nint(-2.5), nint(2.4999)) == (3, -3, 2)
