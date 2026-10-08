"""W2_E2SETUP (pkg/exch2/w2_e2setup.F @63cdc0b): set up the W2_EXCH2 tile topology structures.

    @63cdc0b pkg/exch2/w2_e2setup.F:10-149

Ported as executed by global_ocean.90x40x15 (docs/coverage/global_ocean.90x40x15.md) and the cubed-sphere
experiments (adjustment.cs-32x32x1, solid-body.cs-32x32x1, advect_cs, global_ocean.cs32x15): no blank tiles
(blankList all 0, :62-81 never adds one), preDefTopol = 1 (W2_SET_SINGLE_FACET) or 3 (W2_SET_CS6_FACETS). Blank
tiles and the other topologies (preDefTopol 0: W2_SET_GEN_FACETS, 2: W2_SET_MYOWN_FACETS) are not ported and raise.
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNbFacets
from mitjax.pkg.exch2.w2_set_cs6_facets import w2_set_cs6_facets
from mitjax.pkg.exch2.w2_set_f2f_index import w2_set_f2f_index
from mitjax.pkg.exch2.w2_set_map_cumsum import w2_set_map_cumsum
from mitjax.pkg.exch2.w2_set_map_tiles import w2_set_map_tiles
from mitjax.pkg.exch2.w2_set_single_facet import w2_set_single_facet
from mitjax.pkg.exch2.w2_set_tile2tiles import w2_set_tile2tiles


def w2_e2setup(w2, *, cfg, io, useCubedSphereExchange, myThid=1):
    """W2_E2SETUP( myThid ) on the W2Common `w2`; messages to `io` (unit w2.W2_oUnit). useCubedSphereExchange:
    EEPARAMS.h (from "eedata", read by W2_EEBOOT's caller)."""
    sz = cfg.size
    # --   Initialise parameters from EXCH2_PARAMS common blocks (:48-58)
    for j in range(1, W2_maxNbFacets + 1):
        w2.facet_owns[1, j] = 0
        w2.facet_owns[2, j] = 0
        for i in range(1, 5):
            for k in range(1, 5):
                w2.facet_pij[k, i, j] = 0
            w2.facet_oi[i, j] = 0
            w2.facet_oj[i, j] = 0

    # --   Count Nb of Blank-Tiles and set Number of tiles (:61-82)
    w2.nBlankTiles = 0
    for i in range(1, w2.W2_maxNbTiles + 1):
        if w2.blankList[i] != 0:                                              # :63
            raise NotImplementedError("W2_E2SETUP: blank tiles (w2_e2setup.F:64-79) are not ported")
    w2.exch2_nTiles = w2.nBlankTiles + (sz.nSx * sz.nSy * sz.nPx * sz.nPy)    # :82

    msgBuf = internal_write("(A,I8)", "W2_E2SETUP: number of Active Tiles =", sz.nSx * sz.nSy * sz.nPx * sz.nPy)  # :84-85
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                          # :86
    msgBuf = internal_write("(A,I8)", "W2_E2SETUP: number of Blank Tiles  =", w2.nBlankTiles)                 # :87-88
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                          # :89
    msgBuf = internal_write("(A,I8)", "W2_E2SETUP: Total number of Tiles  =", w2.exch2_nTiles)                # :90-91
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                          # :92

    if w2.exch2_nTiles > w2.W2_maxNbTiles:                                    # :94-102
        raise RuntimeError(f"W2_E2SETUP: Number of Tiles={w2.exch2_nTiles:8d} >{w2.W2_maxNbTiles:8d} =W2_maxNbTiles; "
                           "ABNORMAL END: S/R W2_E2SETUP (nTiles>maxNbTiles)")

    # --   Check blankList (:105-115): nBlankTiles = 0, no iteration

    # --   Define Facet (sub-domain) Topology: Size and Connections (:118-128)
    if w2.preDefTopol == 0:
        raise NotImplementedError("W2_E2SETUP: preDefTopol=0 (W2_SET_GEN_FACETS, data.exch2 topology) is not ported")
    elif w2.preDefTopol == 1:
        w2_set_single_facet(w2, cfg=cfg, io=io, myThid=myThid)                # :121
    elif w2.preDefTopol == 2:
        raise NotImplementedError("W2_E2SETUP: preDefTopol=2 (W2_SET_MYOWN_FACETS) is not ported")
    elif w2.preDefTopol == 3:
        w2_set_cs6_facets(w2, cfg=cfg, io=io, myThid=myThid)                  # :125
    else:
        raise RuntimeError("ABNORMAL END: S/R W2_E2SETUP (invalid preDefTopol)")   # :127

    msgBuf = internal_write("(A,I8)", "W2_E2SETUP: Total number of Facets =", w2.nFacets)                     # :130-131
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                          # :132

    # --   Check Topology; setup correspondence matrix for connected Facet-Edges
    w2_set_f2f_index(w2, io=io, myThid=myThid)                                # :135
    # --   Define Tile Mapping (+ IO global mapping)
    w2_set_map_tiles(w2, cfg=cfg, io=io, myThid=myThid)                       # :138
    # --   Define Tile Mapping (for Cumulated Sum)
    w2_set_map_cumsum(w2, cfg=cfg, io=io, useCubedSphereExchange=useCubedSphereExchange, myThid=myThid)                      # :141
    # --   Set-up tile neighbours and index relations for EXCH2
    w2_set_tile2tiles(w2, cfg=cfg, io=io, myThid=myThid)                      # :144
    return w2
