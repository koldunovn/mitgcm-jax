"""W2_MAP_PROCS (pkg/exch2/w2_map_procs.F @63cdc0b): which process owns which W2 tile (W2_tileProc, W2_tileIndex,
W2_myTileList, W2_procTileList), the tile-size check and the tile-connection print-out with W2_myCommFlag.

    @63cdc0b pkg/exch2/w2_map_procs.F:7-170

Ported as executed by the oracle (one process: myProcId = 0, eesupp/src/eeboot_minimal.F:85; nPx = nPy = 1): every
neighbour is on this process, so every communication flag is 'P' (:157-163). The 'M' (MPI message) branch
(:150-156), the tile-count STOP (:99-102) and the tile-size errors (:111-131) raise.

Note for the JAX port: W2_myTileList is the Fortran's own tile ordering (the tiles this process holds, by bi, bj);
the sharded JAX run splits that list over devices (mitjax/eesupp/shard.py) and has no Fortran process counterpart.
"""

from mitjax.eesupp.print import SQUEEZE_BOTH, SQUEEZE_RIGHT, internal_write, print_message
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNeighbours, idiv, imod


def w2_map_procs(w2, *, cfg, io, myThid=1):
    """W2_MAP_PROCS( myThid ) on the W2Common `w2`; `io.myProcId` is myProcId."""
    sz = cfg.size
    nSx, nSy, nPx, nPy = sz.nSx, sz.nSy, sz.nPx, sz.nPy
    # --   Initialise common blocs W2_MAP_TILE2PROC & W2_EXCH2_COMMFLAG (:47-62)
    for I in range(1, w2.W2_maxNbTiles + 1):
        w2.W2_tileProc[I] = 0
        w2.W2_tileIndex[I] = 0
    for bj in range(1, nSy + 1):
        for bi in range(1, nSx + 1):
            w2.W2_myTileList[bi, bj] = 0
            for np_ in range(1, nPx * nPy + 1):
                w2.W2_procTileList[bi, bj, np_] = 0
            for J in range(1, W2_maxNeighbours + 1):
                w2.W2_myCommFlag[J, bi, bj] = " "

    # Number of tiles I handle is nSx*nSy (:72-102)
    thisProc = 1 + io.myProcId                                                # :72
    J = 0                                                                     # :73
    for I in range(1, w2.exch2_nTiles + 1):                                   # :74
        if w2.exch2_myFace[I] != 0:                                           # :75
            # --   new ordering: for single sub-domain (nFacets=1) case, match default setting
            jj = idiv(J, nSx * nPx)                                           # :82
            ii = imod(J, nSx * nPx)                                           # :83
            # --   switch processor order to match MPI_CART set-up
            np_ = 1 + idiv(jj, nSy) + idiv(ii, nSx) * nPy                     # :87
            bj = 1 + imod(jj, nSy)                                            # :88
            bi = 1 + imod(ii, nSx)                                            # :89
            w2.W2_tileProc[I] = np_                                           # :91
            w2.W2_tileIndex[I] = bi + (bj - 1) * nSx                          # :92
            w2.W2_procTileList[bi, bj, np_] = I                               # :93
            if np_ == thisProc:
                w2.W2_myTileList[bi, bj] = I                                  # :94
            J = J + 1                                                         # :95
    if J != nSx * nSy * nPx * nPy:                                            # :99-102
        raise RuntimeError("ERROR W2_MAP_PROCS: number of active tiles not =nPx*nSx*nPy*nSy")

    # --   Check tile sizes (:105-131)
    iErr = 0
    for bj in range(1, nSy + 1):
        for bi in range(1, nSx + 1):
            myTileId = w2.W2_myTileList[bi, bj]                               # :108
            tNx = w2.exch2_tNx[myTileId]                                      # :109
            tNy = w2.exch2_tNy[myTileId]                                      # :110
            if tNx != sz.sNx:                                                 # :111-118
                iErr = iErr + 1
            if tNy != sz.sNy:                                                 # :119-126
                iErr = iErr + 1
    if iErr != 0:                                                             # :129-131
        raise RuntimeError("ABNORMAL END: W2_MAP_PROCS (tile size differs from sNx, sNy)")

    # --   Print tiles connection for this process and set myCommonFlag (:134-167)
    msgBuf = internal_write("(A)", "===== W2 TILE TOPOLOGY =====")           # :134
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_BOTH, myThid, io=io)          # :135
    for bj in range(1, nSy + 1):
        for bi in range(1, nSx + 1):
            myTileId = w2.W2_myTileList[bi, bj]                               # :138
            msgBuf = internal_write("(A,I5,A,2I4,2A,I3)", " TILE: ", myTileId, " (bi,bj=", bi, bj, " )",
                                    ", Nb of Neighbours =", w2.exch2_nNeighbours[myTileId])         # :139-141
            print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                       # :145
            for J in range(1, w2.exch2_nNeighbours[myTileId] + 1):           # :146
                commFlag = "M"                                                # :147
                jj = w2.exch2_neighbourId[J, myTileId]                        # :148
                if w2.W2_tileProc[jj] == thisProc:
                    commFlag = "P"                                            # :149
                if commFlag == "M":                                           # :150-156
                    raise NotImplementedError("W2_MAP_PROCS: neighbour on another process (Comm = MSG) is not "
                                              "ported")
                if commFlag == "P":                                           # :157
                    msgBuf = internal_write(
                        "(A,I3,A,I8,A,I3,2A,I8,A)", "    NEIGHBOUR", J, " = TILE", w2.exch2_neighbourId[J, myTileId],
                        " (n=", w2.exch2_opposingSend[J, myTileId], ") Comm = PUT",
                        " (PROC=", w2.W2_tileProc[w2.exch2_neighbourId[J, myTileId]], ")")              # :158-161
                    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                   # :162
                w2.W2_myCommFlag[J, bi, bj] = commFlag                        # :164
    return w2
