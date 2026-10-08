"""W2_SET_MAP_CUMSUM (pkg/exch2/w2_set_map_cumsum.F @63cdc0b): facet (and, with W2_CUMSUM_USE_MATRIX, tile) mapping for
the global cumulated sum.

    @63cdc0b pkg/exch2/w2_set_map_cumsum.F:8-368

Ported in full for one process (myProcId = 0): the facet progression (:74-218, all three connection types), the
facet matrix print (:228-236), the cubed-sphere missing-corner tiles (:241-259), and the tile matrix under
W2_CUMSUM_USE_MATRIX (:264-359, defined by adjustment.cs-32x32x1's W2_OPTIONS.h; W2_cumSum_tiles in
/W2_CUMSUM_MATRIX/) or the skip message (:361-364). The missing-connection warning (:219-227) raises (no ported
topology takes it). Integer code; `jj = INT(facet_link)`, `ii = MOD(NINT(facet_link*10.), 10)` with the REAL*4
product.
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.exch2.w2_exch2_h import (W2_EAST, W2_NORTH, W2_SOUTH, W2_WEST, W2_maxNbFacets, FIntArray, idiv,
                                         imod, int_, nint, r4)


def _copy_facet(w2, jj, j):
    """DO k=1,nFacets: W2_cumSum_facet(1:2,k,jj) = W2_cumSum_facet(1:2,k,j)."""
    for k in range(1, w2.nFacets + 1):
        w2.W2_cumSum_facet[1, k, jj] = w2.W2_cumSum_facet[1, k, j]
        w2.W2_cumSum_facet[2, k, jj] = w2.W2_cumSum_facet[2, k, j]


def w2_set_map_cumsum(w2, *, cfg, io, useCubedSphereExchange, myThid=1):
    """W2_SET_MAP_CUMSUM( myThid ) on the W2Common `w2` (useCubedSphereExchange: EEPARAMS.h, from "eedata")."""
    sz = cfg.size
    useMatrix = bool(cfg.cpp.W2_CUMSUM_USE_MATRIX)
    if useMatrix != (w2.W2_cumSum_tiles is not None):
        raise ValueError("W2Common declared without /W2_CUMSUM_MATRIX/ for a W2_CUMSUM_USE_MATRIX build (or with it "
                         "for one without)")
    msgBuf = internal_write("(2A)", "W2_SET_MAP_CUMSUM: ", "setting Facet Matrix for CUMUL-SUM")    # :49-50
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                              # :51
    prtFlag = abs(w2.W2_printMsg) >= 2 or (w2.W2_printMsg != 0 and io.myProcId == 0)             # :52-53

    # --   Initialise Common-block (:56-71)
    w2.W2_tMC1 = 0
    w2.W2_tMC2 = 0
    for j in range(1, W2_maxNbFacets + 1):
        for i in range(1, W2_maxNbFacets + 1):
            w2.W2_cumSum_facet[1, i, j] = 0
            w2.W2_cumSum_facet[2, i, j] = 0
    if useMatrix:                                                             # :64-71
        for j in range(1, w2.W2_maxNbTiles + 1):
            for i in range(1, w2.W2_maxNbTiles + 1):
                w2.W2_cumSum_tiles[1, i, j] = 0
                w2.W2_cumSum_tiles[2, i, j] = 0

    # --   Start setting cumul-sum Face mapping (:74-78)
    fCnt = 0
    fIsSet = {0: True}                                                        # LOGICAL fIsSet(0:W2_maxNbFacets)
    for j in range(1, w2.nFacets + 1):
        fIsSet[j] = False

    # --   Start with first non-empty face (:81-92)
    nActiveFacets = 0
    for j in range(1, w2.nFacets + 1):
        if w2.facet_dims[2 * j - 1] * w2.facet_dims[2 * j] >= 1:
            nActiveFacets = nActiveFacets + 1
            if fCnt == 0:
                fIsSet[j] = True
                fCnt = 1
                if abs(w2.W2_printMsg) >= 2:                                  # :88-89
                    io.write(w2.W2_oUnit, fortran_write("(A,I4)", " CumSum starts @ SW.corner of facet #", j))

    def set_from(jj, j, npass, nType):
        if abs(w2.W2_printMsg) >= 2:                                          # e.g. :121-123
            io.write(w2.W2_oUnit, fortran_write("(5(A,I4))", " CumSum SW.corner of facet #", jj, " set from facet",
                                                j, " (pass,type=", npass, ",", nType, ")"))

    # --   Go through list of connections (:97-218)
    prev_fCnt = 0
    npass = 0
    while fCnt > prev_fCnt:                                                   # :99
        npass = npass + 1
        prev_fCnt = fCnt
        for nType in range(1, 4):                                             # :102
            if fCnt == prev_fCnt:                                             # :103
                for j in range(1, w2.nFacets + 1):
                    if fIsSet[j]:
                        for i in range(1, 5):
                            # -    connected to:
                            jj = int_(w2.facet_link[i, j])                    # :108
                            ii = imod(nint(w2.facet_link[i, j] * r4(10.0)), 10)   # :109
                            # --   1) with same orientation ( N <-> S or E <-> W ), forward progression
                            if not fIsSet[jj] and nType == 1:                 # :111
                                if i == W2_NORTH and ii == W2_SOUTH:          # :113  N <-- S
                                    _copy_facet(w2, jj, j)                    # :114-117
                                    w2.W2_cumSum_facet[2, j, jj] = w2.W2_cumSum_facet[2, j, jj] + 1     # :118
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                                if i == W2_EAST and ii == W2_WEST:            # :126  E <-- W
                                    _copy_facet(w2, jj, j)                    # :127-130
                                    w2.W2_cumSum_facet[1, j, jj] = w2.W2_cumSum_facet[1, j, jj] + 1     # :131
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                            # --   2) with same orientation ( N <-> S or E <-> W ), backward progression
                            if not fIsSet[jj] and nType == 2:                 # :140
                                if i == W2_SOUTH and ii == W2_NORTH:          # :142  S <-- N
                                    _copy_facet(w2, jj, j)                    # :143-146
                                    w2.W2_cumSum_facet[2, jj, jj] = w2.W2_cumSum_facet[2, jj, jj] - 1   # :147
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                                if i == W2_WEST and ii == W2_EAST:            # :155  W <-- E
                                    _copy_facet(w2, jj, j)                    # :156-159
                                    w2.W2_cumSum_facet[1, jj, jj] = w2.W2_cumSum_facet[1, jj, jj] - 1   # :160
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                            # --   3) with different orientation ( N <-> W or S <-> E )
                            if not fIsSet[jj] and nType == 3 and fCnt == prev_fCnt:   # :170-171
                                if ((i == W2_NORTH and ii == W2_WEST)
                                        or (i == W2_WEST and ii == W2_NORTH)):  # :173-174
                                    _copy_facet(w2, jj, j)                    # :175-178
                                    w2.W2_cumSum_facet[2, j, jj] = w2.W2_cumSum_facet[2, j, jj] + 1     # :179
                                    w2.W2_cumSum_facet[2, jj, jj] = w2.W2_cumSum_facet[2, jj, jj] - 1   # :180
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                                if ((i == W2_EAST and ii == W2_SOUTH)
                                        or (i == W2_SOUTH and ii == W2_EAST)):  # :188-189
                                    _copy_facet(w2, jj, j)                    # :190-193
                                    w2.W2_cumSum_facet[1, j, jj] = w2.W2_cumSum_facet[1, j, jj] + 1     # :194
                                    w2.W2_cumSum_facet[1, jj, jj] = w2.W2_cumSum_facet[1, jj, jj] - 1   # :195
                                    fCnt = fCnt + 1
                                    fIsSet[jj] = True
                                    set_from(jj, j, npass, nType)
                if fCnt > prev_fCnt:                                          # :208-213
                    msgBuf = internal_write("(2A,3(I4,A),I2,A)", "W2_SET_MAP_CUMSUM: ", "set ", fCnt - prev_fCnt,
                                            " /", nActiveFacets, " active facets (pass,type=", npass, ",", nType,
                                            ")")
                    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)
    if fCnt < nActiveFacets:                                                  # :219-227
        raise NotImplementedError("W2_SET_MAP_CUMSUM: missing connections in Cumulated Sum (warning branch) is "
                                  "not ported")
    if io.myProcId == 0:                                                      # :228-236
        io.write(w2.W2_oUnit, fortran_write("(2A,2(I4,A))", " Facet Matrix for CUMUL-SUM (", "nFacets=", w2.nFacets,
                                            ", nActive=", nActiveFacets, " ):"))
        for j in range(1, w2.nFacets + 1):
            items = []
            for i in range(1, w2.nFacets + 1):
                items += [w2.W2_cumSum_facet[1, i, j], w2.W2_cumSum_facet[2, i, j], " ,"]
            io.write(w2.W2_oUnit, fortran_write("(A,I3,A,30(2I3,A))", "- facet", j, " :", *items))

    # --   record "missing corner" tile (:241-259)
    if useCubedSphereExchange:
        for j in range(1, w2.nFacets + 1):
            fNx = w2.facet_dims[2 * j - 1]                                    # :243
            fNy = w2.facet_dims[2 * j]                                        # :244
            if fNx * fNy >= 1:                                                # :245
                nbTx = idiv(fNx, sz.sNx)                                      # :246
                nbTy = idiv(fNy, sz.sNy)                                      # :247
                tN = w2.facet_owns[1, j] - 1 + nbTx                           # :248
                if w2.W2_tMC2 == 0 and imod(j, 2) == 0 and w2.exch2_myFace[tN] != 0:   # :249-250
                    w2.W2_tMC2 = tN
                tN = w2.facet_owns[1, j] + (nbTy - 1) * nbTx                  # :251
                if w2.W2_tMC1 == 0 and imod(j, 2) == 1 and w2.exch2_myFace[tN] != 0:   # :252-253
                    w2.W2_tMC1 = tN
        if io.myProcId == 0:                                                  # :256-258
            io.write(w2.W2_oUnit, fortran_write("(3(A,I8))", " missing-corner Tile for CUMUL-SUM (nTiles=",
                                                w2.exch2_nTiles, " ): W2_tMC1=", w2.W2_tMC1, " , W2_tMC2=",
                                                w2.W2_tMC2))

    # --   Now Set cumul-sum Tile mapping (:264-365)
    if useMatrix:
        _tile_matrix(w2, sz, io, prtFlag, myThid)
    else:
        msgBuf = internal_write("(2A)", "W2_SET_MAP_CUMSUM: ", "done (skip Tile Matrix setting)")   # :362-363
        print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                         # :364
    return w2


def _tile_matrix(w2, sz, io, prtFlag, myThid):
    """The W2_CUMSUM_USE_MATRIX block (w2_set_map_cumsum.F:265-359)."""
    mT = w2.W2_maxNbTiles
    facetXYSum = FIntArray("facetXYSum", 2, mT, W2_maxNbFacets)               # :42
    facet_CSum = FIntArray("facet_CSum", 2, mT, W2_maxNbFacets)               # :43
    cst = w2.W2_cumSum_tiles
    for j in range(1, W2_maxNbFacets + 1):                                    # :265-272
        for i in range(1, mT + 1):
            facetXYSum[1, i, j] = 0
            facetXYSum[2, i, j] = 0
            facet_CSum[1, i, j] = 0
            facet_CSum[2, i, j] = 0

    # -    First within each face (:275-314)
    for j in range(1, w2.nFacets + 1):
        fNx = w2.facet_dims[2 * j - 1]
        fNy = w2.facet_dims[2 * j]
        if fNx * fNy >= 1:
            nbTx = idiv(fNx, sz.sNx)                                          # :279
            nbTy = idiv(fNy, sz.sNy)                                          # :280
            for bj in range(1, nbTy + 1):                                     # :282-290
                for bi in range(1, nbTx - 1 + 1):
                    tS = w2.facet_owns[1, j] - 1 + bi                         # :284
                    tN = tS + 1 + (bj - 1) * nbTx                             # :285
                    for k in range(w2.facet_owns[1, j], tS + 1):              # :286-288
                        cst[1, k, tN] = 1
            tN = w2.facet_owns[1, j] - 1 + nbTx                               # :291
            facetXYSum[1, tN, j] = 1                                          # :292
            for k in range(w2.facet_owns[1, j], tN - 1 + 1):                  # :293-295
                facetXYSum[1, k, j] = cst[1, k, tN]

            for bj in range(1, nbTy - 1 + 1):                                 # :297-306
                for bi in range(1, nbTx + 1):
                    tS = w2.facet_owns[1, j] - 1 + bi                         # :299
                    tN = tS + bj * nbTx                                       # :300
                    for k in range(1, bj + 1):                                # :301-304
                        l_ = tS + (k - 1) * nbTx                              # :302  l = tS + (k-1)*nbTx
                        cst[2, l_, tN] = 1
            tN = w2.facet_owns[1, j] + (nbTy - 1) * nbTx                      # :307
            facetXYSum[2, tN, j] = 1                                          # :308
            for k in range(w2.facet_owns[1, j], tN - 1 + 1):                  # :309-311
                facetXYSum[2, k, j] = cst[2, k, tN]

    # -    Then across facet (:317-326)
    for j in range(1, w2.nFacets + 1):
        for k in range(1, w2.exch2_nTiles + 1):
            for i in range(1, w2.nFacets + 1):
                facet_CSum[1, k, j] = facet_CSum[1, k, j] + w2.W2_cumSum_facet[1, i, j] * facetXYSum[1, k, i]
                facet_CSum[2, k, j] = facet_CSum[2, k, j] + w2.W2_cumSum_facet[2, i, j] * facetXYSum[2, k, i]

    # -    Finally, account for cumulated sum at facet origin (:329-338)
    for j in range(1, w2.nFacets + 1):
        for tN in range(w2.facet_owns[1, j], w2.facet_owns[2, j] + 1):
            for k in range(1, w2.exch2_nTiles + 1):
                cst[1, k, tN] = cst[1, k, tN] + facet_CSum[1, k, j]
                cst[2, k, tN] = cst[2, k, tN] + facet_CSum[2, k, j]

    if prtFlag:                                                               # :340-355
        io.write(w2.W2_oUnit, fortran_write("(A,I8,A)", " Tile Matrix for CUMUL-SUM (nTiles=", w2.exch2_nTiles,
                                            " ):"))
        for j in range(1, w2.exch2_nTiles + 1):
            for is_ in range(1, w2.exch2_nTiles + 1, 10):
                ie = min(is_ + 9, w2.exch2_nTiles)                            # :345 MINMAX-INT: integer (no tie or NaN case)
                items = []
                for i in range(is_, ie + 1):
                    items += [cst[1, i, j], cst[2, i, j], " ,"]
                if is_ == 1:
                    io.write(w2.W2_oUnit, fortran_write("(3(I8,A),10(2I3,A))", j, " ,", is_, " ->", ie, " :",
                                                        *items))
                else:
                    io.write(w2.W2_oUnit, fortran_write("(8X,2(I8,A),10(2I3,A))", is_, " ->", ie, " :", *items))

    msgBuf = internal_write("(2A)", "W2_SET_MAP_CUMSUM: ", "setting Tile Matrix for CUMUL-SUM : done")   # :357-358
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                  # :359
