"""W2_SET_TILE2TILES (pkg/exch2/w2_set_tile2tiles.F @63cdc0b): tile neighbours and index relations for EXCH2
(exch2_neighbourId, exch2_pij, exch2_oi/oj, exch2_iLo/iHi/jLo/jHi, exch2_opposingSend, exch2_neighbourDir and the
facet-edge flags exch2_isN/S/E/Wedge).

    @63cdc0b pkg/exch2/w2_set_tile2tiles.F:9-342

Ported as executed by global_ocean.90x40x15: active tiles only (no blank tile), internal connections (:113-140) and
external connections across the facet edges (:145-213, which the single facet's periodic links reach) with
increasing index ranges; at most W2_maxNeighbours neighbours; every neighbour has exactly one reciprocal entry.
Decreasing index ranges (:162-171) come from the orientation-changing facet links of the cube. The error branches
raise.
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNeighbours, FIntArray, idiv, imod, int_, nint, r4


def w2_set_tile2tiles(w2, *, cfg, io, myThid=1):
    """W2_SET_TILE2TILES( myThid ) on the W2Common `w2`."""
    sz = cfg.size
    msgBuf = internal_write("(2A)", "W2_SET_TILE2TILES:", " tile neighbours and index connection:")   # :52-53
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                              # :54
    prtFlag = abs(w2.W2_printMsg) >= 2 or (w2.W2_printMsg != 0 and io.myProcId == 0)             # :55-56

    # --   Initialise local arrays (:59-64)
    tile_edge2edge = FIntArray("tile_edge2edge", W2_maxNeighbours, w2.W2_maxNbTiles)              # :37
    for is_ in range(1, w2.W2_maxNbTiles + 1):
        for ns in range(1, W2_maxNeighbours + 1):
            tile_edge2edge[ns, is_] = 0
            w2.exch2_neighbourDir[ns, is_] = 0

    tNx = sz.sNx                                                              # :66
    tNy = sz.sNy                                                              # :67
    for is_ in range(1, w2.exch2_nTiles + 1):                                 # :68
        js = w2.exch2_myFace[is_]                                             # :69
        # test "myFace" for blank tile; no need for connection if tile is blank
        if js != 0:                                                           # :71
            js = w2.exch2_myFace[is_]                                         # :72
            iLo = w2.exch2_tBasex[is_] + 1                                    # :73
            iHi = w2.exch2_tBasex[is_] + w2.exch2_tNx[is_]                    # :74
            jLo = w2.exch2_tBasey[is_] + 1                                    # :75
            jHi = w2.exch2_tBasey[is_] + w2.exch2_tNy[is_]                    # :76

            nbNeighb = 0                                                      # :78
            for i in range(1, 5):                                             # :79
                ii1 = iLo                                                     # :80
                ii2 = iHi                                                     # :81
                jj1 = jLo                                                     # :82
                jj2 = jHi                                                     # :83
                if i == 1:                                                    # :84  Northern Edge
                    jj1 = jHi + 1                                             # :86
                    jj2 = jHi + 1                                             # :87
                    internConnect = jHi < w2.exch2_mydNy[is_]                 # :88
                    if not internConnect:
                        w2.exch2_isNedge[is_] = 1                             # :89
                elif i == 2:                                                  # :90  Southern Edge
                    jj1 = jLo - 1                                             # :92
                    jj2 = jLo - 1                                             # :93
                    internConnect = jLo > 1                                   # :94
                    if not internConnect:
                        w2.exch2_isSedge[is_] = 1                             # :95
                elif i == 3:                                                  # :96  Eastern Edge
                    ii1 = iHi + 1                                             # :98
                    ii2 = iHi + 1                                             # :99
                    internConnect = iHi < w2.exch2_mydNx[is_]                 # :100
                    if not internConnect:
                        w2.exch2_isEedge[is_] = 1                             # :101
                else:                                                         # :102  Western Edge
                    ii1 = iLo - 1                                             # :104
                    ii2 = iLo - 1                                             # :105
                    internConnect = iLo > 1                                   # :106
                    if not internConnect:
                        w2.exch2_isWedge[is_] = 1                             # :107
                ddi = min(ii2 - ii1, 1)                                       # :109; MINMAX-INT: integer (no tie or NaN case)
                ddj = min(jj2 - jj1, 1)                                       # :110; MINMAX-INT: integer (no tie or NaN case)

                if internConnect:                                             # :112
                    # ---  Internal (from the same facet): get tile neighbour Id "it"
                    nbTx = idiv(w2.facet_dims[2 * js - 1], tNx)               # :116
                    ii = 1 + imod(i, 2)                                       # :117
                    it = 2 * ii - 3                                           # :118
                    if i <= 2:                                                # :119
                        it = is_ + it * nbTx                                  # :120
                    else:
                        it = is_ + it                                         # :122
                        ii = ii + 2                                           # :123
                    if w2.exch2_myFace[it] != 0:                              # :125
                        nbNeighb = nbNeighb + 1                               # :126
                        ns = min(nbNeighb, W2_maxNeighbours)                  # :127; MINMAX-INT: integer (no tie or NaN case)
                        w2.exch2_neighbourId[ns, is_] = it                    # :128
                        tile_edge2edge[ns, is_] = 10 * i + ii                 # :129
                        w2.exch2_pij[1, ns, is_] = 1                          # :130
                        w2.exch2_pij[2, ns, is_] = 0                          # :131
                        w2.exch2_pij[3, ns, is_] = 0                          # :132
                        w2.exch2_pij[4, ns, is_] = 1                          # :133
                        w2.exch2_oi[ns, is_] = 0                              # :134
                        w2.exch2_oj[ns, is_] = 0                              # :135
                        w2.exch2_iLo[ns, is_] = ii1 - ddi - w2.exch2_tBasex[is_]   # :136
                        w2.exch2_iHi[ns, is_] = ii2 + ddi - w2.exch2_tBasex[is_]   # :137
                        w2.exch2_jLo[ns, is_] = jj1 - ddj - w2.exch2_tBasey[is_]   # :138
                        w2.exch2_jHi[ns, is_] = jj2 + ddj - w2.exch2_tBasey[is_]   # :139
                else:
                    # ---  External (from an other facet)
                    jt = int_(w2.facet_link[i, js])                           # :145
                    ii = imod(nint(w2.facet_link[i, js] * r4(10.0)), 10)      # :146
                    if jt > 0:                                                # :147
                        # -    index range on target facet (:150-157)
                        P, oi, oj = w2.facet_pij, w2.facet_oi, w2.facet_oj
                        ibnd1 = P[1, ii, jt] * ii1 + P[2, ii, jt] * jj1 + oi[ii, jt]
                        ibnd2 = P[1, ii, jt] * ii2 + P[2, ii, jt] * jj2 + oi[ii, jt]
                        jbnd1 = P[3, ii, jt] * ii1 + P[4, ii, jt] * jj1 + oj[ii, jt]
                        jbnd2 = P[3, ii, jt] * ii2 + P[4, ii, jt] * jj2 + oj[ii, jt]
                        if ibnd1 <= ibnd2:                                    # :159
                            txbnd1 = idiv(ibnd1 - 1, tNx)                     # :160
                            txbnd2 = idiv(ibnd2 - 1, tNx)                     # :161
                        else:                                                 # :162
                            txbnd1 = idiv(ibnd2 - 1, tNx)                     # :163
                            txbnd2 = idiv(ibnd1 - 1, tNx)                     # :164
                        if jbnd1 <= jbnd2:                                    # :166
                            tybnd1 = idiv(jbnd1 - 1, tNy)                     # :167
                            tybnd2 = idiv(jbnd2 - 1, tNy)                     # :168
                        else:                                                 # :169
                            tybnd1 = idiv(jbnd2 - 1, tNy)                     # :170
                            tybnd2 = idiv(jbnd1 - 1, tNy)                     # :171
                        nbTx = idiv(w2.facet_dims[2 * jt - 1], tNx)           # :173
                        for ty in range(tybnd1, tybnd2 + 1):                  # :174
                            for tx in range(txbnd1, txbnd2 + 1):              # :175
                                it = w2.facet_owns[1, jt] + tx + ty * nbTx    # :176
                                if w2.exch2_myFace[it] != 0:                  # :177
                                    # -    Save to common block this neighbour connection
                                    nbNeighb = nbNeighb + 1                   # :179
                                    ns = min(nbNeighb, W2_maxNeighbours)      # :180; MINMAX-INT: integer (no tie or NaN case)
                                    w2.exch2_neighbourId[ns, is_] = it        # :181
                                    tile_edge2edge[ns, is_] = 10 * i + ii     # :182
                                    for k in range(1, 5):                     # :183-185
                                        w2.exch2_pij[k, ns, is_] = P[k, i, js]
                                    w2.exch2_oi[ns, is_] = oi[i, js]          # :186
                                    w2.exch2_oj[ns, is_] = oj[i, js]          # :187
                                    # Edge length to be exchanged between tiles is & it (:189-196)
                                    tbx, tby = w2.exch2_tBasex[it], w2.exch2_tBasey[it]
                                    itbd1 = min(max(ibnd1, tbx + 1), tbx + tNx)  # MINMAX-INT: integer (no tie or NaN case)
                                    itbd2 = min(max(ibnd2, tbx + 1), tbx + tNx)  # MINMAX-INT: integer (no tie or NaN case)
                                    jtbd1 = min(max(jbnd1, tby + 1), tby + tNy)  # MINMAX-INT: integer (no tie or NaN case)
                                    jtbd2 = min(max(jbnd2, tby + 1), tby + tNy)  # MINMAX-INT: integer (no tie or NaN case)
                                    # (:197-204)
                                    isbd1 = P[1, i, js] * itbd1 + P[2, i, js] * jtbd1 + oi[i, js]
                                    isbd2 = P[1, i, js] * itbd2 + P[2, i, js] * jtbd2 + oi[i, js]
                                    jsbd1 = P[3, i, js] * itbd1 + P[4, i, js] * jtbd1 + oj[i, js]
                                    jsbd2 = P[3, i, js] * itbd2 + P[4, i, js] * jtbd2 + oj[i, js]
                                    w2.exch2_iLo[ns, is_] = isbd1 - ddi - w2.exch2_tBasex[is_]   # :205
                                    w2.exch2_iHi[ns, is_] = isbd2 + ddi - w2.exch2_tBasex[is_]   # :206
                                    w2.exch2_jLo[ns, is_] = jsbd1 - ddj - w2.exch2_tBasey[is_]   # :207
                                    w2.exch2_jHi[ns, is_] = jsbd2 + ddj - w2.exch2_tBasey[is_]   # :208
            w2.exch2_nNeighbours[is_] = nbNeighb                              # :220
            if prtFlag:                                                       # :221-234
                io.write(w2.W2_oUnit, fortran_write(
                    "(A,I8,A,I3,A,4(A,I2))", "Tile", is_, " : nbNeighb=", nbNeighb, " ; is-at-Facet-Edge:",
                    " N=", w2.exch2_isNedge[is_], " , S=", w2.exch2_isSedge[is_],
                    " , E=", w2.exch2_isEedge[is_], " , W=", w2.exch2_isWedge[is_]))
                for ns in range(1, min(nbNeighb, W2_maxNeighbours) + 1):  # MINMAX-INT: integer (no tie or NaN case)
                    io.write(w2.W2_oUnit, fortran_write(
                        "(A,I3,A,I8,2(A,2I6),A,4I3,A,2I6,A)", " ns:", ns, " it=", w2.exch2_neighbourId[ns, is_],
                        ", iLo,iHi=", w2.exch2_iLo[ns, is_], w2.exch2_iHi[ns, is_],
                        ", jLo,jHi=", w2.exch2_jLo[ns, is_], w2.exch2_jHi[ns, is_]))

    # -  Check nbNeighb =< W2_maxNeighbours (:241-261)
    nbNeighb = 0
    it = 0
    for is_ in range(1, w2.exch2_nTiles + 1):
        if w2.exch2_nNeighbours[is_] > nbNeighb:
            nbNeighb = w2.exch2_nNeighbours[is_]
            it = is_
    msgBuf = internal_write("(A,I5,A,I3)", "current Max.Nb.Neighbours (e.g., on tile", it, " ) =", nbNeighb)  # :249-250
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                       # :251
    if nbNeighb > W2_maxNeighbours:                                           # :252-261
        raise RuntimeError(f"W2_SET_TILE2TILES: Max.Nb.Neighbours={nbNeighb:4d} >{W2_maxNeighbours:4d} "
                           "=W2_maxNeighbours; ABNORMAL END: S/R W2_SET_TILE2TILES (W2_maxNeighbours)")

    # -    Set exch2_opposingSend(ns,is) (:266-308)
    errCnt = 0
    for is_ in range(1, w2.exch2_nTiles + 1):
        for ns in range(1, w2.exch2_nNeighbours[is_] + 1):
            i = idiv(tile_edge2edge[ns, is_], 10)                             # :269
            ii = imod(tile_edge2edge[ns, is_], 10)                            # :270
            if ii != 0:                                                       # :271-273
                w2.exch2_neighbourDir[ns, is_] = i
            it = w2.exch2_neighbourId[ns, is_]                                # :274
            for nt in range(1, w2.exch2_nNeighbours[it] + 1):                 # :275
                ii = imod(tile_edge2edge[nt, it], 10)                         # :277
                if w2.exch2_neighbourId[nt, it] == is_ and ii == i:           # :278
                    if w2.exch2_opposingSend[ns, is_] == 0:                   # :279
                        w2.exch2_opposingSend[ns, is_] = nt                   # :280
                    else:                                                     # :281-290
                        errCnt = errCnt + 1
            if w2.exch2_opposingSend[ns, is_] == 0:                           # :294-299
                errCnt = errCnt + 1
    if errCnt > 0:                                                            # :303-308
        raise RuntimeError(f" W2_SET_TILE2TILES: found{errCnt:3d} Dbl/No connection; "
                           "ABNORMAL END: S/R W2_SET_TILE2TILES (tile connection)")
    # --  Check opposingSend reciprocity (:310-339)
    errCnt = 0
    for is_ in range(1, w2.exch2_nTiles + 1):
        for ns in range(1, w2.exch2_nNeighbours[is_] + 1):
            it = w2.exch2_neighbourId[ns, is_]                                # :313
            nt = w2.exch2_opposingSend[ns, is_]                               # :314
            ii = w2.exch2_neighbourId[nt, it]                                 # :315
            nn = w2.exch2_opposingSend[nt, it]                                # :316
            if ii != is_ or nn != ns:                                         # :317-331
                errCnt = errCnt + 1
    if errCnt > 0:                                                            # :334-339
        raise RuntimeError(f" W2_SET_TILE2TILES: found{errCnt:3d} opposingSend error; "
                           "ABNORMAL END: S/R W2_SET_TILE2TILES (opposingSend)")
    return w2
