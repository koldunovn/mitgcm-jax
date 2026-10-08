"""W2_SET_MAP_TILES (pkg/exch2/w2_set_map_tiles.F @63cdc0b): tile mapping within each facet and the global IO map.

    @63cdc0b pkg/exch2/w2_set_map_tiles.F:14-217

Ported as executed by global_ocean.90x40x15 and the cubed-sphere experiments: tile sizes divide the facet sizes, tile
count matches SIZE.h, no blank tiles; W2_mapIO = -1 (old format, facets side by side in X: :136-138, :189-192) and
W2_mapIO = 1 (solid-body.cs-32x32x1's data.exch2: compact map piled in Y, FIND_GCD_N :102-119, :142-145, :198-202).
W2_mapIO = 0 (:139-141, :193-197) is not set by any ported experiment and raises; the error branches raise with the
Fortran message.
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNbFacets, idiv, imod


def w2_set_map_tiles(w2, *, cfg, io, myThid=1):
    """W2_SET_MAP_TILES( myThid ) on the W2Common `w2`."""
    sz = cfg.size
    msgBuf = internal_write("(2A)", "W2_SET_MAP_TILES:", " tile mapping within facet and global Map:")   # :51-52
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                 # :53
    prtFlag = abs(w2.W2_printMsg) >= 2 or (w2.W2_printMsg != 0 and io.myProcId == 0)                # :54-55

    tNx = sz.sNx                                                              # :57
    tNy = sz.sNy                                                              # :58
    # --   Check that tile dims divide facet dims (:60-90)
    errCnt = 0
    tCnt = 0
    nbPts = 0
    for j in range(1, w2.nFacets + 1):
        fNx = w2.facet_dims[2 * j - 1]                                        # :64
        fNy = w2.facet_dims[2 * j]                                            # :65
        nbTx = idiv(fNx, tNx)                                                 # :66
        nbTy = idiv(fNy, tNy)                                                 # :67
        if nbTx * tNx != fNx:                                                 # :68-73
            errCnt = errCnt + 1
        if nbTy * tNy != fNy:                                                 # :74-79
            errCnt = errCnt + 1
        w2.facet_owns[1, j] = tCnt + 1                                        # :80
        tCnt = tCnt + nbTx * nbTy                                             # :81
        w2.facet_owns[2, j] = tCnt                                            # :82
        nbPts = nbPts + fNx * fNy                                             # :83
    if errCnt > 0:                                                            # :85-90
        raise RuntimeError(f" W2_SET_MAP_TILES: found{errCnt:3d} Fatal errors (facet size not a multiple of the "
                           "tile size); ABNORMAL END: S/R W2_SET_MAP_TILES")
    # --   Check that domain size and (SIZE.h + blankList) match (:92-100)
    if tCnt != w2.exch2_nTiles:
        raise RuntimeError(f"W2_SET_MAP_TILES: Domain Total # of tiles ={tCnt:8d} does not match "
                           f"(SIZE.h+blankList)={w2.exch2_nTiles:8d}; ABNORMAL END: S/R W2_SET_MAP_TILES")

    if w2.W2_mapIO == 1:                                                      # :102
        # --   Compact IO map (mostly in Y dir): search for Greatest Common Divisor of all x-size
        k = 0                                                                 # :105
        nnx = [0] * (W2_maxNbFacets + 1)                                      # :45 INTEGER nnx(W2_maxNbFacets)
        nnx[1] = 0                                                            # :106
        for j in range(1, w2.nFacets + 1):                                    # :107-113
            if w2.facet_dims[2 * j - 1] > 0:                                  # skip empty facet
                k = k + 1
                nnx[k] = idiv(w2.facet_dims[2 * j - 1], tNx)
        divide = find_gcd_n(nnx, k)                                           # :114
        w2.W2_mapIO = divide * tNx                                            # :115
        msgBuf = internal_write("(A,2(I5,A))", " W2_mapIO =", w2.W2_mapIO, " (=", divide, "*sNx)")   # :116-117
        print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                           # :118

    # --   Global Map size: facets stacked in x direction (:123-128)
    w2.exch2_xStack_Nx = 0
    w2.exch2_xStack_Ny = 0
    for j in range(1, w2.nFacets + 1):
        w2.exch2_xStack_Nx = w2.exch2_xStack_Nx + w2.facet_dims[2 * j - 1]
        w2.exch2_xStack_Ny = max(w2.exch2_xStack_Ny, w2.facet_dims[2 * j])  # MINMAX-INT: integer (no tie or NaN case)
    # facets stacked in y direction (:130-135)
    w2.exch2_yStack_Nx = 0
    w2.exch2_yStack_Ny = 0
    for j in range(1, w2.nFacets + 1):
        w2.exch2_yStack_Nx = max(w2.exch2_yStack_Nx, w2.facet_dims[2 * j - 1])  # MINMAX-INT: integer (no tie or NaN case)
        w2.exch2_yStack_Ny = w2.exch2_yStack_Ny + w2.facet_dims[2 * j]
    if w2.W2_mapIO == -1:                                                     # :136
        w2.exch2_global_Nx = w2.exch2_xStack_Nx                               # :137
        w2.exch2_global_Ny = w2.exch2_xStack_Ny                               # :138
    elif w2.W2_mapIO == 0:                                                    # :139-141
        raise NotImplementedError("W2_SET_MAP_TILES: W2_mapIO=0 (one long line) is not ported")
    else:
        w2.exch2_global_Nx = w2.W2_mapIO                                      # :143
        w2.exch2_global_Ny = idiv(nbPts, w2.W2_mapIO)                         # :144
    msgBuf = internal_write("(A,2(A,I8))", " Global Map (IO):", " X-size=", w2.exch2_global_Nx,
                            " , Y-size=", w2.exch2_global_Ny)                                       # :146-147
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                               # :148

    # --   Set tiles mapping within facet (sub-domain) and within Global Map (:151-214)
    msgBuf = internal_write("(2A)", "W2_SET_MAP_TILES:", " tile offset within facet and global Map:")    # :151-152
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                 # :153
    tId = 0
    nbPts = 0
    fBaseX = 0
    fBaseY = 0
    for j in range(1, w2.nFacets + 1):
        fNx = w2.facet_dims[2 * j - 1]                                        # :159
        fNy = w2.facet_dims[2 * j]                                            # :160
        nbTx = idiv(fNx, tNx)                                                 # :161
        nbTy = idiv(fNy, tNy)                                                 # :162
        io.write(w2.W2_oUnit, fortran_write("(A,I3,2(A,I6),A,I8,2(A,I4),A)", "- facet", j, " : X-size=", fNx,
                                            " , Y-size=", fNy, " ;", nbTx * nbTy, " tiles (Tx,Ty=", nbTx, ",",
                                            nbTy, ")"))                                             # :163-165
        for ty in range(1, nbTy + 1):
            for tx in range(1, nbTx + 1):
                tId = tId + 1                                                 # :169
                # --   Tags blank tile by removing facet # (exch2_myFace) but keeps its location
                tileIsActive = True                                           # :171
                for k in range(1, w2.nBlankTiles + 1):                        # :172-174
                    if w2.blankList[k] == tId:
                        tileIsActive = False
                if tileIsActive:
                    w2.exch2_myFace[tId] = j                                  # :175
                w2.exch2_mydNx[tId] = fNx                                     # :176
                w2.exch2_mydNy[tId] = fNy                                     # :177
                w2.exch2_tNx[tId] = tNx                                       # :178
                w2.exch2_tNy[tId] = tNy                                       # :179
                w2.exch2_tBasex[tId] = (tx - 1) * tNx                         # :180
                w2.exch2_tBasey[tId] = (ty - 1) * tNy                         # :181
                # --   Global IO Mappings: these are for OBCS (vertical slices) (:184-187)
                w2.exch2_txXStackLo[tId] = 1 + w2.exch2_tBasex[tId] + fBaseX
                w2.exch2_tyXStackLo[tId] = 1 + w2.exch2_tBasey[tId]
                w2.exch2_txYStackLo[tId] = 1 + w2.exch2_tBasex[tId]
                w2.exch2_tyYStackLo[tId] = 1 + w2.exch2_tBasey[tId] + fBaseY
                # and these for global files (3d files/horizontal 2d files)
                if w2.W2_mapIO == -1:                                         # :189
                    w2.exch2_txGlobalo[tId] = 1 + w2.exch2_tBasex[tId] + fBaseX   # :191
                    w2.exch2_tyGlobalo[tId] = 1 + w2.exch2_tBasey[tId]            # :192
                elif w2.W2_mapIO == 0:                                        # :193-197
                    raise NotImplementedError("W2_SET_MAP_TILES: W2_mapIO=0 (one long line) is not ported")
                else:
                    # Compact format: piled in the Y direction (:199-202)
                    ii = nbPts + w2.exch2_tBasex[tId] + w2.exch2_tBasey[tId] * fNx
                    w2.exch2_txGlobalo[tId] = 1 + imod(ii, w2.W2_mapIO)
                    w2.exch2_tyGlobalo[tId] = 1 + idiv(ii, w2.W2_mapIO)
                if prtFlag:                                                   # :204-208
                    io.write(w2.W2_oUnit, fortran_write(
                        "(A,I8,3(A,I3),2A,2I5,2A,2I8)", "  tile", tId, " on facet", w2.exch2_myFace[tId], " (",
                        tx, ",", ty, "):", " offset=", w2.exch2_tBasex[tId], w2.exch2_tBasey[tId], " ;",
                        " on Glob.Map=", w2.exch2_txGlobalo[tId], w2.exch2_tyGlobalo[tId]))
        fBaseX = fBaseX + fNx                                                 # :211
        fBaseY = fBaseY + fNy                                                 # :212
        nbPts = nbPts + fNx * fNy                                             # :213
    return w2


def find_gcd_n(fldList, nFld):
    """FIND_GCD_N( fldList, nFld ) (w2_set_map_tiles.F:224-290): the greatest common divisor of fldList(1:nFld)
    (a list indexed from 1; divided in place and multiplied back, :266-282, as the Fortran does)."""
    mnFld = fldList[1]                                                        # :250
    for j in range(1, nFld + 1):                                              # :251-253
        mnFld = min(mnFld, fldList[j])                                        # MINMAX-INT: integer (no tie or NaN case)
    if mnFld > 1:                                                             # :256
        divide = 1
        ii = 2
        while ii <= mnFld:                                                    # :259
            flag = True
            for j in range(1, nFld + 1):
                flag = flag and (imod(fldList[j], ii) == 0)                   # :263
            if flag:
                divide = divide * ii                                          # :266
                for j in range(1, nFld + 1):
                    fldList[j] = idiv(fldList[j], ii)                         # :268
                mnFld = idiv(mnFld, ii)                                       # :272
            else:
                ii = ii + 2                                                   # :274
                if ii == 4:
                    ii = 3                                                    # :275
        for j in range(1, nFld + 1):                                          # :280-282
            fldList[j] = fldList[j] * divide
    else:
        divide = max(0, mnFld)                                                # :284 MINMAX-INT: integer (no tie or NaN case)
    return divide                                                             # :287
