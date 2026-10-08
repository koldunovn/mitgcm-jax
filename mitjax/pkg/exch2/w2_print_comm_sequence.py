"""W2_PRINT_COMM_SEQUENCE (pkg/exch2/w2_print_comm_sequence.F @63cdc0b): write the communication sequence of the
W2 topology (which points each tile sends and receives, cell-centred scalar exchange with corners).

    @63cdc0b pkg/exch2/w2_print_comm_sequence.F:8-152

W2_PRINT_PREFIX is #undef'd in the file itself (:2), so the records are written with `WRITE(W2_oUnit,'(A)')
msgBuf(1:iLen)` (:100-101, :110-111, :144-145), without the PRINT_MESSAGE prefix.
"""

from mitjax.eesupp.print import ilnblnk, internal_write
from mitjax.pkg.exch2.exch2_get_scal_bounds import exch2_get_scal_bounds


def w2_print_comm_sequence(w2, *, cfg, io, myThid=1):
    """W2_PRINT_COMM_SEQUENCE( myThid ) on the W2Common `w2`."""
    sz = cfg.size

    def write_trimmed(msgBuf):
        iLen = ilnblnk(msgBuf)
        io.write(w2.W2_oUnit, msgBuf[:iLen])

    # Send loop for cell centered (:62-115)
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            myTileId = w2.W2_myTileList[bi, bj]                               # :64
            nN = w2.exch2_nNeighbours[myTileId]                               # :65
            sourceProc = w2.W2_tileProc[myTileId]                             # :66
            for N in range(1, nN + 1):
                targetTile = w2.exch2_neighbourId[N, myTileId]                # :68
                targetProc = w2.W2_tileProc[targetTile]                       # :69
                tN = w2.exch2_opposingSend[N, myTileId]                       # :70
                pi = (w2.exch2_pij[1, N, myTileId], w2.exch2_pij[2, N, myTileId])   # :71-72
                pj = (w2.exch2_pij[3, N, myTileId], w2.exch2_pij[4, N, myTileId])   # :73-74
                oi = w2.exch2_oi[N, myTileId]                                 # :75
                oj = w2.exch2_oj[N, myTileId]                                 # :76
                targetIlo, targetIhi, targetJlo, targetJhi, iStride, jStride = exch2_get_scal_bounds(
                    "T ", sz.OLx, True, targetTile, tN, w2=w2)                # :77-82
                itb = w2.exch2_tBasex[targetTile]                             # :84
                jtb = w2.exch2_tBasey[targetTile]                             # :85
                isb = w2.exch2_tBasex[myTileId]                               # :86
                jsb = w2.exch2_tBasey[myTileId]                               # :87
                sourceIlo = pi[0] * (targetIlo + itb) + pi[1] * (targetJlo + jtb) + oi - isb   # :88
                sourceJlo = pj[0] * (targetIlo + itb) + pj[1] * (targetJlo + jtb) + oj - jsb   # :89
                sourceIhi = pi[0] * (targetIhi + itb) + pi[1] * (targetJhi + jtb) + oi - isb   # :90
                sourceJhi = pj[0] * (targetIhi + itb) + pj[1] * (targetJhi + jtb) + oj - jsb   # :91
                # Tile XX sends to points i=ilo:ihi,j=jlo:jhi in tile YY
                msgBuf = internal_write("(A,I8,A,I8,A,4(A,I4))", "Tile", myTileId, " (pr=", sourceProc, ")",
                                        " sends pts i=", sourceIlo, ":", sourceIhi,
                                        ", j=", sourceJlo, ":", sourceJhi)     # :93-96
                write_trimmed(msgBuf)                                         # :100-101
                msgBuf = internal_write("(26X,4(A,I4),A,I8,A,I8,A)", "    to pts i=", targetIlo, ":", targetIhi,
                                        ", j=", targetJlo, ":", targetJhi,
                                        " in tile ", targetTile, " (pr=", targetProc, ")")   # :103-106
                write_trimmed(msgBuf)                                         # :110-111

    # Recv loop for cell centered (:118-149)
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            myTileId = w2.W2_myTileList[bi, bj]                               # :120
            nN = w2.exch2_nNeighbours[myTileId]                               # :121
            sourceProc = w2.W2_tileProc[myTileId]                             # :122
            for N in range(1, nN + 1):
                targetTile = w2.exch2_neighbourId[N, myTileId]                # :124
                targetProc = w2.W2_tileProc[targetTile]                       # :125
                # Find entry for tile targetTile entry that sent to this edge.
                tN = w2.exch2_opposingSend[N, myTileId]                       # :127  # noqa: F841
                # Get the range of points associated with that entry
                targetIlo, targetIhi, targetJlo, targetJhi, iStride, jStride = exch2_get_scal_bounds(
                    "T ", sz.OLx, True, myTileId, N, w2=w2)                   # :129-134
                # Tile XX receives points i=ilo:ihi,j=jlo:jhi in tile YY
                msgBuf = internal_write("(A,I8,A,I8,A,4(A,I4),A,I8,A,I8,A)", "Tile", myTileId, " (pr=", sourceProc,
                                        ")", " recv pts i=", targetIlo, ":", targetIhi,
                                        ", j=", targetJlo, ":", targetJhi,
                                        " from tile", targetTile, " (pr=", targetProc, ")")   # :136-140
                write_trimmed(msgBuf)                                         # :144-145
    return w2
