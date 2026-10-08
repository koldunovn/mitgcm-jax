"""EXCH2_GET_UV_BOUNDS (pkg/exch2/exch2_get_uv_bounds.F @63cdc0b): the index ranges and step of the overlap regions
of target tile `tgTile` updated by the exchange with its neighbour entry `tgNb`, two-component (UV) version, and the
target-to-source index offsets of both components.

    @63cdc0b pkg/exch2/exch2_get_uv_bounds.F:7-264

Host-side integer routine, called by the table builder of the exch2 C-grid vector exchanges
(mitjax/pkg/exch2/exch2_rx2_cube.py). fCode 'Cg' (C grid) and 'Ag' are the codes the Fortran accepts; any other
stops (:259-261). Returns (tIlo1, tIhi1, tJlo1, tJhi1, tIlo2, tIhi2, tJlo2, tJhi2, tiStride, tjStride, e2_oi1,
e2_oj1, e2_oi2, e2_oj2); an output the Fortran leaves unset (no branch taken) is None.
"""


def exch2_get_uv_bounds(fCode, eWdth, updateCorners, tgTile, tgNb, *, w2):
    """EXCH2_GET_UV_BOUNDS( fCode, eWdth, updateCorners, tgTile, tgNb, tIlo1, ..., e2_oj2, myThid ) on the
    W2Common `w2`."""
    tIlo1 = tIhi1 = tJlo1 = tJhi1 = tiStride = tjStride = None
    # ---  Initialise index range from Topology values (:71-81)
    tIlo = w2.exch2_iLo[tgNb, tgTile]
    tIhi = w2.exch2_iHi[tgNb, tgTile]
    tJlo = w2.exch2_jLo[tgNb, tgTile]
    tJhi = w2.exch2_jHi[tgNb, tgTile]
    soNb = w2.exch2_opposingSend[tgNb, tgTile]
    soTile = w2.exch2_neighbourId[tgNb, tgTile]
    e2_oi1 = w2.exch2_oi[soNb, soTile]
    e2_oj1 = w2.exch2_oj[soNb, soTile]
    e2_pij = [None] + [w2.exch2_pij[i, soNb, soTile] for i in range(1, 5)]     # e2_pij(1:4)

    # ---  Expand index range according to exchange-Width "eWdth"
    if tIlo == tIhi and tIlo == 0:                                            # :84  west edge overlap
        tIlo1 = 1 - eWdth                                                     # :86
        tIhi1 = 0                                                             # :87
        tiStride = 1                                                          # :88
        tjStride = 1 if tJlo <= tJhi else -1                                  # :89-93
        if updateCorners:                                                     # :94-100
            tJlo1 = tJlo - tjStride * (eWdth - 1)
            tJhi1 = tJhi + tjStride * (eWdth - 1)
        else:
            tJlo1 = tJlo + tjStride
            tJhi1 = tJhi - tjStride
    if tIlo == tIhi and tIlo > 1:                                             # :102  east edge overlap
        tIlo1 = tIlo                                                          # :104
        tIhi1 = tIhi + eWdth - 1                                              # :105
        tiStride = 1                                                          # :106
        tjStride = 1 if tJlo <= tJhi else -1                                  # :107-111
        if updateCorners:                                                     # :112-118
            tJlo1 = tJlo - tjStride * (eWdth - 1)
            tJhi1 = tJhi + tjStride * (eWdth - 1)
        else:
            tJlo1 = tJlo + tjStride
            tJhi1 = tJhi - tjStride
    if tJlo == tJhi and tJlo == 0:                                            # :120  south edge overlap
        tJlo1 = 1 - eWdth                                                     # :122
        tJhi1 = 0                                                             # :123
        tjStride = 1                                                          # :124
        tiStride = 1 if tIlo <= tIhi else -1                                  # :125-129
        if updateCorners:                                                     # :130-136
            tIlo1 = tIlo - tiStride * (eWdth - 1)
            tIhi1 = tIhi + tiStride * (eWdth - 1)
        else:
            tIlo1 = tIlo + tiStride
            tIhi1 = tIhi - tiStride
    if tJlo == tJhi and tJlo > 1:                                             # :138  north edge overlap
        tJlo1 = tJlo                                                          # :140
        tJhi1 = tJhi + eWdth - 1                                              # :141
        tjStride = 1                                                          # :142
        tiStride = 1 if tIlo <= tIhi else -1                                  # :143-147
        if updateCorners:                                                     # :148-154
            tIlo1 = tIlo - tiStride * (eWdth - 1)
            tIhi1 = tIhi + tiStride * (eWdth - 1)
        else:
            tIlo1 = tIlo + tiStride
            tIhi1 = tIhi - tiStride

    # ---  copy to 2nd set of indices (:158-163)
    tIlo2 = tIlo1
    tIhi2 = tIhi1
    tJlo2 = tJlo1
    tJhi2 = tJhi1
    e2_oi2 = e2_oi1
    e2_oj2 = e2_oj1

    if fCode == "Cg":                                                         # :165
        # ---  half grid-cell location with inverse index relation (:172-185)
        if e2_pij[1] == -1:
            e2_oi1 = e2_oi1 + 1
        if e2_pij[3] == -1:
            e2_oj1 = e2_oj1 + 1
        if e2_pij[2] == -1:
            e2_oi2 = e2_oi2 + 1
        if e2_pij[4] == -1:
            e2_oj2 = e2_oj2 + 1
        # ---  adjust index lower and upper bounds (fct of updateCorners)
        if updateCorners:                                                     # :188
            if e2_pij[1] == -1 or e2_pij[3] == -1:                            # :192
                tIlo1 = tIlo1 + 1
            if e2_pij[2] == -1 or e2_pij[4] == -1:                            # :193
                tJlo2 = tJlo2 + 1
            # ---  Avoid updating (some) tile-corner halo region if across faces (:208-243; :196-207, :220-231 are
            # commented out in the Fortran)
            if tIlo == tIhi and tIlo > 1:                                     # :208
                if w2.exch2_isSedge[tgTile] == 1:                             # :209-213  East edge touching S edge
                    tJlo1 = tJlo + 1
                    tJlo2 = tJlo + 1
                if w2.exch2_isNedge[tgTile] == 1:                             # :214-218  East edge touching N edge
                    tJhi1 = tJhi - 1
                    tJhi2 = tJhi
            if tJlo == tJhi and tJlo > 1:                                     # :232
                if w2.exch2_isWedge[tgTile] == 1:                             # :233-237  North edge touching W edge
                    tIlo1 = tIlo + 1
                    tIlo2 = tIlo + 1
                if w2.exch2_isEedge[tgTile] == 1:                             # :238-242  North edge touching E edge
                    tIhi1 = tIhi
                    tIhi2 = tIhi - 1
        else:                                                                 # :245-255
            if e2_pij[1] == -1 or e2_pij[3] == -1:
                tIlo1 = tIlo1 + 1
                tIhi1 = tIhi1 + 1
            if e2_pij[2] == -1 or e2_pij[4] == -1:
                tJlo2 = tJlo2 + 1
                tJhi2 = tJhi2 + 1
    elif fCode != "Ag":                                                       # :259-261
        raise RuntimeError("ABNORMAL END: S/R EXCH2_GET_UV_BOUNDS (wrong fCode)")
    return (tIlo1, tIhi1, tJlo1, tJhi1, tIlo2, tIhi2, tJlo2, tJhi2, tiStride, tjStride,
            e2_oi1, e2_oj1, e2_oi2, e2_oj2)
