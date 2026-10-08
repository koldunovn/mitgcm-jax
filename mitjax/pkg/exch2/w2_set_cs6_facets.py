"""W2_SET_CS6_FACETS (pkg/exch2/w2_set_cs6_facets.F @63cdc0b): the 6-facet cube topology (facet links) and the facet
dimensions.

    @63cdc0b pkg/exch2/w2_set_cs6_facets.F:9-184

Ported as the cubed-sphere experiments execute it (adjustment.cs-32x32x1, solid-body.cs-32x32x1, advect_cs,
global_ocean.cs32x15: no facet_dims in data.exch2): the facet links of the regular cube (:59-79), facet dims unset so
the single size is fitted from the tile count (:99-110), the other dims derived from the connection graph
(:134-163). REAL*4 arithmetic as declared: `facet_link` and `tmpVar` are Real*4, `0.4` etc. and `6.` are REAL*4
literals, FLOAT gives REAL*4, SQRT of a REAL*4 is REAL*4. The error branches (:53-56, :85-94, :111-121 with :124-127,
:165-181) raise with the Fortran message.
"""

import numpy as np

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNbFacets, idiv, imod, int_, nint, r4

edge = ("N", "S", "E", "W")         # :43  DATA edge / 'N' , 'S' , 'E' , 'W' /


def _float(n):
    """FLOAT(n): INTEGER -> default REAL (REAL*4)."""
    return np.float32(n)


def w2_set_cs6_facets(w2, *, cfg, io, myThid=1):
    """W2_SET_CS6_FACETS( myThid ) on the W2Common `w2`."""
    sz = cfg.size
    msgBuf = internal_write("(2A,I3,A)", "W2_SET_CS6_FACETS:", " preDefTopol=", w2.preDefTopol, " selected")  # :45-46
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                       # :47
    prtFlag = abs(w2.W2_printMsg) >= 2 or (w2.W2_printMsg != 0 and io.myProcId == 0)                      # :48-49

    # --   Number of facets (:52-56)
    w2.nFacets = 6
    if w2.nFacets > W2_maxNbFacets:
        raise RuntimeError("ABNORMAL END: S/R W2_SET_CS6_FACETS (nFacets>maxNbFacets)")

    # --   Facet Edge connections ( edges order: N,S,E,W <==> 1,2,3,4 ) (:59-79); REAL*4 sums stored in Real*4
    for j in range(1, w2.nFacets + 1):
        if imod(j, 2) == 1:
            jj = j + 2
            w2.facet_link[1, j] = r4(0.4) + _float(1 + imod(jj + 5, 6))       # :62
            jj = j - 1
            w2.facet_link[2, j] = r4(0.1) + _float(1 + imod(jj + 5, 6))       # :64
            jj = j + 1
            w2.facet_link[3, j] = r4(0.4) + _float(1 + imod(jj + 5, 6))       # :66
            jj = j - 2
            w2.facet_link[4, j] = r4(0.1) + _float(1 + imod(jj + 5, 6))       # :68
        else:
            jj = j + 1
            w2.facet_link[1, j] = r4(0.2) + _float(1 + imod(jj + 5, 6))       # :71
            jj = j - 2
            w2.facet_link[2, j] = r4(0.3) + _float(1 + imod(jj + 5, 6))       # :73
            jj = j + 2
            w2.facet_link[3, j] = r4(0.2) + _float(1 + imod(jj + 5, 6))       # :75
            jj = j - 1
            w2.facet_link[4, j] = r4(0.3) + _float(1 + imod(jj + 5, 6))       # :77

    # --   facet dimension: take the 1rst 3 numbers from facet_dims (:82-94)
    nRd = w2.facet_dims[1]
    nGr = w2.facet_dims[2]
    nBl = w2.facet_dims[3]
    for j in range(4, W2_maxNbFacets * 2 + 1):
        if w2.facet_dims[j] != 0:
            raise RuntimeError("W2_SET_CS6_FACETS: no more than 3 dims (nRd,nGr,nBl) expected for CS-6 Topol; "
                               "ABNORMAL END: S/R W2_SET_CS6_FACETS: allows 3 dims only")
    if nRd > 0 and nGr + nBl == 0:                                            # :95
        # -    Only 1rst dim is set: assuming a regular Cube
        nGr = nRd                                                             # :97
        nBl = nRd                                                             # :98
    elif nRd + nGr + nBl == 0:                                                # :99
        # -    try to get cube size from number of tiles, assuming a regular Cube
        nGr = w2.exch2_nTiles * sz.sNx * sz.sNy                               # :101
        tmpVar = np.float32(_float(nGr) / r4(6.0))                            # :102  FLOAT(nGr)/6.
        tmpVar = np.sqrt(tmpVar, dtype=np.float32)                            # :103  SQRT (REAL*4)
        nRd = nint(tmpVar)                                                    # :104
        if nRd * nRd * 6 == nGr:                                              # :105
            nGr = nRd                                                         # :106
            nBl = nRd                                                         # :107
            msgBuf = internal_write("(2A,I5)", "W2_SET_CS6_FACETS:", " facet-dims Unset; assume nRd=nGr=nBl=",
                                    nRd)                                                           # :108-109
            print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                       # :110
        else:                                                                 # :111-122
            raise RuntimeError(f" nTiles*sNx*sNy={w2.exch2_nTiles:8d} x{sz.sNx:4d} x{sz.sNy:4d} ={nGr:10d} not "
                               f"equal to 6 x{nRd:6d}^2; W2_SET_CS6_FACETS: facet-dims Unset; attempt to fit "
                               "single dim FAIL")
    if nRd * nGr * nBl == 0:                                                  # :124-127
        raise RuntimeError("ABNORMAL END: S/R W2_SET_CS6_FACETS (Dims are missing)")

    # --   Set facet dimension : 1rst 3 are known (:130-132)
    w2.facet_dims[1] = nRd
    w2.facet_dims[2] = nGr
    w2.facet_dims[3] = nBl
    # -    Derive the other using from connection graph (topology) (:134-163)
    setDims = 3
    addDims = 1
    while addDims > 0:
        addDims = 0
        for j in range(2, w2.nFacets + 1):
            for i in range(1, 5):
                # -    connected to:
                jj = int_(w2.facet_link[i, j])                                # :141
                ii = imod(nint(w2.facet_link[i, j] * r4(10.0)), 10)           # :142
                if 1 <= jj <= w2.nFacets and 1 <= ii <= 4:                    # :143-144
                    # -    Length of N or S Edge = x-size, E or W Edge = y-size
                    lo = 2 * (j - 1) + idiv(i + 1, 2)                         # :146
                    # -    Corresponding Edge length
                    ll = 2 * (jj - 1) + idiv(ii + 1, 2)                       # :148
                    if w2.facet_dims[lo] == 0 and w2.facet_dims[ll] > 0:      # :149
                        addDims = addDims + 1
                        w2.facet_dims[lo] = w2.facet_dims[ll]
                        if prtFlag:                                           # :152-157
                            msgBuf = internal_write("(A,I3,3A,2(I4,A),I3,3A,I8)", " facet", j, ".", edge[i - 1],
                                                   " set dim", lo, " = dim", ll, " from", jj, ".", edge[ii - 1],
                                                   " :", w2.facet_dims[ll])
                            print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)
        setDims = setDims + addDims                                           # :162

    if setDims != w2.nFacets * 2:                                             # :165-181
        raise RuntimeError(f" W2_SET_CS6_FACETS:{w2.nFacets * 2 - setDims:3d} facet-dims left Unset; "
                           "ABNORMAL END: S/R W2_SET_CS6_FACETS (unset facet dims)")
    return w2
