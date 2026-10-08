"""EXCH2_GET_SCAL_BOUNDS (pkg/exch2/exch2_get_scal_bounds.F @63cdc0b): the index range and step of the overlap
region of target tile `tgTile` updated by the exchange with its neighbour entry `tgNb` (scalar field).

    @63cdc0b pkg/exch2/exch2_get_scal_bounds.F:7-128

Host-side integer routine (called by W2_PRINT_COMM_SEQUENCE here; the exchanges themselves are the measured gather
maps of mitjax/eesupp/exch_maps.py). `fCode` is not read by the Fortran (grid-centred scalars only, :18-19).
Returns (tIlo, tIhi, tJlo, tJhi, tiStride, tjStride); a stride the Fortran leaves unset (no branch taken) is None.
"""


def exch2_get_scal_bounds(fCode, eWdth, updateCorners, tgTile, tgNb, *, w2):
    """EXCH2_GET_SCAL_BOUNDS( fCode, eWdth, updateCorners, tgTile, tgNb, tIlo, tIhi, tJlo, tJhi, tiStride,
    tjStride, myThid ) on the W2Common `w2`."""
    tiStride = tjStride = None
    # ---  Initialise index range from Topology values (:52-55)
    tIlo = w2.exch2_iLo[tgNb, tgTile]
    tIhi = w2.exch2_iHi[tgNb, tgTile]
    tJlo = w2.exch2_jLo[tgNb, tgTile]
    tJhi = w2.exch2_jHi[tgNb, tgTile]

    # ---  Expand index range according to exchange-Width "eWdth"
    if tIlo == tIhi and tIlo == 0:                                            # :58  west edge overlap
        tIlo = 1 - eWdth                                                      # :60
        tiStride = 1                                                          # :61
        tjStride = 1 if tJlo <= tJhi else -1                                  # :62-66
        if updateCorners:                                                     # :67-73
            tJlo = tJlo - tjStride * (eWdth - 1)
            tJhi = tJhi + tjStride * (eWdth - 1)
        else:
            tJlo = tJlo + tjStride
            tJhi = tJhi - tjStride
    if tIlo == tIhi and tIlo > 1:                                             # :75  east edge overlap
        tIhi = tIhi + eWdth - 1                                               # :77
        tiStride = 1                                                          # :78
        tjStride = 1 if tJlo <= tJhi else -1                                  # :79-83
        if updateCorners:                                                     # :84-90
            tJlo = tJlo - tjStride * (eWdth - 1)
            tJhi = tJhi + tjStride * (eWdth - 1)
        else:
            tJlo = tJlo + tjStride
            tJhi = tJhi - tjStride
    if tJlo == tJhi and tJlo == 0:                                            # :92  south edge overlap
        tJlo = 1 - eWdth                                                      # :94
        tjStride = 1                                                          # :95
        tiStride = 1 if tIlo <= tIhi else -1                                  # :96-100
        if updateCorners:                                                     # :101-107
            tIlo = tIlo - tiStride * (eWdth - 1)
            tIhi = tIhi + tiStride * (eWdth - 1)
        else:
            tIlo = tIlo + tiStride
            tIhi = tIhi - tiStride
    if tJlo == tJhi and tJlo > 1:                                             # :109  north edge overlap
        tJhi = tJhi + eWdth - 1                                               # :111
        tjStride = 1                                                          # :112
        tiStride = 1 if tIlo <= tIhi else -1                                  # :113-117
        if updateCorners:                                                     # :118-124
            tIlo = tIlo - tiStride * (eWdth - 1)
            tIhi = tIhi + tiStride * (eWdth - 1)
        else:
            tIlo = tIlo + tiStride
            tIhi = tIhi - tiStride
    return tIlo, tIhi, tJlo, tJhi, tiStride, tjStride
