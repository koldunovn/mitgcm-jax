"""CAL_COPYDATE: pkg/cal/cal_copydate.F @63cdc0b (lane M4ADCOL)."""


def cal_CopyDate(indate, *, cal=None):
    """cal_CopyDate( indate, outdate, mythid )   @63cdc0b pkg/cal/cal_copydate.F:3-51: outdate(1:4) = indate(1:4)
    (:45-48). Returns the copy (4-list)."""
    return [indate[0], indate[1], indate[2], indate[3]]
