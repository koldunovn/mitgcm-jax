"""CAL_COMPDATES: pkg/cal/cal_compdates.F @63cdc0b (lane M4ADCOL, pkg/ecco's COST_AVERAGESFLAGS)."""


def cal_CompDates(date_a, date_b, *, cal=None):
    """LOGICAL FUNCTION cal_CompDates( date_a, date_b, mythid )   @63cdc0b pkg/cal/cal_compdates.F:3-51

    c     Compare two calendar dates or time intervals.

    :41-48: .true. when all four elements are equal. Host-side (4-lists of ints)."""
    return (date_a[0] == date_b[0] and date_a[1] == date_b[1] and date_a[2] == date_b[2]
            and date_a[3] == date_b[3])
