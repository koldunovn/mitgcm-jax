"""CAL_INTMONTHS, CAL_INTYEARS: pkg/cal/cal_intmonths.F, cal_intyears.F @63cdc0b (lane M4ADCOL, pkg/ecco's
ECCO_COST_INIT_FIXED). CAL_INTDAYS (cal_intdays.F, over CAL_SUBDATES) is not ported: ndaysrec is read only by a
'day' gencost (ecco_cost_init_fixed.F:121-123), which raises there."""

from mitjax.pkg.cal.cal_h import idiv, imod, nMonthYear


def cal_IntMonths(*, cal):
    """INTEGER FUNCTION cal_IntMonths( mythid )   @63cdc0b pkg/cal/cal_intmonths.F:3-73

    c     Return the number of calendar months that are affected by the current model integration.

    50-70 on cal.h's modelstartdate / modelenddate."""
    startmonth = imod(idiv(cal.modelStartDate[0], 100), 100)                  # :50
    endmonth = imod(idiv(cal.modelEndDate[0], 100), 100)                      # :51
    startyear = idiv(cal.modelStartDate[0], 10000)                            # :52
    endyear = idiv(cal.modelEndDate[0], 10000)                                # :53
    if startyear != endyear:                                                  # :59-62
        n = (nMonthYear - startmonth + 1) + nMonthYear*(endyear - startyear - 1) + endmonth
    else:
        n = endmonth - startmonth + 1                                         # :64
    if cal.modelEndDate[1] == 0 and imod(cal.modelEndDate[0], 100) == 1:      # :67-70
        n = n - 1
    return n


def cal_IntYears(*, cal):
    """INTEGER FUNCTION cal_IntYears( mythid )   @63cdc0b pkg/cal/cal_intyears.F:3-53, :43-50."""
    n = (idiv(cal.modelEndDate[0], 10000) - idiv(cal.modelStartDate[0], 10000)) + 1
    if (cal.modelEndDate[1] == 0 and imod(cal.modelEndDate[0], 100) == 1
            and imod(idiv(cal.modelEndDate[0], 100), 100) == 1):
        n = n - 1
    return n
