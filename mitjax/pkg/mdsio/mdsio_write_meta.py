"""MDS_WRITE_META: pkg/mdsio/mdsio_write_meta.F @63cdc0b (host-side text output)."""

from mitjax.io.fortran_format import format_E, format_I

precFloat32, precFloat64 = 32, 64            # EEPARAMS.h:185-186 PARAMETER ( precFloat32 = 32 ), ( precFloat64 = 64 )
oneRL = 1.0                                  # EEPARAMS.h  oneRL = 1.0 _d 0


def _1pe(x, w, d):
    """Fortran 1PEw.d (one digit before the point)."""
    return format_E(float(x), w, d, scale=1)


def mds_write_meta(mFileName, dFileName, simulName, titleLine, filePrec, nDims, dimList, map2gl, nFlds, fldList,
                   nTimRec, timList, misVal, nrecords, myIter, *, useCal=False):
    """MDS_WRITE_META( mFileName, dFileName, simulName, titleLine, filePrec, nDims, dimList, map2gl, nFlds, fldList,
    nTimRec, timList, misVal, nrecords, myIter, myThid )   @63cdc0b pkg/mdsio/mdsio_write_meta.F:6-238

    C     | o Write the meta file of an MDS data file

    dimList: [[n, first, last], ...] (Fortran dimList(1:3, j)); fldList: CHARACTER*8 names. ALLOW_CAL (the
    timeStepDate line, `useCal` = True) raises. The file is opened status='unknown' (:102-103): an existing
    file is overwritten, as the Fortran does."""
    lines = []
    iL = len(str(simulName).rstrip())                                     # :106-109
    if iL > 0:
        lines.append(" simulation = { '" + str(simulName)[:iL] + "' };")
    lines.append(" nDims = [ " + format_I(nDims, 3) + " ];")             # :112  '(1X,A,I3,A)'
    ii = 0                                                                # :119-122
    for j in range(nDims):
        ii = max(dimList[j][0], ii)  # MINMAX-INT: integer (no tie or NaN case)
    lines.append(" dimList = [")                                        # :123
    w = 5 if ii < 10000 else 10                                           # :124-142
    for j in range(nDims):
        if j < nDims - 1:
            lines.append(" " + "".join(format_I(v, w) + "," for v in dimList[j]))
        else:
            lines.append(" " + "".join(format_I(v, w) + "," for v in dimList[j][:2]) + format_I(dimList[j][2], w))
    lines.append(" ];")                                                  # :143
    if map2gl[0] != 0 or map2gl[1] != 1:                                  # :145-148 '(1X,2(A,I5),A)'
        lines.append(" map2glob = [ " + format_I(map2gl[0], 5) + "," + format_I(map2gl[1], 5) + " ];")
    if filePrec == precFloat32:                                           # :151-160
        lines.append(" dataprec = [ 'float32' ];")
    elif filePrec == precFloat64:
        lines.append(" dataprec = [ 'float64' ];")
    else:
        raise ValueError(" MDSWRITEMETA: invalid filePrec\nABNORMAL END: S/R MDSWRITEMETA")
    lines.append(" nrecords = [ " + format_I(nrecords, 10) + " ];")       # :165  '(1X,A,I10,A)'
    if myIter >= 0:                                                       # :174-175
        lines.append(" timeStepNumber = [ " + format_I(myIter, 10) + " ];")
    if useCal and myIter >= 0:                                            # :177-199 ALLOW_CAL timeStepDate
        # lane M4COL: `useCal` is then the calendar context (cal: pkg/cal's common block after CAL_INIT_FIXED,
        # baseTime, deltaTClock: PARAMS.h), as the Fortran reads them
        import numpy as np
        from mitjax.pkg.cal.cal_getdate import cal_GetDate
        myTime = np.float64(useCal.baseTime) + np.float64(myIter)*np.float64(useCal.deltaTClock)   # :182
        myDate = cal_GetDate(int(myIter), myTime, cal=useCal.cal)                           # :183
        from mitjax.pkg.cal.cal_h import idiv, imod
        day = imod(myDate[0], 100)                                        # :189-194 (Fortran MOD and /)
        month = imod(idiv(myDate[0], 100), 100)
        year = idiv(myDate[0], 10000)
        second = imod(myDate[1], 100)
        minute = imod(idiv(myDate[1], 100), 100)
        hour = idiv(myDate[1], 10000)
        lines.append(" timeStepDate = [ '" + f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:"   # :195-198
                     f"{second:02d}" + "Z' ];")
    if nTimRec > 0:                                                       # :205-209  '(1P20E20.12)'
        ii = min(nTimRec, 20)  # MINMAX-INT: integer (no tie or NaN case)
        msgBuf = "".join(_1pe(timList[i], 20, 12) for i in range(ii))
        lines.append(" timeInterval = [" + msgBuf[:20 * ii] + " ];")
    if misVal != oneRL:                                                   # :212-215  '(1X,A,1PE21.14,A)'
        lines.append(" missingValue = [ " + _1pe(misVal, 21, 14) + " ];")
    if nFlds > 0:                                                         # :218-224  '(20(A2,A8,A1))'
        lines.append(" nFlds = [ " + format_I(nFlds, 4) + " ];")
        lines.append(" fldList = {")
        recs = [" '" + f"{str(fldList[i]):<8.8s}" + "'" for i in range(nFlds)]
        for n0 in range(0, nFlds, 20):                                    # format reversion: 20 per record
            lines.append("".join(recs[n0:n0 + 20]))
        lines.append(" };")
    iL = len(str(titleLine).rstrip())                                     # :227-230
    if iL > 0:
        lines.append(" /* " + str(titleLine)[:iL] + " */")
    with open(mFileName, "w") as fh:                                      # :102-103 status='unknown', :233 CLOSE
        fh.write("\n".join(lines) + "\n")
