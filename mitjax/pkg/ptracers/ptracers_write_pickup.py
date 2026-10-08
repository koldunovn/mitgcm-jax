"""PTRACERS_WRITE_PICKUP: pkg/ptracers/ptracers_write_pickup.F @63cdc0b (host side: the concrete State's passive
tracers written to the run directory)."""

from mitjax.io.fortran_format import fortran_write
from mitjax.model.src.ini_parms import PRECFLOAT64
from mitjax.pkg.generic_advdiff.gad_h import nSOM
from mitjax.pkg.mdsio.mdsio_write_field import mds_wr_metafiles
from mitjax.pkg.rw.write_rec import WRITE_REC_3D_RL

_OPT = "PTRACERS_OPTIONS.h"
oneRL = 1.0                                 # EEPARAMS.h  oneRL = 1.0 _d 0


def ptracers_write_pickup(permCheckPoint, suff, myTime, myIter, *, cfg, ptr, io, state, mds):
    """PTRACERS_WRITE_PICKUP( permCheckPoint, suff, myTime, myIter, myThid )
    @63cdc0b pkg/ptracers/ptracers_write_pickup.F:8-216

    C     Writes current state of passive tracers to a pickup file
    C     permCheckPoint  :: permanent or a rolling checkpoint
    C     suff            :: suffix for pickup file (eg. ckptA or 0000000010)

    `ptr`: PTRACERS_PARAMS.h; `state`: the State's PTRACERS_FIELDS.h (and SOM) fields. Returns (the STDOUT records it
    prints without the PRINT_MESSAGE prefix, the file names written). :117-176 (PTRACERS_pickup_write_mdsio =
    .NOT.PTRACERS_pickup_write_mnc, ptracers_readparms.F:289-290): 'pickup_ptracers[.suff]', precFloat64, pTracer of every
    tracer in use as records 1..numInUse, then gpTrNm1 of the tracers with AB (WRITE_REC_3D_RL with -j: no meta file),
    then one meta file with the field list 'pTr01   ', 'gPtr01m1' / 'pTr01Nm1' (MDS_WR_METAFILES :171-175). :180-209
    (PTRACERS_ALLOW_DYN_STATE): for each SOM tracer 'pickup_somTRAC'//ioLabel[.suff] with its two messages and the
    nSOM moments as records 1..nSOM. Raise: MNC pickups (:64-112, PTRACERS_pickup_write_mnc)."""
    if ptr.PTRACERS_pickup_write_mnc:                                           # :64-112
        raise NotImplementedError("PTRACERS_WRITE_PICKUP: MNC pickups (PTRACERS_pickup_write_mnc) are not ported")
    from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
    ptf = ptf_of_state(state, ptr)
    Nr = cfg.size.Nr
    num, nUse = ptr.PTRACERS_num, ptr.PTRACERS_numInUse
    listDim = 3*num                                                             # :61  PARAMETER( listDim = 3*PTRACERS_num )
    wrFldList = [" " * 8] * listDim
    lines, written = [], []
    lChar = len(suff.rstrip())                                                  # :116  ILNBLNK(suff)
    # :117  IF ( PTRACERS_pickup_write_mdsio )
    fn = "pickup_ptracers" if lChar == 0 else "pickup_ptracers." + suff[:lChar]   # :119-123
    prec = PRECFLOAT64                                                          # :124
    j = 0                                                                       # :129
    for iTracer in range(1, nUse+1):                                            # :131-138
        j = j + 1
        WRITE_REC_3D_RL(fn, prec, Nr, ptf.pTracer[iTracer-1], -j, myIter, mds=mds, globalFile=io.globalFiles)
        if j <= listDim:
            wrFldList[j-1] = "pTr" + ptr.PTRACERS_ioLabel[iTracer-1] + "   "
    for iTracer in range(1, nUse+1):                                            # :141-155
        if ptr.PTRACERS_AdamsBashGtr[iTracer-1] or ptr.PTRACERS_AdamsBash_Tr[iTracer-1]:
            j = j + 1
            WRITE_REC_3D_RL(fn, prec, Nr, ptf.gpTrNm1[iTracer-1], -j, myIter, mds=mds, globalFile=io.globalFiles)
            if j <= listDim and ptr.PTRACERS_AdamsBashGtr[iTracer-1]:
                wrFldList[j-1] = "gPtr" + ptr.PTRACERS_ioLabel[iTracer-1] + "m1"
            if j <= listDim and ptr.PTRACERS_AdamsBash_Tr[iTracer-1]:
                wrFldList[j-1] = "pTr" + ptr.PTRACERS_ioLabel[iTracer-1] + "Nm1"
    nWrFlds = j                                                                 # :157
    if nWrFlds > listDim:                                                       # :158-167
        raise ValueError(f"PTRACERS_WRITE_PICKUP: trying to write {nWrFlds:5d} fields\nABNORMAL END: S/R "
                         "PTRACERS_WRITE_PICKUP (list-size Pb)")
    glf = io.globalFiles                                                        # :169
    timList = [float(myTime)]                                                   # :170
    mds_wr_metafiles(fn, prec, glf, False, 0, 0, Nr, " ", nWrFlds, wrFldList, 1, timList, oneRL, j, myIter,
                     mds=mds)                                                   # :171-175
    written.append(fn)
    if cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", _OPT):                         # :180-209
        for iTracer in range(1, nUse+1):
            if not ptr.PTRACERS_SOM_Advection[iTracer-1]:
                continue
            lab = ptr.PTRACERS_ioLabel[iTracer-1]
            fn = "pickup_somTRAC" + lab if lChar == 0 else "pickup_somTRAC" + lab + "." + suff[:lChar]   # :184-189
            lines.append(fortran_write("(A,I4,A)", "PTRACERS_WRITE_PICKUP: iTracer =", iTracer,
                                       " : writing 2nd-order moments"))          # :191-194
            lines.append(fortran_write("(A,A)", " to file: ", fn))              # :195-198
            prec = PRECFLOAT64                                                  # :201
            for n in range(1, nSOM+1):                                          # :203-207
                iRec = n
                WRITE_REC_3D_RL(fn, prec, Nr, ptf.som[iTracer-1][n-1], iRec, myIter, mds=mds,
                                globalFile=io.globalFiles)
            written.append(fn)
    return lines, written
