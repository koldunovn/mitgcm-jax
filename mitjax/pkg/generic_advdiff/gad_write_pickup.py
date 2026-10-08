"""GAD_WRITE_PICKUP: pkg/generic_advdiff/gad_write_pickup.F @63cdc0b (host side: the SOM moments of the concrete State
written to the run directory)."""

from mitjax.model.src.ini_parms import PRECFLOAT64
from mitjax.pkg.generic_advdiff.gad_h import nSOM
from mitjax.pkg.rw.write_rec import WRITE_REC_3D_RL

_OPT = "GAD_OPTIONS.h"      # gad_write_pickup.F:1  #include "GAD_OPTIONS.h"


def gad_write_pickup(suff, myTime, myIter, *, cfg, params, io, state, mds):
    """GAD_WRITE_PICKUP( suff, myTime, myIter, myThid )   @63cdc0b pkg/generic_advdiff/gad_write_pickup.F:7-109

    C     Writes current state of 2nd.Order moments to a pickup file
    C     suff    :: suffix for pickup file (eg. ckptA or 0000000010)
    C     myTime  :: model time
    C     myIter  :: time-step number

    Without GAD_ALLOW_TS_SOM_ADV the routine is empty (:37). :67 `IF ( .TRUE. )` (the pickup_write_mdsio test is
    commented out); :68 lChar = ILNBLNK(suff); :70-85 with tempSOM_Advection the nSOM moments of som_T as records
    1..nSOM of 'pickup_somT.'//suff (or 'pickup_somT' for a blank suffix), precFloat64; :87-102 the same for som_S.
    WRITE_REC_3D_RL with iRec = n > 0 (one meta file per data file, written by MDS_WRITE_FIELD). The MNC message
    (:55-64, useMNC .AND. pickup_write_mnc) raises. Returns the file names written."""
    if not cfg.cpp.flag("GAD_ALLOW_TS_SOM_ADV", _OPT):                         # :37
        return []
    if cfg.cpp.flag("ALLOW_MNC", _OPT) and dict(cfg.use).get("useMNC", False):  # :55-64
        raise NotImplementedError("GAD_WRITE_PICKUP: the useMNC branch (:55-64) is not ported")
    Nr = cfg.size.Nr
    written = []
    lChar = len(suff.rstrip())                                                  # :68  ILNBLNK(suff)
    for flag, base, som in ((params.tempSOM_Advection, "pickup_somT", "som_T"),       # :70-85
                            (params.saltSOM_Advection, "pickup_somS", "som_S")):      # :87-102
        if not flag:
            continue
        fn = base if lChar == 0 else base + "." + suff[:lChar]                  # :72-76 / :89-93
        prec = PRECFLOAT64                                                      # :77 / :94
        for n in range(1, nSOM+1):                                              # :79-84 / :96-101
            iRec = n
            WRITE_REC_3D_RL(fn, prec, Nr, getattr(state, som)[n-1], iRec, myIter, mds=mds,
                            globalFile=io.globalFiles)
        written.append(fn)
    return written
