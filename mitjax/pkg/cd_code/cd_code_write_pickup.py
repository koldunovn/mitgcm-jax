"""CD_CODE_WRITE_PICKUP: pkg/cd_code/cd_code_write_pickup.F @63cdc0b."""

from mitjax.model.src.ini_parms import PRECFLOAT64
from mitjax.pkg.rw.write_rec import WRITE_REC_3D_RL


def cd_code_write_pickup(permPickup, suff, myTime, myIter, *, cfg, io, state, mds):
    """CD_CODE_WRITE_PICKUP( permPickup, suff, myTime, myIter, myThid )   @63cdc0b
    pkg/cd_code/cd_code_write_pickup.F:8-99

    Called by PACKAGES_WRITE_PICKUP when useCDscheme (packages_write_pickup.F:102-107; the caller decides). Writes
    'pickup_cd.'//suff (:86) when pickup_write_mdsio (:84): uVelD, vVelD, uNM1, vNM1 as records 1-4 of Nr levels and
    etaNm1 as record 4*Nr+1 of one level (:88-92, WRITE_REC_3D_RL, precFloat64 :85), on the host from concrete
    State values. useMNC from the package switches (cfg.use), `io`: IOParams (pickup_write_mdsio, globalFiles), `mds`: the
    MdsContext of the run directory (as write_pickup). Raise: the MNC write (useMNC .AND. pickup_write_mnc,
    :46-79). Returns the file name or None."""
    if not cfg.cpp.ALLOW_CD_CODE:
        return None
    if cfg.cpp.ALLOW_MNC and dict(cfg.use).get("useMNC", False) and getattr(io, "pickup_write_mnc", False):
        raise NotImplementedError("CD_CODE_WRITE_PICKUP: the MNC write (:46-79) is not ported")
    if not io.pickup_write_mdsio:                                               # :84
        return None
    prec = PRECFLOAT64                                                          # :85
    fn = "pickup_cd." + suff                                                    # :86  WRITE(fn,'(A,A)')
    Nr = cfg.size.Nr
    g = io.globalFiles
    WRITE_REC_3D_RL(fn, prec, Nr, state.uVelD, 1, myIter, mds=mds, globalFile=g)           # :88
    WRITE_REC_3D_RL(fn, prec, Nr, state.vVelD, 2, myIter, mds=mds, globalFile=g)           # :89
    WRITE_REC_3D_RL(fn, prec, Nr, state.uNM1, 3, myIter, mds=mds, globalFile=g)            # :90
    WRITE_REC_3D_RL(fn, prec, Nr, state.vNM1, 4, myIter, mds=mds, globalFile=g)            # :91
    WRITE_REC_3D_RL(fn, prec, 1, state.etaNm1, 4*Nr+1, myIter, mds=mds, globalFile=g)      # :92
    return fn
