"""CD_CODE_READ_PICKUP: pkg/cd_code/cd_code_read_pickup.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL, EXCH_XY_RL
from mitjax.farray import FArray
from mitjax.model.src.ini_parms import PRECFLOAT64


def cd_code_read_pickup(state, myIter, *, cfg, params, ex, rw):
    """CD_CODE_READ_PICKUP( myIter, myThid )   @63cdc0b pkg/cd_code/cd_code_read_pickup.F:8-85

    C     Read the checkpoint.

    Returns the State with uVelD, vVelD, uNM1, vNM1 (records 1-4, Nr levels each) and etaNm1 (record 4*Nr+1 of one
    level; 6*Nr+1 with usePickupBeforeC54) read from 'pickup_cd.'//suff (precFloat64, READ_REC_3D_RL :66-76),
    then the exchanges :79-81 (EXCH_UV_DGRID_3D_RL(uVelD, vVelD, .TRUE.), EXCH_UV_XYZ_RL(uNM1, vNM1, .TRUE.),
    EXCH_XY_RL(etaNm1)). Suffix (:36-44): myIter as I10.10 (rwSuffixType 0) or pickupSuff (A10). Raise:
    rwSuffixType /= 0 (RW_GET_SUFFIX), the MNC read (useMNC .AND. pickup_read_mnc), pickup_read_mdsio = .FALSE."""
    ip = params.init
    if str(ip.pickupSuff).strip() == "":                                        # :36
        if ip.rwSuffixType == 0:
            suff = f"{myIter:010d}"                                             # :38  WRITE(suff,'(I10.10)')
        else:
            raise NotImplementedError("CD_CODE_READ_PICKUP: rwSuffixType != 0 (RW_GET_SUFFIX) is not ported")
    else:
        suff = f"{str(ip.pickupSuff):>10.10s}"                                  # :43  WRITE(suff,'(A10)')
    prec = PRECFLOAT64                                                          # :45
    if cfg.cpp.ALLOW_MNC and getattr(ip, "useMNC", False):                      # :49 #ifdef ALLOW_MNC
        # :50 IF (useMNC .AND. pickup_read_mnc); pickup_read_mnc from data.mnc, else mnc_readparms.F:98 .FALSE.
        # (as READ_PICKUP, read_pickup.py; lab_sea/input: useMNC with pickup_read_mnc = .FALSE., data.mnc)
        if dict(cfg.static).get(("data.mnc", "mnc_01", "pickup_read_mnc"), False):
            raise NotImplementedError("CD_CODE_READ_PICKUP: the MNC read (:49-62) is not ported")
    fn = "pickup_cd." + suff                                                    # :64  WRITE(fn,'(A,A10)')
    if not ip.pickup_read_mdsio:
        raise NotImplementedError("CD_CODE_READ_PICKUP: pickup_read_mdsio = .FALSE. is not ported")
    Nr = cfg.size.Nr
    uVelD = rw.READ_REC_3D_RL(fn, prec, Nr, state.uVelD, 1, myIter)            # :67
    vVelD = rw.READ_REC_3D_RL(fn, prec, Nr, state.vVelD, 2, myIter)            # :68
    uNM1 = rw.READ_REC_3D_RL(fn, prec, Nr, state.uNM1, 3, myIter)              # :69
    vNM1 = rw.READ_REC_3D_RL(fn, prec, Nr, state.vNM1, 4, myIter)              # :70
    if ip.usePickupBeforeC54:                                                   # :71
        etaNm1 = rw.READ_REC_3D_RL(fn, prec, 1, state.etaNm1, 6*Nr+1, myIter)  # :72
    else:
        etaNm1 = rw.READ_REC_3D_RL(fn, prec, 1, state.etaNm1, 4*Nr+1, myIter)  # :74
    u, v = ex.EXCH_UV_DGRID_3D_RL(uVelD.data, vVelD.data, True)                 # :79
    uVelD = FArray(u, uVelD.name, tiled=uVelD.tiled, _dims=uVelD.dims)
    vVelD = FArray(v, vVelD.name, tiled=vVelD.tiled, _dims=vVelD.dims)
    uNM1, vNM1 = EXCH_UV_XYZ_RL(uNM1, vNM1, True, ex=ex)                        # :80
    etaNm1 = EXCH_XY_RL(etaNm1, ex=ex)                                          # :81
    return state.replace(uVelD=uVelD, vVelD=vVelD, uNM1=uNM1, vNM1=vNM1, etaNm1=etaNm1)
