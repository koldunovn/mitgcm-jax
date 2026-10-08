"""GGL90_WRITE_PICKUP: pkg/ggl90/ggl90_write_pickup.F @63cdc0b (host side)."""

PRECFLOAT64 = 64       # EEPARAMS.h:65  PARAMETER ( precFloat64 = 64 )


def ggl90_write_pickup(permPickup, suff, myTime, myIter, *, cfg, ggl):
    """GGL90_WRITE_PICKUP( permPickup, suff, myTime, myIter, myThid )   @63cdc0b pkg/ggl90/ggl90_write_pickup.F:6-60

    C     permPickup :: write a permanent pickup
    C     suff    :: suffix for pickup file (eg. ckptA or 0000000010)

    Returns the records the routine writes, [(file name, precision, record number, FArray)]: 'pickup_ggl90.'//suff
    (:47), GGL90TKE as record 1 (:49) and, under ALLOW_GGL90_IDEMIX with useIDEMIX, IDEMIX_E as record 2 (:51-55),
    both WRITE_REC_3D_RL with precFloat64 (:46). The caller writes them with its MDS writer."""
    fn = "pickup_ggl90." + suff                                         # :47  WRITE(fn,'(A,A)') 'pickup_ggl90.',suff
    out = [(fn, PRECFLOAT64, 1, ggl.GGL90TKE)]                          # :49
    if cfg.cpp.flag("ALLOW_GGL90_IDEMIX", "GGL90_OPTIONS.h") and ggl.useIDEMIX:
        out.append((fn, PRECFLOAT64, 2, ggl.IDEMIX_E))                  # :53
    return out
