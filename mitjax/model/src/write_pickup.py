"""WRITE_PICKUP: model/src/write_pickup.F @63cdc0b (host side: the concrete State is written to the run directory)."""

from mitjax.model.src.ini_parms import PRECFLOAT64
from mitjax.pkg.mdsio.mdsio_write_field import mds_wr_metafiles
from mitjax.pkg.rw.write_rec import WRITE_REC_3D_RL

listDim = 20                                # write_pickup.F:58  PARAMETER( listDim = 20 )
oneRL = 1.0                                 # EEPARAMS.h  oneRL = 1.0 _d 0


def _cpp(cfg, name):
    """#ifdef NAME of the build (an option no compiled source tests is undefined)."""
    return name in cfg.cpp.known and bool(getattr(cfg.cpp, name))


def write_pickup(permPickup, suffix, myTime, myIter, *, cfg, params, ip, io, state, mds):
    """WRITE_PICKUP( permPickup, suffix, myTime, myIter, myThid )   @63cdc0b model/src/write_pickup.F:8-447

    C     | SUBROUTINE WRITE_PICKUP
    C     | o Write main-model pickup-file for restarting

    Ported (pickup_write_mdsio, ALLOW_ADAMSBASHFORTH_3 undefined): :100-101 fn = 'pickup.'//suffix, fp =
    precFloat64; :103-131 Uvel, Vvel, Theta, Salt (3-D records -j); :242-251 GuNm1, GvNm1 (momStepping); :252-269
    GtNm1/TempNm1, GsNm1/SaltNm1 (AdamsBashforthGt/_T, Gs/_S, GAD.h); :286-291 PhiHyd (storePhiHyd4Phys); :328-333
    EtaN; :349-360 dEtaHdt and EtaH (always written: the IF lines are commented out); :363-386 the field-list check and
    MDS_WR_METAFILES. ADVECT lane: the ALLOW_ADAMSBASHFORTH_3 branch (:133-241) for the tracers (gtNm/gsNm(m1) when
    alph_AB or beta_AB /= 0, (m2) when beta_AB /= 0; momentum AB3 raises). Raise: the GM_InMomAsStress fields (ALLOW_EDDYPSI),
    the ALLOW_NONHYDROSTATIC / ALLOW_QHYD_STAGGER_TS / ALLOW_SMAG_3D_DIFFUSIVITY / ALLOW_ADDFLUID /
    ALLOW_FRICTION_HEATING records when switched on, Phi_rLow (usingPCoords .AND. useSEAICE), MNC (:389-440, only
    when `useMNC .AND. pickup_write_mnc`: mnc_readparms.F:97 default .FALSE.).
    `params`: the time-step Params (ini_parms_dyn); `ip`: InitParams (the AB flags of GAD_INIT_FIXED, storePhiHyd4Phys,
    useMNC); `io`: IOParams (pickup_write_mdsio, globalFiles); `mds`: the MdsContext of the run directory;
    `permPickup` only matters for MNC. Returns the file name (or None when nothing is written)."""
    Nr = cfg.size.Nr
    wrFldList = [" " * 8] * listDim                                             # :88-90
    if _cpp(cfg, "ALLOW_MNC") and ip.useMNC and io.pickup_write_mnc:      # :389 GO lane (M1 accept.)
        raise NotImplementedError("WRITE_PICKUP: MNC pickups (:389-440) are not ported")
    if not io.pickup_write_mdsio:                                           # :98
        return
    fn = "pickup." + suffix                                                     # :100
    fp = PRECFLOAT64                                                            # :101
    j = 0

    def w3(name, fld):
        nonlocal j
        j = j + 1
        WRITE_REC_3D_RL(fn, fp, Nr, fld, -j, myIter, mds=mds, globalFile=io.globalFiles)
        if j <= listDim:
            wrFldList[j - 1] = name
    w3("Uvel    ", state.uVel)                                                 # :103-105
    w3("Vvel    ", state.vVel)                                                 # :106-108
    if _cpp(cfg, "ALLOW_EDDYPSI") and _cpp(cfg, "ALLOW_GMREDI"):                          # :109-123
        raise NotImplementedError("WRITE_PICKUP: GM_InMomAsStress fields (ALLOW_EDDYPSI) are not ported")
    w3("Theta   ", state.theta)                                                # :126-128
    w3("Salt    ", state.salt)                                                 # :129-131
    if _cpp(cfg, "ALLOW_ADAMSBASHFORTH_3"):                                          # :133-241 (ADVECT lane)
        m1 = 1 + (int(myIter)+1) % 2                                            # :135  1+MOD(myIter+1,2), myIter >= 0
        m2 = 1 + int(myIter) % 2                                                # :136
        ab_on = float(params.alph_AB) != 0. or float(params.beta_AB) != 0.     # host side: concrete values
        beta_on = float(params.beta_AB) != 0.
        if params.momStepping:                                                  # :137-164 (GO lane)
            for pair, n1, n2 in (("guNm", "GuNm1   ", "GuNm2   "), ("gvNm", "GvNm1   ", "GvNm2   ")):
                if ab_on:
                    w3(n1, getattr(state, pair)[m1-1])
                if beta_on:
                    w3(n2, getattr(state, pair)[m2-1])
        for on, on_T, pair, n1, n2, t1, t2 in (                                 # :165-206
                (ip.AdamsBashforthGt, ip.AdamsBashforth_T, "gtNm", "GtNm1   ", "GtNm2   ", "TempNm1 ", "TempNm2 "),
                (ip.AdamsBashforthGs, ip.AdamsBashforth_S, "gsNm", "GsNm1   ", "GsNm2   ", "SaltNm1 ", "SaltNm2 ")):
            if on or on_T:
                if ab_on:
                    w3(t1 if on_T else n1, getattr(state, pair)[m1-1])          # the last assignment wins
                if beta_on:
                    w3(t2 if on_T else n2, getattr(state, pair)[m2-1])
        # :207-239 (gwNm, QHydGwNm): refused with the AB2 records below (:270-281), same condition
    else:
        if params.momStepping:                                                  # :243-251
            w3("GuNm1   ", state.guNm1)
            w3("GvNm1   ", state.gvNm1)
        if ip.AdamsBashforthGt or ip.AdamsBashforth_T:                          # :252-260
            j_name = "GtNm1   " if ip.AdamsBashforthGt else "TempNm1 "
            w3(j_name, state.gtNm1)
        if ip.AdamsBashforthGs or ip.AdamsBashforth_S:                          # :261-269
            w3("GsNm1   " if ip.AdamsBashforthGs else "SaltNm1 ", state.gsNm1)
    for opt in ("ALLOW_NONHYDROSTATIC", "ALLOW_QHYD_STAGGER_TS"):              # :270-281
        if _cpp(cfg, opt) and (params.nonHydrostatic or (params.quasiHydrostatic and params.staggerTimeStep)):
            raise NotImplementedError(f"WRITE_PICKUP: the {opt} records are not ported")
    if ip.storePhiHyd4Phys:                                                     # :286-291
        w3("PhiHyd  ", state.totPhiHyd)
    if _cpp(cfg, "ALLOW_NONHYDROSTATIC") and params.use3Dsolver:                     # :292-298
        raise NotImplementedError("WRITE_PICKUP: Phi_NHyd (use3Dsolver) is not ported")
    for opt in ("ALLOW_SMAG_3D_DIFFUSIVITY", "ALLOW_FRICTION_HEATING"):        # :299-308, :318-326
        if _cpp(cfg, opt):
            raise NotImplementedError(f"WRITE_PICKUP: the {opt} record is not ported")
    if _cpp(cfg, "ALLOW_ADDFLUID") and params.selectAddFluid != 0:              # :309-317 (GO lane: the switch)
        raise NotImplementedError("WRITE_PICKUP: the AddMass record (selectAddFluid /= 0) is not ported")
    n3D = j                                                                     # :328
    nj = 0

    def w2(name, fld):
        nonlocal j, nj
        j = j + 1
        nj = -(n3D * (Nr - 1) + j)
        WRITE_REC_3D_RL(fn, fp, 1, fld, nj, myIter, mds=mds, globalFile=io.globalFiles)
        if j <= listDim:
            wrFldList[j - 1] = name
    w2("EtaN    ", state.etaN)                                                 # :329-333
    if params.usingPCoords and dict(cfg.use).get("useSEAICE", False):                       # :334-339
        raise NotImplementedError("WRITE_PICKUP: Phi_rLow (usingPCoords .AND. useSEAICE) is not ported")
    if _cpp(cfg, "ALLOW_NONHYDROSTATIC") and params.selectNHfreeSurf >= 1:            # :340-346
        raise NotImplementedError("WRITE_PICKUP: dPhiNH (selectNHfreeSurf) is not ported")
    w2("dEtaHdt ", state.dEtaHdt)                                              # :348-353 (IF commented out)
    w2("EtaH    ", state.etaHnm1)                                              # :355-360 (IF commented out)
    nWrFlds = j                                                                 # :363
    if nWrFlds > listDim:                                                       # :364-373
        raise ValueError(f"WRITE_PICKUP: trying to write {nWrFlds:5d} fields\nWRITE_PICKUP: field-list dimension "
                         f"(listDim={listDim:5d}) too small\nABNORMAL END: S/R WRITE_PICKUP (list-size Pb)")
    nj = abs(nj)                                                                # :376
    glf = io.globalFiles                                                           # :377
    timList = [float(myTime)]                                                   # :379
    mds_wr_metafiles(fn, fp, glf, False, 0, 0, 1, " ", nWrFlds, wrFldList, 1, timList, oneRL, nj, myIter,
                     mds=mds)                                                   # :380-385
    return fn
