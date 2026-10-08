"""READ_PICKUP: model/src/read_pickup.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_3D_RL, EXCH_UV_3D_RL, EXCH_XY_RL
from mitjax.model.src.ini_parms import PRECFLOAT64
from mitjax.pkg.rw.read_mflds import READ_MFLDS_3D_RL, READ_MFLDS_CHECK, READ_MFLDS_SET

missFldDim = 20                     # read_pickup.F:63  PARAMETER( missFldDim = 20 )


def pickup_name(myIter, params):
    """read_pickup.F:82-91: fn = 'pickup.' // suff, suff = myIter as I10.10 (rwSuffixType 0, pickupSuff blank) or
    pickupSuff (A10). RW_GET_SUFFIX (rwSuffixType != 0) raises."""
    ip = params.init
    if str(ip.pickupSuff).strip() == "":
        if ip.rwSuffixType == 0:
            suff = f"{myIter:010d}"
        else:
            raise NotImplementedError("READ_PICKUP: rwSuffixType != 0 (RW_GET_SUFFIX) is not ported")
    else:
        suff = f"{str(ip.pickupSuff):>10.10s}"
    return "pickup." + suff


def read_pickup(state, myIter, *, cfg, params, ex, rw, dyn=None, host=None):
    """READ_PICKUP( myIter, myThid )   @63cdc0b model/src/read_pickup.F:8-597

    C     | SUBROUTINE READ_PICKUP
    C     | o Reads main-model state from a pickup file

    Ported (pickup_read_mdsio with a field list in the meta file, ALLOW_ADAMSBASHFORTH_3 undefined): :97-105
    READ_MFLDS_SET( fn, nbFields, filePrec, Nr ); :109-117 the file precision must be precFloat64; :253-270 Uvel,
    Vvel, Theta, Salt; :366-389 GuNm1, GvNm1 (momStepping), GtNm1 or TempNm1 (AdamsBashforthGt / AdamsBashforth_T),
    GsNm1 or SaltNm1; :407-410 PhiHyd -> totPhiHyd (storePhiHyd4Phys); :443-463 EtaN, dEtaHdt (exactConserv), EtaH
    (nonlinFreeSurf > 0); :468-482 READ_MFLDS_CHECK / CHECK_PICKUP; :538-565 the exchanges.
    Raise (no M1 variant executes them): the old formats (nbFields <= 0, :122-235), the AB3 reads (:272-364), the
    nonHydrostatic / use3Dsolver / selectNHfreeSurf reads (ALLOW_NONHYDROSTATIC is not compiled in M1), QH_GwNm1
    (quasiHydrostatic .AND. staggerTimeStep, refused by CONFIG_CHECK), SmagDiff, AddMass, FricHeat (options not
    compiled), Phi_rLow (usingPCoords), the MNC read (:487-532), and any missing field (CHECK_PICKUP's tolerance of
    missing fields, check_pickup.F, is not ported: a missing field stops here)."""
    ip = params.init
    Nr = cfg.size.Nr
    fn = pickup_name(myIter, params)
    if not ip.pickup_read_mdsio:
        raise NotImplementedError("READ_PICKUP: pickup_read_mdsio = .FALSE. is not ported")
    fp = PRECFLOAT64                                                            # :99
    mf, nbFields, filePrec = READ_MFLDS_SET(fn, Nr, myIter, rw=rw)              # :102-105
    if nbFields >= 0 and filePrec != fp:                                        # :109-117
        raise ValueError(f"READ_PICKUP: pickup-file binary precision do not match !\nREAD_PICKUP: file prec.="
                         f"{filePrec:4d} but expecting prec.={fp:4d}\nABNORMAL END: S/R READ_PICKUP (data-prec Pb)")
    if nbFields <= 0:                                                           # :122-235
        raise NotImplementedError("READ_PICKUP: a pickup without a field list (old formats) is not ported")
    ab3 = bool(cfg.cpp.ALLOW_ADAMSBASHFORTH_3)
    if ab3 and dyn is None:
        raise ValueError("READ_PICKUP: the ALLOW_ADAMSBASHFORTH_3 reads need alph_AB, beta_AB (`dyn`)")
    st = {}
    nj = 0                                                                      # :253

    def rd(name, fld, nNz):
        nonlocal mf, nj
        out, nj, mf = READ_MFLDS_3D_RL(name, fld, nj, fp, nNz, myIter, mf=mf, rw=rw)
        return out

    st["uVel"] = rd("Uvel    ", state.uVel, Nr)                                # :255-256
    st["vVel"] = rd("Vvel    ", state.vVel, Nr)                                # :257-258
    st["theta"] = rd("Theta   ", state.theta, Nr)                              # :267-268
    st["salt"] = rd("Salt    ", state.salt, Nr)                                # :269-270
    if ab3:                                                                     # :272-364 (GO lane)
        m1 = 1 + (myIter + 1) % 2                                               # :274  1 + MOD(myIter+1,2)
        m2 = 1 + myIter % 2                                                     # :275  1 + MOD( myIter ,2)
        alph_AB, beta_AB = float(dyn.alph_AB), float(dyn.beta_AB)

        def rd_pair(n1, n2, pair):
            pair = list(pair)
            if alph_AB != 0. or beta_AB != 0.:
                pair[m1 - 1] = rd(n1, pair[m1 - 1], Nr)
            if beta_AB != 0.:
                pair[m2 - 1] = rd(n2, pair[m2 - 1], Nr)
            return tuple(pair)
        if ip.momStepping:                                                      # :276-297
            st["guNm"] = rd_pair("GuNm1   ", "GuNm2   ", state.guNm)
            st["gvNm"] = rd_pair("GvNm1   ", "GvNm2   ", state.gvNm)
        if ip.AdamsBashforthGt:                                                 # :298-315
            st["gtNm"] = rd_pair("GtNm1   ", "GtNm2   ", state.gtNm)
        elif ip.AdamsBashforth_T:
            st["gtNm"] = rd_pair("TempNm1 ", "TempNm2 ", state.gtNm)
        if ip.AdamsBashforthGs:                                                 # :316-333
            st["gsNm"] = rd_pair("GsNm1   ", "GsNm2   ", state.gsNm)
        elif ip.AdamsBashforth_S:
            st["gsNm"] = rd_pair("SaltNm1 ", "SaltNm2 ", state.gsNm)
        if cfg.cpp.ALLOW_NONHYDROSTATIC and ip.nonHydrostatic:                  # :334-345
            raise NotImplementedError("READ_PICKUP: GwNm1/GwNm2 (nonHydrostatic, AB3) are not ported")
        if cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep:   # :346-361
            raise NotImplementedError("READ_PICKUP: QH_GwNm1/2 (quasiHydrostatic .AND. staggerTimeStep) are not "
                                      "ported")
    else:
        if ip.momStepping:                                                      # :366-373
            st["guNm1"] = rd("GuNm1   ", state.guNm1, Nr)
            st["gvNm1"] = rd("GvNm1   ", state.gvNm1, Nr)
        if ip.AdamsBashforthGt:                                                 # :375-381
            st["gtNm1"] = rd("GtNm1   ", state.gtNm1, Nr)
        elif ip.AdamsBashforth_T:
            st["gtNm1"] = rd("TempNm1 ", state.gtNm1, Nr)
        if ip.AdamsBashforthGs:                                                 # :383-389
            st["gsNm1"] = rd("GsNm1   ", state.gsNm1, Nr)
        elif ip.AdamsBashforth_S:
            st["gsNm1"] = rd("SaltNm1 ", state.gsNm1, Nr)
    if cfg.cpp.ALLOW_QHYD_STAGGER_TS and ip.quasiHydrostatic and ip.staggerTimeStep:      # :396-402
        raise NotImplementedError("READ_PICKUP: QH_GwNm1 (quasiHydrostatic .AND. staggerTimeStep) is not ported")
    if ip.storePhiHyd4Phys:                                                     # :407-410
        st["totPhiHyd"] = rd("PhiHyd  ", state.totPhiHyd, Nr)
    nj = nj * Nr                                                                # :443
    st["etaN"] = rd("EtaN    ", state.etaN, 1)                                 # :444-445
    if ip.exactConserv:                                                         # :456-459
        st["dEtaHdt"] = rd("dEtaHdt ", state.dEtaHdt, 1)
    if ip.nonlinFreeSurf > 0:                                                   # :460-463
        st["etaH"] = rd("EtaH    ", state.etaH, 1)
    nMissing = missFldDim                                                       # :468
    missFldList, nMissing, mf = READ_MFLDS_CHECK(nMissing, myIter, mf=mf)      # :469-472
    if nMissing > missFldDim:                                                   # :473-478
        raise ValueError(f"READ_PICKUP: missing fields list has been truncated to{missFldDim:4d}\n"
                         "ABNORMAL END: S/R READ_PICKUP (list-size Pb)")
    if nMissing > 0 or nbFields >= 1:                                           # :479-482 CHECK_PICKUP (GO lane)
        start = check_pickup(missFldList, nMissing, nbFields, fn, cfg=cfg, params=params, dyn=dyn)
        if host is not None:
            host.append(("CHECK_PICKUP", myIter, start))
    if cfg.cpp.ALLOW_MNC and ip.useMNC:                                         # :487-532
        # PTRACERS lane (tutorial_advection_in_gyre: useMNC with pickup_read_mnc=.FALSE.): :488 IF (useMNC .AND.
        # pickup_read_mnc); pickup_read_mnc from data.mnc, else mnc_readparms.F:98 pickup_read_mnc = .FALSE.
        if dict(cfg.static).get(("data.mnc", "mnc_01", "pickup_read_mnc"), False):
            raise NotImplementedError("READ_PICKUP: pickups with useMNC (pickup_read_mnc) are not ported")
    state = state.replace(**st)
    # :538-565 exchanges
    uVel, vVel = EXCH_UV_3D_RL(state.uVel, state.vVel, True, Nr, ex=ex)        # :538
    theta = EXCH_3D_RL(state.theta, Nr, ex=ex)                                  # :544
    salt = EXCH_3D_RL(state.salt, Nr, ex=ex)                                    # :545
    out = dict(uVel=uVel, vVel=vVel, theta=theta, salt=salt)
    if ab3:                                                                     # :546-553 (GO lane)
        u1, v1 = EXCH_UV_3D_RL(state.guNm[0], state.gvNm[0], True, Nr, ex=ex)
        u2, v2 = EXCH_UV_3D_RL(state.guNm[1], state.gvNm[1], True, Nr, ex=ex)
        out.update(guNm=(u1, u2), gvNm=(v1, v2),
                   gtNm=(EXCH_3D_RL(state.gtNm[0], Nr, ex=ex), EXCH_3D_RL(state.gtNm[1], Nr, ex=ex)),
                   gsNm=(EXCH_3D_RL(state.gsNm[0], Nr, ex=ex), EXCH_3D_RL(state.gsNm[1], Nr, ex=ex)))
    else:
        guNm1, gvNm1 = EXCH_UV_3D_RL(state.guNm1, state.gvNm1, True, Nr, ex=ex)    # :556
        out.update(guNm1=guNm1, gvNm1=gvNm1, gtNm1=EXCH_3D_RL(state.gtNm1, Nr, ex=ex),   # :557
                   gsNm1=EXCH_3D_RL(state.gsNm1, Nr, ex=ex))                          # :558
    out.update(etaN=EXCH_XY_RL(state.etaN, ex=ex),                              # :560
               etaH=EXCH_XY_RL(state.etaH, ex=ex),                              # :561
               dEtaHdt=EXCH_XY_RL(state.dEtaHdt, ex=ex))                        # :562
    if ip.storePhiHyd4Phys:                                                     # :564-565
        out["totPhiHyd"] = EXCH_3D_RL(state.totPhiHyd, Nr, ex=ex)
    return state.replace(**out)


# GO lane (global_ocean.cs32x15/input_ad): CHECK_PICKUP's handling of missing fields
_STOP_ALWAYS = ("Uvel    ", "Vvel    ", "Theta   ", "Salt    ", "EtaN    ")          # check_pickup.F:153-160
_STOP_NOT_YET = ("PhiHyd  ", "Phi_rLow", "AddMass ", "dEtaHdt ", "EtaH    ")       # :163-171


def check_pickup(missFldList, nMissing, nbFields, fn, *, cfg, params, dyn=None):
    """CHECK_PICKUP( missFldList, nMissing, nbFields, myIter, myThid )   @63cdc0b model/src/check_pickup.F:8-268
    (host side). Returns the RESTART.h start levels it leaves {mom_StartAB, tempStartAB, saltStartAB}: :59-65 with a
    field list (nbFields >= 1) all are nIter0; :173-198 a missing GuNm1/GvNm1 sets mom_StartAB = 0, GuNm2/GvNm2
    MIN(mom_StartAB, 1) (and the Gt/Temp, Gs/Salt pairs likewise). Stops (ValueError with the Fortran message) for
    the fields a restart needs (:150-171), an unrecognised one (:206-212), and any missing field with
    pickupStrictlyMatch (:216-223). The warnings of :224-246 are STDOUT/STDERR lines (not printed here).
    Raises: selectNHfreeSurf >= 1 (:66-77), the old formats (nbFields <= 0, :78-139), GwNm / QH_GwNm (not ported)."""
    nIter0 = params.time.nIter0
    ip = params.init
    if nbFields <= 0:
        raise NotImplementedError("CHECK_PICKUP: the old pickup formats (nbFields <= 0, :78-139) are not ported")
    if getattr(ip, "selectNHfreeSurf", 0) >= 1:
        raise NotImplementedError("CHECK_PICKUP: selectNHfreeSurf >= 1 (:66-77) is not ported")
    start = dict(mom_StartAB=nIter0, tempStartAB=nIter0, saltStartAB=nIter0)      # :59-65
    stop = []
    realFW = bool(getattr(dyn, "useRealFreshWaterFlux", False)) if dyn is not None else False
    for f in list(missFldList)[:nMissing]:                                      # :80-213
        if f == "dEtaHdt " and not realFW:                                      # :81-88 can restart without it
            pass
        elif f in ("dPhiNH  ", "Phi_NHyd", "SmagDiff", "FricHeat") or (f == "EtaN    " and getattr(
                ip, "rigidLid", False)):                                        # :89-145
            raise NotImplementedError(f"CHECK_PICKUP: missing {f!r} (:89-145) is not ported")
        elif f in _STOP_ALWAYS:
            stop.append(f"CHECK_PICKUP: cannot restart without field \"{f}\"")
        elif f in _STOP_NOT_YET:
            stop.append(f"CHECK_PICKUP: cannot currently restart without field \"{f}\"")
        elif f in ("GuNm1   ", "GvNm1   "):
            start["mom_StartAB"] = 0
        elif f in ("GuNm2   ", "GvNm2   "):
            start["mom_StartAB"] = min(start["mom_StartAB"], 1)      # MINMAX-INT: integers
        elif f in ("GtNm1   ", "TempNm1 "):
            start["tempStartAB"] = 0
        elif f in ("GtNm2   ", "TempNm2 "):
            start["tempStartAB"] = min(start["tempStartAB"], 1)      # MINMAX-INT: integers
        elif f in ("GsNm1   ", "SaltNm1 "):
            start["saltStartAB"] = 0
        elif f in ("GsNm2   ", "SaltNm2 "):
            start["saltStartAB"] = min(start["saltStartAB"], 1)      # MINMAX-INT: integers
        elif f in ("GwNm1   ", "GwNm2   ", "QH_GwNm1", "QH_GwNm2"):
            raise NotImplementedError(f"CHECK_PICKUP: missing {f!r} (nHydStartAB / qHydStartAB) is not ported")
        else:
            stop.append(f"CHECK_PICKUP: missing field \"{f}\" not recognized")
    if stop:                                                                    # :214-215
        raise ValueError("\n".join(stop) + "\nABNORMAL END: S/R CHECK_PICKUP")
    if nMissing > 0 and ip.pickupStrictlyMatch:                                 # :216-223
        raise ValueError("CHECK_PICKUP: try with \" pickupStrictlyMatch=.FALSE.,\" in file: \"data\", NameList: "
                         "\"PARM03\"\nABNORMAL END: S/R CHECK_PICKUP")
    return start

