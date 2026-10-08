"""SEAICE_CHECK_PICKUP: pkg/seaice/seaice_check_pickup.F @63cdc0b (host side). Lane M4COL (the restart of
1D_ocean_ice_column)."""


def seaice_check_pickup(missFldList, nMissing, nbFields, myIter, *, cfg, sp, pickupStrictlyMatch):
    """SEAICE_CHECK_PICKUP( missFldList, nMissing, nbFields, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_check_pickup.F:6-226

    C     | o Check that fields that are needed to restart have been
    C     |   read. In case some fields are missing, stop if
    C     |   pickupStrictlyMatch=T or try, if possible, to restart
    C     |   without the missing field.

    Returns the SEAICE_PARAMS.h values it may change ({"SEAICEmomStartBDF": 0} when siUicNm1 / siVicNm1 are missing
    without pickupStrictlyMatch, :118-128; else {}). Stops (ValueError with the Fortran message) where the Fortran
    STOPs (:188-197). The WARNING lines (:91-128, :198-216) go to the error unit: not printed here (STDERR text).
    ALLOW_SITRACER (lane M4LAB session 4): a missing siTrac<nn> only warns ("restart without ... (set to zero)",
    :159-174; the field keeps SEAICE_INIT_VARIA's value, which is 1 for a 'one' tracer: the message is not what
    happens, docs/ISSUES_UPSTREAM.md). Raises: the SEAICE_ITD arms (not compiled in the ported builds)."""
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_CHECK_PICKUP: the SEAICE_ITD arms are not ported")
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    out = {}
    if nMissing >= 1:                                                           # :73
        tIceFlag = 0                                                            # :75
        for nj in range(1, nMissing + 1):                                       # :77-81
            if missFldList[nj - 1] == "siTICES ":
                tIceFlag = tIceFlag + 2
            if missFldList[nj - 1] == "siTICE  ":
                tIceFlag = tIceFlag + 1
        stopFlag = False                                                        # :82
        errors = []
        for nj in range(1, nMissing + 1):                                       # :88-186
            fldName = missFldList[nj - 1]
            if fldName == "siTICE  " and tIceFlag <= 1:                         # :90-97 (warning only)
                pass
            elif fldName == "siTICES " and tIceFlag <= 2:                       # :98-106 (warning only)
                pass
            elif fldName[:6] == "siSigm":                                       # :107-117 (warning only)
                pass
            elif fldName[:8] in ("siUicNm1", "siVicNm1"):                       # :118-128
                if not pickupStrictlyMatch:
                    out["SEAICEmomStartBDF"] = 0                                # :122
            elif fldName in ("siTICES ", "siTICE  ", "siUICE  ", "siVICE  ", "siAREA  ", "siHEFF  ",
                             "siHSNOW ", "siHSALT "):                           # :129-142
                stopFlag = True
                errors.append(f"SEAICE_CHECK_PICKUP: cannot restart without field \"{fldName}\"")
            elif sitracer and fldName[:6] == "siTrac":                          # :159-174 (warning only, ioUnit)
                pass
            else:                                                               # :176-184
                stopFlag = True
                errors.append(f"SEAICE_CHECK_PICKUP: missing field \"{fldName}\" not recognized")
        if stopFlag:                                                            # :188-189
            raise ValueError("\n".join(errors) + "\nABNORMAL END: S/R SEAICE_CHECK_PICKUP")
        elif pickupStrictlyMatch:                                               # :190-197
            raise ValueError("SEAICE_CHECK_PICKUP: try with \" pickupStrictlyMatch=.FALSE.,\" in file: \"data\", "
                             "NameList: \"PARM03\"\nABNORMAL END: S/R SEAICE_CHECK_PICKUP")
    return out
