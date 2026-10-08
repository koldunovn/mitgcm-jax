"""SEAICE_BOTTOMDRAG_COEFFS: pkg/seaice/seaice_bottomdrag_coeffs.F @63cdc0b (lane M4LAB session 3)."""


def seaice_bottomdrag_coeffs(uIceLoc, vIceLoc, HEFFMLoc, HEFFLoc, AREALoc, CbotLoc, iStep, myTime, myIter, *, cfg,
                             sp):
    """SEAICE_BOTTOMDRAG_COEFFS( uIceLoc, vIceLoc, HEFFMLoc, HEFFLoc, AREALoc, CbotLoc, iStep, myTime, myIter,
    myThid )   @63cdc0b pkg/seaice/seaice_bottomdrag_coeffs.F:9-169 (the non-ITD argument list, SEAICE_LSR :384-393)

    C     | o Compute the non-linear drag coefficients for basal stress
    C     |   parameterization (Lemieux et al., 2015)

    Returns CbotLoc. With SEAICE_ALLOW_BOTTOMDRAG the whole body is `IF (SEAICEbasalDragK2.GT.0. _d 0) THEN ...
    ENDIF` (:80-165): with SEAICEbasalDragK2 = 0. (lab_sea: the default, seaice_readparms.F:424) nothing is written
    and CbotLoc (SEAICE.h CbotC) keeps its SEAICE_INIT_VARIA zero (seaice_init_varia.F:116). The coefficient arm
    (:81-163, SEAICEbasalDragK2 > 0) is not ported: SEAICE_READPARMS refuses it (a traced REAL cannot select the arm
    here). Without SEAICE_ALLOW_BOTTOMDRAG the routine is empty (:68-166)."""
    del uIceLoc, vIceLoc, HEFFMLoc, HEFFLoc, AREALoc, iStep, myTime, myIter, sp
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_BOTTOMDRAG_COEFFS: SEAICE_ITD (HEFFITD, AREAITD) is not ported")
    return CbotLoc
