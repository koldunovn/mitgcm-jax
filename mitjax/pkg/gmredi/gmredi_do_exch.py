"""GMREDI_DO_EXCH: pkg/gmredi/gmredi_do_exch.F @63cdc0b."""


def gmredi_do_exch(myTime, myIter, *, cfg, gm, params=None, ex=None):
    """GMREDI_DO_EXCH( myTime, myIter, myThid )   @63cdc0b pkg/gmredi/gmredi_do_exch.F:6-69

    C     | SUBROUTINE GMREDI_DO_EXCH
    C     | o Apply Exchanges to GM-Redi variables when necessary

    Returns `gm`. The only exchange of the routine, EXCH_UV_XYZ_RL( GM_PsiX, GM_PsiY ) (:54), runs under
    GM_BOLUS_ADVEC IF ( useCubedSphereExchange .AND. GM_AdvForm .AND. .NOT.GM_AdvSeparate .AND. useMultiDimAdvec )
    (:46-55): with GM_AdvForm = .FALSE. (both M1 runs) the condition is false whatever the other switches, and the
    routine does nothing (the tensor Kwx..Kvy is not exchanged: its outer halo row keeps its GMREDI_INIT_VARIA or
    ALLOW_AUTODIFF zero). GOADK lane: with GM_AdvForm the whole condition is evaluated (`params`, `ex`), and the
    exchange runs when it holds (global_ocean.90x40x15/code_ad: exch1, no cubed sphere: it does not). Not ported
    (raise): ALLOW_EDDYPSI (:37-38,
    IF ( GM_InMomAsStress )), GM_GEOM_VARIABLE_K (:58-66).
    """
    if cfg.cpp.GM_GEOM_VARIABLE_K:
        raise NotImplementedError("GMREDI_DO_EXCH: GM_GEOM_VARIABLE_K is not ported")
    if cfg.cpp.GM_BOLUS_ADVEC:                                      # :36
        if cfg.cpp.ALLOW_EDDYPSI:
            raise NotImplementedError("GMREDI_DO_EXCH: ALLOW_EDDYPSI is not ported")
        if gm.GM_AdvForm:                                           # :46-49 (the conjunction needs GM_AdvForm)
            # GOADK lane: the whole condition (`params`: PARAMS.h useCubedSphereExchange, useMultiDimAdvec, static)
            if params is None:
                raise ValueError("GMREDI_DO_EXCH: GM_AdvForm needs params (useCubedSphereExchange, useMultiDimAdvec)")
            if params.useCubedSphereExchange and not gm.GM_AdvSeparate and params.useMultiDimAdvec:
                if ex is None:
                    raise ValueError("GMREDI_DO_EXCH: the cube exchange of GM_PsiX/Y needs `ex`")
                from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL
                psx, psy = EXCH_UV_XYZ_RL(gm.GM_PsiX, gm.GM_PsiY, True, ex=ex)                  # :54
                gm = gm.replace(GM_PsiX=psx, GM_PsiY=psy)
    return gm
