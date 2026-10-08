"""ADVECT: pkg/seaice/advect.F @63cdc0b (lane M4ADCOL: 1D_ocean_ice_column/input_ad, SEAICEadvScheme = 2)."""

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.farray import loop_i, loop_j

HALF = 0.5                       # SEAICE_PARAMS.h:  PARAMETER ( HALF = 0.5 _d 0 ) (seaice_params_h.HALF)


def advect(UI, VI, fld, iceMsk, *, cfg, sp, grid, ex):
    """ADVECT( UI, VI, fld, fldNm1, iceMsk, myThid )   @63cdc0b pkg/seaice/advect.F:9-238

    C     | o Calculate ice advection

    The second-order centred advection of the non-multi-dimensional arm of SEAICE_ADVDIFF (seaice_advdiff.F:579-656):
    a two-pass predictor-corrector (`DO k=1,2`, :78-158) on the time-centred field tmpFld = (fld + fldNm1)/2, each
    pass followed by EXCH_XY_RL (:156). Returns (fld, fldNm1).
    Ported: SEAICEuseFluxForm (:130-153). The advective form (:104-129) and the DIFF1 > 0 diffusion (:160-235, DIFFUS)
    raise (not executed by input_ad: coverage of the code_ad build, advect.F lines 105-125, 164-226 never run).
    ALLOW_AUTODIFF (:94-101): afx = afy = 0 on the whole local array before each pass (only i, j = 1..sNx+1 are
    written by the flux-form arm, so the other entries are zero in either case)."""
    if not sp.SEAICEuseFluxForm:                                               # :104
        raise NotImplementedError("ADVECT: SEAICEuseFluxForm = .FALSE. (advect.F:104-129) is not ported")
    # DIFF1 > 0 (:160-235, DIFFUS) raises in SEAICE_READPARMS (DIFF1 is a REAL parameter, traced here)
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    DELTT = sp.SEAICE_deltaTtherm                                              # :63
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    fldNm1 = fld.local("fldNm1").at[iA, jA].set(fld[iA, jA])                   # :65-73
    j1 = loop_j(1, sNy+1)
    i1 = loop_i(1, sNx+1)
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    for k in (1, 2):                                                           # :78 DO k=1,2
        tmpFld = fld.local("tmpFld").at[iA, jA].set(HALF*(fld[iA, jA] + fldNm1[iA, jA]))   # :87-92
        afx = fld.local("afx", 0.)                                             # :94-101 (ALLOW_AUTODIFF): 0.
        afy = fld.local("afy", 0.)
        afx = afx.at[i1, j1].set(grid.dyG[i1, j1] * UI[i1, j1]                 # :137-138
                                 * 0.5 * (tmpFld[i1, j1]+tmpFld[i1-1, j1]))
        afy = afy.at[i1, j1].set(grid.dxG[i1, j1] * VI[i1, j1]                 # :139-140
                                 * 0.5 * (tmpFld[i1, j1]+tmpFld[i1, j1-1]))
        fld = fld.at[i, j].set(fldNm1[i, j]                                    # :145-149
                               - DELTT * (
                                   afx[i+1, j] - afx[i, j]
                                   + afy[i, j+1] - afy[i, j]
                               )*grid.recip_rA[i, j]*grid.maskInC[i, j])
        fld = EXCH_XY_RL(fld, ex=ex)                                           # :156
    return fld, fldNm1
