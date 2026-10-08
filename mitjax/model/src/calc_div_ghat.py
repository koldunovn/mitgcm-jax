"""CALC_DIV_GHAT: model/src/calc_div_ghat.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def calc_div_ghat(k, cg2d_b, cg3d_b, *, cfg, grid, params, state):
    """CALC_DIV_GHAT( bi, bj, k, cg2d_b, cg3d_b, myThid )   @63cdc0b model/src/calc_div_ghat.F:6-150

    C     | S/R CALC_DIV_GHAT
    C     | o Form the right hand-side of the surface pressure eqn.
    C     cg2d_b :: Conjugate Gradient 2-D solver : Right-hand side vector
    C     cg3d_b :: Conjugate Gradient 3-D solver : Right-hand side vector

    Returns (cg2d_b, cg3d_b). Reads gU, gV of level k. xA, yA (_RS locals) set on 1..sNx+1, 1..sNy+1 (:65-72);
    pf local. GO lane (global_ocean.cs32x15): ALLOW_NONHYDROSTATIC compiled with use3Dsolver = .FALSE. (the cg3d_b
    blocks :91-101, :113-123 are skipped; use3Dsolver raises) and ALLOW_ADDFLUID compiled with selectAddFluid = 0
    (:126-147: unitsFac is formed but nothing is added; selectAddFluid >= 1 raises)."""
    if cfg.cpp.ALLOW_NONHYDROSTATIC and params.use3Dsolver:                    # :91-101, :113-123
        raise NotImplementedError("CALC_DIV_GHAT: use3Dsolver (cg3d_b) is not ported")
    if cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1:                  # :129-146
        raise NotImplementedError("CALC_DIV_GHAT: selectAddFluid >= 1 (addMass) is not ported")
    sz = cfg.size
    g = grid
    gU, gV = state.gU, state.gV
    xA = cg2d_b.local("xA")
    yA = cg2d_b.local("yA")
    pf = cg2d_b.local("pf")
    j = loop_j(1, sz.sNy+1)                                                     # :65
    i = loop_i(1, sz.sNx+1)                                                     # :66
    xA = xA.at[i, j].set(g.dyG[i, j]*g.deepFacC[k]                              # :67-68
                         * g.drF[k]*g.hFacW[i, j, k]*params.rhoFacC[k])
    yA = yA.at[i, j].set(g.dxG[i, j]*g.deepFacC[k]                              # :69-70
                         * g.drF[k]*g.hFacS[i, j, k]*params.rhoFacC[k])
    j = loop_j(1, sz.sNy)                                                       # :80
    i = loop_i(1, sz.sNx+1)                                                     # :81
    pf = pf.at[i, j].set(params.implicDiv2DFlow*xA[i, j]*gU[i, j, k] / params.deltaTMom)   # :82
    j = loop_j(1, sz.sNy)                                                       # :85
    i = loop_i(1, sz.sNx)                                                       # :86
    cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j] +                                 # :87-88
                                 pf[i+1, j]-pf[i, j])
    j = loop_j(1, sz.sNy+1)                                                     # :103
    i = loop_i(1, sz.sNx)                                                       # :104
    pf = pf.at[i, j].set(params.implicDiv2DFlow*yA[i, j]*gV[i, j, k] / params.deltaTMom)   # :105
    j = loop_j(1, sz.sNy)                                                       # :108
    i = loop_i(1, sz.sNx)                                                       # :109
    cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j] +                                 # :110-111
                                 pf[i, j+1]-pf[i, j])
    return cg2d_b, cg3d_b
