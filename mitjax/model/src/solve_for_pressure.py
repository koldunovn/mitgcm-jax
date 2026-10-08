"""SOLVE_FOR_PRESSURE: model/src/solve_for_pressure.F @63cdc0b (the solver call itself: CG2D lane, cg2d.cg2d_solve)."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.calc_div_ghat import calc_div_ghat
from mitjax.model.src.cg2d import EXCH_XY_RL, cg2d_solve


def solve_for_pressure(myTime, myIter, *, cfg, grid, params, state, ff, cg2dh, cg2d_params, ex):
    """SOLVE_FOR_PRESSURE( myTime, myIter, myThid )   @63cdc0b model/src/solve_for_pressure.F:7-467

    C     | SUBROUTINE SOLVE_FOR_PRESSURE
    C     | o Controls inversion of two and/or three-dimensional elliptic problems for the pressure field.

    Returns (state, diag): State with etaN (:376-384); `diag` = dict of the solver's scalars for the STDOUT lines
    (firstResidual, minResidualSq, lastResidual, numIters, nIterMin, sumRHS, rhsMax), printed on the host by the
    driver (:106-118 putPmEinXvector line at myIter = 1+nIter0, :331-350 the monitor lines at
    DIFFERENT_MULTIPLE(monitorFreq), cg2d.F:199-200 the Sum(rhs) line: formatters in mitjax/model/src/cg2d.py).
    `cg2d_params`: Cg2dParams (cg2d_h.ini_parms_cg2d). cg2d_x, cg2d_b: locals (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy).
    :103 putPmEinXvector = .FALSE. (the build's line; its alternatives are not compiled), so :151-179 never run.
    GO lane: ALLOW_CD_CODE etaNm1 = etaN (:126-128, CD_CODE_VARS.h in the State) and the useRealFreshWaterFlux
    source (:142-150, EmPmR of `ff`); ALLOW_NONHYDROSTATIC compiled with use3Dsolver off. Raise: use3Dsolver (cg3d),
    ALLOW_OBCS, debugLevel >= debLevD,
    diagFreq output (:268-270)."""
    # exactConserv (:216-226, the etaH source term): R2 arm, tracer lane (plan Task 13).
    # GO lane: master's solve_for_pressure.F has no NONLIN_FRSURF lines (the raise listed it); ALLOW_CD_CODE
    # (:126-128 etaNm1 = etaN) and the useRealFreshWaterFlux source (:142-150, needs `ff`) are ported
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: the ALLOW_OBCS blocks are not ported")
    # GO lane (global_ocean.cs32x15): ALLOW_NONHYDROSTATIC compiled with use3Dsolver = .FALSE. (set_parms.F:104):
    # oldFreeSurfTerm = use3Dsolver .AND. .NOT.exactConserv (:86) is .FALSE., so :193-212 take the exactConserv /
    # else branches of the hydrostatic build; cg3d_b (:133-140 zeroed) is only read under use3Dsolver (:386-460).
    # The ' oldFreeSurfTerm =' line (:112-115) is host output at the first step, like :106-111 (not printed).
    if cfg.cpp.ALLOW_NONHYDROSTATIC and params.use3Dsolver:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: use3Dsolver (CG3D, :386-460) is not ported")
    if params.debugLevel >= 4:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: debugLevel >= debLevD output is not ported")
    if params.diagFreq != 0.:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: diagFreq output of cg2d_b (:268-270) is not ported")
    sz = cfg.size
    g = grid
    cg2d_x = state.etaN.local("cg2d_x")
    cg2d_b = state.etaN.local("cg2d_b")
    cg3d_b = None                                                               # :77 cg3d_b(1), not used in M1
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                        # :124
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                        # :125
    if cfg.cpp.ALLOW_CD_CODE:                                                   # :126-128 (GO lane)
        state = state.replace(etaNm1=state.etaNm1.at[iA, jA].set(state.etaN[iA, jA]))   # :127
    cg2d_x = cg2d_x.at[iA, jA].set(g.Bo_surf[iA, jA]*state.etaN[iA, jA])       # :129
    cg2d_b = cg2d_b.at[iA, jA].set(0.)                                          # :130  0. _d 0
    if params.useRealFreshWaterFlux and params.fluidIsWater:                    # :142-150 (GO lane)
        tmpFac = params.freeSurfFac*params.mass2rUnit*params.implicDiv2DFlow     # :143
        j1 = loop_j(1, sz.sNy)                                                  # :144
        i1 = loop_i(1, sz.sNx)                                                  # :145
        cg2d_b = cg2d_b.at[i1, j1].set(tmpFac*g.rA[i1, j1]*ff.EmPmR[i1, j1]     # :146-147
                                       / params.deltaTMom)
    # :182  DO k=Nr,1,-1 as a level scan (KERNEL_GUIDE §4; no level branches)
    from mitjax.ops.scan_k import scan_levels
    cg2d_b, cg3d_b = scan_levels(lambda k, c: calc_div_ghat(k, *c, cfg=cfg, grid=grid, params=params, state=state),
                                 (cg2d_b, cg3d_b), 1, sz.Nr, down=True)
    j = loop_j(1, sz.sNy)                                                       # :228
    i = loop_i(1, sz.sNx)                                                       # :229
    if params.exactConserv:                                                     # :216-226 (R2 arm, tracer lane)
        ks = g.kSurfC[i, j]                                                     # :220
        deepFac2F_ks = g.deepFac2F.data[ks - 1]                                 # deepFac2F(ks): gather
        cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j]                               # :221-224
                                     - params.freeSurfFac*g.rA[i, j]*deepFac2F_ks
                                     / params.deltaTMom/params.deltaTFreeSurf
                                     * state.etaH[i, j])
    else:                                                                       # :227-237
        ks = g.kSurfC[i, j]                                                     # :230
        deepFac2F_ks = g.deepFac2F.data[ks - 1]                                 # deepFac2F(ks): gather (KERNEL_GUIDE)
        cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j]                               # :231-234
                                     - params.freeSurfFac*g.rA[i, j]*deepFac2F_ks
                                     / params.deltaTMom/params.deltaTFreeSurf
                                     * state.etaN[i, j])
    numIters = params.cg2dMaxIters                                              # :277
    nIterMin = params.cg2dUseMinResSol - 1                                      # :278
    solver = cg2d_solve                                                         # :306-313  CALL CG2D
    if cfg.cpp.ALLOW_CG2D_NSA and cg2d_params.useNSACGSolver:                   # :297-305 (PTRACERS lane)
        from mitjax.model.src.cg2d_nsa import cg2d_nsa_solve
        solver = cg2d_nsa_solve                                                 # CALL CG2D_NSA
    cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed = solver(   # :308
        cg2d_b, cg2d_x, numIters, nIterMin, cfg=cfg, cg2dh=cg2dh, params=cg2d_params, ex=ex)
    cg2d_x = EXCH_XY_RL(cg2d_x, ex=ex)                                          # :315  _EXCH_XY_RL( cg2d_x )
    etaN = state.etaN.at[iA, jA].set(g.recip_Bo[iA, jA]*cg2d_x[iA, jA])        # :380
    diag = dict(firstResidual=firstResidual, minResidualSq=minResidualSq, lastResidual=lastResidual,
                numIters=numIters, nIterMin=nIterMin, sumRHS=printed["sumRHS"], rhsMax=printed["rhsMax"])
    return state.replace(etaN=etaN), diag
