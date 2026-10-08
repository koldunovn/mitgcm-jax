"""INI_CG2D: model/src/ini_cg2d.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_XY_RS
from mitjax.model.grid import zeroRS
from mitjax.model.src.cg2d import fortran_1pe
from mitjax.model.src.cg2d_h import CG2DH, declare_xy
from mitjax.model.src.ini_global_domain import ini_global_domain_2d
from mitjax.ops.fortran_minmax import MAX_CHAIN
from mitjax.ops.safe import safe_div


def ini_cg2d(*, cfg, grid, surface, params, ex):
    """INI_CG2D( myThid )   @63cdc0b model/src/ini_cg2d.F:7-249

    C     | SUBROUTINE INI_CG2D
    C     | o Initialise 2d conjugate gradient solver operators.
    C     | These arrays are purely a function of the basin geom.
    C     | We set then here once and them use then repeatedly.

    Reads (GRID.h) dyG, dxG, drF, hFacW, hFacS, recip_dxC, recip_dyC, rA, kSurfC, deepFac2F (and maskInC under
    ALLOW_OBCS) from `grid`, (SURFACE.h) recip_Bo from `surface` (any object with the FArray `recip_Bo`, set by
    INI_LINEAR_PHISURF), the Cg2dParams `params` (cg2d_h.ini_parms_cg2d); `ex`: the experiment's exchanger.
    Returns the CG2D.h common blocks (`CG2DH`): aW2d, aS2d, aC2d, pW, pS, pC, cg2dNorm, cg2dTolerance_sq,
    cg2dNormaliseRHS; and the line INI_CG2D prints (`ini_cg2d_message`, a host-side formatter of cg2dNorm).

    Vectorisation: the k loop :85-101 is a sum over k (a recursion in k): a Python loop over k in the Fortran order,
    each level vectorised over i, j; every other loop computes each point from inputs only. `myNorm = MAX(ABS(.),
    myNorm)` over all tiles and points followed by _GLOBAL_MAX_RS is an order-free maximum (no NaN: every input is
    finite). Scalar IFs on computed values (:117-121) and pointwise IFs (:215-231) are `where`s with the division
    guarded before it (mitjax/ops/safe.py), so masked lanes stay finite.

    Ported for the M1 variants: cg2dTargetResWunit <= 0 (cg2dNormaliseRHS = .TRUE.: tolerance without unit,
    :149-151); PTRACERS lane (tutorial_tracer_adjsens): cg2dTargetResWunit > 0 (:152-162, `ini_cg2d_tolerance`, with
    INI_GLOBAL_DOMAIN's n2dWetPts and globalArea). ALLOW_OBCS (:104-109) is compiled in no M1 build and raises.
    The message lines (:166-180) go to STDOUT only: `ini_cg2d_message`. The commented-out (CcnhDebug) plots are not
    ported. _RS = Real*8 (no -use_real4 build), so the RS exchanges are the RL exchanges (mitjax/model/grid.py).
    """
    sz = cfg.size
    sNx, sNy, Nr = sz.sNx, sz.sNy, sz.Nr
    if cfg.cpp.ALLOW_OBCS:                                                      # :104-109
        raise NotImplementedError("INI_CG2D: ALLOW_OBCS (maskInC factors, ini_cg2d.F:104-109) is not ported")
    dyG, dxG, drF, hFacW, hFacS = grid.dyG, grid.dxG, grid.drF, grid.hFacW, grid.hFacS
    recip_dxC, recip_dyC, rA, kSurfC, deepFac2F = grid.recip_dxC, grid.recip_dyC, grid.rA, grid.kSurfC, grid.deepFac2F
    recip_Bo = surface.recip_Bo
    implicSurfPress, implicDiv2DFlow = params.implicSurfPress, params.implicDiv2DFlow
    freeSurfFac, deltaTMom, deltaTFreeSurf = params.freeSurfFac, params.deltaTMom, params.deltaTFreeSurf
    cg2dpcOffDFac = params.cg2dpcOffDFac
    nt = dyG.ntiles

    # :49-64  Initialize arrays in common blocs (CG2D.h)
    aW2d = declare_xy("aW2d", sz, fill=0.0, ntiles=nt)
    aS2d = declare_xy("aS2d", sz, fill=0.0, ntiles=nt)
    aC2d = declare_xy("aC2d", sz, fill=0.0, ntiles=nt)
    pW = declare_xy("pW", sz, fill=0.0, ntiles=nt)
    pS = declare_xy("pS", sz, fill=0.0, ntiles=nt)
    pC = declare_xy("pC", sz, fill=0.0, ntiles=nt)

    # :66-71  Init. scalars (cg2dNorm, cg2dNormaliseRHS, cg2dTolerance_sq are set again at :146-163)

    # :73-115  Initialise laplace operator
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    aW2d = aW2d.at[i, j].set(0.0)                                               # :79-84
    aS2d = aS2d.at[i, j].set(0.0)
    for k in range(1, Nr + 1):                                                  # :85-101
        faceArea = dyG[i, j]*drF[k] \
            * hFacW[i, j, k]                                                    # :89-90
        aW2d = aW2d.at[i, j].set(aW2d[i, j]
                                 + implicSurfPress*implicDiv2DFlow
                                 * faceArea*recip_dxC[i, j])                    # :91-93
        faceArea = dxG[i, j]*drF[k] \
            * hFacS[i, j, k]                                                    # :94-95
        aS2d = aS2d.at[i, j].set(aS2d[i, j]
                                 + implicSurfPress*implicDiv2DFlow
                                 * faceArea*recip_dyC[i, j])                    # :96-98
    # :102-113  myNorm = MAX(ABS(aW2d),myNorm); myNorm = MAX(ABS(aS2d),myNorm) from myNorm = 0. _d 0 (:76): one chain
    # over every tile in tile order, then j, i, then the two statements, the new value winning NaN (the oracle
    # table); _GLOBAL_MAX_RS (:116) is the identity for the oracle's single process (global_max.F)
    myNorm = MAX_CHAIN(zeroRS, jnp.stack([jnp.abs(aW2d[i, j]), jnp.abs(aS2d[i, j])], axis=-1),
                       p="a", acc="b", ex=ex)                                   # :110-111
    # :117-121  IF ( myNorm .NE. zeroRS ) THEN myNorm = 1. _d 0/myNorm ELSE myNorm = 1. _d 0
    myNorm = jnp.where(myNorm != zeroRS, safe_div(1.0, myNorm, myNorm != zeroRS), 1.0)
    aW2d = aW2d.at[i, j].set(aW2d[i, j]*myNorm)                                 # :122-131
    aS2d = aS2d.at[i, j].set(aS2d[i, j]*myNorm)

    # :133-138  Update overlap regions
    aW2d, aS2d = EXCH_UV_XY_RS(aW2d, aS2d, False, ex=ex)

    # :144-164  set global parameter in common block; define the solver tolerance
    cg2dNorm = myNorm                                                           # :146
    cg2dNormaliseRHS = params.cg2dTargetResWunit_le_0                           # :148 (cg2dTargetResWunit .LE. 0)
    if cg2dNormaliseRHS:                                                        # :149
        cg2dTolerance = params.cg2dTargetResidual                               # :151
    else:                                                                       # :152-162 (PTRACERS lane)
        cg2dTolerance = ini_cg2d_tolerance(cg2dNorm, *ini_global_domain_2d(cfg=cfg, grid=grid, ex=ex), params)
    cg2dTolerance_sq = cg2dTolerance*cg2dTolerance                              # :163

    # :196-238  Initialise preconditioner
    j0 = loop_j(0, sNy)                                                         # :199-209
    i0 = loop_i(0, sNx)
    ks = kSurfC[i0, j0]
    aC2d = aC2d.at[i0, j0].set(-(
        aW2d[i0, j0] + aW2d[i0+1, j0]
        + aS2d[i0, j0] + aS2d[i0, j0+1]
        + freeSurfFac*myNorm*recip_Bo[i0, j0]*deepFac2F.data[ks - 1]
        * rA[i0, j0]/deltaTMom/deltaTFreeSurf
    ))
    aC = aC2d[i, j]                                                             # :210-236
    aCs = aC2d[i, j-1]
    aCw = aC2d[i-1, j]
    pC = pC.at[i, j].set(jnp.where(aC == zeroRS, 1.0,                           # :215-219
                                   safe_div(1.0, aC, aC != zeroRS)))
    # Fortran `-aW2d/x` is -(aW2d/x) (unary minus binds after the division)
    pW = pW.at[i, j].set(jnp.where(aC + aCw == zeroRS, 0.0,                     # :220-225
                                   -safe_div(aW2d[i, j], (cg2dpcOffDFac*(aCw+aC))**2, aC + aCw != zeroRS)))
    pS = pS.at[i, j].set(jnp.where(aC + aCs == zeroRS, 0.0,                     # :226-231
                                   -safe_div(aS2d[i, j], (cg2dpcOffDFac*(aCs+aC))**2, aC + aCs != zeroRS)))
    # :239-241  Update overlap regions
    pC = EXCH_XY_RS(pC, ex=ex)
    pW, pS = EXCH_UV_XY_RS(pW, pS, False, ex=ex)

    return CG2DH(aW2d=aW2d, aS2d=aS2d, aC2d=aC2d, pW=pW, pS=pS, pC=pC, cg2dNorm=cg2dNorm,
                 cg2dTolerance_sq=cg2dTolerance_sq, cg2dNormaliseRHS=cg2dNormaliseRHS)


def ini_cg2d_tolerance(cg2dNorm, n2dWetPts, globalArea, params):
    """cg2dTolerance of ini_cg2d.F:152-162 (cg2dNormaliseRHS = .FALSE.: the target residual in W units converted to
    the solver unit [m^2/s^2]; PTRACERS lane, tutorial_tracer_adjsens). n2dWetPts, globalArea: GRID.h, set by
    INI_GLOBAL_DOMAIN (mitjax/model/src/ini_global_domain.py). Fortran evaluates the products and quotients left to
    right, as written; `IF ( implicDiv2DFlow.GT.zeroRL )` on the traced value is a `where` (both arms finite)."""
    cg2dTargetResWunit, implicDiv2DFlow, deltaTMom = (params.cg2dTargetResWunit, params.implicDiv2DFlow,
                                                      params.deltaTMom)
    tol_flow = cg2dNorm * cg2dTargetResWunit \
        * implicDiv2DFlow / deltaTMom \
        * globalArea / jnp.sqrt(n2dWetPts)                                      # :155-157
    tol_zero = cg2dNorm * cg2dTargetResWunit \
        * globalArea / deltaTMom                                                # :160-161
    return jnp.where(implicDiv2DFlow > 0.0, tol_flow, tol_zero)                 # :154 implicDiv2DFlow.GT.zeroRL


def ini_cg2d_tolerance_message(cg2dTolerance, n2dWetPts, globalArea):
    """The line INI_CG2D prints when .NOT.cg2dNormaliseRHS (ini_cg2d.F:171-176):
    WRITE(msgBuf,'(2A,1PE22.15,A,1PE16.10,A)') 'INI_CG2D: ', 'cg2dTolerance =', cg2dTolerance,
    ' (Area=', globalArea/SQRT(n2dWetPts), ')'."""
    area = float(jnp.asarray(globalArea) / jnp.sqrt(jnp.asarray(n2dWetPts)))
    return ("INI_CG2D: " + "cg2dTolerance =" + fortran_1pe(float(cg2dTolerance), 22, 15) + " (Area="
            + fortran_1pe(area, 16, 10) + ")")


def ini_cg2d_message(cg2dNorm):
    """The line INI_CG2D prints (ini_cg2d.F:168-170, cg2dNormaliseRHS: the tolerance line :171-176 is skipped):
    WRITE(msgBuf,'(2A,1PE23.16)') 'INI_CG2D: ', 'CG2D normalisation factor = ', cg2dNorm."""
    return "INI_CG2D: " + "CG2D normalisation factor = " + fortran_1pe(float(cg2dNorm), 23, 16)
