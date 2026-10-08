"""PP81_CALC: pkg/pp81/pp81_calc.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.pp81.pp81_ri_number import pp81_ri_number


def pp81_calc(sigmaR, myTime, myIter, *, cfg, grid, params, eos, state, pp):
    """PP81_CALC( bi, bj, sigmaR, myTime, myIter, myThid )   @63cdc0b pkg/pp81/pp81_calc.F:7-129

    C     | SUBROUTINE PP81_CALC                                     |
    C     | o Compute all PP81 fields defined in PP81.h              |
    C     | This subroutine is based on SPEM code                    |
    C global parameters updated by pp_calc
    C     PPviscAz  :: PP eddy viscosity coefficient              (m^2/s)
    C     PPdiffKzT :: PP diffusion coefficient for temperature   (m^2/s)
    C     sigmaR :: Vertical gradient of iso-neutral density
    C     myTime :: Current time in simulation
    C     myIter :: Current time-step number

    Returns `pp` (PP81.h) with PPviscAr, PPdiffKr written on levels 2..Nr at iMin:iMax = 2-OLx:sNx+OLx-1,
    jMin:jMax = 2-OLy:sNy+OLy-1 (:61-64); level 1 and the outermost halo row/column keep their values. sigmaR is not
    read (as in the Fortran). `state`: DYNVARS.h (theta, salt, uVel, vVel, totPhiHyd) for PP81_RI_NUMBER.

    The DO K=2,Nr loop (:66-117) calls PP81_RI_NUMBER once per level and its iterations are independent (level K
    reads only inputs and writes only level K); it is written as the per-level body `level_k(k, c)` with the carried
    fields in `c`, run as a level scan in the Fortran order (KERNEL_GUIDE §4; K = 2 static). Ported branches: #else of ALLOW_3D_DIFFKR (:85-86,
    diffKrNrS); ALLOW_3D_DIFFKR and ALLOW_PP81_LOWERBOUND (:94-106) raise; ALLOW_DIAGNOSTICS fills (:119-124) are
    output only.

    `IF ( RiNumber .LT. RiLimit )` (:73) chooses between two values: a `where` with both branches finite (the ELSE
    branch's PPnu0/(denom**PPnRi) is guarded where it is not taken: there denom can be 0). `denom**PPnRi` with the
    INTEGER PPnRi (static) is `lax.integer_pow` (__powidf2's multiplication sequence). Literals `1.0` (REAL*4, exact).
    """
    sz = cfg.size
    if cfg.cpp.flag("ALLOW_3D_DIFFKR", "PP81_OPTIONS.h"):
        raise NotImplementedError("PP81_CALC: ALLOW_3D_DIFFKR is not ported")
    if cfg.cpp.flag("ALLOW_PP81_LOWERBOUND", "PP81_OPTIONS.h"):
        raise NotImplementedError("PP81_CALC: ALLOW_PP81_LOWERBOUND is not ported")
    iMin = 2-sz.OLx                                                     # :61-64
    iMax = sz.sNx+sz.OLx-1
    jMin = 2-sz.OLy
    jMax = sz.sNy+sz.OLy-1
    RiLimit, PPalpha, PPviscMax, PPnu0, PPnRi = pp.RiLimit, pp.PPalpha, pp.PPviscMax, pp.PPnu0, pp.PPnRi
    # RiNumber is a local of the routine: (1-OLx:sNx+OLx,1-OLy:sNy+OLy), never initialised
    RiNumber0 = FArray(jnp.full((sz.nSx*sz.nSy, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx), jnp.nan), "RiNumber",
                       i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))

    def level_k(K, c):                                                  # DO K = 2, Nr (:66-117)
        PPviscAr, PPdiffKr = c
        RiNumber = pp81_ri_number(K, iMin, iMax, jMin, jMax, RiNumber0, myTime,     # :67-70
                                  cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        j = loop_j(jMin, jMax)                                          # :71
        i = loop_i(iMin, iMax)                                          # :72
        below = RiNumber[i, j] < RiLimit                                # :73
        denom_lim = 1.0 + PPalpha*RiLimit                               # :74
        denom_ri = 1.0 + PPalpha*RiNumber[i, j]                         # :77
        denom = jnp.where(below, denom_lim, denom_ri)
        PPviscTmp = jnp.where(below, PPviscMax,                         # :75, :78
                              safe_div(PPnu0, denom_ri**PPnRi, ~below))
        PPviscAr = PPviscAr.at[i, j, K].set(MAX(PPviscTmp, params.viscArNr[K], p="b"))     # :81
        PPdiffKr = PPdiffKr.at[i, j, K].set(MAX(PPviscAr[i, j, K]/denom,                    # :82-86
                                                params.diffKrNrS[K], p="b"))
        PPviscAr = PPviscAr.at[i, j, K].set(PPviscAr[i, j, K]          # :110-111
                                            * grid.maskC[i, j, K])
        PPdiffKr = PPdiffKr.at[i, j, K].set(PPdiffKr[i, j, K]          # :112-113
                                            * grid.maskC[i, j, K])
        return PPviscAr, PPdiffKr

    c = scan_levels(level_k, (pp.PPviscAr, pp.PPdiffKr), 2, sz.Nr,      # K = 2 static (PP81_RI_NUMBER's
                    peel=(1, 0))                                        # Km1 = MAX(1,K-1))
    return pp.replace(PPviscAr=c[0], PPdiffKr=c[1])
