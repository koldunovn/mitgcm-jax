"""SEAICE_LSR and its routines: pkg/seaice/seaice_lsr.F @63cdc0b (lane M4OFF session 3, the C-grid build of
offline_exf_seaice/input.dyn_lsr).

    seaice_lsr(...)              SEAICE_LSR (:24-1167): the Picard (pseudo-time) iteration with the LSR (line successive
                                 relaxation) linear solver, SEAICEuseLSRflex (residual-norm criterion)
    seaice_residual(...)         SEAICE_RESIDUAL (:1173-1289)
    seaice_lsr_calc_coeffs(...)  SEAICE_LSR_CALC_COEFFS (:1294-1609)
    seaice_lsr_rhsu / _rhsv      SEAICE_LSR_RHSU / RHSV (:1616-1764, :1771-1919)
    seaice_lsr_tridiagu / _v     SEAICE_LSR_TRIDIAGU / V (:1926-2064, :2071-2219): one Gauss-Seidel sweep of
                                 tridiagonal line solves (rows j for u, columns i for v), in the Fortran order
    different_multiple_traced    DIFFERENT_MULTIPLE (eesupp/src/different_multiple.F) on a traced myTime

The iteration is the Fortran's: `DO ipass` (a `lax.fori_loop`), the two `IF ( doNonLinLoop )` blocks (`lax.cond`),
`DO m = 1, linearIterLoc` with its convergence tests every SOLV_NCHECK sweeps (a `lax.while_loop` that stops when the
Fortran's loop would do nothing more: both doIterate4u and doIterate4v .FALSE.), the same residual norms
(SEAICE_RESIDUAL's running sum over every tile in the oracle's single-process order) and the same pass counts. The
STDOUT lines of each pass (printResidual) are returned as values (`out`) and written by the run driver.

The derivative: solver iterations are never differentiated (project rule), with one exception (plan decision 14, A1;
lane M4ADLAB session 4): a build that defines SEAICE_LSR_ADJOINT_ITER (TAF tapes its sweeps) runs the DO m loop as
the fixed-length scan of mitjax/ad/lsr_sweeps.py, whose derivative is the transpose of the executed sweeps, and
SEAICE_DYNSOLVER calls it without the forward-only guard; so does any build with the static option
sp.mjx_lsr_derivative = "sweeps" (plan decision 17; mitjax/ad/modes.py), which the API sets by default wherever the
build does not tape the sweeps (docs plan decision 12 a: lsr_derivative=None). With lsr_derivative="run" on such a
build the solve is forward only: any JVP / VJP through it raises NotImplementedError (mitjax/ad/seaice_lsr_rule.py)."""

import jax.numpy as jnp
from jax import lax

from mitjax.ad.modes import lsr_derivative
from mitjax.eesupp.exch_rs import EXCH_UV_XY_RL
from mitjax.eesupp.global_sum import tile_sum_fortran
from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.ops.fortran_minmax import MAX, MAX_CHAIN, MIN
from mitjax.ops.libm import glibc_log10
from mitjax.ops.scan_k import scan_k
from mitjax.pkg.seaice.seaice_bottomdrag_coeffs import seaice_bottomdrag_coeffs
from mitjax.pkg.seaice.seaice_calc_strainrates import seaice_calc_strainrates
from mitjax.pkg.seaice.seaice_calc_viscosities import seaice_calc_viscosities
from mitjax.pkg.seaice.seaice_oceandrag_coeffs import seaice_oceandrag_coeffs
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE, ZERO

debLevA = 1                     # EEPARAMS.h:91  PARAMETER ( debLevA=1 )
debLevD = 4                     # EEPARAMS.h:94  PARAMETER ( debLevD=4 )
halfRL = 0.5                    # EEPARAMS.h:73  PARAMETER ( twoRL  = 2.0 _d 0 , halfRL = 0.5 _d 0 )

# the per-pass values SEAICE_LSR prints (seaice_lsr.F:767-773, :1007-1035) and its non-convergence warning
# (:1047-1057), returned in `out` with one entry per ipass
OUT_FLOAT = ("LSRflexFac", "LSR_ERRORfu", "LSR_ERRORfv", "residIni", "residUini", "residVini", "residU_fd",
             "residV_fd", "S1", "S2", "residUend", "residVend", "WFAU", "WFAV")
OUT_INT = ("ICOUNT1", "ICOUNT2")
OUT_BOOL = ("printFlex", "printResid", "notConverged", "printFrDrift", "printWarn")


def different_multiple_traced(freq, val1, step):
    """LOGICAL FUNCTION DIFFERENT_MULTIPLE( freq, val1, step )   @63cdc0b eesupp/src/different_multiple.F:7-65, with
    traced arguments (the host version: mitjax/eesupp/different_multiple.py). NINT (:54) rounds half away from zero;
    the INTEGER range check of the host version is not made (the quotient is a time step count)."""
    v1 = val1                                                       # :49
    v2 = val1 - step                                                # :50
    v3 = val1 + step                                                # :51
    q = v1 / jnp.where(freq != 0.0, freq, 1.0)                      # :54 NINT(v1/freq)*freq
    a = jnp.abs(q)
    n = jnp.floor(a)
    n = jnp.where(a - n >= 0.5, n + 1.0, n)
    n = jnp.where(q >= 0.0, n, -n)
    v4 = n * freq
    d1 = v1 - v4                                                    # :55
    d2 = v2 - v4                                                    # :56
    d3 = v3 - v4                                                    # :57
    test = (jnp.abs(d1) < jnp.abs(d2)) & (jnp.abs(d1) <= jnp.abs(d3))   # :58-59
    return jnp.where(freq != 0.0, jnp.where(jnp.abs(step) > freq, True, test), False)   # :41-46


def running_sum(a, ex):
    """`s = 0.; DO bj; DO bi; DO j; DO i; s = s + a(i,j,bi,bj)` with ONE scalar across all tiles (SEAICE_RESIDUAL
    :1237-1275), then _GLOBAL_SUM_RL of the oracle's single process (the identity): a [Tloc, nj, ni] (this device's
    tiles) -> the chain over every tile in tile order (bi inner), the same on every device and every P."""
    full = ex.all_tiles(a)
    T, nj, ni = full.shape
    return tile_sum_fortran(full.reshape(1, T*nj, ni))[0]


def seaice_residual(rhsU, rhsV, uRt1, uRt2, vRt1, vRt2, AU, BU, CU, AV, BV, CV, uFld, vFld, calcMeanResid, *, cfg,
                    grid, ex, globalArea):
    """SEAICE_RESIDUAL( rhsU, rhsV, uRt1, uRt2, vRt1, vRt2, AU, BU, CU, AV, BV, CV, uFld, vFld, residU, residV, uRes,
    vRes, calcMeanResid, myIter, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:1173-1289

    C     | o Compute norm of residual of linear system

    Returns (residU, residV). uRes/vRes (output arrays, :1244-1257) are read by no caller of this run (only the
    LSR_mixIniGuess >= 2 arm, :646-689, which raises): not returned. `calcMeanResid` is traced (printResidual)."""
    sz = cfg.size
    j = loop_j(1, sz.sNy)                                                      # :1242
    i = loop_i(1, sz.sNx)                                                      # :1243
    uRes = (rhsU[i, j]                                                         # :1244-1250
            + uRt1[i, j]*uFld[i, j-1]
            + uRt2[i, j]*uFld[i, j+1]
            - (AU[i, j]*uFld[i-1, j]
               + BU[i, j]*uFld[i, j]
               + CU[i, j]*uFld[i+1, j]))
    vRes = (rhsV[i, j]                                                         # :1251-1257
            + vRt1[i, j]*vFld[i-1, j]
            + vRt2[i, j]*vFld[i+1, j]
            - (AV[i, j]*vFld[i, j-1]
               + BV[i, j]*vFld[i, j]
               + CV[i, j]*vFld[i, j+1]))
    residU = running_sum(uRes*uRes*grid.rAw[i, j]*grid.maskInW[i, j], ex)    # :1237, :1263-1264, :1279
    residV = running_sum(vRes*vRes*grid.rAs[i, j]*grid.maskInS[i, j], ex)    # :1238, :1268-1269, :1280
    residU = jnp.where(calcMeanResid, residU, 0.)
    residV = jnp.where(calcMeanResid, residV, 0.)
    residU = jnp.where(residU > 0., jnp.sqrt(jnp.where(residU > 0., residU, 1.)/globalArea), residU)   # :1282
    residV = jnp.where(residV > 0., jnp.sqrt(jnp.where(residV > 0., residV, 1.)/globalArea), residV)   # :1283
    return residU, residV


def seaice_lsr_calc_coeffs(etaPlusZeta, zetaMinusEta, etaZloc, zetaZloc, dragSym, co, iMin, iMax, jMin, jMax,
                           myTime, myIter, *, cfg, sp, grid, sf, useCubedSphereExchange):
    """SEAICE_LSR_CALC_COEFFS( etaPlusZeta, zetaMinusEta, etaZloc, zetaZloc, dragSym, AU, BU, CU, AV, BV, CV, uRt1,
    uRt2, vRt1, vRt2, iMin, iMax, jMin, jMax, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:1294-1609

    C     | o Calculate coefficient matrix for LSR solver

    `co` {name: FArray} the coefficient arrays AU..CV, uRt1..vRt2 (written on iMin..iMax, jMin..jMax; other points
    as given). Returns co. SEAICEuseBDF2 is not carried (SEAICE_READPARMS raises if data.seaice sets it: .FALSE.,
    :1386-1392); strImpCplFac = 1 with SEAICEuseStrImpCpl (:1396-1397; lane M4CS32ICE).
    The loops are independent per point: vectorised."""
    del myTime, myIter
    AREA, maskU, maskV = sf["AREA"], sf["seaiceMaskU"], sf["seaiceMaskV"]
    k1AtC, k2AtC, k1AtZ, k2AtZ = sf["k1AtC"], sf["k2AtC"], sf["k1AtZ"], sf["k2AtZ"]
    g = grid
    bdfAlphaOverDt = 1.0                                                       # :1385 (SEAICEuseBDF2 .FALSE.)
    bdfAlphaOverDt = bdfAlphaOverDt / sp.SEAICE_deltaTdyn                      # :1394
    strImpCplFac = 0.0                                                         # :1396
    if sp.SEAICEuseStrImpCpl:                                                  # :1397
        strImpCplFac = 1.0
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if sp.SEAICEscaleSurfStress:                                               # :1401-1415
        areaW = 0.5*(AREA[i, j]+AREA[i-1, j])                                  # :1404
        areaS = 0.5*(AREA[i, j]+AREA[i, j-1])                                  # :1405
    else:
        areaW = 1.0                                                            # :1411
        areaS = 1.0                                                            # :1412
    loc = sf["AREA"].local("UXX", 0.)
    jj, ii = loop_j(jMin, jMax), loop_i(iMin-1, iMax)                          # :1419-1420
    UXX = loc.at[ii, jj].set(g.dyF[ii, jj] * etaPlusZeta[ii, jj]               # :1422-1423
                             * g.recip_dxF[ii, jj])
    UXM = loc.at[ii, jj].set(g.dyF[ii, jj] * zetaMinusEta[ii, jj]              # :1425-1426
                             * k1AtC[ii, jj] * 0.5)
    jj, ii = loop_j(jMin, jMax+1), loop_i(iMin, iMax)                          # :1429-1430
    UYY = loc.at[ii, jj].set(g.dxV[ii, jj] *                                   # :1432-1434
                             (etaZloc[ii, jj] + strImpCplFac*zetaZloc[ii, jj])
                             * g.recip_dyU[ii, jj])
    UYM = loc.at[ii, jj].set(g.dxV[ii, jj] * etaZloc[ii, jj]                   # :1436-1437
                             * k2AtZ[ii, jj] * 0.5)
    jj, ii = loop_j(jMin, jMax), loop_i(iMin, iMax+1)                          # :1440-1441
    VXX = loc.at[ii, jj].set(g.dyU[ii, jj] *                                   # :1443-1445
                             (etaZloc[ii, jj] + strImpCplFac*zetaZloc[ii, jj])
                             * g.recip_dxV[ii, jj])
    VXM = loc.at[ii, jj].set(g.dyU[ii, jj] * etaZloc[ii, jj]                   # :1447-1448
                             * k1AtZ[ii, jj] * 0.5)
    jj, ii = loop_j(jMin-1, jMax), loop_i(iMin, iMax)                          # :1451-1452
    VYY = loc.at[ii, jj].set(g.dxF[ii, jj] * etaPlusZeta[ii, jj]               # :1454-1455
                             * g.recip_dyF[ii, jj])
    VYM = loc.at[ii, jj].set(g.dxF[ii, jj] * zetaMinusEta[ii, jj]              # :1457-1458
                             * k2AtC[ii, jj] * 0.5)
    # :1467-1485
    AU = (- UXX[i-1, j] + UXM[i-1, j]) * maskU[i, j]                           # :1470-1471
    CU = (- UXX[i, j] - UXM[i, j]) * maskU[i, j]                               # :1473-1474
    BU = ((ONE - maskU[i, j]) +                                                # :1476-1479
          (UXX[i-1, j] + UXX[i, j] + UYY[i, j+1] + UYY[i, j]
           + UXM[i-1, j] - UXM[i, j] + UYM[i, j+1] - UYM[i, j]
           ) * maskU[i, j])
    uRt1 = UYY[i, j] + UYM[i, j]                                               # :1481
    uRt2 = UYY[i, j+1] - UYM[i, j+1]                                           # :1483
    hFacM = maskU[i, j-1]                                                      # :1492
    hFacP = maskU[i, j+1]                                                      # :1493
    BU = BU + maskU[i, j] * (                                                  # :1497-1499
        (1.0 - hFacM) * (UYY[i, j] + UYM[i, j])
        + (1.0 - hFacP) * (UYY[i, j+1] - UYM[i, j+1]))
    uRt1 = uRt1 * hFacM                                                        # :1501
    uRt2 = uRt2 * hFacP                                                        # :1502
    AU = AU * g.recip_rAw[i, j]                                                # :1509
    CU = CU * g.recip_rAw[i, j]                                                # :1510
    BU = (BU * g.recip_rAw[i, j]                                               # :1514-1519
          + maskU[i, j] *
          (bdfAlphaOverDt*sf["seaiceMassU"][i, j]
           + 0.5 * (dragSym[i, j]
                    + dragSym[i-1, j])*areaW))
    uRt1 = uRt1 * g.recip_rAw[i, j]                                            # :1520
    uRt2 = uRt2 * g.recip_rAw[i, j]                                            # :1521
    # :1530-1548
    AV = (- VYY[i, j-1] + VYM[i, j-1]) * maskV[i, j]                           # :1533-1534
    CV = (- VYY[i, j] - VYM[i, j]) * maskV[i, j]                               # :1536-1537
    BV = ((ONE - maskV[i, j]) +                                                # :1539-1542
          (VXX[i, j] + VXX[i+1, j] + VYY[i, j] + VYY[i, j-1]
           - VXM[i, j] + VXM[i+1, j] - VYM[i, j] + VYM[i, j-1]
           ) * maskV[i, j])
    vRt1 = VXX[i, j] + VXM[i, j]                                               # :1544
    vRt2 = VXX[i+1, j] - VXM[i+1, j]                                           # :1546
    hFacM = maskV[i-1, j]                                                      # :1555
    hFacP = maskV[i+1, j]                                                      # :1556
    BV = BV + maskV[i, j] * (                                                  # :1560-1562
        (1.0 - hFacM) * (VXX[i, j] + VXM[i, j])
        + (1.0 - hFacP) * (VXX[i+1, j] - VXM[i+1, j]))
    vRt1 = vRt1 * hFacM                                                        # :1564
    vRt2 = vRt2 * hFacP                                                        # :1565
    AV = AV * g.recip_rAs[i, j]                                                # :1572
    CV = CV * g.recip_rAs[i, j]                                                # :1573
    BV = (BV * g.recip_rAs[i, j]                                               # :1577-1582
          + maskV[i, j] *
          (bdfAlphaOverDt*sf["seaiceMassV"][i, j]
           + 0.5 * (dragSym[i, j]
                    + dragSym[i, j-1])*areaS))
    vRt1 = vRt1 * g.recip_rAs[i, j]                                            # :1583
    vRt2 = vRt2 * g.recip_rAs[i, j]                                            # :1584
    if (useCubedSphereExchange and (sp.SEAICE_OLx > 0 or sp.SEAICE_OLy > 0)) \
            or sp.SEAICEscaleSurfStress:                                       # :1588-1603
        BU = jnp.where(BU == 0.0, 1.0, BU)                                     # :1599
        BV = jnp.where(BV == 0.0, 1.0, BV)                                     # :1600
    vals = dict(AU=AU, BU=BU, CU=CU, AV=AV, BV=BV, CV=CV, uRt1=uRt1, uRt2=uRt2, vRt1=vRt1, vRt2=vRt2)
    return {n: co[n].at[i, j].set(v) for n, v in vals.items()}


def seaice_lsr_rhsu(zetaMinusEta, etaPlusZeta, etaZloc, zetaZloc, pressLoc, uIceC, vIceC, rhsU, iMin, iMax, jMin,
                    jMax, *, cfg, sp, grid, sf):
    """SEAICE_LSR_RHSU( zetaMinusEta, etaPlusZeta, etaZloc, zetaZloc, pressLoc, uIceC, vIceC, rhsU, iMin, iMax,
    jMin, jMax, bi, bj, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:1616-1764

    C     | o Calculate the right-hand side of the u-momentum equation

    Returns rhsU (updated on iMin..iMax, jMin..jMax). SEAICEuseStrImpCpl (:1704-1726, lane M4CS32ICE): the explicit
    -zetaZ du/dy term added to sig12 (hFacM from seaiceMaskV, as the Fortran writes it);
    SEAICEselectMetricTerms >= 2 (:1741-1761; lane M4LAB session 3, lab_sea's spherical grid): the extra metric terms
    from SEAICE.h e11, e22, e12, ZETA, ETA, PRESS, etaZ (the commons, as the Fortran reads them) and k1AtU, k2AtU."""
    g, HEFFM, maskV = grid, sf["HEFFM"], sf["seaiceMaskV"]
    loc = sf["AREA"].local("sig", 0.)                                          # :1665-1670 sig11 = sig12 = 0
    jj, ii = loop_j(jMin, jMax), loop_i(iMin-1, iMax)                          # :1672-1673
    sig11 = loc.at[ii, jj].set(zetaMinusEta[ii, jj]                            # :1674-1679
                               * (vIceC[ii, jj+1] - vIceC[ii, jj])
                               * g.recip_dyF[ii, jj]
                               + etaPlusZeta[ii, jj] * sf["k2AtC"][ii, jj]
                               * 0.5 * (vIceC[ii, jj+1] + vIceC[ii, jj])
                               - 0.5 * pressLoc[ii, jj])
    jj, ii = loop_j(jMin, jMax+1), loop_i(iMin, iMax)                          # :1683-1684
    hFacM = maskV[ii, jj] - maskV[ii-1, jj]                                    # :1685
    sig12 = loc.at[ii, jj].set(etaZloc[ii, jj] * (                             # :1686-1700
        (vIceC[ii, jj] - vIceC[ii-1, jj])
        * g.recip_dxV[ii, jj]
        - sf["k1AtZ"][ii, jj]
        * 0.5 * (vIceC[ii, jj] + vIceC[ii-1, jj])
    )
        * HEFFM[ii, jj]*HEFFM[ii-1, jj]
        * HEFFM[ii, jj-1]*HEFFM[ii-1, jj-1]
        + etaZloc[ii, jj] * g.recip_dxV[ii, jj]
        * (vIceC[ii, jj] + vIceC[ii-1, jj])
        * hFacM * 2.0)
    if sp.SEAICEuseStrImpCpl:                                                  # :1704-1726
        hFacM = maskV[ii, jj] - maskV[ii-1, jj]                                # :1710
        sig12 = sig12.at[ii, jj].set(sig12[ii, jj] - zetaZloc[ii, jj] * (     # :1711-1723
            (uIceC[ii, jj] - uIceC[ii, jj-1])
            * g.recip_dyU[ii, jj]
        )
            * HEFFM[ii, jj]*HEFFM[ii-1, jj]
            * HEFFM[ii, jj-1]*HEFFM[ii-1, jj-1]
            - zetaZloc[ii, jj] * g.recip_dyU[ii, jj]
            * (uIceC[ii, jj] + uIceC[ii, jj-1])
            * hFacM * 2.0)
    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)                              # :1728-1729
    rhsU = rhsU.at[i, j].set(rhsU[i, j]                                        # :1731-1736
                             + g.recip_rAw[i, j] * sf["seaiceMaskU"][i, j] *
                             (g.dyF[i, j]*sig11[i, j]
                              - g.dyF[i-1, j]*sig11[i-1, j]
                              + g.dxV[i, j+1]*sig12[i, j+1]
                              - g.dxV[i, j]*sig12[i, j]))
    if sp.SEAICEselectMetricTerms >= 2:                                        # :1741-1761 (lane M4LAB)
        sz = cfg.size
        jA, iA = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)   # :1743-1744
        eplus = sf["e11"][iA, jA] + sf["e22"][iA, jA]                          # :1745
        eminus = sf["e11"][iA, jA] - sf["e22"][iA, jA]                         # :1746
        sig11 = sig11.at[iA, jA].set(sf["ZETA"][iA, jA]*eplus - sf["ETA"][iA, jA]*eminus   # :1747-1748
                                     - 0.5 * sf["PRESS"][iA, jA])
        sig12 = sig12.at[iA, jA].set(2.0 * sf["e12"][iA, jA] * sf["etaZ"][iA, jA])   # :1749
        rhsU = rhsU.at[i, j].set(rhsU[i, j]                                    # :1755-1758
                                 + sf["seaiceMaskU"][i, j] * halfRL *
                                 (sf["k2AtU"][i, j] * (sig12[i, j] + sig12[i, j+1])
                                  - sf["k1AtU"][i, j] * (sig11[i, j] + sig11[i-1, j])))
    return rhsU


def seaice_lsr_rhsv(zetaMinusEta, etaPlusZeta, etaZloc, zetaZloc, pressLoc, uIceC, vIceC, rhsV, iMin, iMax, jMin,
                    jMax, *, cfg, sp, grid, sf):
    """SEAICE_LSR_RHSV( zetaMinusEta, etaPlusZeta, etaZloc, zetaZloc, pressLoc, uIceC, vIceC, rhsV, iMin, iMax,
    jMin, jMax, bi, bj, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:1771-1919

    C     | o Calculate the right-hand side of the v-momentum equation

    Returns rhsV (updated on iMin..iMax, jMin..jMax); arms as SEAICE_LSR_RHSU (SEAICEuseStrImpCpl :1860-1882 with
    hFacM from seaiceMaskU, lane M4CS32ICE; the metric terms
    :1896-1916, lane M4LAB session 3, with k1AtV, k2AtV)."""
    g, HEFFM, maskU = grid, sf["HEFFM"], sf["seaiceMaskU"]
    loc = sf["AREA"].local("sig", 0.)                                          # :1820-1825 sig22 = sig12 = 0
    jj, ii = loop_j(jMin-1, jMax), loop_i(iMin, iMax)                          # :1828-1829
    sig22 = loc.at[ii, jj].set(zetaMinusEta[ii, jj]                            # :1830-1835
                               * (uIceC[ii+1, jj] - uIceC[ii, jj])
                               * g.recip_dxF[ii, jj]
                               + etaPlusZeta[ii, jj] * sf["k1AtC"][ii, jj]
                               * 0.5 * (uIceC[ii+1, jj] + uIceC[ii, jj])
                               - 0.5 * pressLoc[ii, jj])
    jj, ii = loop_j(jMin, jMax), loop_i(iMin, iMax+1)                          # :1839-1840
    hFacM = maskU[ii, jj] - maskU[ii, jj-1]                                    # :1841
    sig12 = loc.at[ii, jj].set(etaZloc[ii, jj] * (                             # :1842-1856
        (uIceC[ii, jj] - uIceC[ii, jj-1])
        * g.recip_dyU[ii, jj]
        - sf["k2AtZ"][ii, jj]
        * 0.5 * (uIceC[ii, jj] + uIceC[ii, jj-1])
    )
        * HEFFM[ii, jj]*HEFFM[ii-1, jj]
        * HEFFM[ii, jj-1]*HEFFM[ii-1, jj-1]
        + etaZloc[ii, jj] * g.recip_dyU[ii, jj]
        * (uIceC[ii, jj] + uIceC[ii, jj-1])
        * hFacM * 2.0)
    if sp.SEAICEuseStrImpCpl:                                                  # :1860-1882
        hFacM = maskU[ii, jj] - maskU[ii, jj-1]                                # :1866
        sig12 = sig12.at[ii, jj].set(sig12[ii, jj] - zetaZloc[ii, jj] * (     # :1867-1879
            (vIceC[ii, jj] - vIceC[ii-1, jj])
            * g.recip_dxV[ii, jj]
        )
            * HEFFM[ii, jj]*HEFFM[ii-1, jj]
            * HEFFM[ii, jj-1]*HEFFM[ii-1, jj-1]
            - zetaZloc[ii, jj] * g.recip_dxV[ii, jj]
            * (vIceC[ii, jj] + vIceC[ii-1, jj])
            * hFacM * 2.0)
    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)                              # :1884-1885
    rhsV = rhsV.at[i, j].set(rhsV[i, j]                                        # :1887-1892
                             + g.recip_rAs[i, j] * sf["seaiceMaskV"][i, j] *
                             (g.dyU[i+1, j] * sig12[i+1, j]
                              - g.dyU[i, j] * sig12[i, j]
                              + g.dxF[i, j] * sig22[i, j]
                              - g.dxF[i, j-1] * sig22[i, j-1]))
    if sp.SEAICEselectMetricTerms >= 2:                                        # :1896-1916 (lane M4LAB)
        sz = cfg.size
        jA, iA = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)   # :1898-1899
        eplus = sf["e11"][iA, jA] + sf["e22"][iA, jA]                          # :1900
        eminus = sf["e11"][iA, jA] - sf["e22"][iA, jA]                         # :1901
        sig22 = sig22.at[iA, jA].set(sf["ZETA"][iA, jA]*eplus + sf["ETA"][iA, jA]*eminus   # :1902-1903
                                     - 0.5 * sf["PRESS"][iA, jA])
        sig12 = sig12.at[iA, jA].set(2.0 * sf["e12"][iA, jA] * sf["etaZ"][iA, jA])   # :1904
        rhsV = rhsV.at[i, j].set(rhsV[i, j]                                    # :1910-1913
                                 + sf["seaiceMaskV"][i, j] * halfRL *
                                 (sf["k1AtV"][i, j] * (sig12[i, j] + sig12[i+1, j])
                                  - sf["k2AtV"][i, j] * (sig22[i, j] + sig22[i, j-1])))
    return rhsV


def _thomas(A, B, C, R, sNx_line):
    """The tridiagonal solve of one line in the Fortran order (TRIDIAGU :2011-2044 / TRIDIAGV :2161-2198): A, B, C, R
    [T, n] along the line (n = iMax-iMin+1); `CUU = C; CUU(1) = CUU(1)/B(1); URT(1) = URT(1)/B(1)`, the forward
    elimination `bet = B(i)-A(i)*CUU(i-1); CUU(i) = CUU(i)/bet; URT(i) = (URT(i)-A(i)*URT(i-1))/bet` for i = 2..n,
    then the back substitution `URT(iM) = URT(iM)-CUU(iM)*URT(iM+1)` for iM = n-1 down to 1. Returns URT [T, n]."""
    n = A.shape[-1]
    del sNx_line
    cuu0 = C[:, 0]/B[:, 0]                                                     # :2014 / :2167
    urt0 = R[:, 0]/B[:, 0]                                                     # :2015 / :2168

    def fwd(c, x):
        cuu_m, urt_m = c
        a, b, cc, r = x
        bet = b-a*cuu_m                                                        # :2028 / :2181
        cuu = cc/bet                                                           # :2029 / :2182
        urt = (r-a*urt_m)/bet                                                  # :2030 / :2183
        return (cuu, urt), (cuu, urt)
    _, (cuu, urt) = scan_k(fwd, (cuu0, urt0), range(1, n), lambda k: (A[:, k], B[:, k], C[:, k], R[:, k]))
    cuu = jnp.concatenate([cuu0[None], cuu], axis=0)                           # [n, T]
    urt = jnp.concatenate([urt0[None], urt], axis=0)

    def bwd(urt_p, x):
        u, c = x
        u = u-c*urt_p                                                          # :2043 / :2197
        return u, u
    _, ub = scan_k(bwd, urt[n-1], range(n-2, -1, -1), lambda k: (urt[k], cuu[k]))
    return jnp.moveaxis(jnp.concatenate([ub, urt[n-1][None]], axis=0), 0, 1)  # [T, n]


def _check_back_loop(lo, hi, sN):
    """TRIDIAGU :2040-2041 `DO i=iMin,iMax-1; iM=sNx-i` (and V with sNy): iM must run iMax-1 down to iMin, which
    holds when iMin+iMax = sNx+1 (iMin = 1-SEAICE_OLx, iMax = sNx+SEAICE_OLx)."""
    iMs = [sN-i for i in range(lo, hi)]
    if iMs != list(range(hi-1, lo-1, -1)):
        raise NotImplementedError("SEAICE_LSR_TRIDIAG: iM = sNx-i does not run iMax-1..iMin for these bounds")


def seaice_lsr_tridiagu(AU, BU, CU, uRt1, uRt2, rhsU, uTmp, seaiceMaskU, WFAU, uIce, iMin, iMax, jMin, jMax, *,
                        cfg):
    """SEAICE_LSR_TRIDIAGU( AU, BU, CU, uRt1, uRt2, rhsU, uTmp, seaiceMaskU, WFAU, uIce, iMin, iMax, jMin, jMax, bi,
    bj, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:1926-2064

    C     | o Solve tridiagonal problem in uIce for LSR solver

    No SEAICE_LSR_ZEBRA (jStep = 1, :1979): the rows j = jMin..jMax are solved in order and row j reads row j-1 as
    just updated (`uIce(i,j-1)` at :2006): a `scan_k` over j. Within a row the Thomas algorithm (`_thomas`). Returns
    uIce updated on iMin..iMax, jMin..jMax.
    With SEAICE_LSR_ZEBRA (lane M4LAB session 3, lab_sea; jStep = 2, :1977): `DO k=0,1` (:1993-1994) solves the rows
    jMin+k, jMin+k+2, ... <= jMax; a row reads only rows of the other parity (j-1, j+1) and its own halo points
    (iMin-1, iMax+1), which this sweep does not write, so the rows of one k are independent (vectorised, one Thomas
    solve per row) and the second sweep reads the rows the first one wrote (`_zebra`)."""
    _check_back_loop(iMin, iMax, cfg.size.sNx)
    ii = loop_i(iMin, iMax)
    if cfg.cpp.flag("SEAICE_LSR_ZEBRA", "SEAICE_OPTIONS.h"):
        return _zebra(AU, BU, CU, uRt1, uRt2, rhsU, uTmp, seaiceMaskU, WFAU, uIce, iMin, iMax, jMin, jMax, "u")

    def row(j):
        r = lambda F: F[ii, j][:, 0, :]                                        # noqa: E731  [T, ni]
        return dict(A=r(AU), B=r(BU), C=r(CU), R1=r(uRt1), R2=r(uRt2), rhs=r(rhsU), tmp=r(uTmp),
                    msk=r(seaiceMaskU), nxt=uIce[ii, j+1][:, 0, :],
                    w=uIce[iMin-1, j], e=uIce[iMax+1, j])

    def body(prev, x):
        n = x["A"].shape[-1]
        AA3 = jnp.zeros_like(x["rhs"])                                         # :2000
        AA3 = AA3.at[:, 0].set(AA3[:, 0] - x["A"][:, 0]*x["w"])                # :2001 (i.EQ.iMin)
        AA3 = AA3.at[:, n-1].set(AA3[:, n-1] - x["C"][:, n-1]*x["e"])          # :2002 (i.EQ.iMax)
        URT = (x["rhs"]                                                        # :2004-2007
               + AA3
               + x["R1"]*prev
               + x["R2"]*x["nxt"])
        URT = URT * x["msk"]                                                   # :2008
        URT = _thomas(x["A"], x["B"], x["C"], URT, None)                       # :2011-2044
        new = x["tmp"] + WFAU*(URT - x["tmp"])                                 # :2054-2056
        return new, new
    _, rows = scan_k(body, uIce[ii, jMin-1][:, 0, :], range(jMin, jMax+1), row)
    return uIce.at[ii, loop_j(jMin, jMax)].set(jnp.moveaxis(rows, 0, 1))


def seaice_lsr_tridiagv(AV, BV, CV, vRt1, vRt2, rhsV, vTmp, seaiceMaskV, WFAV, vIce, iMin, iMax, jMin, jMax, *,
                        cfg):
    """SEAICE_LSR_TRIDIAGV( AV, BV, CV, vRt1, vRt2, rhsV, vTmp, seaiceMaskV, WFAV, vIce, iMin, iMax, jMin, jMax, bi,
    bj, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:2071-2219

    C     | o Solve tridiagonal problem in vIce for LSR solver

    As TRIDIAGU with the roles of i and j exchanged: columns i = iMin..iMax in order (iStep = 1, :2124), column i
    reads column i-1 as just updated (:2156). Returns vIce updated on iMin..iMax, jMin..jMax. SEAICE_LSR_ZEBRA (iStep
    = 2, :2122; `DO k=0,1`, :2138-2139) as in TRIDIAGU (`_zebra`)."""
    _check_back_loop(jMin, jMax, cfg.size.sNy)
    jj = loop_j(jMin, jMax)
    if cfg.cpp.flag("SEAICE_LSR_ZEBRA", "SEAICE_OPTIONS.h"):
        return _zebra(AV, BV, CV, vRt1, vRt2, rhsV, vTmp, seaiceMaskV, WFAV, vIce, iMin, iMax, jMin, jMax, "v")

    def col(i):
        c = lambda F: F[i, jj][:, :, 0]                                        # noqa: E731  [T, nj]
        return dict(A=c(AV), B=c(BV), C=c(CV), R1=c(vRt1), R2=c(vRt2), rhs=c(rhsV), tmp=c(vTmp),
                    msk=c(seaiceMaskV), nxt=vIce[i+1, jj][:, :, 0],
                    s=vIce[i, jMin-1], n=vIce[i, jMax+1])

    def body(prev, x):
        n = x["A"].shape[-1]
        AA3 = jnp.zeros_like(x["rhs"])                                         # :2150
        AA3 = AA3.at[:, 0].set(AA3[:, 0] - x["A"][:, 0]*x["s"])                # :2151 (j.EQ.jMin)
        AA3 = AA3.at[:, n-1].set(AA3[:, n-1] - x["C"][:, n-1]*x["n"])          # :2152 (j.EQ.jMax)
        VRT = (x["rhs"]                                                        # :2154-2157
               + AA3
               + x["R1"]*prev
               + x["R2"]*x["nxt"])
        VRT = VRT * x["msk"]                                                   # :2158
        VRT = _thomas(x["A"], x["B"], x["C"], VRT, None)                       # :2161-2198
        new = x["tmp"] + WFAV*(VRT - x["tmp"])                                 # :2206-2209
        return new, new
    _, cols = scan_k(body, vIce[iMin-1, jj][:, :, 0], range(iMin, iMax+1), col)
    return vIce.at[loop_i(iMin, iMax), jj].set(jnp.moveaxis(cols, 0, 2))


def _zebra(A_, B_, C_, Rt1, Rt2, rhs, tmp, msk, WFA, fld, iMin, iMax, jMin, jMax, uv):
    """SEAICE_LSR_TRIDIAGU (uv = "u": lines are rows j, :1993-2061) / TRIDIAGV ("v": lines are columns i,
    :2138-2213) with SEAICE_LSR_ZEBRA: for k = 0, 1 (`DO k=0,1`) the lines lo+k, lo+k+2, ... <= hi (jStep / iStep
    = 2) at once: AA3 (:2000-2002 / :2150-2152), URT/VRT = rhs + AA3 + Rt1*(line-1) + Rt2*(line+1) (:2004-2007 /
    :2154-2157), times the mask (:2008 / :2158), the Thomas solve in the Fortran order (`_thomas`, one per line), the
    relaxation tmp + WFA*(URT - tmp) (:2054-2056 / :2206-2209). Returns fld updated on iMin..iMax, jMin..jMax."""
    if uv == "u":
        lo, hi = jMin, jMax
        along = loop_i(iMin, iMax)
        line = lambda F, n: F[along, n][:, 0, :]                              # noqa: E731  [T, ni]
        edge = lambda n: (fld[iMin-1, n], fld[iMax+1, n])                     # noqa: E731  (i.EQ.iMin, i.EQ.iMax)
        put = lambda f, n, v: f.at[along, loop_j(n, n)].set(v[:, None, :])    # noqa: E731
    else:
        lo, hi = iMin, iMax
        along = loop_j(jMin, jMax)
        line = lambda F, n: F[n, along][:, :, 0]                              # noqa: E731  [T, nj]
        edge = lambda n: (fld[n, jMin-1], fld[n, jMax+1])                     # noqa: E731  (j.EQ.jMin, j.EQ.jMax)
        put = lambda f, n, v: f.at[loop_i(n, n), along].set(v[:, :, None])    # noqa: E731
    for k in (0, 1):                                                           # DO k=0,1 (:1993 / :2138)
        ns = list(range(lo + k, hi + 1, 2))                                    # :1998 / :2147 (step 2)
        if not ns:
            continue
        st = lambda F, d=0: jnp.stack([line(F, n + d) for n in ns], axis=1)   # noqa: E731  [T, nline, nalong]
        A, B, C = st(A_), st(B_), st(C_)
        R1, R2, rh, tp, mk = st(Rt1), st(Rt2), st(rhs), st(tmp), st(msk)
        prev, nxt = st(fld, -1), st(fld, 1)
        w = jnp.stack([edge(n)[0] for n in ns], axis=1)                        # [T, nline]
        e = jnp.stack([edge(n)[1] for n in ns], axis=1)
        nal = A.shape[-1]
        AA3 = jnp.zeros_like(rh)                                               # :2000 / :2150
        AA3 = AA3.at[:, :, 0].set(AA3[:, :, 0] - A[:, :, 0]*w)                # :2001 / :2151
        AA3 = AA3.at[:, :, nal-1].set(AA3[:, :, nal-1] - C[:, :, nal-1]*e)    # :2002 / :2152
        RT = (rh                                                               # :2004-2007 / :2154-2157
              + AA3
              + R1*prev
              + R2*nxt)
        RT = RT * mk                                                           # :2008 / :2158
        T_, nl = RT.shape[0], RT.shape[1]
        flat = lambda X: X.reshape(T_*nl, nal)                                 # noqa: E731
        RT = _thomas(flat(A), flat(B), flat(C), flat(RT), None).reshape(T_, nl, nal)   # :2011-2044 / :2161-2198
        new = tp + WFA*(RT - tp)                                               # :2054-2056 / :2206-2209
        for idx, n in enumerate(ns):
            fld = put(fld, n, new[:, idx])
    return fld


LSR_SF = ("UICE", "VICE", "uIceNm1", "vIceNm1", "e11", "e22", "e12", "ETA", "ZETA", "etaZ", "zetaZ", "PRESS",
          "deltaC", "DWATN", "FORCEX", "FORCEY", "uice_fd", "vice_fd")


def seaice_lsr(myTime, myIter, sf, *, cfg, sp, op, grid, state, ex):
    """SEAICE_LSR( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_lsr.F:24-1167

    C     | o Solve ice momentum equation with an LSR dynamics solver
    C     |   (see Zhang and Hibler,   JGR, 102, 8691-8702, 1997
    C     |    and Zhang and Rothrock, MWR, 131,  845- 861, 2003)

    `sf` SEAICE.h / SEAICE_GRID.h, `state` (uVel, vVel), `op` OceanParams (debugLevel, deltaTClock, globalArea,
    rhoConst, usingPCoords). Returns (sf, out): the SEAICE.h fields LSR_SF updated, `out` the per-pass STDOUT values
    ({name: [nonLinIterLoc]}, OUT_FLOAT / OUT_INT / OUT_BOOL).
    Ported for the build's options (SEAICE_GLOBAL_3DIAG_SOLVER, SEAICE_ALLOW_LSR_FLEX, SEAICE_ALLOW_FREEDRIFT, ALLOW_DEBUG
    defined; ALLOW_AUTODIFF (lane M4ADLAB: ported, below), SEAICE_ALLOW_MOM_ADVECTION, SEAICE_ALLOW_BOTTOMDRAG,
    SEAICE_ALLOW_SIDEDRAG,
    SEAICE_ALLOW_CHECK_LSR_CONVERGENCE not) and the run-time arms of input.dyn_lsr: SEAICEuseLSRflex, LSR_mixIniGuess
    = 1, no multi-tile solver, no Picard preconditioning; the others raise in SEAICE_READPARMS.
    Lane M4LAB session 3 (lab_sea/code + input): the build without SEAICE_ALLOW_LSR_FLEX and
    SEAICE_GLOBAL_3DIAG_SOLVER, with SEAICE_LSR_ZEBRA and SEAICE_ALLOW_BOTTOMDRAG: the pass runs unconditionally
    (doNonLinLoop stays .TRUE., :281), the linear iteration's SOLV_NCHECK test is the max-norm of the update
    (:934-983: MAX_CHAIN in tile order, _GLOBAL_MAX_RL, the WFAU/WFAV safeguard against S1A/S2A = 0.80, :311-312),
    dragSym = DWATN*COSWAT + CbotC (SEAICE_BOTTOMDRAG_COEFFS :384-393 writes nothing with SEAICEbasalDragK2 = 0),
    LSR_mixIniGuess = 0 (the free-drift uIce_fd/vIce_fd of SEAICE_FREEDRIFT are not overwritten, :615-638; their
    residual :639-644 is printed only), SEAICEselectMetricTerms = 2 (SEAICE_LSR_RHSU/V :1741-1761, :1896-1916). The DEBUG_STATS prints
    (:595-602, :1038-1045, debugLevel >= debLevD) raise; the SIlsrRe diagnostic (:1116-1136) is output only;
    useHB87StressCoupling (:1139-1164) raises in SEAICE_OCEAN_STRESS."""
    # lane M4LAB: the build options above were assumed, not checked (lab_sea defines SEAICE_LSR_ZEBRA, :1975-1980,
    # and SEAICE_ALLOW_BOTTOMDRAG, :384-394/:420-422: this port would have run the dyn_lsr arms silently)
    # session 3: SEAICE_LSR_ZEBRA (TRIDIAGU/V) and SEAICE_ALLOW_BOTTOMDRAG (:384-393, :420-422) are ported
    for o in ("SEAICE_VECTORIZE_LSR", "SEAICE_ALLOW_SIDEDRAG", "SEAICE_ALLOW_CHECK_LSR_CONVERGENCE",
              "SEAICE_ALLOW_MOM_ADVECTION"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_LSR: {o} is not ported")
    # lane M4LAB session 2 audit: the defined options of the docstring were not checked either, and ALLOW_OBCS would
    # include OBCS_OPTIONS.h instead of the file's own `#define OBCS_UVICE_OLD` (:2-6), which drops the open-boundary
    # rows (:564-589). Session 3: the build without SEAICE_GLOBAL_3DIAG_SOLVER (only SEAICEuseMultiTileSolver, refused
    # in SEAICE_READPARMS, reads it: :808-875) and without SEAICE_ALLOW_LSR_FLEX (lab_sea) are ported
    # lane M4CS32ICE: the build without SEAICE_ALLOW_FREEDRIFT (global_ocean.cs32x15) is ported: :162-168 (locals),
    # :293-296 (TAF), :614-699 (the free-drift guess, its residual and the LSR_mixIniGuess >= 2 mix) and :1019-1028
    # (the FrDrift STDOUT line) are not compiled there (session 1 ported the arms and left this refusal in place)
    if sp.LSR_mixIniGuess >= 2:                                                # :646-689
        raise NotImplementedError("SEAICE_LSR: LSR_mixIniGuess >= 2 (:646-689) is not ported")
    # the flexible convergence criterion: compiled (SEAICE_ALLOW_LSR_FLEX) and switched on (static)
    flex = cfg.cpp.flag("SEAICE_ALLOW_LSR_FLEX", "SEAICE_OPTIONS.h") and bool(sp.SEAICEuseLSRflex)
    bottomdrag = cfg.cpp.flag("SEAICE_ALLOW_BOTTOMDRAG", "SEAICE_OPTIONS.h")
    if cfg.cpp.flag("ALLOW_OBCS", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_LSR: ALLOW_OBCS is not ported")
    # lane M4ADLAB session 2 (lab_sea/code_ad): ALLOW_AUTODIFF / ALLOW_AUTODIFF_TAMC. Forward arms: the resets
    # :201-220 (below) and, without the STDOUT / errorMessageUnit lines, :686-697 (residUmix: LSR_mixIniGuess >= 2,
    # raises above) and :1003-1058 (the residual lines and the non-convergence WARNING: `out` marks them unprinted,
    # "printWarn"). The other arms are CADJ directives and tape keys (:248-251, :291-306, :357-366, :401-404,
    # :647-650, :784-809, :878-882, :1140-1145): no forward value.
    autodiff = cfg.cpp.flag("ALLOW_AUTODIFF", "SEAICE_OPTIONS.h")
    tamc = cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", "AUTODIFF_OPTIONS.h") if autodiff else False
    if op.debugLevel >= debLevD:
        raise NotImplementedError("SEAICE_LSR: debugLevel >= debLevD (DEBUG_STATS_RL, :595-602) is not ported")
    if sp.useHB87stressCoupling:
        raise NotImplementedError("SEAICE_LSR: useHB87StressCoupling (:1139-1164) is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    printResidual = (op.debugLevel >= debLevA) & different_multiple_traced(   # :173-174
        sp.SEAICE_monFreq, myTime, op.deltaTClock)
    jMin = 1 - sp.SEAICE_OLy                                                   # :177
    jMax = sNy + sp.SEAICE_OLy                                                 # :178
    iMin = 1 - sp.SEAICE_OLx                                                   # :179
    iMax = sNx + sp.SEAICE_OLx                                                 # :180
    linearIterLoc = sp.SEAICElinearIterMax                                     # :182
    # :782-790 with SEAICE_LSR_ADJOINT_ITER (an AD build that tapes every sweep, lnkey :785-789): the DO m loop as
    # plan decision 14's fixed-length scan (mitjax/ad/lsr_sweeps.py, values bitwise the while_loop's); any other
    # build keeps the while_loop (its derivative: mitjax/ad/seaice_lsr_rule.py, forward only)
    # lane M4ADCS32ICE session 2: the choice is mitjax/ad/modes.lsr_derivative's (A1 also by the explicit static
    # option sp.mjx_lsr_derivative = "sweeps", plan decision 17; it refuses a loop bound above SOLV_MAX_FIXED)
    if lsr_derivative(cfg, sp) == "sweeps":
        from mitjax.ad.lsr_sweeps import SOLV_MAX_FIXED, taped_sweeps

        def sweep_loop(cond_, body_, st_):
            return taped_sweeps(cond_, body_, st_, SOLV_MAX_FIXED)
    else:
        sweep_loop = lax.while_loop
    nonLinIterLoc = sp.SEAICEnonLinIterMax                                     # :183 (no Picard precond., :184-187)
    recip_deltaT = 1.0 / sp.SEAICE_deltaTdyn                                   # :189
    kSrf = sz.Nr if op.usingPCoords else 1                                     # :192-196
    SINWAT = jnp.sin(sp.SEAICE_waterTurnAngle*deg2rad)                         # :198 (_RS SINWAT)
    COSWAT = jnp.cos(sp.SEAICE_waterTurnAngle*deg2rad)                         # :199
    uVel, vVel, fCori = state.uVel, state.vVel, grid.fCori
    AREA = sf["AREA"]
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)
    sf = dict(sf)
    if autodiff:                                                               # :201-220 (ALLOW_AUTODIFF)
        for n in ("deltaC", "PRESS", "ZETA", "zetaZ", "ETA", "etaZ", "uIceNm1", "vIceNm1"):   # :207-216
            sf[n] = sf[n].at[iA, jA].set(0.)                  # uIceC, vIceC (:213-214): the zero locals of c0
    # :224-246 areaW, areaS
    areaW = AREA.local("areaW", 1.0)                                           # :228
    areaS = AREA.local("areaS", 1.0)                                           # :229
    areaMinLoc = 0.0                                                           # :232
    if sp.SEAICEscaleSurfStress:                                               # :233-244
        areaMinLoc = 1.0e-10                                                   # :234
        areaW = areaW.at[i, j].set(0.5*(AREA[i, j]+AREA[i-1, j]))              # :238-239
        areaS = areaS.at[i, j].set(0.5*(AREA[i, j]+AREA[i, j-1]))              # :240-241
    # :256-271
    sf["uIceNm1"] = sf["uIceNm1"].at[iA, jA].set(sf["UICE"][iA, jA])           # :260
    sf["vIceNm1"] = sf["vIceNm1"].at[iA, jA].set(sf["VICE"][iA, jA])           # :261
    fxTmp = AREA.local("fxTmp").at[iA, jA].set(sf["FORCEX0"][iA, jA]           # :262-264
                                               + sf["seaiceMassU"][iA, jA]*recip_deltaT
                                               * sf["uIceNm1"][iA, jA])
    fyTmp = AREA.local("fyTmp").at[iA, jA].set(sf["FORCEY0"][iA, jA]           # :265-267
                                               + sf["seaiceMassV"][iA, jA]*recip_deltaT
                                               * sf["vIceNm1"][iA, jA])
    zero = AREA.local("zero", 0.)
    co0 = {n: zero for n in ("AU", "BU", "CU", "AV", "BV", "CV", "uRt1", "uRt2", "vRt1", "vRt2")}
    f64 = lambda x: jnp.asarray(x, jnp.float64)                                # noqa: E731
    i32 = lambda x: jnp.asarray(x, jnp.int32)                                  # noqa: E731
    out0 = {**{n: jnp.zeros((nonLinIterLoc,)) for n in OUT_FLOAT},
            **{n: jnp.zeros((nonLinIterLoc,), jnp.int32) for n in OUT_INT},
            **{n: jnp.zeros((nonLinIterLoc,), bool) for n in OUT_BOOL}}
    c0 = dict(sf={n: sf[n] for n in LSR_SF if n in sf}, uIceC=zero, vIceC=zero, rhsU=zero, rhsV=zero, co=co0,
              dragSym=zero, residIniNonLin=f64(1.0),                           # :274-278 (residkm1/2: below)
              doNonLinLoop=jnp.asarray(True),                                  # :281
              residUini=f64(0.), residVini=f64(0.), residU_fd=f64(0.), residV_fd=f64(0.),
              S1=f64(0.), S2=f64(0.), WFAU=f64(0.), WFAV=f64(0.), ICOUNT1=i32(0), ICOUNT2=i32(0),
              doIterate4u=jnp.asarray(False), doIterate4v=jnp.asarray(False),
              LSR_ERRORfu=f64(0.), LSR_ERRORfv=f64(0.), out=out0)
    # the scalars of the carry stay typed invariant under shard_map: every one derives from constants and from
    # SEAICE_RESIDUAL's sums over all tiles (eesupp all_tiles: invariant), so both branches of the lax.cond below
    # return them alike; the fields derive from the sharded sf (varying)

    def with_sf(c, new):
        return {**c, "sf": {**c["sf"], **{n: v for n, v in new.items() if n in LSR_SF}}}

    def sec_a(c, ipass):
        """The first IF ( doNonLinLoop ) block (:300-721)."""
        s = {**sf, **c["sf"]}
        WFAU = f64(sp.SEAICE_LSRrelaxU) * jnp.ones_like(c["WFAU"])             # :309
        WFAV = f64(sp.SEAICE_LSRrelaxV) * jnp.ones_like(c["WFAV"])             # :310
        S1 = jnp.zeros_like(c["S1"])                                           # :313 (S1 = 0.)
        S2 = jnp.zeros_like(c["S2"])                                           # :314
        # :311-312 WFAU2 = WFAV2 = ZERO and :315-316 S1A = S2A = 0.80 _d 0: read only by the max-norm arm (sec_c)
        # :318-355 uIceC, vIceC
        uIce, vIce = s["UICE"], s["VICE"]
        if sp.SEAICEnonLinIterMax <= 2:                                        # :328 (static part)
            uIce2 = uIce.at[iA, jA].set(HALF*(uIce[iA, jA]+s["uIceNm1"][iA, jA]))   # :332
            vIce2 = vIce.at[iA, jA].set(HALF*(vIce[iA, jA]+s["vIceNm1"][iA, jA]))   # :333
        else:
            uIce2, vIce2 = uIce, vIce
        uAvg = c["uIceC"].at[iA, jA].set(HALF*(uIce[iA, jA]+c["uIceC"][iA, jA]))   # :348
        vAvg = c["vIceC"].at[iA, jA].set(HALF*(vIce[iA, jA]+c["vIceC"][iA, jA]))   # :349
        first = ipass == 1
        second = (ipass == 2) & (sp.SEAICEnonLinIterMax <= 2)
        uIceC = _sel3(first, uIce, second, uIce2, uAvg)                        # :324, :334, :348
        vIceC = _sel3(first, vIce, second, vIce2, vAvg)                        # :325, :335, :349
        uIce = _sel3(first, uIce, second, uIce2, uIce)                         # :332 (ipass 2 only)
        vIce = _sel3(first, vIce, second, vIce2, vIce)
        e11, e22, e12 = seaice_calc_strainrates(uIceC, vIceC, s["e11"], s["e22"], s["e12"], ipass, myTime, myIter,
                                                cfg=cfg, sp=sp, grid=grid, sf=s)   # :368-371
        eta, etaZ, zeta, zetaZ, press, deltaC = seaice_calc_viscosities(       # :373-377
            e11, e22, e12, s["SEAICE_zMin"], s["SEAICE_zMax"], s["HEFFM"], s["PRESS0"], s["tensileStrFac"],
            s["ETA"], s["etaZ"], s["ZETA"], s["zetaZ"], s["PRESS"], s["deltaC"], ipass, myTime, myIter,
            cfg=cfg, sp=sp, grid=grid)
        DWATN = seaice_oceandrag_coeffs(uIceC, vIceC, s["HEFFM"], s["DWATN"], ipass, myTime, myIter,   # :379-382
                                        cfg=cfg, sp=sp, op=op, grid=grid, state=state)
        s = {**s, "UICE": uIce, "VICE": vIce, "e11": e11, "e22": e22, "e12": e12, "ETA": eta, "etaZ": etaZ,
             "ZETA": zeta, "zetaZ": zetaZ, "PRESS": press, "deltaC": deltaC, "DWATN": DWATN}
        # :409-426
        jj, ii = loop_j(jMin-1, jMax), loop_i(iMin-1, iMax)
        etaPlusZeta = zero.at[ii, jj].set(eta[ii, jj]+zeta[ii, jj])            # :413
        zetaMinusEta = zero.at[ii, jj].set(zeta[ii, jj]-eta[ii, jj])           # :414
        if bottomdrag:                                                         # :384-393 (lane M4LAB)
            CbotC = seaice_bottomdrag_coeffs(uIceC, vIceC, s["HEFFM"], s["HEFF"], s["AREA"], s["CbotC"], ipass,
                                             myTime, myIter, cfg=cfg, sp=sp)
            s = {**s, "CbotC": CbotC}
            dragSym = c["dragSym"].at[iA, jA].set(DWATN[iA, jA]*COSWAT          # :419-422
                                                  + CbotC[iA, jA])
        else:
            dragSym = c["dragSym"].at[iA, jA].set(DWATN[iA, jA]*COSWAT)        # :419
        # :431-482 FORCEX, FORCEY
        FORCEX = s["FORCEX"].at[i, j].set(fxTmp[i, j] +                        # :437-447
                                          (0.5 * (DWATN[i, j]+DWATN[i-1, j]) *
                                           COSWAT * uVel[i, j, kSrf]
                                           - jnp.copysign(SINWAT, fCori[i, j]) * 0.5 *
                                           (DWATN[i, j] * 0.5 *
                                            (vVel[i, j, kSrf]-vIceC[i, j]
                                             + vVel[i, j+1, kSrf]-vIceC[i, j+1])
                                            + DWATN[i-1, j] * 0.5 *
                                            (vVel[i-1, j, kSrf]-vIceC[i-1, j]
                                             + vVel[i-1, j+1, kSrf]-vIceC[i-1, j+1])
                                            )) * areaW[i, j])
        FORCEY = s["FORCEY"].at[i, j].set(fyTmp[i, j] +                        # :448-458
                                          (0.5 * (DWATN[i, j]+DWATN[i, j-1]) *
                                           COSWAT * vVel[i, j, kSrf]
                                           + jnp.copysign(SINWAT, fCori[i, j]) * 0.5 *
                                           (DWATN[i, j] * 0.5 *
                                            (uVel[i, j, kSrf]-uIceC[i, j]
                                             + uVel[i+1, j, kSrf]-uIceC[i+1, j])
                                            + DWATN[i, j-1] * 0.5 *
                                            (uVel[i, j-1, kSrf]-uIceC[i, j-1]
                                             + uVel[i+1, j-1, kSrf]-uIceC[i+1, j-1])
                                            )) * areaS[i, j])
        massC = s["seaiceMassC"]
        FORCEX = FORCEX.at[i, j].set(FORCEX[i, j] + HALF *                     # :464-468
                                     (massC[i, j] * fCori[i, j]
                                      * 0.5*(vIceC[i, j]+vIceC[i, j+1])
                                      + massC[i-1, j] * fCori[i-1, j]
                                      * 0.5*(vIceC[i-1, j]+vIceC[i-1, j+1])))
        FORCEY = FORCEY.at[i, j].set(FORCEY[i, j] - HALF *                     # :469-473
                                     (massC[i, j] * fCori[i, j]
                                      * 0.5*(uIceC[i, j]+uIceC[i+1, j])
                                      + massC[i, j-1] * fCori[i, j-1]
                                      * 0.5*(uIceC[i, j-1]+uIceC[i+1, j-1])))
        FORCEX = FORCEX.at[i, j].set(FORCEX[i, j]*s["seaiceMaskU"][i, j])      # :479
        FORCEY = FORCEY.at[i, j].set(FORCEY[i, j]*s["seaiceMaskV"][i, j])      # :480
        s = {**s, "FORCEX": FORCEX, "FORCEY": FORCEY}
        rhsU = c["rhsU"].at[i, j].set(FORCEX[i, j])                            # :489
        rhsU = seaice_lsr_rhsu(zetaMinusEta, etaPlusZeta, etaZ, zetaZ, press, uIceC, vIceC, rhsU,   # :492-496
                               iMin, iMax, jMin, jMax, cfg=cfg, sp=sp, grid=grid, sf=s)
        rhsV = c["rhsV"].at[i, j].set(FORCEY[i, j])                            # :502
        rhsV = seaice_lsr_rhsv(zetaMinusEta, etaPlusZeta, etaZ, zetaZ, press, uIceC, vIceC, rhsV,   # :505-509
                               iMin, iMax, jMin, jMax, cfg=cfg, sp=sp, grid=grid, sf=s)
        co = seaice_lsr_calc_coeffs(etaPlusZeta, zetaMinusEta, etaZ, zetaZ, dragSym, c["co"],   # :542-545
                                    iMin, iMax, jMin, jMax, myTime, myIter, cfg=cfg, sp=sp, grid=grid, sf=s,
                                    useCubedSphereExchange=op.useCubedSphereExchange)
        res = lambda u, v: seaice_residual(rhsU, rhsV, co["uRt1"], co["uRt2"], co["vRt1"], co["vRt2"],  # noqa: E731
                                           co["AU"], co["BU"], co["CU"], co["AV"], co["BV"], co["CV"], u, v,
                                           printResidual, cfg=cfg, grid=grid, ex=ex, globalArea=op.globalArea)
        # :606-612 (LSR_mixIniGuess >= 1 or SEAICEuseLSRflex: always; else with printResidual, where the values
        # are only printed: computed alike, zero without printResidual as SEAICE_RESIDUAL's calcMeanResid)
        residUini, residVini = res(s["UICE"], s["VICE"])
        # :614-645 (#ifdef SEAICE_ALLOW_FREEDRIFT; lane M4CS32ICE: global_ocean.cs32x15 compiles none of it, so
        # neither the free-drift guess nor its printed residual exists there)
        fd = cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h")
        if not fd:
            residU_fd = residV_fd = jnp.float64(0.)
        elif sp.LSR_mixIniGuess >= 1:                                          # :616 (input.dyn_lsr)
            dsym = dragSym
            uIce_fd = s["uice_fd"].at[i, j].set(FORCEX[i, j]                   # :621-626
                                                / (1.0 - s["seaiceMaskU"][i, j]
                                                   + s["seaiceMassU"][i, j]*recip_deltaT
                                                   + HALF*(dsym[i, j] + dsym[i-1, j])
                                                   * MAX(areaW[i, j], areaMinLoc, p="b")))
            vIce_fd = s["vice_fd"].at[i, j].set(FORCEY[i, j]                   # :627-632
                                                / (1.0 - s["seaiceMaskV"][i, j]
                                                   + s["seaiceMassV"][i, j]*recip_deltaT
                                                   + HALF*(dsym[i, j] + dsym[i, j-1])
                                                   * MAX(areaS[i, j], areaMinLoc, p="b")))
            uIce_fd, vIce_fd = EXCH_UV_XY_RL(uIce_fd, vIce_fd, True, ex=ex)   # :637
        else:                     # LSR_mixIniGuess = 0 (lab_sea): SEAICE_FREEDRIFT's values (seaice_dynsolver.F:307)
            uIce_fd, vIce_fd = s["uice_fd"], s["vice_fd"]
        if fd:
            if sp.LSR_mixIniGuess >= 0:                                        # :639-644 (printed only)
                residU_fd, residV_fd = res(uIce_fd, vIce_fd)
            else:
                residU_fd = residV_fd = jnp.float64(0.)
            s = {**s, "uice_fd": uIce_fd, "vice_fd": vIce_fd}
        c = with_sf(c, s)
        return {**c, "uIceC": uIceC, "vIceC": vIceC, "rhsU": rhsU, "rhsV": rhsV, "co": co, "dragSym": dragSym,
                "residUini": residUini, "residVini": residVini, "residU_fd": residU_fd, "residV_fd": residV_fd,
                "S1": S1, "S2": S2, "WFAU": WFAU, "WFAV": WFAV,
                "doIterate4u": jnp.asarray(True), "doIterate4v": jnp.asarray(True),   # :705-706
                "ICOUNT1": i32(linearIterLoc), "ICOUNT2": i32(linearIterLoc)}         # :716-717

    def sec_c(c, ipass):
        """The second IF ( doNonLinLoop ) block (:779-1111): the linear iteration, the residual print, the masks."""
        s = {**sf, **c["sf"]}
        co, rhsU, rhsV = c["co"], c["rhsU"], c["rhsV"]
        res = lambda u, v: seaice_residual(rhsU, rhsV, co["uRt1"], co["uRt2"], co["vRt1"], co["vRt2"],  # noqa: E731
                                           co["AU"], co["BU"], co["CU"], co["AV"], co["BV"], co["CV"], u, v,
                                           printResidual, cfg=cfg, grid=grid, ex=ex, globalArea=op.globalArea)
        maskU, maskV = s["seaiceMaskU"], s["seaiceMaskV"]
        WFAU, WFAV = c["WFAU"], c["WFAV"]

        def cond(st):
            m, _, _, du, dv = st[:5]
            return (m <= linearIterLoc) & (du | dv)                            # :782, :796

        def body(st):
            m, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2 = st
            if op.useCubedSphereExchange:                                      # :798-801
                du = jnp.asarray(True)
                dv = jnp.asarray(True)
            uTmp, vTmp = uIce, vIce                                            # :885-890
            uNew = seaice_lsr_tridiagu(co["AU"], co["BU"], co["CU"], co["uRt1"], co["uRt2"], rhsU, uTmp,   # :894-897
                                       maskU, WFAU, uIce, iMin, iMax, jMin, jMax, cfg=cfg)
            uIce = _sel(du, uNew, uIce)                                        # :892-898
            vNew = seaice_lsr_tridiagv(co["AV"], co["BV"], co["CV"], co["vRt1"], co["vRt2"], rhsV, vTmp,   # :902-905
                                       maskV, WFAV, vIce, iMin, iMax, jMin, jMax, cfg=cfg)
            vIce = _sel(dv, vNew, vIce)                                        # :900-906
            check = (m % sp.SOLV_NCHECK == 0) & (du | dv)                      # :918-919
            r1, r2 = res(uIce, vIce)                                           # :920-924
            S1 = jnp.where(check, r1, S1)
            S2 = jnp.where(check, r2, S2)
            conv_u = check & (S1 < c["LSR_ERRORfu"])                           # :925-928
            conv_v = check & (S2 < c["LSR_ERRORfv"])                           # :929-932
            ICOUNT1 = jnp.where(conv_u, m, ICOUNT1)
            du = du & ~conv_u
            ICOUNT2 = jnp.where(conv_v, m, ICOUNT2)
            dv = dv & ~conv_v
            uIce, vIce = EXCH_UV_XY_RL(uIce, vIce, True, ex=ex)               # :987
            return (m + 1, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2)
        def body_maxnorm(st):
            """One `DO m` iteration of the build / run without the flexible criterion (:782-990, the max-norm
            arm :934-983; lane M4LAB session 3)."""
            m, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2, WFAU_, WFAV_, S1A, S2A = st
            if op.useCubedSphereExchange:                                      # :798-801
                du = jnp.asarray(True)
                dv = jnp.asarray(True)
            uTmp, vTmp = uIce, vIce                                            # :885-890
            uNew = seaice_lsr_tridiagu(co["AU"], co["BU"], co["CU"], co["uRt1"], co["uRt2"], rhsU, uTmp,   # :894-897
                                       maskU, WFAU_, uIce, iMin, iMax, jMin, jMax, cfg=cfg)
            uIce = _sel(du, uNew, uIce)                                        # :892-898
            vNew = seaice_lsr_tridiagv(co["AV"], co["BV"], co["CV"], co["vRt1"], co["vRt2"], rhsV, vTmp,   # :902-905
                                       maskV, WFAV_, vIce, iMin, iMax, jMin, jMax, cfg=cfg)
            vIce = _sel(dv, vNew, vIce)                                        # :900-906
            jI, iI = loop_j(1, sNy), loop_i(1, sNx)                            # :940-941, :965-966
            chk = (m % sp.SOLV_NCHECK) == 0
            cu = du & chk                                                      # :936
            UERR = ((uIce[iI, jI]-uTmp[iI, jI])                                # :942-943
                    * maskU[iI, jI])
            S1n = MAX_CHAIN(ZERO, jnp.abs(UERR), p="a", acc="b", ex=ex)       # :937-948 (MAX :944), :949
            WFAU_ = jnp.where(cu & (m > 1) & (S1n > S1A), WFAU2, WFAU_)        # :953
            S1A = jnp.where(cu, S1n, S1A)                                      # :954
            S1 = jnp.where(cu, S1n, S1)
            conv_u = cu & (S1n < sp.LSR_ERROR)                                 # :955-958
            ICOUNT1 = jnp.where(conv_u, m, ICOUNT1)
            du = du & ~conv_u
            cv = dv & chk                                                      # :961
            UERR = ((vIce[iI, jI]-vTmp[iI, jI])                                # :967-968
                    * maskV[iI, jI])
            S2n = MAX_CHAIN(ZERO, jnp.abs(UERR), p="a", acc="b", ex=ex)       # :962-973 (MAX :969), :974
            WFAV_ = jnp.where(cv & (m > 1) & (S2n > S2A), WFAV2, WFAV_)        # :976
            S2A = jnp.where(cv, S2n, S2A)                                      # :977
            S2 = jnp.where(cv, S2n, S2)
            conv_v = cv & (S2n < sp.LSR_ERROR)                                 # :978-981
            ICOUNT2 = jnp.where(conv_v, m, ICOUNT2)
            dv = dv & ~conv_v
            uIce, vIce = EXCH_UV_XY_RL(uIce, vIce, True, ex=ex)               # :987
            return (m + 1, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2, WFAU_, WFAV_, S1A, S2A)
        if flex:
            st = (i32(1), s["UICE"], s["VICE"], c["doIterate4u"], c["doIterate4v"], c["ICOUNT1"], c["ICOUNT2"],
                  c["S1"], c["S2"])
            _, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2 = sweep_loop(cond, body, st)
        else:
            WFAU2 = ZERO                                                       # :311
            WFAV2 = ZERO                                                       # :312
            S1A0 = f64(0.80) * jnp.ones_like(c["S1"])                          # :315 S1A=0.80 _d 0
            S2A0 = f64(0.80) * jnp.ones_like(c["S2"])                          # :316
            st = (i32(1), s["UICE"], s["VICE"], c["doIterate4u"], c["doIterate4v"], c["ICOUNT1"], c["ICOUNT2"],
                  c["S1"], c["S2"], WFAU, WFAV, S1A0, S2A0)
            (_, uIce, vIce, du, dv, ICOUNT1, ICOUNT2, S1, S2, WFAU, WFAV, _, _) = sweep_loop(cond, body_maxnorm, st)
        residUend, residVend = res(uIce, vIce)                                 # :1007-1013 (printed only)
        uIce = uIce.at[iA, jA].set(uIce[iA, jA]*maskU[iA, jA])                 # :1065
        vIce = vIce.at[iA, jA].set(vIce[iA, jA]*maskV[iA, jA])                 # :1066
        c = with_sf(c, {"UICE": uIce, "VICE": vIce})
        k = ipass - 1
        o = dict(c["out"])
        o["printResid"] = o["printResid"].at[k].set(printResidual)
        o["printFrDrift"] = o["printFrDrift"].at[k].set(                     # :1019-1022 (lane M4CS32ICE)
            cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h") and sp.LSR_mixIniGuess >= 0)
        o["notConverged"] = o["notConverged"].at[k].set(du | dv)              # :1047-1057 (to errorMessageUnit)
        if tamc:                       # :1003-1058 #ifndef ALLOW_AUTODIFF_TAMC: neither the lines nor the WARNING
            o["printResid"] = o["printResid"].at[k].set(False)
        o["printWarn"] = o["printWarn"].at[k].set(not tamc)
        for n, v in (("residUend", residUend), ("residVend", residVend), ("S1", S1), ("S2", S2),
                     ("ICOUNT1", ICOUNT1), ("ICOUNT2", ICOUNT2), ("residUini", c["residUini"]),
                     ("residVini", c["residVini"]), ("residU_fd", c["residU_fd"]), ("residV_fd", c["residV_fd"]),
                     ("WFAU", WFAU), ("WFAV", WFAV)):
            o[n] = o[n].at[k].set(v)
        return {**c, "doIterate4u": du, "doIterate4v": dv, "ICOUNT1": ICOUNT1, "ICOUNT2": ICOUNT2, "S1": S1,
                "S2": S2, "out": o}

    def pass_body(ipass, c):
        if not flex:      # without the flexible criterion doNonLinLoop stays .TRUE. (:281): both blocks always run
            return sec_c(sec_a(c, ipass), ipass)
        c = lax.cond(c["doNonLinLoop"], lambda c_: sec_a(c_, ipass), lambda c_: c_, c)   # :300-721
        # :723-776 (SEAICEuseLSRflex)
        residUini, residVini = c["residUini"], c["residVini"]
        zu = residUini == 0.0                                                  # :725-729
        du = c["doIterate4u"] & ~zu
        ICOUNT1 = jnp.where(zu, 0, c["ICOUNT1"])
        S1 = jnp.where(zu, residUini, c["S1"])
        zv = residVini == 0.0                                                  # :730-734
        dv = c["doIterate4v"] & ~zv
        ICOUNT2 = jnp.where(zv, 0, c["ICOUNT2"])
        S2 = jnp.where(zv, residVini, c["S2"])
        residIni = jnp.sqrt(residUini*residUini+residVini*residVini)           # :736
        residIniNonLin = jnp.where(ipass == 1, residIni, c["residIniNonLin"])  # :740
        doNonLinLoop = c["doNonLinLoop"] & (du | dv)                           # :743
        doNonLinLoop = doNonLinLoop & ~(                                       # :745-747
            (ipass > 2)
            & (residIni < sp.SEAICEnonLinTol*residIniNonLin))
        residIni = jnp.where(residIni == 0.0, 1.0e-20, residIni)               # :750
        LSRflexFac = 1.0 / (1.0 + jnp.abs(glibc_log10(residIni)))             # :754
        LSRflexFac = MIN(LSRflexFac, 0.99, p="a")                              # :757
        LSR_ERRORfu = residUini*LSRflexFac                                     # :764
        LSR_ERRORfv = residVini*LSRflexFac                                     # :765
        k = ipass - 1
        o = dict(c["out"])
        o["printFlex"] = o["printFlex"].at[k].set(printResidual)              # :767-774
        for n, v in (("LSRflexFac", LSRflexFac), ("LSR_ERRORfu", LSR_ERRORfu), ("LSR_ERRORfv", LSR_ERRORfv),
                     ("residIni", residIni)):
            o[n] = o[n].at[k].set(v)
        c = {**c, "doIterate4u": du, "doIterate4v": dv, "ICOUNT1": ICOUNT1, "ICOUNT2": ICOUNT2, "S1": S1,
             "S2": S2, "residIniNonLin": residIniNonLin, "doNonLinLoop": doNonLinLoop, "LSR_ERRORfu": LSR_ERRORfu,
             "LSR_ERRORfv": LSR_ERRORfv, "out": o}
        # :995-998 residkm2 = residkm1, residkm1 = residIni: read only by commented-out alternatives of
        # LSRflexFac (:752-756): not carried
        return lax.cond(c["doNonLinLoop"], lambda c_: sec_c(c_, ipass), lambda c_: c_, c)   # :779-1111

    c = lax.fori_loop(1, nonLinIterLoc + 1, pass_body, c0)                     # :289-1113
    return {**sf, **c["sf"]}, c["out"]


def _sel(flag, a, b):
    """`IF ( flag ) a ELSE b` of two FArrays declared alike (a traced scalar flag)."""
    return type(a)(jnp.where(flag, a.data, b.data), b.name, tiled=b.tiled, _dims=b.dims)


def _sel3(c1, a1, c2, a2, a3):
    """`IF ( c1 ) a1 ELSEIF ( c2 ) a2 ELSE a3` of FArrays declared alike (traced scalar conditions)."""
    return type(a1)(jnp.where(c1, a1.data, jnp.where(c2, a2.data, a3.data)), a3.name, tiled=a3.tiled,
                    _dims=a3.dims)


# ------------------------------------------------------------------------------------------------ host output
def lsr_stdout_lines(out):
    """The records SEAICE_LSR writes to standardMessageUnit in one call, in order, from its `out` (host numpy):
    per pass, the FLEX_FACTOR line (:767-773, with SEAICEuseLSRflex and printResidual) and, when the pass's second
    `IF ( doNonLinLoop )` block ran with printResidual, the residual lines (:1015-1035; the FrDrift line with
    LSR_mixIniGuess >= 0, the Mix-ini line with >= 2 is not reached: LSR_mixIniGuess = 1). Plain WRITEs: no
    PRINT_MESSAGE prefix."""
    from mitjax.io.fortran_format import fortran_write as w
    lines = []
    n = len(out["printFlex"])
    for k in range(n):
        ipass = k + 1
        if out["printFlex"][k]:
            lines.append(w("(A,1X,I5,1X,F10.6,3(1X,E12.6))",
                           " SEAICE_LSR: ipass, FLEX_FACTOR, LSR_ERRORfu, LSR_ERRORfv, residIni =", ipass,
                           float(out["LSRflexFac"][k]), float(out["LSR_ERRORfu"][k]), float(out["LSR_ERRORfv"][k]),
                           float(out["residIni"][k])))
        if out["printResid"][k]:
            lines.append(w("(A,1X,I9,1P3E16.8)", " SEAICE_LSR: Residual Initial ipass,Uice,Vice=", ipass,
                           float(out["residUini"][k]), float(out["residVini"][k])))
            if out.get("printFrDrift", [True]*n)[k]:   # :1019-1022 SEAICE_ALLOW_FREEDRIFT, LSR_mixIniGuess >= 0
                lines.append(w("(A,1P3E16.8)", " SEAICE_LSR: Residual FrDrift U_fd,V_fd=",
                               float(out["residU_fd"][k]), float(out["residV_fd"][k])))
            lines.append(w("(A,I9,A,I9,1P2E16.8)", " SEAICE_LSR (ipass=", ipass, ") iters,dU,Resid=",
                           int(out["ICOUNT1"][k]), float(out["S1"][k]), float(out["residUend"][k])))
            lines.append(w("(A,I9,A,I9,1P2E16.8)", " SEAICE_LSR (ipass=", ipass, ") iters,dV,Resid=",
                           int(out["ICOUNT2"][k]), float(out["S2"][k]), float(out["residVend"][k])))
    return lines


def lsr_stderr_lines(out, myIter):
    """The PRINT_MESSAGE records to errorMessageUnit of a pass that did not converge (:1047-1057), in order."""
    from mitjax.io.fortran_format import fortran_write as w
    lines = []
    for k in range(len(out["notConverged"])):
        if out.get("printWarn", [True]*len(out["notConverged"]))[k] and out["notConverged"][k]:
            lines.append(w("(2A,I10,A,I4,A)", "** WARNING ** SEAICE_LSR ", "(it=", int(myIter), ",", k + 1,
                           ") did not converge :"))
            lines.append(w("(2(A,I6,0PF6.3,1PE14.6))", " nIt,wFU,dU=", int(out["ICOUNT1"][k]),
                           float(out["WFAU"][k]), float(out["S1"][k]), " ; nIt,wFV,dV=", int(out["ICOUNT2"][k]),
                           float(out["WFAV"][k]), float(out["S2"][k])))
    return lines
