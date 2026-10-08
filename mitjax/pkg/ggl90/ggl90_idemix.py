"""GGL90_IDEMIX, IDEMIX_gofx2, IDEMIX_hofx1: pkg/ggl90/ggl90_idemix.F @63cdc0b.

ASIN is glibc's asin (mitjax/ops/libm.glibc_asin, transcribed from the oracle's libm: bit for bit for |x| < 2^-3,
the arguments of wet points, where fxa >= 10; dry and masked points take 1/3 and 1/1.01, where the jnp.arcsin
FALLBACK of glibc_asin is measured equal; mitjax/tests/test_glibc_asin.py). XLA's arcsin is not glibc's
(MEASURED["asin"]): with it ~1500 IDEMIX_E / TKE points per step differ in the dump gate.
"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.model.grid import PI
from mitjax.model.src.solve_tridiagonal import solve_tridiagonal
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_asin, glibc_exp
from mitjax.ops.safe import safe_sqrt
from mitjax.ops.scan_k import level, scan_k
from mitjax.pkg.ggl90.ggl90_h import GGL90eps

ASIN = glibc_asin      # the oracle's libm asin (module docstring)


def idemix_gofx2(xx, toPI):
    """_RL FUNCTION IDEMIX_gofx2(xx,toPI)   @63cdc0b pkg/ggl90/ggl90_idemix.F:581-589

    toPI :: 2.d0/PI. Elementwise on arrays. x**(-2.d0/3.d0) is libm's pow (XLA's power, bitwise:
    MEASURED["pow"]), EXP is glibc_exp, ASIN glibc_asin (module docstring)."""
    x = MAX(3.0, xx, p="b")                                             # :586  MAX(3.d0,xx)
    c = 1.0-toPI*ASIN(1.0/x)                                            # :587
    return toPI/c*0.9*jnp.power(x, -2.0/3.0)*(1.-glibc_exp(-x/4.3))    # :588


def idemix_hofx1(x, toPI):
    """_RL FUNCTION IDEMIX_hofx1(x,toPI)   @63cdc0b pkg/ggl90/ggl90_idemix.F:591-597

    toPI :: 2.d0/PI. Elementwise on arrays."""
    return (toPI/(1.0-toPI*ASIN(1.0/MAX(1.01, x, p="a")))             # :595-596
            * (x-1.0)/(x+1.0))


def ggl90_idemix(hFacI, recip_hFacI, sigmaR, gTKE, myTime, myIter, *, cfg, grid, params, ggl, gm=None):
    """GGL90_IDEMIX( bi, bj, hFacI, recip_hFacI, sigmaR, gTKE, myTime, myIter, myThid )
    @63cdc0b pkg/ggl90/ggl90_idemix.F:18-576

    C     | S/R GGL90_IDEMIX
    C     | IDEMIX1 model as described in
    C     | - Olbers, D. and Eden, C. (2013), JPO, doi:10.1175/JPO-D-12-0207.1
    C     | in a nutshell:
    C     | computes contribution of internal wave field to vertical mixing
    C     hFacI  :: thickness factors for w-cells (interface)
    C               with reciprocal of hFacI = recip_hFacI
    C     sigmaR :: Vertical gradient of iso-neutral density
    C     gTKE   :: dissipation of IW energy (output of S/R GGL90_IDEMIX)

    hFacI, recip_hFacI, sigmaR, gTKE: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). Returns (gTKE, ggl) with IDEMIX_E
    updated in `ggl` (GGL90.h).

    Ported: z coordinates (kSrf = 1, kTop = 2, coordFac = 1), without GGL90_IDEMIX_CVMIX_VERSION (:166-172,
    :194-204, :212-217), IDEMIX_tau_h > 0 and = 0 (:221-235, a host decision on the namelist value), the horizontal
    diffusion of E (:310-357), the vertical solve (:375-524) and gTKE (:527-536). GM_EG_PROGNOSTIC is #undef'd by
    the file itself (:5). Not ported (raise): pressure coordinates, GGL90_IDEMIX_CVMIX_VERSION, IDEMIX_include_GM and
    IDEMIX_include_GM_bottom with useGMRedi (:266-304). Not computed: gm_forc (:241-264), which feeds only
    IDEMIX_include_GM_bottom (raises) and the diagnostic IDEM_F_g; the ALLOW_DIAGNOSTICS block (:538-572) fills
    diagnostics only.

    Loops: every k loop has independent iterations (each level reads inputs of levels k, k-1, k+1 and writes level k
    of its own output) and is vectorised, except the running sum bN0 (:179-186: bN0 accumulates level by level, the
    Fortran's summation order: scan_k). The per-level 2-D fluxes dfx, dfy (:310-357) are 3-D temporaries. kBot =
    MAX(kLowC,1) (:394, :442, :499) is a masked write / gather along k.

    Differentiability: SQRT(Nsquare) (:183, :202) is guarded on Nsquare > 0 (Nsquare = 0 on masked levels; value
    Fortran's: SQRT(+0) = +0) and SQRT(fxc*fxc-1.) (:211) on its argument > 0 (= +0 at fxc = 1).
    """
    sz = cfg.size
    Nr = sz.Nr
    opt = "GGL90_OPTIONS.h"
    if params.usingPCoords:                                             # :113-115
        raise NotImplementedError("GGL90_IDEMIX: pressure coordinates are not ported")
    if cfg.cpp.flag("GGL90_IDEMIX_CVMIX_VERSION", opt):
        raise NotImplementedError("GGL90_IDEMIX: GGL90_IDEMIX_CVMIX_VERSION is not ported")
    useGmredi = cfg.cpp.flag("ALLOW_GMREDI", opt) and cfg.use_flag("useGMRedi")
    if useGmredi and (ggl.IDEMIX_include_GM or ggl.IDEMIX_include_GM_bottom):          # :266, :290
        raise NotImplementedError("GGL90_IDEMIX: IDEMIX_include_GM / IDEMIX_include_GM_bottom are not ported")
    iMin, iMax = 2-sz.OLx, sz.sNx+sz.OLx-1                             # :108-109
    jMin, jMax = 2-sz.OLy, sz.sNy+sz.OLy-1                             # :110-111
    fj, fi = (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx)
    kSrf, kTop = 1, 2                                                   # :117-118
    deltaTloc = params.dTtracerLev[kSrf]                                # :120
    coordFac = 1.0                                                      # :122  1. _d 0
    recip_coordFac = 1./coordFac                                        # :124
    twoOverPi = 2.0/PI                                                  # :126  2. _d 0/PI
    pijstar = PI*ggl.IDEMIX_jstar                                       # :127
    recip_pijstar = 1.0/pijstar                                         # :128  1. _d 0 / pijstar
    maskC = grid.maskC
    E = ggl.IDEMIX_E

    # :131-157  initialize local fields
    k, j, i = loops_kji((1, Nr), fj, fi)
    gTKE = gTKE.at[i, j, k].set(0.0)
    Nsquare = E.local("Nsquare").at[i, j, k].set(0.0)
    delta = E.local("delta").at[i, j, k].set(0.0)
    a3d = E.local("a3d").at[i, j, k].set(0.0)
    b3d = E.local("b3d").at[i, j, k].set(1.0)
    c3d = E.local("c3d").at[i, j, k].set(0.0)
    c0 = E.local("c0").at[i, j, k].set(0.0)
    v0 = E.local("v0").at[i, j, k].set(0.0)
    tau_d = E.local("tau_d").at[i, j, k].set(0.0)
    forc = E.local("forc").at[i, j, k].set(0.0)
    j, i = loop_j(*fj), loop_i(*fi)
    bN0 = grid.Ro_surf.local("bN0").at[i, j].set(0.0)                  # :154

    # :161-175  allow for IW everywhere by limiting buoyancy freq.
    k, j, i = loops_kji((2, Nr), fj, fi)
    N2 = (params.gravity*grid.gravitySign*params.recip_rhoConst         # :164-165
          * sigmaR[i, j, k] * coordFac)
    fxb = MAX(1.e-6, jnp.abs(grid.fCori[i, j]), p="a")                  # :169  MAX( 1. _d -6, ABS( fCori ))
    Nsquare = Nsquare.at[i, j, k].set(MAX(100.*fxb*fxb, N2, p="a")      # :170-171
                                      * maskC[i, j, k]*maskC[i, j, k-1])

    # :179-186  vertically integrated N (a running sum in k)
    j, i = loop_j(*fj), loop_i(*fi)

    def integrate(b, x):
        n2 = x["N2"][i, j]
        return b.at[i, j].set(b[i, j]                                   # :182-183
                              + safe_sqrt(n2, n2 > 0.0)*x["drC"]*recip_coordFac*x["hFacI"][i, j]), None

    bN0, _ = scan_k(integrate, bN0, range(2, Nr+1),
                    lambda k: {"N2": level(Nsquare, k), "drC": grid.drC[k], "hFacI": level(hFacI, k)})

    # :191-220  vertical and horizontal group velocities and constant for dissipation
    k, j, i = loops_kji((2, Nr), fj, fi)
    n2 = Nsquare[i, j, k]
    fxb = MAX(1.e-6, jnp.abs(grid.fCori[i, j]), p="a")                  # :201
    fxa = safe_sqrt(n2, n2 > 0.0)/fxb                                   # :202
    cstar = bN0[i, j]*recip_pijstar                                     # :203
    c0 = c0.at[i, j, k].set(MAX(0.0,                                    # :205-206
                                cstar*ggl.IDEMIX_gamma*idemix_gofx2(fxa, twoOverPi), p="a"))
    v0 = v0.at[i, j, k].set(MAX(0.0,                                    # :207-208
                                cstar*ggl.IDEMIX_gamma*idemix_hofx1(fxa, twoOverPi), p="a"))
    fxc = MAX(1.0, fxa, p="a")                                          # :210
    arg = fxc*fxc - 1.                                                  # :211  `1.` REAL*4, exact
    fxc = jnp.log(fxc + safe_sqrt(arg, arg > 0.0))                      # :211  LOG (XLA's is glibc's: MEASURED)
    tau_d = tau_d.at[i, j, k].set(ggl.IDEMIX_mu0*fxb*fxc                # :215-216
                                  * (pijstar/(GGL90eps+bN0[i, j]))**2)
    if ggl.static_float("IDEMIX_tau_h") > 0.:                           # :221-235
        fxa = jnp.sqrt(1.0/(deltaTloc * ggl.IDEMIX_tau_h))              # :229  SQRT( 1. _d 0/( ... ) )
        fxb = 0.5*MIN(grid.dxF[i, j], grid.dyF[i, j], p="a")*fxa        # :230
        v0 = v0.at[i, j, k].set(MIN(v0[i, j, k], fxb, p="b"))           # :231

    # :310-357  horizontal diffusion of IW energy (z coordinates: kl = k); dfx, dfy per level -> 3-D temporaries
    # dfx, dfy: zero at every point first (:152-153 and :314, :328-330 rewrite the edge row each level)
    k, j, i = loops_kji((2, Nr), fj, fi)
    dfx = E.local("dfx").at[i, j, k].set(0.0)
    dfy = E.local("dfy").at[i, j, k].set(0.0)
    k, j, i = loops_kji((2, Nr), fj, (1-sz.OLx+1, sz.sNx+sz.OLx))
    fxa = ggl.IDEMIX_tau_h*0.5*(                                        # :316-318
        v0[i-1, j, k]*maskC[i-1, j, k]
        + v0[i, j, k]*maskC[i, j, k])
    dfx = dfx.at[i, j, k].set(-fxa*grid.dyG[i, j]*grid.drC[k]           # :319-325
                              * (MIN(.5, grid.hFacW[i, j, k-1], p="a")
                                 + MIN(.5, grid.hFacW[i, j, k], p="a"))
                              * grid.recip_dxC[i, j]
                              * (v0[i, j, k]*E[i, j, k]
                                 - v0[i-1, j, k]*E[i-1, j, k])
                              * grid.maskW[i, j, k])
    k, j, i = loops_kji((2, Nr), (1-sz.OLy+1, sz.sNy+sz.OLy), fi)
    fxa = ggl.IDEMIX_tau_h*0.5*(                                        # :333-335
        v0[i, j, k]*maskC[i, j, k]
        + v0[i, j-1, k]*maskC[i, j-1, k])
    dfy = dfy.at[i, j, k].set(-fxa*grid.dxG[i, j]*grid.drC[k]           # :336-342
                              * (MIN(.5, grid.hFacS[i, j, k-1], p="a")
                                 + MIN(.5, grid.hFacS[i, j, k], p="a"))
                              * grid.recip_dyC[i, j]
                              * (v0[i, j, k]*E[i, j, k]
                                 - v0[i, j-1, k]*E[i, j-1, k])
                              * grid.maskS[i, j, k])
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    E = E.at[i, j, k].set(E[i, j, k]                                    # :350-354
                          + deltaTloc*(-grid.recip_drC[k]*grid.recip_rA[i, j]
                                       * recip_hFacI[i, j, k]
                                       * ((dfx[i+1, j, k]-dfx[i, j, k])+(dfy[i, j+1, k]-dfy[i, j, k])))
                          * maskC[i, j, k])
    # :361-368  add interior forcing
    E = E.at[i, j, k].set(E[i, j, k] + forc[i, j, k]*deltaTloc)

    # :375-398  delta_k = dt tau_v /drF_k (c_k+c_k+1)/2
    if Nr >= 3:
        k, j, i = loops_kji((2, Nr-1), (jMin, jMax), (iMin, iMax))
        delta = delta.at[i, j, k].set(deltaTloc*ggl.IDEMIX_tau_v        # :378-380
                                      * grid.recip_drF[k]*coordFac*grid.recip_hFacC[i, j, k]
                                      * .5*(c0[i, j, k]+c0[i, j, k+1]))
    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    kv = jnp.arange(1, Nr+1, dtype=jnp.int32)[None, :, None, None]
    kBot = jnp.maximum(grid.kLowC[i, j], 1)                            # MINMAX-RAW: integer level index (no tie/NaN)
    at_bot = kv == kBot
    delta = delta.at[i, j, k].set(jnp.where(at_bot, 0.0, delta[i, j, k]))      # :392-397

    # :402-424  lower and upper diagonals
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    a3d = a3d.at[i, j, k].set(-delta[i, j, k-1]                         # :406-408
                              * grid.recip_drC[k]*coordFac*recip_hFacI[i, j, k]
                              * maskC[i, j, k])
    c3d = c3d.at[i, j, k].set(-delta[i, j, k]                           # :419-421
                              * grid.recip_drC[k]*coordFac*recip_hFacI[i, j, k]
                              * maskC[i, j, k-1])
    # :438-448  z coordinates: c3d(kBot) = 0, a3d(kTop) = 0
    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    c3d = c3d.at[i, j, k].set(jnp.where(at_bot, 0.0, c3d[i, j, k]))   # :442-443
    jj, ii = loop_j(jMin, jMax), loop_i(iMin, iMax)
    a3d = a3d.at[ii, jj, kTop].set(0.0)                                 # :445
    a3d = a3d.at[ii, jj, 1].set(0.0)                                    # :453
    c3d = c3d.at[ii, jj, 1].set(0.0)                                    # :454
    b3d = b3d.at[ii, jj, 1].set(1.0)                                    # :456

    # :461-470  center diagonal
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    b3d = b3d.at[i, j, k].set(1.0+deltaTloc*tau_d[i, j, k]              # :464-467
                              * E[i, j, k]
                              * maskC[i, j, k]*maskC[i, j, k-1]
                              - (a3d[i, j, k] + c3d[i, j, k]) * c0[i, j, k])
    # :474-483  kp1 = MIN(k+1,Nr), km1 = MAX(k-1,2): static per level (storage gathers along k)
    pos_km1 = np.array([max(kk-1, 2) for kk in range(2, Nr+1)]) - 1    # MINMAX-INT: static level numbers
    pos_kp1 = np.array([min(kk+1, Nr) for kk in range(2, Nr+1)]) - 1   # MINMAX-INT: static level numbers
    jsl, isl = slice(jMin-fj[0], jMax-fj[0]+1), slice(iMin-fi[0], iMax-fi[0]+1)
    c0_km1 = c0.data[:, pos_km1][:, :, jsl, isl]                        # c0(i,j,km1)
    c0_kp1 = c0.data[:, pos_kp1][:, :, jsl, isl]                        # c0(i,j,kp1)
    a3d = a3d.at[i, j, k].set(a3d[i, j, k]*c0_km1)                      # :479
    c3d = c3d.at[i, j, k].set(c3d[i, j, k]*c0_kp1)                      # :480

    # :485-516  flux boundary conditions (z coordinates: kl = kTop)
    E = E.at[ii, jj, kTop].set(E[ii, jj, kTop]                          # :490-493
                               + deltaTloc*ggl.IDEMIX_F_S[ii, jj]
                               * grid.recip_drC[kTop]*coordFac*recip_hFacI[ii, jj, kTop]
                               * maskC[ii, jj, kTop])
    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    E = E.at[i, j, k].set(jnp.where(at_bot,                              # :499-503
                                    E[i, j, k]
                                    - deltaTloc*ggl.IDEMIX_F_B[i, j]
                                    * grid.recip_drC[k]*coordFac*recip_hFacI[i, j, k]
                                    * maskC[i, j, k],
                                    E[i, j, k]))

    E, _ = solve_tridiagonal(iMin, iMax, jMin, jMax, a3d, b3d, c3d, E, -1, cfg=cfg)    # :519-524

    # :527-536  TKE tendency due to dissipation of IW energy
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    gTKE = gTKE.at[i, j, k].set(tau_d[i, j, k]*E[i, j, k]*E[i, j, k])  # :530-531
    return gTKE, ggl.replace(IDEMIX_E=E)
