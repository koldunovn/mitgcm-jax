"""IMPLDIFF: model/src/impldiff.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import level, scan_k, set_levels

GAD_TR1 = 3                 # pkg/generic_advdiff/GAD.h:124  PARAMETER(GAD_TR1=3)


def impldiff(iMin, iMax, jMin, jMax, tracerId, KappaRX, recip_hFac, gTracer, *, cfg, grid, params, ptr=None):
    """IMPLDIFF( bi, bj, iMin, iMax, jMin, jMax, tracerId, KappaRX, recip_hFac, gTracer, myThid )
    @63cdc0b model/src/impldiff.F:4-395

    C     *==========================================================*
    C     | S/R IMPLDIFF
    C     | o Solve implicit diffusion equation for vertical
    C     |   diffusivity.
    C     *==========================================================*
    C     | o Recoded from 2d intermediate fields to 3d to reduce
    C     |   TAF storage
    C     | o Fixed missing masks for fields a(), c()
    C     *==========================================================*
    C     tracerId   :: tracer Identificator (if > 0) ; = -1 or -2 when
    C                   solving vertical viscosity implicitly for U or V
    C     KappaRk    :: vertical diffusion coefficient
    C     recip_hFac :: Inverse of cell open-depth factor
    C     gTracer    :: future tracer field

    KappaRX, recip_hFac, gTracer: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); tracerId a Python int. Returns gTracer
    (points outside iMin:iMax, jMin:jMax keep their values). Ported without TARGET_NEC_SX (loops over iMin..jMax);
    tracerId >= GAD_TR1 under ALLOW_PTRACERS takes PTRACERS_dTLev from `ptr` (PTRACERS_PARAMS.h; PTRACERS lane arm,
    :89-92). The loops building a, c, b and bet = 1, gam = 0
    have independent k iterations (each level reads only inputs) and are vectorised; the middle of the forward sweep
    (:209-224), the forward elimination (:238-250) and the backward sweep (:254-265) are recursions in k (scan_k, in
    the Fortran order). The ALLOW_DIAGNOSTICS block (:282-392) only fills diagnostics (pkg/diagnostics is not
    ported); it changes no output of this routine.
    """
    if cfg.cpp.TARGET_NEC_SX:
        raise NotImplementedError("IMPLDIFF: TARGET_NEC_SX loop ranges are not ported")
    sz = cfg.size
    Nr = sz.Nr
    # :88-104 time step of each level
    deltaTX = FArray(jnp.zeros(Nr, jnp.float64), "deltaTX", k=(1, Nr), tiled=False)
    if cfg.cpp.ALLOW_PTRACERS and tracerId >= GAD_TR1:                 # :89-92 (PTRACERS lane)
        for k in range(1, Nr+1):
            deltaTX = deltaTX.at[k].set(ptr.PTRACERS_dTLev[k])
    elif tracerId >= 1:                                                 # :93/:95-99
        for k in range(1, Nr+1):
            deltaTX = deltaTX.at[k].set(params.dTtracerLev[k])
    else:                                                               # :100-103
        for k in range(1, Nr+1):
            deltaTX = deltaTX.at[k].set(params.deltaTMom)

    fj = (1-sz.OLy, sz.sNy+sz.OLy)
    fi = (1-sz.OLx, sz.sNx+sz.OLx)
    # :107-113  locTr = 0
    k, j, i = loops_kji((1, Nr), fj, fi)
    locTr = gTracer.local("locTr").at[i, j, k].set(0.0)
    a = gTracer.local("a")
    b = gTracer.local("b")
    c = gTracer.local("c")
    bet = gTracer.local("bet")
    gam = gTracer.local("gam")
    j, i = loop_j(*fj), loop_i(*fi)
    a = a.at[i, j, 1].set(0.0)                                          # :116-120

    # :121-136  old aLower
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    a = a.at[i, j, k].set(jnp.where(
        recip_hFac[i, j, k-1] == 0.0, 0.0,                              # :133  IF (recip_hFac(i,j,k-1).EQ.0.) a=0.
        -deltaTX[k]*recip_hFac[i, j, k]*grid.recip_drF[k]
        * grid.recip_deepFac2C[k]*params.recip_rhoFacC[k]
        * KappaRX[i, j, k]*grid.recip_drC[k]
        * grid.deepFac2F[k]*params.rhoFacF[k]))
    # :139-154  old aUpper
    k, j, i = loops_kji((1, Nr-1), (jMin, jMax), (iMin, iMax))
    c = c.at[i, j, k].set(jnp.where(
        recip_hFac[i, j, k+1] == 0.0, 0.0,                              # :151  IF (recip_hFac(i,j,k+1).EQ.0.) c=0.
        -deltaTX[k]*recip_hFac[i, j, k]*grid.recip_drF[k]
        * grid.recip_deepFac2C[k]*params.recip_rhoFacC[k]
        * KappaRX[i, j, k+1]*grid.recip_drC[k+1]
        * grid.deepFac2F[k+1]*params.rhoFacF[k+1]))
    j, i = loop_j(*fj), loop_i(*fi)
    c = c.at[i, j, Nr].set(0.0)                                         # :155-159
    # :162-175  old aCenter
    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    b = b.at[i, j, k].set(1.0 - (a[i, j, k] + c[i, j, k]))
    # :178-185  old and new gam, bet are the same
    k, j, i = loops_kji((1, Nr), fj, fi)
    bet = bet.at[i, j, k].set(1.0)
    gam = gam.at[i, j, k].set(0.0)

    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)
    if Nr > 1:                                                          # :188-203
        nz = b[i, j, 1] != 0.0                                          # :199  IF (b(i,j,1).NE.0.) bet = 1/b
        bet = bet.at[i, j, 1].set(jnp.where(nz, safe_div(1.0, b[i, j, 1], nz), bet[i, j, 1]))

    if Nr >= 2:                                                         # :206-226  middle of forward sweep
        def middle(bet_km1, x):
            gam_k = x["gam"].at[i, j].set(x["c_km1"][i, j]*bet_km1[i, j])           # :218
            den = x["b"][i, j] - x["a"][i, j]*gam_k[i, j]               # :219-220
            nzk = den != 0.0
            bet_k = x["bet"].at[i, j].set(jnp.where(nzk, safe_div(1.0, den, nzk), x["bet"][i, j]))
            return bet_k, {"gam": gam_k, "bet": bet_k}

        _, mid = scan_k(middle, level(bet, 1), range(2, Nr+1),
                        lambda k: {"c_km1": level(c, k-1), "a": level(a, k), "b": level(b, k),
                                   "gam": level(gam, k), "bet": level(bet, k)})
        gam = set_levels(gam, mid["gam"], range(2, Nr+1))
        bet = set_levels(bet, mid["bet"], range(2, Nr+1))

    # :228-237  k = 1
    locTr = locTr.at[i, j, 1].set(gTracer[i, j, 1]*bet[i, j, 1])

    # :238-250  forward elimination
    def elim(locTr_km1, x):
        locTr_k = x["locTr"].at[i, j].set(x["bet"][i, j]
                                          * (x["g"][i, j] - x["a"][i, j]*locTr_km1[i, j]))
        return locTr_k, locTr_k

    if Nr >= 2:
        _, fw = scan_k(elim, level(locTr, 1), range(2, Nr+1),
                       lambda k: {"locTr": level(locTr, k), "bet": level(bet, k), "g": level(gTracer, k),
                                  "a": level(a, k)})
        locTr = set_levels(locTr, fw, range(2, Nr+1))

    # :252-265  backward sweep
    def back(locTr_kp1, x):
        locTr_k = x["locTr"].at[i, j].set(x["locTr"][i, j] - x["gam_kp1"][i, j]*locTr_kp1[i, j])
        return locTr_k, locTr_k

    if Nr >= 2:
        _, bw = scan_k(back, level(locTr, Nr), range(Nr-1, 0, -1),
                       lambda k: {"locTr": level(locTr, k), "gam_kp1": level(gam, k+1)})
        locTr = set_levels(locTr, bw, range(1, Nr))

    # :267-280  gTracer = locTr (locTr = locUpdate is only read by the diagnostics)
    k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))
    gTracer = gTracer.at[i, j, k].set(locTr[i, j, k])
    return gTracer
