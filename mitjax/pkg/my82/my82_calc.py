"""MY82_CALC: pkg/my82/my82_calc.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.ops.scan_k import level, scan_k, scan_levels
from mitjax.pkg.my82.my82_h import B1
from mitjax.pkg.my82.my82_ri_number import my82_ri_number


def my82_calc(sigmaR, myTime, myIter, *, cfg, grid, params, eos, state, my):
    """MY82_CALC( bi, bj, sigmaR, myTime, myIter, myThid )   @63cdc0b pkg/my82/my82_calc.F:7-190

    C     | SUBROUTINE MY82_CALC                                     |
    C     | o Compute all MY82 fields defined in MY82.h              |
    C     | This subroutine is based on SPEM code                    |
    C global parameters updated by pp_calc
    C     MYviscAz  :: MY82 eddy viscosity coefficient              (m^2/s)
    C     MYdiffKzT :: MY82 diffusion coefficient for temperature   (m^2/s)
    C     RiNumber - Richardson Number = -GH/GM
    C     GH       - buoyancy freqency after call to Ri_number, later
    C                GH of M. Satoh, Eq. (11.3.45)
    C     GM       - vertical shear of velocity after call Ri_number,
    C                later GM of M. Satoh, Eq. (11.3.44)

    Returns `my` (MY82.h) with MYhbl, MYviscAr, MYdiffKr written on iMin:iMax = 2-OLx:sNx+OLx-1, jMin:jMax =
    2-OLy:sNy+OLy-1 (:69-72; MYviscAr/MYdiffKr on every level); other points keep their values. sigmaR is not read.
    `state`: DYNVARS.h (theta, salt, uVel, vVel, totPhiHyd) for MY82_RI_NUMBER.

    Loops: the first k loop (:92-120) calls MY82_RI_NUMBER once per level and each level writes and then reads only
    its own GH, GM, RiNumber points: a per-level body `level_k(K, c)` run as a level scan in the Fortran order
    (KERNEL_GUIDE §4; K = 2 static). The second k loop (:134-142) accumulates GM, GH from the surface
    down: a recursion, `scan_k`. The third (:154-179) has independent levels: vectorised over k. Branches: #else of
    ALLOW_3D_DIFFKR (diffKrNrS); ALLOW_3D_DIFFKR raises; diagnostics fills (:181-187) are output only.

    Guards (values on the Fortran's lanes unchanged): `-GM/GH*MYhblScale` only where GH .NE. 0 (:146-150); the SQRT
    of tkesquare (>= 0 after the MAX; its derivative is infinite at 0) masked at tkesquare = +-0 with the value itself
    (SQRT(+-0) = +-0). The SQRT of :103 has a positive argument for every RiTmp (the quadratic b4^2 R^2 + (2 b1 b4 -
    4 b2 b3) R + b1^2 of the V4r4 constants has no real root). `b1` at :111 is the PARAMETER B1 of MY82.h (Fortran
    is case-insensitive).
    """
    sz = cfg.size
    Nr = sz.Nr
    if cfg.cpp.flag("ALLOW_3D_DIFFKR", "MY82_OPTIONS.h"):
        raise NotImplementedError("MY82_CALC: ALLOW_3D_DIFFKR is not ported")
    iMin = 2-sz.OLx                                                     # :69-72
    iMax = sz.sNx+sz.OLx-1
    jMin = 2-sz.OLy
    jMax = sz.sNy+sz.OLy-1
    T = sz.nSx*sz.nSy
    ij = dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    shp2 = (T, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)
    shp3 = (T, Nr) + shp2[1:]
    RiNumber = FArray(jnp.full(shp2, jnp.nan), "RiNumber", **ij)        # local, never initialised
    GH = FArray(jnp.zeros(shp2), "GH", **ij)                            # :75-80
    GM = FArray(jnp.zeros(shp2), "GM", **ij)
    SH = FArray(jnp.zeros(shp3), "SH", **ij, k=(1, Nr))                 # :81-89
    SM = FArray(jnp.zeros(shp3), "SM", **ij, k=(1, Nr))
    tke = FArray(jnp.zeros(shp3), "tke", **ij, k=(1, Nr))
    alpha1, alpha2 = my.alpha1, my.alpha2
    beta1, beta2, beta3, beta4 = my.beta1, my.beta2, my.beta3, my.beta4

    def level_k(K, c):                                                  # first k-loop, DO K = 2, Nr (:92-120)
        RiNumber, GH, GM, SH, SM, tke = c
        RiNumber, GH, GM = my82_ri_number(K, iMin, iMax, jMin, jMax, RiNumber, GH, GM, myTime,    # :93-96
                                          cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        j = loop_j(jMin, jMax)                                          # :97
        i = loop_i(iMin, iMax)                                          # :98
        RiTmp = MIN(RiNumber[i, j], my.RiMax, p="a")                    # :99
        btmp = beta1+beta4*RiTmp                                        # :100
        RiFlux = ((btmp - jnp.sqrt(btmp*btmp - 4.0*beta2*beta3*RiTmp))  # :103-104
                  / (2.0*beta2))
        SHtmp = (alpha1-alpha2*RiFlux)/(1.0-RiFlux)                     # :106
        SH = SH.at[i, j, K].set(SHtmp)                                  # :107
        SM = SM.at[i, j, K].set(SHtmp*(beta1-beta2*RiFlux)/(beta3-beta4*RiFlux))   # :108
        tkesquare = MAX(0.0,                                            # :110-111
                        B1*(SH[i, j, K]*GH[i, j] + SM[i, j, K]*GM[i, j]), p="b")
        tke = tke.at[i, j, K].set(safe_sqrt(tkesquare, tkesquare > 0.0, fill=tkesquare))   # :112
        return RiNumber, GH, GM, SH, SM, tke

    c = scan_levels(level_k, (RiNumber, GH, GM, SH, SM, tke), 2, Nr,    # K = 2 static (MY82_RI_NUMBER's
                    peel=(1, 0))                                        # Km1 = MAX(1,K-1))
    RiNumber, GH, GM, SH, SM, tke = c

    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    GH = GH.at[i, j].set(0.0)                                           # :124-129
    GM = GM.at[i, j].set(0.0)

    def accumulate(carry, x):                                           # second k-loop, DO K = 2, Nr (:134-142)
        GM_, GH_ = carry
        GM_ = GM_.at[i, j].set(GM_[i, j] + x["tke"][i, j]*x["rF"])      # :137
        GH_ = GH_.at[i, j].set(GH_[i, j] + x["tke"][i, j])              # :138
        return (GM_, GH_), None

    (GM, GH), _ = scan_k(accumulate, (GM, GH), range(2, Nr+1),
                         lambda K: {"tke": level(tke, K), "rF": grid.rF[K]})

    nz = GH[i, j] != 0.0                                                # :146-150
    MYhbl = my.MYhbl.at[i, j].set(jnp.where(
        nz, -(safe_div(GM[i, j], GH[i, j], nz)*my.MYhblScale), 0.0))

    k, j3, i3 = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))          # third k-loop (:154-179)
    hbl = MYhbl[i3, j3]
    tkel = hbl*tke[i3, j3, k]                                           # :158
    vis = hbl*tkel*SM[i3, j3, k]                                        # :160
    dif = hbl*tkel*SH[i3, j3, k]                                        # :161
    vis = MAX(vis, params.viscArNr[k], p="b")                           # :163-164
    dif = MAX(dif, params.diffKrNrS[k], p="b")                          # :165-169
    vis = MIN(vis, my.MYviscMax, p="b")*grid.maskC[i3, j3, k]           # :172-173
    dif = MIN(dif, my.MYdiffMax, p="b")*grid.maskC[i3, j3, k]           # :174-175
    return my.replace(MYhbl=MYhbl, MYviscAr=my.MYviscAr.at[i3, j3, k].set(vis),
                      MYdiffKr=my.MYdiffKr.at[i3, j3, k].set(dif))
