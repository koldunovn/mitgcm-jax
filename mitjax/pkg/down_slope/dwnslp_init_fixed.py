"""DWNSLP_INIT_FIXED: pkg/down_slope/dwnslp_init_fixed.F @63cdc0b (lane M4ADLAB session 3)."""

import numpy as np

from mitjax.ops.fortran_minmax_host import MIN


def _fmod(a, b):
    """Fortran MOD of INTEGERs (the sign of a)."""
    r = abs(a) % abs(b)
    return r if a >= 0 else -r


def _fdiv(a, b):
    """Fortran INTEGER division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def dwnslp_init_fixed(dp, *, cfg, grid, usingPCoords):
    """DWNSLP_INIT_FIXED( myThid )   @63cdc0b pkg/down_slope/dwnslp_init_fixed.F:6-345

    C     | SUBROUTINE DWNSLP_INIT_FIXED
    C     | o Routine to initialize Down-Sloping arrays

    Host routine (numpy; the grid is fixed). `dp` DwnslpParams, `grid` the GRID.h fields (FArrays, [tile, (k,) j,
    i] with halos). Returns dict(NbSite [T], ijDeep [T, NS], shVsD [T, NS], Gamma [T, NS], kshelf [T, NS],
    kdeepMax [T, NS], log, stdout) with NS = the largest DWNSLP_NbSite (entries n > NbSite of a tile are zero and
    never read): DWNSLP_ijDeep (1-based flat index of the deep column in the tile with overlap, `1 + (i+OLx-1) +
    (j+OLy-1)*xSize`), DWNSLP_shVsD, DWNSLP_Gamma of :56-63, :69-192 (the z-coordinate block :129-187; the
    p-coordinate block :71-128 raises), :207-272; `kshelf` = kBottom of the shelf column and `kdeepMax` = kBottom of
    the deep column (kLowC, the values DWNSLP_CALC_FLOW / DWNSLP_APPLY read). DWNSLP_size (DWNSLP_SIZE.h:14 =
    xySize) bounds NbSite (:195-204 STOP). `log`: the lines of down_slope.0000.log (:282-332, written when
    debugLevel >= debLevA and closed below debLevD), `stdout`: the DWNSLP_INIT: DWNSLP_NbSite= messages (:294-297).
    Loops in the Fortran order (j outer, i inner; X then Y); MIN winners :231-232, :239-244, :259-266 "b" (site
    table)."""
    if usingPCoords:
        raise NotImplementedError("DWNSLP_INIT_FIXED: p coordinates (gravitySign > 0, :86-146) are not ported")
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    xSize = OLx + sNx + OLx                                                    # DWNSLP_SIZE.h:7
    DWNSLP_size = xSize * (OLy + sNy + OLy)                                    # DWNSLP_SIZE.h:14
    g = {n: np.asarray(getattr(grid, n).data) for n in ("kSurfW", "kSurfS", "kLowC", "hFacC", "hFacW", "hFacS",
                                                         "dyG", "dxG", "recip_dxC", "recip_dyC", "R_low")}
    drF = np.asarray(grid.drF.data, np.float64)
    T = g["kLowC"].shape[0]

    def at(name, t, i, j, k=None):                                             # Fortran (i,j[,k]) of tile t
        a = g[name]
        return a[t, j - 1 + OLy, i - 1 + OLx] if k is None else a[t, k - 1, j - 1 + OLy, i - 1 + OLx]

    sites = []
    for t in range(T):
        ijDeep, shVsD = [], []
        ncount = 0                                                             # :69
        for j in range(1, sNy + 1):                                            # :132-157  X direction (U-flow)
            for i in range(1, sNx + 2):
                if at("kSurfW", t, i, j) <= Nr:
                    if at("kLowC", t, i, j) > at("kLowC", t, i-1, j):
                        ncount += 1
                        ijDeep.append(1 + (i+OLx-1) + (j+OLy-1)*xSize)
                        shVsD.append(-1)
                    if at("kLowC", t, i, j) < at("kLowC", t, i-1, j):
                        ncount += 1
                        ijDeep.append(1 + (i-1+OLx-1) + (j+OLy-1)*xSize)
                        shVsD.append(1)
        for j in range(1, sNy + 2):                                            # :160-186  Y direction (V-flow)
            for i in range(1, sNx + 1):
                if at("kSurfS", t, i, j) <= Nr:
                    if at("kLowC", t, i, j) > at("kLowC", t, i, j-1):
                        ncount += 1
                        ijDeep.append(1 + (i+OLx-1) + (j+OLy-1)*xSize)
                        shVsD.append(-xSize)
                    if at("kLowC", t, i, j) < at("kLowC", t, i, j-1):
                        ncount += 1
                        ijDeep.append(1 + (i+OLx-1) + (j-1+OLy-1)*xSize)
                        shVsD.append(xSize)
        if ncount > DWNSLP_size:                                               # :195-204
            raise RuntimeError(f" DWNSLP_INIT: DWNSLP_size={DWNSLP_size:8d} too small !\n"
                               "ABNORMAL END: S/R DWNSLP_INIT")
        Gamma, kshelfs, kdeeps, cols = [], [], [], []
        for n in range(ncount):                                                # :207-272
            ijd = ijDeep[n]
            ideep = 1-OLx + _fmod(ijd-1, xSize)
            jdeep = 1-OLy + _fdiv(ijd-1, xSize)
            ijr = shVsD[n]
            ishelf = ideep + _fmod(ijr, xSize)
            jshelf = jdeep + _fdiv(ijr, xSize)
            kdeep = int(at("kLowC", t, ideep, jdeep))                          # :219-222 (z coordinates)
            kshelf = int(at("kLowC", t, ishelf, jshelf))
            downward = 1
            i = max(ideep, ishelf)                                             # :225  MINMAX-INT: INTEGER MAX
            j = max(jdeep, jshelf)                                             # :226  MINMAX-INT: INTEGER MAX
            drFlowMin = np.float64(dp.DWNSLP_drFlow)                           # :229
            for k in range(kshelf, kdeep + downward, downward):                # :230-233
                drFlowMin = MIN(drFlowMin,                                     # :231-232
                                drF[k-1]*at("hFacC", t, ideep, jdeep, k), p="b")
            if dp.DWNSLP_slope != 0.:                                          # :235-245
                if abs(ijr) == 1:
                    gam = (np.float64(dp.DWNSLP_slope)*at("dyG", t, i, j)       # :239-240
                           * MIN(drF[kshelf-1]*at("hFacW", t, i, j, kshelf), drFlowMin, p="b"))
                else:
                    gam = (np.float64(dp.DWNSLP_slope)*at("dxG", t, i, j)       # :243-244
                           * MIN(drF[kshelf-1]*at("hFacS", t, i, j, kshelf), drFlowMin, p="b"))
            else:                                                              # :246-269 local slope
                dz_bottom = at("R_low", t, ishelf, jshelf) - at("R_low", t, ideep, jdeep)   # :254-255
                if abs(ijr) == 1:
                    gam = (dz_bottom*at("recip_dxC", t, i, j)                  # :259-261
                           * at("dyG", t, i, j)
                           * MIN(drF[kshelf-1]*at("hFacW", t, i, j, kshelf), drFlowMin, p="b"))
                else:
                    gam = (dz_bottom*at("recip_dyC", t, i, j)                  # :264-266
                           * at("dxG", t, i, j)
                           * MIN(drF[kshelf-1]*at("hFacS", t, i, j, kshelf), drFlowMin, p="b"))
            Gamma.append(np.float64(gam))
            kshelfs.append(kshelf)
            kdeeps.append(kdeep)
            cols.append((ideep, jdeep, kshelf, kdeep - kshelf))
        sites.append((ijDeep, shVsD, Gamma, kshelfs, kdeeps, cols))
    NS = max(1, max(len(s[0]) for s in sites))          # MINMAX-INT: host array size (integer)
    # padding entries n > NbSite (never active): a valid column (ijDeep 1, shVsD 0) and level 1, so the vectorised
    # kernels gather finite values there
    out = {n: np.full((T, NS), 1 if n != "shVsD" else 0, np.int32) for n in ("ijDeep", "shVsD", "kshelf",
                                                                            "kdeepMax")}
    out["Gamma"] = np.zeros((T, NS), np.float64)
    out["NbSite"] = np.array([len(s[0]) for s in sites], np.int32)
    for t, (ijDeep, shVsD, Gamma, kshelfs, kdeeps, _) in enumerate(sites):
        m = len(ijDeep)
        out["ijDeep"][t, :m], out["shVsD"][t, :m], out["Gamma"][t, :m] = ijDeep, shVsD, Gamma
        out["kshelf"][t, :m], out["kdeepMax"][t, :m] = kshelfs, kdeeps
    # :282-332 the log and the STDOUT lines (tile order bj outer, bi inner; one process: procId 0)
    from mitjax.model.src.cg2d import fortran_1pe
    log, stdout = [], []
    for t in range(T):
        bi, bj = t % sz.nSx + 1, t // sz.nSx + 1
        stdout.append(f"DWNSLP_INIT: DWNSLP_NbSite={bi:4d}{bj:4d}{int(out['NbSite'][t]):8d}")     # :294-297
        if dp.write_init_log:
            ijDeep, shVsD, Gamma, _, _, cols = sites[t]
            log.append(f" DWNSLP_INIT: bi,bj, DWNSLP_NbSite, xSize ={bi:4d}{bj:4d}"
                       f"{len(ijDeep):8d}{xSize:8d}")                           # :300-302
            log.append("  bi  bj     n :     ijd  is  js ,   ijr  ks dkMx  Gamma :")   # :303-304
            for n, (ideep, jdeep, kshelf, dkMx) in enumerate(cols):            # :305-322
                log.append(f"{bi:4d}{bj:4d}{n+1:6d} :{ijDeep[n]:8d}{ideep:4d}{jdeep:4d} ,{shVsD[n]:6d}"
                           f"{kshelf:4d}{dkMx:4d}{fortran_1pe(float(Gamma[n]), 14, 6)}")
            log.append("")                                                     # :323 WRITE(DWNSLP_ioUnit,*)
    out["log"] = log
    out["stdout"] = stdout
    return out
