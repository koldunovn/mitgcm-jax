"""INI_PRESSURE: model/src/ini_pressure.F @63cdc0b."""

import numpy as np

from mitjax.farray import loops_kji


def ini_pressure(state, *, cfg, params, grid=None, dyn=None, eos=None):
    """INI_PRESSURE( myThid )   @63cdc0b model/src/ini_pressure.F:7-205

    C     | SUBROUTINE INI_PRESSURE
    C     | o Calculate pressure field from initial conditions

    :62-72 totPhiHyd = 0. _d 0 on every point. vermix lane (M3 Task 30): :74-185 (storePhiHyd4Phys, set by
    selectP_inEOS_Zc >= 2 for a pressure-dependent EOS, set_parms.F:304; #ifndef ALLOW_AUTODIFF): the iterative
    initial hydrostatic pressure, up to 15 passes of CALC_PHI_HYD over every level with myIter = -1 (its FIND_RHO_2D
    arm, :138-149, density from the current totPhiHyd), iMin..jMax the whole tile; the RMS difference of each pass is
    a host value (the set-up runs eagerly): rmsTile and sumTile are chains over DO k / DO j=1,sNy / DO i=1,sNx in
    that order (np.add.accumulate: left to right), rmspp and count chains over the tiles in tile order,
    _GLOBAL_SUM_RL of a scalar the identity for one process. `grid`, `dyn` (the time-step PARAMS.h of
    ini_parms_dyn), `eos` are needed there. A non-converging iteration STOPs (:187-198, raised). ALLOW_AUTODIFF builds
    raise (the iteration is not compiled there). The messages (:55-60, :174-202) are STDOUT only."""
    sz = cfg.size
    k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    state = state.replace(totPhiHyd=state.totPhiHyd.at[i3, j3, k3].set(0.))      # :67
    if not params.init.storePhiHyd4Phys:                                       # :74
        return state
    if cfg.cpp.ALLOW_AUTODIFF:
        raise NotImplementedError("INI_PRESSURE: storePhiHyd4Phys in an ALLOW_AUTODIFF build (not compiled there)")
    if grid is None or dyn is None or eos is None:
        raise ValueError("INI_PRESSURE: storePhiHyd4Phys needs grid, dyn (PARAMS.h) and eos")
    from mitjax.farray import loop_i, loop_j
    from mitjax.model.grid import declare
    from mitjax.model.src.calc_phi_hyd import calc_phi_hyd
    Nr = sz.Nr
    iMin, iMax, jMin, jMax = 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy  # :80-83
    jA, iA = loop_j(jMin, jMax), loop_i(iMin, iMax)
    z2 = declare("Bo_surf", sz).at[iA, jA].set(0.)
    jI, iI = loop_j(1, sz.sNy), loop_i(1, sz.sNx)
    rmspp = np.float64(1.0)                                                    # :84  1. _d 0
    rmsppold = np.float64(0.0)                                                 # :85  0. _d 0
    npiter = 0
    for npiter in range(1, 16):                                                # :87  DO npiter= 1, 15
        if not rmspp > 0.0:                                                    # :88  IF ( rmspp.GT.zeroRL )
            continue
        rmsppold = rmspp                                                       # :89
        rmspp = np.float64(0.0)                                                # :90
        count = np.float64(0.0)                                                # :91
        phiHydF = z2                                                           # :96-100  phiHydF = 0. _d 0
        phiHydC, dPhiHydX, dPhiHydY = z2.local("phiHydC"), z2.local("dPhiHydX"), z2.local("dPhiHydY")
        diffs, masks = [], []
        for k in range(1, Nr + 1):                                             # :101-123
            oldPhi = np.asarray(state.totPhiHyd[iI, jI, k])                    # :102-106 (read on 1..sNy, 1..sNx)
            state, phiHydF, phiHydC, dPhiHydX, dPhiHydY = calc_phi_hyd(        # :107-111
                iMin, iMax, jMin, jMax, k, phiHydF, phiHydC, dPhiHydX, dPhiHydY, dyn.startTime, -1,
                cfg=cfg, grid=grid, params=dyn, state=state, phi0surf=z2, eos=eos)
            d = np.asarray(state.totPhiHyd[iI, jI, k]) - oldPhi
            m = np.asarray(grid.maskC[iI, jI, k])
            diffs.append(d * d * m)                                            # :114-116 (..)**2 * maskC
            masks.append(m)
        for t in range(diffs[0].shape[0]):                                     # tiles in tile order
            v = np.concatenate([np.ravel(x[t]) for x in diffs])                # DO k / DO j / DO i
            w = np.concatenate([np.ravel(x[t]) for x in masks])
            rmsTile = np.add.accumulate(np.concatenate([[0.0], v]))[-1]        # :92, :114-116
            sumTile = np.add.accumulate(np.concatenate([[0.0], w]))[-1]        # :93, :117
            rmspp = rmspp + rmsTile                                            # :121
            count = count + sumTile                                            # :122
        # :126-127 _GLOBAL_SUM_RL: identity for one process
        if count == 0.:                                                        # :128-132
            rmspp = np.float64(0.0)
        else:
            rmspp = np.sqrt(rmspp/count)
    if rmspp != 0. and not rmspp == rmsppold:                                  # :174-198
        raise RuntimeError(f"Initial hydrostatic pressure did not converge after {npiter-1:2d} steps\n"
                           "ABNORMAL END: S/R INI_PRESSURE")
    return state
