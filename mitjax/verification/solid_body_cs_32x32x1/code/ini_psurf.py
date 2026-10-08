"""INI_PSURF of solid-body.cs-32x32x1: verification/solid-body.cs-32x32x1/code/ini_psurf.F @63cdc0b (the experiment's
own version; lane B, plan Task 25)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XY_RL
from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.rw.read_rec import READ_FLD_XY_RL


def ini_psurf(state, *, cfg, grid, params, ex, rw):
    """INI_PSURF( myThid )   @63cdc0b verification/solid-body.cs-32x32x1/code/ini_psurf.F:7-118

    The surface pressure of the solid-body rotation in gradient-wind balance (:50-67):
    etaN = 0. + psFac*( snFac*fCori*fCori - 1. _d 0 / 3. _d 0 )*recip_Bo with
    psFac = -(rSphere*rSphere)*omegaPrime*( Omega + omegaPrime*0.5 ), snFac = 1/(4*Omega*Omega), on every point;
    pSurfInitFile (:69-73); ALLOW_CD_CODE etaNm1 (:75-86); etaH = etaHnm1 = etaN, dEtaHdt = 0 (:88-99);
    ALLOW_SHELFICE (:101-114) raises."""
    sz = cfg.size
    gp, ip = params.grid, params.init
    rSphere, Omega = gp.rSphere, gp.omega
    omegaPrime = 80.0 / rSphere                                                # :50
    psfac = -(rSphere*rSphere)*omegaPrime*(Omega + omegaPrime*0.5)             # :53-54
    snFac = 1.0 / (4.0*Omega*Omega)                                            # :55
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    fCori = grid.fCori[i, j]
    etaN = state.etaN.at[i, j].set(_zero_plus(                                 # :62-64
        psfac*(snFac*fCori*fCori - 1.0/3.0)*grid.recip_Bo[i, j]))
    if str(ip.pSurfInitFile).strip() != "":                                    # :69-73
        etaN = READ_FLD_XY_RL(ip.pSurfInitFile, " ", etaN, 0, rw=rw)
        etaN = EXCH_XY_RL(etaN, ex=ex)
    out = {"etaN": etaN}
    if cfg.cpp.ALLOW_CD_CODE:                                                  # :75-86
        out["etaNm1"] = state.etaNm1.at[i, j].set(etaN[i, j])
    out["etaH"] = state.etaH.at[i, j].set(etaN[i, j])                          # :92
    out["etaHnm1"] = state.etaHnm1.at[i, j].set(etaN[i, j])                    # :93
    out["dEtaHdt"] = state.dEtaHdt.at[i, j].set(0.)                            # :94
    if cfg.cpp.ALLOW_SHELFICE:                                                 # :101-114
        raise NotImplementedError("INI_PSURF (solid-body): the ALLOW_SHELFICE load anomaly is not ported")
    return state.replace(**out)
