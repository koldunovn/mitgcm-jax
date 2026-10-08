"""SEAICE_DIFFUSION: pkg/seaice/seaice_diffusion.F @63cdc0b (lane M4ADLAB session 2: lab_sea/input_ad,
SEAICEdiffKhArea = 200)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
from mitjax.pkg.generic_advdiff.gad_diff_x import gad_diff_x
from mitjax.pkg.generic_advdiff.gad_diff_y import gad_diff_y


def seaice_diffusion(tracerIdentity, diffKh, fac, iceFld, iceMask, xA, yA, gFld, *, cfg, grid):
    """SEAICE_DIFFUSION( tracerIdentity, diffKh, fac, iceFld, iceMask, xA, yA, gFld, bi, bj, myTime, myIter,
    myThid )   @63cdc0b pkg/seaice/seaice_diffusion.F:6-106

    C     | o Add tendency from horizontal diffusion

    The tile loop is the leading axis (the callers loop over bi, bj, :584-596 of seaice_advdiff.F). `diffKh`, `fac`
    traced floats (SEAICEdiffKh*, SEAICE_deltaTtherm or ONE); `iceFld` the field diffused (the caller's fldNm1),
    `iceMask` HEFFM, `xA`, `yA` the face areas (seaice_advdiff.F:102-111), `gFld` updated: returned. The body is
    compiled with ALLOW_GENERIC_ADVDIFF only (:48-103; else gFld is returned unchanged). `IF ( diffKh .GT. 0. )`
    (:63) on a traced REAL: a `where` (both arms finite). fZon, fMer = 0 on every point (:72-77), GAD_DIFF_X / Y
    (:79, :81, k = 1, :71), then the divergence on 1-OLx..sNx+OLx-1, 1-OLy..sNy+OLy-1 (:83-89). The diagnostics
    (:49-53, :64-69, :91-100) are output only; `tracerIdentity` selects only their suffix."""
    del tracerIdentity
    if not cfg.cpp.flag("ALLOW_GENERIC_ADVDIFF"):                              # :48-103
        return gFld
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    k = 1                                                                      # :71
    fZon = gFld.local("fZon", 0.)                                              # :72-77 (0. _d 0)
    fMer = gFld.local("fMer", 0.)
    kc = gad_kernel_cfg(cfg)
    fZon = gad_diff_x(k, xA, diffKh, iceFld, fZon, cfg=kc, grid=grid)          # :79
    fMer = gad_diff_y(k, yA, diffKh, iceFld, fMer, cfg=kc, grid=grid)          # :81
    j = loop_j(1-OLy, sNy+OLy-1)                                               # :83-84
    i = loop_i(1-OLx, sNx+OLx-1)
    new = (gFld[i, j]                                                          # :85-88
           - fac*iceMask[i, j]*grid.recip_rA[i, j]
           * ((fZon[i+1, j]-fZon[i, j])
              + (fMer[i, j+1]-fMer[i, j])))
    return gFld.at[i, j].set(jnp.where(diffKh > 0.0, new, gFld[i, j]))         # :63 IF ( diffKh .GT. 0. _d 0 )
