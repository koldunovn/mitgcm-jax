"""INI_DYNVARS: model/src/ini_dynvars.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j, loops_kji

_XYZ = ("uVel", "vVel", "wVel", "theta", "salt", "gU", "gV")                    # :58-65
_AB2 = ("guNm1", "gvNm1", "gtNm1", "gsNm1")                                     # :80-83
_AB3 = ("guNm", "gvNm", "gtNm", "gsNm")                                         # :71-78
_DIAG = ("totPhiHyd", "rhoInSitu", "IVDConvCount")                              # :91-93
_XY = ("etaN", "etaH", "phiHydLow", "hMixLayer")                                # :111-114


def ini_dynvars(state, *, cfg, grid=None, dyn=None):
    """INI_DYNVARS( myThid )   @63cdc0b model/src/ini_dynvars.F:13-136

    C     | SUBROUTINE INI_DYNVARS
    C     | o Initialise to zero all DYNVARS.h arrays
    C     | NOTE: If a field is not initialised it may be set to NaN (Not a Number)

    Every point (halos included) of the DYNVARS.h fields this build declares is set to 0. _d 0: uVel ... gV (:58-65),
    the AB histories (guNm(:,:,:,:,:,1:2) under ALLOW_ADAMSBASHFORTH_3, :71-78; guNm1 ... otherwise, :80-83), diffKr
    (ALLOW_3D_DIFFKR or ALLOW_DIFFKR_CONTROL, :85-87), totPhiHyd, rhoInSitu, IVDConvCount (:91-93), etaN, etaH,
    phiHydLow, hMixLayer (:111-114), gT, gS under USE_OLD_EXTERNAL_FORCING (:66-69, adjustment.cs), sigmaRfield under
    ALLOW_LEITH_QG (:94-96, M3 lane MLAdjust). Branches of
    options no ported build sets (ALLOW_SMAG_3D_DIFFUSIVITY, INCLUDE_SOUNDSPEED_CALC_CODE) are refused by
    mitjax/model/state.fields_of. Lane B (Task 25, cs32x15): ALLOW_SOLVE4_PS_AND_DRAG, dU_psFacX = dV_psFacY = 0
    (:97-100), then dU_psFacX = maskW*recip_deepFacC(k)*recip_rhoFacC(k), dV_psFacY with maskS (:117-130; `grid`,
    `dyn`: the Params with recip_rhoFacC). Pointwise; vectorised over k (the iterations are independent)."""
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    names = _XYZ + (_AB3 if cfg.cpp.ALLOW_ADAMSBASHFORTH_3 else _AB2) + _DIAG
    if "sigmaRfield" in state:                                                 # :94-96 ALLOW_LEITH_QG (M3 MLAdjust)
        names = names + ("sigmaRfield",)
    if "diffKr" in state:
        names = names + ("diffKr",)
    if cfg.cpp.USE_OLD_EXTERNAL_FORCING:                                        # lane B (Task 25): :66-69
        names = names + ("gT", "gS")
    if cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG:                                        # lane B (Task 25): :97-100
        names = names + ("dU_psFacX", "dV_psFacY")
    out = {}
    for n in names:
        f = getattr(state, n)
        if isinstance(f, tuple):                                                # guNm(...,1), guNm(...,2)
            out[n] = tuple(a.at[i, j, k].set(0.) for a in f)
        else:
            out[n] = f.at[i, j, k].set(0.)
    j2 = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i2 = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    for n in _XY:
        out[n] = getattr(state, n).at[i2, j2].set(0.)
    if cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG:                                        # :117-130 (lane B)
        out["dU_psFacX"] = out["dU_psFacX"].at[i, j, k].set(
            grid.maskW[i, j, k]*grid.recip_deepFacC[k]*dyn.recip_rhoFacC[k])   # :123-124
        out["dV_psFacY"] = out["dV_psFacY"].at[i, j, k].set(
            grid.maskS[i, j, k]*grid.recip_deepFacC[k]*dyn.recip_rhoFacC[k])   # :125-126
    return state.replace(**out)
