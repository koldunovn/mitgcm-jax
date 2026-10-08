"""pkg/mom_common/mom_calc_relvort3.F: relative vorticity, incl. the cube-corner points (MOM_CALC_RELVORT3)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_relvort3.F:1


def mom_calc_relvort3(k, uFld, vFld, hFacZ, vort3, *, cfg, grid, params, w2=None):
    """MOM_CALC_RELVORT3(bi,bj,k, uFld, vFld, hFacZ, vort3, myThid)
    @63cdc0b pkg/mom_common/mom_calc_relvort3.F:4-307

    C     | S/R MOM_CALC_RELVORT3
    C       Horizontal curl of flow field - ignoring lopping factors

    Returns vort3. hFacZ is not read (the lopping variant :63-74 is commented out). #ifdef ALLOW_AUTODIFF (:42-48) is
    ported (global_ocean.90x40x15/code_ad). `#undef CALC_CS_CORNER_EXTENDED` (:2): the extended corner arms
    (:127-138, :184-195, :241-252, :291-302) are dead code. The point loop :50-77 runs on the whole (i,j) range.

    Cube corners (:80-304, useCubedSphereExchange, EEPARAMS.h, static `params.useCubedSphereExchange`): ported for
    ALLOW_EXCH2 (global_ocean.cs32x15, solid-body.cs-32x32x1); the non-exch2 cube (`myFace = bi`, :92-98) raises.
    `w2` is the W2_EXCH2_TOPOLOGY.h view of this routine's tiles (the one addition to the Fortran signature):
    per-tile arrays [tile] in tile storage order, i.e. already at myTile = W2_myTileList(bi,bj) (:82-83):
    w2.exch2_myFace, w2.exch2_isWedge, w2.exch2_isSedge, w2.exch2_isEedge, w2.exch2_isNedge (integers). The four
    corner flags and the face tests select per tile (`jnp.where` over the tile axis; every arm is computed, all finite);
    the corner statement writes the single point (I,J) of each tile whose flag is set, as the Fortran does.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxC, dyC, recip_rAz, recip_deepFacC = grid.dxC, grid.dyC, grid.recip_rAz, grid.recip_deepFacC

    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                       # :42-48
        jA = loop_j(1-OLy, sNy+OLy)
        iA = loop_i(1-OLx, sNx+OLx)
        vort3 = vort3.at[iA, jA].set(0.)

    J = loop_j(2-OLy, sNy+OLy)                                      # :50-77
    I = loop_i(2-OLx, sNx+OLx)
#       Horizontal curl of flow field - ignoring lopping factors
    vort3 = vort3.at[I, J].set(
        recip_rAz[I, J]*(
            (vFld[I, J]*dyC[I, J]
             -vFld[I-1, J]*dyC[I-1, J])
            -(uFld[I, J]*dxC[I, J]
              -uFld[I, J-1]*dxC[I, J-1])
                            )*recip_deepFacC[k])

#     Special stuff for Cubed Sphere
    if params.useCubedSphereExchange:                               # :80
        if not cfg.cpp.flag("ALLOW_EXCH2", _OPT):                  # :92-98
            raise NotImplementedError("MOM_CALC_RELVORT3: the cube without ALLOW_EXCH2 (myFace = bi) is not ported")
        if w2 is None:
            raise ValueError("MOM_CALC_RELVORT3: useCubedSphereExchange needs the W2 tile view `w2`")
        myFace = jnp.asarray(w2.exch2_myFace)                       # :82-91
        isW, isE = jnp.asarray(w2.exch2_isWedge) == 1, jnp.asarray(w2.exch2_isEedge) == 1
        isS, isN = jnp.asarray(w2.exch2_isSedge) == 1, jnp.asarray(w2.exch2_isNedge) == 1
        southWestCorner = isW & isS
        southEastCorner = isE & isS
        northEastCorner = isE & isN
        northWestCorner = isW & isN

#--    South West corner (:100-139)
        I, J = 1, 1                                                 # :110-111
        new = (+recip_rAz[I, J]*(                                   # :114-119
               (vFld[I, J]*dyC[I, J]
                -uFld[I, J]*dxC[I, J])
               + uFld[I, J-1]*dxC[I, J-1]
               )*recip_deepFacC[k])
        vort3 = vort3.at[I, J].set(jnp.where(southWestCorner, new, vort3[I, J]))

#--    South East corner (:141-196)
        I, J = sNx+1, 1                                             # :151-152
        f2 = (+recip_rAz[I, J]*(                                    # :155-161 myFace.EQ.2
              (-uFld[I, J]*dxC[I, J]
               -vFld[I-1, J]*dyC[I-1, J])
              + uFld[I, J-1]*dxC[I, J-1]
              )*recip_deepFacC[k])
        f4 = (+recip_rAz[I, J]*(                                    # :162-168 myFace.EQ.4
              (-vFld[I-1, J]*dyC[I-1, J]
               +uFld[I, J-1]*dxC[I, J-1])
              - uFld[I, J]*dxC[I, J]
              )*recip_deepFacC[k])
        fo = (+recip_rAz[I, J]*(                                    # :169-175 ELSE
              (+uFld[I, J-1]*dxC[I, J-1]
               -uFld[I, J]*dxC[I, J])
              - vFld[I-1, J]*dyC[I-1, J]
              )*recip_deepFacC[k])
        new = jnp.where(myFace == 2, f2, jnp.where(myFace == 4, f4, fo))
        vort3 = vort3.at[I, J].set(jnp.where(southEastCorner, new, vort3[I, J]))

#--    North West corner (:198-253)
        I, J = 1, sNy+1                                             # :208-209
        f1 = (+recip_rAz[I, J]*(                                    # :212-218 myFace.EQ.1
              (+uFld[I, J-1]*dxC[I, J-1]
               +vFld[I, J]*dyC[I, J])
              - uFld[I, J]*dxC[I, J]
              )*recip_deepFacC[k])
        f3 = (+recip_rAz[I, J]*(                                    # :219-225 myFace.EQ.3
              (-uFld[I, J]*dxC[I, J]
               +uFld[I, J-1]*dxC[I, J-1])
              + vFld[I, J]*dyC[I, J]
              )*recip_deepFacC[k])
        fo = (+recip_rAz[I, J]*(                                    # :226-232 ELSE
              (+vFld[I, J]*dyC[I, J]
               -uFld[I, J]*dxC[I, J])
              + uFld[I, J-1]*dxC[I, J-1]
              )*recip_deepFacC[k])
        new = jnp.where(myFace == 1, f1, jnp.where(myFace == 3, f3, fo))
        vort3 = vort3.at[I, J].set(jnp.where(northWestCorner, new, vort3[I, J]))

#--    North East corner (:255-303)
        I, J = sNx+1, sNy+1                                         # :265-266
        fodd = (+recip_rAz[I, J]*(                                  # :269-275 MOD(myFace,2).EQ.1
                (-uFld[I, J]*dxC[I, J]
                 -vFld[I-1, J]*dyC[I-1, J])
                + uFld[I, J-1]*dxC[I, J-1]
                )*recip_deepFacC[k])
        fevn = (+recip_rAz[I, J]*(                                  # :276-282 ELSE
                (+uFld[I, J-1]*dxC[I, J-1]
                 -uFld[I, J]*dxC[I, J])
                - vFld[I-1, J]*dyC[I-1, J]
                )*recip_deepFacC[k])
        new = jnp.where(jnp.mod(myFace, 2) == 1, fodd, fevn)
        vort3 = vort3.at[I, J].set(jnp.where(northEastCorner, new, vort3[I, J]))

    return vort3
