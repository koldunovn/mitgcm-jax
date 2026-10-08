"""GAD_SOM_ADV_Y: Second-Order Moments advection of a tracer in the Y direction (Prather, 1986).

Constants of EEPARAMS.h are defined here with their citation (GAD-A's `gad_h.py` and an EEPARAMS module are not merged
yet; the main session consolidates them at merge).
"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div

# eesupp/inc/EEPARAMS.h:72-73   PARAMETER ( zeroRL = 0.0 _d 0 , oneRL = 1.0 _d 0 ); ( twoRL = 2.0 _d 0 , ... )
zeroRL = 0.0
twoRL = 2.0
# pkg/generic_advdiff/gad_som_adv_y.F:94-95   _RL three; PARAMETER( three = 3. _d 0 )
three = 3.0


def gad_som_adv_y(k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge,
                  deltaTloc, vTrans, maskIn,
                  sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
                  vT, *, cfg):
    """GAD_SOM_ADV_Y(bi,bj,k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge, deltaTloc, vTrans,
                     maskIn, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, vT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_som_adv_y.F:13-331

    C !DESCRIPTION:
    C  Calculates the area integrated meridional flux due to advection
    C  of a tracer using
    C        Second-Order Moments Advection of tracer in Y-direction
    C        ref: M.J.Prather, 1986, JGR, 91, D6, pp 6671-6681.
    C      The 3-D grid has dimension  (Nx,Ny,Nz) with corresponding
    C      velocity field (U,V,W).  Parallel subroutine calculate
    C      advection in the X- and Z- directions.
    C      The moment [Si] are as defined in the text, Sm refers to
    C      the total mass in each grid box
    C      the moments [Fi] are similarly defined and used as temporary
    C      storage for portions of the grid boxes in transit.
    C !INPUT PARAMETERS:
    C  k             :: vertical level
    C  limiter       :: 0: no limiter ; 1: Prather, 1986 limiter
    C  overlapOnly   :: only update the edges of myTile, but not the interior
    C  interiorOnly  :: only update the interior of myTile, but not the edges
    C [N,S,E,W]_edge :: true if N,S,E,W edge of myTile is an Edge of the cube
    C  vTrans        :: zonal volume transport
    C  maskIn        :: 2-D array Interior mask
    C !OUTPUT PARAMETERS:
    C  sm_v         :: volume of grid cell
    C  sm_o         :: tracer content of grid cell (zero order moment)
    C  sm_x,y,z     :: 1rst order moment of tracer distribution, in x,y,z direction
    C  sm_xx,yy,zz  ::  2nd order moment of tracer distribution, in x,y,z direction
    C  sm_xy,xz,yz  ::  2nd order moment of tracer distr., in cross direction xy,xz,yz
    C  vT           :: meridional advective flux

    Returns (sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, vT); points the Fortran does not
    write keep their input values. All arrays are 2-D (one level) FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy); `limiter`,
    `overlapOnly`, `interiorOnly` and the edge flags are static; `deltaTloc` is traced.

    Ported branches: the full-tile update (overlapOnly = interiorOnly = .FALSE., one strip); limiter 0 and 1. Not
    ported (raise): overlapOnly or interiorOnly (:142-159, cubed-sphere passes; nbStrips = 2), ALLOW_OBCS (:255-257,
    :283-285). The CADJ lines are TAF directives (no forward effect). The header comment calls vTrans "zonal" (:55,
    copied verbatim); it is the meridional transport.

    Every point loop (:169-184, :193-238, :253-276, :281-324) runs on its whole (i,j) range at once (same argument as
    GAD_SOM_ADV_X with i and j exchanged). `0.` is an exact REAL*4 literal (the limiter's `sm_o(i,j).GT.0.` at :174 is
    the same comparison as the X routine's `.GT.zeroRL`); the others are `_d 0` doubles.
    """
    if overlapOnly or interiorOnly:
        raise NotImplementedError("GAD_SOM_ADV_Y: overlapOnly / interiorOnly (cubed-sphere passes, "
                                  "gad_som_adv_y.F:142-159) are not ported")
    if cfg.ALLOW_OBCS:
        raise NotImplementedError("GAD_SOM_ADV_Y: ALLOW_OBCS (maskIn test, gad_som_adv_y.F:255-257) is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy

    recip_dT = safe_div(1.0, deltaTloc, deltaTloc > zeroRL, 0.)     # :133-134

#-    Set loop ranges for updating tracer field (splitted in 2 strips)
    nbStrips = 1                                                    # :137-141
    iMinUpd = 1-OLx
    iMaxUpd = sNx+OLx
    jMinUpd = 1-OLy+1
    jMaxUpd = sNy+OLy-1
    assert nbStrips == 1                                            # :142-159 (overlapOnly = interiorOnly = F)

    alp = vT.local("alp")                                           # :104-127 local 2-D arrays
    aln = vT.local("aln")
    fp_v, fn_v = vT.local("fp_v"), vT.local("fn_v")
    fp_o, fn_o = vT.local("fp_o"), vT.local("fn_o")
    fp_x, fn_x = vT.local("fp_x"), vT.local("fn_x")
    fp_y, fn_y = vT.local("fp_y"), vT.local("fn_y")
    fp_z, fn_z = vT.local("fp_z"), vT.local("fn_z")
    fp_xx, fn_xx = vT.local("fp_xx"), vT.local("fn_xx")
    fp_yy, fn_yy = vT.local("fp_yy"), vT.local("fn_yy")
    fp_zz, fn_zz = vT.local("fp_zz"), vT.local("fn_zz")
    fp_xy, fn_xy = vT.local("fp_xy"), vT.local("fn_xy")
    fp_xz, fn_xz = vT.local("fp_xz"), vT.local("fn_xz")
    fp_yz, fn_yz = vT.local("fp_yz"), vT.local("fn_yz")

#--   start 1rst loop on strip number "ns"   (ns = 1 only)
    if limiter == 1:                                                # :164-185
        j = loop_j(jMinUpd-1, jMaxUpd+1)
        i = loop_i(iMinUpd, iMaxUpd)
#     If flux-limiting transport is to be applied, place limits on
#     appropriate moments before transport.
        slpmax = jnp.where(sm_o[i, j] > 0., sm_o[i, j], 0.)
        s1max = slpmax*1.5
        s1new = MIN(s1max, MAX(-s1max, sm_y[i, j], p="a"), p="b")    # :176
        s2new = MIN((slpmax+slpmax-jnp.abs(s1new)/three),             # :177-178
                     MAX(jnp.abs(s1new)-slpmax, sm_yy[i, j], p="a"), p="b")
        sm_xy = sm_xy.at[i, j].set(MIN(slpmax, MAX(-slpmax, sm_xy[i, j], p="a"), p="b"))  # :179
        sm_yz = sm_yz.at[i, j].set(MIN(slpmax, MAX(-slpmax, sm_yz[i, j], p="a"), p="b"))  # :180
        sm_y = sm_y.at[i, j].set(s1new)
        sm_yy = sm_yy.at[i, j].set(s2new)

#---  part.1 : calculate flux for all moments                     # :192-238
    j = loop_j(jMinUpd, jMaxUpd+1)
    i = loop_i(iMinUpd, iMaxUpd)
    vLoc = vTrans[i, j]*deltaTloc
#--    Flux from (j-1) to (j) when V>0 (i.e., take right side of box j-1)
    fp_v = fp_v.at[i, j].set(MAX(zeroRL, vLoc, p="b"))    # :197
    alp = alp.at[i, j].set(fp_v[i, j]/sm_v[i, j-1])
    alpq = alp[i, j]*alp[i, j]
    alp1 = 1.0 - alp[i, j]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "p" for positive V
    fp_o = fp_o.at[i, j].set(alp[i, j]*(sm_o[i, j-1] + alp1*sm_y[i, j-1]
                                        + alp1*(alp1-alp[i, j])*sm_yy[i, j-1]
                                        ))
    fp_y = fp_y.at[i, j].set(alpq*(sm_y[i, j-1] + three*alp1*sm_yy[i, j-1]))
    fp_yy = fp_yy.at[i, j].set(alp[i, j]*alpq*sm_yy[i, j-1])
    fp_x = fp_x.at[i, j].set(alp[i, j]*(sm_x[i, j-1] + alp1*sm_xy[i, j-1]))
    fp_z = fp_z.at[i, j].set(alp[i, j]*(sm_z[i, j-1] + alp1*sm_yz[i, j-1]))

    fp_xy = fp_xy.at[i, j].set(alpq*sm_xy[i, j-1])
    fp_yz = fp_yz.at[i, j].set(alpq*sm_yz[i, j-1])
    fp_xx = fp_xx.at[i, j].set(alp[i, j]*sm_xx[i, j-1])
    fp_zz = fp_zz.at[i, j].set(alp[i, j]*sm_zz[i, j-1])
    fp_xz = fp_xz.at[i, j].set(alp[i, j]*sm_xz[i, j-1])
#--    Flux from (j) to (j-1) when V<0 (i.e., take left side of box j)
    fn_v = fn_v.at[i, j].set(MAX(zeroRL, -vLoc, p="a"))   # :217
    aln = aln.at[i, j].set(fn_v[i, j]/sm_v[i, j])
    alnq = aln[i, j]*aln[i, j]
    aln1 = 1.0 - aln[i, j]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "n" for negative V
    fn_o = fn_o.at[i, j].set(aln[i, j]*(sm_o[i, j] - aln1*sm_y[i, j]
                                        + aln1*(aln1-aln[i, j])*sm_yy[i, j]
                                        ))
    fn_y = fn_y.at[i, j].set(alnq*(sm_y[i, j] - three*aln1*sm_yy[i, j]))
    fn_yy = fn_yy.at[i, j].set(aln[i, j]*alnq*sm_yy[i, j])
    fn_x = fn_x.at[i, j].set(aln[i, j]*(sm_x[i, j] - aln1*sm_xy[i, j]))
    fn_z = fn_z.at[i, j].set(aln[i, j]*(sm_z[i, j] - aln1*sm_yz[i, j]))
    fn_xy = fn_xy.at[i, j].set(alnq*sm_xy[i, j])
    fn_yz = fn_yz.at[i, j].set(alnq*sm_yz[i, j])
    fn_xx = fn_xx.at[i, j].set(aln[i, j]*sm_xx[i, j])
    fn_zz = fn_zz.at[i, j].set(aln[i, j]*sm_zz[i, j])
    fn_xz = fn_xz.at[i, j].set(aln[i, j]*sm_xz[i, j])
#--    Save zero-order flux:
    vT = vT.at[i, j].set((fp_o[i, j] - fn_o[i, j])*recip_dT)

#---  part.2 : re-adjust moments remaining in the box               # :251-276
#      take off from grid box (j): negative V(j) and positive V(j+1)
    j = loop_j(jMinUpd, jMaxUpd)
    i = loop_i(iMinUpd, iMaxUpd)
    alf1 = 1.0 - aln[i, j] - alp[i, j+1]
    alf1q = alf1*alf1
    alpmn = alp[i, j+1] - aln[i, j]
    sm_v = sm_v.at[i, j].set(sm_v[i, j] - fn_v[i, j] - fp_v[i, j+1])
    sm_o = sm_o.at[i, j].set(sm_o[i, j] - fn_o[i, j] - fp_o[i, j+1])
    sm_y = sm_y.at[i, j].set(alf1q*(sm_y[i, j] - three*alpmn*sm_yy[i, j]))
    sm_yy = sm_yy.at[i, j].set(alf1*alf1q*sm_yy[i, j])
    sm_xy = sm_xy.at[i, j].set(alf1q*sm_xy[i, j])
    sm_yz = sm_yz.at[i, j].set(alf1q*sm_yz[i, j])
    sm_x = sm_x.at[i, j].set(sm_x[i, j] - fn_x[i, j] - fp_x[i, j+1])
    sm_xx = sm_xx.at[i, j].set(sm_xx[i, j] - fn_xx[i, j] - fp_xx[i, j+1])
    sm_z = sm_z.at[i, j].set(sm_z[i, j] - fn_z[i, j] - fp_z[i, j+1])
    sm_zz = sm_zz.at[i, j].set(sm_zz[i, j] - fn_zz[i, j] - fp_zz[i, j+1])
    sm_xz = sm_xz.at[i, j].set(sm_xz[i, j] - fn_xz[i, j] - fp_xz[i, j+1])

#---  part.3 : Put the temporary moments into appropriate neighboring boxes   # :279-324
#      add into grid box (j): positive V(j) and negative V(j+1)
    sm_v = sm_v.at[i, j].set(sm_v[i, j] + fp_v[i, j] + fn_v[i, j+1])
    alfp = fp_v[i, j]/sm_v[i, j]
    alfn = fn_v[i, j+1]/sm_v[i, j]
    alf1 = 1.0 - alfp - alfn
    alp1 = 1.0 - alfp
    aln1 = 1.0 - alfn
    alpmn = alfp - alfn
    locTp = alfp*sm_o[i, j] - alp1*fp_o[i, j]
    locTn = alfn*sm_o[i, j] - aln1*fn_o[i, j+1]
    sm_yy = sm_yy.at[i, j].set(alf1*alf1*sm_yy[i, j] + alfp*alfp*fp_yy[i, j]
                                                     + alfn*alfn*fn_yy[i, j+1]
                               - 5.0*(-(alpmn*alf1*sm_y[i, j]) + alfp*alp1*fp_y[i, j]
                                                               - alfn*aln1*fn_y[i, j+1]
                                      + twoRL*alfp*alfn*sm_o[i, j] + (alp1-alfp)*locTp
                                                                   + (aln1-alfn)*locTn
                                      ))
    sm_xy = sm_xy.at[i, j].set(alf1*sm_xy[i, j] + alfp*fp_xy[i, j]
                                                + alfn*fn_xy[i, j+1]
                               + three*(alpmn*sm_x[i, j] - alp1*fp_x[i, j]
                                                         + aln1*fn_x[i, j+1]
                                        ))
    sm_yz = sm_yz.at[i, j].set(alf1*sm_yz[i, j] + alfp*fp_yz[i, j]
                                                + alfn*fn_yz[i, j+1]
                               + three*(alpmn*sm_z[i, j] - alp1*fp_z[i, j]
                                                         + aln1*fn_z[i, j+1]
                                        ))
    sm_y = sm_y.at[i, j].set(alf1*sm_y[i, j] + alfp*fp_y[i, j] + alfn*fn_y[i, j+1]
                             + three*(locTp - locTn))
    sm_o = sm_o.at[i, j].set(sm_o[i, j] + fp_o[i, j] + fn_o[i, j+1])
    sm_x = sm_x.at[i, j].set(sm_x[i, j] + fp_x[i, j] + fn_x[i, j+1])
    sm_xx = sm_xx.at[i, j].set(sm_xx[i, j] + fp_xx[i, j] + fn_xx[i, j+1])
    sm_z = sm_z.at[i, j].set(sm_z[i, j] + fp_z[i, j] + fn_z[i, j+1])
    sm_zz = sm_zz.at[i, j].set(sm_zz[i, j] + fp_zz[i, j] + fn_zz[i, j+1])
    sm_xz = sm_xz.at[i, j].set(sm_xz[i, j] + fp_xz[i, j] + fn_xz[i, j+1])
#--   end 2nd loop on strip number "ns"

    return sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, vT
