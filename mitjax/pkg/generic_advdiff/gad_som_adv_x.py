"""GAD_SOM_ADV_X: Second-Order Moments advection of a tracer in the X direction (Prather, 1986).

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
# pkg/generic_advdiff/gad_som_adv_x.F:94-95   _RL three; PARAMETER( three = 3. _d 0 )
three = 3.0


def gad_som_adv_x(k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge,
                  deltaTloc, uTrans, maskIn,
                  sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
                  uT, *, cfg):
    """GAD_SOM_ADV_X(bi,bj,k, limiter, overlapOnly, interiorOnly, N_edge, S_edge, E_edge, W_edge, deltaTloc, uTrans,
                     maskIn, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, uT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_som_adv_x.F:13-330

    C !DESCRIPTION:
    C  Calculates the area integrated zonal flux due to advection
    C  of a tracer using
    C        Second-Order Moments Advection of tracer in X-direction
    C        ref: M.J.Prather, 1986, JGR, 91, D6, pp 6671-6681.
    C      The 3-D grid has dimension  (Nx,Ny,Nz) with corresponding
    C      velocity field (U,V,W).  Parallel subroutine calculate
    C      advection in the Y- and Z- directions.
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
    C  uTrans        :: zonal volume transport
    C  maskIn        :: 2-D array Interior mask
    C !OUTPUT PARAMETERS:
    C  sm_v         :: volume of grid cell
    C  sm_o         :: tracer content of grid cell (zero order moment)
    C  sm_x,y,z     :: 1rst order moment of tracer distribution, in x,y,z direction
    C  sm_xx,yy,zz  ::  2nd order moment of tracer distribution, in x,y,z direction
    C  sm_xy,xz,yz  ::  2nd order moment of tracer distr., in cross direction xy,xz,yz
    C  uT           :: zonal advective flux

    Returns (sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, uT); points the Fortran does not
    write keep their input values. All arrays are 2-D (one level) FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy); `limiter`,
    `overlapOnly`, `interiorOnly` and the edge flags are static; `deltaTloc` is traced.

    Ported branches: the full-tile update (overlapOnly = interiorOnly = .FALSE., one strip: every caller that is not a
    cubed-sphere pass, gad_som_advect.F:333-337); limiter 0 and 1. Not ported (raise): overlapOnly or interiorOnly
    (:142-159, cubed-sphere passes; nbStrips = 2), ALLOW_OBCS (:254-256, :282-284, maskIn test). The CADJ lines are TAF
    directives (no forward effect).

    Every point loop (:169-184, :193-237, :252-275, :280-323) runs on its whole (i,j) range at once: a statement at
    (i,j) reads the moments only at (i,j) in the limiter and in parts 2 and 3 (where the statement order is kept, so a
    moment is read before or after its own update exactly as in the Fortran), and part 1 reads the moments at i-1 and
    i but writes only the flux arrays. `0.` is an exact REAL*4 literal; the others are `_d 0` doubles. `recip_dT`
    (:133-134) is guarded before the division (safe_div) so that deltaTloc = 0 has a finite derivative.
    """
    if overlapOnly or interiorOnly:
        raise NotImplementedError("GAD_SOM_ADV_X: overlapOnly / interiorOnly (cubed-sphere passes, "
                                  "gad_som_adv_x.F:142-159) are not ported")
    if cfg.ALLOW_OBCS:
        raise NotImplementedError("GAD_SOM_ADV_X: ALLOW_OBCS (maskIn test, gad_som_adv_x.F:254-256) is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy

    recip_dT = safe_div(1.0, deltaTloc, deltaTloc > zeroRL, 0.)     # :133-134

#-    Set loop ranges for updating tracer field (splitted in 2 strips)
    nbStrips = 1                                                    # :137-141
    iMinUpd = 1-OLx+1
    iMaxUpd = sNx+OLx-1
    jMinUpd = 1-OLy
    jMaxUpd = sNy+OLy
    assert nbStrips == 1                                            # :142-159 (overlapOnly = interiorOnly = F)

    alp = uT.local("alp")                                           # :104-127 local 2-D arrays
    aln = uT.local("aln")
    fp_v, fn_v = uT.local("fp_v"), uT.local("fn_v")
    fp_o, fn_o = uT.local("fp_o"), uT.local("fn_o")
    fp_x, fn_x = uT.local("fp_x"), uT.local("fn_x")
    fp_y, fn_y = uT.local("fp_y"), uT.local("fn_y")
    fp_z, fn_z = uT.local("fp_z"), uT.local("fn_z")
    fp_xx, fn_xx = uT.local("fp_xx"), uT.local("fn_xx")
    fp_yy, fn_yy = uT.local("fp_yy"), uT.local("fn_yy")
    fp_zz, fn_zz = uT.local("fp_zz"), uT.local("fn_zz")
    fp_xy, fn_xy = uT.local("fp_xy"), uT.local("fn_xy")
    fp_xz, fn_xz = uT.local("fp_xz"), uT.local("fn_xz")
    fp_yz, fn_yz = uT.local("fp_yz"), uT.local("fn_yz")

#--   start 1rst loop on strip number "ns"   (ns = 1 only)
    if limiter == 1:                                                # :164-185
        j = loop_j(jMinUpd, jMaxUpd)
        i = loop_i(iMinUpd-1, iMaxUpd+1)
#     If flux-limiting transport is to be applied, place limits on
#     appropriate moments before transport.
        slpmax = jnp.where(sm_o[i, j] > zeroRL, sm_o[i, j], 0.)
        s1max = slpmax*1.5
        s1new = MIN(s1max, MAX(-s1max, sm_x[i, j], p="a"), p="b")    # :176
        s2new = MIN((slpmax+slpmax-jnp.abs(s1new)/three),             # :177-178
                     MAX(jnp.abs(s1new)-slpmax, sm_xx[i, j], p="a"), p="b")
        sm_xy = sm_xy.at[i, j].set(MIN(slpmax, MAX(-slpmax, sm_xy[i, j], p="a"), p="b"))  # :179
        sm_xz = sm_xz.at[i, j].set(MIN(slpmax, MAX(-slpmax, sm_xz[i, j], p="a"), p="b"))  # :180
        sm_x = sm_x.at[i, j].set(s1new)
        sm_xx = sm_xx.at[i, j].set(s2new)

#---  part.1 : calculate flux for all moments                     # :192-237
    j = loop_j(jMinUpd, jMaxUpd)
    i = loop_i(iMinUpd, iMaxUpd+1)
    uLoc = uTrans[i, j]*deltaTloc
#--    Flux from (i-1) to (i) when U>0 (i.e., take right side of box i-1)
    fp_v = fp_v.at[i, j].set(MAX(zeroRL, uLoc, p="b"))    # :197
    alp = alp.at[i, j].set(fp_v[i, j]/sm_v[i-1, j])
    alpq = alp[i, j]*alp[i, j]
    alp1 = 1.0 - alp[i, j]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "p" for positive U
    fp_o = fp_o.at[i, j].set(alp[i, j]*(sm_o[i-1, j] + alp1*sm_x[i-1, j]
                                        + alp1*(alp1-alp[i, j])*sm_xx[i-1, j]
                                        ))
    fp_x = fp_x.at[i, j].set(alpq*(sm_x[i-1, j] + three*alp1*sm_xx[i-1, j]))
    fp_xx = fp_xx.at[i, j].set(alp[i, j]*alpq*sm_xx[i-1, j])
    fp_y = fp_y.at[i, j].set(alp[i, j]*(sm_y[i-1, j] + alp1*sm_xy[i-1, j]))
    fp_z = fp_z.at[i, j].set(alp[i, j]*(sm_z[i-1, j] + alp1*sm_xz[i-1, j]))
    fp_xy = fp_xy.at[i, j].set(alpq*sm_xy[i-1, j])
    fp_xz = fp_xz.at[i, j].set(alpq*sm_xz[i-1, j])
    fp_yy = fp_yy.at[i, j].set(alp[i, j]*sm_yy[i-1, j])
    fp_zz = fp_zz.at[i, j].set(alp[i, j]*sm_zz[i-1, j])
    fp_yz = fp_yz.at[i, j].set(alp[i, j]*sm_yz[i-1, j])
#--    Flux from (i) to (i-1) when U<0 (i.e., take left side of box i)
    fn_v = fn_v.at[i, j].set(MAX(zeroRL, -uLoc, p="a"))   # :216
    aln = aln.at[i, j].set(fn_v[i, j]/sm_v[i, j])
    alnq = aln[i, j]*aln[i, j]
    aln1 = 1.0 - aln[i, j]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "n" for negative U
    fn_o = fn_o.at[i, j].set(aln[i, j]*(sm_o[i, j] - aln1*sm_x[i, j]
                                        + aln1*(aln1-aln[i, j])*sm_xx[i, j]
                                        ))
    fn_x = fn_x.at[i, j].set(alnq*(sm_x[i, j] - three*aln1*sm_xx[i, j]))
    fn_xx = fn_xx.at[i, j].set(aln[i, j]*alnq*sm_xx[i, j])
    fn_y = fn_y.at[i, j].set(aln[i, j]*(sm_y[i, j] - aln1*sm_xy[i, j]))
    fn_z = fn_z.at[i, j].set(aln[i, j]*(sm_z[i, j] - aln1*sm_xz[i, j]))
    fn_xy = fn_xy.at[i, j].set(alnq*sm_xy[i, j])
    fn_xz = fn_xz.at[i, j].set(alnq*sm_xz[i, j])
    fn_yy = fn_yy.at[i, j].set(aln[i, j]*sm_yy[i, j])
    fn_zz = fn_zz.at[i, j].set(aln[i, j]*sm_zz[i, j])
    fn_yz = fn_yz.at[i, j].set(aln[i, j]*sm_yz[i, j])
#--    Save zero-order flux:
    uT = uT.at[i, j].set((fp_o[i, j] - fn_o[i, j])*recip_dT)

#---  part.2 : re-adjust moments remaining in the box               # :250-275
#      take off from grid box (i): negative U(i) and positive U(i+1)
    j = loop_j(jMinUpd, jMaxUpd)
    i = loop_i(iMinUpd, iMaxUpd)
    alf1 = 1.0 - aln[i, j] - alp[i+1, j]
    alf1q = alf1*alf1
    alpmn = alp[i+1, j] - aln[i, j]
    sm_v = sm_v.at[i, j].set(sm_v[i, j] - fn_v[i, j] - fp_v[i+1, j])
    sm_o = sm_o.at[i, j].set(sm_o[i, j] - fn_o[i, j] - fp_o[i+1, j])
    sm_x = sm_x.at[i, j].set(alf1q*(sm_x[i, j] - three*alpmn*sm_xx[i, j]))
    sm_xx = sm_xx.at[i, j].set(alf1*alf1q*sm_xx[i, j])
    sm_xy = sm_xy.at[i, j].set(alf1q*sm_xy[i, j])
    sm_xz = sm_xz.at[i, j].set(alf1q*sm_xz[i, j])
    sm_y = sm_y.at[i, j].set(sm_y[i, j] - fn_y[i, j] - fp_y[i+1, j])
    sm_yy = sm_yy.at[i, j].set(sm_yy[i, j] - fn_yy[i, j] - fp_yy[i+1, j])
    sm_z = sm_z.at[i, j].set(sm_z[i, j] - fn_z[i, j] - fp_z[i+1, j])
    sm_zz = sm_zz.at[i, j].set(sm_zz[i, j] - fn_zz[i, j] - fp_zz[i+1, j])
    sm_yz = sm_yz.at[i, j].set(sm_yz[i, j] - fn_yz[i, j] - fp_yz[i+1, j])

#---  part.3 : Put the temporary moments into appropriate neighboring boxes   # :278-323
#      add into grid box (i): positive U(i) and negative U(i+1)
    sm_v = sm_v.at[i, j].set(sm_v[i, j] + fp_v[i, j] + fn_v[i+1, j])
    alfp = fp_v[i, j]/sm_v[i, j]
    alfn = fn_v[i+1, j]/sm_v[i, j]
    alf1 = 1.0 - alfp - alfn
    alp1 = 1.0 - alfp
    aln1 = 1.0 - alfn
    alpmn = alfp - alfn
    locTp = alfp*sm_o[i, j] - alp1*fp_o[i, j]
    locTn = alfn*sm_o[i, j] - aln1*fn_o[i+1, j]
    sm_xx = sm_xx.at[i, j].set(alf1*alf1*sm_xx[i, j] + alfp*alfp*fp_xx[i, j]
                                                     + alfn*alfn*fn_xx[i+1, j]
                               - 5.0*(-(alpmn*alf1*sm_x[i, j]) + alfp*alp1*fp_x[i, j]
                                                               - alfn*aln1*fn_x[i+1, j]
                                      + twoRL*alfp*alfn*sm_o[i, j] + (alp1-alfp)*locTp
                                                                   + (aln1-alfn)*locTn
                                      ))
    sm_xy = sm_xy.at[i, j].set(alf1*sm_xy[i, j] + alfp*fp_xy[i, j]
                                                + alfn*fn_xy[i+1, j]
                               + three*(alpmn*sm_y[i, j] - alp1*fp_y[i, j]
                                                         + aln1*fn_y[i+1, j]
                                        ))
    sm_xz = sm_xz.at[i, j].set(alf1*sm_xz[i, j] + alfp*fp_xz[i, j]
                                                + alfn*fn_xz[i+1, j]
                               + three*(alpmn*sm_z[i, j] - alp1*fp_z[i, j]
                                                         + aln1*fn_z[i+1, j]
                                        ))
    sm_x = sm_x.at[i, j].set(alf1*sm_x[i, j] + alfp*fp_x[i, j] + alfn*fn_x[i+1, j]
                             + three*(locTp - locTn))
    sm_o = sm_o.at[i, j].set(sm_o[i, j] + fp_o[i, j] + fn_o[i+1, j])
    sm_y = sm_y.at[i, j].set(sm_y[i, j] + fp_y[i, j] + fn_y[i+1, j])
    sm_yy = sm_yy.at[i, j].set(sm_yy[i, j] + fp_yy[i, j] + fn_yy[i+1, j])
    sm_z = sm_z.at[i, j].set(sm_z[i, j] + fp_z[i, j] + fn_z[i+1, j])
    sm_zz = sm_zz.at[i, j].set(sm_zz[i, j] + fp_zz[i, j] + fn_zz[i+1, j])
    sm_yz = sm_yz.at[i, j].set(sm_yz[i, j] + fp_yz[i, j] + fn_yz[i+1, j])
#--   end 2nd loop on strip number "ns"

    return sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, uT
