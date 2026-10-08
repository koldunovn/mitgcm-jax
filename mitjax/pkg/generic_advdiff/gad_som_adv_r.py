"""GAD_SOM_ADV_R: Second-Order Moments advection of a tracer in the vertical (Prather, 1986), one level.

Constants of EEPARAMS.h are defined here with their citation (GAD-A's `gad_h.py` and an EEPARAMS module are not merged
yet; the main session consolidates them at merge).
"""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_som_lim_r import adv_r_ranges

# eesupp/inc/EEPARAMS.h:72-73   PARAMETER ( zeroRL = 0.0 _d 0 , oneRL = 1.0 _d 0 ); ( twoRL = 2.0 _d 0 , ... )
zeroRL = 0.0
twoRL = 2.0
# pkg/generic_advdiff/gad_som_adv_r.F:116-117   _RL three; PARAMETER( three = 3. _d 0 )
three = 3.0


def gad_som_adv_r(k, kUp, kDw, deltaTloc, rTrans, maskUp, maskIn,
                  sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
                  alp, aln, fp_v, fn_v, fp_o, fn_o,
                  fp_x, fn_x, fp_y, fn_y, fp_z, fn_z,
                  fp_xx, fn_xx, fp_yy, fn_yy, fp_zz, fn_zz,
                  fp_xy, fn_xy, fp_xz, fn_xz, fp_yz, fn_yz,
                  wT, *, cfg):
    """GAD_SOM_ADV_R(bi,bj,k, kUp, kDw, deltaTloc, rTrans, maskUp, maskIn,
                     sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
                     alp, aln, fp_v, fn_v, fp_o, fn_o, fp_x, fn_x, fp_y, fn_y, fp_z, fn_z,
                     fp_xx, fn_xx, fp_yy, fn_yy, fp_zz, fn_zz, fp_xy, fn_xy, fp_xz, fn_xz, fp_yz, fn_yz,
                     wT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_som_adv_r.F:13-370

    C !DESCRIPTION:
    C  Calculates the area integrated vertical flux due to advection
    C  of a tracer using
    C        Second-Order Moments Advection of tracer in Z-direction
    C        ref: M.J.Prather, 1986, JGR, 91, D6, pp 6671-6681.
    C      The 3-D grid has dimension  (Nx,Ny,Nz) with corresponding
    C      velocity field (U,V,W).  Parallel subroutine calculate
    C      advection in the X- and Y- directions.
    C      The moment [Si] are as defined in the text, Sm refers to
    C      the total mass in each grid box
    C      the moments [Fi] are similarly defined and used as temporary
    C      storage for portions of the grid boxes in transit.
    C !INPUT PARAMETERS:
    C  k            :: vertical level
    C  kUp          :: index into 2 1/2D array, toggles between 1 and 2
    C  kDw          :: index into 2 1/2D array, toggles between 2 and 1
    C  rTrans       :: vertical volume transport
    C  maskUp       :: 2-D array mask for W points
    C  maskIn       :: 2-D array Interior mask
    C !OUTPUT PARAMETERS:
    C  sm_v         :: volume of grid cell
    C  sm_o         :: tracer content of grid cell (zero order moment)
    C  sm_x,y,z     :: 1rst order moment of tracer distribution, in x,y,z direction
    C  sm_xx,yy,zz  ::  2nd order moment of tracer distribution, in x,y,z direction
    C  sm_xy,xz,yz  ::  2nd order moment of tracer distr., in cross direction xy,xz,yz
    C  wT           :: vertical advective flux

    Returns the 11 moments (3-D FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:Nr)), the 24 flux work arrays alp .. fn_yz
    (FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:2), indexed by kUp/kDw) and wT (2-D), in the argument order; points the
    Fortran does not write keep their input values. `k`, `kUp`, `kDw` are static; `deltaTloc` is traced. Loop ranges
    iMinAdvR:iMaxAdvR = 1:sNx, jMinAdvR:jMaxAdvR = 1:sNy (GAD.h:108-109).

    Ported branches: the surface flux at k = 1 (:198-225), the interior W<0 flux at k > 1 (:226-260); the ALLOW_AUTODIFF
    self-assignments x(1,1,kDw) = x(1,1,kDw) (:127-155, TAF hints) leave every value unchanged and are not written out.
    Not ported (raise): the w_surf term of a linear free surface whose surface is not at k = 1 (:261-289,
    .NOT.uniformFreeSurfLev .AND. k.NE.1 .AND. .NOT.noFlowAcrossSurf), ALLOW_OBCS (:296-298, :324-326).

    Every point loop runs on its whole (i,j) range at once: part 1 writes only the kUp slot of the flux arrays from
    the moments at k (and km1); parts 2 and 3 read and write the moments only at (i,j,k), in the Fortran statement
    order. In the k = 1 branch `alnq` and `aln1` are computed and not used (:208-209), as in the Fortran. `0.` is not
    used here; all literals are `_d 0` doubles. `recip_dT` (:157-158) is guarded before the division (safe_div).
    """
    if cfg.ALLOW_OBCS:
        raise NotImplementedError("GAD_SOM_ADV_R: ALLOW_OBCS (maskIn test, gad_som_adv_r.F:296-298) is not ported")
    iMinAdvR, iMaxAdvR, jMinAdvR, jMaxAdvR = adv_r_ranges(cfg)

    recip_dT = safe_div(1.0, deltaTloc, deltaTloc > zeroRL, zeroRL)  # :157-158
    noFlowAcrossSurf = (cfg.rigidLid or cfg.nonlinFreeSurf >= 1     # :159-160
                        or cfg.select_rStar != 0)

#---  part.1 : calculate flux for all moments                     # :167-197
    j = loop_j(jMinAdvR, jMaxAdvR)
    i = loop_i(iMinAdvR, iMaxAdvR)
    wLoc = rTrans[i, j]*deltaTloc
#--    Flux from (k) to (k-1) when W>0 (i.e., take upper side of box k)
#- note: Linear free surface case: this takes care of w_surf advection out
#       of the domain since for this particular case, rTrans is not masked
    fp_v = fp_v.at[i, j, kUp].set(MAX(zeroRL, wLoc, p="b"))    # :174
    alp = alp.at[i, j, kUp].set(fp_v[i, j, kUp]/sm_v[i, j, k])
    alpq = alp[i, j, kUp]*alp[i, j, kUp]
    alp1 = 1.0 - alp[i, j, kUp]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "p" for positive W
    fp_o = fp_o.at[i, j, kUp].set(alp[i, j, kUp]*
                                  (sm_o[i, j, k] + alp1*sm_z[i, j, k]
                                   + alp1*(alp1-alp[i, j, kUp])*sm_zz[i, j, k]
                                   ))
    fp_z = fp_z.at[i, j, kUp].set(alpq*
                                  (sm_z[i, j, k] + three*alp1*sm_zz[i, j, k]))
    fp_zz = fp_zz.at[i, j, kUp].set(alp[i, j, kUp]*alpq*sm_zz[i, j, k])
    fp_x = fp_x.at[i, j, kUp].set(alp[i, j, kUp]*
                                  (sm_x[i, j, k] + alp1*sm_xz[i, j, k]))
    fp_y = fp_y.at[i, j, kUp].set(alp[i, j, kUp]*
                                  (sm_y[i, j, k] + alp1*sm_yz[i, j, k]))
    fp_xz = fp_xz.at[i, j, kUp].set(alpq*sm_xz[i, j, k])
    fp_yz = fp_yz.at[i, j, kUp].set(alpq*sm_yz[i, j, k])
    fp_xx = fp_xx.at[i, j, kUp].set(alp[i, j, kUp]*sm_xx[i, j, k])
    fp_yy = fp_yy.at[i, j, kUp].set(alp[i, j, kUp]*sm_yy[i, j, k])
    fp_xy = fp_xy.at[i, j, kUp].set(alp[i, j, kUp]*sm_xy[i, j, k])
    if k == 1:                                                      # :198-225
#--   Linear free surface, calculate w_surf (<0) advection term
        km1 = 1
        wLoc = rTrans[i, j]*deltaTloc
#-     Flux from above to (k) when W<0 , surface case:
#      take box k=1, assuming zero 1rst & 2nd moment in Z dir.
        fn_v = fn_v.at[i, j, kUp].set(MAX(zeroRL, -wLoc, p="a"))   # :206
        aln = aln.at[i, j, kUp].set(fn_v[i, j, kUp]/sm_v[i, j, km1])
        alnq = aln[i, j, kUp]*aln[i, j, kUp]                        # noqa: F841 (:208, not used, as in the Fortran)
        aln1 = 1.0 - aln[i, j, kUp]                                 # noqa: F841 (:209, not used, as in the Fortran)
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "n" for negative W
        fn_o = fn_o.at[i, j, kUp].set(aln[i, j, kUp]*sm_o[i, j, km1])
        fn_z = fn_z.at[i, j, kUp].set(zeroRL)
        fn_zz = fn_zz.at[i, j, kUp].set(zeroRL)
        fn_x = fn_x.at[i, j, kUp].set(aln[i, j, kUp]*sm_x[i, j, km1])
        fn_y = fn_y.at[i, j, kUp].set(aln[i, j, kUp]*sm_y[i, j, km1])
        fn_xz = fn_xz.at[i, j, kUp].set(zeroRL)
        fn_yz = fn_yz.at[i, j, kUp].set(zeroRL)
        fn_xx = fn_xx.at[i, j, kUp].set(aln[i, j, kUp]*sm_xx[i, j, km1])
        fn_yy = fn_yy.at[i, j, kUp].set(aln[i, j, kUp]*sm_yy[i, j, km1])
        fn_xy = fn_xy.at[i, j, kUp].set(aln[i, j, kUp]*sm_xy[i, j, km1])
#--    Save zero-order flux:
        wT = wT.at[i, j].set((fp_o[i, j, kUp] - fn_o[i, j, kUp])*recip_dT)
    else:                                                           # :226-260
#--   Interior only: mask rTrans (if not already done)
        km1 = k-1
        wLoc = maskUp[i, j]*rTrans[i, j]*deltaTloc
#-     Flux from (k-1) to (k) when W<0 (i.e., take lower side of box k-1)
        fn_v = fn_v.at[i, j, kUp].set(MAX(zeroRL, -wLoc, p="a"))   # :233
        aln = aln.at[i, j, kUp].set(fn_v[i, j, kUp]/sm_v[i, j, km1])
        alnq = aln[i, j, kUp]*aln[i, j, kUp]
        aln1 = 1.0 - aln[i, j, kUp]
#-     Create temporary moments/masses for partial boxes in transit
#       use same indexing as velocity, "n" for negative W
        fn_o = fn_o.at[i, j, kUp].set(aln[i, j, kUp]*
                                      (sm_o[i, j, km1] - aln1*sm_z[i, j, km1]
                                       + aln1*(aln1-aln[i, j, kUp])*sm_zz[i, j, km1]
                                       ))
        fn_z = fn_z.at[i, j, kUp].set(alnq*
                                      (sm_z[i, j, km1] - three*aln1*sm_zz[i, j, km1]))
        fn_zz = fn_zz.at[i, j, kUp].set(aln[i, j, kUp]*alnq*sm_zz[i, j, km1])
        fn_x = fn_x.at[i, j, kUp].set(aln[i, j, kUp]*
                                      (sm_x[i, j, km1] - aln1*sm_xz[i, j, km1]))
        fn_y = fn_y.at[i, j, kUp].set(aln[i, j, kUp]*
                                      (sm_y[i, j, km1] - aln1*sm_yz[i, j, km1]))
        fn_xz = fn_xz.at[i, j, kUp].set(alnq*sm_xz[i, j, km1])
        fn_yz = fn_yz.at[i, j, kUp].set(alnq*sm_yz[i, j, km1])
        fn_xx = fn_xx.at[i, j, kUp].set(aln[i, j, kUp]*sm_xx[i, j, km1])
        fn_yy = fn_yy.at[i, j, kUp].set(aln[i, j, kUp]*sm_yy[i, j, km1])
        fn_xy = fn_xy.at[i, j, kUp].set(aln[i, j, kUp]*sm_xy[i, j, km1])
#--    Save zero-order flux:
        wT = wT.at[i, j].set((fp_o[i, j, kUp] - fn_o[i, j, kUp])*recip_dT)
#--   end surface/interior cases for W<0 advective fluxes
    if not cfg.uniformFreeSurfLev and k != 1 and not noFlowAcrossSurf:
        raise NotImplementedError("GAD_SOM_ADV_R: linear free surface with the surface not at k=1 "
                                  "(gad_som_adv_r.F:261-289, .NOT.uniformFreeSurfLev) is not ported")

#---  part.2 : re-adjust moments remaining in the box               # :291-317
#      take off from grid box (k): negative W(kDw) and positive W(kUp)
    alf1 = 1.0 - aln[i, j, kDw] - alp[i, j, kUp]
    alf1q = alf1*alf1
    alpmn = alp[i, j, kUp] - aln[i, j, kDw]
    sm_v = sm_v.at[i, j, k].set(sm_v[i, j, k] - fn_v[i, j, kDw] - fp_v[i, j, kUp])
    sm_o = sm_o.at[i, j, k].set(sm_o[i, j, k] - fn_o[i, j, kDw] - fp_o[i, j, kUp])
    sm_z = sm_z.at[i, j, k].set(alf1q*(sm_z[i, j, k] - three*alpmn*sm_zz[i, j, k]))
    sm_zz = sm_zz.at[i, j, k].set(alf1*alf1q*sm_zz[i, j, k])
    sm_xz = sm_xz.at[i, j, k].set(alf1q*sm_xz[i, j, k])
    sm_yz = sm_yz.at[i, j, k].set(alf1q*sm_yz[i, j, k])
    sm_x = sm_x.at[i, j, k].set(sm_x[i, j, k] - fn_x[i, j, kDw] - fp_x[i, j, kUp])
    sm_xx = sm_xx.at[i, j, k].set(sm_xx[i, j, k] - fn_xx[i, j, kDw] - fp_xx[i, j, kUp])
    sm_y = sm_y.at[i, j, k].set(sm_y[i, j, k] - fn_y[i, j, kDw] - fp_y[i, j, kUp])
    sm_yy = sm_yy.at[i, j, k].set(sm_yy[i, j, k] - fn_yy[i, j, kDw] - fp_yy[i, j, kUp])
    sm_xy = sm_xy.at[i, j, k].set(sm_xy[i, j, k] - fn_xy[i, j, kDw] - fp_xy[i, j, kUp])

#---  part.3 : Put the temporary moments into appropriate neighboring boxes   # :319-366
#      add into grid box (k): positive W(kDw) and negative W(kUp)
    sm_v = sm_v.at[i, j, k].set(sm_v[i, j, k] + fp_v[i, j, kDw] + fn_v[i, j, kUp])
    alfp = fp_v[i, j, kDw]/sm_v[i, j, k]
    alfn = fn_v[i, j, kUp]/sm_v[i, j, k]
    alf1 = 1.0 - alfp - alfn
    alp1 = 1.0 - alfp
    aln1 = 1.0 - alfn
    alpmn = alfp - alfn
    locTp = alfp*sm_o[i, j, k] - alp1*fp_o[i, j, kDw]
    locTn = alfn*sm_o[i, j, k] - aln1*fn_o[i, j, kUp]
    sm_zz = sm_zz.at[i, j, k].set(alf1*alf1*sm_zz[i, j, k] + alfp*alfp*fp_zz[i, j, kDw]
                                                           + alfn*alfn*fn_zz[i, j, kUp]
                                  - 5.0*(-(alpmn*alf1*sm_z[i, j, k]) + alfp*alp1*fp_z[i, j, kDw]
                                                                     - alfn*aln1*fn_z[i, j, kUp]
                                         + twoRL*alfp*alfn*sm_o[i, j, k] + (alp1-alfp)*locTp
                                                                         + (aln1-alfn)*locTn
                                         ))
    sm_xz = sm_xz.at[i, j, k].set(alf1*sm_xz[i, j, k] + alfp*fp_xz[i, j, kDw]
                                                      + alfn*fn_xz[i, j, kUp]
                                  + three*(alpmn*sm_x[i, j, k] - alp1*fp_x[i, j, kDw]
                                                               + aln1*fn_x[i, j, kUp]
                                           ))
    sm_yz = sm_yz.at[i, j, k].set(alf1*sm_yz[i, j, k] + alfp*fp_yz[i, j, kDw]
                                                      + alfn*fn_yz[i, j, kUp]
                                  + three*(alpmn*sm_y[i, j, k] - alp1*fp_y[i, j, kDw]
                                                               + aln1*fn_y[i, j, kUp]
                                           ))
    sm_z = sm_z.at[i, j, k].set(alf1*sm_z[i, j, k] + alfp*fp_z[i, j, kDw]
                                                   + alfn*fn_z[i, j, kUp]
                                + three*(locTp - locTn))
    sm_o = sm_o.at[i, j, k].set(sm_o[i, j, k] + fp_o[i, j, kDw] + fn_o[i, j, kUp])
    sm_x = sm_x.at[i, j, k].set(sm_x[i, j, k] + fp_x[i, j, kDw] + fn_x[i, j, kUp])
    sm_xx = sm_xx.at[i, j, k].set(sm_xx[i, j, k] + fp_xx[i, j, kDw] + fn_xx[i, j, kUp])
    sm_y = sm_y.at[i, j, k].set(sm_y[i, j, k] + fp_y[i, j, kDw] + fn_y[i, j, kUp])
    sm_yy = sm_yy.at[i, j, k].set(sm_yy[i, j, k] + fp_yy[i, j, kDw] + fn_yy[i, j, kUp])
    sm_xy = sm_xy.at[i, j, k].set(sm_xy[i, j, k] + fp_xy[i, j, kDw] + fn_xy[i, j, kUp])

    return (sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz,
            alp, aln, fp_v, fn_v, fp_o, fn_o,
            fp_x, fn_x, fp_y, fn_y, fp_z, fn_z,
            fp_xx, fn_xx, fp_yy, fn_yy, fp_zz, fn_zz,
            fp_xy, fn_xy, fp_xz, fn_xz, fp_yz, fn_yz,
            wT)
