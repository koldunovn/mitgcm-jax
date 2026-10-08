"""ECCO_DIFFMSK, ECCO_ADDCOST, ECCO_MULT, ECCO_READWEI   @63cdc0b pkg/ecco/ecco_toolbox.F (lane M4ADCOL session 3)

The routines COST_GENLOOP calls for this build, on interior values [tile, nz, sNy, sNx] (each loops k = 1..nLev,
j = 1..sNy, i = 1..sNx). ECCO_ZERO (:30-68) is the zero (or spzero) fill of a local. ECCO_READWEI reads a file: host
numpy (IEEE float64 arithmetic, as gfortran's). ECCO_READBAR is the read-back of the carried bar records
(cost_averagesfields.py): the interior of record iRec (:845-866, ACTIVE_READ_XYZ with lAdInit = .FALSE.).
"""

import jax.numpy as jnp
import numpy as np

from mitjax.pkg.cost.cost_h import chain_sum


def ecco_diffmsk(localBar, localObs, localMask, spMinLoc, spMaxLoc, spzeroLoc):
    """ECCO_DIFFMSK( localBar, localObs, localMask, nzIn, nLev, spMinLoc, spMaxLoc, spzeroLoc, localDif, difMask,
    myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:74-132, :115-122: difMask = localMask; difMask = 0. _d 0 where
    localObs .LT. spMinLoc .OR. .GT. spMaxLoc .OR. .EQ. spzeroLoc; localDif = difMask*(localBar-localObs).
    Returns (localDif, difMask)."""
    bad = (localObs < spMinLoc) | (localObs > spMaxLoc) | (localObs == spzeroLoc)            # :116-118
    difMask = jnp.where(bad, 0.0, localMask)                                                 # :115, :119
    localDif = difMask * (localBar - localObs)                                               # :121-122
    return localDif, difMask


def ecco_addcost(localDif, localWeight, difMask, doSumSq, objf_local, num_local):
    """ECCO_ADDCOST( localDif, localWeight, difMask, nzIn, nLev, doSumSq, objf_local, num_local, myThid )
    @63cdc0b pkg/ecco/ecco_toolbox.F:238-303, :280-296: per tile localcost = 0. _d 0, then over k, j, i (outer to
    inner) localwww = localWeight*difMask, junk = localDif, localcost = localcost + junk*junk*localwww (doSumSq; else
    + junk*localwww), num_local + 1. _d 0 where localwww .NE. 0.; objf_local = objf_local + localcost.
    Returns (objf_local, num_local) [tile]."""
    localwww = localWeight * difMask                                                         # :284
    junk = localDif                                                                          # :285
    term = junk * junk * localwww if doSumSq else junk * localwww                            # :286-290
    localcost = chain_sum(term)                                                              # :280, :287
    num_local = num_local + chain_sum(jnp.where(localwww != 0.0, 1.0, 0.0))                  # :291-292
    return objf_local + localcost, num_local                                                 # :296


def ecco_mult(fld, multLoc):
    """ECCO_MULT( fld, multLoc, nzIn, nLev, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:573-615, :605:
    fld = fld*multLoc."""
    return fld * multLoc


def ecco_readwei(raw, doSumSq):
    """The conversion of ECCO_READWEI   @63cdc0b pkg/ecco/ecco_toolbox.F:912-978 on the values READ_REC_LEV_RL read
    (:950-951; `raw` REAL*8 numpy): :959-967 -- missing values (.LT. -9900.: 0. _d 0), else where .NE. 0.:
    oneRL/w/w with doSumSq, oneRL/w otherwise. Host numpy (left-to-right divisions, IEEE)."""
    w = np.asarray(raw, np.float64)
    nz = w != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        conv = np.where(nz, (1.0 / w) / w if doSumSq else 1.0 / w, w)
    return np.where(w < -9900.0, 0.0, conv)


# lane M4ADLAB session 4: the routines of COST_GENLOOP's 'mean' / 'offset' / 'mindepth' pre-processing
# (lab_sea/input_ad, gencost 'mdt')

def ecco_cp(fldIn, fldOut):
    """ECCO_CP( fldIn, fldOut, nzIn, nLev, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:138-183, :167-177:
    fldOut(i,j,k,bi,bj) = fldIn(i,j,k,bi,bj) over the interior, k = 1..nLev. On interior values the copy is fldIn;
    `fldOut` only fixes the shape."""
    return jnp.broadcast_to(fldIn, jnp.shape(fldOut))


def ecco_divfield(fld, fldDenom):
    """ECCO_DIVFIELD( fld, fldDenom, nzIn, nLev, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:523-567, :553-557:
    fld = fld/fldDenom where fldDenom .NE. 0. _d 0, else 0. _d 0. Guard before the division: the divisor is 1 on the
    lanes that take the ELSE arm (mitjax/ops/safe.py rule)."""
    nz = fldDenom != 0.0
    return jnp.where(nz, fld / jnp.where(nz, fldDenom, 1.0), 0.0)


def ecco_addmask(fldIn, fldInmask, fldOut, fldOutnum):
    """ECCO_ADDMASK( fldIn, fldInmask, fldOut, fldOutnum, nzIn, nLev, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:
    414-466, :451-455: where fldInmask .NE. 0. _d 0: fldOut = fldOut + fldIn, fldOutnum = fldOutnum + 1. _d 0.
    Returns (fldOut, fldOutnum)."""
    m = fldInmask != 0.0
    return jnp.where(m, fldOut + fldIn, fldOut), jnp.where(m, fldOutnum + 1.0, fldOutnum)


def ecco_maskmindepth(difMask, R_low, topoMin):
    """ECCO_MASKMINDEPTH( difMask, nzIn, nLev, topoMin, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:669-714,
    :701-705: difMask(i,j,k) = zeroRL for k = 1..nLev where R_low(i,j,bi,bj) .GT. topoMin. `R_low` [tile, 1, sNy, sNx]
    (GRID.h, broadcast over k)."""
    return jnp.where(R_low > topoMin, 0.0, difMask)


def ecco_offset(fld, difMask, ex=None):
    """ECCO_OFFSET( fName, fld, difMask, nzIn, nLev, myThid )   @63cdc0b pkg/ecco/ecco_toolbox.F:720-811: per tile
    (:756-770) volTile = 0. _d 0, sumTile = 0. _d 0, then over k, j, i (outer to inner) tmpVol = difMask, volTile =
    volTile + tmpVol, sumTile = sumTile + tmpVol*fld; GLOBAL_SUM_TILE_RL of both (:772-773); IF volGlob .GT. zeroRL
    (:775-789) theMean = sumGlob/volGlob and fld = fld - theMean where difMask .NE. 0. _d 0, ELSE theMean = 0. _d 0
    (:791). The prints (:795-806) are `cost_gencost_all.ecco_offset_lines` on the host.
    Returns (fld, volGlob, theMean)."""
    from mitjax.eesupp.global_sum import global_sum_tile
    tmpVol = difMask                                                                         # :763
    volTile = chain_sum(tmpVol)                                                              # :758, :764
    sumTile = chain_sum(tmpVol * fld)                                                        # :759, :765
    gs = global_sum_tile if ex is None else ex.global_sum_tile
    volGlob = gs(volTile)                                                                    # :772
    sumGlob = gs(sumTile)                                                                    # :773
    pos = volGlob > 0.0                                                                      # :775
    theMean = jnp.where(pos, sumGlob / jnp.where(pos, volGlob, 1.0), 0.0)                    # :776, :791
    fld = jnp.where(pos & (difMask != 0.0), fld - theMean, fld)                              # :782-784
    return fld, volGlob, theMean
