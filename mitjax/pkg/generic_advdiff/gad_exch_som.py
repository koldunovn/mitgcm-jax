"""GAD_EXCH_SOM: halo exchanges of the 1rst and 2nd-order moments of one tracer (pkg/generic_advdiff).

The exchanges are lane B's `mitjax.eesupp` Exchanger (probed maps, Task 7b): EXCH_UV_AGRID_3D_RL with and without
signs, EXCH_3D_RL and EXCH_SM_3D_RL(.TRUE.) are all probed on every M1 layout (mitjax/eesupp/exch_maps.py).
"""

from mitjax.farray import FArray


def _exchanged(A, data):
    """A with its storage replaced by the exchanged array `data` (same declaration; the shape is checked)."""
    return FArray(data, A.name, tiled=A.tiled, _dims=A.dims)


def gad_exch_som(smTr, myNz, *, ex):
    """GAD_EXCH_SOM(smTr, myNz, myThid)   @63cdc0b pkg/generic_advdiff/gad_exch_som.F:6-71

    C     | SUBROUTINE GAD_EXCH_SOM
    C     | o Apply exchanges to update overlaps of 1srt & 2nd.Order
    C     |   Moments array, corresponding to 1 tracer
    C     smTr   :: tracer 1rst & 2nd Order moments
    C     myNz   ::  3rd dimension of array to exchange

    `smTr` is a tuple of nSOM FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:myNz), all tiles: smTr[n-1] is the Fortran
    smTr(:,:,:,:,:,n), n = 1..9 = x, y, z, xx, yy, zz, xy, xz, yz. Returns the exchanged tuple. `ex` is the
    experiment's Exchanger (or ShardedExchanger); the exchanges act on the storage [tile, k, j, i].
    """
    for a in smTr:
        if a.data.shape[1] != myNz:
            raise ValueError(f"GAD_EXCH_SOM: {a.name} has {a.data.shape[1]} levels, myNz = {myNz}")
    s = [a.data for a in smTr]
#--   Apply exchanges to 1rst.O.Moments:
#-    Sx,Sy :                                                       # :170-173
    s[0], s[1] = ex.EXCH_UV_AGRID_3D_RL(s[0], s[1], True)
#-    Sz :                                                          # :175-177
    s[2] = ex.EXCH_3D_RL(s[2])
#--   Apply exchanges to 2nd.O.Moments:
#-    Sxx,Syy :                                                     # :181-184
    s[3], s[4] = ex.EXCH_UV_AGRID_3D_RL(s[3], s[4], False)
#-    Szz :                                                         # :186-188
    s[5] = ex.EXCH_3D_RL(s[5])
#-    Sxy :                                                         # :190-192
    s[6] = ex.EXCH_SM_3D_RL(s[6], True)
#-    Sxz,Syz :                                                     # :194-197
    s[7], s[8] = ex.EXCH_UV_AGRID_3D_RL(s[7], s[8], True)
    return tuple(_exchanged(a, d) for a, d in zip(smTr, s))
