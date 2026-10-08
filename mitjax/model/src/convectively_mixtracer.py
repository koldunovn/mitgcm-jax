"""CONVECTIVELY_MIXTRACER: model/src/convectively_mixtracer.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def convectively_mixtracer(k, weightA, weightB, Tracer, *, cfg):
    """CONVECTIVELY_MIXTRACER( bi, bj, k, weightA, weightB, Tracer, myThid )
    @63cdc0b model/src/convectively_mixtracer.F:6-58

    C Mixes a tracer over two layers according to the weights pre-calculated
    C as a function of stability.
    C Mixing is represented by:
    C                       T(k-1) = T(k-1) + A * ( T(k) - T(k-1) )
    C                       T(k)   = T(k)   + B * ( T(k-1) - T(k) )
    C     weightA :: weight for level k-1
    C     weightB :: weight for level  k

    k: Python int. weightA, weightB: (1-OLx:sNx+OLx,1-OLy:sNy+OLy); Tracer: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) of all
    tiles. Returns Tracer with levels k-1 and k mixed. Each point reads and writes only its own column (delTrac from
    both levels before either is written, :46): vectorised over (i, j)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :43
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :44
    delTrac = Tracer[i, j, k]-Tracer[i, j, k-1]                                 # :46
    Tracer = Tracer.at[i, j, k-1].set(Tracer[i, j, k-1]                         # :47-48
                                      + weightA[i, j]*delTrac)
    Tracer = Tracer.at[i, j, k].set(Tracer[i, j, k]                             # :49-50
                                    - weightB[i, j]*delTrac)
    return Tracer
