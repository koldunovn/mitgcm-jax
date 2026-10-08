"""EXF_SWAPFFIELDS: pkg/exf/exf_swapffields.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def exf_SwapFFields(ffld0, ffld1, *, cfg):
    """EXF_SWAPFFIELDS( ffld0, ffld1, myThid )   @63cdc0b pkg/exf/exf_swapffields.F:3-67

    C     | o Copy a forcing field (the next record) into the previous record; zero the next record.

    Returns (ffld0, ffld1)."""
    sz = cfg.size
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    ffld0 = ffld0.at[i, j].set(ffld1[i, j])                                    # :59
    ffld1 = ffld1.at[i, j].set(0.)                                             # :60
    return ffld0, ffld1
