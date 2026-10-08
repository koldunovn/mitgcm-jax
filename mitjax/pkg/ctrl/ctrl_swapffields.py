"""CTRL_SWAPFFIELDS   @63cdc0b pkg/ctrl/ctrl_swapffields.F:12-75"""

from mitjax.farray import loop_i, loop_j


def ctrl_swapffields(ffld0, ffld1, *, sz):
    """CTRL_SWAPFFIELDS( ffld0, ffld1, myThid )

    ffld0 = ffld1 and ffld1 = 0 on the interior (:55-72); returns (ffld0, ffld1)."""
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    ffld0 = ffld0.at[i, j].set(ffld1[i, j])                            # :59
    ffld1 = ffld1.at[i, j].set(0.0)                                    # :69  0. _d 0
    return ffld0, ffld1
