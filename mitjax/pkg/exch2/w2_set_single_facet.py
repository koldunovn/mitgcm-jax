"""W2_SET_SINGLE_FACET (pkg/exch2/w2_set_single_facet.F @63cdc0b): set up the simple single-facet (domain in one
piece) topology.

    @63cdc0b pkg/exch2/w2_set_single_facet.F:9-80

The facet links are REAL*4 literals (`1.2` etc., no D exponent): stored as float32, as the Fortran stores them.
The error branches (:60-66, :69-75) raise with the Fortran message.
"""

from mitjax.eesupp.print import SQUEEZE_RIGHT, internal_write, print_message
from mitjax.pkg.exch2.w2_exch2_h import W2_maxNbFacets, r4


def w2_set_single_facet(w2, *, cfg, io, myThid=1):
    """W2_SET_SINGLE_FACET( myThid ) on the W2Common `w2`."""
    sz = cfg.size
    msgBuf = internal_write("(2A,I3,A)", "W2_SET_SINGLE_FACET:", " preDefTopol=", w2.preDefTopol, " selected")  # :36-37
    print_message(msgBuf, w2.W2_oUnit, SQUEEZE_RIGHT, myThid, io=io)                                          # :38

    # --   Number of facets:
    w2.nFacets = 1                                                            # :41

    # --   Set Facet Edge connections (topology graph) ignoring any previous setting from data.exch2
    w2.facet_link[1, 1] = r4(1.2)                                             # :46  N -> face 1 S
    w2.facet_link[2, 1] = r4(1.1)                                             # :48  S -> face 1 N
    w2.facet_link[3, 1] = r4(1.4)                                             # :50  E -> face 1 W
    w2.facet_link[4, 1] = r4(1.3)                                             # :52  W -> face 1 E

    # --   Facet dimension: take the 1rst 2 numbers from facet_dims (if correct)
    if w2.facet_dims[1] == 0 and w2.facet_dims[2] == 0:                       # :55
        # -    Default: take global dimension from SIZE.h (will fail with blank tiles)
        w2.facet_dims[1] = sz.Nx                                              # :57
        w2.facet_dims[2] = sz.Ny                                              # :58
    if w2.facet_dims[1] <= 0 or w2.facet_dims[2] <= 0:                        # :60-66
        raise RuntimeError(f"W2_SET_SINGLE_FACET: unvalid 1rst 2 dimensions:{w2.facet_dims[1]:5d}"
                           f"{w2.facet_dims[2]:5d}; ABNORMAL END: S/R W2_SET_SINGLE_FACET: unvalid dims")
    for j in range(3, W2_maxNbFacets * 2 + 1):                                # :68-76
        if w2.facet_dims[j] != 0:
            raise RuntimeError("W2_SET_SINGLE_FACET: no more than 2 dims (X,Y) expected for single facet; "
                               "ABNORMAL END: S/R W2_SET_SINGLE_FACET: unexpected dims")
    return w2
