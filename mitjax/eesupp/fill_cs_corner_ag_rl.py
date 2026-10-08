"""FILL_CS_CORNER_AG_RL (eesupp/src/fill_cs_corner_ag_rl.F @63cdc0b): fill the corner-halo regions of an A-grid vector
(u, v at cell centres) across the cube's facet corners, for a 1-D operation in X (fill4dirX) or Y.

    @63cdc0b eesupp/src/fill_cs_corner_ag_rl.F:9-244

    uFld, vFld = fill_cs_corner_ag_rl(fill4dirX, withSigns, uFld, vFld, corners, useCubedSphereExchange, ...)

Arrays [T, ..., ny, nx], corners [T, 4] bool (fill_cs_corner_tr_rl.cs_corner_flags). negOne = 1. or -1. (:68-69);
the products are written `fld*negOne` (:133-232). One gather from the inputs equals the in-place loops (no statement
reads a written point; checked).
"""

from mitjax.eesupp.fill_cs_corner_tr_rl import apply_corner_statements


def ag_corner_statements(fill4dirX, withSigns, sNx, sNy, OLx, OLy):
    """{corner: [(dst, (j0, i0), src, (j0, i0), mul)]} of FILL_CS_CORNER_AG_RL (fill_cs_corner_ag_rl.F:126-235)."""
    negOne = 1.0                                                              # :68  negOne = 1.
    if withSigns:
        negOne = -1.0                                                         # :69
    m = ("r", negOne)

    def P(i, j):
        return (j - 1 + OLy, i - 1 + OLx)
    st = {0: [], 1: [], 2: [], 3: []}
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            if fill4dirX:
                st[0] += [("u", P(1 - i, 1 - j), "v", P(1 - j, i), m),                        # :133
                          ("v", P(1 - i, 1 - j), "u", P(1 - j, i), None)]                     # :134
                st[1] += [("u", P(sNx + i, 1 - j), "v", P(sNx + j, i), None),                 # :141
                          ("v", P(sNx + i, 1 - j), "u", P(sNx + j, i), m)]                    # :142
                st[2] += [("u", P(1 - i, sNy + j), "v", P(1 - j, sNy + 1 - i), None),         # :149
                          ("v", P(1 - i, sNy + j), "u", P(1 - j, sNy + 1 - i), m)]            # :150
                st[3] += [("u", P(sNx + i, sNy + j), "v", P(sNx + j, sNy + 1 - i), m),        # :157
                          ("v", P(sNx + i, sNy + j), "u", P(sNx + j, sNy + 1 - i), None)]     # :158
            else:
                st[0] += [("u", P(1 - i, 1 - j), "v", P(j, 1 - i), None),                    # :207
                          ("v", P(1 - i, 1 - j), "u", P(j, 1 - i), m)]                        # :208
                st[1] += [("u", P(sNx + i, 1 - j), "v", P(sNx + 1 - j, 1 - i), m),            # :215
                          ("v", P(sNx + i, 1 - j), "u", P(sNx + 1 - j, 1 - i), None)]         # :216
                st[2] += [("u", P(1 - i, sNy + j), "v", P(j, sNy + i), m),                    # :223
                          ("v", P(1 - i, sNy + j), "u", P(j, sNy + i), None)]                 # :224
                st[3] += [("u", P(sNx + i, sNy + j), "v", P(sNx + 1 - j, sNy + i), None),     # :231
                          ("v", P(sNx + i, sNy + j), "u", P(sNx + 1 - j, sNy + i), m)]        # :232
    return st


def fill_cs_corner_ag_rl(fill4dirX, withSigns, uFld, vFld, corners, useCubedSphereExchange, *, sNx, sNy, OLx, OLy):
    """FILL_CS_CORNER_AG_RL( fill4dirX, withSigns, uFld, vFld, bi, bj, myThid ) on every tile."""
    if not useCubedSphereExchange:                                            # :71
        return uFld, vFld
    out = apply_corner_statements({"u": uFld, "v": vFld}, corners,
                                  ag_corner_statements(fill4dirX, withSigns, sNx, sNy, OLx, OLy))
    return out["u"], out["v"]
