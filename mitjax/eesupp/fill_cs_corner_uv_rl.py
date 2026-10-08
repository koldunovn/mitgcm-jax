"""FILL_CS_CORNER_UV_RL / FILL_CS_CORNER_UV_RS (eesupp/src/fill_cs_corner_uv_rl.F, fill_cs_corner_uv_rs.F @63cdc0b:
the same statements on _RL / _RS arrays, both Real*8 in these builds): fill the corner-halo regions of a C-grid
vector (u at west faces, v at south faces) across the cube's facet corners.

    @63cdc0b eesupp/src/fill_cs_corner_uv_rl.F:9-188 (fill_cs_corner_uv_rs.F:9-186 identical but for the types)

    uFld, vFld = fill_cs_corner_uv_rl(withSigns, uFld, vFld, corners, useCubedSphereExchange, sNx=..., ...)

Arrays [T, ..., ny, nx] (every tile), corners [T, 4] bool (SW, SE, NW, NE; fill_cs_corner_tr_rl.cs_corner_flags).
negOne is set only inside IF (useCubedSphereExchange) (:67-70); SW and NE multiply (`negOne*vFld(...)`), SE and NW
copy (:139-166). No statement reads a point any statement writes (checked), so one gather from the inputs equals the
Fortran's in-place loops.
"""

from mitjax.eesupp.fill_cs_corner_tr_rl import apply_corner_statements


def _P(i, j, OLx, OLy):
    return (j - 1 + OLy, i - 1 + OLx)


def uv_corner_statements(withSigns, sNx, sNy, OLx, OLy):
    """{corner: [(dst, (j0, i0), src, (j0, i0), mul)]} of FILL_CS_CORNER_UV_RL (fill_cs_corner_uv_rl.F:122-182)."""
    negOne = 1.0                                                              # :69  negOne = 1. _d 0
    if withSigns:
        negOne = -1.0                                                         # :70
    P = lambda i, j: _P(i, j, OLx, OLy)                                       # noqa: E731
    st = {0: [], 1: [], 2: [], 3: []}
    # southWestCorner (:122-137)
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            st[0].append(("u", P(1 - i, 1 - j), "v", P(1 - j, 1 + i), ("l", negOne)))   # :128
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            st[0].append(("v", P(1 - i, 1 - j), "u", P(1 + j, 1 - i), ("l", negOne)))   # :134
    # southEastCorner (:138-152)
    for j in range(1, OLy + 1):
        for i in range(2, OLx + 1):
            st[1].append(("u", P(sNx + i, 1 - j), "v", P(sNx + j, i), None))          # :143
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            st[1].append(("v", P(sNx + i, 1 - j), "u", P(sNx + 1 - j, 1 - i), None))  # :149
    # northWestCorner (:153-167)
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            st[2].append(("u", P(1 - i, sNy + j), "v", P(1 - j, sNy + 1 - i), None))  # :158
    for j in range(2, OLy + 1):
        for i in range(1, OLx + 1):
            st[2].append(("v", P(1 - i, sNy + j), "u", P(j, sNy + i), None))          # :164
    # northEastCorner (:168-182)
    for j in range(1, OLy + 1):
        for i in range(2, OLx + 1):
            st[3].append(("u", P(sNx + i, sNy + j), "v", P(sNx + j, sNy + 2 - i), ("l", negOne)))   # :173
    for j in range(2, OLy + 1):
        for i in range(1, OLx + 1):
            st[3].append(("v", P(sNx + i, sNy + j), "u", P(sNx + 2 - j, sNy + i), ("l", negOne)))   # :179
    return st


def fill_cs_corner_uv_rl(withSigns, uFld, vFld, corners, useCubedSphereExchange, *, sNx, sNy, OLx, OLy):
    """FILL_CS_CORNER_UV_RL( withSigns, uFld, vFld, bi, bj, myThid ) on every tile."""
    if not useCubedSphereExchange:                                            # :67
        return uFld, vFld
    out = apply_corner_statements({"u": uFld, "v": vFld}, corners,
                                  uv_corner_statements(withSigns, sNx, sNy, OLx, OLy))
    return out["u"], out["v"]
