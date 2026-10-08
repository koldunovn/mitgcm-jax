"""FILL_CS_CORNER_UV_RS (eesupp/src/fill_cs_corner_uv_rs.F @63cdc0b, :9-186): FILL_CS_CORNER_UV_RL's statements on
_RS arrays (Real*8 in these builds: no -use_real4), see fill_cs_corner_uv_rl.py."""

from mitjax.eesupp.fill_cs_corner_uv_rl import fill_cs_corner_uv_rl


def fill_cs_corner_uv_rs(withSigns, uFld, vFld, corners, useCubedSphereExchange, *, sNx, sNy, OLx, OLy):
    """FILL_CS_CORNER_UV_RS( withSigns, uFld, vFld, bi, bj, myThid ) on every tile."""
    return fill_cs_corner_uv_rl(withSigns, uFld, vFld, corners, useCubedSphereExchange, sNx=sNx, sNy=sNy, OLx=OLx,
                                OLy=OLy)
