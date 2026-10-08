"""Gather tables of the eesupp (exch1) halo exchanges, replayed on the host from the Fortran loops (docs plan 20261006
Task 3; the exch1 counterpart of mitjax/pkg/exch2/exch2_cube_tables.py).

    @63cdc0b eesupp/src/exch1_rx.template:170-273 (FORWARD_SIMULATION), exch_rx_send_put_x.template,
             exch_rx_recv_get_x.template, exch_rx_send_put_y.template, exch_rx_recv_get_y.template,
             ini_communication_patterns.F:99-306 (tile neighbours and communication modes),
             exch_xy_rx.template:57-74, exch_3d_rx.template:56-73, exch_z_3d_rx.template, exch_sm_3d_rx.template,
             exch_s3d_rx.template:57-74, exch_uv_xy_rx.template, exch_uv_3d_rx.template,
             exch_uv_agrid_3d_rx.template, exch_uv_bgrid_3d_rx.template, exch_uv_dgrid_3d_rx.template;
             DISCONNECTED_TILES: exch0_rx.template:66-89 (EXCH0_RX), :101-238 (FILL_HALO_LOCAL_RX)

Without pkg/exch2 and without useCubedSphereExchange every exchange routine of mitjax/eesupp/exch_maps.py calls
EXCH1_RX once per array: with myOLw = myOLe = exchWidthX = OLx, myOLs = myOLn = exchWidthY = OLy and
EXCH_UPDATE_CORNERS, except EXCH_S3D_RL (arrays (0:sNx+1, 0:sNy+1): widths 1, EXCH_IGNORE_CORNERS). The vector
routines exchange u and v separately (no sign: negOne multiplies only in their cube branches), so every map is a
copy from the array's own input.

One process, one thread (the port's runs): INI_COMMUNICATION_PATTERNS makes the tile grid periodic in both directions
(west of bi = 1 is bi = nSx, south of bj = 1 is bj = nSy) with every neighbour in the same process, so every
communication mode is COMM_PUT (:263-306). EXCH1_RX then runs, in order (exch1_rx.template:170-200):
SEND_PUT_X, [RECV_GET_X if EXCH_UPDATE_CORNERS], SEND_PUT_Y, [RECV_GET_X otherwise], RECV_GET_Y, and the Nx = 1 /
Ny = 1 copies (:236-273).
With DISCONNECTED_TILES (CPP_EEOPTIONS.h; vermix/code) the same wrappers call EXCH0_RX with the same arguments: each
tile fills its own halo from its own interior, locally periodic (FILL_HALO_LOCAL_RX). A SEND_PUT fills the receiving tile's buffer from the array as it is when the pass starts
(the _BARRIER between the sends and the receives), in the (k, j, i) order the RECV_GET reads it back.

Symbolic replay with mitjax.pkg.exch2.exch2_cube_tables.Sym: every point holds (comp, src, sign, neg); a copy
statement copies the tuple. Not ported (raise): useCubedSphereExchange without pkg/exch2 (EXCH1_RX_CUBE,
EXCH1_UV_RX_CUBE, EXCH1_BG_RX_CUBE, EXCH1_Z_RX_CUBE), nPx*nPy > 1.
"""

from mitjax.pkg.exch2.exch2_cube_tables import Sym

# exch_maps kind -> (myOL / exchange width: "OL" (OLx, OLy) or 1, cornerMode update?) ; vector kinds: both arrays
EXCH1_SCALAR = {"XY": ("OL", True), "3D": ("OL", True), "Z": ("OL", True), "SMs": ("OL", True),
                "S3D": (1, False)}
EXCH1_VECTOR = ("UVs", "UVn", "As", "An", "Bs", "Bn", "Ds", "UV3s")


def exch1_rx(sym, c, sz, myOLx, myOLy, exchWidthX, exchWidthY, update_corners):
    """EXCH1_RX( array_c, myOLx, myOLx, myOLy, myOLy, 1, exchWidthX, exchWidthY, cornerMode ) on the symbolic array c,
    FORWARD_SIMULATION (exch1_rx.template:170-200, 236-273). myOLx, myOLy: the array's own halo widths (OLx, OLy;
    the (0:sNx+1, 0:sNy+1) arrays of EXCH_S3D_RL: 1, 1); `sz` the build's Size."""
    sNx, sNy, nSx, nSy = sz.sNx, sz.sNy, sz.nSx, sz.nSy
    myOLw = myOLe = myOLx
    myOLs = myOLn = myOLy

    def tile(bi, bj):                                   # tileNo with nPx = nPy = 1 (ini_communication_patterns.F)
        return (bj - 1) * nSx + bi

    def west(bi):                                       # :110-118
        return nSx if bi - 1 < 1 else bi - 1

    def east(bi):                                       # :137-145
        return 1 if bi + 1 > nSx else bi + 1

    def north(bj):                                      # :166-174
        return 1 if bj + 1 > nSy else bj + 1

    def south(bj):                                      # :188-196
        return nSy if bj - 1 < 1 else bj - 1

    def send_put_x():
        """EXCH_RX_SEND_PUT_X, COMM_PUT (exch_rx_send_put_x.template): the west interior columns into the west
        neighbour's eastRecvBuf, the east ones into the east neighbour's westRecvBuf."""
        east_recv, west_recv = {}, {}
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                east_recv[(west(bi), bj)] = [sym.get(c, t, i, j) for j in range(1, sNy + 1)
                                             for i in range(1, 1 + exchWidthX - 1 + 1)]
                west_recv[(east(bi), bj)] = [sym.get(c, t, i, j) for j in range(1, sNy + 1)
                                             for i in range(sNx - exchWidthX + 1, sNx + 1)]
        return east_recv, west_recv

    def recv_get_x(bufs):
        """EXCH_RX_RECV_GET_X, COMM_PUT (exch_rx_recv_get_x.template): east halo columns sNx+1..sNx+exchWidthX, then
        west halo columns 1-exchWidthX..0, rows 1..sNy."""
        east_recv, west_recv = bufs
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                buf = iter(east_recv[(bi, bj)])
                for j in range(1, sNy + 1):
                    for i in range(sNx + 1, sNx + exchWidthX + 1):
                        sym.put(c, t, i, j, next(buf))
                buf = iter(west_recv[(bi, bj)])
                for j in range(1, sNy + 1):
                    for i in range(1 - exchWidthX, 0 + 1):
                        sym.put(c, t, i, j, next(buf))

    def i_range():
        if update_corners:                              # send_put_y / recv_get_y: EXCH_UPDATE_CORNERS
            return range(1 - exchWidthX, sNx + exchWidthX + 1)
        return range(1, sNx + 1)

    def send_put_y():
        """EXCH_RX_SEND_PUT_Y, COMM_PUT (exch_rx_send_put_y.template): the south interior rows into the south
        neighbour's northRecvBuf, the north ones into the north neighbour's southRecvBuf."""
        north_recv, south_recv = {}, {}
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                north_recv[(bi, south(bj))] = [sym.get(c, t, i, j) for j in range(1, 1 + exchWidthY - 1 + 1)
                                               for i in i_range()]
                south_recv[(bi, north(bj))] = [sym.get(c, t, i, j) for j in range(sNy - exchWidthY + 1, sNy + 1)
                                               for i in i_range()]
        return north_recv, south_recv

    def recv_get_y(bufs):
        """EXCH_RX_RECV_GET_Y, COMM_PUT (exch_rx_recv_get_y.template): north halo rows sNy+1..sNy+exchWidthY, then
        south halo rows 1-exchWidthY..0."""
        north_recv, south_recv = bufs
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                buf = iter(north_recv[(bi, bj)])
                for j in range(sNy + 1, sNy + exchWidthY + 1):
                    for i in i_range():
                        sym.put(c, t, i, j, next(buf))
                buf = iter(south_recv[(bi, bj)])
                for j in range(1 - exchWidthY, 0 + 1):
                    for i in i_range():
                        sym.put(c, t, i, j, next(buf))

    bx = send_put_x()                                   # exch1_rx.template:174-177
    if update_corners:                                  # :178-183
        recv_get_x(bx)
    by = send_put_y()                                   # :184-187
    if not update_corners:                              # :191-196
        recv_get_x(bx)
    recv_get_y(by)                                      # :197-200
    if sz.Nx == 1:                                      # :238-254
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                for j in range(1 - myOLs, sNy + myOLn + 1):
                    for i in range(1 - myOLw, sNx + myOLe + 1):
                        sym.copy(c, t, i, j, c, 1, j)
    if sz.Ny == 1:                                      # :255-271
        for bj in range(1, nSy + 1):
            for bi in range(1, nSx + 1):
                t = tile(bi, bj)
                for j in range(1 - myOLs, sNy + myOLn + 1):
                    for i in range(1 - myOLw, sNx + myOLe + 1):
                        sym.copy(c, t, i, j, c, i, 1)


def exch0_rx(sym, c, sz, myOLx, myOLy, update_corners):
    """EXCH0_RX( array_c, myOLx, myOLx, myOLy, myOLy, 1, exchWidthX, exchWidthY, cornerMode ) under
    DISCONNECTED_TILES (exch0_rx.template:79-87): FILL_HALO_LOCAL_RX on every tile (:101-238); the exchange widths
    are only checked (:66-77), the halo widths decide."""
    sNx, sNy = sz.sNx, sz.sNy
    myOLw = myOLe = myOLx
    myOLs = myOLn = myOLy
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            t = (bj - 1) * sz.nSx + bi

            def cp(i, j, i2, j2):
                sym.copy(c, t, i, j, c, i2, j2)
            if update_corners:                                                 # :148-158
                iMin, iMax, jMin, jMax = 1 - myOLw, sNx + myOLe, 1 - myOLs, sNy + myOLn
            else:
                iMin, iMax, jMin, jMax = 1, sNx, 1, sNy
            if sNx == 1:                                                       # :161-169
                for j in range(jMin, jMax + 1):
                    for i in range(1 - myOLw, sNx + myOLe + 1):
                        cp(i, j, 1, j)
            elif sNx < myOLw:                                                  # :170-183
                for j in range(jMin, jMax + 1):
                    for i in range(0, 1 - myOLw - 1, -1):
                        cp(i, j, i + sNx, j)
                    for i in range(1, myOLe + 1):
                        cp(i + sNx, j, i, j)
            else:                                                              # :184-194
                for j in range(jMin, jMax + 1):
                    for i in range(1 - myOLw, 0 + 1):
                        cp(i, j, i + sNx, j)
                    for i in range(1, myOLe + 1):
                        cp(i + sNx, j, i, j)
            if sNy == 1:                                                       # :198-206
                for j in range(1 - myOLs, sNy + myOLn + 1):
                    for i in range(iMin, iMax + 1):
                        cp(i, j, i, 1)
            elif sNy < myOLs:                                                  # :207-222
                for j in range(0, 1 - myOLs - 1, -1):
                    for i in range(iMin, iMax + 1):
                        cp(i, j, i, j + sNy)
                for j in range(1, myOLn + 1):
                    for i in range(iMin, iMax + 1):
                        cp(i, j + sNy, i, j)
            else:                                                              # :223-235
                for j in range(1 - myOLs, 0 + 1):
                    for i in range(iMin, iMax + 1):
                        cp(i, j, i, j + sNy)
                for j in range(1, myOLn + 1):
                    for i in range(iMin, iMax + 1):
                        cp(i, j + sNy, i, j)


def exch1_exchange_maps(sz, layout, cpp=None, useCubedSphereExchange=False):
    """{key: (src int32, comp int8, sign int8)} of every exchange kind the probe measures (exch_maps.SCALAR, VECTOR)
    on an exch1 build: `sz` its Size (SIZE.h), `layout` its TileLayout."""
    if useCubedSphereExchange:
        raise NotImplementedError("exch1 with useCubedSphereExchange (EXCH1_RX_CUBE, EXCH1_UV_RX_CUBE, "
                                  "EXCH1_BG_RX_CUBE, EXCH1_Z_RX_CUBE) is not ported")
    disconnected = cpp is not None and "DISCONNECTED_TILES" in cpp.known and bool(cpp.DISCONNECTED_TILES)

    def rx(s, c, olx, oly, wx, wy, update):           # exch_*_rx.template: EXCH0_RX or EXCH1_RX, same arguments
        if disconnected:
            exch0_rx(s, c, sz, olx, oly, update)
        else:
            exch1_rx(s, c, sz, olx, oly, wx, wy, update)
    if sz.nPx * sz.nPy != 1:
        raise NotImplementedError(f"exch1 with nPx*nPy = {sz.nPx * sz.nPy} processes is not ported")
    if (layout.sNx, layout.sNy, layout.OLx, layout.OLy, layout.nTiles) != (sz.sNx, sz.sNy, sz.OLx, sz.OLy,
                                                                          sz.nSx * sz.nSy):
        raise ValueError(f"layout {layout} does not match SIZE.h {sz}")
    maps = {}
    for kind, (ol, update) in EXCH1_SCALAR.items():
        s = Sym(layout, 1)
        if ol == "OL":                                   # myOLw = myOLe = exchWidthX = OLx, ... = OLy
            rx(s, 1, sz.OLx, sz.OLy, sz.OLx, sz.OLy, update)
        else:                                            # EXCH_S3D_RL: OLw = ... = exchWidthY = 1
            rx(s, 1, 1, 1, 1, 1, update)
        src, comp, sign, neg = s.table(1, 1)
        assert not neg.any()
        maps[kind] = (src, comp, sign)
    for kind in EXCH1_VECTOR:
        s = Sym(layout, 2)
        for c in (1, 2):                                 # EXCH1_RX( uPhi ), then EXCH1_RX( vPhi )
            rx(s, c, sz.OLx, sz.OLy, sz.OLx, sz.OLy, True)
        for c, suf in ((1, "_u"), (2, "_v")):
            src, comp, sign, neg = s.table(c, c)
            assert not neg.any()
            maps[kind + suf] = (src, comp, sign)
    return maps
