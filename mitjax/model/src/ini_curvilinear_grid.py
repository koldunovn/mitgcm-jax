"""INI_CURVILINEAR_GRID: model/src/ini_curvilinear_grid.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.eesupp import exch_rs as X
from mitjax.farray import FArray
from mitjax.model.src.calc_grid_angles import calc_grid_angles
from mitjax.pkg.mdsio.mdsio_facef_read import PRECFLOAT64, mds_facef_read_rs

# the records of a grid file in the order INI_CURVILINEAR_GRID reads them (ini_curvilinear_grid.F:294-339)
RECORDS = ("xC", "yC", "dxF", "dyF", "rA", "xG", "yG", "dxV", "dyU", "rAz", "dxC", "dyC", "rAw", "rAs", "dxG", "dyG")
ANGLE_RECORDS = ("angleCosC", "angleSinC")           # records 17, 18 (:342-351), only with horizGridFile
# the fields the radius rescaling multiplies by tmpFac (lengths) and tmpFac2 (areas), in the Fortran order (:390-413)
LENGTHS = ("dxC", "dyC", "dxG", "dyG", "dxF", "dyF", "dxV", "dyU")
AREAS = ("rA", "rAz", "rAw", "rAs")


def grid_file_name(horizGridFile, iG):
    """fName of facet iG (ini_curvilinear_grid.F:278-284): 'tile<iG as I3.3>.mitgrid' without horizGridFile, else
    horizGridFile(1:iLen)//'.face'//<iG as I3.3>//'.bin'."""
    iLen = len(horizGridFile.rstrip(" "))                                     # ILNBLNK(horizGridFile)
    if iLen == 0:
        return f"tile{iG:03d}.mitgrid"
    return f"{horizGridFile[:iLen]}.face{iG:03d}.bin"


def ini_curvilinear_grid(grid, *, cfg, params, ex, rw):
    """INI_CURVILINEAR_GRID( myThid )   @63cdc0b model/src/ini_curvilinear_grid.F:8-451

    C     | o Initialise curvilinear coordinate system
    C     | Curvilinear grid settings are read from a file rather than coded in-line as for cartesian and
    C     | spherical-polar.

    The branch the cubed-sphere experiments compile and run: OLD_GRID_IO undefined (:73-205 not compiled), no MNC grid
    read (readgrid_mnc = .FALSE., :209-245 skipped; .TRUE. raises), ALLOW_EXCH2 and ALLOW_MDSIO: per tile, the facet
    file of facet exch2_myFace(W2_myTileList(bi,bj)) (`grid_file_name`) is read with MDS_FACEF_READ_RS
    (fp = precFloat64, :252) into the points 1..sNx+1, 1..sNy+1 of the 16 fields (records 1-16) and, with
    horizGridFile, of angleCosC, angleSinC (records 17, 18; anglesAreSet = .TRUE., else .FALSE., :340-354). Then the
    exchanges of :374-384 in that order, the rescaling from radius_fromHorizGrid to rSphere (:390-413, products in
    the Fortran order), CALC_GRID_ANGLES( anglesAreSet ) (:417) and EXCH_UV_AGRID_3D_RS( angleSinC, angleCosC,
    .TRUE. ) (:420). The PRINT_MESSAGE lines of the read loop (STDOUT only, no model variable) and the
    plotLevel >= debLevC field plots (:429-448) are not ported. Host-side reads (numpy), placed as jax arrays.

    `ex`: the exchanger of the cube (mitjax/eesupp/exch_maps.load_cube_maps); `rw`: the run directory
    (mitjax/pkg/rw/read_rec.RW: files are opened relative to it, as the Fortran opens them in the run directory).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    try:
        old_io = bool(cfg.cpp.OLD_GRID_IO)
    except KeyError:
        old_io = False
    if old_io:
        raise NotImplementedError("INI_CURVILINEAR_GRID: OLD_GRID_IO is not ported")
    if params.useMNC and params.readgrid_mnc:                                 # :209
        raise NotImplementedError("INI_CURVILINEAR_GRID: readgrid_mnc (MNC grid read) is not ported")
    if not cfg.cpp.ALLOW_EXCH2:
        raise NotImplementedError("INI_CURVILINEAR_GRID: the non-exch2 file naming (:270-274) is not ported")
    if not cfg.cpp.ALLOW_MDSIO:
        raise RuntimeError("INI_CURVILINEAR_GRID: Needs to compile MDSIO pkg; ABNORMAL END: S/R INI_CURVILINEAR_GRID")
    ex2 = params.exch2
    if ex2 is None or not ex2.exch2_myFace:
        raise ValueError("INI_CURVILINEAR_GRID: needs the W2 topology with exch2_myFace (w2_eeboot.exch2_topology)")
    fp = PRECFLOAT64                                                          # :252  fp = precFloat64

    names = RECORDS + ANGLE_RECORDS
    store = {n: np.array(getattr(grid, n).data, dtype=np.float64) for n in names}
    anglesAreSet = False
    for bj in range(1, sz.nSy + 1):                                           # :261-366
        for bi in range(1, sz.nSx + 1):
            t = (bj - 1) * sz.nSx + bi                                        # storage tile of (bi,bj)
            jG = ex2.W2_myTileList[t - 1]                                     # :265
            iG = ex2.exch2_myFace[jG - 1]                                     # :266
            fName = grid_file_name(params.horizGridFile, iG)                  # :278-284
            path = rw.path(fName)
            geo = dict(sNx=sNx, sNy=sNy, OLx=OLx, OLy=OLy, dNx=ex2.exch2_mydNx[jG - 1], dNy=ex2.exch2_mydNy[jG - 1],
                       tBx=ex2.exch2_tBasex[jG - 1], tBy=ex2.exch2_tBasey[jG - 1])
            for irec, n in enumerate(RECORDS, start=1):                       # :294-339
                store[n][t - 1] = mds_facef_read_rs(path, fp, irec, store[n][t - 1], **geo)
            if len(params.horizGridFile.rstrip(" ")) > 0:                     # :340-354
                for irec, n in enumerate(ANGLE_RECORDS, start=17):
                    store[n][t - 1] = mds_facef_read_rs(path, fp, irec, store[n][t - 1], **geo)
                anglesAreSet = True
            else:
                anglesAreSet = False
    f = {n: FArray(jnp.asarray(store[n]), n, tiled=getattr(grid, n).tiled, _dims=getattr(grid, n).dims)
         for n in names}

    # exchanges (:374-384), in the Fortran order
    f["xC"] = X.EXCH_XY_RS(f["xC"], ex=ex)
    f["yC"] = X.EXCH_XY_RS(f["yC"], ex=ex)
    f["dxF"], f["dyF"] = X.EXCH_UV_AGRID_3D_RS(f["dxF"], f["dyF"], False, 1, ex=ex)
    f["rA"] = X.EXCH_XY_RS(f["rA"], ex=ex)
    f["xG"] = X.EXCH_Z_3D_RS(f["xG"], 1, ex=ex)
    f["yG"] = X.EXCH_Z_3D_RS(f["yG"], 1, ex=ex)
    f["dxV"], f["dyU"] = X.EXCH_UV_BGRID_3D_RS(f["dxV"], f["dyU"], False, 1, ex=ex)
    f["rAz"] = X.EXCH_Z_3D_RS(f["rAz"], 1, ex=ex)
    f["dxC"], f["dyC"] = X.EXCH_UV_XY_RS(f["dxC"], f["dyC"], False, ex=ex)
    f["rAw"], f["rAs"] = X.EXCH_UV_XY_RS(f["rAw"], f["rAs"], False, ex=ex)
    f["dyG"], f["dxG"] = X.EXCH_UV_XY_RS(f["dyG"], f["dxG"], False, ex=ex)

    # -    Rescale from the radius of the input grid to rSphere (:390-413; whole arrays, 1-OLx:sNx+OLx x 1-OLy:sNy+OLy)
    if params.rSphere != params.radius_fromHorizGrid:
        tmpFac = params.rSphere / params.radius_fromHorizGrid                 # :391
        tmpFac2 = tmpFac*tmpFac                                               # :392
        for n in LENGTHS:
            f[n] = FArray(f[n].data*tmpFac, n, tiled=f[n].tiled, _dims=f[n].dims)
        for n in AREAS:
            f[n] = FArray(f[n].data*tmpFac2, n, tiled=f[n].tiled, _dims=f[n].dims)

    grid = grid.replace(**{n: f[n] for n in names})
    grid = calc_grid_angles(grid, anglesAreSet, cfg=cfg, params=params)      # :417
    sinC, cosC = X.EXCH_UV_AGRID_3D_RS(grid.angleSinC, grid.angleCosC, True, 1, ex=ex)   # :420
    return grid.replace(angleSinC=sinC, angleCosC=cosC)
