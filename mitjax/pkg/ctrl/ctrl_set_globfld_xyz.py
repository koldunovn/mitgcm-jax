"""CTRL_SET_GLOBFLD_XYZ: pkg/ctrl/ctrl_set_globfld_xyz.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad)."""

import numpy as np


def ctrl_set_globfld_xyz(nRecArg, *, sz):
    """CTRL_SET_GLOBFLD_XYZ( fname, nRecArg, filePrec, myThid )   @63cdc0b pkg/ctrl/ctrl_set_globfld_xyz.F:3-60

    The first-guess control file CTRL_INIT_CTRLVAR writes with doInitXX at optimcycle 0 (ctrl_init_ctrlvar.F:140-142):
    globfld3d = 0. _d 0 at every point (:41-51), written as nRecArg records of Nr levels (:53-57). Here the file is the
    control array (the JAX control vector convention of mitjax/pkg/ctrl): returns the nRecArg records as numpy
    [tile, Nr, j, i] arrays with halos (only the interior goes to the file)."""
    shape = (sz.nSx * sz.nSy, sz.Nr, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)
    return [np.zeros(shape) for _ in range(nRecArg)]                # :46  0. _d 0


def ctrl_set_globfld_xy(nRecArg, *, sz):
    """CTRL_SET_GLOBFLD_XY( fname, nRecArg, filePrec, myThid )   @63cdc0b pkg/ctrl/ctrl_set_globfld_xy.F:3-77

    As CTRL_SET_GLOBFLD_XYZ for a 2-D control (ctrl_init_ctrlvar.F:149-151): globfld2d / globfld3d = 0. _d 0 (:43-62);
    INT(nRecArg/Nr) records of Nr levels, then the remaining records of one level (:63-74). Returns the records in
    file order: numpy [tile, Nr, j, i] for the Nr-level records, [tile, 1, j, i] for the one-level records."""
    nrec_nl = nRecArg // sz.Nr                                      # :63  INT(nRecArg/Nr)
    ny, nx, nt = sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx, sz.nSx * sz.nSy
    return ([np.zeros((nt, sz.Nr, ny, nx)) for _ in range(nrec_nl)]                       # :64-68
            + [np.zeros((nt, 1, ny, nx)) for _ in range(nrec_nl * sz.Nr + 1, nRecArg + 1)])   # :69-73
