"""CTRL_INIT_FIXED (the unit weight it writes): pkg/ctrl/ctrl_init_fixed.F @63cdc0b (GOADK lane, M2:
global_ocean.90x40x15/code_ad, whose genarr controls name `wunit.data` as their weight)."""

import numpy as np


def ctrl_init_fixed_wunit(*, sz):
    """CTRL_INIT_FIXED( myThid ) :92-110 @63cdc0b pkg/ctrl/ctrl_init_fixed.F

    C     Set unit weight to 1

    wunit(k,bi,bj) = 1. _d 0 and loctmp3d = 1. _d 0 at every point (:93-104), written to the file `wunit`
    (`wunit.data`) by ACTIVE_WRITE_XYZ (:106-107, ALLOW_AUTODIFF) at initialisation; a weight file named `wunit.data`
    in data.ctrl is therefore this record. Returns it as numpy [tile, Nr, j, i] (the file holds the interior). The rest
    of CTRL_INIT_FIXED (the control registration, ctrl_init_ctrlvar.py; the wet counts, grdchk.py) is ported
    separately."""
    return np.ones((sz.nSx * sz.nSy, sz.Nr, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))     # :99  1. _d 0
