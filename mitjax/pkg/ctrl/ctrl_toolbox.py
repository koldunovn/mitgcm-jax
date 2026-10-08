"""CTRL_CPRSRS   @63cdc0b pkg/ctrl/ctrl_toolbox.F:154-197 (and the 2-D declaration helper of pkg/ctrl)"""

from mitjax.farray import FArray, loop_i, loop_j


def xy(data, name, sz):
    """An `_RL/_RS name(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)` FArray (all tiles)."""
    return FArray(data, name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))


def ctrl_cprsrs(fldIn, nzIn, fldOut, nzOut, *, sz):
    """CTRL_CPRSRS( fldIn, nzIn, fldOut, nzOut, myThid )

    C     copy a field (RS) to another array (RS)

    Here nzOut = 1 (the only call of this experiment, CTRL_GET_MASK2D): fldIn is level 1 of a 3-D FArray
    (k=1..nzIn), fldOut a 2-D FArray. Loops :180-190 (DO k = 1,nzOut; all points incl. halos)."""
    if nzOut != 1:
        raise NotImplementedError("CTRL_CPRSRS with nzOut > 1 is not ported")
    j = loop_j(1 - sz.OLy, sz.sNy + sz.OLy)                          # :182
    i = loop_i(1 - sz.OLx, sz.sNx + sz.OLx)                          # :183
    return fldOut.at[i, j].set(fldIn[i, j, 1])                       # :184
