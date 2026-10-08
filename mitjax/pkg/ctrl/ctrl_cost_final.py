"""CTRL_COST_FINAL   @63cdc0b pkg/ctrl/ctrl_cost_final.F:9-234 (lane M4ADCOL)"""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import global_sum_tile
from mitjax.io.fortran_format import fortran_write


def ctrl_cost_final(cost, t2, a3, *, useCtrlCostContribution, maxCtrlTim2D, maxCtrlArr3D, mult_gentim2d,
                    mult_genarr3d, ex=None, a2=None, maxCtrlArr2D=0, mult_genarr2d=None):
    """CTRL_COST_FINAL( ifc, optimcycle, myThid )

    `t2`, `a3`: CTRL_COST_DRIVER's {iarr: (num, objf)} [tile]; the other slots are CTRL_INIT_VARIABLES' zeros.
    `mult_gentim2d` / `mult_genarr3d`: {iarr: value} for iarr = 1..maxCtrlTim2D / maxCtrlArr3D (data.ctrl or the
    default of ctrl_readparms.F:481-500). tile_fc(bi,bj) = tile_fc + mult_gentim2d(num_var)*objf_gentim2d (:91-99,
    num_var = 1..maxCtrlTim2D) then + mult_genarr3d(num_var)*objf_genarr3d (:113-121); f_ and no_ by
    _GLOBAL_SUM_RL of the per-tile sums (:70-73, :82-85, :95-98, :117-120, :128-130, :158-160).
    ALLOW_GENARR2D_CONTROL: `a2` {iarr: (num, objf)} (None when the option is undefined: no :101-111, :142-156);
    lane M4ADLAB (lab_sea/code_ad): + mult_genarr2d(num_var)*objf_genarr2d between the two (:101-111, per tile in
    that order). Returns (cost, {"tim2d": [(num_var, f, no)], "arr2d": [...], "arr3d": [...]}) with host-side
    printing in `ctrl_cost_final_lines`."""
    out = {"tim2d": [], "arr2d": [], "arr3d": []}
    if not useCtrlCostContribution:                                    # :66
        return cost, out
    nT = cost.tile_fc.shape[0]
    z = jnp.zeros((nT,), jnp.float64)
    tile_fc = cost.tile_fc
    f2 = {}
    for nv in range(1, maxCtrlTim2D + 1):                              # :91-99
        num, objf = t2.get(nv, (z, z))
        tile_fc = tile_fc + mult_gentim2d[nv] * objf
        f2[nv] = (objf, num)
    f22 = {}
    if a2 is not None:
        for nv in range(1, maxCtrlArr2D + 1):                          # :101-111
            num, objf = a2.get(nv, (z, z))
            tile_fc = tile_fc + mult_genarr2d[nv] * objf
            f22[nv] = (objf, num)
    f3 = {}
    for nv in range(1, maxCtrlArr3D + 1):                              # :113-121
        num, objf = a3.get(nv, (z, z))
        tile_fc = tile_fc + mult_genarr3d[nv] * objf
        f3[nv] = (objf, num)
    cost.tile_fc = tile_fc
    gs = global_sum_tile if ex is None else ex.global_sum_tile
    for nv in range(1, maxCtrlTim2D + 1):                              # :128-140
        out["tim2d"].append((nv, gs(f2[nv][0]), gs(f2[nv][1])))
    for nv in sorted(f22):                                             # :143-155
        out["arr2d"].append((nv, gs(f22[nv][0]), gs(f22[nv][1])))
    for nv in range(1, maxCtrlArr3D + 1):                              # :158-170
        out["arr3d"].append((nv, gs(f3[nv][0]), gs(f3[nv][1])))
    return cost, out


def ctrl_cost_final_lines(out, *, files_tim2d, files_arr3d, mult_gentim2d, mult_genarr3d, files_arr2d=None,
                          mult_genarr2d=None):
    """(STDOUT records without the PID prefix, costfunction_ctrl lines) of :131-139, :146-154 (genarr2d), :161-169
    and :185-223 (host)."""
    std, cf = [], []
    kinds = (("tim2d", files_tim2d, mult_gentim2d, "gentim2d"), ("arr2d", files_arr2d, mult_genarr2d, "genarr2d"),
             ("arr3d", files_arr3d, mult_genarr3d, "genarr3d"))
    for kind, files, mult, fmt_tag in kinds:
        for nv, f, no in out.get(kind, []):
            if float(no) > 0.0:                                        # :131 / :161
                name = files[nv].rstrip()                              # ILNBLNK
                std.append(fortran_write("(A,1PE22.14,I2,1X,1PE9.2,1X,3A)", f" --> f_{fmt_tag} =", float(f), nv,
                                         float(mult[nv]), "(", name, ")"))
    for kind, files, mult, fmt_tag in kinds:
        for nv, f, no in out.get(kind, []):
            if float(no) > 0.0:                                        # :186 / :214
                IL = max(len(files[nv].rstrip()), 15)   # :188-189 / :216-217  MINMAX-INT: INTEGER lengths
                cf.append(fortran_write("(2A,I2.0,A,1PE22.14,1PE22.14,1X,1PE9.2)", files[nv].ljust(IL)[:IL],
                                        f" ({fmt_tag} ", nv, ") = ", float(f), float(no), float(mult[nv])))
    return std, cf
