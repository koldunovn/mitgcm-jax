"""FILL_CS_CORNER_TR_RL (eesupp/src/fill_cs_corner_tr_rl.F @63cdc0b): fill the corner-halo regions of a tracer field
on the cubed sphere so that a 1-D advection in X (fill4dir = 1) or Y (fill4dir = 2) sees the right neighbour values
across a facet corner; fill4dir = 0 zeros them.

    @63cdc0b eesupp/src/fill_cs_corner_tr_rl.F:9-270

    trFld = fill_cs_corner_tr_rl(fill4dir, withSigns, trFld, corners, useCubedSphereExchange)

trFld: [T, ..., 1-OLy:sNy+OLy, 1-OLx:sNx+OLx] (every tile at once; a model level axis may sit between tile and j).
corners: [T, 4] bool, the facet corners of each tile (`cs_corner_flags`: SW, SE, NW, NE as the Fortran derives them
from exch2_isW/S/E/Nedge of W2_myTileList(bi,bj), :75-82), a host table (sharded: the local block's rows).
fill4dir and withSigns are static (Python). Within a corner the statements read only points outside that corner and
outside the other corners (every source has one index in 1..sN), so the Fortran's in-place loops equal one gather
from the input: a corner value is `negOne*trFld(source)` (negOne = -1. or 1., :68-69, as written: a product), the
rest is unchanged. Linear: JAX's transpose of the gathers and products is the adjoint.
"""

import itertools

import jax.numpy as jnp
import numpy as np


def cs_corner_flags(w2):
    """[nTiles, 4] bool (SW, SE, NW, NE) of the W2 topology, tile order W2_myTileList(bi,bj) (bi fastest):
    southWestCorner = exch2_isWedge.EQ.1 .AND. exch2_isSedge.EQ.1 etc. (fill_cs_corner_tr_rl.F:75-82)."""
    sz = w2.size
    out = []
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            t = w2.W2_myTileList[bi, bj]
            W, S, E, N = (w2.exch2_isWedge[t] == 1, w2.exch2_isSedge[t] == 1, w2.exch2_isEedge[t] == 1,
                          w2.exch2_isNedge[t] == 1)
            out.append((W and S, E and S, W and N, E and N))
    return np.array(out, bool)


def _corner_ij(sNx, sNy, OLx, OLy, corner, fill4dir):
    """[(target (j, i), source (j, i))] in 0-based padded indices for one corner, as the Fortran loops write them."""
    def P(i, j):                                                              # Fortran (i,j) -> padded (j0, i0)
        return (j - 1 + OLy, i - 1 + OLx)
    pairs = []
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            if fill4dir == 1:
                tgt, src = {"SW": ((1 - i, 1 - j), (1 - j, i)),                           # :165
                            "SE": ((sNx + i, 1 - j), (sNx + j, i)),                       # :172
                            "NW": ((1 - i, sNy + j), (1 - j, sNy + 1 - i)),               # :179
                            "NE": ((sNx + i, sNy + j), (sNx + j, sNy + 1 - i))}[corner]  # :186
            elif fill4dir == 2:
                tgt, src = {"SW": ((1 - i, 1 - j), (j, 1 - i)),                           # :235
                            "SE": ((sNx + i, 1 - j), (sNx + 1 - j, 1 - i)),               # :242
                            "NW": ((1 - i, sNy + j), (j, sNy + i)),                       # :249
                            "NE": ((sNx + i, sNy + j), (sNx + 1 - j, sNy + i))}[corner]  # :256
            else:
                tgt = {"SW": (1 - i, 1 - j), "SE": (sNx + i, 1 - j), "NW": (1 - i, sNy + j),
                       "NE": (sNx + i, sNy + j)}[corner]                                  # :96-117
                src = None
            pairs.append((P(*tgt), None if src is None else P(*src)))
    return pairs


def apply_corner_statements(arrays, corners, statements):
    """Apply per-corner statements `{corner k: [(dst, (j0, i0), src, (j0, i0) or None, mul), ...]}` to the arrays
    (dict name -> [T, ..., ny, nx]): dst(target) = src(source) * mul (mul a Python float, or None for a plain copy;
    a None source writes 0.). Every source is read from the INPUT arrays: the callers' statements never read a point
    any statement of the routine writes (checked here), so this equals the Fortran's in-place loops. Tiles whose
    corner flag is False keep their values."""
    written = {(d, t) for st in statements.values() for d, t, _, _, _ in st}
    for st in statements.values():
        for _, _, s_, q, _ in st:
            if s_ is not None and q is not None and (s_, q) in written:
                raise AssertionError(f"corner statement reads a written point {s_}{q}")
    a0 = {k: jnp.asarray(v) for k, v in arrays.items()}
    out = dict(a0)
    some = next(iter(a0.values()))
    lead = (some.shape[0],) + (1,) * (some.ndim - 3)
    corners = jnp.asarray(corners)
    for k, st in statements.items():
        flag = corners[:, k].reshape(lead + (1,))
        for name in a0:
            mine = [x for x in st if x[0] == name]
            if not mine:
                continue
            tj = np.array([x[1][0] for x in mine])
            ti = np.array([x[1][1] for x in mine])
            # GO lane session 10: consecutive statements with the same source array and product are read with one
            # gather (each element of a gather is the element read alone; the product is elementwise: the same
            # values) instead of one slice per point -- the per-point reads were a third of the cs32x15 step's
            # equations (gradient program 378 065 -> 282 497 ops, compile 437-476 -> 306 s, gradient bitwise: jobs
            # 27843047/48 -> 27843193)
            new = []
            for (s_, mul), grp in itertools.groupby(mine, key=lambda x: (x[2], x[4])):
                grp = list(grp)
                if s_ is None:
                    new.append(jnp.zeros(some.shape[:-2] + (len(grp),), some.dtype))
                else:
                    g = a0[s_][..., np.array([x[3][0] for x in grp]), np.array([x[3][1] for x in grp])]
                    new.append(g if mul is None else _mul(g, mul))
            new = new[0] if len(new) == 1 else jnp.concatenate(new, axis=-1)
            cur = out[name][..., tj, ti]
            out[name] = out[name].at[..., tj, ti].set(jnp.where(flag, new, cur))
    return out


def _mul(g, mul):
    """The Fortran product as written: ('l', c) is c*x (negOne*fld), ('r', c) is x*c (fld*negOne)."""
    side, c = mul
    return c * g if side == "l" else g * c


def fill_cs_corner_tr_rl(fill4dir, withSigns, trFld, corners, useCubedSphereExchange, *, sNx, sNy, OLx, OLy):
    """FILL_CS_CORNER_TR_RL( fill4dir, withSigns, trFld, bi, bj, myThid ) on every tile of trFld."""
    if fill4dir not in (0, 1, 2):
        raise ValueError("FILL_CS_CORNER_TR_RL: fill4dir has illegal value")   # :263 STOP
    negOne = 1.0                                                              # :68  negOne = 1.
    if withSigns:
        negOne = -1.0                                                         # :69
    if not useCubedSphereExchange:                                            # :71
        return trFld
    statements = {}
    for k, corner in enumerate(("SW", "SE", "NW", "NE")):
        statements[k] = [("tr", tgt, None if src is None else "tr", src, ("l", negOne))   # negOne*trFld( source )
                         for tgt, src in _corner_ij(sNx, sNy, OLx, OLy, corner, fill4dir)]  # or 0. _d 0
    return apply_corner_statements({"tr": trFld}, corners, statements)["tr"]
