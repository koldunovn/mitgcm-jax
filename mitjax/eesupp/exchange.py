"""Halo exchanges on one device, one gather per call from the probed maps (plan Task 7b; [E§7], L-COPY-2).

`Exchanger(maps)` applies the Fortran exchanges of an experiment (eesupp exch1 or pkg/exch2: one format,
eesupp/exch_maps.py) to `[tile, j, i]` or `[tile, k, j, i]` arrays holding ALL tiles. Every output point is
    keep (comp 0): its own value;  comp 1: the first input at `src`;  comp 2: the second input at `src`
times the map's sign (+-1, exact). A copy, so -0 and NaN pass unchanged and bit for bit; the JAX transpose of the
gather is the matching scatter-add (no hand-written adjoint).

Except the exch2 C-grid vector exchanges (EXCH_UV_XY_RL, EXCH_UV_XYZ_RL via exch_rs.py, EXCH_UV_3D_RL,
EXCH_UV_DGRID_3D_RL on a pkg/exch2 layout; `ExchangeMaps.rx2`, lane A's finding 2026-10-01): EXCH2_PUT_RX2 builds
every halo value as `sa1*A1(isl,jsl) + sa2*A2(isl,jsl)` (pkg/exch2/exch2_put_rx2.template:229-230, 317-318), so
`apply_rx2` evaluates exactly that per EXCH2_RX2_CUBE pass (two passes: corners ignored, then updated). Without
rotation the u halo is `1*u + 0*v`: a -0 becomes +0 where v there is >= +0, NaN where v is Inf/NaN. Transpose, per
pass and array c (derived; JAX obtains the same from the gathers and products): xbar_c = ybar_c on the points the pass
does not write, plus, for every written point p of either array c', sa_c(c')[p]*ybar_c'[p] added at src_c'[p] --
including the zero coefficients, which put 0*ybar into the other component's source points (a copy's transpose would
not); passes in reverse order. Gate: mitjax/tests/test_exch2_vector.py. The methods carry the Fortran names and arguments minus
`myThid` (and the `myNz` argument, which the array shape gives); a `withSigns` value the probe did not measure, or an
exchange it did not probe, is an error (unported option).

A pytree whose leaves are the integer map arrays, so the maps travel as jit ARGUMENTS (closed over, they become
compile-time constants and XLA constant-folds the transposes' scatter-adds; ECCO port: compile 64 s closed over vs
26 s as an argument). No float leaf: differentiating with respect to a pytree that holds an Exchanger gives float0
tangents for the maps.

Global reductions and sharding hooks share their names with `ShardedExchanger` (sharded_exchange.py), so a kernel
calls `ex.global_sum_tile(...)`, `ex.vary(...)` and runs unchanged at P=1 and P=N.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.eesupp import global_sum as gs
from mitjax.eesupp.exch_maps import CUBE_SCALAR, CUBE_VECTOR, SCALAR, VECTOR, ExchangeMaps


def _flat(a, n_points):
    """[T, ..., ny, nx] -> ([..., T*ny*nx], lead): the tile axis moved next to (j, i) and flattened with them."""
    a = jnp.moveaxis(jnp.asarray(a), 0, -3)
    lead = a.shape[:-3]
    if int(np.prod(a.shape[-3:])) != n_points:
        raise ValueError(f"array {a.shape} does not match the exchange layout ({n_points} points per level)")
    return a.reshape(*lead, n_points), lead


def _unflat(f, lead, shape3):
    return jnp.moveaxis(f.reshape(*lead, *shape3), -3, 0)


def apply_map(m, own, fa, fb):
    """One output of an exchange: own = the output's current values (kept where comp == 0), fa/fb = the first/second
    input (fb None for a scalar exchange), all [..., N]."""
    src, comp, sign = m
    out = jnp.where(comp == 1, fa[..., src], own)
    if fb is not None:
        out = jnp.where(comp == 2, fb[..., src], out)
    return out * sign.astype(out.dtype)


def apply_post(post, fu, fv):
    """The cube statements after the EXCH2_RX2_CUBE passes (corner fixes of exch2_uv_3d_rx.template:79-227, D-grid
    sign changes of exch2_uv_dgrid_3d_rx.template:91-196; pkg/exch2/exch2_cube_tables.py): per output array a gather
    from the passes' outputs (comp 1: u, 2: v, 0: keep), times its sign (the product of the `negOne` factors, +-1),
    negated where `neg` (`-vPhi(...)`, a negation, not a product)."""
    outs = []
    for c, own in ((1, fu), (2, fv)):
        src, comp, sign, neg = post[c]
        g = jnp.where(comp == 1, fu[..., src], jnp.where(comp == 2, fv[..., src], own)) * sign.astype(own.dtype)
        outs.append(jnp.where(neg, -g, g))
    return outs[0], outs[1]


def apply_rx2(passes, swap, fu, fv):
    """An exch2 C-grid vector exchange (EXCH2_UV_3D_RX / EXCH2_UV_DGRID_3D_RX, pkg/exch2/exch2_rx2_cube.py): per
    EXCH2_RX2_CUBE pass, every written point of array c becomes `sa1*A1[src] + sa2*A2[src]` from the arrays as they are
    before the pass (exch2_put_rx2.template:229-230, 317-318: two products and one sum in float64, that operand
    order, both arrays at the same source index), every other point keeps its value; swap: array1 is v (DGRID).
    fu, fv: [..., N]. Linear in (fu, fv): its transpose is derived by JAX from the gathers and products (see
    ExchangeMaps.rx2; mitjax/tests/test_exch2_vector.py gates it against the Fortran)."""
    a1, a2 = (fv, fu) if swap else (fu, fv)
    for p in passes:
        outs = []
        for c, own in ((1, a1), (2, a2)):
            src, written, sa1, sa2 = p[c]
            val = sa1.astype(own.dtype) * a1[..., src] + sa2.astype(own.dtype) * a2[..., src]
            outs.append(jnp.where(written, val, own))
        a1, a2 = outs
    return (a2, a1) if swap else (a1, a2)


def _check_signs(routine, withSigns, probed):
    if withSigns not in probed:
        raise NotImplementedError(f"{routine} with withSigns={withSigns} was not probed (jaxdump.F JAXDUMP_EXCH_PROBE)")


@jax.tree_util.register_pytree_node_class
class Exchanger:
    """Halo exchanges and global reductions on arrays holding every tile (single device)."""

    def __init__(self, maps: ExchangeMaps):
        self.layout = maps.layout
        self.m = {k: tuple(jnp.asarray(x) for x in v) for k, v in maps.maps.items()}
        # exch2 C-grid vector exchanges: kind -> (swap, [ {1: (src, written, sa1, sa2), 2: ...} per pass ], post)
        # with post None (single facet) or {1: (src, comp, sign, neg), 2: ...} (cube statements after the passes)
        self.rx2 = {}
        for k, entry in getattr(maps, "rx2", {}).items():
            swap, passes = entry[0], entry[1]
            post = entry[2] if len(entry) > 2 else None
            self.rx2[k] = (swap, [{c: (jnp.asarray(p[c][0], jnp.int32), jnp.asarray(p[c][1]),
                                       jnp.asarray(p[c][2], jnp.int8), jnp.asarray(p[c][3], jnp.int8))
                                   for c in (1, 2)} for p in passes],
                           None if post is None else {c: (jnp.asarray(post[c][0], jnp.int32),
                                                          jnp.asarray(post[c][1], jnp.int8),
                                                          jnp.asarray(post[c][2], jnp.int8),
                                                          jnp.asarray(post[c][3])) for c in (1, 2)})

    def tree_flatten(self):
        keys = tuple(sorted(self.m))
        rkeys = tuple(sorted(self.rx2))
        leaves = [self.m[k] for k in keys]
        rx2_aux = []
        for k in rkeys:
            swap, passes, post = self.rx2[k]
            rx2_aux.append((k, swap, len(passes), post is not None))
            leaves += [p[c] for p in passes for c in (1, 2)]
            if post is not None:
                leaves += [post[1], post[2]]
        return tuple(leaves), (keys, self.layout, tuple(rx2_aux))

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        keys, layout, rx2_aux = aux
        ex = object.__new__(cls)
        ex.layout = layout
        ex.m = dict(zip(keys, leaves[:len(keys)]))
        i = len(keys)
        ex.rx2 = {}
        for k, swap, npass, has_post in rx2_aux:
            passes = []
            for _ in range(npass):
                passes.append({1: leaves[i], 2: leaves[i + 1]})
                i += 2
            post = None
            if has_post:
                post = {1: leaves[i], 2: leaves[i + 1]}
                i += 2
            ex.rx2[k] = (swap, passes, post)
        return ex

    # ---------------------------------------------------------------- generic (map names of exch_maps.py)
    def scalar(self, name, phi):
        if name not in SCALAR and name not in CUBE_SCALAR:
            raise KeyError(f"{name} is not a scalar exchange")
        L = self.layout
        f, lead = _flat(phi, L.npoints)
        return _unflat(apply_map(self.m[name], f, f, None), lead, L.shape2d)

    def vector(self, name, u, v):
        if name not in VECTOR and name not in CUBE_VECTOR:
            raise KeyError(f"{name} is not a vector exchange")
        L = self.layout
        fu, lead = _flat(u, L.npoints)
        fv, lead_v = _flat(v, L.npoints)
        if lead != lead_v:
            raise ValueError(f"u and v shapes differ: {jnp.shape(u)} vs {jnp.shape(v)}")
        if name in self.rx2:
            swap, passes, post = self.rx2[name]
            uo, vo = apply_rx2(passes, swap, fu, fv)
            if post is not None:
                uo, vo = apply_post(post, uo, vo)
        else:
            uo = apply_map(self.m[name + "_u"], fu, fu, fv)
            vo = apply_map(self.m[name + "_v"], fv, fu, fv)
        return _unflat(uo, lead, L.shape2d), _unflat(vo, lead, L.shape2d)

    # ---------------------------------------------------------------- Fortran names (eesupp/src/exch_*_rx.template)
    def EXCH_XY_RL(self, phi):
        return self.scalar("XY", phi)

    def EXCH_3D_RL(self, phi):
        return self.scalar("3D", phi)

    def EXCH_Z_3D_RL(self, phi):
        return self.scalar("Z", phi)

    def EXCH_SM_3D_RL(self, phi, withSigns):
        _check_signs("EXCH_SM_3D_RL", withSigns, self._signs("SMs", "SMn"))
        return self.scalar("SMs" if withSigns else "SMn", phi)

    def EXCH_S3D_RL(self, phi):
        """EXCH_S3D_RL on the cg2d work arrays (0:sNx+1, 0:sNy+1), here held inside full [tile, j, i] arrays: only the
        width-1 ring is written (its 4 corners are not, except where the probe measured otherwise)."""
        return self.scalar("S3D", phi)

    def EXCH_UV_XY_RL(self, u, v, withSigns):
        return self.vector("UVs" if withSigns else "UVn", u, v)

    def EXCH_UV_AGRID_3D_RL(self, u, v, withSigns):
        return self.vector("As" if withSigns else "An", u, v)

    def EXCH_UV_BGRID_3D_RL(self, u, v, withSigns):
        return self.vector("Bs" if withSigns else "Bn", u, v)

    def EXCH_UV_DGRID_3D_RL(self, u, v, withSigns):
        _check_signs("EXCH_UV_DGRID_3D_RL", withSigns, self._signs("Ds", "Dn"))
        return self.vector("Ds" if withSigns else "Dn", u, v)

    def EXCH_UV_3D_RL(self, u, v, withSigns):
        _check_signs("EXCH_UV_3D_RL", withSigns, self._signs("UV3s", "UV3n"))
        return self.vector("UV3s" if withSigns else "UV3n", u, v)

    def _signs(self, ks, kn):
        """The withSigns values this exchanger has tables for (the probe measured only .TRUE. for some M1 kinds; the
        cube tables replay both)."""
        have = set(self.m) | set(self.rx2) | {k[:-2] for k in self.m if k[-2:] in ("_u", "_v")}
        return tuple(v for v, k in ((True, ks), (False, kn)) if k in have)

    # ---------------------------------------------------------------- global reductions (single device: all tiles)
    def all_tiles(self, per_tile):
        """[nTiles, ...] per-tile values -> the same (every tile is here, in tile order)."""
        return per_tile

    def global_sum_tile(self, phiTile):
        """GLOBAL_SUM_TILE_RL of per-tile partials [nTiles] (eesupp/global_sum.py)."""
        return gs.global_sum_tile(phiTile)

    def global_sum_rl(self, a):
        """Per-tile `DO j; DO i` partials of a [nTiles, nj, ni], then GLOBAL_SUM_TILE_RL."""
        return gs.global_sum_rl(a, self)

    def global_max(self, a):
        """_GLOBAL_MAX_RL over every tile of a [nTiles, ...] array (a max is order-free)."""
        return jnp.max(a)

    # ---------------------------------------------------------------- sharding hooks (identity on one device)
    def vary(self, x):
        """Typed as varying over the tile axis (sharded); the identity on one device."""
        return x

    def tile_index(self):
        """Global 0-based tile number of every tile held, or None for all tiles in order (eesupp/tiles.tile_rows)."""
        return None

    def real_tiles(self):
        """[T] bool: the tiles held that are real (sharded: not padding)."""
        return jnp.ones((self.layout.nTiles,), bool)

    def zero_padding(self, a):
        """Padding tiles set to 0 (sharded); the identity on one device."""
        return a

    def tile_mask(self, mask2d):
        """[ny, nx] bool -> [T, ny, nx] for the tiles held, False on padding tiles (sharded)."""
        return jnp.broadcast_to(jnp.asarray(mask2d), (self.layout.nTiles,) + tuple(np.shape(mask2d)))
