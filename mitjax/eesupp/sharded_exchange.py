"""Halo exchanges and Fortran-order global sums on tiles sharded over devices (plan Task 7b; [E§7], [E§11], L-COPY-3).

`ShardedExchanger` has the methods of the single-device `Exchanger` (exchange.py: the Fortran-named exchanges,
global_sum_tile, global_max, vary, ...) and is used INSIDE `jax.shard_map(..., check_vma=True)` over the mesh axis
`AXIS`, on the local block of every [tile, ..., j, i] array. It is a pytree: its per-device tables are leaves with a
leading device axis (shard them with PartitionSpec(AXIS); `device_arrays`); the round structure is static.

Tile blocks (`TileBlocks`): the nTiles tiles (experiment order, eesupp/tiles.py) are split into P contiguous blocks of
Tloc = ceil(nTiles/P); device d holds padded positions d*Tloc .. d*Tloc+Tloc-1; Tpad = P*Tloc. A padding tile is a
REPLICA of tile 1 (the donor; L-PAR-6): it gets tile 1's data and tables and tile 1's halo sources, so it computes
bit for bit what tile 1 computes (finite wherever tile 1 is), a global max is unchanged, and nothing reads it (every
map source is a real tile; global sums drop it). Its outputs are dropped. Gradients: a padding tile's cotangent must
be 0 (`zero_padding`), because the exchange transpose would carry it onto tile 1's halo sources; solves zero the
padding (`tile_mask`: the cg2d rule's `interior` is all False there), since the padded operator is not symmetric.

Exchanges: the maps are the single-device ones (exch_maps.py). The host classifies every output point's source as
local (same device: a gather) or remote; remote requests are collected per (sender, receiver) pair and the pairs are
greedily coloured into rounds in which every device sends to at most one device and receives from at most one: one
`lax.ppermute` per round ([L-PAR-8]). A vector exchange (u and v) shares one set of rounds. On a device an exchange
is one gather of the send buffer, K ppermutes, one concatenation, one gather, `where(the Fortran does not write the
point, own value, gathered)`, times the sign: the single-device formula with a larger source buffer, so the same
copies with the same signs, bitwise equal to `Exchanger` for every P. Transpose: gather -> scatter-add, ppermute ->
inverse ppermute; JAX derives both. `ragged_all_to_all` is banned (its transpose is wrong, [L-PAR-7];
test_banned_transforms.py).

Global sums: eesupp/global_sum.py (`all_tiles`: psum of the zero-padded block vector, then the ordered chain).
`global_max` gathers per-tile maxima the same way (`lax.pmax` has no JVP) and is invariant across the mesh axis.

Sharding types ([L-PAR-10]): JAX 0.10.1 re-traces `custom_linear_solve` at lowering and fails on an invariant ->
varying pvary inside it, so values a solve closes over (cg2d: `interior`, `cg2dNorm`) are made varying first
(`vary`, a `pcast`). Copied from the ECCO port's `parallel/sharded_exchange.py`; changed: map names of exch_maps.py,
integer sign tables, `tile_mask`, `tile_index`.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitjax.eesupp import global_sum as gs
from mitjax.eesupp.exch_maps import CUBE_SCALAR, CUBE_VECTOR, SCALAR, VECTOR, output_names
from mitjax.eesupp.exchange import Exchanger

AXIS = "tile"


@dataclass(frozen=True)
class TileBlocks:
    """Contiguous tile blocks of P devices (tile t, 0-based, experiment order, lives on device t // Tloc)."""
    nTiles: int
    P: int
    donor: int = 0

    @property
    def Tloc(self):
        return -(-self.nTiles // self.P)

    @property
    def Tpad(self):
        return self.P * self.Tloc

    @property
    def source_tile(self):
        """[Tpad] int32: the real tile whose data, tables and halo sources every padded position carries."""
        return np.array(list(range(self.nTiles)) + [self.donor] * (self.Tpad - self.nTiles), np.int32)

    def pad(self, a):
        """[nTiles, ...] -> [Tpad, ...] (padding tiles = copies of the donor tile)."""
        a = np.asarray(a)
        if a.shape[0] != self.nTiles:
            raise ValueError(f"leading axis {a.shape[0]} is not the tile axis ({self.nTiles})")
        return a[self.source_tile]

    def unpad(self, a):
        return a[:self.nTiles]


# ---------------------------------------------------------------------------------------------------------------------
# host-side plan


def colour_pairs(pairs, P):
    """Directed device pairs (sender, receiver) -> rounds, each a partial permutation (every device sends at most
    once and receives at most once per round). Greedy in the order of the cyclic shift (receiver - sender) mod P, so a
    complete neighbour graph takes P-1 rounds (one per shift)."""
    rounds = []
    for e, d in sorted(pairs, key=lambda ed: ((ed[1] - ed[0]) % P, ed[0])):
        for r in rounds:
            if e not in r[1] and d not in r[2]:
                r[0].append((e, d))
                r[1].add(e)
                r[2].add(d)
                break
        else:
            rounds.append(([(e, d)], {e}, {d}))
    return [tuple(r[0]) for r in rounds]


@dataclass(frozen=True)
class KindPlan:
    """Static round structure of one exchange (hashable: pytree aux data)."""
    outputs: tuple      # map keys of the outputs: ("XY",) or ("UVs_u", "UVs_v")
    ncomp: int          # source arrays in the local buffer (1 scalar, 2 vector: [u | v])
    perms: tuple        # per round: ((sender, receiver), ...)
    slots: tuple        # per round: values per message
    offs: tuple         # per round: offset in the send buffer


def build_kind(maps, outputs, blocks, layout, ncomp=None):
    """Plan and per-device tables of one exchange. maps: ExchangeMaps.maps; outputs: the map keys of the outputs;
    ncomp: source arrays in the buffer (default: one per output).

    Returns (KindPlan, tables) with tables (numpy, leading device axis P):
      loc [P, nout, N] int32   index into the receiver's source vector [own buffer (ncomp*N) | round 0 | round 1 ...]
      keep [P, nout, N] bool   the Fortran exchange does not write the point (keeps its own value)
      sign [P, nout, N] int8   sign of the copy (+1 where kept)
      send_idx [P, S] int32    index into the sender's own buffer of every value it sends (S = max(total, 1))
      send_ok [P, S] bool      slot holds a requested value (else padding of a shorter message, sent as 0)
    """
    L = layout
    n = L.ny * L.nx
    P, Tloc = blocks.P, blocks.Tloc
    N = Tloc * n
    ncomp = len(outputs) if ncomp is None else ncomp
    st = blocks.source_tile
    nout = len(outputs)
    loc = np.zeros((P, nout, N), np.int64)
    keep = np.zeros((P, nout, N), bool)
    sign = np.ones((P, nout, N), np.int8)
    remote = []
    req = {}                                 # (sender, receiver) -> requested buffer indices
    for o, name in enumerate(outputs):
        src, comp, sgn = maps[name]
        src = np.asarray(src, np.int64).reshape(blocks.nTiles, n)
        comp = np.asarray(comp, np.int64).reshape(blocks.nTiles, n)
        sgn = np.asarray(sgn, np.int8).reshape(blocks.nTiles, n)
        for d in range(P):
            tiles = st[d * Tloc:(d + 1) * Tloc]      # map rows of the local positions (padding: the donor's)
            s, c, g = src[tiles].reshape(-1), comp[tiles].reshape(-1), sgn[tiles].reshape(-1)
            ts, p = s // n, s % n                     # source tile (real) and point
            if np.any(ts >= blocks.nTiles):
                raise ValueError(f"{name}: a map source outside the real tiles")
            e = ts // Tloc                            # device holding the source tile
            bidx = (np.maximum(c, 1) - 1) * N + (ts - e * Tloc) * n + p   # index in e's [u | v] buffer
            w = c > 0
            keep[d, o] = ~w
            sign[d, o] = g
            lo = w & (e == d)
            loc[d, o, lo] = bidx[lo]
            rm = w & (e != d)
            remote.append((d, o, rm, e, bidx))
            for ee in np.unique(e[rm]):
                req.setdefault((int(ee), d), []).append(bidx[rm & (e == ee)])
    req = {k: np.unique(np.concatenate(v)) for k, v in req.items()}
    rounds = colour_pairs(list(req), P)
    slots = tuple(int(max(len(req[pr]) for pr in r)) for r in rounds)
    offs = tuple(int(x) for x in np.concatenate([[0], np.cumsum(slots)])[:-1]) if slots else ()
    round_of = {pr: k for k, r in enumerate(rounds) for pr in r}
    S = max(int(sum(slots)), 1)
    send_idx = np.zeros((P, S), np.int64)
    send_ok = np.zeros((P, S), bool)
    for (e, d), idx in req.items():
        k = round_of[(e, d)]
        send_idx[e, offs[k]:offs[k] + len(idx)] = idx
        send_ok[e, offs[k]:offs[k] + len(idx)] = True
    for d, o, rm, e, bidx in remote:
        for ee in np.unique(e[rm]):
            m = rm & (e == ee)
            k = round_of[(int(ee), d)]
            loc[d, o, m] = ncomp * N + offs[k] + np.searchsorted(req[(int(ee), d)], bidx[m])
    plan = KindPlan(tuple(outputs), ncomp, tuple(tuple(r) for r in rounds), slots, offs)
    tabs = dict(loc=loc.astype(np.int32), keep=keep, sign=sign, send_idx=send_idx.astype(np.int32),
                send_ok=send_ok)
    return plan, tabs


def build_rx2_pass(tabs_p, blocks, layout):
    """Plan and per-device tables of one EXCH2_RX2_CUBE pass (pkg/exch2/exch2_rx2_cube.py tables): four gathers from
    the [A1 | A2] buffer -- A1 and A2 at the source index of array 1's points, then at array 2's -- with `keep` where
    the pass does not write, plus `sa` [P, 2 (array), 2 (term), N] int8: the coefficients sa1, sa2 of the local
    points."""
    n = layout.ny * layout.nx
    synth, names = {}, []
    for c in (1, 2):
        src, written, _, _ = tabs_p[c]
        one = np.ones(len(src), np.int8)
        for term in (1, 2):
            key = f"c{c}a{term}"
            synth[key] = (src, np.where(written, term, 0).astype(np.int8), one)
            names.append(key)
    plan, tab = build_kind(synth, tuple(names), blocks, layout, ncomp=2)
    st = blocks.source_tile
    sa = np.zeros((blocks.P, 2, 2, blocks.Tloc * n), np.int8)
    for c in (1, 2):
        for term in (1, 2):
            full = np.asarray(tabs_p[c][1 + term], np.int8).reshape(blocks.nTiles, n)
            for d in range(blocks.P):
                sa[d, c - 1, term - 1] = full[st[d * blocks.Tloc:(d + 1) * blocks.Tloc]].reshape(-1)
    tab["sa"] = sa
    return plan, tab


def build_post(post, blocks, layout):
    """Plan and per-device tables of the cube statements after the EXCH2_RX2_CUBE passes (exchange.apply_post): a
    copy kind over the [u | v] outputs of the passes (build_kind) plus `neg` [P, 2, N] bool (the negations)."""
    n = layout.ny * layout.nx
    synth = {"post_u": post[1][:3], "post_v": post[2][:3]}
    plan, tab = build_kind(synth, ("post_u", "post_v"), blocks, layout, ncomp=2)
    st = blocks.source_tile
    neg = np.zeros((blocks.P, 2, blocks.Tloc * n), bool)
    for c in (1, 2):
        full = np.asarray(post[c][3], bool).reshape(blocks.nTiles, n)
        for d in range(blocks.P):
            neg[d, c - 1] = full[st[d * blocks.Tloc:(d + 1) * blocks.Tloc]].reshape(-1)
    tab["neg"] = neg
    return plan, tab


# ---------------------------------------------------------------------------------------------------------------------
# device side


@jax.tree_util.register_pytree_node_class
class ShardedExchanger(Exchanger):
    """Halo exchanges and global reductions on the local tile block, inside shard_map over AXIS.

    Build on the host with `ShardedExchanger.build(maps, blocks)`, place with `device_arrays(mesh)`, pass it into the
    shard_map'd function with in_specs PartitionSpec(AXIS) for all its leaves. The Fortran-named methods are
    inherited from Exchanger and call `scalar` / `vector` below."""

    def __init__(self, layout, blocks, plans, tabs, axis_name=AXIS, rx2=None):
        self.layout = layout             # the global layout (index ranges and map size only)
        self.blocks = blocks
        self.plans = plans               # map name (exch2 C-grid vector kinds: "<kind>#<pass>") -> KindPlan
        self.tabs = tabs                 # map name -> dict of arrays [P or 1, ...]
        self.axis_name = axis_name
        self.rx2 = dict(rx2 or {})       # exch2 C-grid vector kind -> (swap, number of passes, has post stage)

    @classmethod
    def build(cls, maps, blocks, names=None, axis_name=AXIS):
        """maps: exch_maps.ExchangeMaps; blocks: TileBlocks; names: the exchanges to prepare (default: all)."""
        L = maps.layout
        if L.nTiles != blocks.nTiles:
            raise ValueError("maps and tile blocks disagree on the number of tiles")
        if names is None:
            names = [k for k in list(SCALAR) + list(VECTOR) + list(CUBE_SCALAR) + list(CUBE_VECTOR)
                     if k in getattr(maps, "rx2", {}) or all(o in maps.maps for o in output_names(k))]
        plans, tabs, rx2 = {}, {}, {}
        for k in names:
            if k in getattr(maps, "rx2", {}):
                entry = maps.rx2[k]
                swap, passes = entry[0], entry[1]
                post = entry[2] if len(entry) > 2 else None
                for p, tabs_p in enumerate(passes):
                    plans[f"{k}#{p}"], tabs[f"{k}#{p}"] = build_rx2_pass(tabs_p, blocks, L)
                if post is not None:
                    plans[f"{k}#post"], tabs[f"{k}#post"] = build_post(post, blocks, L)
                rx2[k] = (swap, len(passes), post is not None)
            else:
                plans[k], tabs[k] = build_kind(maps.maps, output_names(k), blocks, L)
        return cls(L, blocks, plans, tabs, axis_name, rx2)

    def tree_flatten(self):
        names = tuple(sorted(self.tabs))
        fields = tuple(tuple(sorted(self.tabs[k])) for k in names)
        leaves = tuple(self.tabs[k][nm] for k, fs in zip(names, fields) for nm in fs)
        aux = (self.layout, self.blocks, tuple((k, self.plans[k], fs) for k, fs in zip(names, fields)),
               self.axis_name, tuple(sorted(self.rx2.items())))
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        layout, blocks, plans, axis_name, rx2 = aux
        tabs, i = {}, 0
        for k, _, fs in plans:
            tabs[k] = dict(zip(fs, leaves[i:i + len(fs)]))
            i += len(fs)
        return cls(layout, blocks, {k: pl for k, pl, _ in plans}, tabs, axis_name, dict(rx2))

    def device_arrays(self, mesh):
        """The same exchanger with its tables placed on `mesh`, sharded over the device axis."""
        from jax.sharding import NamedSharding, PartitionSpec
        sh = NamedSharding(mesh, PartitionSpec(self.axis_name))
        return jax.tree.map(lambda a: jax.device_put(a, sh), self)

    # ---------------------------------------------------------------- exchanges (inside shard_map)
    def _flat(self, a):
        a = jnp.moveaxis(jnp.asarray(a), 0, -3)
        lead, T = a.shape[:-3], a.shape[-3]
        if T != self.blocks.Tloc or a.shape[-2:] != (self.layout.ny, self.layout.nx):
            raise ValueError(f"local block {a.shape} (tile axis moved) does not match Tloc={self.blocks.Tloc}, "
                             f"(ny, nx)={(self.layout.ny, self.layout.nx)}")
        return a.reshape(*lead, T * self.layout.ny * self.layout.nx), lead

    def _unflat(self, f, lead):
        return jnp.moveaxis(f.reshape(*lead, self.blocks.Tloc, self.layout.ny, self.layout.nx), -3, 0)

    def _local(self, name):
        if self.tabs[name]["loc"].shape[0] != 1:
            raise ValueError("ShardedExchanger works inside shard_map (tables sharded over the device axis)")
        return self.plans[name], {k: v[0] for k, v in self.tabs[name].items()}     # local block: [1, ...] -> [...]

    def _sources(self, plan, tab, own):
        """The receiver's source vector [own buffer | round 0 | round 1 ...] (ppermute rounds of the plan)."""
        buf = own[0] if len(own) == 1 else jnp.concatenate(own, axis=-1)
        parts = [buf]
        if plan.perms:
            send = jnp.where(tab["send_ok"], buf[..., tab["send_idx"]], jnp.zeros((), buf.dtype))
            for perm, s, o in zip(plan.perms, plan.slots, plan.offs):
                parts.append(lax.ppermute(send[..., o:o + s], self.axis_name, perm=list(perm)))
        return (jnp.concatenate(parts, axis=-1) if len(parts) > 1 else buf), buf

    def _exchange_rx2(self, kind, u, v):
        """An exch2 C-grid vector exchange (exchange.apply_rx2) on the local block: per EXCH2_RX2_CUBE pass, the
        gathers of A1 and A2 at each array's source points (local or received), then sa1*A1 + sa2*A2 where the pass
        writes."""
        swap, npass, has_post = self.rx2[kind]
        a1, a2 = (v, u) if swap else (u, v)
        for p in range(npass):
            plan, tab = self._local(f"{kind}#{p}")
            (f1, lead), (f2, _) = self._flat(a1), self._flat(a2)
            src, buf = self._sources(plan, tab, [f1, f2])
            g = [src[..., tab["loc"][o]] for o in range(4)]          # c1a1, c1a2, c2a1, c2a2
            sa = tab["sa"].astype(buf.dtype)
            o1 = jnp.where(tab["keep"][0], f1, sa[0, 0] * g[0] + sa[0, 1] * g[1])
            o2 = jnp.where(tab["keep"][2], f2, sa[1, 0] * g[2] + sa[1, 1] * g[3])
            a1, a2 = self._unflat(o1, lead), self._unflat(o2, lead)
        uo, vo = (a2, a1) if swap else (a1, a2)
        if has_post:
            plan, tab = self._local(f"{kind}#post")
            (fu, lead), (fv, _) = self._flat(uo), self._flat(vo)
            src, buf = self._sources(plan, tab, [fu, fv])
            outs = []
            for o, own in ((0, fu), (1, fv)):
                g = jnp.where(tab["keep"][o], own, src[..., tab["loc"][o]]) * tab["sign"][o].astype(buf.dtype)
                outs.append(self._unflat(jnp.where(tab["neg"][o], -g, g), lead))
            uo, vo = outs
        return uo, vo

    def _exchange(self, name, arrays):
        plan, tab = self._local(name)
        flats = [self._flat(a) for a in arrays]
        lead = flats[0][1]
        own = [f for f, _ in flats]
        src, buf = self._sources(plan, tab, own)
        outs = []
        for o in range(len(plan.outputs)):
            out = jnp.where(tab["keep"][o], own[o], src[..., tab["loc"][o]]) * tab["sign"][o].astype(buf.dtype)
            outs.append(self._unflat(out, lead))
        return outs

    def scalar(self, name, phi):
        if name not in SCALAR and name not in CUBE_SCALAR:
            raise KeyError(f"{name} is not a scalar exchange")
        return self._exchange(name, [phi])[0]

    def _signs(self, ks, kn):
        have = set(self.plans) | set(self.rx2)
        return tuple(v for v, k in ((True, ks), (False, kn)) if k in have)

    def vector(self, name, u, v):
        if name not in VECTOR and name not in CUBE_VECTOR:
            raise KeyError(f"{name} is not a vector exchange")
        if name in self.rx2:
            return self._exchange_rx2(name, u, v)
        uo, vo = self._exchange(name, [u, v])
        return uo, vo

    # ---------------------------------------------------------------- global reductions (inside shard_map)
    def all_tiles(self, per_tile):
        """[Tloc, ...] values of the local tiles -> [nTiles, ...] of every real tile, tile order, on every device."""
        return gs.all_tiles(per_tile, self.blocks, self.axis_name)

    def global_sum_tile(self, phiTile):
        """GLOBAL_SUM_TILE_RL: local per-tile partials [Tloc] -> 0 + tile 1 + ... + tile nTiles (every P)."""
        return gs.global_sum_tile(self.all_tiles(phiTile))

    def global_max(self, a):
        """_GLOBAL_MAX_RL over every real tile of a local [Tloc, ...] array."""
        a = jnp.asarray(a)
        return jnp.max(self.all_tiles(jnp.max(a.reshape(a.shape[0], -1), axis=1)))

    # ---------------------------------------------------------------- sharding types and padding (inside shard_map)
    def vary(self, x):
        """x (pytree) typed as varying over the tile axis; values unchanged ([L-PAR-10])."""
        ax = self.axis_name

        def one(a):
            varying = getattr(getattr(jax.typeof(a), "mat", None), "varying", frozenset())
            return a if ax in varying else lax.pcast(a, ax, to="varying")
        return jax.tree.map(one, x)

    def tile_index(self):
        """[Tloc] int32: the global 0-based tile number of every local position (padding: the donor's)."""
        st = jnp.asarray(self.blocks.source_tile)
        return lax.dynamic_slice_in_dim(st, lax.axis_index(self.axis_name) * self.blocks.Tloc, self.blocks.Tloc)

    def real_tiles(self):
        """[Tloc] bool: the local tiles that are real (not padding)."""
        return lax.axis_index(self.axis_name) * self.blocks.Tloc + jnp.arange(self.blocks.Tloc) < self.blocks.nTiles

    def zero_padding(self, a):
        """a [Tloc, ...] with the padding tiles set to 0."""
        a = jnp.asarray(a)
        return jnp.where(self.real_tiles().reshape((-1,) + (1,) * (a.ndim - 1)), a, jnp.zeros((), a.dtype))

    def tile_mask(self, mask2d):
        """[ny, nx] bool -> [Tloc, ny, nx], False on padding tiles, typed varying."""
        m = jnp.asarray(mask2d)
        return self.real_tiles().reshape(-1, 1, 1) & m[None]

    def first_device(self, x):
        """x (pytree) as held by device 0, typed invariant (psum of x on device 0 and zeros elsewhere: exact)."""
        ax = self.axis_name
        on0 = lax.axis_index(ax) == 0
        return jax.tree.map(lambda a: lax.psum(jnp.where(on0, a, jnp.zeros_like(a)), ax), x)

    def rounds(self, name):
        """Number of ppermute rounds (collective-permutes) of one exchange (an exch2 C-grid vector exchange: summed
        over its EXCH2_RX2_CUBE passes)."""
        if name in self.rx2:
            return (sum(len(self.plans[f"{name}#{p}"].perms) for p in range(self.rx2[name][1]))
                    + (len(self.plans[f"{name}#post"].perms) if self.rx2[name][2] else 0))
        return len(self.plans[name].perms)
