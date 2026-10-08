"""Fortran-index arrays: `uFld[i+1, j]` and `KE.at[i, j].set(...)` with MITgcm's own index ranges (plan Task 8).

The array style of all physics code (decided by Nikolay 2026-10-01 from the Task 8 side-by-side page
`dev/prototype/side_by_side.html`; conventions in `docs/KERNEL_GUIDE.md`).

What a Fortran MITgcm developer writes, and what it means:

    # _RL uFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy)          (one level; the bi,bj tile loop is implicit)
    uFld = FArray(data, "uFld", i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    # _RS hFacW(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
    hFacW = FArray(data, "hFacW", i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr))
    # _RL recip_deepFacC(Nr)                          (no bi,bj: not tiled)
    recip_deepFacC = FArray(data, "recip_deepFacC", k=(1, Nr), tiled=False)

    j = loop_j(1-OLy, sNy+OLy-1)                      # DO j=1-OLy,sNy+OLy-1
    i = loop_i(1-OLx, sNx+OLx-1)                      #  DO i=1-OLx,sNx+OLx-1
    KE = KE.at[i, j].set(0.125*((uFld[i, j] + uFld[i+1, j])**2   #   KE(i,j) = 0.125*(
                                + (vFld[i, j] + vFld[i, j+1])**2))  #     ...)

* Declarations. Dimensions are named i, j, k and given in Fortran order with their declared bounds (`k=(1, Nr)` or
  `k=(1, Nr+1)`: shapes come from the declaration, never from a loop range [P§5]). Storage is the plan's layout:
  `[tile, k, j, i]` (a declared dimension that is absent is absent from the storage), so Fortran index `i` sits at
  storage index `i - lo_i` (`i-1+OLx` for `lo_i = 1-OLx`). The tile axis (bi,bj) is implicit and first; arrays
  without bi,bj in their Fortran declaration are `tiled=False`. The storage shape is checked against the
  declaration.
* Indices, in Fortran order: a loop index (`loop_i`, `loop_j`, `loop_k`, shifted by `+`/`-` an integer) or an
  integer. A loop index may only index the dimension of its own kind (`uFld[j, i]` is an error), every declared
  dimension needs an index, and every index must lie inside the declared bounds; errors name the array and the
  indices in Fortran order, e.g. `uFld(i+1, j): i+1 = 2..14 is outside the declared 1-OLx:sNx+OLx = -2..13`.
* Reading `A[...]` returns a plain jax array (no wrapper: arithmetic, `jnp.where`, `jnp.abs` are ordinary jax).
  With at least one loop index its axes are `[tile, (k), j, i]` -- tile only for tiled arrays, k only in a
  k-vectorised nest (`loops_kji`) -- with length 1 where the array has no such dimension or is indexed by an
  integer, so that every operand of one statement broadcasts on the right axes (`cosFacU[j]` is `[tile, nj, 1]`,
  which multiplies `[tile, nj, ni]` along i; a misaligned `[tile, nj]` against `[tile, nj, ni]` would silently
  align j with i when nj == ni). With integer indices only (`recip_deepFacC[k]`) it is what numpy indexing gives
  (a scalar here).
* Writing `A.at[...].set(value)` returns a new FArray in which exactly the indexed points hold `value` (broadcast
  as above) and every other point keeps the prior array [E§2: KG:29-30]: the Fortran semantics of a loop that does
  not cover the whole declared range.
* An FArray is a pytree (its storage is the only leaf; name and declaration are static), so it passes through jit,
  grad, vmap, scan and shard_map unchanged. All index arithmetic happens at trace time on Python integers: the
  traced program is the one plain slicing gives (`[:, j0:j1, i0+1:i1+1]`), checked by the identical-HLO gate of
  `mitjax/tests/test_style_prototype.py`.

No JAX transforms here; jax.numpy indexing only.
"""

import jax
import jax.numpy as jnp
import numpy as np

AXES = ("k", "j", "i")      # storage order after the tile axis: [tile, k, j, i]


class Loop:
    """A Fortran DO-loop index: `axis` runs lo..hi (inclusive); `i+1` is the same loop shifted by +1."""

    __slots__ = ("axis", "lo", "hi", "off", "nest_k")

    def __init__(self, axis, lo, hi, off=0, nest_k=False):
        if axis not in AXES:
            raise ValueError(f"loop axis must be one of {AXES}, got {axis!r}")
        if not (isinstance(lo, int) and isinstance(hi, int) and isinstance(off, int)):
            raise TypeError(f"DO {axis}={lo},{hi}: loop bounds and shifts are Python integers")
        if hi < lo:
            raise ValueError(f"DO {axis}={lo},{hi}: empty loop")
        self.axis, self.lo, self.hi, self.off, self.nest_k = axis, lo, hi, off, nest_k

    def __add__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        return Loop(self.axis, self.lo, self.hi, self.off + n, self.nest_k)

    __radd__ = __add__

    def __sub__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        return Loop(self.axis, self.lo, self.hi, self.off - n, self.nest_k)

    @property
    def first(self):
        return self.lo + self.off

    @property
    def last(self):
        return self.hi + self.off

    def __len__(self):
        return self.hi - self.lo + 1

    def __repr__(self):
        s = self.axis if self.off == 0 else f"{self.axis}{self.off:+d}"
        return f"{s} (DO {self.axis}={self.lo},{self.hi})"

    def label(self):
        return self.axis if self.off == 0 else f"{self.axis}{self.off:+d}"


def loop_i(lo, hi):
    """DO i=lo,hi"""
    return Loop("i", lo, hi)


def loop_j(lo, hi):
    """DO j=lo,hi"""
    return Loop("j", lo, hi)


def loops_kji(k, j, i):
    """A k-vectorised nest DO k / DO j / DO i, each given as (lo, hi); returns the loop indices (k, j, i).

    Only for loops whose k iterations are independent (the routine's docstring says so). Inside such a nest every
    read has a k axis (length 1 for an array without k), so 2-D and 3-D fields broadcast correctly."""
    return (Loop("k", *k, nest_k=True), Loop("j", *j, nest_k=True), Loop("i", *i, nest_k=True))


def _bounds_label(lo, hi):
    return f"{lo}:{hi}"


@jax.tree_util.register_pytree_node_class
class FArray:
    """An array with Fortran index ranges; storage `[tile, k, j, i]` (see the module docstring)."""

    __slots__ = ("data", "name", "dims", "tiled")

    def __init__(self, data, name="array", *, i=None, j=None, k=None, tiled=True, _dims=None):
        if _dims is None:
            dims = []
            for axis, b in (("i", i), ("j", j), ("k", k)):
                if b is not None:
                    lo, hi = b
                    if not (isinstance(lo, int) and isinstance(hi, int)) or hi < lo:
                        raise ValueError(f"{name}: bad bounds {axis}={b!r}")
                    dims.append((axis, lo, hi))
            if not dims:
                raise ValueError(f"{name}: no dimension declared")
            _dims = tuple(dims)
        self.data, self.name, self.dims, self.tiled = data, name, _dims, tiled
        if not _is_placeholder(data):
            want = self.storage_shape(None)
            got = tuple(data.shape)
            if len(got) != len(want) or any(w is not None and w != g for w, g in zip(want, got)):
                raise ValueError(f"{self.decl()} needs storage {self._shape_label()}, got shape {got}")

    # -- pytree ---------------------------------------------------------------------------------------------------
    def tree_flatten(self):
        return (self.data,), (self.name, self.dims, self.tiled)

    @classmethod
    def tree_unflatten(cls, aux, children):
        name, dims, tiled = aux
        obj = object.__new__(cls)
        obj.data, obj.name, obj.dims, obj.tiled = children[0], name, dims, tiled
        return obj

    # -- declaration ----------------------------------------------------------------------------------------------
    def _storage_dims(self):
        """The declared dims in storage order (k, j, i)."""
        by_axis = {d[0]: d for d in self.dims}
        return [by_axis[a] for a in AXES if a in by_axis]

    def storage_shape(self, ntiles):
        shp = tuple(hi - lo + 1 for _, lo, hi in self._storage_dims())
        return ((ntiles,) if self.tiled else ()) + shp

    def _shape_label(self):
        names = (["tile"] if self.tiled else []) + [str(hi - lo + 1) for _, lo, hi in self._storage_dims()]
        return "[" + ", ".join(names) + "]"

    def decl(self):
        return f"{self.name}({','.join(_bounds_label(lo, hi) for _, lo, hi in self.dims)})"

    @property
    def ntiles(self):
        return self.data.shape[0] if self.tiled else None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"FArray {self.decl()}{' tiled' if self.tiled else ''} storage {tuple(self.data.shape)}"

    def local(self, name, fill=jnp.nan):
        """A local array declared like this one (same bounds and tiles), every point `fill` (default NaN, so that
        a read of a point the Fortran never wrote shows up)."""
        return FArray(jnp.full_like(self.data, fill), name, tiled=self.tiled, _dims=self.dims)

    # -- indexing -------------------------------------------------------------------------------------------------
    def _resolve(self, idx):
        """Fortran index tuple -> (storage index tuple, canonical shape insertion plan).

        Returns (sidx, loops, expand) where sidx indexes `data` with slices/ints (numpy semantics), loops is the set
        of loop axes present, and expand(x) inserts the unit axes of the canonical read shape."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        label = f"{self.name}({', '.join(_ilabel(x) for x in idx)})"
        if len(idx) != len(self.dims):
            raise IndexError(f"{label}: {len(idx)} indices for the {len(self.dims)} dimensions of {self.decl()}")
        per_axis = {}
        for n, (x, (axis, lo, hi)) in enumerate(zip(idx, self.dims), start=1):
            if isinstance(x, Loop):
                if x.axis != axis:
                    raise IndexError(f"{label}: index {n} is a {x.axis} loop, but dimension {n} of {self.decl()} "
                                     f"is {axis}")
                if x.first < lo or x.last > hi:
                    raise IndexError(f"{label}: {x.label()} = {x.first}..{x.last} is outside the declared "
                                     f"{_bounds_label(lo, hi)} of dimension {n} of {self.decl()}")
                per_axis[axis] = slice(x.first - lo, x.last - lo + 1)
            elif isinstance(x, int) and not isinstance(x, bool):
                if x < lo or x > hi:
                    raise IndexError(f"{label}: index {n} = {x} is outside the declared {_bounds_label(lo, hi)} of "
                                     f"{self.decl()}")
                per_axis[axis] = x - lo
            elif isinstance(x, KIdx):                                       # a traced level (scan_levels)
                if x.lo < lo or x.hi > hi:
                    raise IndexError(f"{label}: index {n} = {x!r} is outside the declared {_bounds_label(lo, hi)} of "
                                     f"{self.decl()}")
                per_axis[axis] = x.value - lo
            else:
                raise TypeError(f"{label}: index {n} must be a loop index or a Python int, got {type(x).__name__}")
        sidx = ((slice(None),) if self.tiled else ()) + tuple(per_axis[a] for a in AXES if a in per_axis)
        loops = [x for x in idx if isinstance(x, Loop)]
        return sidx, loops, label

    def _canonical(self, sidx, loops):
        """Index tuple with `None` inserted so that the read has axes [tile?, (k), j, i] (see module docstring)."""
        nest_k = any(x.nest_k for x in loops)
        out = list(sidx[:1]) if self.tiled else []
        rest = list(sidx[1:] if self.tiled else sidx)
        by_axis = {}
        for d, s in zip(self._storage_dims(), rest):
            by_axis[d[0]] = s
        for axis in AXES:
            if axis == "k" and not nest_k:
                if "k" in by_axis:                 # integer k (a loop k needs a k nest)
                    if isinstance(by_axis["k"], slice):
                        raise IndexError("a k loop index must come from loops_kji (a k-vectorised nest)")
                    out.append(by_axis["k"])
                continue
            s = by_axis.get(axis)
            if s is None:
                out.append(None)                    # no such dimension: unit axis
            elif isinstance(s, slice):
                out.append(s)
            else:
                out.extend([s, None])               # integer index: drop the axis, then a unit axis
        return tuple(out)

    def __getitem__(self, idx):
        sidx, loops, _ = self._resolve(idx)
        if not loops:
            return self.data[sidx]
        return self.data[self._canonical(sidx, loops)]

    @property
    def at(self):
        return _At(self)


class _At:
    __slots__ = ("arr",)

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, idx):
        return _Setter(self.arr, idx)


class _Setter:
    __slots__ = ("arr", "idx")

    def __init__(self, arr, idx):
        self.arr, self.idx = arr, idx

    def set(self, value):
        """A(idx) = value: the indexed points get `value`, all other points keep the prior array."""
        a = self.arr
        sidx, loops, label = a._resolve(self.idx)
        v = value
        if loops and getattr(v, "ndim", 0) > 0:
            view = np.broadcast_to(np.zeros((), bool), a.data.shape)     # zero-size view: shapes only
            sh = tuple(0 if hasattr(x, "aval") or (hasattr(x, "shape") and not isinstance(x, slice)) else x
                       for x in sidx)                                    # a traced level: shape of any level
            cshape, wshape = view[a._canonical(sh, loops)].shape, view[sh].shape
            if v.ndim > len(cshape):
                raise ValueError(f"{label} = value: value has {v.ndim} axes, the target has {len(cshape)} "
                                 f"{tuple(cshape)}")
            try:
                bshape = jnp.broadcast_shapes(v.shape, cshape)
            except ValueError:
                raise ValueError(f"{label} = value: value shape {tuple(v.shape)} does not broadcast to the "
                                 f"target {tuple(cshape)}") from None
            if bshape != tuple(cshape):
                raise ValueError(f"{label} = value: value shape {tuple(v.shape)} does not broadcast to the "
                                 f"target {tuple(cshape)}")
            if tuple(cshape) != tuple(wshape):           # integer indices: drop their unit axes
                v = jnp.broadcast_to(v, cshape).reshape(wshape)
        return FArray(a.data.at[sidx].set(v), a.name, tiled=a.tiled, _dims=a.dims)


class KIdx:
    """A level index k that is TRACED (the input of a scan over levels, mitjax/ops/scan_k.scan_levels; KERNEL_GUIDE
    §4) but carries the static facts the per-level code branches on: the range lo..hi of its possible values and
    its parity. `k == 1`, `k >= 2`, `k == Nr` are decided statically when the range decides them (else TypeError:
    the caller must peel that level); `k % 2` is static when the parity is known; `k + 1`, `k - 1` shift the range.
    FArray indexing with it is a dynamic index into the level axis (bounds checked on the whole range)."""

    __slots__ = ("value", "lo", "hi", "parity")

    def __init__(self, value, lo, hi, parity=None):
        self.value, self.lo, self.hi, self.parity = value, int(lo), int(hi), parity

    def _shift(self, n):
        if not isinstance(n, int):
            return NotImplemented
        par = None if self.parity is None else (self.parity + n) % 2
        return KIdx(self.value + n, self.lo + n, self.hi + n, par)

    def __add__(self, n):
        return self._shift(n)

    __radd__ = __add__

    def __sub__(self, n):
        return self._shift(-n) if isinstance(n, int) else NotImplemented

    def __rsub__(self, n):
        """n - k (e.g. kUp = 1+MOD(Nr-k,2)): a level index with the range mirrored."""
        if not isinstance(n, int):
            return NotImplemented
        par = None if self.parity is None else (n - self.parity) % 2
        return KIdx(n - self.value, n - self.hi, n - self.lo, par)

    def __mul__(self, n):
        """n*k (e.g. phiRef(2*k)): a level index with the range scaled."""
        if not isinstance(n, int) or n <= 0:
            return NotImplemented
        par = None if self.parity is None else (0 if n % 2 == 0 else self.parity)
        return KIdx(self.value * n, self.lo * n, self.hi * n, par)

    __rmul__ = __mul__

    def __mod__(self, n):
        if n == 2 and self.parity is not None:
            return self.parity
        raise TypeError(f"KIdx {self.lo}..{self.hi}: k % {n} is not static")

    def _cmp(self, other, op):
        if hasattr(other, "shape") and not isinstance(other, int):      # an array: a traced pointwise comparison
            return op(self.value, other)
        if not isinstance(other, int):
            raise TypeError("KIdx compares with Python ints and arrays only")
        res = {op(v, other) for v in (self.lo, self.hi)} | {op(v, other) for v in range(self.lo, self.hi + 1)}
        if len(res) != 1:
            raise TypeError(f"KIdx {self.lo}..{self.hi}: comparison with {other} is not static (peel that level)")
        return res.pop()

    def __eq__(self, o):
        import operator
        return self._cmp(o, operator.eq)

    def __ne__(self, o):
        import operator
        return self._cmp(o, operator.ne)

    def __lt__(self, o):
        import operator
        return self._cmp(o, operator.lt)

    def __le__(self, o):
        import operator
        return self._cmp(o, operator.le)

    def __gt__(self, o):
        import operator
        return self._cmp(o, operator.gt)

    def __ge__(self, o):
        import operator
        return self._cmp(o, operator.ge)

    __hash__ = None

    def __index__(self):
        raise TypeError(f"KIdx {self.lo}..{self.hi} is traced: no Python int (index a Python list with it?)")

    def __repr__(self):
        return f"KIdx({self.lo}..{self.hi}, parity={self.parity})"


def _ilabel(x):
    if isinstance(x, Loop):
        return x.label()
    return str(x)


def _is_placeholder(data):
    """Pytree internals (jax.tree_util) build FArrays with non-array leaves; skip the shape check for those."""
    return not hasattr(data, "shape")
