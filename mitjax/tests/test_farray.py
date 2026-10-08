"""mitjax/farray.py (plan Task 8): Fortran-index reads and writes against a numpy reference written as explicit Fortran
loops, the error messages (Fortran order), and the pytree round trip through jit, grad, scan, vmap and shard_map at
P=4 fake devices. Each check has a planted-error twin that must fail."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from mitjax.farray import FArray, Loop, loop_i, loop_j, loops_kji

sNx, sNy, OLx, OLy, Nr, T = 5, 4, 2, 3, 3, 8          # deliberately sNx != sNy, OLx != OLy (no silent transposes)
Nx, Ny = sNx + 2 * OLx, sNy + 2 * OLy
I_B, J_B, K_B = (1 - OLx, sNx + OLx), (1 - OLy, sNy + OLy), (1, Nr)


def rnd(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def f2(a, i, j):
    """Fortran A(i,j) of a [tile, j, i] array declared (1-OLx:..,1-OLy:..), for every tile."""
    return a[:, j - J_B[0], i - I_B[0]]


def test_reads_match_fortran_loops():
    u, h, c, d = rnd(T, Ny, Nx), rnd(T, Nr, Ny, Nx, seed=1), rnd(T, Ny, seed=2), rnd(Nr, seed=3)
    uF = FArray(jnp.asarray(u), "uFld", i=I_B, j=J_B)
    hF = FArray(jnp.asarray(h), "hFacW", i=I_B, j=J_B, k=K_B)
    cF = FArray(jnp.asarray(c), "cosFacU", j=J_B)
    dF = FArray(jnp.asarray(d), "recip_deepFacC", k=K_B, tiled=False)
    j = loop_j(2 - OLy, sNy + OLy - 1)
    i = loop_i(1 - OLx, sNx + OLx - 1)
    k = 2
    got = np.asarray(uF[i + 1, j - 1] * hF[i, j, k] * cF[j] * dF[k])
    ref = np.empty((T, len(j), len(i)))
    for jj in range(j.lo, j.hi + 1):                              # DO j / DO i, explicit Fortran indices
        for ii in range(i.lo, i.hi + 1):
            ref[:, jj - j.lo, ii - i.lo] = (f2(u, ii + 1, jj - 1) * h[:, k - 1, jj - J_B[0], ii - I_B[0]]
                                            * c[:, jj - J_B[0]] * d[k - 1])
    assert np.array_equal(got, ref)
    # canonical shapes: [tile, j, i]; a missing or integer-indexed dimension is a unit axis; integers only -> numpy
    assert cF[j].shape == (T, len(j), 1)
    assert uF[1 - OLx, j].shape == (T, len(j), 1) and uF[i, 3].shape == (T, 1, len(i))
    assert np.array_equal(np.asarray(uF[1 - OLx, j])[:, :, 0], u[:, j.lo - J_B[0]:j.hi - J_B[0] + 1, 0])
    assert dF[k].shape == () and float(dF[k]) == d[k - 1]
    # negative control: the same reference with the i shift dropped differs
    assert not np.array_equal(np.asarray(uF[i, j - 1] * hF[i, j, k] * cF[j] * dF[k]), ref)


def test_k_vectorised_nest():
    h, u, d = rnd(T, Nr, Ny, Nx, seed=4), rnd(T, Ny, Nx, seed=5), rnd(Nr, seed=6)
    hF = FArray(jnp.asarray(h), "hFacC", i=I_B, j=J_B, k=K_B)
    uF = FArray(jnp.asarray(u), "etaN", i=I_B, j=J_B)
    dF = FArray(jnp.asarray(d), "recip_drF", k=K_B, tiled=False)
    k, j, i = loops_kji((1, Nr), (1, sNy), (1, sNx))
    got = np.asarray(hF[i, j, k] * uF[i, j] * dF[k])
    assert got.shape == (T, Nr, sNy, sNx)
    ref = (h[:, :, OLy:OLy + sNy, OLx:OLx + sNx] * u[:, None, OLy:OLy + sNy, OLx:OLx + sNx] * d[None, :, None, None])
    assert np.array_equal(got, ref)
    with pytest.raises(IndexError, match="loops_kji"):
        hF[loop_i(1, sNx), loop_j(1, sNy), Loop("k", 1, Nr)]


def test_writes_keep_unwritten_points():
    prior = rnd(T, Ny, Nx, seed=7)
    v = rnd(T, Ny, Nx, seed=8)
    KE = FArray(jnp.asarray(prior), "KE", i=I_B, j=J_B)
    V = FArray(jnp.asarray(v), "v", i=I_B, j=J_B)
    j = loop_j(1 - OLy, sNy + OLy - 1)
    i = loop_i(2 - OLx, sNx + OLx - 1)
    out = KE.at[i, j].set(2.0 * V[i, j])
    ref = prior.copy()
    for jj in range(j.lo, j.hi + 1):
        for ii in range(i.lo, i.hi + 1):
            ref[:, jj - J_B[0], ii - I_B[0]] = 2.0 * f2(v, ii, jj)
    assert np.array_equal(np.asarray(out.data), ref)                 # the rim keeps the prior array [E§2]
    assert out.decl() == KE.decl() and isinstance(out, FArray)
    # integer index, scalar and broadcast values (uT(1-OLx,j) = 0.; A(i,j) = c(j))
    jall = loop_j(*J_B)
    z = KE.at[1 - OLx, jall].set(0.0)
    assert np.all(np.asarray(z.data)[:, :, 0] == 0) and np.array_equal(np.asarray(z.data)[:, :, 1:], prior[:, :, 1:])
    c = FArray(jnp.asarray(rnd(T, Ny, seed=9)), "cosFacU", j=J_B)
    b = KE.at[i, j].set(c[j])
    assert np.array_equal(np.asarray(b.data)[:, 0:Ny - 1, 1:Nx - 1],
                          np.broadcast_to(np.asarray(c.data)[:, 0:Ny - 1, None], (T, Ny - 1, Nx - 2)))
    w = KE.at[1 - OLx, jall].set(c[jall])                             # [tile, j, 1] value into an int-i column
    assert np.array_equal(np.asarray(w.data)[:, :, 0], np.asarray(c.data))
    # negative control: a full-range write does not keep the rim
    full = KE.at[loop_i(*I_B), loop_j(*J_B)].set(2.0 * V[loop_i(*I_B), loop_j(*J_B)])
    assert not np.array_equal(np.asarray(full.data), ref)


def test_errors_in_fortran_order():
    a = FArray(jnp.zeros((T, Ny, Nx)), "uFld", i=I_B, j=J_B)
    i = loop_i(1 - OLx, sNx + OLx - 1)
    j = loop_j(1 - OLy, sNy + OLy)
    with pytest.raises(IndexError, match=r"uFld\(i\+2, j\): i\+2 = 1\.\.8 is outside the declared -1:7 of dimension 1"):
        a[i + 2, j]
    with pytest.raises(IndexError, match=r"uFld\(j, i\): index 1 is a j loop, but dimension 1 of uFld\(-1:7,-2:7\) is i"):
        a[j, i]
    with pytest.raises(IndexError, match=r"uFld\(i\): 1 indices for the 2 dimensions"):
        a[i]
    with pytest.raises(IndexError, match=r"uFld\(8, j\): index 1 = 8 is outside"):
        a[8, j]
    with pytest.raises(ValueError, match=r"hFacW\(-1:7,-2:7,1:3\) needs storage \[tile, 3, 10, 9\], got shape"):
        FArray(jnp.zeros((T, Nr + 1, Ny, Nx)), "hFacW", i=I_B, j=J_B, k=K_B)     # Nr+1 levels for an Nr array [P§5]
    FArray(jnp.zeros((Nr + 1,)), "rF", k=(1, Nr + 1), tiled=False)           # declared Nr+1: fine
    with pytest.raises(ValueError, match="does not broadcast"):
        a.at[loop_i(1, sNx), loop_j(1, sNy)].set(jnp.zeros((T, sNx, sNy)))    # a transposed value
    with pytest.raises(TypeError, match="loop index or a Python int"):
        a[i, 1.0]


def _ke(uFld, vFld, KE):
    """A KE-like kernel (mom_calc_ke.F:77-86) in the Fortran-index style."""
    j = loop_j(1 - OLy, sNy + OLy - 1)
    i = loop_i(1 - OLx, sNx + OLx - 1)
    return KE.at[i, j].set(0.25 * ((uFld[i, j] * uFld[i, j] + uFld[i + 1, j] * uFld[i + 1, j])
                                   + (vFld[i, j] * vFld[i, j] + vFld[i, j + 1] * vFld[i, j + 1])))


def _ke_slices(u, v, ke):
    j, jp, i, ip = slice(0, Ny - 1), slice(1, Ny), slice(0, Nx - 1), slice(1, Nx)
    return ke.at[:, j, i].set(0.25 * ((u[:, j, i] * u[:, j, i] + u[:, j, ip] * u[:, j, ip])
                                      + (v[:, j, i] * v[:, j, i] + v[:, jp, i] * v[:, jp, i])))


def _args(seed=10):
    mk = lambda s: FArray(jnp.asarray(rnd(T, Ny, Nx, seed=s)), "f", i=I_B, j=J_B)
    return mk(seed), mk(seed + 1), mk(seed + 2)


def test_traces_to_plain_slicing():
    u, v, ke = _args()
    ja = str(jax.make_jaxpr(lambda a, b, c: _ke(a, b, c).data)(u, v, ke))
    jb = str(jax.make_jaxpr(_ke_slices)(u.data, v.data, ke.data))
    assert ja == jb
    # negative control: a different association is a different program
    jc = str(jax.make_jaxpr(lambda a, b, c: c.at[:, 0:Ny - 1, 0:Nx - 1].set(
        0.25 * (a[:, 0:Ny - 1, 0:Nx - 1] * a[:, 0:Ny - 1, 0:Nx - 1] + (a[:, 0:Ny - 1, 1:Nx] * a[:, 0:Ny - 1, 1:Nx]
                + (b[:, 0:Ny - 1, 0:Nx - 1] * b[:, 0:Ny - 1, 0:Nx - 1] + b[:, 1:Ny, 0:Nx - 1] * b[:, 1:Ny, 0:Nx - 1])))))
               (u.data, v.data, ke.data))
    assert jc != jb


def test_pytree_jit_grad_scan_vmap():
    u, v, ke = _args()
    ref = np.asarray(_ke_slices(u.data, v.data, ke.data))
    out = jax.jit(_ke)(u, v, ke)
    assert isinstance(out, FArray) and out.decl() == ke.decl() and np.array_equal(np.asarray(out.data), ref)
    # grad w.r.t. an FArray is an FArray; the prior KE gets cotangent only where KE is not written
    g = jax.grad(lambda a, b, c: jnp.sum(_ke(a, b, c).data))(u, v, ke)
    assert isinstance(g, FArray)
    gk = jax.grad(lambda c: jnp.sum(_ke(u, v, c).data))(ke)
    mask = np.ones((T, Ny, Nx))
    mask[:, :Ny - 1, :Nx - 1] = 0.0
    assert np.array_equal(np.asarray(gk.data), mask)
    gu = jax.grad(lambda a: jnp.sum(_ke_slices(a, v.data, ke.data)))(u.data)
    assert np.array_equal(np.asarray(g.data), np.asarray(gu))
    # scan with an FArray carry
    def body(c, _):
        return _ke(u, v, c), None
    last, _ = jax.lax.scan(body, ke, None, length=3)
    assert isinstance(last, FArray) and np.array_equal(np.asarray(last.data), ref)
    # vmap over a leading batch axis of the storage
    ub = jax.tree_util.tree_map(lambda x: jnp.stack([x, 2 * x]), u)
    vb = jax.vmap(lambda a: _ke(a, v, ke).data)(ub)
    assert np.array_equal(np.asarray(vb[0]), ref)


def test_shard_map_tiles_at_P4():
    assert len(jax.devices()) >= 4, "the gate XLA_FLAGS give 4 fake CPU devices (conftest.py)"
    mesh = Mesh(np.array(jax.devices()[:4]), ("tiles",))
    u, v, ke = _args()
    ref = np.asarray(_ke(u, v, ke).data)
    spec = P("tiles")
    f = jax.jit(jax.shard_map(_ke, mesh=mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=True))
    out = f(u, v, ke)
    assert isinstance(out, FArray) and np.array_equal(np.asarray(out.data), ref)
    g = jax.jit(jax.grad(lambda a: jnp.sum(jax.shard_map(_ke, mesh=mesh, in_specs=(spec, spec, spec),
                                                         out_specs=spec, check_vma=True)(a, v, ke).data)))(u)
    gref = jax.grad(lambda a: jnp.sum(_ke(a, v, ke).data))(u)
    assert np.array_equal(np.asarray(g.data), np.asarray(gref.data))
