"""Gates of lane EOSAB: FIND_ALPHA and FIND_BETA, 'MDJWF' arms (model/src/find_alpha.F:222-279, :539-589), against
gfortran (M3 Task 30 prerequisite: vermix's CALC_OCE_MXLAYER and KPP's STATEKPP).

Oracle: the EOSAB replay harness (reference/replay_eosab; vermix/code build, THE_MAIN_LOOP replaced; find_alpha.f,
find_rho.f, pressure_for_eos.f, ini_eos.f byte-identical to the M3 oracle build), run listed in
reference/replay_eosab/CURRENT (relative to $MJX_RUNS). vermix/input's namelist (eosType = 'MDJWF', shared by
input.ggl90, .gglLC, .my82, .opps, .pp81; selectP_inEOS_Zc = 2 by SET_PARMS). 64 synthetic samples of the 5 x 5
(1 x 1 + halos 2) x 26-level tile (reference/replay_eosab/replay_io.py: land points with zero or ocean-like values,
fresh columns, salt = +0, -0, < 0 and 1e-14..1e-6 at isolated points, totPhiHyd anomalies, priors of both outputs),
three passes: 1 kRef = k, full tile; 2 kRef = 1, full tile; 3 kRef = Nr+1-k on the range i = 2-OLx..sNx+OLx-1,
j = 1-OLy..sNy (points outside keep the prior). The samples are fed to the ported routines as 64 tiles (the routines
are tile-local); levels by a Python loop in the Fortran order and by scan_levels (k a traced KIdx, as a scanned
caller passes it; pass 3's kRef = Nr+1-k is a KIdx too).

Gates: bitwise (bit patterns, all points incl. halos and land, both finite) for every output of every pass; INI_EOS
MDJWF coefficients bitwise vs the harness's EOS.h. Negative controls: planted changes measured to bite (re-association
in each routine, kRef replaced by k, the `s1 = 0` of the ELSE branch dropped in each routine). Gradients (w.r.t.
theta, salt, totPhiHyd): finite on every lane, tangent vs adjoint dot test, central FD h-sweep at smooth points.
All tier1x (no tier-1 test). JAX transforms are used here, never in mitjax/model.
"""

import functools
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.farray import FArray
from mitjax.tests.col_replay import Common, bit_equal, planted

REPO = Path(__file__).resolve().parents[2]


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rio = _load_by_path("_mjx_replay_eosab_io", REPO / "reference" / "replay_eosab" / "replay_io.py")


def current_runs():
    out = {}
    for ln in (REPO / "reference" / "replay_eosab" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out[inp] = Path(paths.RUNS) / rel
    return out


class _Size:
    """vermix's SIZE.h with the samples as tiles (nSx = number of samples)."""

    def __init__(self, sz, ntiles):
        for k in ("sNx", "sNy", "OLx", "OLy", "Nr", "nPx", "nPy"):
            setattr(self, k, getattr(sz, k))
        self.nSx, self.nSy = ntiles, 1


class _Cfg:
    def __init__(self, c, ntiles):
        self.cpp, self._c = c.cpp, c
        self.size = _Size(c.size, ntiles)

    def use_flag(self, name):
        return self._c.use_flag(name)


@functools.lru_cache(maxsize=None)
def experiment(inp):
    from mitjax.config import params as cp
    return cp.load("vermix", inp)


def our_eos(eos_type):
    """EOS.h from our INI_EOS port (gated against the harness's own EOS.h in test_ini_eos_mdjwf_coefficients)."""
    from types import SimpleNamespace
    from mitjax.model.src.ini_eos import UNSET_RL, ini_eos
    p = SimpleNamespace(fluidIsWater=True, eosType=eos_type.ljust(6)[:6], usingPCoords=False,
                        tAlpha=UNSET_RL, sBeta=UNSET_RL)
    cfg = SimpleNamespace(cpp=SimpleNamespace(TARGET_NEC_SX=False))
    return ini_eos(cfg=cfg, params=p)


class Replay:
    """The harness run: inputs, Fortran outputs, dumped parameters; the commons for samples `sel` as tiles."""

    def __init__(self, inp="input"):
        runs = current_runs()
        if inp not in runs:
            raise FileNotFoundError(f"no EOSAB replay run for vermix/{inp} in reference/replay_eosab/CURRENT")
        self.rundir = runs[inp]
        self.exp = experiment(inp)
        sz = self.exp.cfg.size
        self.sz = {k: getattr(sz, k) for k in rio.SIZE_KEYS}
        self.inputs = rio.read_inputs(self.rundir, self.sz)
        self.out = rio.read_outputs(self.rundir, self.sz)
        self.g = rio.read_grid(self.rundir)
        self.orig = rio.read_orig(self.rundir)

    def samples(self, name, sel):
        a = self.inputs[name] if name in self.inputs else self.out[name]
        return a[sel][:, 0]

    def commons(self, sel):
        """(cfg, grid, params, eos, state) for the samples `sel` as tiles."""
        n = len(np.arange(rio.NSAMP)[sel])
        cfg = _Cfg(self.exp.cfg, n)
        s = cfg.size
        Nr = s.Nr
        ij = dict(i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))
        f3 = lambda name: FArray(jnp.asarray(self.samples(name, sel)), name, k=(1, Nr), **ij)   # noqa: E731
        g = self.g
        vec = lambda name, n_: FArray(jnp.asarray(g[name]), name, k=(1, n_), tiled=False)      # noqa: E731
        grid = Common(rC=vec("rC", Nr))
        static = {"selectP_inEOS_Zc": int(g["selectP_inEOS_Zc"]), "usingPCoords": bool(g["usingPCoords"]),
                  "usingZCoords": not bool(g["usingPCoords"])}
        par = dict(pRef4EOS=vec("pRef4EOS", Nr), phiRef=vec("phiRef", 2*Nr+1))
        par.update({k: jnp.float64(g[k]) for k in ("rhoConst", "rhoNil", "surf_pRef")})
        params = Common(static, {}, **par)
        eos = our_eos("MDJWF")
        state = Common(theta=f3("theta"), salt=f3("salt"), totPhiHyd=f3("totPhiHyd"))
        return cfg, grid, params, eos, state

    def prior(self, name, sel, cfg):
        s = cfg.size
        return FArray(jnp.asarray(self.samples(name, sel)), name, k=(1, s.Nr),
                      i=(1-s.OLx, s.sNx+s.OLx), j=(1-s.OLy, s.sNy+s.OLy))

    def pass_args(self, p):
        """(iMin, iMax, jMin, jMax, kRef(k)) of harness pass p (the_main_loop.F; the ranges as the harness wrote
        them to cm_grid.bin)."""
        g, Nr = self.g, self.sz["Nr"]
        rng = tuple(int(g[f"{n}_p{p}"]) for n in ("iMin", "iMax", "jMin", "jMax"))
        kref = {1: lambda k: k, 2: lambda k: 1, 3: lambda k: Nr+1-k}[p]
        return rng, kref


@functools.lru_cache(maxsize=None)
def replay():
    return Replay()


def check(label, got, want):
    nd, nt, fin = bit_equal(got, want)
    print(f"{label}: {nt - nd}/{nt} points bitwise equal, finite={fin}")
    assert fin, f"{label}: non-finite values"
    assert nd == 0, f"{label}: {nd} of {nt} points differ (max |diff| " \
                    f"{np.max(np.abs(np.asarray(got) - np.asarray(want))):.3e})"


def _level(fld, k):
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = fld.dims
    kk = k - klo
    return FArray(fld.data[:, getattr(kk, "value", kk)], fld.name, i=(ilo, ihi), j=(jlo, jhi))


def _set_level(fld, k, lev):
    (_, _, _), (_, _, _), (_, klo, _) = fld.dims
    kk = k - klo
    return FArray(fld.data.at[:, getattr(kk, "value", kk)].set(lev.data), fld.name, tiled=fld.tiled, _dims=fld.dims)


ALL = slice(None)


def run_pass(R, p, sel=ALL, mod=None, scan=False):
    """FIND_ALPHA and FIND_BETA for every level k = 1..Nr of pass p, as the harness calls them (prior of level k in,
    output of level k out). Returns (alpha, beta) storage [tile, k, j, i]. `mod`: a (planted) copy of the module;
    scan=True: the level loop as scan_levels (k a traced KIdx, kRef = k, 1 or Nr+1-k)."""
    from mitjax.model.src import find_alpha as fa
    mod = mod or fa
    cfg, grid, params, eos, state = R.commons(sel)
    Nr = cfg.size.Nr
    (iMin, iMax, jMin, jMax), kref = R.pass_args(p)
    aP, bP = R.prior("alphaPrior", sel, cfg), R.prior("betaPrior", sel, cfg)

    def f(grid, params, eos, state, aP, bP):
        def body(k, c):
            A, B = c
            a = mod.find_alpha(iMin, iMax, jMin, jMax, k, kref(k), _level(aP, k), params=params, eos=eos,
                               cfg=cfg, grid=grid, state=state)
            b = mod.find_beta(iMin, iMax, jMin, jMax, k, kref(k), _level(bP, k), params=params, eos=eos,
                              cfg=cfg, grid=grid, state=state)
            return _set_level(A, k, a), _set_level(B, k, b)

        c = (aP, bP)
        if scan:
            from mitjax.ops.scan_k import scan_levels
            c = scan_levels(body, c, 1, Nr)
        else:
            for k in range(1, Nr+1):
                c = body(k, c)
        return c[0].data, c[1].data

    return jax.jit(f)(grid, params, eos, state, aP, bP)


# ---------------------------------------------------------------------------------------------------------------
# replay gates

def test_ini_eos_mdjwf_coefficients():
    """INI_EOS: our MDJWF coefficients and eosRefP0 == the harness's EOS.h (bit patterns): the precondition of every
    gate below (they use our INI_EOS)."""
    R = replay()
    mine = our_eos("MDJWF")
    for name in ("eosMDJWFnum", "eosMDJWFden"):
        check(f"INI_EOS MDJWF {name}", np.asarray(getattr(mine, name).data), R.orig[name])
    assert float(mine.eosRefP0) == R.orig["eosRefP0"] == R.g["eosRefP0"]
    assert int(R.g["selectP_inEOS_Zc"]) == 2 and R.g["usingPCoords"] == 0.0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_find_alpha_beta_mdjwf_replay(p):
    """FIND_ALPHA, FIND_BETA 'MDJWF' == gfortran on every point (halos, land, unwritten points) of pass p."""
    R = replay()
    a, b = run_pass(R, p)
    check(f"FIND_ALPHA MDJWF pass {p}", a, R.samples(f"alpha_p{p}", ALL))
    check(f"FIND_BETA MDJWF pass {p}", b, R.samples(f"beta_p{p}", ALL))
    if p == 3:     # the partial range: the prior is kept outside it (and was overwritten inside it)
        kept = np.asarray(a) >= 1000.0
        assert kept.sum() == 64*26*(25 - 9), kept.sum()


@pytest.mark.parametrize("p", [1, 2, 3])
def test_find_alpha_beta_mdjwf_scan_levels(p):
    """The same with the level loop as scan_levels (k a traced KIdx, as a scanned caller passes it): bitwise."""
    R = replay()
    a, b = run_pass(R, p, scan=True)
    check(f"FIND_ALPHA MDJWF pass {p} (scan_levels)", a, R.samples(f"alpha_p{p}", ALL))
    check(f"FIND_BETA MDJWF pass {p} (scan_levels)", b, R.samples(f"beta_p{p}", ALL))


def test_salinity_branch_exercised():
    """The inputs reach both arms of `IF ( s1 .GT. 0. _d 0 )` (:251, :568) on points the passes write."""
    R = replay()
    s = R.inputs["salt"]
    n_le0, n_neg, n_tiny = int((s <= 0.0).sum()), int((s < 0.0).sum()), int(((s > 0) & (s < 1e-6)).sum())
    print(f"salt <= 0: {n_le0} points ({n_neg} < 0, {int(np.signbit(s[s == 0.0]).sum())} = -0); "
          f"0 < salt < 1e-6: {n_tiny}")
    assert n_neg > 0 and n_le0 > n_neg and n_tiny > 0


def test_mdjwf_needs_common_blocks():
    """'MDJWF' without theta/salt (state) or PRESSURE_FOR_EOS's fields refuses (the LINEAR callers pass neither)."""
    from mitjax.model.src.find_alpha import find_alpha, find_beta
    R = replay()
    cfg, grid, params, eos, state = R.commons(slice(0, 1))
    aP = _level(R.prior("alphaPrior", slice(0, 1), cfg), 1)
    for fn in (find_alpha, find_beta):
        with pytest.raises(ValueError, match="cfg=, grid=, state= are required"):
            fn(-1, 3, -1, 3, 1, 1, aP, params=params, eos=eos)


# ---------------------------------------------------------------------------------------------------------------
# negative controls: planted changes, each measured to bite on the replay

PLANTS = [
    # (label, old, new, pass, which output must differ)
    # 3.*eosMDJWFnum(3)*t1 as 3.*(eosMDJWFnum(3)*t1) (:262). Not used: 4.*den(4)*t1 -> 4.*(den(4)*t1) (a power of
    # two: exact, 0 points) and p1*p1*(..) -> p1*(p1*(..)) (0 points; measured on the login node, 2026-10-02)
    ("FIND_ALPHA re-association 3.*num(3)*t1", "+ t1*(2.0*num[2] + 3.0*num[3]*t1)",
     "+ t1*(2.0*num[2] + 3.0*(num[3]*t1))", 1, "alpha"),
    ("FIND_BETA re-association 1.5*sp5*(...)", "+ 1.5*sp5*(den_[8] + den_[9]*t2))",
     "+ 1.5*(sp5*(den_[8] + den_[9]*t2)))", 1, "beta"),
    ("FIND_ALPHA kRef -> k in PRESSURE_FOR_EOS", "jMax, kRef, dp0, locPres,      # :225-228",
     "jMax, k, dp0, locPres,      # :225-228", 2, "alpha"),
    ("FIND_ALPHA ELSE s1 = 0 dropped", "s1 = jnp.where(pos, s1, 0.0)                                    # :254",
     "s1 = s1  # :254", 1, "alpha"),
    ("FIND_BETA ELSE s1 = 0 dropped", "s1 = jnp.where(pos, s1, 0.0)                                    # :571",
     "s1 = s1  # :571", 1, "beta"),
]


@pytest.mark.parametrize("label,old,new,p,which", PLANTS, ids=[x[0] for x in PLANTS])
def test_negative_controls(label, old, new, p, which):
    R = replay()
    mod = planted("find_alpha", old, new)
    a, b = run_pass(R, p, mod=mod)
    got = {"alpha": a, "beta": b}[which]
    nd, nt, _ = bit_equal(got, R.samples(f"{which}_p{p}", ALL))
    print(f"planted '{label}': {nd}/{nt} {which} points differ (pass {p})")
    assert nd > 0, label


# ---------------------------------------------------------------------------------------------------------------
# gradients: finite on every lane (land, halos, salt <= 0 and near-zero lanes), dot test, central FD at smooth points

GRAD_SEL = slice(0, 4)


def test_gradients():
    R = replay()
    from mitjax.model.src.find_alpha import find_alpha, find_beta
    from mitjax.ops.scan_k import scan_levels
    cfg, grid, params, eos, state = R.commons(GRAD_SEL)
    Nr = cfg.size.Nr
    s = cfg.size
    full = (1-s.OLx, s.sNx+s.OLx, 1-s.OLy, s.sNy+s.OLy)
    aP, bP = R.prior("alphaPrior", GRAD_SEL, cfg), R.prior("betaPrior", GRAD_SEL, cfg)
    x0 = jnp.stack([state.theta.data, state.salt.data, state.totPhiHyd.data])

    def fwd(x):
        st = state.replace(theta=FArray(x[0], "theta", tiled=True, _dims=state.theta.dims),
                           salt=FArray(x[1], "salt", tiled=True, _dims=state.salt.dims),
                           totPhiHyd=FArray(x[2], "totPhiHyd", tiled=True, _dims=state.totPhiHyd.dims))
        # every level k, kRef = k (STATEKPP :1918-1924) and kRef = 1 (:1859-1865, CALC_OCE_MXLAYER), as scanned
        # callers run them (scan_levels: the unrolled 104-call program took 6.4 min to compile, job 27842232)
        def body(k, c):
            out = []
            for X, (fn, P, kRef) in zip(c, ((find_alpha, aP, k), (find_alpha, aP, 1), (find_beta, bP, k),
                                            (find_beta, bP, 1))):
                y = fn(*full, k, kRef, _level(P, k), params=params, eos=eos, cfg=cfg, grid=grid, state=st)
                out.append(_set_level(X, k, y))
            return tuple(out)

        c = scan_levels(body, (aP, aP, bP, bP), 1, Nr)
        return jnp.stack([X.data for X in c])

    rng = np.random.default_rng(11)
    y0 = jax.jit(fwd)(x0)
    w = jnp.asarray(rng.standard_normal(y0.shape))
    J = lambda x: jnp.sum(w*fwd(x))                                         # noqa: E731
    g = np.asarray(jax.jit(jax.grad(J))(x0))
    assert np.all(np.isfinite(g)), f"{np.sum(~np.isfinite(g))} non-finite gradient lanes"
    salt = np.asarray(x0[1])
    print(f"grad finite on {g.size} lanes (incl. {int((salt <= 0).sum())} salt <= 0 points, "
          f"{int(((salt > 0) & (salt < 1e-6)).sum())} with 0 < salt < 1e-6), |g|max {np.abs(g).max():.3e}")
    v = jnp.asarray(rng.standard_normal(x0.shape))
    _, tl = jax.jit(lambda x, v: jax.jvp(fwd, (x,), (v,)))(x0, v)
    _, vjp = jax.vjp(fwd, x0)
    (ad,) = jax.jit(vjp)(w)
    lhs, rhs = float(jnp.sum(w*tl)), float(jnp.sum(ad*v))
    rel = abs(lhs - rhs)/max(abs(lhs), 1e-300)
    print(f"dot test {lhs:.15e} vs {rhs:.15e} (rel {rel:.1e})")
    assert rel < 1e-12
    # central FD h-sweep at smooth points (salt > 1: far from the s1 > 0 switch and the SQRT), the two largest
    # gradient lanes of each input field
    Jj = jax.jit(J)
    smooth = np.broadcast_to(salt > 1.0, g.shape)
    worst = 0.0
    for c in range(3):
        gc = np.where(smooth[c], np.abs(g[c]), 0.0).ravel()
        for idx in np.argsort(-gc)[:2]:
            lane = (c,) + np.unravel_index(idx, g[c].shape)
            e = np.zeros(g.shape)
            e[lane] = 1.0
            e = jnp.asarray(e)
            scale = max(1.0, abs(float(x0[lane])))
            fds = [(float(Jj(x0 + h*scale*e)) - float(Jj(x0 - h*scale*e)))/(2*h*scale)
                   for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]   # totPhiHyd lanes: plateau at h >= 1e-3
            best = min(abs(f - g[lane])/abs(g[lane]) for f in fds)
            worst = max(worst, best)
            print(f"  lane {lane}: AD {g[lane]:.10e}, FD {[f'{f:.10e}' for f in fds]} (best rel {best:.1e})")
    assert worst < 1e-6, worst    # measured (login node, 2026-10-02): 5.5e-9
