"""pkg/kpp/kpp_routines.F @63cdc0b: KPPMIX, BLDEPTH, WSCALE, RI_IWMIX, SMOOTH_HORIZ, BLMIX, ENHANCE, STATEKPP,
KPP_DOUBLEDIFF.

The NCOM column routines (KPPMIX and its callees) work on arrays dimensioned `(imt)` / `(imt,Nr)` /
`(imt,0:Nrp1,mdiff)`, imt = (sNx+2*OLx)*(sNy+2*OLy) (KPP_PARAMS.h:23), which KPP_CALC passes its (i,j) tile arrays
to (Fortran sequence association: point n of imt is (i,j) with i fastest). Here an `(imt)` array is an FArray
declared `i=(1, imt)` (and `k=` for the level dimension) whose storage is the caller's [tile, (k), j, i] storage
reshaped (`imt_view` / `xy_view` next to the call, KERNEL_GUIDE §4 "sequence association"); `DO i = 1, imt` is
`loop_i(1, imt)`. The `mdiff` dimension of diffus, blmc, dkm1 is a dict keyed by the Fortran index md = 1, 2, 3
(KERNEL_GUIDE §4, leading coefficient dimensions). Per-point level indices (kbl, kn, ki, kmtj) are int32 arrays; a
read `A(i, kl)` with a per-point kl is a gather along the level axis (`_gk`, `_gz`), a write a select over the levels.

Level loops: an independent `DO k` is a k-vectorised nest (`loops_kji`, the docstring says so); a recursion in k
without a routine call (kbl searches, RI_IWMIX's copy-down) is a Python loop in the Fortran's order; the per-level
callers (BLDEPTH's and BLMIX's `DO kl` around WSCALE, STATEKPP's `DO k` around FIND_RHO_2D) are level scans
(`scan_levels`, KERNEL_GUIDE §4, decided 2026-10-02) of the body `level_k(k, c)`, k a traced level index (KIdx);
BLDEPTH peels kl = Nr (the value of sigma it returns).

Pointwise IFs are `jnp.where` with both branches finite: every division and SQRT whose operand can be zero or
negative on a lane the IF does not select is guarded before the operation (mitjax/ops/safe.py). MAX/MIN of REAL
values with the per-site winners of $MJX_REFERENCE/minmax_sites/vermix-code-63cdc0b-463504e.json (and, for the
SHORTWAVE_HEATING sites :696, :832 of 1D_ocean_ice_column, 1D_ocean_ice_column-code-63cdc0b-704fd6b.json). The
diagnostics (DIAGNOSTICS_FILL of KPPbfsfc, KPPRi, KPPdbsfc, KPPdbloc, KPPnuddt, KPPnudds) are output only and not
ported; the CADJ STOREs are TAF directives.
"""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp, powi
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.pkg.kpp.kpp_params_h import MDIFF, NNI, NNJ, imt as _imt_of

MD = tuple(range(1, MDIFF + 1))


# ---------------------------------------------------------------------------------------------------------------
# sequence association and per-point level indices

def imt_view(A, name=None):
    """An (i,j[,k]) tile array passed to an `(imt[,k])` dummy: the same storage, i fastest (sequence association)."""
    dims = {d[0]: d for d in A.dims}
    (_, ilo, ihi), (_, jlo, jhi) = dims["i"], dims["j"]
    n = (ihi - ilo + 1)*(jhi - jlo + 1)
    T = A.data.shape[0]
    if "k" in dims:
        _, klo, khi = dims["k"]
        return FArray(A.data.reshape(T, khi - klo + 1, n), name or A.name, i=(1, n), k=(klo, khi))
    return FArray(A.data.reshape(T, n), name or A.name, i=(1, n))


def xy_view(B, like, name=None):
    """An `(imt[,k])` dummy back as the caller's (i,j[,k]) actual argument `like` (same storage)."""
    return FArray(B.data.reshape(like.data.shape[:1] + B.data.shape[1:-1] + like.data.shape[-2:]),
                  name or like.name, tiled=like.tiled, _dims=like.dims)


def _gk(A, kidx):
    """A(i, kidx(i)) of an (imt, k) FArray for a per-point Fortran level index kidx [tile, 1, imt] -> [tile, 1, imt]."""
    (_, klo, _) = next(d for d in A.dims if d[0] == "k")
    return jnp.take_along_axis(A.data, kidx - klo, axis=1)


def _gz(Z, kidx):
    """Z(kidx(i)) of a not-tiled level vector (zgrid, hwide: k = 0..Nr+1) -> the shape of kidx."""
    (_, klo, _) = Z.dims[0]
    return Z.data[kidx - klo]


def _klev(lo, hi):
    """The Fortran level index of a k-vectorised nest DO k = lo, hi as a value, shaped [1, nk, 1, 1]."""
    return jnp.arange(lo, hi + 1, dtype=jnp.int32)[None, :, None, None]


def _swfrac(fact, worka):
    """CALL SWFRAC( imt, fact, worka, myTime, myIter, myThid ) (model/src/swfrac.F) on an (imt) value: the points
    are independent, so the [tile, 1, imt] value is passed flattened (lane M4LAB, the .NOT.KPPuseSWfrac3D arms)."""
    from mitjax.model.src.swfrac import swfrac
    return swfrac(worka.size, fact, worka.reshape(-1), 0., 0).reshape(worka.shape)


def _salt_plume_args(cfg, boplume, SPDepth, who):
    """#ifdef ALLOW_SALT_PLUME: boplume, SPDepth are arguments (:31-36, :215-220, :312-317), read only by the
    IF ( useSALT_PLUME ) arms (raise: SALT_PLUME_FRAC is not ported); SALT_PLUME_VOLUME / SALT_PLUME_SPLIT_BASIN
    raise (lane M4LAB: lab_sea compiles pkg/salt_plume with useSALT_PLUME = .FALSE.)."""
    sp = cfg.cpp.flag("ALLOW_SALT_PLUME", "KPP_OPTIONS.h")
    if sp != (boplume is not None and SPDepth is not None):
        raise TypeError(f"{who}: boplume, SPDepth are arguments exactly when ALLOW_SALT_PLUME is defined")
    if sp:
        for opt in ("SALT_PLUME_VOLUME", "SALT_PLUME_SPLIT_BASIN"):
            if cfg.cpp.flag(opt, "KPP_OPTIONS.h"):
                raise NotImplementedError(f"{who}: {opt} is not ported")
        if cfg.use_flag("useSALT_PLUME"):
            raise NotImplementedError(f"{who}: the useSALT_PLUME arms (SALT_PLUME_FRAC) are not ported")


def _sign(a, b):
    """Fortran SIGN(a, b) (gfortran -fsign-zero: SIGN(a, -0.) = -|a|) = copysign."""
    return jnp.copysign(a, b)


# ---------------------------------------------------------------------------------------------------------------

def kppmix(kmtj, shsq, dvsq, ustar, msk, bo, bosol, dbloc, Ritop, coriol, diffusKzS, diffusKzT, ikey, diffus, ghat,
           hbl, myTime, myIter, *, cfg, kpp, params, swatt=None, fp=None, boplume=None, SPDepth=None):
    """KPPMIX( kmtj, shsq, dvsq, ustar, msk, bo, bosol, dbloc, Ritop, coriol, diffusKzS, diffusKzT, ikey, diffus,
    ghat, hbl, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:28-305

    c     Main driver subroutine for kpp vertical mixing scheme and
    c     interface to greater ocean model
    c     written by: bill large,    june  6, 1994
    c     input  kmtj   number of vertical layers on this row; shsq (imt,Nr) (local velocity shear)^2;
    c     dvsq (imt,Nr) (velocity shear re sfc)^2; ustar (imt) surface friction velocity; bo, bosol (imt) surface
    c     turbulent / radiative buoyancy forcing; dbloc (imt,Nr) local delta buoyancy across interfaces; Ritop (imt,Nr)
    c     numerator of bulk Richardson Number; coriol (imt) Coriolis parameter; diffusKzS, diffusKzT (imt,Nr)
    c     background vertical diffusivities
    c     output diffus(imt,0:Nrp1,1..3) vertical viscosity, scalar diffusivity, temperature diffusivity; hbl (imt)
    c     boundary layer depth;  update ghat (imt,Nr) nonlocal transport coefficient (in: dbloc for RI_IWMIX)

    All arrays `(imt)`-shaped FArrays (module docstring); `diffus` a dict {md: FArray (imt, 0:Nrp1)} (prior: points
    the routine writes all of them); msk, myTime, myIter unused by the live code. Returns (diffus, ghat, hbl), and
    with SHORTWAVE_HEATING (diffus, ghat, hbl, kbl): the argument lists of :28-50 (swatt (imt,Nr+1) an input after
    coriol, :38-40, and kbl (imt) an output after hbl, :46-48, only #ifdef SHORTWAVE_HEATING; lane M4COL,
    1D_ocean_ice_column/code/CPP_OPTIONS.h:21 defines it). `swatt`, `fp` (forcing parameters: selectPenetratingSW)
    keyword-only, passed with SHORTWAVE_HEATING. ALLOW_SALT_PLUME (lane M4LAB): `boplume`, `SPDepth` keyword-only
    arguments (:31-36), passed on to BLDEPTH, read only with useSALT_PLUME (raise, `_salt_plume_args`).
    ALLOW_SHELFICE raises. RI_IWMIX, the zeroing
    below kmtj (:198-204, levels independent: k-vectorised), BLDEPTH, BLMIX, ENHANCE, and the combination with the
    interior values (:290-300, levels independent: k-vectorised)."""
    if cfg.cpp.flag("ALLOW_SHELFICE", "KPP_OPTIONS.h"):
        raise NotImplementedError("KPPMIX: ALLOW_SHELFICE is not ported")
    _salt_plume_args(cfg, boplume, SPDepth, "KPPMIX")
    sw_heating = cfg.cpp.flag("SHORTWAVE_HEATING", "KPP_OPTIONS.h")
    if sw_heating != (swatt is not None):                                      # :38-40 the argument list
        raise TypeError("KPPMIX: swatt is an argument exactly when SHORTWAVE_HEATING is defined")
    sz = cfg.size
    n = _imt_of(sz)
    Nr = sz.Nr
    diffus = ri_iwmix(kmtj, shsq, dbloc, ghat, diffusKzS, diffusKzT, ikey, diffus, cfg=cfg, kpp=kpp,
                      params=params)                                           # :180-184
    k, _, i = loops_kji((1, Nr + 1), (1, 1), (1, n))                           # :198-204 DO md / DO k=1,Nrp1
    below = _klev(1, Nr + 1) >= kmtj[i]
    for md in MD:
        diffus[md] = diffus[md].at[i, k].set(jnp.where(below, 0.0, diffus[md][i, k]))   # 0.0 (REAL*4, exact)
    hbl, bfsfc, stable, casea, kbl, Rib, sigma = bldepth(                      # :212-227
        kmtj, dvsq, dbloc, Ritop, ustar, bo, bosol, coriol, ikey, myTime, myIter, cfg=cfg, kpp=kpp, swatt=swatt,
        fp=fp, boplume=boplume, SPDepth=SPDepth)
    dkm1, blmc, ghat, sigma = blmix(ustar[loop_i(1, n)], bfsfc, hbl, stable, casea, diffus, kbl, ikey,   # :238-241
                                    ghat=ghat, sigma=sigma, cfg=cfg, kpp=kpp)
    ghat, blmc = enhance(dkm1, hbl, kbl, diffus, casea, ghat, blmc, cfg=cfg, kpp=kpp)        # :252-256
    k, _, i = loops_kji((1, Nr), (1, 1), (1, n))                               # :290-300
    above = _klev(1, Nr) < kbl[:, None]                                        # IF (k .LT. kbl(i))
    i1 = loop_i(1, n)
    d1 = MAX(blmc[1][i, k], params.viscArNr[1], p="b")                         # :293
    d2 = MAX(blmc[2][i, k], diffusKzS[i1, Nr][:, None], p="b")                 # :294
    d3 = MAX(blmc[3][i, k], diffusKzT[i1, Nr][:, None], p="b")                 # :295
    diffus[1] = diffus[1].at[i, k].set(jnp.where(above, d1, diffus[1][i, k]))
    diffus[2] = diffus[2].at[i, k].set(jnp.where(above, d2, diffus[2][i, k]))
    diffus[3] = diffus[3].at[i, k].set(jnp.where(above, d3, diffus[3][i, k]))
    ghat = ghat.at[i, k].set(jnp.where(above, ghat[i, k], 0.))                 # :297  0. _d 0
    if sw_heating:
        return diffus, ghat, hbl, kbl
    return diffus, ghat, hbl


def bldepth(kmtj, dvsq, dbloc, Ritop, ustar, bo, bosol, coriol, ikey, myTime, myIter, *, cfg, kpp, swatt=None,
            fp=None, boplume=None, SPDepth=None):
    """BLDEPTH( kmtj, dvsq, dbloc, Ritop, ustar, bo, bosol, coriol, ikey, hbl, bfsfc, stable, casea, kbl, Rib,
    sigma, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:309-919

    c     the oceanic planetary boundary layer depth, hbl, is determined as
    c     the shallowest depth where the bulk Richardson number is
    c     equal to the critical value, Ricr.
    c     bulk Richardson numbers are evaluated by computing velocity and
    c     buoyancy differences between values at zgrid(kl) < 0 and surface
    c     reference values.
    c     in this configuration, the reference values are equal to the
    c     values in the surface layer.
    c     when using a very fine vertical grid, these values should be
    c     computed as the vertical average of velocity and buoyancy from
    c     the surface down to epsilon*zgrid(kl).
    c     when the bulk Richardson number at k exceeds Ricr, hbl is
    c     linearly interpolated between grid levels zgrid(k) and zgrid(k-1).
    c     The water column and the surface forcing are diagnosed for
    c     stable/ustable forcing conditions, and where hbl is relative
    c     to grid points (caseA), so that conditional branches can be
    c     avoided in later subroutines.

    Returns (hbl, bfsfc, stable, casea, kbl, Rib, sigma); kbl int32 [tile, 1, imt]. ALLOW_SALT_PLUME (lane
    M4LAB): `boplume`, `SPDepth` arguments read only by the IF ( useSALT_PLUME ) arms (:530-565, :723-758, :859-894:
    raise). KPPBFSFC is a diagnostics local: not ported. SHORTWAVE_HEATING (lane M4COL; `swatt` (imt,Nr+1) an argument,
    :319-321, `fp` gives selectPenetratingSW): the three `bfsfc = bo + bosol*(1. - worka)` blocks (:485-528,
    :684-721, :823-858) with worka from swatt (KPPuseSWfrac3D, a static logical) or, lane M4LAB (lab_sea:
    KPPuseSWfrac3D = .FALSE., kpp_readparms.F:87 default), from SWFRAC (:504-511 worka = zgrid(kl) with fact = hbf,
    the same value on every point; :699-706 and :835-842 worka = hbl with fact = minusone = -1.0, :427 PARAMETER,
    REAL*4 exact);
    `0.5*` (:495) and `1.` REAL*4 literals, exact; the per-point k = kbl(i) reads swatt(i,k), swatt(i,k+1),
    zgrid(k), hwide(k) as gathers (kbl in 1..Nr: k+1 <= Nr+1 is in swatt's bounds). The DO kl = 2,Nr loop around
    WSCALE (:477-636) is a per-level caller (Python loop, body `level_k`); the two kbl searches (:653-657,
    :811-817) are recursions in kl (Python loops). LimitHblStable is a static logical. `Rib(i,1) = 0.` (:462) and the
    kl = 2..Nr levels are written; sigma is the value of the last kl iteration (KPPMIX passes it to BLMIX, which
    overwrites it before reading it)."""
    sz = cfg.size
    n = _imt_of(sz)
    Nr = sz.Nr
    p5, eins = 0.5, 1.0                                                        # :425-426 PARAMETER (REAL*4, exact)
    zgrid = kpp.zgrid
    i = loop_i(1, n)
    km = kmtj[i]                                                               # [tile, 1, imt] int32
    Rib = FArray(jnp.full_like(dvsq.data, jnp.nan), "Rib", i=(1, n), k=(1, Nr))   # tile-varying (scan carry)
    Rib = Rib.at[i, 1].set(0.)                                                 # :462  0. _d 0
    kbl = jnp.where(km < 1, 1, km)                                             # :463-464
    hbl = -_gz(zgrid, kbl)                                                     # :465-466
    us = ustar[i]
    bo_ = bo[i]
    sw_heating = cfg.cpp.flag("SHORTWAVE_HEATING", "KPP_OPTIONS.h")
    if sw_heating != (swatt is not None):                                      # :319-321 the argument list
        raise TypeError("BLDEPTH: swatt is an argument exactly when SHORTWAVE_HEATING is defined")
    sw_on = sw_heating and fp.selectPenetratingSW >= 1                         # :486, :685, :824 (INTEGER, static)
    _salt_plume_args(cfg, boplume, SPDepth, "BLDEPTH")
    minusone = -1.0                                                            # :427  PARAMETER ( minusone=-1.0 )
    bosol_ = bosol[i] if sw_on else None

    def worka_at_kbl(kb, hb):
        """:691-697 / :830-833  k = kbl(i); rFac = MAX( (hbl(i)+zgrid(k)+p5*hwide(k))/hwide(k), zeroRL );
        worka(i) = swatt(i,k) + rFac*(swatt(i,k+1)-swatt(i,k))"""
        hw = _gz(kpp.hwide, kb)
        rFac = (hb+_gz(zgrid, kb)+p5*hw)/hw
        return rFac, _gk(swatt, kb), _gk(swatt, kb+1)

    def level_k(kl, c):                                                        # :477-636  DO kl = 2, Nr
        Rib, = c
        if sw_on:                                                              # :486-520
            if kpp.KPPuseSWfrac3D:
                worka = 0.5*(swatt[i, kl] + swatt[i, kl+1])                    # :491-496
            else:                                                              # :504-511 (lane M4LAB)
                worka = _swfrac(kpp.hbf, jnp.broadcast_to(zgrid[kl], bo_.shape))   # worka(i) = zgrid(kl)
            bfsfc = bo_ + bosol_*(1. - worka)                                  # :518-520
        else:
            bfsfc = bo_                                                        # :523-525
        stable = p5 + _sign(p5, bfsfc)                                         # :577
        sigma = stable + (1. - stable) * kpp.epsilon                           # :578
        casea = -zgrid[kl] + jnp.zeros_like(bfsfc)                             # :580
        wm, ws = wscale(sigma, casea, us, bfsfc, cfg=cfg, kpp=kpp)             # :587-589
        bvsq = p5 * (dbloc[i, kl-1] / (zgrid[kl-1]-zgrid[kl])                  # :600-602
                     + dbloc[i, kl] / (zgrid[kl]-zgrid[kl+1]))
        nz = bvsq != 0.                                                        # :609  IF (bvsq .EQ. 0. _d 0)
        vtsq = jnp.where(nz, -zgrid[kl] * ws * safe_sqrt(jnp.abs(bvsq), nz) * kpp.Vtc, 0.)   # :610, :612
        tempVar1 = dvsq[i, kl] + vtsq                                          # :627
        tempVar2 = MAX(tempVar1, kpp.phepsi, p="a")                            # :631
        Rib = Rib.at[i, kl].set(Ritop[i, kl] / tempVar2)                       # :633
        return (Rib,), (bfsfc, stable, sigma, casea)

    from mitjax.ops.scan_k import scan_levels                                  # KERNEL_GUIDE §4: a level scan
    c = (Rib,)
    if Nr > 2:
        c = scan_levels(lambda kl, c: level_k(kl, c)[0], c, 2, Nr - 1)          # kl = 2 .. Nr-1
    c, last = level_k(Nr, c)                                                   # kl = Nr (sigma: its value)
    Rib, = c
    _, _, sigma, _ = last
    for kl in range(2, Nr + 1):                                                # :653-657 (recursion in kl)
        kbl = jnp.where((kbl == km) & (Rib[i, kl] > kpp.Ricr), kl, kbl)
    kl = kbl                                                                   # :663-674
    sel = (kl > 1) & (kl < km)
    kls = jnp.where(sel, kl, 2)                                                # a valid level on lanes not selected
    tempVar1 = _gk(Rib, kls) - _gk(Rib, kls - 1)                               # :667
    hbl = jnp.where(sel, -_gz(zgrid, kls-1) + safe_div((_gz(zgrid, kls-1) - _gz(zgrid, kls))   # :668-669
                                                       * (kpp.Ricr - _gk(Rib, kls-1)), tempVar1, sel), hbl)
    if sw_on and kpp.KPPuseSWfrac3D:                                           # :685-713
        rFac, s_k, s_kp1 = worka_at_kbl(kbl, hbl)
        rFac = MAX(rFac, 0.0, p="a")                                           # :696  zeroRL = 0.0 _d 0
        worka = s_k + rFac*(s_kp1-s_k)                                         # :697
        bfsfc = bo_ + bosol_ * (1. - worka)                                    # :711-713
    elif sw_on:                                                                # :699-706 (lane M4LAB)
        worka = _swfrac(minusone, hbl)                                         # worka(i) = hbl(i)
        bfsfc = bo_ + bosol_ * (1. - worka)                                    # :711-713
    else:
        bfsfc = bo_                                                            # :716-718
    stable = p5 + _sign(p5, bfsfc)                                             # :765
    bfsfc = _sign(eins, bfsfc) * MAX(kpp.phepsi, jnp.abs(bfsfc), p="b")        # :766
    if kpp.LimitHblStable:                                                     # :781-792
        pos = bfsfc > 0.0                                                      # :783  0.0 (REAL*4, exact)
        hekman = kpp.cekman * us / MAX(jnp.abs(coriol[i]), kpp.phepsi, p="b")  # :784
        hmonob = kpp.cmonob * us*us*us / kpp.vonk / bfsfc                      # :785-786 (|bfsfc| >= phepsi)
        hlimit = (stable * MIN(hekman, hmonob, p="b")                          # :787-788
                  + (stable - 1.) * zgrid[Nr])
        hbl = jnp.where(pos, MIN(hbl, hlimit, p="b"), hbl)                     # :789
    hbl = MAX(hbl, kpp.minKPPhbl, p="a")                                       # :799
    kbl = km                                                                   # :800
    for kl in range(2, Nr + 1):                                                # :811-817 (recursion in kl)
        kbl = jnp.where((kbl == km) & (-zgrid[kl] > hbl), kl, kbl)
    if sw_on and kpp.KPPuseSWfrac3D:                                           # :824-850
        rFac, s_k, s_kp1 = worka_at_kbl(kbl, hbl)
        rFac = MAX(rFac, 0.0, p="a")                                           # :832  zeroRL = 0.0 _d 0
        worka = s_k + rFac*(s_kp1-s_k)                                         # :833
        bfsfc = bo_ + bosol_ * (1. - worka)                                    # :848-850
    elif sw_on:                                                                # :835-842 (lane M4LAB)
        worka = _swfrac(minusone, hbl)                                         # worka(i) = hbl(i)
        bfsfc = bo_ + bosol_ * (1. - worka)                                    # :848-850
    else:
        bfsfc = bo_                                                            # :853-855
    stable = p5 + _sign(p5, bfsfc)                                             # :902
    bfsfc = _sign(eins, bfsfc) * MAX(kpp.phepsi, jnp.abs(bfsfc), p="b")        # :903
    casea = p5 + _sign(p5, -_gz(zgrid, kbl) - p5*_gz(kpp.hwide, kbl) - hbl)    # :910-914
    return hbl, bfsfc, stable, casea, kbl, Rib, sigma


def wscale(sigma, hbl, ustar, bfsfc, *, cfg, kpp):
    """WSCALE( sigma, hbl, ustar, bfsfc, wm, ws, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:923-1028

    c     compute turbulent velocity scales.
    c     use a 2D-lookup table for unstable conditions.
    c     input  sigma (imt) normalized depth (d/hbl); hbl (imt) boundary layer depth (m); ustar (imt) surface
    c     friction velocity (m/s); bfsfc (imt) total surface buoyancy flux (m^2/s^3)
    c     output wm, ws (imt) turbulent velocity scales at sigma

    Arguments are the values of the (imt) arrays ([tile, 1, imt] jax arrays, the caller's `A[i]`); returns (wm, ws).
    The points are independent. The table branch (zehat .LE. zmax, :978-1012): INT is a truncation toward zero, the
    clamps MIN(iz, nni), MAX(iz, 0) (INTEGER) are taken on the truncated float before the conversion, which gives
    gfortran's integer for every |zdiff/deltaz| < 2**31 and also where INT overflows below (cvttsd2si's -2**31 is
    clamped to 0 as the float is); `float(iz)` (REAL*4) is exact for 0 <= iz <= nni + 1. The table gathers read
    wmt(iz,ju) (storage [ju, iz]). The other branch's division (:1018) is guarded (its denominator can vanish on
    table lanes)."""
    zehat = kpp.vonk*sigma*hbl*bfsfc                                           # :976
    table = zehat <= kpp.zmax                                                  # :978
    zdiff = zehat - kpp.zmin                                                   # :980
    izf = jnp.maximum(jnp.minimum(jnp.trunc(zdiff / kpp.deltaz), float(NNI)), 0.)   # :991-993 MINMAX-RAW: INTEGER
    iz = izf.astype(jnp.int32)                                                 #   MIN/MAX of INT(...) (no tie, NaN)
    izp1 = iz + 1                                                              # :994
    udiff = ustar - kpp.umin                                                   # :996
    juf = jnp.maximum(jnp.minimum(jnp.trunc(udiff / kpp.deltau), float(NNJ)), 0.)   # :997-999 MINMAX-RAW: INTEGER
    ju = juf.astype(jnp.int32)                                                 #   MIN/MAX of INT(...) (no tie, NaN)
    jup1 = ju + 1                                                              # :1000
    zfrac = zdiff / kpp.deltaz - izf                                           # :1002  float(iz)
    ufrac = udiff / kpp.deltau - juf                                           # :1003  float(ju)
    fzfrac = 1. - zfrac                                                        # :1005
    wmt, wst = kpp.wmt.data, kpp.wst.data                                      # storage [j, i] = [ju, iz]
    wam = fzfrac * wmt[jup1, iz] + zfrac * wmt[jup1, izp1]                     # :1006
    wbm = fzfrac * wmt[ju, iz] + zfrac * wmt[ju, izp1]                         # :1007
    wm_t = (1. - ufrac) * wbm + ufrac * wam                                    # :1008
    was = fzfrac * wst[jup1, iz] + zfrac * wst[jup1, izp1]                     # :1010
    wbs = fzfrac * wst[ju, iz] + zfrac * wst[ju, izp1]                         # :1011
    ws_t = (1. - ufrac) * wbs + ufrac * was                                    # :1012
    u3 = ustar * ustar * ustar                                                 # :1016
    tempVar = u3 + kpp.conc1 * zehat                                           # :1017
    wm_s = safe_div(kpp.vonk * ustar * u3, tempVar, ~table)                    # :1018
    wm = jnp.where(table, wm_t, wm_s)
    ws = jnp.where(table, ws_t, wm_s)                                          # :1019  ws(i) = wm(i)
    return wm, ws


def ri_iwmix(kmtj, shsq, dbloc, dblocSm, diffusKzS, diffusKzT, ikey, diffus, *, cfg, kpp, params):
    """RI_IWMIX( kmtj, shsq, dbloc, dblocSm, diffusKzS, diffusKzT, ikey, diffus, myThid )
    @63cdc0b pkg/kpp/kpp_routines.F:1032-1226

    c     compute interior viscosity diffusivity coefficients due
    c     to shear instability (dependent on a local Richardson number),
    c     to background internal wave activity, and
    c     to static instability (local Richardson number < 0).

    Returns diffus {md: FArray (imt, 0:Nrp1)}. num_v_smooth_Ri = 0 (KPP_READPARMS refuses > 0: Z121 not ported), no
    ALLOW_KPP_VERTICALLY_SMOOTH, no KPP_SCALE_SHEARMIXING, no EXCLUDE_KPP_SHEAR_MIX (raise); ALLOW_AUTODIFF (lane
    M4ADCOL): the diffus reset :1105-1115 and the inAdMode arm :1197-1205 (below). The
    first DO ki (:1117-1135) is a recursion in ki (the ELSEIF copies level ki-1): Python loop; the second (:1165-1212)
    has independent levels: Python loop over ki (kp1 = MIN(ki+1,Nr) a host integer)."""
    for opt in ("ALLOW_KPP_VERTICALLY_SMOOTH", "KPP_SMOOTH_REGULARISATION", "KPP_SCALE_SHEARMIXING",
                "EXCLUDE_KPP_SHEAR_MIX"):
        if cfg.cpp.flag(opt, "KPP_OPTIONS.h"):
            raise NotImplementedError(f"RI_IWMIX: {opt} is not ported")
    # ALLOW_AUTODIFF (lane M4ADCOL): :1197-1201 `IF ( inAdMode .AND. .NOT. inAdExact )` -- inAdMode is .FALSE. in the
    # forward (AUTODIFF_INADMODE_SET is empty, autodiff_inadmode_set.F), so the ELSE arm (:1203-1205) runs; in TAF's
    # adjoint inAdMode = .TRUE., and the Model refuses inAdExact = .FALSE. (pkg/autodiff/autodiff_readparms.
    # adjoint_mode_check), so the derivative is of the same arm
    sz = cfg.size
    n = _imt_of(sz)
    Nr = sz.Nr
    zgrid = kpp.zgrid
    c1, c0 = 1.0, 0.0                                                          # :1097-1098  1. _d 0, 0. _d 0
    i = loop_i(1, n)
    km = kmtj[i]
    diffus = dict(diffus)
    if cfg.cpp.flag("ALLOW_AUTODIFF"):                                         # :1105-1115 (lane M4ADCOL)
        # "break data flow dependence on diffus": REAL*4 literals 0.0 / 0. (exact zero); every value set here is
        # overwritten below before it is read (measured on the stage gate), the statements are kept literal
        diffus[1] = diffus[1].at[loop_i(1, 1), 1].set(0.0)                     # :1107  diffus(1,1,1) = 0.0
        for ki in range(1, Nr + 1):                                            # :1108-1114
            for md in (1, 2, 3):
                diffus[md] = diffus[md].at[i, ki].set(0.)
    for ki in range(1, Nr + 1):                                                # :1117-1135
        land = km <= 1                                                         # :1119
        deep = ki >= km                                                        # :1122
        d1 = (dblocSm[i, ki] * (zgrid[ki]-zgrid[ki+1])                         # :1126, :1130
              / MAX(shsq[i, ki], kpp.phepsi, p="b"))
        d2 = dbloc[i, ki] / (zgrid[ki]-zgrid[ki+1])                            # :1132
        diffus[1] = diffus[1].at[i, ki].set(jnp.where(land, 0., jnp.where(deep, diffus[1][i, ki-1], d1)))
        diffus[2] = diffus[2].at[i, ki].set(jnp.where(land, 0., jnp.where(deep, diffus[2][i, ki-1], d2)))
    for ki in range(1, Nr + 1):                                                # :1165-1212
        Rig = MAX(diffus[2][i, ki], kpp.BVSQcon, p="a")                        # :1170
        ratio = MIN((kpp.BVSQcon - Rig) / kpp.BVSQcon, c1, p="b")              # :1171
        fcon = c1 - ratio * ratio                                              # :1172
        fcon = fcon * fcon * fcon                                              # :1173
        Rig = MAX(diffus[1][i, ki], c0, p="b")                                 # :1177
        ratio = MIN(Rig / kpp.Riinfty, c1, p="b")                              # :1178
        fRi = c1 - ratio * ratio                                               # :1179
        fRi = fRi * fRi * fRi                                                  # :1180
        kp1 = min(ki + 1, Nr)                                                  # :1191  MINMAX-INT: host level index
        diffus[1] = diffus[1].at[i, ki].set(params.viscArNr[1] + fcon*kpp.difmcon + fRi*kpp.difm0)   # :1206
        diffus[2] = diffus[2].at[i, ki].set(diffusKzS[i, kp1]+fcon*kpp.difscon+fRi*kpp.difs0)       # :1207
        diffus[3] = diffus[3].at[i, ki].set(diffusKzT[i, kp1]+fcon*kpp.diftcon+fRi*kpp.dift0)       # :1208
    for md in MD:                                                              # :1217-1221
        diffus[md] = diffus[md].at[i, 0].set(c0)
    return diffus


def smooth_horiz(k, fld, *, cfg, grid):
    """SMOOTH_HORIZ( k, bi, bj, fld, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:1311-1391

    c     Apply horizontal smoothing to global _RL 2-D array
    c     k      : vertical index used for masking
    c     fld    : 2-D array to be smoothed

    Lane M4COL (KPP_CALC's KPP_SMOOTH_DBLOC arm, the default pkg/kpp/KPP_OPTIONS.h:22). `fld` a
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy) FArray (the caller's level section, sequence association); `k` a Python int or a
    traced level (KIdx, KPP_CALC's level scan). Returns fld with the points iMin..iMax, jMin..jMax (:1341-1342,
    iMin=2-OLx, iMax=sNx+OLx-1) replaced by fld_tmp (:1382-1386); the other points keep their values. The points
    are independent (fld_tmp is a separate array: every point reads the unsmoothed fld). The IF (:1363-1377) is a
    `where` with the division by tempVar guarded on the lanes that keep fld (tempVar < p25, e.g. 0 on land)."""
    sz = cfg.size
    iMin, iMax, jMin, jMax = 2-sz.OLx, sz.sNx+sz.OLx-1, 2-sz.OLy, sz.sNy+sz.OLy-1   # :1341-1342
    p25, p125, p0625 = 0.25, 0.125, 0.0625                                      # :1344-1345 (REAL*4, exact)
    m = grid.maskC
    j = loop_j(jMin, jMax)                                                      # :1347-1379
    i = loop_i(iMin, iMax)
    tempVar = (p25   *   m[i  , j  , k]   +                                    # :1353-1362
               p125  * (m[i-1, j  , k]   +
                        m[i+1, j  , k]   +
                        m[i  , j-1, k]   +
                        m[i  , j+1, k]) +
               p0625 * (m[i-1, j-1, k]   +
                        m[i-1, j+1, k]   +
                        m[i+1, j-1, k]   +
                        m[i+1, j+1, k]))
    smooth = tempVar >= p25                                                     # :1363
    num = (p25  * fld[i  , j  ]*m[i  , j  , k] +                                # :1364-1374
           p125 * (fld[i-1, j  ]*m[i-1, j  , k] +
                   fld[i+1, j  ]*m[i+1, j  , k] +
                   fld[i  , j-1]*m[i  , j-1, k] +
                   fld[i  , j+1]*m[i  , j+1, k]) +
           p0625 * (fld[i-1, j-1]*m[i-1, j-1, k] +
                    fld[i-1, j+1]*m[i-1, j+1, k] +
                    fld[i+1, j-1]*m[i+1, j-1, k] +
                    fld[i+1, j+1]*m[i+1, j+1, k]))
    fld_tmp = jnp.where(smooth, safe_div(num, tempVar, smooth), fld[i, j])     # :1363-1377
    return fld.at[i, j].set(fld_tmp)                                           # :1382-1386


def blmix(ustar, bfsfc, hbl, stable, casea, diffus, kbl, ikey, *, ghat, sigma, cfg, kpp):
    """BLMIX( ustar, bfsfc, hbl, stable, casea, diffus, kbl, dkm1, blmc, ghat, sigma, ikey, myThid )
    @63cdc0b pkg/kpp/kpp_routines.F:1395-1692

    c     mixing coefficients within boundary layer depend on surface
    c     forcing and the magnitude and gradient of interior mixing below
    c     the boundary layer ("matching").
    c     caution: if mixing bottoms out at hbl = -zgrid(Nr) then
    c     fictitious layer at Nrp1 is needed with small but finite width
    c     hwide(Nrp1) (eg. epsln = 1.e-20).
    c     output dkm1 (imt,mdiff) boundary layer difs at kbl-1 level; blmc (imt,Nr,mdiff) boundary layer mixing
    c     coefficients; sigma (imt) normalized depth (d / hbl)

    ustar, bfsfc, hbl, stable, casea: [tile, 1, imt] values; kbl int32 [tile, 1, imt]; `ghat`, `sigma` are output
    arguments whose priors KPPMIX passes (ghat FArray (imt,Nr), sigma unused before it is set). Returns
    (dkm1 {md: [tile,1,imt]}, blmc {md: FArray (imt, Nr)}, ghat, sigma). The live arms: matching of diffusivities
    and derivatives (no KPP_DO_NOT_MATCH_*), no KPP_SMOOTH_REGULARISATION (raise). The DO ki loop (:1585-1647) is a
    per-level caller of WSCALE (Python loop, body `level_k`); the other loops are over points."""
    for opt in ("KPP_DO_NOT_MATCH_DIFFUSIVITIES", "KPP_DO_NOT_MATCH_DERIVATIVES", "KPP_SMOOTH_REGULARISATION"):
        if cfg.cpp.flag(opt, "KPP_OPTIONS.h"):
            raise NotImplementedError(f"BLMIX: {opt} is not ported")
    sz = cfg.size
    n = _imt_of(sz)
    Nr = sz.Nr
    p0, eins = 0.0, 1.0                                                        # :1466-1467 PARAMETER
    zgrid, hwide = kpp.zgrid, kpp.hwide
    i = loop_i(1, n)
    sigma = stable * 1.0 + (1. - stable) * kpp.epsilon                         # :1477-1479
    wm, ws = wscale(sigma, hbl, ustar, bfsfc, cfg=cfg, kpp=kpp)                # :1484-1486
    wm = _sign(eins, wm) * MAX(kpp.phepsi, jnp.abs(wm), p="b")                 # :1493
    ws = _sign(eins, ws) * MAX(kpp.phepsi, jnp.abs(ws), p="b")                 # :1494
    ia = jnp.trunc(casea + kpp.phepsi).astype(jnp.int32)                       # :1503-1504  INT(caseA(i)+phepsi)
    kn = ia * (kbl - 1) + (1 - ia) * kbl
    hw_kn, hw_kn1 = _gz(hwide, kn), _gz(hwide, kn + 1)
    delhat = 0.5*hw_kn - _gz(zgrid, kn) - hbl                                  # :1525
    R = 1.0 - delhat / hw_kn                                                   # :1526
    p_, h_ = {}, {}
    for md in MD:                                                              # :1527-1547
        dvdzup = (_gk(diffus[md], kn-1) - _gk(diffus[md], kn)) / hw_kn
        dvdzdn = (_gk(diffus[md], kn) - _gk(diffus[md], kn+1)) / hw_kn1
        p_[md] = 0.5 * ((1.-R) * (dvdzup + jnp.abs(dvdzup)) +
                        R * (dvdzdn + jnp.abs(dvdzdn)))
        h_[md] = _gk(diffus[md], kn) + p_[md] * delhat
    viscp, difsp, diftp = p_[1], p_[2], p_[3]
    visch, difsh, difth = h_[1], h_[2], h_[3]
    f1 = (stable * kpp.conc1 * bfsfc /                                         # :1550-1554
          MAX(powi(ustar, 4), kpp.phepsi, p="a"))
    gat1m = visch / hbl / wm                                                   # :1556
    dat1m = -viscp / wm + f1 * visch                                           # :1557
    gat1s = difsh / hbl / ws                                                   # :1559
    dat1s = -difsp / ws + f1 * difsh                                           # :1560
    gat1t = difth / hbl / ws                                                   # :1562
    dat1t = -diftp / ws + f1 * difth                                           # :1563
    dat1m = MIN(dat1m, p0, p="a")                                              # :1575
    dat1s = MIN(dat1s, p0, p="a")                                              # :1576
    dat1t = MIN(dat1t, p0, p="a")                                              # :1577
    blmc = {md: FArray(jnp.full_like(ghat.data, jnp.nan), "blmc", i=(1, n), k=(1, Nr)) for md in MD}   # scan carry

    def level_k(ki, c):                                                        # :1585-1647  DO ki = 1, Nr
        blmc, ghat = c
        sig = (-zgrid[ki] + 0.5 * hwide[ki]) / hbl                             # :1596
        sigma = stable*sig + (1.-stable)*MIN(sig, kpp.epsilon, p="b")          # :1597
        wm, ws = wscale(sigma, hbl, ustar, bfsfc, cfg=cfg, kpp=kpp)            # :1606-1608
        sig = (-zgrid[ki] + 0.5 * hwide[ki]) / hbl                             # :1619
        a1 = sig - 2.                                                          # :1620
        a2 = 3. - 2. * sig                                                     # :1621
        a3 = sig - 1.                                                          # :1622
        Gm = a1 + a2 * gat1m + a3 * dat1m                                      # :1624
        Gs = a1 + a2 * gat1s + a3 * dat1s                                      # :1625
        Gt = a1 + a2 * gat1t + a3 * dat1t                                      # :1626
        blmc = dict(blmc)
        blmc[1] = blmc[1].at[i, ki].set(hbl * wm * sig * (1. + sig * Gm))      # :1632
        blmc[2] = blmc[2].at[i, ki].set(hbl * ws * sig * (1. + sig * Gs))      # :1633
        blmc[3] = blmc[3].at[i, ki].set(hbl * ws * sig * (1. + sig * Gt))      # :1634
        tempVar = ws * hbl                                                     # :1640
        ghat = ghat.at[i, ki].set((1.-stable) * kpp.cg / MAX(kpp.phepsi, tempVar, p="b"))   # :1644
        return (blmc, ghat), sigma

    from mitjax.ops.scan_k import scan_levels                                  # KERNEL_GUIDE §4: a level scan
    c = scan_levels(lambda ki, c: level_k(ki, c)[0], (blmc, ghat), 1, Nr)       # DO ki = 1, Nr
    blmc, ghat = c                                                             # (sigma: rewritten at :1656)
    kl = kbl                                                                   # :1653-1658
    sig = -_gz(zgrid, kl-1) / hbl
    sigma = (stable * sig
             + (1. - stable) * MIN(sig, kpp.epsilon, p="b"))                   # :1656-1657
    wm, ws = wscale(sigma, hbl, ustar, bfsfc, cfg=cfg, kpp=kpp)                # :1667-1669
    sig = -_gz(zgrid, kl-1) / hbl                                              # :1677
    a1 = sig - 2.                                                              # :1678
    a2 = 3. - 2. * sig                                                         # :1679
    a3 = sig - 1.                                                              # :1680
    Gm = a1 + a2 * gat1m + a3 * dat1m                                          # :1681
    Gs = a1 + a2 * gat1s + a3 * dat1s                                          # :1682
    Gt = a1 + a2 * gat1t + a3 * dat1t                                          # :1683
    dkm1 = {1: hbl * wm * sig * (1. + sig * Gm),                               # :1684
            2: hbl * ws * sig * (1. + sig * Gs),                               # :1685
            3: hbl * ws * sig * (1. + sig * Gt)}                               # :1686
    return dkm1, blmc, ghat, sigma


def enhance(dkm1, hbl, kbl, diffus, casea, ghat, blmc, *, cfg, kpp):
    """ENHANCE( dkm1, hbl, kbl, diffus, casea, ghat, blmc, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:1696-1762

    c     enhance the diffusivity at the kbl-.5 interface

    Returns (ghat, blmc). Per point: ki = kbl-1 and, where 1 <= ki < Nr, level ki of blmc and ghat is rewritten
    (:1739-1757); the write at a per-point level is a select over the levels (the other levels keep their values).
    The gathers read a valid level (ki = 1) on the lanes the IF does not select."""
    sz = cfg.size
    n = _imt_of(sz)
    Nr = sz.Nr
    zgrid = kpp.zgrid
    ki = kbl - 1                                                               # :1740
    sel = (ki >= 1) & (ki < Nr)                                                # :1741
    kis = jnp.where(sel, ki, 1)
    delta = (hbl + _gz(zgrid, kis)) / (_gz(zgrid, kis) - _gz(zgrid, kis + 1))   # :1742
    k, _, i = loops_kji((1, Nr), (1, 1), (1, n))
    at_ki = (_klev(1, Nr) == kis[:, None]) & sel[:, None]                      # the level ki of each point
    blmc = dict(blmc)
    for md in MD:                                                              # :1743-1754
        dkmp5 = casea * _gk(diffus[md], kis) + (1. - casea) * _gk(blmc[md], kis)   # :1744-1745
        dstar = powi(1. - delta, 2) * dkm1[md] + powi(delta, 2) * dkmp5        # :1750-1751
        new = (1. - delta) * _gk(diffus[md], kis) + delta * dstar              # :1752-1753
        blmc[md] = blmc[md].at[i, k].set(jnp.where(at_ki, new[:, None], blmc[md][i, k]))
    newg = (1. - casea) * _gk(ghat, kis)                                       # :1755
    ghat = ghat.at[i, k].set(jnp.where(at_ki, newg[:, None], ghat[i, k]))
    return ghat, blmc


def statekpp(ikey, *, cfg, grid, params, eos, state):
    """STATEKPP( RHO1, DBLOC, DBSFC, TTALPHA, SSBETA, ikey, bi, bj, myThid )   @63cdc0b pkg/kpp/kpp_routines.F:1766-1958

    c     compute all necessary input arrays for the kpp mixing scheme
    c     output: rho1 (nx,ny) potential density of surface layer (kg/m^3); dbloc (nx,ny,Nr) local buoyancy gradient
    c     at Nr interfaces g/rho{k+1,k+1} * [ drho{k,k+1}-drho{k+1,k+1} ] (m/s^2); dbsfc (nx,ny,Nr) buoyancy
    c     difference with respect to the surface g * [ drho{1,k}/rho{1,k} - drho{k,k}/rho{k,k} ] (m/s^2);
    c     ttalpha (nx,ny,Nr+1) d(rho)/ d(potential temperature) (kg/m^3/C); ssbeta (nx,ny,Nr+1) d(rho) / d(salinity)
    c     (kg/m^3/PSU)

    Returns (RHO1, DBLOC, DBSFC, TTALPHA, SSBETA) as (i,j[,k]) FArrays (every point written). theta, salt from `state`
    (DYNVARS.h). The DO k = 2,Nr loop (:1879-1937) is a per-level caller of FIND_RHO_2D, FIND_ALPHA, FIND_BETA (a
    level scan, KERNEL_GUIDE §4). FIND_ALPHA / FIND_BETA get cfg, grid, state (their MDJWF arm reads theta, salt and
    PRESSURE_FOR_EOS' totPhiHyd). No KPP_AUTODIFF_MORE_STORE / ALLOW_AUTODIFF (TAF directives only). `DBSFC(i,j,1) =
    0.` (REAL*4, exact)."""
    from mitjax.model.src.find_alpha import find_alpha, find_beta
    from mitjax.model.src.find_rho import find_rho_2d
    sz = cfg.size
    Nr = sz.Nr
    lo_i, hi_i, lo_j, hi_j = 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy
    th, sa = state.theta, state.salt

    def lev(A, k):                                       # A(1-OLx,1-OLy,k,bi,bj) as a 2-D dummy; k may be a KIdx
        kk = k - 1
        return FArray(A.data[:, getattr(kk, "value", kk)], A.name, i=(lo_i, hi_i), j=(lo_j, hi_j))

    xy = lev(th, 1).local("WORK")
    j = loop_j(lo_j, hi_j)
    i = loop_i(lo_i, hi_i)
    nan3 = jnp.full_like(th.data, jnp.nan)     # from the tiled theta: tile-varying under shard_map (DBLOC/DBSFC are
    nan3p = jnp.full(th.data.shape[:1] + (Nr + 1,) + th.data.shape[2:], jnp.nan)   # scan carries written per tile)
    decl = dict(i=(lo_i, hi_i), j=(lo_j, hi_j))
    DBLOC = FArray(nan3, "DBLOC", **decl, k=(1, Nr))
    DBSFC = FArray(nan3, "DBSFC", **decl, k=(1, Nr))
    TTALPHA = FArray(nan3p, "TTALPHA", **decl, k=(1, Nr + 1))
    SSBETA = FArray(nan3p, "SSBETA", **decl, k=(1, Nr + 1))
    k = 1                                                                      # :1843
    WORK1 = find_rho_2d(lo_i, hi_i, lo_j, hi_j, 1, lev(th, k), lev(sa, k), xy.local("WORK1"), k,   # :1849-1853
                        cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    WORK2 = find_alpha(lo_i, hi_i, lo_j, hi_j, 1, 1, xy.local("WORK2"), params=params, eos=eos,    # :1859-1861
                       cfg=cfg, grid=grid, state=state)
    WORK3 = find_beta(lo_i, hi_i, lo_j, hi_j, 1, 1, xy.local("WORK3"), params=params, eos=eos,     # :1863-1865
                      cfg=cfg, grid=grid, state=state)
    RHO1 = xy.local("RHO1").at[i, j].set(WORK1[i, j] + params.rhoConst)        # :1869
    TTALPHA = TTALPHA.at[i, j, 1].set(WORK2[i, j])                             # :1870
    SSBETA = SSBETA.at[i, j, 1].set(WORK3[i, j])                               # :1871
    DBSFC = DBSFC.at[i, j, 1].set(0.)                                          # :1872

    def level_k(k, c):                                                         # :1879-1937  DO k = 2, Nr
        TTALPHA, SSBETA, DBLOC, DBSFC = c
        RHOK = find_rho_2d(lo_i, hi_i, lo_j, hi_j, k, lev(th, k), lev(sa, k), xy.local("RHOK"), k,   # :1886-1890
                           cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        RHOKM1 = find_rho_2d(lo_i, hi_i, lo_j, hi_j, k, lev(th, k-1), lev(sa, k-1), xy.local("RHOKM1"),   # :1896-1900
                             k-1, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        RHO1K = find_rho_2d(lo_i, hi_i, lo_j, hi_j, k, lev(th, 1), lev(sa, 1), xy.local("RHO1K"), 1,   # :1906-1910
                            cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        W1 = find_alpha(lo_i, hi_i, lo_j, hi_j, k, k, xy.local("WORK1"), params=params, eos=eos,    # :1918-1920
                       cfg=cfg, grid=grid, state=state)
        W2 = find_beta(lo_i, hi_i, lo_j, hi_j, k, k, xy.local("WORK2"), params=params, eos=eos,     # :1922-1924
                       cfg=cfg, grid=grid, state=state)
        TTALPHA = TTALPHA.at[i, j, k].set(W1[i, j])                            # :1928
        SSBETA = SSBETA.at[i, j, k].set(W2[i, j])                              # :1929
        DBLOC = DBLOC.at[i, j, k-1].set(params.gravity * (RHOK[i, j] - RHOKM1[i, j]) /   # :1930-1931
                                        (RHOK[i, j] + params.rhoConst))
        DBSFC = DBSFC.at[i, j, k].set(params.gravity * (RHOK[i, j] - RHO1K[i, j]) /      # :1932-1933
                                      (RHOK[i, j] + params.rhoConst))
        return TTALPHA, SSBETA, DBLOC, DBSFC

    from mitjax.ops.scan_k import scan_levels                                  # KERNEL_GUIDE §4: a level scan
    TTALPHA, SSBETA, DBLOC, DBSFC = scan_levels(level_k, (TTALPHA, SSBETA, DBLOC, DBSFC), 2, Nr)   # DO k = 2, Nr
    TTALPHA = TTALPHA.at[i, j, Nr+1].set(TTALPHA[i, j, Nr])                    # :1942
    SSBETA = SSBETA.at[i, j, Nr+1].set(SSBETA[i, j, Nr])                       # :1943
    DBLOC = DBLOC.at[i, j, Nr].set(0.)                                         # :1944
    return RHO1, DBLOC, DBSFC, TTALPHA, SSBETA


def kpp_doublediff(TTALPHA, SSBETA, kappaRT, kappaRS, ikey, iMin, iMax, jMin, jMax, *, cfg, kpp, state):
    """KPP_DOUBLEDIFF( TTALPHA, SSBETA, kappaRT, kappaRS, ikey, iMin, iMax, jMin, jMax, bi, bj, myThid )
    @63cdc0b pkg/kpp/kpp_routines.F:1962-2121

    C     Calculate additional diffusivities due to double diffusion
    C     Large et al. 1994, eq. 30-34; salt fingering (Rrho > 1) and diffusive convection (0 < Rrho < 1)

    Returns (kappaRT, kappaRS) (i,j,k) FArrays. theta, salt from `state`. Every level k is independent (it reads the
    inputs at km1 = MAX(k-1,1) and k and updates its own level): k-vectorised over 2..Nr after the k = 1 level,
    which only adds nuddt = nudds = 0. _d 0 (the IF ( k .GT. 1 ) block is skipped). `exp` is glibc's
    (libm.glibc_exp); the divisions are guarded on the lanes their branch does not select (Rrho = 1 there, so the
    nested exponentials stay finite)."""
    sz = cfg.size
    Nr = sz.Nr
    numol = 1.5e-06                                                            # :2036  1.5 _d -06
    rFac = 1.0 / (kpp.Rrho0 - 1.0)                                             # :2037
    th, sa = state.theta, state.salt
    # k = 1 (Km1 = 1): alphaDT = betaDS = 0 contributions; nuddt = nudds = 0. _d 0 on every point (:2056-2057)
    j1 = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i1 = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    kappaRT = kappaRT.at[i1, j1, 1].set(kappaRT[i1, j1, 1] + 0.)               # :2106 (+ nuddt = 0. _d 0)
    kappaRS = kappaRS.at[i1, j1, 1].set(kappaRS[i1, j1, 1] + 0.)               # :2107
    if Nr < 2:
        return kappaRT, kappaRS
    k, j, i = loops_kji((2, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :2048-2059
    alphaDT = ((th[i, j, k-1] - th[i, j, k])                                   # :2052-2053 (km1 = k-1)
               * 0.5 * jnp.abs(TTALPHA[i, j, k-1] + TTALPHA[i, j, k]))
    betaDS = ((sa[i, j, k-1] - sa[i, j, k])                                    # :2054-2055
              * 0.5 * (SSBETA[i, j, k-1] + SSBETA[i, j, k]))
    finger = (alphaDT > betaDS) & (betaDS > 0.)                                # :2067-2068
    diffu = (alphaDT < 0.) & (betaDS < 0.) & (alphaDT > betaDS)                # :2076-2078
    Rrho_f = MIN(safe_div(alphaDT, betaDS, finger, fill=1.0), kpp.Rrho0, p="a")   # :2069
    nutmp = (1.0 - (Rrho_f - 1.0) * rFac)                                      # :2072
    nudds_f = kpp.dsfmax * nutmp * nutmp * nutmp                               # :2073
    nuddt_f = 0.7 * nudds_f                                                    # :2075  0.7 _d 0
    Rrho_d = safe_div(alphaDT, betaDS, diffu, fill=1.0)                        # :2083
    nuddt_d = (numol * 0.909                                                   # :2085-2087
               * glibc_exp(4.6 * glibc_exp(-5.4 * (safe_div(1.0, Rrho_d, diffu, fill=1.0) - 1.0))))
    nudds_d = nuddt_d * MAX(0.15 * Rrho_d,                                     # :2092-2093
                            1.85 * Rrho_d - 0.85, p="b")
    nuddt = jnp.where(finger, nuddt_f, jnp.where(diffu, nuddt_d, 0.))         # :2063-2098 (ELSE: 0. _d 0 kept)
    nudds = jnp.where(finger, nudds_f, jnp.where(diffu, nudds_d, 0.))
    inner = jnp.zeros_like(nuddt, dtype=bool)
    jI, iI = slice(jMin - (1-sz.OLy), jMax - (1-sz.OLy) + 1), slice(iMin - (1-sz.OLx), iMax - (1-sz.OLx) + 1)
    inner = inner.at[:, :, jI, iI].set(True)                                   # DO j = jMin, jMax / DO i = iMin, iMax
    nuddt = jnp.where(inner, nuddt, 0.)
    nudds = jnp.where(inner, nudds, 0.)
    kappaRT = kappaRT.at[i, j, k].set(kappaRT[i, j, k] + nuddt)                # :2104-2109
    kappaRS = kappaRS.at[i, j, k].set(kappaRS[i, j, k] + nudds)
    return kappaRT, kappaRS
