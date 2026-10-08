"""Elementary functions of the Fortran oracle's math libraries, for kernels whose gates are bitwise (plan Task 7c).

The master oracle (gfortran 11.2.0, -O0 -ffp-contract=off, glibc 2.28 = glibc-2.28-251.el8_10.40, /lib64/libm-2.28.so,
AMD EPYC 7763; reference/README.md) calls scalar libm and libgcc functions. Its imports, from `nm -D --undefined-only`
of the six M1 oracle binaries `$MJX_REFERENCE/bin/<exp>-<code>-63cdc0b-f48c062/mitgcmuv` (2026-10-01; symbol version
GLIBC_2.2.5, no libmvec: an -O0 build vectorises nothing):
    every binary  atan atan2 cos exp fmod log log10f lround pow sin tan __powidf2 (libgcc)
    some          round (diagnostics I/O), tanh (global_ocean.90x40x15, tutorial_global_oce_optim: gmredi), asin
                  (global_ocean.90x40x15: ggl90_idemix)
SQRT is the inline `sqrtsd` instruction (correctly rounded, as XLA's sqrt). Which of them an M1 run reaches (call
sites from `nm` of the build's objects, guards from master's sources and the runs' namelists):
    sin, cos      ini_spherical_polar_grid.F:103-236, ini_cori.F:92-96 (R2, R4, R5: spherical polar grids)
    tan           ini_spherical_polar_grid.F:245-247 tanPhiAtU/V (R2, R4, R5)
    pow           ini_spherical_polar_grid.F:259-261 cosFacU/V = |cos|**cosPower, cosPower = 0. (set_defaults.F:152,
                  no M1 run sets it): pow(x, 0) = 1 exactly; the other real-exponent sites (atm_kappa) are
                  atmosphere-only branches
    tanh          gmredi_slope_limit.F:565 (GM_taper_scheme 'dm95': R5; R4 uses 'gkw91')
    exp           advect_xy/code/ini_theta.F:67 (R3's own initial condition); swfrac.F needs SHORTWAVE_HEATING, which
                  no M1 build defines
    __powidf2     x**n with an INTEGER n, called even for constant n at -O0 (no powi expansion without optimisation):
                  gad_pqm_fun.F:243-270 (R3 pqm), ini_eos.F:320-323, the gad_ppm/pqm flux routines, mom_calc_visc.F
    not reached   atan (calc_3d_diffusivity.F:86-89: diffKrBL79/BLEQ unset), atan2 (rotate_spherical_polar_grid.F:
                  rotateGrid unset), asin (ggl90 off at run time in R4), log (port_rand.F), fmod/lround/round/log10f
                  (calendar, I/O, STDOUT formatting)
`test_libm.py` measures every function against that libm (ctypes, the library the oracle's own ldd.txt names) on
sampled arguments; `MEASURED` below is its result and the test fails when it changes.

Contents:
  - `glibc_exp`: libm's scalar `exp`, bit for bit, copied from the ECCO port's ops/libm.py (R) and re-verified
    against this glibc (tables and 13 constants equal the bytes of /lib64/libm-2.28.so; 0 mismatches on the sampled
    arguments): XLA's exp is not glibc's (measured: MEASURED["exp"]).
  - `glibc_tanh`: libm's `tanh` (fdlibm s_tanh.c + s_expm1.c), bit for bit, new: XLA's tanh is not glibc's
    (MEASURED["tanh"]) and R5's dm95 taper calls it.
  - `powi`: libgcc's `__powidf2` (right-to-left binary powering, then 1/y for n < 0), as written in
    libgcc2.c; for a static int n it is the same multiplication sequence as `x ** n` (lax.integer_pow), measured.
  - `M1_LIBM`: the function each M1 call site uses (jnp's where bitwise, the transcription where not).
  Not copied from R: `exp_libmvec` (libmvec's `_ZGVbN2v_exp`; no M1 binary imports libmvec).
Every bitwise claim needs the gate XLA flags (no FMA contraction, no algsimp; conftest.py, mitjax/xla_flags.py).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["glibc_exp", "glibc_tanh", "glibc_asin", "glibc_log10", "glibc_asin_constants_vs_libm", "fma_emulated", "exp_tables",
           "glibc_exp_tables_vs_libm", "powi", "Libm", "M1_LIBM", "MEASURED"]


# ------------------------------------------------------------------------------------------------ glibc 2.28 exp
#
# glibc_exp: bit-for-bit emulation of the exp() the gfortran oracle calls. RHEL8 glibc 2.28, /lib64/libm-2.28.so:
# `exp` -> `__exp_finite` (ifunc at 0x28380) -> on AMD EPYC 7763 (FMA + AVX2) the FMA-compiled IBM Accurate
# Mathematical Library exp at 0x771e0 (sysdeps/ieee754/dbl-64/e_exp.c after the 2.28 slow-path removal; NOT correctly
# rounded: 11 of 1e5 random results differ from the correctly rounded exp). XLA's exp differs from it by 1-2 ulp at
# ~14 % of arguments. Paths (hx = high word of x, n = hx & 0x7fffffff), transcribed from the disassembly, every
# vfmadd/vfnmadd as an (emulated, exact) fused multiply-add:
#     n <= 0x3e2fffff                 (|x| < 2^-28)       1 + x                                  (0x77253)
#     0x3e2fffff < n <= 0x3ecfffff    (|x| < ~3.8e-6)    fma(fma(x, 0.5, 1), x, 1)              (0x77490)
#     0x3ecfffff < n <= 0x3f862e41    (|x| <= ~0.01085)  1 + (x + poly(x))                      (0x775a0)
#     0x3f862e41 < n <= 0x3ff0a2b1    (|x| <= ~1.0397)   accurate-table path                    (0x772b0)
#     0x3ff0a2b1 < n <= 0x40862001    (|x| <= ~708.0039) coar/fine table path (IBM __exp)       (0x774b0)
#     otherwise (near overflow/underflow, inf, nan)      jnp.exp FALLBACK, not bitwise          (0x77368, specials)
# The rounding-mode save/restore (vstmxcsr) is a no-op in round-to-nearest. Tables (.rodata, VA == file offset): coar
# (0xfa4a0) = exp(k*2^-9), k = -178..177, fine (0xf84a0) = exp(m*2^-18), m = 0..511, as (hi, lo), hi = exp rounded to
# 27 significant bits, lo = RN(exp - hi): regenerated below (decimal, 80 digits), bit-identical to the binary; the
# accurate table (0xf7c40) holds 67 + 67 Gal-type nodes x_k ~ +-(k+1)/64 (listed: they cannot be derived) and
# RN(exp(x_k)) (regenerated). `glibc_exp_tables_vs_libm()` compares all of them and the 13 constants with the binary.
# FMA without hardware FMA: TwoProduct (Veltkamp split) a*b = uh + ul, TwoSum c + uh = th + tl, then
# fma = RN(th + RO(tl + ul)) (Boldo & Melquiond 2008; round-to-odd from TwoSum + one integer step of the bit pattern).
# Needs IEEE arithmetic as written (the gate flags: no FMA contraction, no algsimp); bexp and base are read from the
# integer words of y and yy (exact identities) because algsimp folds (c + t) - c -> t. Measured in the ECCO port (R)
# on a compute node with 2e6 uniform arguments per range vs Python math.exp (same glibc, same CPU model): 0 mismatches
# in [-15,-0.075], [+-0.0108,+-1.0397], [1,8], [20,30], log-uniform |x| in [0.0108,708] and [1e-12,0.0108] under the
# gate flags; default XLA flags: <= 7 per 2e6 for |x| >= 0.0108, 1.9 % below (all <= 1 ulp). Cost ~28 ms per 1e6
# values (jnp.exp 1.2 ms).
# Re-verified for mitjax 2026-10-01 against the master oracle's libm (test_libm.py: tables, constants, 0 mismatches).

_E_THREE51 = float.fromhex("0x1.8p+52")               # 0xd3320
_E_LOG2E = float.fromhex("0x1.71547652b82fep+0")      # 0xde4e0
_E_THREE33 = float.fromhex("0x1.8p+34")               # 0xfbae0
_E_LN_TWO1 = float.fromhex("0x1.62e42fefa3800p-1")    # 0xfbb00
_E_LN_TWO2 = float.fromhex("0x1.ef35793c76730p-45")   # 0xfbaf8
_E_P3 = float.fromhex("0x1.5555555555a0fp-3")         # 0xfbae8
_E_P2 = float.fromhex("0x1.00000000004dcp-1")         # 0xfbaf0
_E_C120 = float.fromhex("0x1.1111111111111p-7")       # 0xc55c8
_E_C24 = float.fromhex("0x1.5555555555555p-5")        # 0xc55d0
_E_C720 = float.fromhex("0x1.6c16c16c16c17p-10")      # 0xc55d8
_E_C6 = float.fromhex("0x1.5555555555555p-3")         # 0xc55e0
_E_HALF = 0.5                                         # 0xad0a8
_E_ONE = 1.0                                          # 0xac8b0
_E_CONST_ADDR = {0xd3320: _E_THREE51, 0xde4e0: _E_LOG2E, 0xfbae0: _E_THREE33, 0xfbb00: _E_LN_TWO1,
                 0xfbaf8: _E_LN_TWO2, 0xfbae8: _E_P3, 0xfbaf0: _E_P2, 0xc55c8: _E_C120, 0xc55d0: _E_C24,
                 0xc55d8: _E_C720, 0xc55e0: _E_C6, 0xad0a8: _E_HALF, 0xac8b0: _E_ONE}
_E_N_TINY, _E_N_SMALL, _E_N_POLY, _E_N_MED, _E_N_MAIN = 0x3e2fffff, 0x3ecfffff, 0x3f862e41, 0x3ff0a2b1, 0x40862001
_E_SPLIT = 134217729.0  # 2^27 + 1
# accurate-table nodes (0xf7c40, even entries): 67 positive, then 67 negative
_E_NODES_HEX = (
    "0x1.ffffffffffc82p-7", "0x1.fffffffffffdbp-6", "0x1.80000000000a0p-5", "0x1.fffffffffff79p-5",
    "0x1.3fffffffffffcp-4", "0x1.8000000000060p-4", "0x1.c000000000061p-4", "0x1.fffffffffffd6p-4",
    "0x1.1ffffffffff58p-3", "0x1.3ffffffffff75p-3", "0x1.5ffffffffff00p-3", "0x1.8000000000020p-3",
    "0x1.9ffffffffa629p-3", "0x1.c00000000000fp-3", "0x1.e00000000007fp-3", "0x1.0000000000072p-2",
    "0x1.0fffffffffecap-2", "0x1.1ffffffffff8fp-2", "0x1.300000000003bp-2", "0x1.4000000000034p-2",
    "0x1.4ffffffffff89p-2", "0x1.5ffffffffffe7p-2", "0x1.6ffffffffff78p-2", "0x1.7ffffffffff65p-2",
    "0x1.8ffffffffffd5p-2", "0x1.9ffffffffff6ep-2", "0x1.affffffffffc3p-2", "0x1.c000000000053p-2",
    "0x1.d00000000004dp-2", "0x1.e000000000096p-2", "0x1.efffffffffefap-2", "0x1.fffffffffffd0p-2",
    "0x1.0800000000002p-1", "0x1.100000000001fp-1", "0x1.17ffffffffff8p-1", "0x1.1fffffffffffap-1",
    "0x1.27fffffffffc4p-1", "0x1.2fffffffffffdp-1", "0x1.380000000001fp-1", "0x1.3ffffffffffd8p-1",
    "0x1.4800000000052p-1", "0x1.4ffffffffffc8p-1", "0x1.5800000000013p-1", "0x1.5ffffffffffbcp-1",
    "0x1.680000000002dp-1", "0x1.7000000000040p-1", "0x1.780000000004fp-1", "0x1.7ffffffffff6fp-1",
    "0x1.87fffffffffe5p-1", "0x1.9000000000035p-1", "0x1.97fffffffffb3p-1", "0x1.a000000000000p-1",
    "0x1.a80000000004ap-1", "0x1.affffffffffedp-1", "0x1.b7ffffffffffbp-1", "0x1.c00000000001dp-1",
    "0x1.c800000000079p-1", "0x1.cffffffffff51p-1", "0x1.d7fffffffff74p-1", "0x1.e000000000011p-1",
    "0x1.e80000000001ep-1", "0x1.effffffffff9ep-1", "0x1.f7fffffffffedp-1", "0x1.0000000000034p+0",
    "0x1.03fffffffffe2p+0", "0x1.07fffffffff4bp+0", "0x1.0bffffffffffdp+0", "-0x1.fffffffffffe4p-7",
    "-0x1.ffffffffffb0bp-6", "-0x1.7ffffffffffa7p-5", "-0x1.ffffffffffea8p-5", "-0x1.3ffffffffffb3p-4",
    "-0x1.7ffffffffffe3p-4", "-0x1.bffffffffff9ap-4", "-0x1.fffffffffff98p-4", "-0x1.1ffffffffffe9p-3",
    "-0x1.3ffffffffffe0p-3", "-0x1.5fffffffff553p-3", "-0x1.7ffffffffff8bp-3", "-0x1.9fffffffffe51p-3",
    "-0x1.bffffffffff6ep-3", "-0x1.dffffffffff7fp-3", "-0x1.fffffffffff7ap-3", "-0x1.0fffffffffffep-2",
    "-0x1.1ffffffffff41p-2", "-0x1.2ffffffffffbap-2", "-0x1.3fffffffffff8p-2", "-0x1.4ffffffffff90p-2",
    "-0x1.5ffffffffffdbp-2", "-0x1.6ffffffffff9ap-2", "-0x1.7ffffffffff9fp-2", "-0x1.8ffffffffffeep-2",
    "-0x1.9fffffffffc4ap-2", "-0x1.affffffffff30p-2", "-0x1.bfffffffffff0p-2", "-0x1.cfffffffffff3p-2",
    "-0x1.dfffffffffff3p-2", "-0x1.effffffffff80p-2", "-0x1.fffffffffffdfp-2", "-0x1.0800000000000p-1",
    "-0x1.0ffffffffffa4p-1", "-0x1.17fffffffff0ap-1", "-0x1.2000000000000p-1", "-0x1.27fffffffffbbp-1",
    "-0x1.2fffffffffe32p-1", "-0x1.37ffffffff042p-1", "-0x1.3ffffffffff77p-1", "-0x1.47fffffffff6bp-1",
    "-0x1.4fffffffffff1p-1", "-0x1.57ffffffffe02p-1", "-0x1.5ffffffffffe5p-1", "-0x1.67fffffffffb0p-1",
    "-0x1.6ffffffffffb2p-1", "-0x1.77fffffffff7fp-1", "-0x1.7ffffffffffe8p-1", "-0x1.87fffffffffc8p-1",
    "-0x1.8fffffffffb30p-1", "-0x1.97fffffffffefp-1", "-0x1.9ffffffffffa7p-1", "-0x1.a7fffffffffdcp-1",
    "-0x1.affffffffff95p-1", "-0x1.b7fffffffffcbp-1", "-0x1.bffffffffff32p-1", "-0x1.c7fffffffff6ap-1",
    "-0x1.cffffffffffb6p-1", "-0x1.d7fffffffffcap-1", "-0x1.dffffffffffcdp-1", "-0x1.e7ffffffffffbp-1",
    "-0x1.effffffffff88p-1", "-0x1.f7fffffffffbbp-1", "-0x1.fffffffffffdbp-1", "-0x1.03fffffffff00p+0",
    "-0x1.07ffffffffe6fp+0", "-0x1.0bfffffffffd6p+0",
)


def _glibc_exp_tables():
    """(coar[712], fine[1024], acc[268]) float64, regenerated from first principles (see the section comment)."""
    import decimal
    from fractions import Fraction

    def dec_exp(q):
        with decimal.localcontext() as ctx:
            ctx.prec = 80
            return Fraction((decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator)).exp())

    def rn_bits(v, nb):
        e = v.numerator.bit_length() - v.denominator.bit_length()
        while Fraction(2) ** e > v:
            e -= 1
        while Fraction(2) ** (e + 1) <= v:
            e += 1
        scale = Fraction(2) ** (nb - 1 - e)
        return Fraction(round(v * scale)) / scale

    def hi_lo(q):
        ex = dec_exp(q)
        hi = rn_bits(ex, 27)
        return float(hi), float(ex - hi)

    coar = np.empty(712)
    for n, k in enumerate(range(-178, 178)):
        coar[2 * n], coar[2 * n + 1] = hi_lo(Fraction(k, 2 ** 9))
    fine = np.empty(1024)
    for m in range(512):
        fine[2 * m], fine[2 * m + 1] = hi_lo(Fraction(m, 2 ** 18))
    acc = np.empty(268)
    for n, h in enumerate(_E_NODES_HEX):
        x = float.fromhex(h)
        acc[2 * n], acc[2 * n + 1] = x, float(dec_exp(Fraction(x)))
    return coar, fine, acc


_E_TABLES = _glibc_exp_tables()


def glibc_exp_tables_vs_libm(path="/lib64/libm-2.28.so"):
    """List of differences between the regenerated tables/constants and the bytes of the libm binary ([] = none)."""
    import struct

    b = open(path, "rb").read()
    bad = [hex(a) for a, v in _E_CONST_ADDR.items() if struct.unpack("<d", b[a:a + 8])[0] != v]
    coar, fine, acc = _E_TABLES
    for name, a, t in (("acc", 0xf7c40, acc), ("fine", 0xf84a0, fine), ("coar", 0xfa4a0, coar)):
        nd = int(np.sum(np.frombuffer(b[a:a + 8 * t.size], "<u8") != t.view(np.uint64)))
        if nd:
            bad.append(f"{name}: {nd} entries differ")
    return bad


def _f2i(v):
    return jax.lax.bitcast_convert_type(v, jnp.int64)


def _i2f(v):
    return jax.lax.bitcast_convert_type(v, jnp.float64)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):
    def split(v):
        c = _E_SPLIT * v
        vh = c - (c - v)
        return vh, v - vh

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def fma_emulated(a, b, c):
    """RN(a*b + c) in float64 without a hardware FMA (Boldo-Melquiond; exact barring overflow/underflow)."""
    a, b, c = (jnp.asarray(v, jnp.float64) for v in (a, b, c))
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    s, e = _two_sum(tl, ul)                                  # round-to-odd of tl + ul
    sb = _f2i(s)
    step = jnp.where(jnp.signbit(e) == jnp.signbit(s), jnp.int64(1), jnp.int64(-1))
    sb = jnp.where((e != 0) & ((sb & 1) == 0), sb + step, sb)
    return th + _i2f(sb)


def _exp_poly(d):
    """d + d^2*(1/2 + d/6) + d^4*(1/24 + d/120 + d^2/720) in the binary's FMA order (0x772b0 / 0x775a0)."""
    a = fma_emulated(d, _E_C120, _E_C24)
    b = fma_emulated(d, _E_C6, _E_HALF)
    d2 = d * d
    a = fma_emulated(d2, _E_C720, a)
    a = a * (d2 * d2)
    b = fma_emulated(b, d2, a)
    return b + d


def _signed_low(v):
    k = _f2i(v) & 0xffffffff
    return (k ^ 0x80000000) - 0x80000000


def _glibc_exp_impl(x):
    coar, fine, acc = (jnp.asarray(t, jnp.float64) for t in _E_TABLES)
    x = jnp.asarray(x, jnp.float64)
    hx = _f2i(x) >> 32
    n = hx & 0x7fffffff
    # main path (0x774b0)
    y = fma_emulated(x, _E_LOG2E, _E_THREE51)
    ky = _signed_low(y)
    bexp = ky.astype(jnp.float64)                            # == y - THREE51 exactly (0x774d4)
    t = fma_emulated(-bexp, _E_LN_TWO1, x)
    yy = _E_THREE33 + t
    kb = _signed_low(yy)
    base = kb.astype(jnp.float64) * 3.814697265625e-06       # == yy - THREE33 exactly (0x774fb), 2^-18
    i = jnp.clip(((kb >> 8) & ~1) + 356, 0, 710)
    j = jnp.clip((kb * 2) & 0x3fe, 0, 1022)
    dl = fma_emulated(-bexp, _E_LN_TWO2, t - base)
    pp = fma_emulated(dl, _E_P3, _E_P2)
    eps = fma_emulated(dl * dl, pp, dl)
    ch, cl = jnp.take(coar, i), jnp.take(coar, i + 1)
    fh, fl = jnp.take(fine, j), jnp.take(fine, j + 1)
    al = ch * fh
    bet = fma_emulated(ch, fl, fh * cl)
    bet = fma_emulated(cl, fl, bet)
    r = fma_emulated(bet, eps, bet)
    r = fma_emulated(eps, al, r)
    res_main = (r + al) * _i2f(((ky + 1023) & 0xfff) << 52)
    # accurate-table path (0x772b0)
    M = (hx & 0xfffff) | 0x100000
    m = M >> jnp.clip(1036 - (n >> 20), 0, 63)
    idx = (m - 1) & ~1
    idx = jnp.clip(jnp.where(hx < 0, idx + 134, idx), 0, 266)
    a0, a1 = jnp.take(acc, idx), jnp.take(acc, idx + 1)
    res_med = fma_emulated(a1, _exp_poly(x - a0), a1)
    # small |x| paths
    res_poly = _exp_poly(x) + _E_ONE
    res_quad = fma_emulated(fma_emulated(x, _E_HALF, _E_ONE), x, _E_ONE)
    res_tiny = _E_ONE + x
    return jnp.where(n <= _E_N_TINY, res_tiny,
                     jnp.where(n <= _E_N_SMALL, res_quad,
                               jnp.where(n <= _E_N_POLY, res_poly,
                                         jnp.where(n <= _E_N_MED, res_med,
                                                   jnp.where(n <= _E_N_MAIN, res_main, jnp.exp(x))))))


@jax.custom_jvp
def glibc_exp(x):
    """exp(x), float64, bit for bit the glibc 2.28 exp of the oracle for |x| <= ~708 (jnp.exp beyond). The derivative is
    glibc_exp(x) * x_dot (AD never enters the bit arithmetic)."""
    return _glibc_exp_impl(x)


@glibc_exp.defjvp
def _glibc_exp_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    y = glibc_exp(x)
    return y, y * xd


def exp_tables():
    """(coar[712], fine[1024]) as glibc stores them (libm 0xfa4a0, 0xf84a0): hi, lo pairs of e^((k-178)/512) and
    e^(m/2^18)."""
    return _E_TABLES[0], _E_TABLES[1]


# ------------------------------------------------------------------------------------------------ glibc 2.28 tanh
#
# glibc_tanh: bit-for-bit the tanh() of the oracle's libm. In /lib64/libm-2.28.so `tanh` (0x382b0) is the fdlibm
# s_tanh.c, with no ifunc variant, and calls `expm1` (0x30c40) directly, which is the fdlibm s_expm1.c with no FMA
# instruction (objdump, 2026-10-01). Transcribed from glibc 2.28 sysdeps/ieee754/dbl-64/s_{tanh,expm1}.c; the
# operation order of expm1's polynomial was checked against the disassembly (0x30d06-0x30da9: hfx = 0.5 x,
# hxs = x hfx, R1 = 1 + hxs Q1, h2 = hxs^2, R2 = Q2 + hxs Q3, h4 = h2^2, R3 = Q4 + hxs Q5, r1 = (R1 + h2 R2) + h4 R3,
# t = 3 - r1 hfx, e = hxs ((r1 - t) / (6 - x t))). XLA's tanh is a different (rational) approximation: measured
# 46-63 % of arguments differ, up to 6 ulp (MEASURED["tanh"]). Needs the gate XLA flags; under them XLA:CPU's float64
# divide is correctly rounded (measured: 0 of 2e5 quotients differ from numpy's), so the divisions are plain `/`.

_X_LN2_HI = 6.93147180369123816490e-01     # 0x3fe62e42 fee00000
_X_LN2_LO = 1.90821492927058770002e-10     # 0x3dea39ef 35793c76
_X_INVLN2 = 1.44269504088896338700e+00     # 0x3ff71547 652b82fe
_X_Q1 = -3.33333333333331316428e-02        # BFA11111 111110F4
_X_Q2 = 1.58730158725481460165e-03         # 3F5A01A0 19FE5585
_X_Q3 = -7.93650757867487942473e-05        # BF14CE19 9EAADBB7
_X_Q4 = 4.00821782732936239552e-06         # 3ED0CFCA 86E65239
_X_Q5 = -2.01099218183624371326e-07        # BE8AFDB7 6E09C32D


def _add_exponent(y, k):
    """SET_HIGH_WORD(y, high + (k << 20)): add k to y's binary exponent (integer add on the bit pattern)."""
    return _i2f(_f2i(y) + (k.astype(jnp.int64) << 52))


def _high_word_double(hw):
    """The double whose high word is hw and low word 0 (fdlibm `t = one; SET_HIGH_WORD(t, hw)`)."""
    return _i2f(hw.astype(jnp.int64) << 32)


def _glibc_expm1_impl(x):
    """expm1(x) of glibc 2.28 for |x| < 709.78 (s_expm1.c; jnp.expm1 beyond, not bitwise)."""
    x = jnp.asarray(x, jnp.float64)
    hx = (_f2i(x) >> 32) & 0xffffffff
    pos = (hx & 0x80000000) == 0
    ax = hx & 0x7fffffff
    # argument reduction
    hi1 = jnp.where(pos, x - _X_LN2_HI, x + _X_LN2_HI)          # 0.5 ln2 < |x| < 1.5 ln2: k = +-1
    lo1 = jnp.where(pos, _X_LN2_LO, -_X_LN2_LO)
    k1 = jnp.where(pos, 1, -1)
    k2 = jnp.trunc(_X_INVLN2 * x + jnp.where(pos, 0.5, -0.5)).astype(jnp.int64)   # (int) conversion truncates
    t2 = k2.astype(jnp.float64)
    hi2 = x - t2 * _X_LN2_HI
    lo2 = t2 * _X_LN2_LO
    mid = ax < 0x3FF0A2B2
    hi, lo, k = jnp.where(mid, hi1, hi2), jnp.where(mid, lo1, lo2), jnp.where(mid, k1, k2)
    xr = hi - lo
    c = (hi - xr) - lo
    red = ax > 0x3fd62e42
    xr, c, k = jnp.where(red, xr, x), jnp.where(red, c, 0.0), jnp.where(red, k, 0)
    # x in the primary range
    hfx = 0.5 * xr
    hxs = xr * hfx
    R1 = 1.0 + hxs * _X_Q1
    h2 = hxs * hxs
    R2 = _X_Q2 + hxs * _X_Q3
    h4 = h2 * h2
    R3 = _X_Q4 + hxs * _X_Q5
    r1 = (R1 + h2 * R2) + h4 * R3
    t = 3.0 - r1 * hfx
    e = hxs * ((r1 - t) / (6.0 - xr * t))
    res_k0 = xr - (xr * e - hxs)
    e = ((xr * (e - c)) - c) - hxs
    res_km1 = 0.5 * (xr - e) - 0.5
    res_k1 = jnp.where(xr < -0.25, -2.0 * (e - (xr + 0.5)), 1.0 + 2.0 * (xr - e))
    res_far = _add_exponent(1.0 - (e - xr), k) - 1.0                          # k <= -2 or k > 56
    kc = jnp.clip(k, 0, 63)
    t_lt20 = _high_word_double(0x3ff00000 - (jnp.int64(0x200000) >> kc))       # 1 - 2^-k
    res_lt20 = _add_exponent(t_lt20 - (e - xr), k)                             # 2 <= k < 20
    t_ge20 = _high_word_double((0x3ff - kc) << 20)                            # 2^-k
    res_ge20 = _add_exponent((xr - (e + t_ge20)) + 1.0, k)                     # 20 <= k <= 56
    res = jnp.where(k == 0, res_k0,
                    jnp.where(k == -1, res_km1,
                              jnp.where(k == 1, res_k1,
                                        jnp.where((k <= -2) | (k > 56), res_far,
                                                  jnp.where(k < 20, res_lt20, res_ge20)))))
    res = jnp.where(ax < 0x3c900000, x, res)                                   # |x| < 2^-54: x
    res = jnp.where((ax >= 0x4043687A) & ~pos, -1.0, res)                      # x < -56 ln2: tiny - one = -1
    return jnp.where(ax >= 0x40862E42, jnp.expm1(x), res)                      # overflow, inf, nan: not bitwise


def _glibc_tanh_impl(x):
    x = jnp.asarray(x, jnp.float64)
    bits = _f2i(x)
    jx = bits >> 32                                                            # signed high word
    ix = jx & 0x7fffffff
    ax = jnp.abs(x)
    big = ix >= 0x3ff00000                                                     # |x| >= 1
    t = _glibc_expm1_impl(jnp.where(big, 2.0 * ax, -2.0 * ax))
    z = jnp.where(big, 1.0 - 2.0 / (t + 2.0), -t / (t + 2.0))
    z = jnp.where(ix < 0x40360000, z, 1.0 - 1.0e-300)                          # |x| >= 22: one - tiny
    z = jnp.where(jx >= 0, z, -z)
    z = jnp.where(ix < 0x3c800000, x * (1.0 + x), z)                           # |x| < 2^-55 (and +-0)
    return jnp.where(ix >= 0x7ff00000, jnp.tanh(x), z)                         # inf, nan


@jax.custom_jvp
def glibc_tanh(x):
    """tanh(x), float64, bit for bit the glibc 2.28 tanh of the oracle (all finite x). Derivative 1 - tanh(x)^2 (AD
    never enters the bit arithmetic)."""
    return _glibc_tanh_impl(x)


@glibc_tanh.defjvp
def _glibc_tanh_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    y = glibc_tanh(x)
    return y, (1.0 - y * y) * xd


# ------------------------------------------------------------------------------------------------ glibc 2.28 asin
#
# glibc_asin: bit-for-bit emulation of the asin() the gfortran oracle calls (ggl90_idemix.F:587, :595), for
# |x| < 2^-3, the range IDEMIX's wet points reach. /lib64/libm-2.28.so: `asin` (0xebf0) -> `__asin_finite` (ifunc at
# 0x23800) -> on AMD EPYC 7763 (FMA + AVX2) the FMA-compiled IBM Accurate Mathematical Library __ieee754_asin at
# 0x79ec0 (sysdeps/ieee754/dbl-64/e_asin.c with its 2.28 slow paths). Transcribed from the disassembly, every
# vfmadd/vfmsub as an (emulated, exact) fused multiply-add. k = high word of |x|:
#     k <  0x3e500000   (|x| < 2^-26)      x                                                     (0x7a180)
#     k <  0x3fc00000   (|x| < 2^-3)       stage 1: Taylor polynomial, accepted if res == res + 1.025*cor (0x79f03);
#                                          stage 2: split x = x1 + x2, accepted with 1.00014 (0x79f65);
#                                          stage 3: __doasin (double-double Taylor series, 0x84270), accepted
#                                          with 1.00000001 (0x7a300); else __sin32 (multi-precision, 0x83850):
#                                          NOT transcribed -> jnp.arcsin FALLBACK (never reached in the sampled
#                                          arguments: 0 of 2.5e6, mitjax/tests/test_glibc_asin.py)
#     otherwise                            jnp.arcsin FALLBACK, not bitwise in general (~24 % differ by 1 ulp);
#                                          equal at 1/3 and 1/1.01, IDEMIX's dry-column arguments (measured)
# Constants: .rodata (VA == file offset) 0xb6648-0xb66c0, 0x113c00/08 (a2, a1), 0xf43a8-0xf4410 (__doasin: c1..c4,
# cc1..cc4, d5..d11), 0xc55e0; `glibc_asin_constants_vs_libm()` compares them with the binary. Measured on the login
# node (same CPU model and libm) against the oracle's libm via ctypes: host transcription 0 mismatches of 2.5e6
# arguments in [2^-26, 2^-3) (stage 1 ~97.6 %, stage 2 ~2.4 %, stage 3 ~1.6e-4); the jax version: test_glibc_asin.py.
_A_TINY_K = 0x3e500000
_A_SMALL_K = 0x3fc00000
_A_ADDR = {"c0": 0xb6648, "c1": 0xb6650, "c2": 0xb6658, "c3": 0xb6660, "c4": 0xb6668, "c5": 0xb6670,
           "r1": 0xb6678, "big": 0xb6680, "e7": 0xb6688, "e6": 0xb6690, "e5": 0xb6698, "e4": 0xb66a0,
           "e3": 0xb66a8, "e2": 0xb66b0, "r2": 0xb66b8, "r3": 0xb66c0, "a2": 0x113c00, "a1": 0x113c08,
           "d11": 0xf43e0, "d10": 0xf43e8, "d9": 0xf43f0, "d8": 0xf43f8, "d7": 0xf4400, "d6": 0xf4408,
           "d5": 0xf4410, "c4d": 0xf43d8, "cc4": 0xf43d0, "c3d": 0xf43c8, "cc3": 0xf43c0, "c2d": 0xf43b8,
           "cc2": 0xf43b0, "c1d": 0xc55e0, "cc1": 0xf43a8}
_A = {k: float.fromhex(v) for k, v in {
    "c0": "0x1.292d80f453c72p-6", "c1": "0x1.6e442c822d419p-6", "c2": "0x1.f1c7e04f4ad99p-6",
    "c3": "0x1.6db6dae42c0e4p-5", "c4": "0x1.333333336127dp-4", "c5": "0x1.55555555554f9p-3",
    "r1": "0x1.0666666666666p+0", "big": "0x1.8p+36",
    "e7": "0x1.e20777c52ca64p-7", "e6": "0x1.1bfe83c1e2c9bp-6", "e5": "0x1.6e8cb40cbd654p-6",
    "e4": "0x1.f1c71a6fd9132p-6", "e3": "0x1.6db6db6ebd269p-5", "e2": "0x1.3333333332f18p-4",
    "r2": "0x1.00092ccf6be38p+0", "r3": "0x1.0000002af31dcp+0", "a2": "-0x1.555555555233p-18", "a1": "0x1.5558p-3",
    "d11": "0x1.04687a52e4bcdp-7", "d10": "0x1.1211420bda560p-7", "d9": "0x1.3fe397e225ac3p-7",
    "d8": "0x1.7a87733820feap-7", "d7": "0x1.c99999cd28e76p-7", "d6": "0x1.1c4ec4ec23223p-6",
    "d5": "0x1.6e8ba2e8ba5f3p-6", "c4d": "0x1.f1c71c71c71c5p-6", "cc4": "-0x1.2b240ff23ed1ep-63",
    "c3d": "0x1.6db6db6db6db7p-5", "cc3": "-0x1.20fc03d5cf0c5p-60", "c2d": "0x1.3333333333333p-4",
    "cc2": "0x1.9999363f1a115p-59", "c1d": "0x1.5555555555555p-3", "cc1": "0x1.5555555775389p-57"}.items()}


def glibc_asin_constants_vs_libm(path="/lib64/libm-2.28.so"):
    """[] if every asin constant equals the 8 bytes at its address in the oracle's libm, else the differences."""
    import struct
    data = open(path, "rb").read()
    return [f"{k}: {_A[k]!r} vs {struct.unpack('<d', data[a:a + 8])[0]!r}" for k, a in _A_ADDR.items()
            if struct.unpack("<d", data[a:a + 8])[0] != _A[k]]


def _add2(x, xx, y, yy):
    """dla.h ADD2 (double-double sum) as compiled: the branch on |x| > |y| (vcomisd) selects the summation order."""
    r = x + y
    s = jnp.where(jnp.abs(x) > jnp.abs(y), (((x - r) + y) + yy) + xx, (((y - r) + x) + xx) + yy)
    z = r + s
    return z, (r - z) + s


def _mul2(x, xx, y, yy):
    """dla.h MUL2 with DLA_FMS: c = x*y, cc = (x*yy + xx*y) + fms(x,y,c) with x*yy fused (vfmadd132sd)."""
    c = x * y
    cc = fma_emulated(x, yy, xx * y) + fma_emulated(x, y, -c)
    z = c + cc
    return z, (c - z) + cc


def _doasin(x):
    """__doasin(x, dx=0, w) (dbl-64/doasincos.c, FMA build at 0x84270): (w0, w1) double-double asin(x)."""
    A = _A
    dx = jnp.zeros_like(x)
    c = x * x
    xx = fma_emulated(x + x, dx, c)                                   # 0x84290  x*x + 2.0*x*dx
    t = fma_emulated(xx, A["d11"], A["d10"])
    for d in ("d9", "d8", "d7", "d6", "d5"):
        t = fma_emulated(xx, t, A[d])
    p = t * xx                                                        # 0x842d9
    cc = fma_emulated(x * dx, 2.0, fma_emulated(x, x, -c))            # 0x84295, 0x842a3 (MUL2(x,dx,x,dx))
    u = c + cc
    uu = (c - u) + cc
    p, pp = _add2(p, 0.0, A["c4d"], A["cc4"])                          # pp = 0: the literal 0.0 additions
    for cn, ccn in (("c3d", "cc3"), ("c2d", "cc2"), ("c1d", "cc1")):
        p, pp = _mul2(p, pp, u, uu)
        p, pp = _add2(p, pp, A[cn], A[ccn])
    p, pp = _mul2(p, pp, u, uu)
    p, pp = _mul2(p, pp, x, dx)
    return _add2(p, pp, x, dx)


def _asin_parts(x):
    """The three stages of the |x| < 2^-3 path: (res, ok1, res2, ok2, w0, ok3)."""
    A = _A
    # stage 1 (0x79f03)
    xx = x * x
    p = fma_emulated(xx, A["c0"], A["c1"])
    for cn in ("c2", "c3", "c4", "c5"):
        p = fma_emulated(xx, p, A[cn])
    x3 = x * xx
    res = fma_emulated(p, x3, x)
    cor = fma_emulated(p, x3, x - res)
    ok1 = fma_emulated(cor, A["r1"], res) == res
    # stage 2 (0x79f65)
    x1 = (x + A["big"]) - A["big"]
    q = fma_emulated(xx, A["e7"], A["e6"])
    for cn in ("e5", "e4", "e3", "e2"):
        q = fma_emulated(xx, q, A[cn])
    x2 = x - x1
    pp = (x1 * x1) * x1
    hx = (x1 * 0.5) * x
    t1 = (q * xx) * xx
    s = fma_emulated((A["a1"] + A["a2"]) * x2, x2, hx)
    res1 = fma_emulated(A["a1"], pp, x)
    s = s * x2
    t1 = fma_emulated(t1, x, s)
    s1c = fma_emulated(A["a1"], pp, x - res1)
    t1 = fma_emulated(A["a2"], pp, t1)
    s2 = s1c + t1
    res2 = res1 + s2
    cor2 = (res1 - res2) + s2
    ok2 = fma_emulated(cor2, A["r2"], res2) == res2
    # stage 3 (0x7a300): __doasin; __sin32 (not transcribed) -> FALLBACK
    w0, w1 = _doasin(x)
    ok3 = fma_emulated(w1, A["r3"], w0) == w0
    return res, ok1, res2, ok2, w0, ok3


def _glibc_asin_impl(x):
    x = jnp.asarray(x, jnp.float64)
    k = (_f2i(x) >> 32) & 0x7fffffff
    res, ok1, res2, ok2, w0, ok3 = _asin_parts(x)
    small = jnp.where(ok1, res, jnp.where(ok2, res2, jnp.where(ok3, w0, jnp.arcsin(x))))
    return jnp.where(k < _A_TINY_K, x, jnp.where(k < _A_SMALL_K, small, jnp.arcsin(x)))


def glibc_asin_stage(x):
    """The path glibc's asin takes for each x (tests): 0 tiny, 1-3 the stages, 4 __sin32 (FALLBACK), 5 |x| >= 2^-3
    (FALLBACK)."""
    x = jnp.asarray(x, jnp.float64)
    k = (_f2i(x) >> 32) & 0x7fffffff
    _, ok1, _, ok2, _, ok3 = _asin_parts(x)
    st = jnp.where(ok1, 1, jnp.where(ok2, 2, jnp.where(ok3, 3, 4)))
    return jnp.where(k < _A_TINY_K, 0, jnp.where(k < _A_SMALL_K, st, 5))


@jax.custom_jvp
def glibc_asin(x):
    """asin(x), float64, bit for bit the glibc 2.28 asin of the oracle for |x| < 2^-3 (jnp.arcsin beyond, and on
    the never-observed __sin32 path; see the block comment). Derivative x_dot / sqrt(1 - x*x) (AD never enters the
    bit arithmetic)."""
    return _glibc_asin_impl(x)


@glibc_asin.defjvp
def _glibc_asin_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    return glibc_asin(x), xd / jnp.sqrt(1.0 - x * x)


# ------------------------------------------------------------------------------------------------ glibc log10
# lane M4OFF session 3 (SEAICE_LSR's LSRflexFac, seaice_lsr.F:754 LOG10): sysdeps/ieee754/dbl-64/e_log10.c of glibc
# 2.28 (x86_64 has no assembly or FMA variant of log10), whose only libm call is __ieee754_log; XLA:CPU's log is
# bitwise glibc's log on 200000 sampled arguments in [1e-6, 1e2] (0 mismatches, measured 2026-10-03 under the gate
# XLA flags) while XLA's log10 is not (16.5 % of the same arguments differ). The transcription below: 0 mismatches
# against /lib64/libm-2.28.so log10 on 1.5e6 arguments in [1e-30, 1e30] (mitjax/tests/test_m4off_lsr.py).
_L10_TWO54 = 1.80143985094819840000e+16      # 0x43500000 00000000   e_log10.c two54
_L10_IVLN10 = 4.34294481903251816668e-01     # 0x3FDBCB7B 1526E50E   ivln10
_L10_LOG10_2HI = 3.01029995663611771306e-01  # 0x3FD34413 509F6000   log10_2hi
_L10_LOG10_2LO = 3.69423907715893078616e-13  # 0x3D59FEF3 11F12B36   log10_2lo


def _glibc_log10_impl(x):
    """__ieee754_log10(x) of glibc 2.28 (e_log10.c) for normal x > 0; zero, negative, subnormal (XLA:CPU flushes
    them), inf and NaN arguments return jnp.log10's value (not bitwise, not reached by the ported call sites)."""
    x = jnp.asarray(x, jnp.float64)
    bits = _f2i(x)
    hx = (bits >> 32).astype(jnp.int32)                         # EXTRACT_WORDS (hx, lx, x)
    k = (hx >> 20) - 1023                                       # k += (hx >> 20) - 1023
    i = ((k.astype(jnp.uint32) & jnp.uint32(0x80000000)) >> 31).astype(jnp.int32)
    hx2 = (hx & 0x000fffff) | ((0x3ff - i) << 20)
    y = (k + i).astype(jnp.float64)                             # y = (double) (k + i)
    xm = _i2f((hx2.astype(jnp.int64) << 32) | (bits & jnp.int64(0xffffffff)))   # SET_HIGH_WORD (x, hx)
    z = y*_L10_LOG10_2LO + _L10_IVLN10*jnp.log(xm)
    r = z + y*_L10_LOG10_2HI
    normal = (hx >= 0x00100000) & (hx < 0x7ff00000)
    return jnp.where(normal, r, jnp.log10(x))


@jax.custom_jvp
def glibc_log10(x):
    """log10(x), float64, bit for bit glibc 2.28's for normal x > 0. Derivative x_dot / (x ln 10)."""
    return _glibc_log10_impl(x)


@glibc_log10.defjvp
def _glibc_log10_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    return glibc_log10(x), xd * (_L10_IVLN10 / x)


# ------------------------------------------------------------------------------------------------ libgcc __powidf2


def powi(x, n):
    """x**n for a static Python int n, exactly libgcc's __powidf2 (libgcc2.c, GCC 11):

        unsigned int n = m < 0 ? -(unsigned int) m : (unsigned int) m;
        DFtype y = n % 2 ? x : 1;
        while (n >>= 1) { x = x * x; if (n % 2) y = y * x; }
        return m < 0 ? 1/y : y;

    (the multiplication by 1 when n is even is exact and omitted). gfortran -O0 calls it for every REAL**INTEGER."""
    n = int(n)
    m = abs(n)
    x = jnp.asarray(x, jnp.float64)
    y = x if m % 2 else None
    m >>= 1
    while m:
        x = x * x
        if m % 2:
            y = x if y is None else y * x
        m >>= 1
    if y is None:
        y = jnp.ones_like(x)
    return 1.0 / y if n < 0 else y


# ------------------------------------------------------------------------------------------------ bundles


class Libm(NamedTuple):
    """The elementary functions the M1 kernels call (the gfortran binary calls glibc or libgcc for each)."""
    exp: Callable
    sin: Callable
    cos: Callable
    tan: Callable
    tanh: Callable
    pow: Callable
    powi: Callable
    sqrt: Callable



# XLA's function (jax 0.10.1, gate XLA flags, AMD EPYC 7763) vs the oracle's libm: elements that differ, summed over
# test_libm.samples() (20 000 per range; 2026-10-01, login node, re-measured by test_libm.py on every tier-1 run).
# 0 = bitwise on every sampled range.
MEASURED = {
    "sin": 0, "cos": 0, "tan": 0, "log": 0, "atan": 0, "atan2": 0, "pow": 0, "fmod": 0, "sqrt": 0,
    "exp": 5810,      # 13.5 % of [-50, 0] and [-708, 708], 1.7 % of |x| in [1e-12, 1]; 1 ulp (glibc_exp: 0)
    "tanh": 31987,    # 46-63 % per range, up to 6 ulp (glibc_tanh: 0)
    "asin": 4764,     # 23.8 % of [-1, 1], 1 ulp; not on an M1 path (ggl90_idemix, ggl90 off)
}

# The function each M1 call site uses: jnp's where it is glibc's bit for bit, the transcription where not.
M1_LIBM = Libm(exp=glibc_exp, sin=jnp.sin, cos=jnp.cos, tan=jnp.tan, tanh=glibc_tanh, pow=jnp.power, powi=powi,
               sqrt=jnp.sqrt)
