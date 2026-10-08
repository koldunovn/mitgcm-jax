"""SWFRAC: model/src/swfrac.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.ops.libm import glibc_exp

# :74-76  DATA rfac / 0.58 _d 0, 0.62 _d 0, ... /, a1 / 0.35 _d 0, 0.6 _d 0, ... /, a2 / 23.0 _d 0, 20.0 _d 0, ... /
rfac = (0.58, 0.62, 0.67, 0.77, 0.78)
a1 = (0.35, 0.6, 1.0, 1.5, 1.4)
a2 = (23.0, 20.0, 17.0, 14.0, 7.9)


def swfrac(imax, fact, swdk, myTime, myIter):
    """SWFRAC( imax, fact, swdk, myTime, myIter, myThid )   @63cdc0b model/src/swfrac.F:7-108

    C     | o Compute solar short-wave flux penetration.
    C     | Compute fraction of solar short-wave flux penetrating to
    C     | specified depth, swdk, due to exponential decay in
    C     | Jerlov water type jwtype.
    C     | Reference : Two band solar absorption model of Paulson
    C     |             and Simpson (1977, JPO, 7, 952-956)
    C     | Parameter jwtype is hardcoded to 2 for time being.
    C     | Below 200m the solar penetration gets set to zero,
    C     | otherwise the limit for the exponent (+/- 5678) needs to
    C     | be taken care of.
    C     imax    :: number of vertical grid points
    C     fact    :: scale  factor to apply to depth array
    C     swdk    :: on input: vertical depth for desired sw fraction
    C               (fact*swdk) is negative distance (m) from surface
    C     swdk    :: on output: short wave (radiation) fractional decay

    swdk: a jnp vector of length imax. Returns swdk. jwtype = 2 (:92 / :94, both arms of ALLOW_CAL); the DATA values
    are double literals (`_d 0`). exp is glibc's (mitjax/ops/libm.glibc_exp; XLA's exp is not bitwise). The points
    are independent (vectorised); the IF (:99-104) is a `where`: both branches finite (facz/a1 >= -200/0.6 there)."""
    jwtype = 2                                                                  # :94
    swdk = jnp.asarray(swdk)
    assert swdk.shape == (imax,), (swdk.shape, imax)
    facz = fact*swdk                                                            # :98
    deep = facz < -200.                                                         # :99
    fz = jnp.where(deep, 0., facz)                                              # finite exponent on the THEN lanes
    return jnp.where(deep, 0.,                                                  # :100
                     rfac[jwtype-1]*glibc_exp(fz/a1[jwtype-1])                  # :102-103
                     + (1.-rfac[jwtype-1])*glibc_exp(fz/a2[jwtype-1]))
