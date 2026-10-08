"""The adjoint exchanges ADEXCH_3D_RL / ADEXCH_UV_3D_RL and the copies COPY_ADVAR_OUTP / COPY_AD_UV_OUTP that the
adjoint monitor and the adjoint dumps use (pkg/autodiff/copy_advar_outp.F, copy_ad_uv_outp.F @63cdc0b, with
mon_AdVarExch = 2).

ADEXCH is the adjoint (transpose) of the forward exchange: every halo value is added to the interior point it was
copied from, and the halos are zeroed. Here it is the transpose of the experiment's own exchange (mitjax/eesupp:
the probe-measured maps), taken with jax.vjp: the input cotangent of `out = EXCH(in)` -- the interior points keep
their value plus the halo copies of them, the input halos (overwritten by the exchange, never read) get zero. This
is AD machinery (a JAX transform), so it lives in mitjax/ad, not in mitjax/pkg.
"""

import jax
import jax.numpy as jnp

from mitjax.farray import FArray


def _fa(like, data):
    return FArray(data, like.name, tiled=like.tiled, _dims=like.dims)


def adexch_3d_rl(fld, *, ex):
    """ADEXCH_3D_RL( fld, nNz ): the transpose of EXCH_3D_RL (2-D fields: of EXCH_XY_RL) on the FArray `fld`."""
    exch = ex.EXCH_3D_RL if len(fld.dims) == 3 else ex.EXCH_XY_RL
    _, vjp = jax.vjp(exch, jnp.zeros_like(fld.data))
    return _fa(fld, vjp(fld.data)[0])


def adexch_uv_3d_rl(u, v, withSigns, *, ex):
    """ADEXCH_UV_3D_RL( u, v, withSigns, nNz ): the transpose of EXCH_UV_3D_RL on the pair."""
    _, vjp = jax.vjp(lambda a, b: ex.EXCH_UV_3D_RL(a, b, withSigns), jnp.zeros_like(u.data), jnp.zeros_like(v.data))
    cu, cv = vjp((u.data, v.data))
    return _fa(u, cu), _fa(v, cv)


def copy_advar_outp(fld, vType, *, ex):
    """COPY_ADVAR_OUTP( inpFldRS, inpFldRL, outFld, nNz, vType ) for the vTypes the monitor uses (11, 12: centred,
    no sign; copy_advar_outp.F:61-110): a copy, then ADEXCH_3D_RL (:120-125). Other vTypes STOP in the Fortran."""
    gridloc, kind = vType // 10, vType % 10
    if kind < 1 or kind > 4 or gridloc < 1 or gridloc > 2:
        raise RuntimeError("ABNORMAL END: COPY_ADVAR_OUTP invalid vType")
    if gridloc != 1 or kind >= 3:
        raise RuntimeError(f"ABNORMAL END: COPY_ADVAR_OUTP vType={vType} (wSign / loc=2) not coded")
    return adexch_3d_rl(fld, ex=ex)


def copy_ad_uv_outp(u, v, vType, *, ex):
    """COPY_AD_UV_OUTP( ..., uFldOut, vFldOut, nNz, vType ) for vType 33/34 (C-grid pair; copy_ad_uv_outp.F): a
    copy, then ADEXCH_UV_3D_RL( uFldOut, vFldOut, wSign ) with wSign = MOD(vType,10) >= 3 (:69-74). The other
    grid locations STOP in the Fortran."""
    gridloc, kind = vType // 10, vType % 10
    if kind < 1 or kind > 4 or gridloc < 1 or gridloc > 4:
        raise RuntimeError("ABNORMAL END: COPY_AD_UV_OUTP invalid vType")
    if gridloc != 3:
        raise RuntimeError(f"ABNORMAL END: COPY_AD_UV_OUTP missing vType={vType}")
    return adexch_uv_3d_rl(u, v, kind >= 3, ex=ex)
