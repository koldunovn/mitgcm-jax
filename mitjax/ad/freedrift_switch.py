"""SEAICEuseFREEDRIFTswitchInAd as a backward-only switch (plan decision 17 revised, decision 11 pattern; lane
M4ADCS32ICE session 2; global_ocean.cs32x15/input_ad.seaice_dynmix, offline_exf_seaice/input_ad.obcs).

What TAF does (63cdc0b). ADAUTODIFF_INADMODE_SET (autodiff_inadmode_set_ad.F:69-72), run first in the reverse sweep
of every step: `IF ( SEAICEuseFREEDRIFTswitchInAd ) SEAICEuseFREEDRIFT = .NOT.SEAICEuseFREEDRIFTinFwdMode;
SEAICEuseLSR = .NOT.SEAICEuseFREEDRIFT`. With the forward's SEAICEuseFREEDRIFT = .FALSE. the reverse sweep therefore
differentiates SEAICE_DYNSOLVER's AD-mode block (seaice_dynsolver.F:303-346):
    CALL SEAICE_FREEDRIFT                       (:304-307: called in both modes when LSR_mixIniGuess = 0)
    uIce = uIce_fd, vIce = vIce_fd, stressDivergenceX = stressDivergenceY = 0.   (:308-321, every point 1-OLx:sNx+OLx)
    no SEAICE_LSR                               (:343-346, SEAICEuseLSR = .FALSE.)
TAF's TLM never sets inAdMode (G_AUTODIFF_INADMODE_SET): the TLM differentiates the LSR.

Hook (`lsr_with_freedrift_in_ad`): the call "SEAICE_LSR" of seaice_dynsolver.py keeps its forward value; its reverse
derivative is that of the AD-mode block above with respect to the same inputs: the uIce / vIce cotangents go to
uice_fd / vice_fd (SEAICE_FREEDRIFT's outputs, differentiated by the forward code that computed them), the
stressDivergenceX/Y cotangents are dropped, every other SEAICE.h field passes through (the AD-mode block does not
write it), and nothing reaches SEAICE_LSR's own inputs. The LSR's monitor outputs (residuals) get no derivative.
Static option `SeaiceParams.mjx_freedrift_in_ad` (not a Fortran parameter; absent = off), set by
drivers/ad_switches.with_run_switches from the run's data.autodiff. The forward program is the same function with
the option on or off (fc bit for bit); forward-mode differentiation of the hooked call is not defined (custom_vjp).
"""

import jax
import jax.numpy as jnp

from mitjax.ad.approx_advection import backward_as

OPTION = "mjx_freedrift_in_ad"


def freedrift_in_ad(sp):
    v = getattr(sp, OPTION, False)
    if v not in (False, True):
        raise ValueError(f"sp.{OPTION} = {v!r}: True or False")
    return v


def _with(a, data):
    return type(a)(data, a.name, tiled=a.tiled, _dims=a.dims)


def lsr_with_freedrift_in_ad(lsr, myTime, myIter, sf, *, cfg, sp, op, grid, state, ex):
    """(sf, lsr_out) = lsr(myTime, myIter, sf, ...) whose reverse derivative is that of the AD-mode free-drift block
    (module docstring). `sf` holds uice_fd / vice_fd (SEAICE_FREEDRIFT ran before, seaice_dynsolver.F:304-307)."""
    def f_fwd(t, it, s, p, o, g, st):
        return lsr(t, it, s, cfg=cfg, sp=p, op=o, grid=g, state=st, ex=ex)

    def f_ad(t, it, s, p, o, g, st):
        out_shape = jax.eval_shape(lambda *a: f_fwd(*a)[1], t, it, s, p, o, g, st)
        new = dict(s)
        new["UICE"] = _with(s["UICE"], s["uice_fd"].data)                     # :312 uIce = uIce_fd
        new["VICE"] = _with(s["VICE"], s["vice_fd"].data)                     # :313 vIce = vIce_fd
        for n in ("stressDivergenceX", "stressDivergenceY"):                  # :314-315  0. _d 0
            if n in s:
                new[n] = _with(s[n], jnp.zeros_like(s[n].data))
        full_shape = jax.eval_shape(lambda *a: f_fwd(*a)[0], t, it, s, p, o, g, st)
        if set(full_shape) != set(new):
            raise ValueError(f"free-drift switch: SEAICE_LSR returns fields {sorted(set(full_shape) - set(new))} "
                             "that are not in its input (the AD-mode block must return the same fields)")
        return new, jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), out_shape)
    return backward_as(f_fwd, f_ad, myTime, myIter, sf, sp, op, grid, state)
