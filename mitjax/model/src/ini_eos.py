"""INI_EOS: model/src/ini_eos.F @63cdc0b, and EOS.h (model/inc/EOS.h) as INI_EOS leaves it.

`EOS` holds the EOS.h common-block variables by their Fortran names: `equationOfState` (CHARACTER*(6), static), the
traced scalars eosRefP0, tAlpha, sBeta, and the coefficient vectors as FArrays with their declared bounds
(eosJMDCFw(6) -> `k=(1, 6)`, eosMDJWFnum(0:11) -> `k=(0, 11)`, untiled; the axis is named k only because FArray
names its axes i, j, k: `eos.eosJMDCFw[2]` is eosJMDCFw(2)). A pytree (equationOfState is static aux data).
"""

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp

from mitjax.farray import FArray

UNSET_RL = 1.234567e5       # EEPARAMS.h:81  PARAMETER ( UNSET_RL = 1.234567D5 )

# EOS.h declarations: name -> (lo, hi)
_VECTORS = {"eosJMDCFw": (1, 6), "eosJMDCSw": (1, 9), "eosJMDCKFw": (1, 5), "eosJMDCKSw": (1, 7),
            "eosJMDCKP": (1, 14), "eosMDJWFnum": (0, 11), "eosMDJWFden": (0, 12), "teos": (1, 48)}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EOS:
    """model/inc/EOS.h @63cdc0b (the variables INI_EOS sets and the ported FIND_RHO routines read)."""
    equationOfState: str        # EOS.h:26  CHARACTER*(6) (static)
    eosRefP0: object            # EOS.h:25
    tAlpha: object              # EOS.h:32
    sBeta: object               # EOS.h:33
    eosJMDCFw: FArray           # EOS.h:58
    eosJMDCSw: FArray
    eosJMDCKFw: FArray          # EOS.h:59
    eosJMDCKSw: FArray
    eosJMDCKP: FArray
    eosMDJWFnum: FArray         # EOS.h:62
    eosMDJWFden: FArray
    teos: FArray                # EOS.h:67

    def tree_flatten(self):
        names = [f.name for f in fields(self) if f.name != "equationOfState"]
        return tuple(getattr(self, n) for n in names), (self.equationOfState, tuple(names))

    @classmethod
    def tree_unflatten(cls, aux, children):
        eqs, names = aux
        return cls(equationOfState=eqs, **dict(zip(names, children)))

    def replace(self, **kw):
        return replace(self, **kw)


def _vector(name, values=None):
    lo, hi = _VECTORS[name]
    data = jnp.zeros(hi - lo + 1, jnp.float64) if values is None else jnp.asarray(values, jnp.float64)
    return FArray(data, name, k=(lo, hi), tiled=False)


def _set(v, n, value):
    return v.at[n].set(value)


def ini_eos(*, cfg, params):
    """INI_EOS( myThid )   @63cdc0b model/src/ini_eos.F:9-377

    C     *==========================================================*
    C     | SUBROUTINE INI_EOS
    C     | o Initialise coefficients of equation of state.
    C     *==========================================================*

    Host-side (runs once, eagerly). `params`: the PARAMS.h / EOS.h values INI_PARMS leaves: fluidIsWater,
    usingPCoords, eosType (static), tAlpha and sBeta (the namelist value or UNSET_RL, ini_parms.F:421-422). Returns
    the EOS common block, or None when the fluid is not water (:45, the RETURN before anything is set).

    Ported: 'LINEAR' (:84-86), 'JMD95Z'/'JMD95P' (:105-175) and 'MDJWF' (:233-260); 'POLY3' (reads
    POLY3.COEFFS), 'UNESCO', 'TEOS10' and 'IDEALG' raise. EOS_CHECK (:379-) is empty: INCLUDE_EOS_CHECK is #undef'd
    at :2.
    """
    if not params.fluidIsWater:                                         # :45
        return None
    equationOfState = params.eosType                                    # :50
    eqs = equationOfState.rstrip()
    # :54-77 zero all coefficients
    v = {name: _vector(name) for name in _VECTORS}
    for k in range(1, 7):
        v["eosJMDCFw"] = _set(v["eosJMDCFw"], k, 0.0)
    for k in range(1, 10):
        v["eosJMDCSw"] = _set(v["eosJMDCSw"], k, 0.0)
    for k in range(1, 6):
        v["eosJMDCKFw"] = _set(v["eosJMDCKFw"], k, 0.0)
    for k in range(1, 8):
        v["eosJMDCKSw"] = _set(v["eosJMDCKSw"], k, 0.0)
    for k in range(1, 15):
        v["eosJMDCKP"] = _set(v["eosJMDCKP"], k, 0.0)
    for k in range(0, 12):
        v["eosMDJWFnum"] = _set(v["eosMDJWFnum"], k, 0.0)
    for k in range(0, 13):
        v["eosMDJWFden"] = _set(v["eosMDJWFden"], k, 0.0)
    for k in range(1, 49):
        v["teos"] = _set(v["teos"], k, 0.0)
    eosRefP0 = jnp.float64(101325.0)                                    # :82  101325. _d 0
    tAlpha, sBeta = params.tAlpha, params.sBeta

    if eqs == "LINEAR":                                                 # :84-86
        if tAlpha == UNSET_RL:
            tAlpha = 2.0e-4                                             # :85  2.  _d -4
        if sBeta == UNSET_RL:
            sBeta = 7.4e-4                                              # :86  7.4 _d -4
    elif eqs == "POLY3":                                                # :87-103
        raise NotImplementedError("INI_EOS: eosType = 'POLY3' (POLY3.COEFFS) is not ported")
    elif eqs in ("JMD95Z", "JMD95P", "UNESCO"):                         # :105-107
        if eqs == "JMD95Z" and params.usingPCoords:                     # :114-125
            raise ValueError("ini_eos: equation of state 'JMD95Z' should not\n"
                             "         be used together with pressure coordinates.\n"
                             "         Use only 'JMD95P' with 'OCEANICP'.\nABNORMAL END: S/R INI_EOS")
        # :129-134  1. density of fresh water at p = 0
        Fw = v["eosJMDCFw"]
        Fw = _set(Fw, 1, 999.842594e+00)
        Fw = _set(Fw, 2, 6.793952e-02)
        Fw = _set(Fw, 3, -9.095290e-03)
        Fw = _set(Fw, 4, 1.001685e-04)
        Fw = _set(Fw, 5, -1.120083e-06)
        Fw = _set(Fw, 6, 6.536332e-09)
        # :136-144  2. density of sea water at p = 0
        Sw = v["eosJMDCSw"]
        Sw = _set(Sw, 1, 8.24493e-01)
        Sw = _set(Sw, 2, -4.0899e-03)
        Sw = _set(Sw, 3, 7.6438e-05)
        Sw = _set(Sw, 4, -8.2467e-07)
        Sw = _set(Sw, 5, 5.3875e-09)
        Sw = _set(Sw, 6, -5.72466e-03)
        Sw = _set(Sw, 7, 1.0227e-04)
        Sw = _set(Sw, 8, -1.6546e-06)
        Sw = _set(Sw, 9, 4.8314e-04)
        v["eosJMDCFw"], v["eosJMDCSw"] = Fw, Sw
        if equationOfState[:5] == "JMD95":                              # :145
            # :147-151  3. secant bulk modulus K of fresh water at p = 0
            KFw = v["eosJMDCKFw"]
            KFw = _set(KFw, 1, 1.965933e+04)
            KFw = _set(KFw, 2, 1.444304e+02)
            KFw = _set(KFw, 3, -1.706103e+00)
            KFw = _set(KFw, 4, 9.648704e-03)
            KFw = _set(KFw, 5, -4.190253e-05)
            # :153-159  4. secant bulk modulus K of sea water at p = 0
            KSw = v["eosJMDCKSw"]
            KSw = _set(KSw, 1, 5.284855e+01)
            KSw = _set(KSw, 2, -3.101089e-01)
            KSw = _set(KSw, 3, 6.283263e-03)
            KSw = _set(KSw, 4, -5.084188e-05)
            KSw = _set(KSw, 5, 3.886640e-01)
            KSw = _set(KSw, 6, 9.085835e-03)
            KSw = _set(KSw, 7, -4.619924e-04)
            # :161-174  5. secant bulk modulus K of sea water at p
            KP = v["eosJMDCKP"]
            KP = _set(KP, 1, 3.186519e+00)
            KP = _set(KP, 2, 2.212276e-02)
            KP = _set(KP, 3, -2.984642e-04)
            KP = _set(KP, 4, 1.956415e-06)
            KP = _set(KP, 5, 6.704388e-03)
            KP = _set(KP, 6, -1.847318e-04)
            KP = _set(KP, 7, 2.059331e-07)
            KP = _set(KP, 8, 1.480266e-04)
            KP = _set(KP, 9, 2.102898e-04)
            KP = _set(KP, 10, -1.202016e-05)
            KP = _set(KP, 11, 1.394680e-07)
            KP = _set(KP, 12, -2.040237e-06)
            KP = _set(KP, 13, 6.128773e-08)
            KP = _set(KP, 14, 6.207323e-10)
            v["eosJMDCKFw"], v["eosJMDCKSw"], v["eosJMDCKP"] = KFw, KSw, KP
        else:                                                           # :176-  'UNESCO'
            raise NotImplementedError("INI_EOS: eosType = 'UNESCO' is not ported")
    elif eqs == "MDJWF":                                                # :233-260 (lane COLMIX, M3 vermix)
        num = v["eosMDJWFnum"]
        num = _set(num, 0, 9.99843699e+02)                              # :235  9.99843699 _d +02
        num = _set(num, 1, 7.35212840e+00)                              # :236
        num = _set(num, 2, -5.45928211e-02)                             # :237
        num = _set(num, 3, 3.98476704e-04)                              # :238
        num = _set(num, 4, 2.96938239e+00)                              # :239
        num = _set(num, 5, -7.23268813e-03)                             # :240
        num = _set(num, 6, 2.12382341e-03)                              # :241
        num = _set(num, 7, 1.04004591e-02)                              # :242
        num = _set(num, 8, 1.03970529e-07)                              # :243
        num = _set(num, 9, 5.18761880e-06)                              # :244
        num = _set(num, 10, -3.24041825e-08)                            # :245
        num = _set(num, 11, -1.23869360e-11)                            # :246
        den = v["eosMDJWFden"]
        den = _set(den, 0, 1.00000000e+00)                              # :248  1.00000000 _d +00
        den = _set(den, 1, 7.28606739e-03)                              # :249
        den = _set(den, 2, -4.60835542e-05)                             # :250
        den = _set(den, 3, 3.68390573e-07)                              # :251
        den = _set(den, 4, 1.80809186e-10)                              # :252
        den = _set(den, 5, 2.14691708e-03)                              # :253
        den = _set(den, 6, -9.27062484e-06)                             # :254
        den = _set(den, 7, -1.78343643e-10)                             # :255
        den = _set(den, 8, 4.76534122e-06)                              # :256
        den = _set(den, 9, 1.63410736e-09)                              # :257
        den = _set(den, 10, 5.30848875e-06)                             # :258
        den = _set(den, 11, -3.03175128e-16)                            # :259
        den = _set(den, 12, -1.27934137e-17)                            # :260
        v["eosMDJWFnum"], v["eosMDJWFden"] = num, den
    elif eqs in ("TEOS10", "IDEALG"):                                   # :262-358
        raise NotImplementedError(f"INI_EOS: eosType = '{eqs}' is not ported")
    else:                                                               # :360-366
        raise ValueError(f'INI_EOS: eosType= "{eqs}" not valid\nABNORMAL END: S/R INI_EOS')

    # :371  CALL EOS_CHECK: empty (INCLUDE_EOS_CHECK #undef'd at :2)
    return EOS(equationOfState=equationOfState, eosRefP0=eosRefP0, tAlpha=jnp.float64(tAlpha),
               sBeta=jnp.float64(sBeta), **v)
