"""KPP_INIT_FIXED: pkg/kpp/kpp_init_fixed.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import UNSET_RL
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.libm import powi
from mitjax.pkg.kpp.kpp_params_h import NNI, NNJ


def kpp_init_fixed(kpp, *, cfg, grid):
    """KPP_INIT_FIXED( myThid )   @63cdc0b pkg/kpp/kpp_init_fixed.F:6-193

    C     | SUBROUTINE KPP_INIT_FIXED
    C     | o Routine to initialize KPP parameters and variables.
    C     | Initialize KPP parameters and variables that do not
    C     | change during the model run.

    `kpp`: the Kpp of KPP_READPARMS (None when useKPP is .FALSE.: returned as is). Returns the Kpp with Vtc, cg
    (:125-126), deltaz, deltau (:132-133), the velocity-scale tables wmt, wst(0:nni+1,0:nnj+1) (:135-156), minKPPhbl
    (:162-164), zgrid, hwide(0:Nr+1) (:165-181). Computed once on the host, eagerly (jax on concrete values, one
    operation per XLA call, as the grid builders), in the Fortran's order and association: `p25 = 0.25 _d 0`,
    `p33 = 1. _d 0 / 3. _d 0`; `x**p25`, `x**p33` are libm `pow` (XLA's pow is glibc's bit for bit, libm.MEASURED);
    `usta**3`, `vonk**2` are libgcc's __powidf2 (libm.powi); an INTEGER in a REAL expression (`deltaz*i`,
    `(nni + 1)`) is converted to double first, as gfortran does. The table loops (:135-156) are independent per
    (i, j) and run as array statements over the whole table, both branches of each IF computed and selected
    (forward only: the tables are constants of the model). The MNC variable definitions (:44-112) are output setup
    and KPP_DIAGNOSTICS_INIT (:187-189) is output only: not ported. Every value is gated against the replay
    harness's kpp_fixed.bin (reference/replay_kpp)."""
    if kpp is None:
        return None
    sz = cfg.size
    Nr = sz.Nr
    p25 = 0.25                                                                 # :118  0.25 _d 0
    p33 = np.float64(1.0) / np.float64(3.0)                                    # :119  1. _d 0 / 3. _d 0
    f = lambda x: jnp.asarray(x, jnp.float64)                                  # noqa: E731
    concv, concs, epsilon, vonk, Ricr = (f(kpp.concv), f(kpp.concs), f(kpp.epsilon), f(kpp.vonk), f(kpp.Ricr))
    # :125  Vtc = concv * SQRT(0.2 _d 0 /concs/epsilon) / vonk**2 / Ricr
    Vtc = concv * jnp.sqrt(0.2 / concs / epsilon) / powi(vonk, 2) / Ricr
    # :126  cg = cstar * vonk * (concs * vonk * epsilon)**p33
    cg = f(kpp.cstar) * vonk * jnp.power(concs * vonk * epsilon, p33)
    zmin, zmax, umin, umax = f(kpp.zmin), f(kpp.zmax), f(kpp.umin), f(kpp.umax)
    deltaz = (zmax - zmin) / float(NNI + 1)                                    # :132
    deltau = (umax - umin) / float(NNJ + 1)                                    # :133
    # :135-156  DO i = 0, nni + 1 / DO j = 0, nnj + 1 (independent points); storage [j, i] of wmt(i, j)
    ii = jnp.arange(0, NNI + 2, dtype=jnp.float64)[None, :]                    # DO i = 0, nni + 1
    jj = jnp.arange(0, NNJ + 2, dtype=jnp.float64)[:, None]                    # DO j = 0, nnj + 1
    zehat = deltaz*ii + zmin                                                   # :136
    usta = deltau*jj + umin                                                    # :138
    zeta = zehat / MAX(f(kpp.phepsi), powi(usta, 3), p="b")                    # :139
    conc1, conc2, conc3 = f(kpp.conc1), f(kpp.conc2), f(kpp.conc3)
    wm_pos = vonk*usta/(1. + conc1*zeta)                                       # :141  1. (REAL*4, exact)
    wm_a = vonk*usta*jnp.power(1. - conc2*zeta, p25)                           # :145
    wm_b = vonk*jnp.power(f(kpp.conam)*powi(usta, 3) - f(kpp.concm)*zehat, p33)   # :147
    ws_a = vonk*usta*jnp.sqrt(1.0 - conc3*zeta)                                # :150  1. _d 0
    ws_b = vonk*jnp.power(f(kpp.conas)*powi(usta, 3) - concs*zehat, p33)       # :152
    pos = zehat >= 0.                                                          # :140
    wmt = jnp.where(pos, wm_pos, jnp.where(zeta > f(kpp.zetam), wm_a, wm_b))   # :140-148
    wst = jnp.where(pos, wm_pos, jnp.where(zeta > f(kpp.zetas), ws_a, ws_b))   # :142, :149-153
    minKPPhbl = kpp.minKPPhbl
    if minKPPhbl == UNSET_RL:                                                  # :162-164
        minKPPhbl = -np.float64(grid.rC.data[0])                               # -rC(1)
    rC = np.asarray(grid.rC.data, np.float64)
    drF = np.asarray(grid.drF.data, np.float64)
    zg = np.empty(Nr + 2)
    hw = np.empty(Nr + 2)
    zg[0] = kpp.phepsi                                                         # :165  zgrid(0)  =  phepsi
    hw[0] = kpp.phepsi                                                         # :166  hwide(0)  =  phepsi
    zg[1:Nr+1] = rC                                                            # :174-177 zgrid(k) = rC(k)
    hw[1:Nr+1] = drF                                                           #          hwide(k) = drF(k)
    zg[Nr+1] = zg[Nr] * 100.                                                   # :179  zgrid(Nrp1) = zgrid(Nr) * 100.
    hw[Nr+1] = kpp.phepsi                                                      # :181  hwide(Nrp1) = phepsi
    return kpp.replace(
        Vtc=np.float64(Vtc), cg=np.float64(cg), deltaz=np.float64(deltaz), deltau=np.float64(deltau),
        minKPPhbl=np.float64(minKPPhbl),
        wmt=FArray(jnp.asarray(wmt), "wmt", i=(0, NNI+1), j=(0, NNJ+1), tiled=False),
        wst=FArray(jnp.asarray(wst), "wst", i=(0, NNI+1), j=(0, NNJ+1), tiled=False),
        zgrid=FArray(jnp.asarray(zg), "zgrid", k=(0, Nr+1), tiled=False),
        hwide=FArray(jnp.asarray(hw), "hwide", k=(0, Nr+1), tiled=False))
