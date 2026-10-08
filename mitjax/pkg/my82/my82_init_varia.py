"""MY82_INIT_VARIA: pkg/my82/my82_init_varia.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.pkg.my82.my82_h import A1, A2, B1, B2, C1, declare


def my82_init_varia(*, cfg, params, my):
    """MY82_INIT_VARIA( myThid )   @63cdc0b pkg/my82/my82_init_varia.F:3-62

    C     | SUBROUTINE MY82_INIT_VARIA                               |
    C     | o Routine to initialize MY82 parameters and variables.   |
    C     | Initialize MY92 parameters and variables.                |

    Returns `my` (MY82.h) with the magic parameters alpha1, alpha2, beta1..beta4 (:32-39, M. Satoh p. 314; host
    float64, one IEEE operation per Fortran operation in the Fortran association) and MYhbl = 0, MYviscAr(k) =
    viscArNr(k), MYdiffKr(k) = diffKrNrS(k) at every point, halos included (:45-57; levels independent).
    """
    one, two, three, six = np.float64(1.0), np.float64(2.0), np.float64(3.0), np.float64(6.0)
    a1, a2, b1, b2, c1 = (np.float64(x) for x in (A1, A2, B1, B2, C1))
    gam1 = one/three - two*a1/b1                                        # :32
    gam2 = (b2+six*a1)/b1                                               # :33
    alpha1 = three*a2*gam1                                              # :34
    alpha2 = three*a2*(gam1+gam2)                                       # :35
    beta1 = a1*b1*(gam1-c1)                                             # :36
    beta2 = a1*(b1*(gam1-c1) + six*a1 + three*a2)                       # :37
    beta3 = a2*b1*gam1                                                  # :38
    beta4 = a2*(b1*(gam1+gam2) - three*a1)                              # :39
    sz = cfg.size
    T = sz.nSx*sz.nSy
    shape = (T, sz.Nr, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)
    hbl = jnp.zeros((T, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx))                                    # :50
    vis = jnp.broadcast_to(jnp.asarray(params.viscArNr.data)[None, :, None, None], shape)     # :51
    dif = jnp.broadcast_to(jnp.asarray(params.diffKrNrS.data)[None, :, None, None], shape)    # :52
    return my.replace(alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4,
                      MYhbl=declare("MYhbl", sz, hbl), MYviscAr=declare("MYviscAr", sz, vis),
                      MYdiffKr=declare("MYdiffKr", sz, dif))
