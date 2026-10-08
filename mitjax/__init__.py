"""MITgcm master (commit 63cdc0b) in JAX: literal, differentiable, parallel, readable for a Fortran MITgcm developer.

float64 is mandatory: x64 is enabled on import, before any array is created.

The API (mitjax/api.py) is exported lazily (PEP 562): `import mitjax` does not import the drivers.
    mitjax.load(exp_dir, variant=), mitjax.compare(run, results)
    mitjax.MinMaxDefaultWarning   the warning of a build-dependent MAX/MIN site that took its documented default
                                  (plan decision 7; mitjax/ops/fortran_minmax.py), e.g. for warnings.simplefilter
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"

_API = ("load", "compare")
_OPS = ("MinMaxDefaultWarning",)


def __getattr__(name):
    if name in _API:
        from mitjax import api
        return getattr(api, name)
    if name in _OPS:
        from mitjax.ops import fortran_minmax
        return getattr(fortran_minmax, name)
    raise AttributeError(f"module 'mitjax' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_API) + list(_OPS))
