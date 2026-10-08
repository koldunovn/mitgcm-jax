"""Build identity (docs plan 20261006 Task 4, R6 of docs/PORTABILITY.md): which compiled MITgcm a configuration is.

The identity of a build is a hash of what its preprocessing fixes: the MITgcm commit, the compiled packages (genmake2
order), the macros in effect after every *OPTIONS.h and the model/src prologue (cfg.cpp.headers, without the
compiler's own predefined macros; the DEFINES of the toolchain are part of them) and the SIZE.h parameters. It does
not depend on the experiment's name or location: a renamed or moved copy of a verification experiment is the same
build, and an edited code directory (another CPP option, another SIZE.h) is a new build.

`KNOWN` names the identities of the verification builds of the registry (reference/reference_runs.py) by their
oracle build label `<experiment>-<code dir>-<upstream commit[:7]>` (the oracle binary's name without its
reference-tool commit, e.g. lab_sea-code_ad-63cdc0b for lab_sea-code_ad-63cdc0b-704fd6b); mitjax/tests/
test_build_identity.py recomputes every entry from the configuration. Used for the build-dependent MAX/MIN winners
of plan decision 15 (mitjax/ops/fortran_minmax.build_winner): a build that is not in KNOWN was never compiled by the
oracle, so its winners were never measured (decision 7: the documented default and one warning per site).
"""

import functools
import hashlib
import json

SIZE_NAMES = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "nPx", "nPy", "Nx", "Ny", "Nr", "MAX_OLX", "MAX_OLY")


def identity_text(cfg):
    """The canonical text the identity hashes (JSON, sorted keys)."""
    return json.dumps({"upstream_commit": cfg.upstream_commit,
                       "packages": list(cfg.packages),
                       "cpp": [[h, [list(kv) for kv in items]] for h, items in cfg.cpp.headers],
                       "size": {k: int(getattr(cfg.size, k)) for k in SIZE_NAMES}},
                      sort_keys=True, separators=(",", ":"))


@functools.lru_cache(maxsize=256)
def _identity_cached(cfg):
    return hashlib.sha256(identity_text(cfg).encode()).hexdigest()


def identity(cfg):
    """sha256 (hex) of identity_text(cfg)."""
    try:
        return _identity_cached(cfg)
    except TypeError:                       # an unhashable stand-in configuration
        return hashlib.sha256(identity_text(cfg).encode()).hexdigest()


# identity -> oracle build label, for every (experiment, code dir) of the registry's variants (M1-M4 and code_min);
# computed with the default toolchain (mitjax/config/cpp_options.py) at 63cdc0b, checked by test_build_identity.py
KNOWN = {
    "bcf8faba602bc48e57b06df3d7b586a5b9cfc66a8f7b87efa4a925ad5709ba77": "1D_ocean_ice_column-code-63cdc0b",
    "2b265d50747f3641d111d7b3331dee8a14309568e436aad4bcf0da5eb89f765e": "1D_ocean_ice_column-code_ad-63cdc0b",
    "80d5a978da6b558afd940da5ebe55ada0481c11914c1d432105c711b249cc454": "MLAdjust-code-63cdc0b",
    "b21da5fc4a778d52dea4e11b14a1f806a67e658a9886d4d3830e95957498bee4": "adjustment.cs-32x32x1-code-63cdc0b",
    "2e4871f080f3b0a61014cf5ef39484230e38343735b709a93c989abcb9c2fc9b": "adjustment.cs-32x32x1-code_min-63cdc0b",
    "b0fe990f9029c295f904f9c29672fdea80c0db8146623743179126c5b55d4009": "advect_cs-code-63cdc0b",
    "db79a798c45383be26d40ff29c03214a4d4c89c7c8ba0684538f56616f542fcf": "advect_xy-code-63cdc0b",
    "fc70fd67ab44d2a5ade0b567396f797b61f9d6bd383c7a4268697f1b75afc5b0": "advect_xz-code-63cdc0b",
    "c1e078afbe2fc9638f5a4c7268d1857033a576ec7a2c8500d4d44aa3264fe4ba": "front_relax-code-63cdc0b",
    "601ee8e5fc87c7f21469894ab4a604279d2d6ceca0c0ce97e28ba1a1abc728cd": "global_ocean.90x40x15-code-63cdc0b",
    "6fdb287f7483e03a5c8e68448364568098a76a8ee12225e73f7c3c0f65913464": "global_ocean.90x40x15-code_ad-63cdc0b",
    "b0378a16cfb26407519d0e9a19406ce429b0c293c9e7ef85a7ea9135ff9f3661": "global_ocean.cs32x15-code-63cdc0b",
    "bfc1114fb25db96ffd5482599611accc473792e1d00e01f6b39042ba77e7cb48": "global_ocean.cs32x15-code_ad-63cdc0b",
    "399432fbc27709f55ce913641c035c733615e32264469ff94e2d462a616f5714": "ideal_2D_oce-code-63cdc0b",
    "29db1bdf89edb0398f7b4ff25fe16f933dad5a7a9699925898a4556ba2955941": "lab_sea-code-63cdc0b",
    "c578a64db498032ce32f6d0452f36ef9962d1d2f9105cd47c9923ea4732136c3": "lab_sea-code_ad-63cdc0b",
    "f58ef923a18a80ba020256dc4cae272f58e77394ac0fb43a4bc290745235329e": "offline_exf_seaice-code-63cdc0b",
    "684d958c2d7ac98a3970179caf674a23d65f4867d6359bbe88304da73c3e6971": "offline_exf_seaice-code_ad-63cdc0b",
    "7a3a0494033d4bee805b9bcf5be20ff8f2ef882c2db48774c874b6086cfb559d": "seaice_itd-code-63cdc0b",
    "b1d81f3862f3d255af3c22e7388086094468b64e3dbdf45c11059ae68dafd51a": "solid-body.cs-32x32x1-code-63cdc0b",
    "79de90358cc32e5ecf4b4f9b221f141b573c90d90de0eacc8b6fbc2cd56f43e4": "tutorial_advection_in_gyre-code-63cdc0b",
    "03da3f9f6f012d6c1ee4de263491289b013ae27c1d602d3c4214436cf970dc1f": "tutorial_baroclinic_gyre-code-63cdc0b",
    "5a14c2408611c80742fd8531fa01374338bacbf961b8108f237e4ab4045951ce": "tutorial_barotropic_gyre-code-63cdc0b",
    "667ba28b142fbcf683961e2cbf21208101045d646dc02088b42842e015619a19": "tutorial_global_oce_latlon-code-63cdc0b",
    "9680d1319828c92939fbce62f63c96d4c00051d6e53351f6aa1fee848a4fb28f": "tutorial_global_oce_optim-code_ad-63cdc0b",
    "81c15cc88dc27ea1365817954ae90f36bde49d36cecdfc230a071e6701a915de": "tutorial_reentrant_channel-code-63cdc0b",
    "1fc04984da640f98310444c0bc5bcc961751f46518972eeda8c8719bec708d03": "tutorial_tracer_adjsens-code_ad-63cdc0b",
    "c72e8c65d4c59c1eb0bc5d7ed6de5c00ff678c9f735e4d6b1d1bff119a7c6266": "vermix-code-63cdc0b",
}


def label(cfg):
    """The oracle build label of cfg's build (KNOWN), or None for a build the oracle never compiled."""
    return KNOWN.get(identity(cfg))
