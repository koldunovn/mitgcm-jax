"""INI_PARMS / SET_DEFAULTS / SET_PARMS (model/src/ini_parms.F, set_defaults.F, set_parms.F @63cdc0b), r* group: the
PARAMS.h values the r* / nonlinear free-surface routines read that `ini_parms_dyn` (mitjax/model/src/ini_parms.py, M1
core lane) does not provide yet: CALC_R_STAR (hFacInf, hFacSup, selectKEscheme), FORWARD_STEP (doResetHFactors).

Same construction as mitjax/model/src/ini_parms.py: the namelist value from the run's parameter file (mitjax/config),
else the cited SET_DEFAULTS line (`fortran_default`, checked live in this build), and the SET_PARMS statements that
change a value ported as code with their lines. Kept in the r* lane's own file because ini_parms.py belongs to the
core lane; to be folded into `ini_parms_dyn` when the lanes merge.

Static (aux data of `Params`): the INTEGER / LOGICAL switches. Traced (leaves): the REAL values, float64.
"""

import numpy as np

from mitjax.model.src.ini_parms import _get
from mitjax.params_io import RunParams

SD = "model/src/set_defaults.F"


def ini_parms_rstar(exp, params):
    """`params` (a `Params` of ini_parms_dyn) with the r* values added (module docstring), each cited.

    * PARM01 hFacInf (set_defaults.F:258 `hFacInf = 0.2 _d 0`), hFacSup (:259 `hFacSup = 2.0 _d 0`): traced REALs
      (CALC_R_STAR compares rStarFac with them, calc_r_star.F:185-196);
    * PARM01 selectKEscheme (:239 `selectKEscheme = 0`): static (CALC_R_STAR's rStarAreaWeight, calc_r_star.F:66-68);
    * PARM01 doResetHFactors (:185 `doResetHFactors = .FALSE.`), then set_parms.F:177-182: `.TRUE.` under
      ALLOW_AUTODIFF, `.FALSE.` without NONLIN_FRSURF (FORWARD_STEP's RESET_NLFS_VARS / UPDATE_R_STAR(.FALSE.) block,
      forward_step.F:465-494).
    """
    cfg = exp.cfg
    rp = RunParams(exp.run)
    g = lambda grp, key, line: _get(exp, rp, grp, key, f"{SD}:{line}")       # noqa: E731
    hFacInf = float(g("PARM01", "hFacInf", 258))
    hFacSup = float(g("PARM01", "hFacSup", 259))
    selectKEscheme = int(g("PARM01", "selectKEscheme", 239))
    doResetHFactors = bool(g("PARM01", "doResetHFactors", 185))
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # set_parms.F:177-179
        doResetHFactors = True
    if not cfg.cpp.NONLIN_FRSURF:                                               # set_parms.F:180-182
        doResetHFactors = False
    return params.replace(static=dict(selectKEscheme=selectKEscheme, doResetHFactors=doResetHFactors),
                          traced=dict(hFacInf=np.float64(hFacInf), hFacSup=np.float64(hFacSup)))
