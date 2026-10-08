"""CTRL_MAP_INI_GENARR: pkg/ctrl/ctrl_map_ini_genarr.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad)."""

from mitjax.pkg.ctrl.ctrl_map_genarr import ctrl_map_genarr2d, ctrl_map_genarr3d
from mitjax.pkg.ctrl.ctrl_readparms import fstr_blank, fstr_prefix


def ctrl_map_ini_genarr(fields, *, cfg, genarr2d, genarr3d, xx_in, weight_in, maskC, ex, ptr=None):
    """CTRL_MAP_INI_GENARR( myThid )   @63cdc0b pkg/ctrl/ctrl_map_ini_genarr.F:24-438

    C     | Add the generic arrays to the
    C     | corresponding model variables

    `fields`: {Fortran name: FArray} of the model variables a control of this build can reach: theta, salt (DYNVARS.h),
    GM_inpK3dGM, GM_inpK3dRedi (GMREDI.h), bottomDragFld (CTRL_FIELDS.h; CTRL_INIT_VARIABLES sets it to 0. _d 0 at
    every point first, ctrl_init_variables.F:83-91). `genarr2d` / `genarr3d`: ctrl_readparms_genarr(run, 2 / 3);
    `xx_in` / `weight_in`: {(dim, iarr): FArray} the control and weight records (see ctrl_map_genarr3d).
    Returns (fields with the controls added, {(dim, iarr): effective record}).

    Branches (global_ocean.90x40x15/code_ad, CTRL_OPTIONS.h / GMREDI_OPTIONS.h): ALLOW_GENARR2D_CONTROL,
    ALLOW_GENARR3D_CONTROL, ALLOW_BOTTOMDRAG_CONTROL, ALLOW_KAPGM_CONTROL with GM_READ_K3D_GM, ALLOW_KAPREDI_CONTROL with
    GM_READ_K3D_REDI defined; ALLOW_SEAICE, ALLOW_SHELFICE, ALLOW_STREAMICE, ALLOW_DIC, ALLOW_PTRACERS,
    ALLOW_GEOTHERMAL_FLUX, ALLOW_UVEL0/VVEL0_CONTROL, ALLOW_DIFFKR_CONTROL with ALLOW_3D_DIFFKR undefined (their
    blocks are not compiled: a control named for them is matched (igen_*) but never applied, as in the Fortran).
    Not ported (raise): xx_etan (:200-202, CTRL_MAP_GENARR2D of etaN with its own exchanges).
    Lane M4ADLAB (lab_sea/code_ad, ALLOW_SEAICE): xx_siarea / xx_siheff map onto SEAICE.h AREA / HEFF (:162-166,
    :217-222; fields["AREA"], fields["HEFF"]); a SHELFICE / STREAMICE / DIC control name raises.
    PTRACERS lane (tutorial_tracer_adjsens/code_ad, xx_ptr1): under ALLOW_PTRACERS the xx_ptr<n> controls (:347-351,
    :378-387, :412-421) map onto fields["pTracer_<nn>"]; `ptr`: the PTRACERS_PARAMS.h values (usePTRACERS,
    PTRACERS_num, PTRACERS_numInUse); None when usePTRACERS is .FALSE. (no igen_ptr is set, :379)."""
    out = dict(fields)
    eff = {}
    cpp = cfg.cpp
    # :130-205  2-D
    igen_etan = igen_bdrag = igen_geoth = 0                         # :133-135
    igen_siarea = igen_siheff = 0                                   # :136-139 (ALLOW_SEAICE; lane M4ADLAB)
    seaice = bool(cpp.flag("ALLOW_SEAICE"))
    for g in genarr2d:                                              # :168-196 (SHELFICE / STREAMICE / DIC names)
        if not fstr_blank(g.weight) and any(fstr_prefix(g.file, len(n), n) for n in (
                "xx_shicoeff", "xx_shicdrag", "xx_bglen", "xx_rlow_streamice", "xx_beta", "xx_bdot",
                "xx_h_streamice", "xx_alpha")):
            raise NotImplementedError(f"CTRL_MAP_INI_GENARR: {g.file.rstrip()} is not ported")
    for g in genarr2d:                                              # :153
        if not fstr_blank(g.weight):                                # :154
            if fstr_prefix(g.file, 7, "xx_etan"):                   # :156-157
                igen_etan = g.iarr
            if fstr_prefix(g.file, 13, "xx_bottomdrag"):            # :158-159
                igen_bdrag = g.iarr
            if fstr_prefix(g.file, 13, "xx_geothermal"):            # :160-161
                igen_geoth = g.iarr
            if seaice and fstr_prefix(g.file, 9, "xx_siarea"):      # :162-164
                igen_siarea = g.iarr
            if seaice and fstr_prefix(g.file, 9, "xx_siheff"):      # :165-166
                igen_siheff = g.iarr
    if igen_etan > 0:                                               # :200-202
        raise NotImplementedError("CTRL_MAP_INI_GENARR: xx_etan is not ported")
    if cpp.flag("ALLOW_BOTTOMDRAG_CONTROL", "CTRL_OPTIONS.h") and igen_bdrag > 0:   # :203-206
        g = genarr2d[igen_bdrag - 1]
        out["bottomDragFld"], eff[(2, igen_bdrag)], _ = ctrl_map_genarr2d(
            out["bottomDragFld"], igen_bdrag, cfg=cfg, genarr2d=g, xx_in=xx_in[(2, igen_bdrag)],
            weight_in=weight_in[(2, igen_bdrag)], maskC=maskC, ex=ex)
    if cpp.flag("ALLOW_GEOTHERMAL_FLUX") and igen_geoth > 0:        # :207-215
        raise NotImplementedError("CTRL_MAP_INI_GENARR: xx_geothermal is not ported")
    for name, igen in (("AREA", igen_siarea), ("HEFF", igen_siheff)):   # :217-222 (lane M4ADLAB: lab_sea/code_ad)
        if seaice and igen > 0:                                     # CTRL_MAP_GENARR2D( AREA / HEFF, igen, myThid )
            if name not in fields:
                raise ValueError(f"CTRL_MAP_INI_GENARR: xx_si{name.lower()} needs SEAICE.h {name} in `fields`")
            out[name], eff[(2, igen)], _ = ctrl_map_genarr2d(
                out[name], igen, cfg=cfg, genarr2d=genarr2d[igen - 1], xx_in=xx_in[(2, igen)],
                weight_in=weight_in[(2, igen)], maskC=maskC, ex=ex)

    # :336-430  3-D
    igen_theta0 = igen_salt0 = igen_kapgm = igen_kapredi = igen_diffkr = 0   # :339-343
    for g in genarr3d:                                              # :353
        if not fstr_blank(g.weight):                                # :354
            if fstr_prefix(g.file, 8, "xx_theta"):                  # :355-356
                igen_theta0 = g.iarr
            if fstr_prefix(g.file, 7, "xx_salt"):                   # :357-358
                igen_salt0 = g.iarr
            if fstr_prefix(g.file, 8, "xx_kapgm"):                  # :359-360
                igen_kapgm = g.iarr
            if fstr_prefix(g.file, 10, "xx_kapredi"):               # :361-362
                igen_kapredi = g.iarr
            if fstr_prefix(g.file, 9, "xx_diffkr"):                 # :363-364
                igen_diffkr = g.iarr
    targets = [("theta", igen_theta0, True), ("salt", igen_salt0, True),             # :392-395
               ("GM_inpK3dGM", igen_kapgm,                                          # :396-399
                cpp.flag("ALLOW_KAPGM_CONTROL", "CTRL_OPTIONS.h") and cpp.flag("GM_READ_K3D_GM", "GMREDI_OPTIONS.h")),
               ("GM_inpK3dRedi", igen_kapredi,                                      # :400-403
                cpp.flag("ALLOW_KAPREDI_CONTROL", "CTRL_OPTIONS.h")
                and cpp.flag("GM_READ_K3D_REDI", "GMREDI_OPTIONS.h"))]
    # :404-407 GO lane (global_ocean.cs32x15/code_ad): CTRL_MAP_GENARR3D( diffKr, igen_diffkr ) (DYNVARS.h diffKr)
    targets.append(("diffKr", igen_diffkr,
                    cpp.flag("ALLOW_DIFFKR_CONTROL", "CTRL_OPTIONS.h") and cpp.flag("ALLOW_3D_DIFFKR")))
    if igen_diffkr > 0 and targets[-1][2] and "diffKr" not in fields:
        raise ValueError("CTRL_MAP_INI_GENARR: xx_diffkr needs the State's diffKr in `fields`")
    for name, igen, compiled in targets:
        if compiled and igen > 0:
            g = genarr3d[igen - 1]
            out[name], eff[(3, igen)], _ = ctrl_map_genarr3d(
                out[name], igen, cfg=cfg, genarr3d=g, xx_in=xx_in[(3, igen)], weight_in=weight_in[(3, igen)],
                maskC=maskC, ex=ex)
    if cpp.ALLOW_PTRACERS and ptr is not None:                      # PTRACERS lane: :347-351, :378-387, :412-421
        igen_ptr = [0]*ptr.PTRACERS_num                             # :348-350
        for g in genarr3d:                                          # :353
            if not fstr_blank(g.weight) and ptr.usePTRACERS:        # :354, :379
                iLen = len(g.file.rstrip())                         # :380  ILNBLNK
                if iLen == 7 and g.file[:6] == "xx_ptr":            # :381-382
                    iPtr = int(g.file[6:7])                         # :383  READ(file(7:7),*) iPtr
                    if 1 <= iPtr <= ptr.PTRACERS_numInUse:          # :384-385
                        igen_ptr[iPtr - 1] = g.iarr
        for iPtr in range(1, ptr.PTRACERS_num + 1):                 # :415-421
            igen = igen_ptr[iPtr - 1]
            if igen > 0:
                name = f"pTracer_{iPtr:02d}"
                out[name], eff[(3, igen)], _ = ctrl_map_genarr3d(
                    out[name], igen, cfg=cfg, genarr3d=genarr3d[igen - 1], xx_in=xx_in[(3, igen)],
                    weight_in=weight_in[(3, igen)], maskC=maskC, ex=ex)
    return out, eff
