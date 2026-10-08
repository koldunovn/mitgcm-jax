#!/usr/bin/env python3
"""Insert jaxdump calls into copies of MITgcm master sources (plan Task 4; copied from the ECCO port's
reference/jaxdump/instrument.py and adapted to master 63cdc0b and the M1 experiments).

    instrument.py --root ROOT [--mods DIR] OUTDIR    write instrumented copies + jaxdump.F + JAXDUMP.h to OUTDIR
                                                    (ROOT = a MITgcm tree, e.g. the build snapshot; a file in the
                                                    experiment's code dir DIR wins over ROOT, as in genmake2)
    instrument.py --git UPSTREAM [--rev pinned] --check   anchor counts and CPP contexts on a git revision (no output)
    instrument.py --git UPSTREAM --markdown          print the SUBSTEPS table (reference/jaxdump/SUBSTEPS.md)

Each STAGES entry names a source file, an anchor (regex matched against non-comment lines), which occurrence to use,
how many occurrences the file must contain (a drifted source fails loudly), and what to dump right after the anchor
statement (after its continuation lines). Every instrumented file lives in model/src, which every build compiles, so
the copies never pull an unused package into a build; package fields are dumped by jaxdump.F under #ifdef ALLOW_<PKG>.
Sources are never modified in place.

Insertion points carry the CPP context of the anchor (the stack of enclosing #if lines). It is printed in the
SUBSTEPS table: a stage inside `#ifdef NONLIN_FRSURF` exists only in builds that define it, and a stage inside a
Fortran IF block only when that branch runs. A missing stage is information, never silently dropped (the tests list
the stages compiled into each build from its preprocessed sources).

Iteration of a record. forward_step.F advances the counter right after DYNAMICS (`myIter = nIter0 + iLoop`,
forward_step.F:807 at 63cdc0b; :823 at c66g). The ECCO port passed myIter-1 statically to every later stage. Here
THERMODYNAMICS runs either before DYNAMICS (staggerTimeStep=.FALSE., forward_step.F:733) or after the update
(staggerTimeStep=.TRUE., :1005), so one static choice cannot be right for both; instead the shim keeps a runtime
shift: JAXDUMP_ITERSHIFT(0) at the start of the step (before stage S00) and JAXDUMP_ITERSHIFT(1) right after the
update line, and every record carries myIter - shift = the step's START iteration. The assert of the ECCO port is
kept: the update line must exist exactly once, after CALL DYNAMICS and before CALL UPDATE_R_STAR(.TRUE.), and both
shift calls must sit outside any CPP conditional.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (file, anchor, occurrence, expected total, stage, dump statements, scope, what the substep does, options)
# dump statements: 'S:<groups>' -> JAXDUMP_STATE(stage, groups, bi0, bj0) (groups: jaxdump.F, JAXDUMP_STATE);
# 'T:<name>:<kind>:<nz>' -> JAXDUMP_TILE of a routine-local array of the current tile (halos included);
# 'G:<name>:<kind>:<nz>' -> JAXDUMP_LOCAL of a routine-local all-tile array;
# 'K:<name>:<kind>[:<expr>]' -> JAXDUMP_TILEK of the 2-D level k (loop variable k) of the current tile inside a k
#   loop, field <name>_k<kkk>; <expr> (default <name>) is the array element the 2-D slice starts at;
# 'U:<name>:<kind>:<nz>' -> JAXDUMP_TILEI of a routine-local INTERIOR-only array (1:sNx,1:sNy,nz), halos written 0;
# 'N:<name>[:<expr>]' -> JAXDUMP_SCALAR of a scalar (integer or real) as a constant field on every tile (kind N).
# scope 'all' dumps every tile (bi0=0), 'tile' only the current bi,bj (inside a tile loop).
# Anchors: 'BEFORE:<re>' inserts before the statement; 'ENDDO<n>:<re>' after the ENDDO that closes the n-th DO loop
# enclosing the matched line (the matched line counts as level 1 when it is a DO statement); 'CPPEND:<re>' after the
# #endif closing the innermost CPP block around the matched line; a plain '<re>' after the statement.
# options: 'pkg' the stage list it belongs to (PACKAGES below); 'loop' an integer Fortran expression appended to the
# stage name as '_p<n>'; 'cond' a Fortran logical expression guarding the dumps; 'iter' the Fortran expression passed
# as the iteration (default myIter; routines without a myIter argument, e.g. INITIALISE_VARIA and INI_FIELDS, pass
# nIter0 from PARAMS.h, which is the myIter of READ_PICKUP and of the initial MONITOR call, initialise_varia.F:383).
FS, OP, DY, TH, SP = "forward_step.F", "do_oceanic_phys.F", "dynamics.F", "thermodynamics.F", "solve_for_pressure.F"
TI, SI, ML, IO = "temp_integrate.F", "salt_integrate.F", "the_main_loop.F", "do_the_model_io.F"
IV, IF_, RP = "initialise_varia.F", "ini_fields.F", "read_pickup.F"
TC = "tracers_correction_step.F"
FILES = {f: "model/src/" + f for f in (FS, OP, DY, TH, SP, TI, SI, ML, IO, IV, IF_, RP, TC)}
# M4 (lane A session 10): package files. A copy in the -mods directory is compiled whether or not its package is
# enabled (tools/genmake2:2945-2961), so a package file is instrumented only for a build whose packages.conf requests
# that package (FILE_PKG; reference/check_build.py requested_packages, the set check_build verifies against
# ENABLED_PACKAGES); `instrument.py --root --mods` prints each skipped file.
IM, IY, XG = "seaice_model.F", "seaice_dynsolver.F", "exf_getforcing.F"
FILES.update({IM: "pkg/seaice/" + IM, IY: "pkg/seaice/" + IY, XG: "pkg/exf/" + XG})
FILE_PKG = {IM: "seaice", IY: "seaice", XG: "exf"}
PACKAGES = {
    "core": "model/src time step: forcing, density, momentum, free surface, continuity, tracers (M1 all)",
    "gad": "generic_advdiff inside TEMP_/SALT_INTEGRATE: advection, explicit tendency, implicit vertical",
    "gmredi": "GM/Redi tensor, its exchange, bolus (residual) velocity (R5; R4 compiles gmredi, useGMRedi=F)",
    "ggl90": "GGL90 TKE mixing (R4)",
    "monitor": "state at the MONITOR call (all M1 runs with monitorFreq > 0)",
    "sbo": "SBO_CALC scalars (R4)",
    "ctrl": "CTRL_MAP_FORCING: time-varying controls added to the forcing (R5, xx_qnet)",
    "cost": "COST_TILE per step and COST_FINAL after the loop (R5)",
    "exch": "exchange probe: index-coded and signed-zero fields through every exchange routine (all layouts)",
    "init": "INITIALISE_VARIA: READ_PICKUP before and after its exchanges, state after INI_FIELDS (iteration nIter0)",
    "ptracers": "PTRACERS_INTEGRATE inside THERMODYNAMICS (M2: tutorial_global_oce_latlon, "
                "tutorial_advection_in_gyre, tutorial_tracer_adjsens)",
    "kpp": "KPP_CALC and KPP_DO_EXCH (M3: vermix input, input.dd with KPPuseDoubleDiff)",
    "pp81": "PP81_CALC (M3: vermix input.pp81)",
    "my82": "MY82_CALC (M3: vermix input.my82)",
    "opps": "OPPS_INTERFACE in TRACERS_CORRECTION_STEP (M3: vermix input.opps)",
    "exf": "EXF_GETFORCING (pkg/exf/exf_getforcing.F, instrumented only in builds requesting exf): read and "
           "interpolate, radiation, wind, bulk formulae, surface fluxes, map to the model forcing (M4)",
    "seaice": "SEAICE_MODEL and SEAICE_DYNSOLVER (pkg/seaice, instrumented only in builds requesting seaice) and the "
              "state after SEAICE_MODEL in DO_OCEANIC_PHYS (M4)",
    "thsice": "THSICE_MAIN in DO_OCEANIC_PHYS and THSICE_DO_ADVECT in SEAICE_MODEL (M4)",
    "salt_plume": "SALT_PLUME_DO_EXCH in DO_OCEANIC_PHYS (M4: lab_sea)",
}
STAGES = [
    (RP, r"BEFORE:CALL EXCH_UV_3D_RL\(\s*uVel,\s*vVel,", 1, 1, "I00_pickup_read", ["S:dta"], "all",
     "pickup fields as read (interior from the file; halos still those of INI_DYNVARS), before READ_PICKUP's "
     "exchanges (read_pickup.F:538-567)", {"pkg": "init"}),
    (IF_, r"CALL READ_PICKUP\(", 1, 1, "I01_read_pickup", ["S:dta"], "all",
     "state after READ_PICKUP (its exchanges included)", {"pkg": "init", "iter": "nIter0"}),
    (IV, r"CALL INI_FIELDS\(", 1, 1, "I02_ini_fields", ["S:dta"], "all",
     "state after INI_FIELDS (cold start: INI_VEL ... INI_PRESSURE; restart: READ_PICKUP)",
     {"pkg": "init", "iter": "nIter0"}),
    (FS, r"CPPEND:CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "S00_begin", ["S:dtarfmkgcP"], "all",
     "state at the start of the step (after the AD-only iteration reset and AUTODIFF_INADMODE_UNSET, "
     "forward_step.F:427-435, which plain builds do not compile)", {"pkg": "core", "shift": 0}),
    (FS, r"CPPEND:CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "G00_geometry", ["S:GVR"], "all",
     "grid, masks, 3-D mixing parameters, packed vertical grid, r* fields", {"pkg": "core"}),
    (FS, r"CPPEND:CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "X00_exch_probe", ["S:X"], "all",
     "exchange probe (jaxdump.F JAXDUMP_EXCH_PROBE): every exchange routine on index-coded and +0/-0 fields",
     {"pkg": "exch"}),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.FALSE\.", 1, 1, "S01_update_rstar_F", ["S:rd"], "all",
     "RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) (select_rStar > 0)", {"pkg": "core"}),
    (XG, r"CALL EXF_GETFFIELDS\(", 1, 1, "X01_exf_getffields", ["S:x"], "all",
     "EXF_GETFFIELDS: forcing records read and time-interpolated (A-grid stress not yet exchanged)",
     {"pkg": "exf"}),
    (XG, r"CALL EXF_RADIATION\(", 1, 1, "X02_exf_radiation", ["S:x"], "all",
     "EXF_RADIATION: lwflux (emissivity, surface temperature), swflux", {"pkg": "exf"}),
    (XG, r"CALL EXF_WIND\(", 1, 1, "X03_exf_wind", ["S:x"], "all",
     "EXF_WIND: wind speed and stress (wStress, cw, sw, sh; uwind/vwind from the stress when useAtmWind=F)",
     {"pkg": "exf"}),
    (XG, r"CALL EXF_BULKFORMULAE\(", 1, 1, "X04_exf_bulkformulae", ["S:x"], "all",
     "EXF_BULKFORMULAE: hs, hl, evap, the stress over open ocean", {"pkg": "exf"}),
    (XG, r"BEFORE:CALL EXF_GETSURFACEFLUXES\(", 1, 1, "X05_exf_hflux_sflux", ["S:x"], "all",
     "hflux, sflux (net heat and fresh-water fluxes, runoff, masks) and the stress exchange", {"pkg": "exf"}),
    (XG, r"CALL EXF_MAPFIELDS\(", 1, 1, "X06_exf_mapfields", ["S:xf"], "all",
     "EXF_MAPFIELDS: fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad (end of EXF_GETFORCING)", {"pkg": "exf"}),
    (FS, r"CALL LOAD_FIELDS_DRIVER\(", 1, 1, "S02_load_fields", ["S:f"], "all",
     "external forcing fields read and time-interpolated", {"pkg": "core"}),
    (FS, r"CALL CTRL_MAP_FORCING\(", 1, 1, "S03_ctrl_map_forcing", ["S:f"], "all",
     "time-varying controls added to the forcing (useCTRL)", {"pkg": "ctrl"}),
    (OP, r"CALL THSICE_MAIN\(", 1, 1, "P12_thsice_main", ["S:hf"], "all",
     "THSICE_MAIN: thermodynamic sea ice (pkg/thsice) and the ocean forcing it modifies", {"pkg": "thsice"}),
    (IM, r"BEFORE:CALL SEAICE_DYNSOLVER\s*\(", 1, 1, "I00_seaice_begin", ["S:Iix"], "all",
     "SEAICE_MODEL inputs: masks and metric terms, ice state, EXF fields (uwind/vwind after EXCH_UV_AGRID_3D_RL)",
     {"pkg": "seaice"}),
    (IM, r"BEFORE:CALL DYNSOLVER\s*\(", 1, 1, "I00b_seaice_begin", ["S:IixB"], "all",
     "SEAICE_MODEL inputs, B-grid build (SEAICE_BGRID_DYNAMICS): masks, ice state, B-grid fields, EXF fields",
     {"pkg": "seaice"}),
    (IM, r"CALL DYNSOLVER\s*\(", 1, 1, "I01b_dynsolver", ["S:iyBf"], "all",
     "DYNSOLVER (B grid, pkg/seaice/dynsolver.F): forcing, PRESS0/zMax/zMin, LSR if SEAICEuseDYNAMICS, then OSTRES "
     "(fu, fv under ice) in every case", {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_GET_DYNFORCING\s*\(", 1, 1, "Y01_get_dynforcing", ["G:TAUX:W:1", "G:TAUY:S:1"], "all",
     "SEAICE_GET_DYNFORCING: wind stress on the ice TAUX, TAUY (routine-local all-tile arrays)", {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_CALC_ICE_STRENGTH\(", 1, 1, "Y02_ice_strength", ["S:y"], "tile",
     "ice mass, FORCEX0/Y0 (stress + tilt) and SEAICE_CALC_ICE_STRENGTH: PRESS0, SEAICE_zMax, SEAICE_zMin",
     {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_FREEDRIFT\(", 1, 1, "Y03_freedrift", ["S:y"], "all",
     "SEAICE_FREEDRIFT: uice_fd, vice_fd (SEAICEuseFREEDRIFT, EVP or LSR_mixIniGuess)", {"pkg": "seaice"}),
    (IY, r"CPPEND:CALL OBCS_APPLY_UVICE\(", 1, 1, "Y04_solver_inputs", ["S:iy"], "all",
     "all inputs of the momentum solver (EVP, LSR, Krylov or JFNK)", {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_EVP\(", 1, 1, "Y05_evp", ["S:iy"], "all", "SEAICE_EVP result (EVP, mEVP, aEVP)",
     {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_LSR\(", 1, 1, "Y06_lsr", ["S:iy"], "all", "SEAICE_LSR result", {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_KRYLOV\(", 1, 1, "Y07_krylov", ["S:iy"], "all", "SEAICE_KRYLOV result",
     {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_JFNK\(", 1, 1, "Y08_jfnk", ["S:iy"], "all", "SEAICE_JFNK result", {"pkg": "seaice"}),
    (IY, r"CALL SEAICE_OCEAN_STRESS\s*\(", 1, 1, "Y09_ocean_stress", ["S:f"], "all",
     "SEAICE_OCEAN_STRESS: fu, fv under ice (SEAICEupdateOceanStress)", {"pkg": "seaice"}),
    (IM, r"CALL SEAICE_DYNSOLVER\s*\(", 1, 1, "I01_dynsolver", ["S:iyf"], "all",
     "SEAICE_DYNSOLVER incl. velocity clipping (SEAICE_clipVelocities)", {"pkg": "seaice"}),
    (IM, r"CALL THSICE_DO_ADVECT\(", 1, 1, "I02a_thsice_advect", ["S:h"], "all",
     "THSICE_DO_ADVECT: pkg/thsice fields advected by the sea-ice velocity (useThSice)", {"pkg": "thsice"}),
    (IM, r"CALL SEAICE_ADVDIFF\(", 2, 2, "I02_advdiff", ["S:i"], "all",
     "SEAICE_ADVDIFF (C grid): HEFF, AREA, HSNOW (and ITD categories) advected and diffused", {"pkg": "seaice"}),
    (IM, r"CALL SEAICE_REG_RIDGE\(", 1, 1, "I03_reg_ridge", ["S:in"], "all",
     "SEAICE_REG_RIDGE: negative-value and area regularisation (d_HEFFbyNEG, d_HSNWbyNEG)", {"pkg": "seaice"}),
    (IM, r"CALL SEAICE_GROWTH_ADX\(", 1, 1, "I04a_growth_adx", ["S:infp"], "all",
     "SEAICE_GROWTH_ADX (SEAICE_USE_GROWTH_ADX): thermodynamics and the ocean forcing", {"pkg": "seaice"}),
    (IM, r"CALL SEAICE_GROWTH\(", 1, 1, "I04_growth", ["S:infp"], "all",
     "SEAICE_GROWTH: thermodynamics, ocean forcing Qnet/Qsw/EmPmR/saltFlux, sIceLoad, salt-plume flux",
     {"pkg": "seaice"}),
    (OP, r"CALL SEAICE_MODEL\(", 1, 1, "P13_seaice_model", ["S:iyfh"], "all",
     "all of SEAICE_MODEL (after its HEFF/AREA/HSNOW and forcing exchanges)", {"pkg": "seaice"}),
    (OP, r"CALL SALT_PLUME_DO_EXCH\(", 1, 1, "P14_salt_plume_exch", ["S:p"], "all",
     "salt-plume depth and flux after SALT_PLUME_DO_EXCH (useSALT_PLUME)", {"pkg": "salt_plume"}),
    (OP, r"CALL EXTERNAL_FORCING_SURF\(", 1, 1, "P01_external_forcing_surf", ["S:f"], "all",
     "surface forcing arrays (before the tile loop)", {"pkg": "core"}),
    (OP, r"ENDDO1:CALL GRAD_SIGMA\(", 1, 1, "P02_rho_sigma_ivdc",
     ["S:m", "T:sigmaX:W:Nr", "T:sigmaY:S:Nr", "T:sigmaR:C:Nr"], "tile",
     "FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC: after the k loop (do_oceanic_phys.F:803-887)", {"pkg": "core"}),
    (OP, r"CALL CALC_OCE_MXLAYER\(", 1, 1, "P03_mxlayer", ["S:m"], "tile",
     "CALC_OCE_MXLAYER (calcGMRedi or doDiagsRho odd)", {"pkg": "core"}),
    (OP, r"CALL GGL90_CALC\(", 1, 1, "P04_ggl90", ["S:k"], "tile", "GGL90 TKE, viscosity, diffusivity",
     {"pkg": "ggl90"}),
    (OP, r"CALL KPP_CALC\(", 1, 1, "P07_kpp", ["S:K"], "tile",
     "KPP_CALC: viscosity, diffusivities (double diffusion with KPPuseDoubleDiff), non-local term, boundary-layer "
     "depth (calcKPP; KPP_CALC_DUMMY in AD builds is not this stage)", {"pkg": "kpp"}),
    (OP, r"CALL PP81_CALC\(", 1, 1, "P08_pp81", ["S:Q"], "tile", "PP81_CALC: Richardson-number viscosity, diffusivity",
     {"pkg": "pp81"}),
    (OP, r"CALL MY82_CALC\(", 1, 1, "P09_my82", ["S:Y"], "tile",
     "MY82_CALC: Mellor-Yamada viscosity, diffusivity, boundary-layer depth", {"pkg": "my82"}),
    (OP, r"CALL KPP_DO_EXCH\(", 1, 1, "P10_kpp_exch", ["S:K"], "all", "KPP fields after their halo exchange",
     {"pkg": "kpp"}),
    (OP, r"CALL GGL90_EXCHANGES\(", 1, 1, "P11_ggl90_exch", ["S:k"], "all",
     "GGL90 fields after GGL90_EXCHANGES (useGGL90; the dump runs after the one-line IF)", {"pkg": "ggl90"}),
    (OP, r"CALL GMREDI_CALC_TENSOR\(", 1, 1, "P05_gmredi_tensor", ["S:g"], "tile", "GM/Redi slopes, taper, tensor",
     {"pkg": "gmredi"}),
    (OP, r"CALL GMREDI_DO_EXCH\(", 1, 1, "P06_gmredi_exch", ["S:g"], "all", "GM/Redi tensor halo exchange",
     {"pkg": "gmredi"}),
    (FS, r"CALL DO_OCEANIC_PHYS\(", 1, 1, "S04_oceanic_phys", ["S:fmkgrt"], "all", "all of DO_OCEANIC_PHYS",
     {"pkg": "core"}),
    (FS, r"CALL THERMODYNAMICS\(", 1, 2, "S05_thermodynamics_sync", ["S:ta"], "all",
     "THERMODYNAMICS before DYNAMICS (staggerTimeStep=.FALSE.)", {"pkg": "core"}),
    (DY, r"CALL CALC_PHI_HYD\(", 1, 1, "D00a_phi_hyd",
     ["K:dPhiHydX:W", "K:dPhiHydY:S", "K:phiHydC:C", "K:phiHydF:C"], "tile",
     "hydrostatic pressure (per level k): gradient terms dPhiHydX/Y, phiHydC, phiHydF (next interface)",
     {"pkg": "core"}),
    (DY, r"CALL MOM_FLUXFORM\(", 1, 1, "D00b_mom_fluxform",
     ["K:gU:W:gU(1-OLx,1-OLy,k,bi,bj)", "K:gV:S:gV(1-OLx,1-OLy,k,bi,bj)", "K:guDissip:W", "K:gvDissip:S"], "tile",
     "flux-form momentum tendency of level k (gU, gV) and dissipation kept out of AB", {"pkg": "core"}),
    (DY, r"CALL MOM_VECINV\(", 1, 1, "D00c_mom_vecinv",
     ["K:gU:W:gU(1-OLx,1-OLy,k,bi,bj)", "K:gV:S:gV(1-OLx,1-OLy,k,bi,bj)", "K:guDissip:W", "K:gvDissip:S"], "tile",
     "vector-invariant momentum tendency of level k (gU, gV) and dissipation kept out of AB", {"pkg": "core"}),
    (DY, r"BEFORE:CALL IMPLDIFF\(", 1, 4, "D01_before_impl_visc", ["S:a", "T:kappaRU:W:Nr+1", "T:kappaRV:S:Nr+1"],
     "tile", "explicit gU, gV (after TIMESTEP) and vertical viscosities, input of IMPLDIFF (implicitViscosity)",
     {"pkg": "core"}),
    (DY, r"CALL IMPLDIFF\(", 2, 4, "D02_after_impl_visc", ["S:a"], "tile", "gU, gV after implicit viscosity",
     {"pkg": "core"}),
    (FS, r"CALL DYNAMICS\(", 1, 1, "S06_dynamics", ["S:adm"], "all",
     "phi_hyd, momentum tendencies, AB, implicit viscosity -> gU, gV", {"pkg": "core"}),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.TRUE\.", 1, 1, "S07_update_rstar_T", ["S:r"], "all", "r* at the new time",
     {"pkg": "core"}),
    (FS, r"CALL UPDATE_CG2D\(", 1, 1, "S08_update_cg2d", ["S:c"], "all", "cg2d operator + preconditioner",
     {"pkg": "core"}),
    (SP, r"BEFORE:CALL CG2D\(", 1, 1, "C01_cg2d_inputs", ["G:cg2d_b:C:1", "G:cg2d_x:C:1", "S:c"], "all",
     "cg2d right-hand side, first guess and operator", {"pkg": "core"}),
    (SP, r"CALL CG2D\(", 1, 1, "C02_cg2d_solution",
     ["G:cg2d_x:C:1", "N:numIters", "N:nIterMin", "N:firstResidual", "N:minResidualSq", "N:lastResidual"], "all",
     "cg2d solution (before its exchange), iteration count and residuals", {"pkg": "core"}),
    (FS, r"CALL SOLVE_FOR_PRESSURE\(", 1, 1, "S09_solve_for_pressure", ["S:d"], "all", "cg2d solve -> etaN",
     {"pkg": "core"}),
    (FS, r"CALL MOMENTUM_CORRECTION_STEP\(", 1, 1, "S10_momentum_correction", ["S:d"], "all", "u, v corrected",
     {"pkg": "core"}),
    (FS, r"CALL INTEGR_CONTINUITY\(", 1, 1, "S11_integr_continuity", ["S:d"], "all", "w, etaH",
     {"pkg": "core"}),
    (FS, r"CALL CALC_R_STAR\(", 1, 1, "S12_calc_rstar", ["S:r"], "all", "rStarFac from etaH", {"pkg": "core"}),
    (FS, r"CALL DO_STAGGER_FIELDS_EXCHANGES\(", 2, 2, "S13_stagger_exchanges", ["S:d"], "all",
     "exchanges before the staggered tracer step (staggerTimeStep=.TRUE.)", {"pkg": "core"}),
    (TH, r"CALL GMREDI_RESIDUAL_FLOW\(", 1, 1, "T01_residual_flow", ["T:uFld:W:Nr", "T:vFld:S:Nr", "T:wFld:C:Nr"],
     "tile", "Eulerian + bolus velocity used by tracer advection", {"pkg": "gmredi"}),
    (TI, r"CALL GAD_ADVECTION\(", 1, 1, "T10_temp_adv", ["T:gT_loc:C:Nr"], "tile",
     "theta: multi-dimensional advective tendency (multiDimAdvection schemes)", {"pkg": "gad"}),
    (TI, r"BEFORE:CALL TIMESTEP_TRACER\(", 1, 1, "T11_temp_gT", ["T:gT_loc:C:Nr"], "tile",
     "theta: total explicit tendency after forcing, diffusion, AB and r* rescale", {"pkg": "gad"}),
    (TI, r"CALL TIMESTEP_TRACER\(", 1, 1, "T12_temp_step", ["T:gT_loc:C:Nr"], "tile", "theta: T + dt*gT",
     {"pkg": "gad"}),
    (TI, r"BEFORE:IF \( AdamsBashforth_T \) THEN", 2, 2, "T13_temp_impl", ["T:gT_loc:C:Nr", "T:kappaRk:C:Nr"],
     "tile", "theta after the implicit vertical step (GAD_IMPLICIT_R or IMPLDIFF, temp_integrate.F:480-504, "
     "when one runs) and its input kappaRk", {"pkg": "gad"}),
    (TH, r"CALL TEMP_INTEGRATE\(", 1, 1, "T02_temp_integrate", ["S:ta"], "tile", "theta advanced",
     {"pkg": "core"}),
    (SI, r"CALL GAD_ADVECTION\(", 1, 1, "T20_salt_adv", ["T:gS_loc:C:Nr"], "tile",
     "salt: multi-dimensional advective tendency", {"pkg": "gad"}),
    (SI, r"BEFORE:CALL TIMESTEP_TRACER\(", 1, 1, "T21_salt_gS", ["T:gS_loc:C:Nr"], "tile",
     "salt: total explicit tendency after forcing, diffusion, AB and r* rescale", {"pkg": "gad"}),
    (SI, r"CALL TIMESTEP_TRACER\(", 1, 1, "T22_salt_step", ["T:gS_loc:C:Nr"], "tile", "salt: S + dt*gS",
     {"pkg": "gad"}),
    (SI, r"BEFORE:IF \( AdamsBashforth_S \) THEN", 2, 2, "T23_salt_impl", ["T:gS_loc:C:Nr", "T:kappaRk:C:Nr"],
     "tile", "salt after the implicit vertical step (GAD_IMPLICIT_R or IMPLDIFF) and its input kappaRk",
     {"pkg": "gad"}),
    (TH, r"CALL SALT_INTEGRATE\(", 1, 1, "T03_salt_integrate", ["S:ta"], "tile", "salt advanced",
     {"pkg": "core"}),
    (TH, r"CALL PTRACERS_INTEGRATE\(", 1, 1, "T04_ptracers_integrate", ["S:P"], "tile",
     "passive tracers advanced (PTRACERS_INTEGRATE, thermodynamics.F:348; DO_PTRACERS_HERE = ALLOW_PTRACERS without "
     "ALLOW_LONGSTEP)", {"pkg": "ptracers"}),
    (FS, r"CALL THERMODYNAMICS\(", 2, 2, "S14_thermodynamics_stagger", ["S:ta"], "all",
     "THERMODYNAMICS after the momentum step (staggerTimeStep=.TRUE.)", {"pkg": "core"}),
    (TC, r"CALL OPPS_INTERFACE\(", 1, 1, "T05_opps", ["S:t"], "tile",
     "theta, salt after the OPPS convective adjustment (useOPPS, tracers_correction_step.F:104-113)",
     {"pkg": "opps"}),
    (TC, r"CALL CONVECTIVE_ADJUSTMENT\(", 1, 1, "T06_convective_adjustment", ["S:t"], "tile",
     "theta, salt after CONVECTIVE_ADJUSTMENT (INCLUDE_CONVECT_CALL, .NOT.useOPPS and cAdjFreq /= 0; M3: "
     "front_relax input.bvp/.mxl/.top, cAdjFreq=-1)", {"pkg": "core"}),
    (FS, r"CALL TRACERS_CORRECTION_STEP\(", 1, 1, "S15_tracers_correction", ["S:tP"], "all", "end of the physics",
     {"pkg": "core"}),
    (FS, r"CALL DO_FIELDS_BLOCKING_EXCHANGES\(", 1, 1, "S16_blocking_exchanges", ["S:dtP"], "all",
     "state after the end-of-step exchanges (input of MONITOR, cost and output)", {"pkg": "core"}),
    (FS, r"CALL MONITOR\(", 1, 1, "S17_monitor", ["S:dtP"], "all", "state at MONITOR (its statistics: STDOUT)",
     {"pkg": "monitor"}),
    (FS, r"CALL COST_TILE\s*\(", 1, 1, "S18_cost_tile", ["S:q"], "all", "cost terms after COST_TILE",
     {"pkg": "cost"}),
    (IO, r"CALL SBO_CALC\(", 1, 1, "S19_sbo_calc", ["S:o"], "all", "SBO_CALC: OAM, mass, centre of mass scalars",
     {"pkg": "sbo"}),
    (ML, r"CALL COST_FINAL\s*\(", 1, 1, "E01_cost_final", ["S:q"], "all",
     "COST_FINAL after the time loop (records carry the start iteration of the LAST step)", {"pkg": "cost"}),
]

_COMMENT = re.compile(r"^[cC*!]")
_CPP = re.compile(r"^#")
_CONT = re.compile(r"^     [^ 0]")
_DO = re.compile(r"^\s+(?:\d+\s+)?DO\s+(?:[A-Za-z]\w*\s*=|WHILE\b)", re.I)
_LABELLED_DO = re.compile(r"^\s+(?:\d+\s+)?DO\s+\d+", re.I)
_ENDDO = re.compile(r"^\s+(?:\d+\s+)?END\s*DO\b", re.I)
ITER_UPDATE = re.compile(r"^\s+myIter = nIter0 \+ iLoop\s*$")
MARK = "C--   jaxdump (mitjax Task 4) -- no effect unless JAXDUMP_DIR is set"


def entries(pkgs=None, requested=None):
    """STAGES as 9-tuples (file, anchor, occ, total, stage, dumps, scope, what, opts), optionally of some packages;
    with `requested` (the build's requested package set) the stages of package files (FILE_PKG) whose package is not
    requested are left out (their file is not copied)."""
    return [e for e in STAGES if (pkgs is None or e[8]["pkg"] in pkgs)
            and (requested is None or e[0] not in FILE_PKG or FILE_PKG[e[0]] in requested)]


def requested_packages(rootdir, mods):
    """The package set a build's packages.conf requests (reference/check_build.py, groups expanded, `-pkg` removed):
    the same set check_build verifies against genmake2's ENABLED_PACKAGES. The build directory holds no packages.conf
    when instrument.py runs (step 1b, before genmake2), so the -mods directory decides."""
    spec = __import__("importlib.util").util.spec_from_file_location("_mjx_check_build", HERE.parent / "check_build.py")
    cb = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    groups = cb.read_groups(Path(rootdir) / "pkg" / "pkg_groups")
    req, _, _ = cb.requested_packages(cb.find_packages_conf(Path(mods), [Path(mods)]), groups)
    return req


class Source:
    """Where instrument() reads master's files: a MITgcm tree on disk (+ experiment mods), or a git revision."""

    def __init__(self, root=None, mods=None, git=None, rev="pinned"):
        self.root, self.mods, self.git, self.rev = (Path(root) if root else None, Path(mods) if mods else None,
                                                    git, rev)

    def path(self, fname):
        if self.mods and (self.mods / fname).exists():
            return self.mods / fname, f"<mods>/{fname}"
        if self.git:
            return None, FILES[fname]
        return self.root / FILES[fname], FILES[fname]

    def text(self, fname):
        p, label = self.path(fname)
        if p is not None:
            return p.read_text(), label
        out = subprocess.run(["git", "-C", str(self.git), "show", f"{self.rev}:{FILES[fname]}"], check=True,
                             capture_output=True, text=True).stdout
        return out, label


def _wrap(lines):
    """Re-wrap one generated CALL to fixed-form 72 columns when the default two-line layout is too long."""
    if all(len(ln) <= 72 for ln in lines):
        return lines
    text = " ".join(ln[6:].strip() if ln.startswith("     &") else ln.strip() for ln in lines)
    head, inner = text.split("(", 1)
    inner = inner.rsplit(")", 1)[0]
    args, depth, cur_arg, quoted = [], 0, "", False  # split at top-level commas only (expressions keep theirs)
    for ch in inner:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            args.append(cur_arg.strip())
            cur_arg = ""
        else:
            cur_arg += ch
    args.append(cur_arg.strip())
    out, cur = [], f"      {head.strip()}( "
    for n, a in enumerate(args):
        tok = a + (", " if n < len(args) - 1 else " )")
        if len(cur) + len(tok.rstrip()) > 72:
            out.append(cur.rstrip())
            cur = "     &   "
        cur += tok
    out.append(cur.rstrip())
    assert all(len(ln) <= 72 for ln in out), out
    return out


def _calls(stage, dumps, scope, opts=None):
    opts = opts or {}
    out = []
    for d in dumps:
        out += _wrap(_call(stage, d, scope, opts.get("iter", "myIter")))
    if "loop" in opts:  # stage name gets '_p<value>' (JAXDUMP_PASS; 0 switches the suffix off again)
        out = [f"      CALL JAXDUMP_PASS( {opts['loop']} )"] + out + ["      CALL JAXDUMP_PASS( 0 )"]
    if "cond" in opts:
        out = [f"      IF ( {opts['cond']} ) THEN"] + out + ["      ENDIF"]
    if "shift" in opts:  # start of the step: records carry myIter - 0 until the update line (module docstring)
        out = [f"      CALL JAXDUMP_ITERSHIFT( {opts['shift']} )"] + out
    return out


def _call(stage, d, scope, it="myIter"):
    """Fortran lines of one dump statement; `it` is the iteration expression (the shim subtracts its runtime shift,
    module docstring)."""
    bi, bj = ("bi", "bj") if scope == "tile" else ("0", "0")
    kind, rest = d.split(":", 1)
    if kind == "S":
        return [f"      CALL JAXDUMP_STATE( '{stage}', '{rest}',",
                f"     &                    {bi}, {bj}, {it}, myThid )"]
    if kind == "T":
        name, pk, nz = rest.split(":")
        return [f"      CALL JAXDUMP_TILE( '{stage}', '{name}', '{pk}',",
                f"     &                   {name}, {nz}, bi, bj, {it}, myThid )"]
    if kind == "K":
        parts = rest.split(":", 2)
        name, pk = parts[0], parts[1]
        expr = parts[2] if len(parts) > 2 else name
        return [f"      CALL JAXDUMP_TILEK( '{stage}', '{name}', '{pk}',",
                f"     &   {expr}, k, bi, bj, {it}, myThid )"]
    if kind == "U":
        name, pk, nz = rest.split(":")
        return [f"      CALL JAXDUMP_TILEI( '{stage}', '{name}', '{pk}',",
                f"     &                    {name}, {nz}, bi, bj, {it}, myThid )"]
    if kind == "N":
        parts = rest.split(":", 1)
        name = parts[0]
        expr = parts[1] if len(parts) > 1 else name
        return [f"      CALL JAXDUMP_SCALAR( '{stage}', '{name}',",
                f"     &                     DBLE({expr}), {it}, myThid )"]
    if kind == "G":
        name, pk, nz = rest.split(":")
        return [f"      CALL JAXDUMP_LOCAL( '{stage}', '{name}', '{pk}',",
                f"     &                    {name}, {nz}, {it}, myThid )"]
    raise SystemExit(f"{stage}: unknown dump kind {d!r}")


def _is_code(ln):
    return not _COMMENT.match(ln) and not _CPP.match(ln)


def _statement_end(lines, i):
    """Index of the last line of the statement starting at line i: continuation lines, including those that follow
    comment or preprocessor lines inside the statement (e.g. an #ifdef'd argument)."""
    last, j = i, i + 1
    while j < len(lines):
        if _CONT.match(lines[j]):
            last, j = j, j + 1
            continue
        k = j
        while k < len(lines) and not _is_code(lines[k]):
            k += 1
        if k > j and k < len(lines) and _CONT.match(lines[k]):
            j = k
            continue
        break
    return last


def _enddo_after(lines, i, level, src):
    """Line index of the ENDDO closing the `level`-th DO loop enclosing line i (line i counts when it is a DO)."""
    def check(ln, k):
        if _LABELLED_DO.match(ln):
            raise SystemExit(f"{src}:{k + 1}: labelled DO loop, ENDDO anchors do not support it")
    enclosing = []
    if _DO.match(lines[i]):
        enclosing.append(i)
    depth, k = 0, i - 1
    while len(enclosing) < level and k >= 0:
        ln = lines[k]
        if _is_code(ln):
            check(ln, k)
            if _ENDDO.match(ln):
                depth += 1
            elif _DO.match(ln):
                if depth:
                    depth -= 1
                else:
                    enclosing.append(k)
        k -= 1
    if len(enclosing) < level:
        raise SystemExit(f"{src}:{i + 1}: fewer than {level} enclosing DO loops")
    d0 = enclosing[level - 1]
    depth = 0
    for k in range(d0, len(lines)):
        ln = lines[k]
        if not _is_code(ln):
            continue
        check(ln, k)
        if _DO.match(ln):
            depth += 1
        elif _ENDDO.match(ln):
            depth -= 1
            if depth == 0:
                indent = lambda s: len(s) - len(s.lstrip())  # noqa: E731
                if indent(ln) != indent(lines[d0]):
                    raise SystemExit(f"{src}:{k + 1}: ENDDO indentation differs from its DO at line {d0 + 1}")
                return k
    raise SystemExit(f"{src}:{d0 + 1}: no matching ENDDO")


def cpp_context(lines, j):
    """Stack of the #if/#ifdef/#ifndef lines (with #else noted) that enclose an insertion BEFORE line index j."""
    stack = []
    for ln in lines[:j]:
        s = ln.strip()
        if re.match(r"#\s*if", s):
            stack.append(s.split("/*")[0].strip())
        elif re.match(r"#\s*el", s):
            if not stack:
                raise SystemExit(f"unbalanced {s!r}")
            stack[-1] = stack[-1] + " [" + s.split("/*")[0].strip() + "]"
        elif re.match(r"#\s*endif", s):
            if not stack:
                raise SystemExit("unbalanced #endif")
            stack.pop()
    return stack


def _cppend_after(lines, i, src):
    """Line index of the #endif closing the innermost CPP block that contains line i."""
    depth = 0
    for k in range(i + 1, len(lines)):
        s = lines[k].strip()
        if re.match(r"#\s*if", s):
            depth += 1
        elif re.match(r"#\s*endif", s):
            if depth == 0:
                return k
            depth -= 1
    raise SystemExit(f"{src}:{i + 1}: no enclosing CPP block")


def _parse_anchor(anchor):
    """-> (mode, level, regex): mode 'after' | 'before' | 'enddo' | 'cppend'."""
    if anchor.startswith("BEFORE:"):
        return "before", 0, anchor[len("BEFORE:"):]
    if anchor.startswith("CPPEND:"):
        return "cppend", 0, anchor[len("CPPEND:"):]
    m = re.match(r"ENDDO(\d+):(.*)$", anchor)
    if m:
        return "enddo", int(m.group(1)), m.group(2)
    return "after", 0, anchor


def _hits(lines, pat):
    return [i for i, ln in enumerate(lines) if not _COMMENT.match(ln) and not _CPP.match(ln) and re.search(pat, ln)]


def plan(source, pkgs=None, requested=None):
    """Insertion plan per file: {fname: (lines, label, {line index: [inserted lines]}, report rows)}.
    Fails (SystemExit) on any anchor-count drift or a misplaced iteration update."""
    stages = entries(pkgs, requested)
    out = {}
    for fname in sorted({s[0] for s in stages}):
        text, label = source.text(fname)
        lines = text.split("\n")
        inserts, rows = {}, []
        for f, anchor, occ, total, stage, dumps, scope, what, opts in stages:
            if f != fname:
                continue
            mode, level, pat = _parse_anchor(anchor)
            hits = _hits(lines, pat)
            if len(hits) != total:
                raise SystemExit(f"{label}: anchor {pat!r} found {len(hits)} times, expected {total}")
            i = hits[occ - 1]
            if mode == "before":
                j = i
            elif mode == "enddo":
                j = _enddo_after(lines, i, level, label) + 1
            elif mode == "cppend":
                j = _cppend_after(lines, i, label) + 1
            else:
                j = _statement_end(lines, i) + 1
            ctx = cpp_context(lines, j)
            if "shift" in opts and ctx:
                raise SystemExit(f"{label}: {stage} sets the iteration shift inside CPP block(s) {ctx}")
            inserts.setdefault(j, []).extend(_calls(stage, dumps, scope, opts))
            rows.append({"stage": stage, "file": fname, "label": label, "line": i + 1, "insert_before": j + 1,
                         "mode": mode, "scope": scope, "dumps": dumps, "what": what, "opts": opts, "cpp": ctx})
        if fname == FS:
            upd = [i for i, ln in enumerate(lines) if ITER_UPDATE.match(ln)]
            dyn = _hits(lines, r"CALL DYNAMICS\(")
            rst = _hits(lines, r"CALL UPDATE_R_STAR\(\s*\.TRUE\.")
            if not (len(upd) == 1 and len(dyn) == 1 and len(rst) == 1 and dyn[0] < upd[0] < rst[0]):
                raise SystemExit(f"{label}: iteration-counter update not between DYNAMICS and UPDATE_R_STAR(.TRUE.)"
                                 f" (update lines {[u + 1 for u in upd]})")
            if cpp_context(lines, upd[0] + 1):
                raise SystemExit(f"{label}:{upd[0] + 1}: iteration update inside a CPP block")
            inserts.setdefault(upd[0] + 1, []).insert(0, "      CALL JAXDUMP_ITERSHIFT( 1 )")
            rows.append({"stage": "(iteration shift 1)", "file": fname, "label": label, "line": upd[0] + 1,
                         "insert_before": upd[0] + 2, "mode": "after", "scope": "-", "dumps": [], "what":
                         "myIter = nIter0 + iLoop: later records carry myIter-1", "opts": {}, "cpp": []})
        out[fname] = (lines, label, inserts, rows)
    return out


def instrument(source, outdir, pkgs=None, requested=None):
    """Write instrumented copies of every file with a stage + jaxdump.F + JAXDUMP.h into outdir (a new or empty
    directory; nothing is overwritten). `requested`: the build's requested package set (package files of other
    packages are not copied). Returns the report rows."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    for fname, (lines, label, inserts, rows) in plan(source, pkgs, requested).items():
        new = []
        for k, ln in enumerate(lines):
            if k in inserts:
                new += [MARK] + inserts[k]
            new.append(ln)
        if len(lines) in inserts:
            new += [MARK] + inserts[len(lines)]
        dst = outdir / fname
        if dst.exists():
            raise SystemExit(f"{dst} exists (nothing is overwritten)")
        for ln in new:
            if _is_code(ln) and len(ln) > 72 and ln.startswith("      CALL JAXDUMP"):
                raise SystemExit(f"{fname}: generated line longer than 72 columns: {ln!r}")
        dst.write_text("\n".join(new))
        report += rows
    for f in ("jaxdump.F", "JAXDUMP.h"):
        if (outdir / f).exists():
            raise SystemExit(f"{outdir / f} exists (nothing is overwritten)")
        shutil.copy(HERE / f, outdir / f)
    return report


def anchor_counts(source):
    """{(file, anchor): found count} for every stage anchor, and the expected count (tests: drift detection)."""
    out = {}
    for f, anchor, occ, total, *_ in STAGES:
        text, _ = source.text(f)
        out[(f, anchor)] = (len(_hits(text.split("\n"), _parse_anchor(anchor)[2])), total)
    return out


def markdown(source):
    rows = ["| stage | package list | anchor (master `63cdc0b`) | compiled only if | scope | dumps | what |",
            "|---|---|---|---|---|---|---|"]
    allrows = []
    for fname, (_, label, _, rr) in plan(source).items():
        allrows += rr
    order = {e[4]: n for n, e in enumerate(STAGES)}
    allrows.sort(key=lambda r: order.get(r["stage"], -1))
    for r in allrows:
        where = {"before": "before ", "enddo": "after the ENDDO of the loop around ",
                 "cppend": "after the `#endif` around "}.get(r["mode"], "")
        sc = r["scope"]
        if "loop" in r["opts"]:
            sc += f", per `{r['opts']['loop']}` (stage `_p<n>`)"
        if "cond" in r["opts"]:
            sc += f", if `{r['opts']['cond']}`"
        if "iter" in r["opts"]:
            sc += f", iteration `{r['opts']['iter']}`"
        cpp = "; ".join(f"`{c}`" for c in r["cpp"]) or "always"
        rows.append(f"| `{r['stage']}` | {r['opts'].get('pkg', 'core')} | {where}`{r['label']}:{r['line']}` | {cpp} "
                    f"| {sc} | {', '.join(r['dumps'])} | {r['what']} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--root", help="MITgcm tree (the build snapshot)")
    ap.add_argument("--mods", help="experiment code directory (its copy of a file wins)")
    ap.add_argument("--git", help="MITgcm clone (read a revision with git show)")
    ap.add_argument("--rev", default="pinned")
    ap.add_argument("--check", action="store_true", help="anchor counts and CPP checks only")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if not (a.root or a.git):
        ap.error("--root or --git")
    src = Source(root=a.root, mods=a.mods, git=a.git, rev=a.rev)
    if a.markdown:
        print(markdown(src))
        return 0
    if a.check:
        for fname, (_, label, inserts, rows) in plan(src).items():
            for r in rows:
                print(f"{r['stage']:28s} {r['label']}:{r['line']}  cpp={r['cpp']}")
        print("ANCHORS OK")
        return 0
    if not a.outdir:
        ap.error("OUTDIR")
    requested = None
    if a.root and a.mods:  # a build (reference/build.sh step 1b): package files only for requested packages
        requested = requested_packages(a.root, a.mods)
        for f in sorted(FILE_PKG):
            if FILE_PKG[f] not in requested:
                print(f"(skipped {FILES[f]}: package {FILE_PKG[f]} not requested by {a.mods}/packages.conf)")
    for r in instrument(src, a.outdir, requested=requested):
        print(f"{r['stage']:28s} {r['label']}:{r['line']} -> before line {r['insert_before']}  {r['scope']}  "
              f"cpp={r['cpp']}  {r['dumps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
