# Issues found in MITgcm master (63cdc0b)

Bugs or inconsistencies in the upstream Fortran found while porting, for the MITgcm developers. Each entry: file:line at
`63cdc0b`, what happens, how it was found (gate, experiment, job), and whether the port reproduces it (literal port:
yes, unless Nikolay approved a `# DEVIATION:`). Reporting upstream only on Nikolay's say.

Before an upstream report: re-verify every number against the files; rule out configuration explanations -- for an
adjoint, read every cited run's `data.autodiff` and the AD-mode block of its STDOUT (reverse-sweep-only switches such as
useApproxAdvectionInAdMode or SEAICEuseFREEDRIFTswitchInAd make an ADM/TLM gap expected; MITgcm#1041 missed two); check
upstream history and existing issues; show Nikolay the exact text; scan for markers; post from his account.

## pkg/layers: unbalanced `#ifdef ALLOW_LAYERS` between two files
- `pkg/layers/layers_fluxcalc.F:80` opens `#ifdef ALLOW_LAYERS` and the file ends without its `#endif` (26 `#if*`, 25
  `#endif`); `pkg/layers/layers_locate.F:112` has `#endif /* ALLOW_LAYERS */` without an `#if` (1 `#if*`, 2 `#endif`).
- cpp reports `<stdin>:80: error: unterminated #ifdef` and `<stdin>:112: error: #endif without #if` in every build that
  compiles pkg/layers (seen in the make.log of `tutorial_reentrant_channel/code`, lane A session 9, job 27840380), but
  the Makefile rule `cat $< | cpp -traditional -P ... | set64bitConst.sh` takes the exit status of the last command, so
  make goes on: the conditional is closed at the end of the file (ALLOW_LAYERS is defined, so the code is compiled) and
  the stray `#endif` is ignored. The compiled code is as intended; the sources are not.
- Port: nothing to reproduce (the compiled code is the intended one). Our coverage mapping (tools/cpp_live.py, which
  treats a cpp error as fatal) cannot map these two files; recorded as known mapping problems in
  `scripts/tests/test_m3_oracle.py`.

## pkg/seaice: `SEAICEnonLinTol` has no default value
- `SEAICEnonLinTol` (SEAICE_PARAMS.h:557, namelist SEAICE_PARM01 at seaice_readparms.F:147) is read by SEAICE_LSR
  (seaice_lsr.F:747), SEAICE_KRYLOV (seaice_krylov.F:401) and SEAICE_JFNK (seaice_jfnk.F:315), but no routine of
  pkg/seaice assigns it a default (grep over pkg/seaice at 63cdc0b: only the namelist, the retired-parameter message at
  seaice_readparms.F:1124 and SEAICE_SUMMARY). A run with implicit dynamics that does not set it in data.seaice uses an
  uninitialised common-block value. The verification runs that use it set it (offline_exf_seaice/input.dyn_lsr/data.seaice:10,
  1.e-10). Found by lane M4OFF session 3 (2026-10-03).
- Port: refuses a dynamics run without `SEAICEnonLinTol` in data.seaice.
- Further evidence (lane M4LAB session 1, 2026-10-03): lab_sea/input does not set it; its SEAICE_SUMMARY prints the
  uninitialised value, `SEAICEnonLinTol = /* non-linear solver tolerance */ 0.000000000000000E+00` (oracle STDOUT of
  job27855987-jdon, output.txt:1384-1385). Harmless there: the build has no SEAICE_ALLOW_LSR_FLEX, so the reader at
  seaice_lsr.F:747 (inside the flex arm :719-779) is not compiled, and the plain LSR stops on LSR_ERROR (:955, :978).

## pkg/autodiff: TAF's tangent of CG2D keeps the operator passive even with `cg2dFullAdjoint = .TRUE.`
- `pkg/autodiff/cg2d.flow:4-8`: `ADNAME = cg2d_mad`, `FTLNAME = cg2d`, `ACTIVE = 1,2` (cg2d_b, cg2d_x only). The adjoint
  CG2D_MAD differentiates the operator when cg2dFullAdjoint is set (cg2d_mad.F:175-204), but the tangent-linear code
  calls CG2D itself, so the TLM never differentiates the operator. With cg2dFullAdjoint = .TRUE. (global_ocean.90x40x15/
  input_ad.bottomdrag, bottom_ctrl_5x5/input_ad.facg2d) the TLM and the adjoint are then derivatives of different
  maps whenever a control reaches the operator. Found by lane GO (cs32x15 input_ad, 2026-10-02); the size of the effect
  on bottomdrag's TLM output is not measured yet.
- Port: one derivative rule for both directions (mitjax/ad/cg2d_rule.py with ad/modes.py); a TAF-compatible TLM would
  be a separate switch (not implemented).

## pkg/seaice: LSR adjoint without `SEAICE_LSR_ADJOINT_ITER` is wrong by construction, in two verification setups
- seaice_lsr.F:803-808: unless SEAICE_LSR_ADJOINT_ITER is defined, the tape key of the LSR iterate is the same for
  every iteration, "so that the tapes get overwritten for each iterate and the adjoint is necessarily wrong". It is
  #undef'd in global_ocean.cs32x15/code_ad/SEAICE_OPTIONS.h:180 and offline_exf_seaice/code_ad/SEAICE_OPTIONS.h:186
  (used by input_ad.seaice and input_ad.obcs with dynamics); lab_sea/code_ad defines it (:180). The reference adjoint
  gradients of those runs therefore include a knowingly wrong LSR adjoint. Found by lane M4OFF session 3 (2026-10-03).
- **Correction (main session 2026-10-05): only cs32x15/input_ad.seaice actually runs this LSR adjoint, and it is
  confounded there.** offline_exf_seaice/input_ad.obcs sets `SEAICEuseFREEDRIFTswitchInAd = .TRUE.` (data.autodiff;
  results/output_adm.obcs.txt prints T): ADAUTODIFF_INADMODE_SET (autodiff_inadmode_set_ad.F:71-73) swaps the LSR for
  free drift in the reverse sweep, so TAF never differentiates SEAICE_LSR in that run (its TLM keeps the LSR:
  G_AUTODIFF_INADMODE_SET only sets inAdMode = .FALSE.). cs32x15/input_ad.seaice sets `useApproxAdvectionInAdMode =
  .TRUE.` with tempAdvScheme = saltAdvScheme = SEAICEadvScheme = 33: the reverse sweep uses scheme 30 (no flux
  limiter; gad_advection.F:194-202, gad_calc_rhs.F:176-184, seaice_advection.F:177-183) while the TLM keeps 33, so
  its ADM-vs-TLM gap (2-3 digits) mixes the advection approximation with this LSR adjoint. seaice_dynmix, at the same
  check points (xx_theta, i = 1..4, j = 1, k = 1, tile 1), has ADM (free drift) and TLM (LSR) agreeing to 7-8 digits,
  so the ice dynamics probably barely enters these gradients -- the cs32x15.seaice gap is likely mostly the advection
  approximation. To be measured by the port (plan decision 17, revised).
- Port: decided at step 7 (plan decision 11); the comparison with TAF for those runs will document the difference.
- **Measured (lane M4ADCS32ICE session 2, 2026-10-05): the cs32x15.seaice gap is the approximate advection.** At the
  4 grdchk points (xx_theta, i = 1..4, j = 1, k = 1, tile 1) mitjax's exact derivative (scheme 33 in the backward pass,
  the LSR differentiated through its executed sweeps) equals TAF's TLM (results/output_tlm.seaice.txt.gz) to 11-12
  digits (rel 4.9e-13 .. 9.6e-12, dev job 27906279); with useApproxAdvectionInAdMode followed in the backward pass
  (scheme 30, GAD_ADVECTION recomputed per routine, SEAICE_ADVECTION per flux call) it equals TAF's ADM
  (results/output_adm.seaice.txt) to 7-8 digits (rel 8.1e-9, 1.5e-8, 2.5e-8, 3.3e-8; dev job 27906489), against 2-3
  digits without it. So the split, as measured: the approximate advection explains the whole ADM-vs-TLM gap (3e-4 ..
  1.2e-3) except a residual of 8.1e-9 .. 3.3e-8 (relative, the four points), which is what TAF's LSR adjoint without
  SEAICE_LSR_ADJOINT_ITER contributes there (about 1e-8).
  Where the remainder sits (adjoint monitor, reverse of the last step): TAF's ocean adjoints are reproduced (adtheta
  every statistic 16 digits, adsalt >= 9), its sea-ice ones are not (aduice max 1.98 vs 0.21, sd 0.137 vs 0.0083, i.e.
  ~10x, as in lab_sea; adarea mean 3.4e-4 vs 1.6e-3). Both cg2d settings passive (cg2dFullAdjoint = .FALSE.; TAF's
  TLM always passive). This settles the "likely" of MITgcm#1041 for cs32x15.seaice (a follow-up comment only on
  Nikolay's say).
- **Measured (lane M4ADCS32ICE session 3, 2026-10-06): TAF's LSR adjoint passes no sensitivity to the sea-surface
  tilt.** TAF prints ad_dynstat_adeta = 0 exactly at every reverse block of input_ad.seaice (results/
  output_adm.seaice.txt), and also of cs32x15/input_ad (no sea ice). In both seaice variants etaN at the start of a
  step enters the step only through the sea-ice tilt (SEAICEuseTilt default .TRUE., seaice_readparms.F:252;
  phiSurf = Bo_surf*etaN, seaice_dynsolver.F:236, into FORCEX0/FORCEY0 :285-296; exactConserv: the CG2D right-hand
  side uses etaH, solve_for_pressure.F:215-224): with SEAICEuseTilt planted .FALSE. our adeta is exactly 0 at every
  block in both variants (dev jobs 27907697, 27907745). With the tilt on (the runs' setting) the exact adjoint of
  input_ad.seaice (our run mode, the LSR differentiated through its sweeps) has adeta max 9.3e-2 / min -8.9e-2 at
  block 36001, where TAF prints 0; in input_ad.seaice_dynmix, whose reverse sweep differentiates the free drift
  instead of the LSR (SEAICEuseFREEDRIFTswitchInAd), TAF's adeta (max 4.12e-2 .. 4.48e-2) equals ours to 12 digits.
  So TAF's reverse sweep carries the tilt sensitivity through SEAICE_FREEDRIFT but not through its SEAICE_LSR
  adjoint: in this build the LSR adjoint returns no adjoint to the tilt part of the forcing. A concrete, checkable
  symptom of the overwritten LSR tape (adeta should be nonzero; compare with a SEAICE_LSR_ADJOINT_ITER build).
- **Measured (lane M4COSTSHARD session 1, 2026-10-06): the EXF adjoint monitor shows the same split.** ADEXF_MONITOR
  (iwhen = 3, exf_monitor_ad.F:204-218) prints adfu, adfv, adQnet, adEmPmR, adQsw at the reverse of the last call of
  SEAICE_MODEL (seaice_model.F:405). mitjax's run-mode adjoint (the cotangents of FFIELDS.h tapped right after
  SEAICE_MODEL, dev jobs 27908744 / 27908745): input_ad.seaice_dynmix (free drift in TAF's reverse sweep) matches
  TAF's 5 blocks in every record to 9-16 digits; input_ad.seaice matches the last step's block (36001: 10-16 digits)
  but at block 36000, i.e. after TAF's reverse of step 36001's SEAICE_LSR, adfu / adfv / adEmPmR keep only 1-8 digits
  (adfu_mean 1, adfv_mean 2, adempmr_mean 4) while adQnet / adQsw keep 12-16. So the departure enters exactly where
  the reverse sweep crosses the LSR adjoint, and only in the fields the ice dynamics feeds.

## pkg/seaice: the LSR forward in offline_exf_seaice/input.dyn_lsr is far from its own tolerance
- With SEAICEnonLinIterMax = 20 Picard passes and SEAICEnonLinTol = 1.e-10, the nonlinear residual falls only to
  2.5e-2 (step 1) and 0.14-0.29 (steps 2-12) of its first-pass value; each linear LSR solve stops after a 1-63 %
  reduction (SOLV_NCHECK 2). Not a bug (fixed pass counts are a common choice), but the printed tolerance is never
  reached and an adjoint "at the converged solution" does not describe this forward. Measured by lane M4OFF session 3
  (dev jobs 27870929, 27870970; the oracle's 1200 SEAICE_LSR STDOUT lines).

## verification/1D_ocean_ice_column/input/data.exf: three periods assigned twice
- data.exf:34-35 `atempperiod = 2635200.0,` then `= 86400.0,`; :67-68 `swdownperiod` and :72-73 `lwdownperiod` the
  same. The namelist read keeps the last value (86400, confirmed by the run's STDOUT), so the first lines are dead and
  probably leftovers. Found by lane A session 11 (2026-10-02).
- Port: reads the namelist like the Fortran (last assignment wins).

## model/src: a restart does not reproduce the `EtaH` pickup record when nonlinFreeSurf = 0
- WRITE_PICKUP always writes etaHnm1 as `EtaH` (write_pickup.F:354-360, comment: "always write dEtaHdt & EtaH but read
  only if exactConserv & nonlinFreeSurf"); READ_PICKUP reads it only when nonlinFreeSurf > 0 (read_pickup.F:228-229);
  INI_PSURF sets etaH/etaHnm1 = etaN only on a cold start (ini_psurf.F:78-88). With momStepping = .FALSE. nothing
  updates etaHnm1, so after a restart the next pickup's `EtaH` record is 0 where the continuous run writes etaN
  (offline_exf_seaice/input.dyn_lsr, pSurfInitFile set: restart 6 + 6 vs 12 steps, lane M4OFF session 3, job 27870970).
  The record is not read back in that configuration, so the physics is unaffected; restart pickups are not bitwise
  reproducible. Intentional per the comment; listed for completeness.
- Port: reproduces it (the restart test asserts the one differing record).

## pkg/salt_plume: compiled but unused, its parameters are read without ever being set
- `SALT_PLUME_READPARMS` returns at `pkg/salt_plume/salt_plume_readparms.F:46-54` when `useSALT_PLUME = .FALSE.`, before
  the defaults of `SPsalFRAC` (:72, 1.0) and `SaltPlumeSouthernOcean` (:59, .TRUE.) are assigned. `SEAICE_GROWTH` still
  computes `saltPlumeFlux = MAX(tmpscal3-tmpscal2, 0.)*localSPfrac` with `localSPfrac = SPsalFRAC`
  (`pkg/seaice/seaice_growth.F:2008-2045`, inside `#ifdef ALLOW_SALT_PLUME` with no `useSALT_PLUME` test) and tests
  `.NOT.SaltPlumeSouthernOcean` (:2041-2044); `KPP_TRANSPORT_S` reads `KPPplumefrac` (`pkg/kpp/kpp_transport_s.F:99`,
  :115), which only the `useSALT_PLUME` arm of `KPP_CALC` (`kpp_calc.F:680-711`) and `KPP_CALC_DUMMY` (:772) write. All three
  are uninitialised common-block values in a build that compiles pkg/salt_plume with `useSALT_PLUME = .FALSE.`.
- Found in lab_sea/input (packages.conf compiles salt_plume, data.pkg does not switch it on), lane M4LAB session 2
  (2026-10-04): the oracle job27855987-jdon dumps `saltPlumeFlux = +0.` on every point at I04_growth, iterations 1-3
  (gfortran's zero-filled common: SPsalFRAC = 0.). Harmless there: KPP_TRANSPORT_S multiplies the term by
  `tmpFac1 = 0.` (:72-82), and nothing else reads saltPlumeFlux without `useSALT_PLUME`; a compiler or loader that does
  not zero common blocks (or a NaN-initialising debug build) would make it a NaN times zero.
- Port: reproduces the oracle (SALT_PLUME.h fields and KPPplumefrac as zero commons, drivers/model.py; the KPP terms
  formed literally); SEAICE_GROWTH's saltPlumeFlux is ported literally with the never-written SPsalFRAC = 0. and
  SaltPlumeSouthernOcean = .FALSE. (pkg/salt_plume/salt_plume_readparms.py, lane M4LAB session 3), bitwise at
  I04_growth, iterations 1-3 (test_m4lab_seaice; a planted SPsalFRAC = 1 changes saltPlumeFlux).

## pkg/seaice: the plain LSR's "safeguard against bad forcing" freezes the iterate and then reports convergence
- In the LSR without SEAICE_ALLOW_LSR_FLEX (lab_sea's build), `IF(m.GT.1.AND.S1.GT.S1A) WFAU=WFAU2`
  (`pkg/seaice/seaice_lsr.F:953`, likewise `:976` for v with WFAV) switches the relaxation factor to `WFAU2 = ZERO`
  (`:311-312`). TRIDIAGU then returns `uIce = uTmp + WFAU*(URT-uTmp) = uTmp` (`:2054-2056`): the iterate no longer
  moves, the next SOLV_NCHECK test (`:937-958`) measures `S1 = MAX|uIce-uTmp| = 0 < LSR_ERROR` and stops the u
  iteration as converged (`ICOUNT1 = m`), so the "did not converge" warning (`:1047-1057`) cannot fire after the
  safeguard has acted, however far the iterate is from the solution. A relaxation factor of zero is presumably meant
  to be a reduced one (the flex arm has no such safeguard).
- Read from the Fortran (lane M4LAB session 3, 2026-10-04) and verified by the main session (2026-10-04). Not yet seen
  to trigger in a run: in lab_sea/input's whole 9-step run (the port, bitwise with the oracle) WFAU and WFAV still
  hold SEAICE_LSRrelaxU/V = 0.95 at the end of both passes of every step, every pass ending with S1, S2 < 1e-4 =
  LSR_ERROR after 4-30 sweeps (measured: lane M4LAB session 4, dev job 27873741, the run driver's per-pass LSR
  values).
- Port: reproduces it literally (`mitjax/pkg/seaice/seaice_lsr.py` body_maxnorm, `WFAU2 = ZERO`).

## pkg/seaice: SItracer — a misleading restart warning, a diffusion switch on the wrong field, a stale snapshot
- `SEAICE_CHECK_PICKUP` warns `restart without "siTracNN" (set to zero)` for a missing tracer record
  (`pkg/seaice/seaice_check_pickup.F:159-174`), but nothing sets it to zero: `READ_MFLDS_3D_RL` leaves the field
  untouched when the record is missing (`pkg/rw/read_mflds.F:308-321`), so it keeps `SEAICE_INIT_VARIA`'s value, which is
  1 for a tracer named 'one' (`seaice_init_varia.F:137-142`). Seen in lab_sea/input (oracle job27855987: STDERR.0000:8-9
  prints the warning for siTrac01 'age' and siTrac02 'one'; the first SEAICE monitor block prints
  `seaice_sitracer02_max/min/mean = 1.0`, output.txt:3299-3301). Harmless for the values (1 is the right start for
  'one'); the message is wrong. The port reproduces the values (the warning text goes to STDERR, which the port does
  not print for this routine).
- `SEAICE_ADVDIFF` diffuses every sea-ice tracer under `IF ( SEAICEdiffKhHeff .GT. 0. )` (`seaice_advdiff.F:453`) but
  with `SEAICEdiffKhSItr` (`:455-460`), which is `SEAICEdiffKhAREA` for an AREA-mate tracer (`:425`): such a tracer is
  not diffused when only `SEAICEdiffKhArea > 0`, and SEAICE_DIFFUSION is called with a zero (or the AREA) diffusivity
  when only `SEAICEdiffKhHeff > 0`. Read from the Fortran (lane M4LAB session 4); not exercised by lab_sea (all
  SEAICEdiffKh* = 0). The port refuses SEAICEdiffKh* > 0.
- In the non-ITD build `SEAICE_GROWTH` sets `SItrHEFF(:,:,2)` only inside `IF (.NOT.SEAICE_growMeltByConv)`
  (`seaice_growth.F:1299-1342`, the snapshot at `:1335-1337`); with `SEAICE_growMeltByConv = .TRUE.` it keeps the
  previous time step's value (the ITD build zeroes levels 2-5 first, `:548-555`, and accumulates), and
  `SEAICE_TRACER_PHYS` reads it as HEFFprev/HEFFpost of the first two thermodynamic increments of every HEFF-mate tracer
  (`seaice_tracer_phys.F:100-115`). Read from the Fortran (lane M4LAB session 4); not exercised by lab_sea
  (SEAICE_growMeltByConv = .FALSE., seaice_readparms.F:265). The port reproduces it (SItrHEFF level 2 keeps its value).

## pkg/exf + pkg/monitor: a restarted run prints different `exf_uwind_del2` / `exf_vwind_del2` at its first step
- EXF_MONITOR (`pkg/exf/exf_getforcing.F:380`, `exf_monitor.F:132`) prints the del2 statistic of uwind / vwind, and
  MON_STATS_RL's del2 reads the halo neighbours (`pkg/monitor/mon_stats_rl.F:76-81`). EXF_GETFORCING sets uwind / vwind
  on the interior and does not exchange them; in lab_sea the only exchange is SEAICE_MODEL's
  `EXCH_UV_AGRID_3D_RL( uwind, vwind, .TRUE., 1 )` (`pkg/seaice/seaice_model.F:127`), later in the step. So at every
  step but the first the monitor sees the halos of the previous step's exchange, and at the first step of a run the
  never-exchanged halos: a restart prints different values there although the state is identical.
- Measured with the oracle binary (lab_sea-code-63cdc0b-704fd6b, lane M4LAB session 4, job 27874017): A = lab_sea/input
  with pChkptFreq 18000 s (pickups at iterations 5 and 10), B = startTime 18000 s from A's pickups at 5. A's and B's
  pickups at 10 are byte-identical, but at exf_tsnumber 5 A prints exf_uwind_del2 = 8.0270959439526E-02,
  exf_vwind_del2 = 4.8609272056854E-02 and B 1.6626227332346E-01, 1.5093881355919E-01 (and the cold start at
  iteration 1 prints 1.66257...E-01 too). Output only: no state is affected.
- Port: reproduces both runs literally (test_m4lab_run::test_restart_4_pickup_5_equals_9 asserts B's two lines equal
  the Fortran B's).

## pkg/seaice: SEAICEuseStrImpCpl's no-slip term takes the mask of the other velocity component
- `pkg/seaice/seaice_lsr.F:1710` (SEAICE_LSR_RHSU) sets `hFacM = seaiceMaskV(i,j) - seaiceMaskV(i-1,j)` for the
  explicit `-zetaZ*du/dy` term, whose no-slip part (comment :1719-1721, "u(j-1)=-u(j)") multiplies
  `(uIceC(i,j) + uIceC(i,j-1))` (:1722-1724): the ghost value of u across a boundary in y needs the u mask across j,
  `seaiceMaskU(i,j) - seaiceMaskU(i,j-1)`. `:1866` (SEAICE_LSR_RHSV) mirrors it: `seaiceMaskU(i,j) - seaiceMaskU(i,j-1)`
  for `(vIceC(i,j) + vIceC(i-1,j))`, where `seaiceMaskV(i,j) - seaiceMaskV(i-1,j)` belongs. Both lines repeat the
  hFacM of the main sig12 term just above them (:1685, :1841), which is right there (dv/dx in RHSU, du/dy in RHSV).
  The term came with SEAICEuseStrImpCpl (0fd6b2de1, 2014) and was edited in f33d8d9de / 4a08d54d3 without this change.
- Effect in global_ocean.cs32x15/input.seaice (the only verification run with SEAICEuseStrImpCpl = .TRUE.): none
  measured. On the oracle's Y04_solver_inputs / Y06_lsr fields of iterations 36000-36002 the two masks differ at 450
  (u-equation) and 441 (v-equation) Z points of the interior range, and at every one of them
  `zetaZ*(u(j)+u(j-1))` (resp. `zetaZ*(v(i)+v(i-1))`) is exactly 0. (the velocities next to a coast are masked to
  zero), so the term is 0. either way (lane M4CS32ICE session 2, dev job 27877993). A configuration with nonzero
  velocities on those faces could see a difference.
- Port: literal (seaice_lsr.py, the masks as written).

## pkg/seaice: SEAICE_ADVECTION's cube passes set interiorOnly in pass 1 only (GAD_ADVECTION in all three)
- `pkg/seaice/seaice_advection.F:303-318` resets `interiorOnly = .FALSE.` at every pass and sets it only in pass 1
  (`MOD(nCFace,3).NE.0`); `pkg/generic_advdiff/gad_advection.F:347-363`, the routine it was adapted from, also sets
  `interiorOnly = MOD(nCFace,3).EQ.1` in pass 2 and `.TRUE.` in pass 3. In passes 2 and 3 SEAICE_ADVECTION therefore
  updates the overlap rows (columns) of a cube-face-edge tile with fluxes the generic routine does not apply there.
  The corner fills differ too: FILL_CS_CORNER_TR_RL before the fluxes for `overlapOnly .OR. ipass.EQ.1`
  (:356-359, :570-573; gad_advection.F:389-392 only `overlapOnly`), and the fill after the X (Y) fluxes sits outside
  the "fluxes computed" block (:424-427, :638-641).
- Effect in global_ocean.cs32x15/input.seaice: none on any dumped value. With GAD_ADVECTION's pass flags planted into
  the port, I02_advdiff (HEFF, AREA, HSNOW, ...) is bitwise unchanged at 36000-36002 (lane M4CS32ICE session 2, dev
  job 27878076): the overlap points updated in passes 2/3 are not read by a later flux of the same call, and
  SEAICE_ADVDIFF uses gFld on the interior only (seaice_advdiff.F:314-320). Only the halo of gFld differs. A
  consistency question for the developers rather than a bug.
- Port: literal (seaice_advection.py `_cs_pass_flags`, `_cs_passes`); test_m4cs32ice_seaice.py asserts the blind spot.

## verification/1D_ocean_ice_column: results/output_adm.txt is a checkpoint69e output (stale, low priority)
- `verification/1D_ocean_ice_column/results/output_adm.txt` (last changed in 00c7090dc, 2025-07-07) reports
  `MITgcmUV version: checkpoint69e` (its line 8) and carries `ph-ice B` / `ph-ice C` lines that the code at 63cdc0b no
  longer writes (the WRITE statements are commented out, pkg/seaice/seaice_cost_test.F:85, :226), and the old
  `--> f_ice = 0.691935719996437D+06` / `--> f_gencost = ...` formats (seaice_cost_final.F:79-80 now writes
  '(A,1PE22.14,1PE22.14,1X,1PE9.2)').
- Effect: none on what testreport compares. The fc and the grdchk lines (fc+/fc-, FD) of a 63cdc0b forward build of
  code_ad (lane A's FD oracle, job 27856073) equal the reference's in every printed digit; only the cosmetic lines are
  stale. A refreshed output_adm.txt would make the file match the code that produces it.
- Found by lane M4ADCOL session 1 (against lane A's FD oracle, job 27856073).

## pkg/ecco: ECCO_READPARMS checks mult_obcsn twice and never mult_obcss (retired-parameter test)
- `pkg/ecco/ecco_readparms.F:509` reads `IF ( mult_obcsn.NE.UNSET_RL .OR. mult_obcsn.NE.UNSET_RL .OR. ...`: the
  second term repeats `mult_obcsn`, so a data.ecco that still sets the retired `mult_obcss` (declared and defaulted to
  UNSET_RL at :53, :484, and in the namelist at :136) passes the check silently instead of stopping with the
  "mult_obcs* ... set them in data.ctrl instead" error (:513-519). The intended test is evidently `mult_obcss`.
- Effect: none for any verification input at 63cdc0b (no data.ecco sets mult_obcs*); a typo in an error check.
- Found by lane M4ADCOL session 3 (port of ECCO_READPARMS, mitjax/pkg/ecco/ecco_readparms.py, which ports only the
  empty ecco_cost_nml and raises on any key).

## model/src/the_main_loop.F: the after-loop COST_AVERAGESFIELDS is not guarded by useECCO
- Inside the time loop COST_AVERAGESFIELDS runs under `IF ( useECCO )` (`model/src/the_main_loop.F:663-673`); the call
  after the loop with endTime (:737-742) is under `#ifdef ALLOW_ECCO` only. A build that compiles pkg/ecco but runs
  with useECCO = .FALSE. therefore calls COST_AVERAGESFIELDS -> COST_AVERAGESFLAGS (pkg/cal date arithmetic) once at
  the end, with ECCO.h never initialised by ECCO_READPARMS / ECCO_COST_INIT_FIXED (they return early or are not
  called without useECCO); with every using_gencost .FALSE. it only does the calendar arithmetic, but it would fail
  if pkg/cal were not in use (CAL_GETDATE's "called too early" stop, cal_getdate.F:55-63).
- Effect: none on the verification inputs we run (every ALLOW_ECCO run here has useECCO and useCAL). An
  inconsistency of the two call sites.
- Found by lane M4ADCOL session 3 (mitjax ports the after-loop call inside the useECCO arm of COST_FINAL's driver,
  Model._pkg_final; a useECCO = .FALSE. run of an ALLOW_ECCO build is not exercised).

## pkg/seaice: an AD build never reports a non-converged LSR solve
- `pkg/seaice/seaice_lsr.F:1003-1058` is one `#ifndef ALLOW_AUTODIFF_TAMC` block: the residual prints (only with
  printResidual) and also the `** WARNING ** SEAICE_LSR (it=..., ipass) did not converge` message to
  errorMessageUnit (:1047-1057), which the forward build prints whenever doIterate4u/v is still .TRUE. after the
  linear loop. The comment (:1004-1006) explains the residual part (TAF would recompute it); the warning reads only
  logicals and counters already taped (doIterate4u/v, :793) and needs no recomputation, yet it is hidden too. Every
  code_ad build with the C-grid LSR (lab_sea, offline_exf_seaice, global_ocean.cs32x15) therefore runs its forward
  sweep silently even when SEAICElinearIterMax (= SOLV_MAX_FIXED = 500 under ALLOW_AUTODIFF,
  seaice_readparms.F:887-889) is exhausted. The free-drift residual print (:690-696) is hidden the same way.
- Effect: none in lab_sea/input_ad (measured with an instrumented copy of the code_ad oracle, job 27882556: every
  solve converges in 28-82 sweeps); a diagnostic gap in AD builds.
- Found by lane M4ADLAB session 1 (reading seaice_lsr.F at 63cdc0b; the instrumented oracle run is job 27882556).

## pkg/seaice/seaice_check.F: the areaLossFormula range message says 1..2, the check allows 1..3
- `pkg/seaice/seaice_check.F:102-107` (63cdc0b): `IF ((SEAICE_areaLossFormula.LT.1).OR.(SEAICE_areaLossFormula.GT.3))`
  prints ' SEAICE_areaLossFormula must be between 1 and 2'; formula 3 (predicted melt by the atmosphere,
  seaice_growth.F:1840-1848) is valid and used by lab_sea/input_ad and input_ad.noseaicedyn. A copy of the
  areaGainFormula message (:95-100, 1..2).
- Effect: cosmetic (only the text of the error for a value outside 1..3).
- Found by lane M4ADLAB session 3 while porting formula 3 (G3).

## pkg/ecco/cost_generic.F: the 2-D misfit / weight files of a multi-tile process hold tile (1,1)'s other levels
- `pkg/ecco/cost_generic.F:451-452, :498-499, :508-509` (63cdc0b) call `WRITE_REC_XY_RL( fname3, localdif, ... )`,
  `WRITE_REC_XY_RL( fname3, localdifmeanOut, ... )` and `WRITE_REC_XY_RL( fname3, localweight, ... )` for a 2-D
  gencost (nnzobs = 1), but COST_GENLOOP declares these locals with Nr levels,
  `_RL localdif(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)` (:232-243, localdifmeanOut :215), while WRITE_REC_XY_RL's
  dummy is `field(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)` (pkg/rw/write_rec.F:162). By sequence association the routine
  writes, for tile (bi,bj), the ((bi-1)+(bj-1)*nSx+1)-th xy slab of the local: level 1 of tile (1,1) for tile (1,1)
  (correct), levels 2, 3, ... of tile (1,1) (ECCO_ZERO'd: zeros) for every other tile of the process.
- Evidence: lab_sea/input_ad (one process, nSx = nSy = 2): the code_ad oracle's `misfit_sst.00[12].00[12].data`,
  `misfit_mdt.*` and `weight_mdt.*` are all zeros in tiles 001.002, 002.001, 002.002 while tile 001.001 holds the
  misfit (job27855987-plain / -jdon run directories; the bar files m_sst_month / m_eta_month, written from the 2-D
  gencost_barfld, are correct in every tile). The cost itself is right (ECCO_ADDCOST reads the arrays with nLev = 1);
  only the diagnostic files are wrong. With one tile per process (nSx = nSy = 1) the files are correct.
- mitjax reproduces the Fortran's files (cost_gencost_all.py `_write_xy_or_xyz`, sequence association made explicit).
- Found by lane M4ADLAB session 4 (dev job 27895800: our tiles 2-4 held the misfits, the oracle's zeros).

## pkg/ecco/ecco_toolbox.F: typo in ECCO_OFFSET's message
- `pkg/ecco/ecco_toolbox.F:797` prints 'ecco_offset: # of nonzero constributions to mean of ' ("constributions").
  Cosmetic. Found by lane M4ADLAB session 4 (the line is part of lab_sea/input_ad's STDOUT, reproduced literally).

## verification/lab_sea: TAF's adjoint (output_adm.txt) disagrees with TAF's own TLM, the Tapenade adjoint and FD
- `verification/lab_sea/results/` (all checkpoint69p, built 2026-08-12): for the same grdchk (xx_atemp record 1,
  tile (1,1), (i,j) = (6..10, 8), eps 1e-3) and the same forward (fcref 7.23649445797779E+03 and every fc+ / fc- equal
  in all three files), the TAF adjoint `ADM adjoint_gradient` of output_adm.txt (code_ad + input_ad) is
  1.84490010605889E-04, 1.78821436501969E-04, 2.32024652275522E-04, 2.97982702085000E-04, 3.74871810303031E-04, while
  TAF's own tangent-linear model `TLM tangent-lin_grad` (output_tlm.txt.gz, the same code_ad + input_ad) gives
  1.84507951894972E-04, 1.78840653676620E-04, 2.32137879127948E-04, 2.98245428792074E-04, 3.75155817043441E-04 and
  the Tapenade adjoint (output_tap_adj.txt: code_tap, which differs from code_ad only in AUTODIFF_OPTIONS.h /
  CTRL_OPTIONS.h AD-tool switches and packages.conf, + input_tap, which links input_ad's data files) gives
  1.84507951894957E-04, 1.78840653676567E-04, 2.32137879127976E-04, 2.98245428792293E-04, 3.75155817043462E-04.
  TLM and Tapenade agree to 12-13 digits and with the finite differences (1-FD/grad 0.8e-6 .. 3.6e-6, the FD noise
  at eps 1e-3); TAF's adjoint is off by 9.7e-5, 1.1e-4, 4.9e-4, 8.8e-4, 7.6e-4 relative (its own grdchk prints
  1-FD/AD = -9.4e-5 .. -8.9e-4, RMS 5.7e-4, against 2.6e-6 for TLM and Tapenade). So TAF's adjoint of this run is
  not the transpose of TAF's tangent of the same code.
- Independent check: mitjax's adjoint of lab_sea/input_ad (JAX reverse mode through the literal port; the LSR
  through its executed sweeps as SEAICE_LSR_ADJOINT_ITER tapes them) gives 1.84507951894917E-04,
  1.78840653676585E-04, 2.32137879127995E-04, 2.98245428792268E-04, 3.75155817043323E-04: 12 digits vs the TLM,
  12-13 vs Tapenade, our central FD at eps 1e-2 within 1.1e-7 .. 7.9e-7 (dev jobs 27897240, 27898125; dot test
  6e-14, job 27897171). The MAX/MIN tie and ABS-at-0 conventions of the derivative do not move it (probes, jobs
  27897980-27897983: at most 16 points changed by 4e-14), so the gap is not a kink convention.
- Localisation (TAF's `%MON ad_dynstat` / `ad_seaice` blocks vs mitjax's exact ones, lane M4ADLAB
  session 5, jobs 27898939 and 27899470): every record agrees at tsnumber 4; at tsnumber 3 (after the reverse of the
  last step) adtheta, adsalt,
  adwvel, adhsnow, adhsalt agree in all 16 digits, while aduice, advice, adarea (every digit), adheff (4 digits),
  aduvel, advvel, adeta (1-4 digits) do not -- exactly the inputs of SEAICE_DYNSOLVER / SEAICE_LSR (with
  SEAICEaddSnowMass = .FALSE., input_ad/data.seaice:35, HSNOW is not one). TAF's ad_seaice_aduice is 2-3 orders of
  magnitude larger than the exact adjoint (max 2.417 vs 0.0324 at tsnumber 3, 114 vs 0.056 at tsnumber 2). So the
  departure is in TAF's adjoint of the sea-ice dynamics (seaice_lsr.F with SEAICE_LSR_ADJOINT_ITER, code_ad
  SEAICE_OPTIONS.h:180). Possibly related: output_adm.noseaicedyn.txt (xx_salt, eps 1e-5) shows 1-FD/AD up to 4.1e-5, far above its
  FD noise, while output_adm.noseaice.txt stays at <= 1.8e-7.
- Effect: testreport compares the adjoint gradient of lab_sea only with the stored output_adm.txt, so the gap is
  invisible there; a user who checks the TAF gradient of this setup against FD sees 1e-4..9e-4.
- EXF adjoint monitor (lane M4COSTSHARD session 1, dev job 27908743): TAF's `ad_exf` iwhen = 3 block at tsnumber 3
  (the reverse of the last step's SEAICE_MODEL, before any LSR adjoint) equals mitjax's in every record to 13-16
  digits; the block at tsnumber 0, after the reverse of three steps of TAF's LSR adjoint, agrees in adfu / adfv in no
  digit (O(1) relative), adQnet / adQsw 2-9, adEmPmR 5-7 digits: the same localisation.
- Found by lane M4ADLAB session 5.
- **Reported as MITgcm#1041** (2026-10-05 17:04 UTC, Nikolay's account, his approved text). Its second table first
  presented cs32x15.seaice and offline_exf_seaice.obcs as further cases; both run AD-mode approximations
  (useApproxAdvectionInAdMode; SEAICEuseFREEDRIFTswitchInAd), see the LSR entry above. Title and body edited at
  17:21 UTC with Nikolay's yes (title now "(lab_sea)"; the table has an AD-mode column and an "Edited" note). GitHub
  keeps the first version in the edit history. The edited text calls the cs32x15.seaice attribution "likely"; the
  port's measurement (plan decision 17, revised) can settle it in a follow-up comment (only on Nikolay's say).
- **Follow-up posted** 2026-10-06 07:57 UTC (Nikolay's approval; issuecomment-6011945051): the cs32x15.seaice split
  measured (approximate advection: 7-8 digits vs TAF's ADM; exact: 11-12 digits vs TAF's TLM; TAF's LSR adjoint ~1e-8
  at the check points) and TAF's ad_dynstat_adeta = 0 through the sea-ice tilt. Text: the comment itself on MITgcm#1041.

## Candidates (not yet confirmed)
- pkg/exf/exf_bulkformulae.F:324-333 with :373-375 (solve4Stress = .FALSE., i.e. useAtmWind = .FALSE. without
  ALLOW_BULK_LARGEYEAGER04): ustar = SQRT(wStress/atmrho) and huol = .../(ustar*ustar) divide by zero where wStress = 0
  at a point with atemp /= 0 (calm or masked stress over a forced point); EXF_WIND guards the same root "to please AD
  tools" (exf_wind.F:194-199), this arm does not, and its adjoint at such a point is infinite. Not met in
  global_ocean.cs32x15/input_ad.seaice* (wStress >= 1.2e-3 on every interior point; lane M4ADCS32ICE session 2).
  Low priority; the port keeps the Fortran forward (only the root's derivative is guarded, safe_sqrt).
- lab_sea/input.hb87 (PS, Uav, Vav: 4 digits) and seaice_itd/input.thermo (ice velocities: 6-9 digits) agree poorly with
  their `results/output.txt`, while our plain, dumps-off and dumps-on builds agree bitwise with each other (lane A
  session 11). Possibly stale reference outputs; to be checked with a second compiler/build before reporting.
