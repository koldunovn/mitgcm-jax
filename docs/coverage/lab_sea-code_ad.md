# Coverage: lab_sea (code_ad) — M4 porting worklist

Written by `tools/coverage.py` (mitjax 7c620b3) from the gcov build `lab_sea-code_ad-63cdc0b-dad681f-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 922 files from the link farm, 109 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_ad` | `job27856010` | 5 | 0 / 0 | 509 | yes | identical (965 lines) |
| `input_ad.noseaice` | `job27856010` | 13 | 0 / 0 | 574 | yes | identical (2093 lines) |
| `input_ad.noseaicedyn` | `job27856010` | 13 | 0 / 0 | 504 | yes | identical (2509 lines) |

## lab_sea/input_ad

Run `$MJX_REFERENCE/coverage/lab_sea/input_ad/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/lab_sea/input_ad/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice volume changes | 5 blocks, first 1.000000e+00, last 1.785896e-01, min 1.785896e-01, max 1.000000e+00 | yes |
| sea ice moves | 5 blocks, first 0.000000e+00, last 8.098258e-02, min 0.000000e+00, max 8.098258e-02 | yes |
| sea-ice thermodynamics | seaice_growth (4 calls) | yes |
| LSR momentum solver | seaice_lsr (4 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (4 calls) | yes |
| cost function | global fc = 7236.49445797779; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

(parameter files could not be read by mitjax.config: ValueError: $MJX_UPSTREAM/verification/lab_sea/input_ad/data.err:1: text outside a namelist group: '0.25\n    0.5201    0.2676\n    '; all printed switches are listed)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, doSaltClimRelax, dumpInitAndLast, fluidIsWater, implicitDiffusion, implicitFreeSurface, implicitViscosity, inAdExact, KPP_ghatUseTotalDiffus, KPPwriteState, LimitHblStable, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICEadvArea, SEAICEadvHeff, SEAICEadvSnow, SEAICEupdateOceanStress, SEAICEuseDYNAMICS, SEAICEuseFlooding, SEAICEuseFluxForm, SEAICEuseLSR, SEAICEuseTilt, SEAICEwriteState, snapshot_mdsio, staggerTimeStep, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCDscheme, useCoriolis, useCtrlCostContribution, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useKPPinAdMode, useMaykutSatVapPoly, useMultiDimAdvec, usePW79thermodynamics, useSEAICEinAdMode, usingGregorianCalendar, usingSphericalPolarGrid, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, doResetHFactors, doThetaClimRelax, exactConserv, fluidIsAir, globalFiles, GM_AdvForm, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, implicitIntGravWave, interDiffKr_pCell, interViscAr_pCell, KPPuseDoubleDiff, KPPuseSWfrac3D, linFSConserveTr, momImplVertAdv, monitor_mnc, no_slip_sides, noNegativeEvap, nonHydrostatic, pickup_read_mnc, pickup_write_mnc, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICE_2ndOrderBC, SEAICE_clipVeloctities, SEAICE_doOpenWaterGrowth, SEAICE_doOpenWaterMelt, SEAICE_dump_mnc, SEAICE_growMeltByConv, SEAICE_maskRHS, SEAICE_mcPheeStepFunc, SEAICE_mon_mnc, SEAICE_no_slip, SEAICE_salinityTracer, SEAICE_useMultDimSnow, SEAICEaddSnowMass, SEAICEadvSalt, SEAICEheatConsFix, SEAICEmomAdvection, SEAICEmultiDimAdvection, SEAICErestoreUnderIce, SEAICEscaleSurfStress, SEAICEuseBDF2, SEAICEuseDYNAMICSswitchInAd, SEAICEuseEVP, SEAICEuseFREEDRIFT, SEAICEuseFREEDRIFTswitchInAd, SEAICEuseJFNK, SEAICEuseKrylov, SEAICEuseLSRflex, SEAICEuseMultiTileSolver, SEAICEusePicardAsPrecon, SEAICEuseStrImpCpl, SEAICEuseTEM, snapshot_mnc, stressIsOnCgrid, tempImplVertAdv, tempSOM_Advection, twoDigitYear, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useHB87stressCoupling, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCurvilinearGrid, usingCylindricalGrid, usingJulianCalendar, usingModelCalendar, usingMPI, usingNoLeapYearCal, usingPCoords, vectorInvariantMomentum
- selectors: exf_adjMonSelect=3, monitorSelect=3, pCellMix_select=0, saltAdvScheme=30, saltVertAdvScheme=30, SEAICEadvScheme=2, SEAICEetaZmethod=0, SEAICEselectMetricTerms=2, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempAdvScheme=30, tempVertAdvScheme=30

### Executed routines (509)

**eesupp/src** (78)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 5442 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 6733 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 77 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 56/77 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1504 | 27/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 120 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 4 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1624 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1504 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1504 | 38/90 | 289-324, 346-381 | 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1504 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1504 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 120 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 120 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 120 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 120 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 360 | 11/12 | 60 | 56:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 4 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 12 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_dgrid_3d_rl` | exch_uv_dgrid_3d_rl.F:8 | 4 | 14/32 | 88-177 | 72:1/2, 74:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 497 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 17 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 89 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 84 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 17 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 25641 | 2/2 | - | - |
| `gather_2d_r8` | gather_2d_r8.F:7 | 544 | 14/14 | - | 61:1/2, 63:1/2 |
| `global_max_r8` | global_max.F:97 | 774 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 73 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 144 | 12/12 | - | 154:1/2 |
| `global_sum_singlecpu_rl` | global_sum_singlecpu.F:15 | 544 | 22/22 | - | 98:1/2, 107:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 884 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 3562 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 16114 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 27 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 1305 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1146 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 1019 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 6496 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 383 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 14 | 30/40 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 72 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 171 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 308 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 4361 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 236 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 236 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 236 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 119 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 117 | 4/4 | - | - |
| `ucase` | utils.F:311 | 475 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 11 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 62 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 171 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 262 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 45 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (103)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 736 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 368 | 13/25 | 838, 840, 842, 913-918, 925-929, 955 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2, 951:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 368 | 19/55 | 467, 469, 471, 563-612, 627-632, 639-643, 723 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2, 719:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 368 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 368 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 64 | 26/32 | 164-166, 195-197 | 132:1/2, 180:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 736 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 368 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 368 | 19/19 | - | - |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 16 | 8/8 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 16 | 6/81 | 76, 81, 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 368 | 36/162 | 149, 184, 212-240, 268-610, 636-648 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 16 | 12/16 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 4 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `config_check` | config_check.F:14 | 1 | 96/545 | 92-97, 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 986-989, 992-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 985:1/2, 991:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 443/541 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 533-538, 554-588, 756-762, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 16 | 24/57 | 79, 83, 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 32 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 368 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 368 | 24/32 | 79-84, 123-125 | 61:1/2, 76:1/2, 121:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 4 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 4 | 10/16 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 4 | 112/151 | 257-264, 467-471, 548, 557, 759-784, 797-798, 830-833, 869-874, 882, 906, 934, 962, 1043, 1054, 1116, 1119, 1123, 1127 | 248:1/2, 251:1/2, 256:1/2, 421:1/2, 450:1/2, 546:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 734:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 867:1/2, 878:1/2, 896:1/2, 903:1/2, 932:1/2, 951:1/2, 953:1/2, 1030:1/2, 1032:1/2, 1049:1/2, 1051:1/2, 1096:1/2, 1102:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1126:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 4 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 5 | 14/21 | 97-110, 242 | 96:1/2, 116:1/2, 144:1/2, 187:1/2, 193:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 4 | 22/26 | 82, 84, 115-117 | 66:1/2, 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 4 | 58/84 | 337-340, 358, 533, 684-700, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 615:1/2, 682:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 4 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 4 | 42/69 | 75, 151-153, 250, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 247:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_alpha` | find_alpha.F:14 | 368 | 29/76 | 77-79, 85-99, 222-337 | 75:1/2, 83:1/2, 113:1/2 |
| `find_beta` | find_alpha.F:347 | 368 | 27/74 | 410-412, 418-430, 539-641 | 408:1/2, 416:1/2, 444:1/2 |
| `find_bulkmod` | find_rho.F:412 | 2620 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 1884 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 67 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 2620 | 16/16 | - | - |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 4 | 12/21 | 64, 93-104, 245-252 | 63:1/2, 75:1/2, 242:1/2 |
| `forward_step` | forward_step.F:70 | 4 | 80/102 | 508-512, 730-734, 767-769, 813, 980, 989-991, 1109-1111, 1176, 1201-1202 | 424:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 812:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1169:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 368 | 20/22 | 61, 74 | 59:1/2, 72:1/2 |
| `impldiff` | impldiff.F:7 | 96 | 65/102 | 286-298, 304-382 | 199:1/2, 219:1/2, 284:1/2, 303:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/71 | 57-63, 70-79, 106-112, 121-180, 191, 196-201 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 195:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/76 | 63-66, 94-98, 107-114, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 106:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 57/87 | 52, 60, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/121 | 131, 134-144, 192, 197-202 | 130:1/2, 132:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 171:1/2, 175:1/2, 185:1/2, 189:1/2, 196:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 177/193 | 210-212, 247-261, 435, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 35/58 | 146-179, 191-195, 219-227 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 206:1/2, 217:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 404/881 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 584, 586-589, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1351-1355, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 585:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1116:1/2, 1117:1/2, 1118:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 20/22 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 26/45 | 68-73, 100, 109-124, 130 | 65:1/2, 67:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_spherical_polar_grid` | ini_spherical_polar_grid.F:8 | 1 | 75/85 | 258-263, 277-280 | 119:1/2, 224:1/2, 237:1/2, 257:1/2, 276:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 27/54 | 69-74, 101, 110-125, 131-138, 151 | 66:1/2, 68:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 5 | 14/65 | 92-185, 211-221, 279-282, 329-331 | 90:1/2, 207:1/2, 277:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 460 | 17/28 | 95-117 | 93:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 4 | 26/34 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 27/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161, 173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 4 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 4 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 110/124 | 242-261 | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 64/98 | 207, 212, 280, 289, 304, 321, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 252:1/2, 260:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 339:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 428:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 44/52 | 170-176, 521-523, 675-677 | 144:1/2, 158:1/2, 160:1/2, 167:1/2, 193:1/2, 200:1/2, 202:1/2, 209:1/2, 211:1/2, 249:1/2, 251:1/2, 317:1/2, 319:1/2, 327:1/2, 329:1/2, 347:1/2, 349:1/2, 358:1/2, 365:1/2, 367:1/2, 374:1/2, 379:1/2, 510:1/2, 512:1/2, 519:1/2, 651:1/2, 655:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 27/31 | 169, 174, 445, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 204:1/2, 206:1/2, 251:1/2, 253:1/2, 270:1/2, 285:1/2, 291:1/2, 293:1/2, 435:1/2, 444:1/2, 601:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 24 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 17/17 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 3 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 14/15 | 225 | 103:1/2, 124:1/2, 184:1/2, 223:1/2, 237:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 2620 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 14 | 50/64 | 67-70, 78, 100-101, 168-174 | 66:1/2, 77:1/2, 99:1/2, 111:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 16 | 58/70 | 167, 169, 192, 326, 367-369, 386-390, 454, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 444:1/2, 448:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 96/166 | 56-57, 60-63, 67, 71, 76, 109-115, 120-126, 171-175, 186-189, 192-195, 202, 208, 214, 220, 226, 232, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 4 | 59/94 | 143-147, 152-157, 164, 170-176, 218-224, 265, 269, 319, 327, 342-344, 355-366, 371 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 401 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 16 | 58/70 | 169, 171, 194, 328, 369-371, 388-392, 456, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 446:1/2, 450:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 53/53 | - | 292:1/2, 382:1/2, 472:1/2, 664:1/2, 666:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/45 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 4 | 44/52 | 158, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 368 | 58/94 | 118-121, 140-143, 189-190, 220-223, 328-332, 392-410, 413-414, 417-418, 422-423 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 229:1/2, 321:1/2, 391:1/2, 412:1/2, 416:1/2, 421:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 32 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 4 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/20 | - | 55:1/2, 78:1/2, 88:1/2, 106:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 47/110 | 74, 98-101, 106-111, 117, 119-121, 133-140, 164-208, 217-219 | 73:1/2, 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2, 162:1/2, 214:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 52/100 | 253-257, 261-265, 288-290, 335-338, 365-371, 391-438 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2, 390:1/2 |
| `write_state` | write_state.F:11 | 5 | 19/46 | 87, 126, 177-209 | 84:1/2, 95:1/2, 123:1/2, 175:1/2 |

**pkg/autodiff** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 85 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 79 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 6 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 41 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 38 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 3 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 3 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 4 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 4 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 18/145 | 69-83, 123-403 | 66:1/2, 68:1/2, 121:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 81/148 | 63-68, 127-132, 157, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 6 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 6 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 14 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 5 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 4 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 4 | 2/2 | - | - |

**pkg/cal** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_addtime` | cal_addtime.F:3 | 209 | 47/107 | 66-75, 79-81, 89-106, 114-116, 127-140, 169-171, 179, 181-186, 192-199 | 65:1/2, 78:1/2, 88:1/2, 110:1/2, 124:1/2, 167:1/2, 168:1/2, 177:1/2, 180:1/2, 191:1/2, 200:2/2, 214:1/2, 218:1/2 |
| `cal_checkdate` | cal_checkdate.F:3 | 47 | 22/54 | 61-63, 65-70, 76-81, 89-93, 96, 100-102, 106-108, 111, 123-126, 129-132, 135-138 | 59:1/2, 64:1/2, 73:1/2, 88:1/2, 94:1/2, 98:1/2, 103:1/2, 109:1/2, 116:1/2, 122:1/2, 128:1/2, 134:1/2 |
| `cal_compdates` | cal_compdates.F:3 | 10 | 5/5 | - | - |
| `cal_convdate` | cal_convdate.F:3 | 520 | 21/37 | 51-57, 66-68, 71-73, 87-89 | 50:1/2, 65:1/2, 70:1/2, 82:1/2 |
| `cal_copydate` | cal_copydate.F:3 | 97 | 6/6 | - | - |
| `cal_fulldate` | cal_fulldate.F:3 | 47 | 15/37 | 59-65, 71-74, 87-93, 98-101 | 58:1/2, 70:1/2, 77:1/2, 84:1/2 |
| `cal_getdate` | cal_getdate.F:3 | 686 | 16/23 | 57-63 | 55:1/2 |
| `cal_init_fixed` | cal_init_fixed.F:8 | 1 | 6/6 | - | 28:1/2, 42:1/2 |
| `cal_intdays` | cal_intdays.F:3 | 2 | 9/10 | 58 | 55:1/2 |
| `cal_intmonths` | cal_intmonths.F:3 | 2 | 9/11 | 62, 69 | 59:1/2, 67:1/2 |
| `cal_intyears` | cal_intyears.F:3 | 2 | 4/5 | 49 | 47:1/2 |
| `cal_isleap` | cal_isleap.F:3 | 19319 | 9/21 | 41-47, 60-67 | 40:1/2, 50:1/2 |
| `cal_numints` | cal_numints.F:3 | 3 | 7/10 | 61-63 | 53:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 19/23 | 70-76 | 65:1/2, 68:1/2, 79:1/2, 114:1/2 |
| `cal_set` | cal_set.F:3 | 1 | 60/93 | 104-114, 163-184, 202-204, 207-209, 212-214 | 82:1/2, 99:1/2, 119:1/2, 160:1/2, 201:1/2, 206:1/2, 211:1/2 |
| `cal_subdates` | cal_subdates.F:3 | 2 | 6/24 | 48-57, 65-69, 77-79 | 47:1/2, 60:1/2, 63:1/2 |
| `cal_summary` | cal_summary.F:3 | 1 | 43/43 | - | 49:1/2 |
| `cal_time2dump` | cal_time2dump.F:3 | 8 | 3/15 | 36-51 | 31:1/2 |
| `cal_timeinterval` | cal_timeinterval.F:3 | 221 | 15/36 | 60-66, 73-92 | 57:1/2, 59:1/2 |
| `cal_timepassed` | cal_timepassed.F:3 | 213 | 52/75 | 67-76, 98, 120-127, 148-149, 182-184 | 66:1/2, 87:1/2, 97:1/2, 119:1/2, 147:1/2 |
| `cal_toseconds` | cal_toseconds.F:3 | 167 | 15/26 | 57-63, 69, 92-94 | 56:1/2, 67:1/2, 73:1/2 |

**pkg/cd_code** (4)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cd_code_ini_vars` | cd_code_ini_vars.F:3 | 1 | 15/16 | 53 | 52:1/2 |
| `cd_code_init_fixed` | cd_code_init_fixed.F:3 | 1 | 3/23 | 25-67 | 22:1/2 |
| `cd_code_scheme` | cd_code_scheme.F:7 | 368 | 47/49 | 82-83 | 78:1/2 |
| `cd_code_write_pickup` | cd_code_write_pickup.F:8 | 1 | 12/26 | 47-66 | 70:1/2, 85:1/2 |

**pkg/cost** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 18/18 | - | - |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 12/12 | - | 44:1/2, 48:1/2, 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 47/47 | - | 82:1/2, 97:1/2, 102:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 21/21 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 4 | 2/2 | - | - |

**pkg/ctrl** (33)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 4 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 20 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 2 | 11/13 | 51, 54 | 41:1/2, 50:1/2, 53:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 31/67 | 102-110, 124-128, 139-143, 182-186, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 121:1/2, 136:1/2, 179:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 32/37 | 74, 83-84, 113, 140 | 65:1/2, 73:1/2, 78:1/2, 82:1/2, 112:1/2, 120:1/2, 139:1/2, 147:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 83/83 | - | 66:1/2, 131:1/2, 146:1/2, 161:1/2, 176:1/2, 186:1/2, 200:1/2, 214:1/2 |
| `ctrl_cost_gen2d` | ctrl_cost_gen.F:12 | 11 | 41/42 | 158 | 121:1/2, 157:1/2, 162:1/2 |
| `ctrl_cost_gen3d` | ctrl_cost_gen.F:187 | 2 | 41/42 | 322 | 283:1/2, 319:1/2, 326:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 71 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 36 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 36 | 35/66 | 99, 104-108, 145, 174-197, 206-223 | 79:1/2, 92:1/2, 100:1/2, 144:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 67 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 4 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 31 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 64/73 | 272-273, 283, 285, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 18 | 21/36 | 72-76, 87-88, 90-91, 106-112, 118-124 | 86:1/2, 89:1/2, 93:1/2, 104:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 168:1/2, 174:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 4 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr2d` | ctrl_map_genarr.F:13 | 2 | 42/57 | 91-93, 97-99, 102, 108-109, 153-160, 168-170 | 71:1/2, 90:1/2, 95:1/2, 101:1/2, 104:1/2, 145:1/2, 149:1/2, 152:1/2, 167:1/2, 199:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 2 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 4 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 42/49 | 157, 159, 161, 201, 360, 362, 364 | 128:1/2, 154:1/2, 156:1/2, 158:1/2, 160:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 71/127 | 151-153, 157-159, 162, 183-190, 254-261, 276-393, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 253:1/2, 272:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 217/245 | 181-186, 565, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 539:1/2, 540:2/4, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 31 | 15/16 | 60 | 42:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 29 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 2 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 102/140 | 153-155, 169-171, 184-187, 218-222, 237-241, 267-292, 304-306 | 136:1/2, 146:1/2, 162:1/2, 178:1/2, 217:1/2, 236:1/2, 255:1/2, 259:1/4, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 9 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 1476 | 10/16 | 43-45, 55-57 | 41:1/2, 42:2/2, 50:1/2, 52:1/2, 53:2/2 |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/370 | 123-653 | 109:1/2, 111:1/2 |

**pkg/down_slope** (6)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `dwnslp_apply` | dwnslp_apply.F:6 | 32 | 30/49 | 104-115, 120-122, 128-132, 170-175, 183-184, 191 | 95:1/2, 100:1/2, 103:1/2, 119:1/2, 127:1/2, 147:4/6, 152:4/6, 169:1/2, 182:1/2, 190:1/2 |
| `dwnslp_calc_flow` | dwnslp_calc_flow.F:6 | 16 | 22/48 | 79-83, 123-128, 139-157, 163-164 | 74:1/2, 77:1/2, 109:4/8, 122:1/2, 138:1/2, 162:1/2 |
| `dwnslp_calc_rho` | dwnslp_calc_rho.F:6 | 368 | 8/8 | - | - |
| `dwnslp_init_fixed` | dwnslp_init_fixed.F:6 | 1 | 94/138 | 75-121, 197-202, 216-218, 248-266, 313-314, 337 | 71:1/2, 140:1/2, 149:1/2, 168:1/2, 177:1/2, 195:1/2, 215:1/2, 230:4/6, 235:1/2, 280:1/2, 283:1/2, 286:1/2, 299:1/2, 312:1/2, 328:1/2, 336:1/2 |
| `dwnslp_init_varia` | dwnslp_init_varia.F:6 | 1 | 7/7 | - | - |
| `dwnslp_readparms` | dwnslp_readparms.F:6 | 1 | 23/32 | 44-50, 93-95, 99-101 | 42:1/2, 53:1/2, 91:1/2, 97:1/2, 105:1/2 |

**pkg/ecco** (39)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_averagesfields` | cost_averagesfields.F:3 | 5 | 14/14 | - | 74:1/2, 127:1/2 |
| `cost_averagesflags` | cost_averagesflags.F:4 | 5 | 74/95 | 153, 174-183, 207, 228-239, 263, 285-289 | 152:1/2, 170:1/2, 206:1/2, 224:1/2, 262:1/2, 280:1/2 |
| `cost_averagesgeneric` | cost_averagesgeneric.F:3 | 20 | 38/50 | 113-134, 209 | 111:1/2, 193:1/2 |
| `cost_averagesinit` | cost_averagesinit.F:3 | 1 | 20/20 | - | - |
| `cost_gencal` | cost_gencal.F:7 | 8 | 17/35 | 75-77, 80-98 | 74:1/2, 78:1/2, 112:1/2 |
| `cost_gencost_all` | cost_gencost_all.F:3 | 1 | 21/23 | 54-56 | 53:1/2, 93:1/2, 94:1/2, 95:1/2, 96:1/2 |
| `cost_gencost_assignperiod` | cost_gencost_assignperiod.F:3 | 5 | 12/37 | 58-62, 70-93 | 56:1/2, 63:1/2 |
| `cost_gencost_boxmean` | cost_gencost_boxmean.F:3 | 1 | 4/45 | 76-169 | 71:1/2 |
| `cost_gencost_bpv4` | cost_gencost_bpv4.F:3 | 1 | 7/104 | 95-361 | 80:1/2, 84:1/2 |
| `cost_gencost_customize` | cost_gencost_customize.F:20 | 5 | 61/97 | 128, 131, 137, 140, 143, 147, 166, 169, 172, 175, 179, 182, 185, 190, 193, 197, 210, 213, 217, 220, 223, 247-250, 253-256, 259-261, 264-266, 269-271 | 126:1/2, 129:1/2, 135:1/2, 138:1/2, 141:1/2, 144:1/2, 164:1/2, 167:1/2, 170:1/2, 173:1/2, 177:1/2, 180:1/2, 183:1/2, 188:1/2, 191:1/2, 195:1/2, 207:1/2, 211:1/2, 214:1/2, 218:1/2, 221:1/2, 246:1/2, 252:1/2, 258:1/2, 263:1/2, 268:1/2 |
| `cost_gencost_glbmean` | cost_gencost_glbmean.F:3 | 1 | 2/2 | - | - |
| `cost_gencost_moc` | cost_gencost_moc.F:3 | 1 | 14/89 | 95-270 | 92:1/2 |
| `cost_generic` | cost_generic.F:12 | 4 | 18/18 | - | 101:1/2 |
| `cost_genloop` | cost_generic.F:147 | 4 | 99/125 | 294-295, 298-299, 302, 321-342, 407, 458-464, 501, 511 | 285:1/2, 286:1/2, 287:1/2, 293:1/2, 297:1/2, 301:1/2, 312:1/2, 362:1/2, 370:1/2, 375:1/2, 393:1/2, 406:1/2, 448:1/2, 457:1/2, 495:1/2, 498:1/2, 500:1/2, 505:1/2, 508:1/2, 510:1/2 |
| `cost_genread` | cost_genread.F:7 | 4 | 7/27 | 64-94 | 108:1/2 |
| `ecco_addcost` | ecco_toolbox.F:238 | 4 | 18/20 | 276, 289 | 275:1/2, 286:1/2 |
| `ecco_addmask` | ecco_toolbox.F:414 | 1 | 13/14 | 445 | 444:1/2 |
| `ecco_check` | ecco_check.F:10 | 1 | 59/300 | 248-252, 261-266, 269-274, 277-282, 285-290, 297-301, 320-325, 330-335, 347, 352-358, 365-366, 369-370, 373-374, 380, 382, 384, 388-393, 397-402, 413-432, 440-534, 543-679, 698-703, 715-748, 759-763 | 50:1/2, 247:1/2, 260:1/2, 268:1/2, 276:1/2, 284:1/2, 296:1/2, 305:1/2, 318:1/2, 328:1/2, 344:1/2, 348:1/2, 351:1/2, 364:1/2, 368:1/2, 372:1/2, 376:1/2, 379:1/2, 381:1/2, 383:1/2, 386:1/2, 395:1/2, 412:1/2, 438:1/2, 540:1/2, 688:1/2, 696:1/2, 714:1/2, 755:1/2, 758:1/2 |
| `ecco_check_files` | ecco_check.F:785 | 4 | 13/36 | 836-841, 852-874, 883-888 | 833:1/2, 845:1/2, 847:1/2, 851:1/2, 879:1/2, 880:1/2, 897:1/2 |
| `ecco_cost_final` | ecco_cost_final.F:6 | 1 | 34/36 | 139-141 | 111:1/2 |
| `ecco_cost_init_barfiles` | ecco_cost_init_barfiles.F:7 | 1 | 28/28 | - | 109:1/2 |
| `ecco_cost_init_fixed` | ecco_cost_init_fixed.F:7 | 1 | 36/48 | 132-133, 138-152, 171 | 83:1/2, 130:1/2, 134:1/2, 168:1/2, 238:1/2 |
| `ecco_cost_init_varia` | ecco_cost_init_varia.F:12 | 1 | 14/14 | - | 68:1/2 |
| `ecco_cp` | ecco_toolbox.F:138 | 8 | 11/12 | 166 | 165:1/2 |
| `ecco_diffmsk` | ecco_toolbox.F:74 | 4 | 14/15 | 109 | 108:1/2 |
| `ecco_divfield` | ecco_toolbox.F:523 | 8 | 11/12 | 547 | 546:1/2 |
| `ecco_init_fixed` | ecco_init_fixed.F:8 | 1 | 5/6 | 37 | 31:1/2, 36:1/2 |
| `ecco_init_varia` | ecco_init_varia.F:3 | 1 | 5/5 | - | - |
| `ecco_maskmindepth` | ecco_toolbox.F:669 | 1 | 11/12 | 696 | 695:1/2 |
| `ecco_mult` | ecco_toolbox.F:573 | 4 | 10/11 | 599 | 598:1/2 |
| `ecco_offset` | ecco_toolbox.F:720 | 1 | 34/36 | 754, 791 | 775:1/2, 795:1/2, 811:1/2 |
| `ecco_phys` | ecco_phys.F:17 | 5 | 83/126 | 292-294, 365-389, 411-412, 425-435, 450-451, 454-455, 457-458, 492-499, 555-557, 567, 577-598, 620-628 | 112:1/2, 291:1/2, 364:1/2, 406:1/2, 417:1/2, 449:1/2, 452:1/2, 456:1/2, 491:1/2, 548:1/2, 563:1/2, 572:1/2, 618:1/2 |
| `ecco_readbar` | ecco_toolbox.F:817 | 4 | 16/18 | 875-876 | 869:1/2 |
| `ecco_readparms` | ecco_readparms.F:3 | 1 | 371/497 | 204-209, 513-519, 522-525, 528-531, 534-537, 541-548, 552-558, 561-568, 692-693, 700-710, 715-721, 725-731, 738-739, 774-822, 834-836, 839-841, 844-846, 855, 868-875, 880-887, 891-923 | 202:1/2, 212:1/2, 511:1/2, 521:1/2, 527:1/2, 533:1/2, 539:1/2, 550:1/2, 560:1/2, 576:1/2, 597:1/2, 690:1/2, 698:1/2, 714:1/2, 724:1/2, 736:1/2, 769:1/2, 833:1/2, 838:1/2, 843:1/2, 852:1/2, 854:1/2, 865:1/2, 878:1/2, 890:1/2, 934:1/2 |
| `ecco_readwei` | ecco_toolbox.F:912 | 4 | 14/16 | 960, 967 | 959:1/2, 965:1/2 |
| `ecco_summary` | ecco_summary.F:3 | 1 | 65/89 | 76-82, 90-92, 98-101, 111-114, 133-135, 138-140 | 69:1/2, 75:1/2, 88:1/2, 97:1/2, 110:1/2, 132:1/2, 137:1/2 |
| `ecco_write_pickup` | ecco_write_pickup.F:6 | 1 | 3/3 | - | - |
| `ecco_zero` | ecco_toolbox.F:30 | 341 | 8/8 | - | - |
| `stergloh_output` | stergloh_output.F:8 | 5 | 2/2 | - | - |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 12 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 4 | 90/101 | 236, 241-242, 325-333, 403, 502-506 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 30/140 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 333-342, 353-357, 399-402, 550-569, 575-578, 616-621, 635-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 625:1/2, 634:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 28/76 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 4 | 3/26 | 41-79 | 39:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 16 | 12/22 | 66-78 | 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 8 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 4 | 13/14 | 89 | 68:1/2, 88:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 8 | 12/55 | 65-76, 86-92, 106-141 | 80:1/2, 85:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 32 | 18/97 | 101-109, 121-132, 139, 155-190, 206-259 | 97:1/2, 112:1/2, 119:1/2, 138:1/2, 196:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 4 | 60/76 | 97, 147-152, 315-320, 507, 512, 525, 540 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 505:1/2, 510:1/2, 523:1/2, 538:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 4 | 45/53 | 175-181, 202-203, 329, 388 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 4 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 16 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 89/133 | 64-65, 149-155, 159-179, 185-191, 195-201, 207-213, 262-268, 282-289, 309-315, 322-329, 359-366, 387-394, 474-481, 489, 590-593, 617-619 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 137:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 320:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 385:1/2, 472:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 19 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 41/49 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 4 | 69/95 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 4 | 69/86 | 74, 79-89, 114-116, 164, 186, 192, 204, 210, 222 | 64:1/2, 67:1/2, 71:1/2, 78:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 202:1/2, 208:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 4 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 76 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 4 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 170/208 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 402-408, 487-493, 500-501, 548-555, 571-581, 585-586, 625-626, 636-642, 684-690, 714-720, 761-767, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 633:1/2, 653:1/2, 666:1/2, 681:1/2, 711:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 8 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 4 | 37/83 | 104-115, 138-140, 158-240 | 79:1/2, 102:1/2, 126:1/2, 133:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 32 | 126/221 | 199-202, 213-219, 253-266, 279-282, 335-337, 351-364, 394, 414, 418, 423-446, 461, 475-539, 615, 635, 639, 644-667, 682, 696-761, 824-827, 844-845, 848-849, 866, 991, 995, 1000-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 419:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 640:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 736 | 82/206 | 181-184, 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 614, 620, 646-647, 684-685, 696-700, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 669:1/2, 695:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diff_x` | gad_diff_x.F:7 | 48 | 6/6 | - | - |
| `gad_diff_y` | gad_diff_y.F:7 | 48 | 7/7 | - | - |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 704 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 736 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 736 | 17/17 | - | 82:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 96/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/gmredi** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 32 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 16 | 144/218 | 145-147, 166-209, 301-303, 529, 620-627, 689-691, 713-716, 768, 816-837, 851-886, 908-911, 965, 1013-1034, 1048-1083, 1122 | 144:1/2, 164:1/2, 291:1/2, 526:1/2, 610:1/2, 688:1/2, 702:1/2, 765:1/2, 815:1/2, 850:1/2, 897:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 4 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 19/27 | 63-64, 67-68, 82, 86, 142, 148 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 141:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 4 | 3/25 | 50-82 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 107/145 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 16 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 736 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 1088 | 52/87 | 176, 234, 397, 474, 479, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 473:1/2, 478:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 736 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 736 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 49/71 | 96, 121-130, 190-199, 218-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 113:1/2, 115:1/2, 116:1/2, 189:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 87-89, 116-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 54/118 | 116-126, 134, 163, 192-202, 278-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 185:1/2, 186:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `bldepth` | kpp_routines.F:309 | 16 | 84/117 | 494-495, 523-524, 534-549, 640-641, 691-697, 716-717, 730-744, 830-833, 853-854, 867-881 | 486:1/2, 491:1/2, 532:1/2, 639:1/2, 685:1/2, 690:1/2, 728:1/2, 781:1/2, 824:1/2, 829:1/2, 865:1/2 |
| `blmix` | kpp_routines.F:1395 | 16 | 72/72 | - | - |
| `enhance` | kpp_routines.F:1696 | 16 | 11/11 | - | 1741:1/2 |
| `kpp_calc` | kpp_calc.F:19 | 16 | 74/88 | 496, 537, 644-649, 682-701 | 223:1/2, 495:1/2, 528:1/2, 636:1/2, 641:1/2, 680:1/2 |
| `kpp_calc_diff_s` | kpp_calc_diff_s.F:3 | 16 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_diff_t` | kpp_calc_diff_t.F:3 | 16 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_visc` | kpp_calc_visc.F:3 | 368 | 8/8 | - | - |
| `kpp_check` | kpp_check.F:3 | 1 | 53/62 | 130-132, 137-139, 142-144 | 28:1/2, 128:1/2, 136:1/2, 141:1/2 |
| `kpp_do_exch` | kpp_do_exch.F:6 | 4 | 3/3 | - | - |
| `kpp_forcing_surf` | kpp_forcing_surf.F:10 | 16 | 47/54 | 263-272, 282-285, 508 | 233:1/2, 247:1/2, 281:1/2, 507:1/2 |
| `kpp_init_fixed` | kpp_init_fixed.F:6 | 1 | 35/64 | 48-111, 188 | 45:1/2, 116:1/2, 162:1/2, 187:1/2 |
| `kpp_init_varia` | kpp_init_varia.F:7 | 1 | 17/17 | - | - |
| `kpp_output` | kpp_output.F:11 | 5 | 8/50 | 118-181, 192-227 | 105:1/2, 114:1/2, 191:1/2 |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 73/112 | 61-66, 171-176, 196-199, 202-205, 208-211, 214-217, 220-226, 230-238 | 59:1/2, 69:1/2, 169:1/2, 195:1/2, 201:1/2, 207:1/2, 213:1/2, 219:1/2, 229:1/2 |
| `kpp_transport_s` | kpp_transport_s.F:9 | 352 | 9/11 | 75, 89 | 73:1/2, 86:1/2 |
| `kpp_transport_t` | kpp_transport_t.F:6 | 352 | 8/9 | 72 | 69:1/2 |
| `kppmix` | kpp_routines.F:28 | 16 | 17/17 | - | - |
| `ri_iwmix` | kpp_routines.F:1032 | 16 | 39/42 | 1199-1201 | 1198:1/2 |
| `smooth_horiz` | kpp_routines.F:1311 | 352 | 15/15 | - | - |
| `statekpp` | kpp_routines.F:1766 | 16 | 30/32 | 1950-1951 | 1949:1/2 |
| `wscale` | kpp_routines.F:923 | 752 | 28/28 | - | - |

**pkg/mdsio** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 51 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 26 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 238 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 134 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 487, 535-538, 543, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 530:1/2, 540:1/2, 541:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 185 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 691 | 52/64 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mnc** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mnc_readparms` | mnc_readparms.F:13 | 1 | 5/85 | 68-212 | 55:1/2, 57:1/2 |

**pkg/mom_common** (5)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 368 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 368 | 12/26 | 62-66, 77-84, 90-97, 115-137 | 61:1/2, 70:1/2, 88:1/2, 101:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 38/41 | 51-52, 230 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 368 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 368 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |

**pkg/mom_fluxform** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_rtrans` | mom_calc_rtrans.F:6 | 384 | 11/11 | - | - |
| `mom_fluxform` | mom_fluxform.F:42 | 368 | 161/276 | 265, 276, 331-363, 457, 514-523, 547-594, 607, 622-623, 648-651, 663-666, 690-693, 738-748, 764-767, 773-780, 843-890, 902, 917-918, 943-946, 958-961, 985-988, 1033-1042, 1058-1061, 1067-1074, 1092-1106, 1113-1124, 1142-1146 | 257:1/2, 264:1/2, 270:1/2, 330:1/2, 422:1/2, 444:1/2, 453:1/2, 476:1/2, 510:1/2, 512:1/2, 546:1/2, 601:1/2, 605:1/2, 621:1/2, 647:1/2, 656:1/2, 671:1/2, 689:1/2, 735:1/2, 752:1/2, 753:1/2, 762:1/2, 772:1/2, 789:1/2, 822:1/2, 842:1/2, 897:1/2, 900:1/2, 916:1/2, 942:1/2, 951:1/2, 966:1/2, 984:1/2, 1030:1/2, 1046:1/2, 1047:1/2, 1056:1/2, 1066:1/2, 1082:1/2, 1112:1/2, 1141:1/2 |
| `mom_u_adv_uu` | mom_u_adv_uu.F:7 | 368 | 5/5 | - | - |
| `mom_u_adv_vu` | mom_u_adv_vu.F:7 | 368 | 6/9 | 65-74 | 46:1/2 |
| `mom_u_adv_wu` | mom_u_adv_wu.F:7 | 384 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_u_metric_sphere` | mom_u_metric_sphere.F:7 | 368 | 6/9 | 59-73 | 46:1/2 |
| `mom_u_xviscflux` | mom_u_xviscflux.F:10 | 368 | 5/5 | - | - |
| `mom_u_yviscflux` | mom_u_yviscflux.F:10 | 368 | 5/5 | - | - |
| `mom_v_adv_uv` | mom_v_adv_uv.F:7 | 368 | 5/5 | - | - |
| `mom_v_adv_vv` | mom_v_adv_vv.F:7 | 368 | 5/5 | - | - |
| `mom_v_adv_wv` | mom_v_adv_wv.F:7 | 384 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_v_metric_sphere` | mom_v_metric_sphere.F:7 | 368 | 6/9 | 60-76 | 44:1/2 |
| `mom_v_xviscflux` | mom_v_xviscflux.F:10 | 368 | 5/5 | - | - |
| `mom_v_yviscflux` | mom_v_yviscflux.F:10 | 368 | 5/5 | - | - |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 10 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 5 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 5 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 4 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 16 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 124 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 25 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 5 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 952 | 52/57 | 213-219 | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2, 211:1/2 |
| `mon_out_i` | mon_out.F:8 | 14 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 938 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 56 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 5 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 5 | 39/50 | 95, 123-132, 178-179, 196-198 | 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 5 | 74/127 | 145-237, 244-260, 263-278 | 143:1/2, 242:1/2, 243:1/2, 262:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 124 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 25 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 5 | 65/76 | 57, 62-72, 119-120 | 48:1/2, 50:1/2, 54:1/2, 61:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 38 | 7/7 | - | - |
| `read_rec_lev_rl` | read_rec.F:445 | 8 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 17 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 22 | 16/17 | 38 | 37:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 83 | 7/7 | - | - |
| `write_rec_xy_rl` | write_rec.F:145 | 3 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 2 | 7/7 | - | - |

**pkg/salt_plume** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `salt_plume_readparms` | salt_plume_readparms.F:6 | 1 | 5/32 | 56-124 | 46:1/2, 48:1/2 |

**pkg/seaice** (38)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `advect` | advect.F:9 | 12 | 30/63 | 105-125, 164-226 | 104:1/2, 160:1/2 |
| `seaice_advdiff` | seaice_advdiff.F:10 | 4 | 28/74 | 138-401, 642-652 | 135:1/2, 579:1/2, 584:1/2, 598:1/2, 603:1/2, 617:1/2, 622:1/2, 638:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 16 | 6/6 | - | - |
| `seaice_calc_ice_strength` | seaice_calc_ice_strength.F:9 | 16 | 15/19 | 107-108, 111-112 | 105:1/2, 109:1/2 |
| `seaice_calc_strainrates` | seaice_calc_strainrates.F:11 | 8 | 32/39 | 159-186 | 79:1/2, 158:1/2 |
| `seaice_calc_viscosities` | seaice_calc_viscosities.F:8 | 8 | 63/67 | 126-135 | 95:1/2, 98:1/2, 117:1/2, 467:1/2 |
| `seaice_check` | seaice_check.F:12 | 1 | 91/489 | 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 231-237, 243-249, 253-259, 371-404, 434-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 567-569, 700-702, 725-730, 733-741, 744-752, 789-794, 819-826, 836-845, 852-869, 889-902, 907-916, 923-925, 935-940, 947-952, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1119-1124, 1130-1135, 1141-1144, 1147-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1205-1209, 1216-1221, 1228-1231, 1234-1239, 1242-1247, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 230:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 433:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 788:1/2, 798:1/2, 818:1/2, 828:1/2, 850:1/2, 888:1/2, 904:1/2, 921:1/2, 933:1/2, 945:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1140:1/2, 1146:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1204:1/2, 1214:1/2, 1226:1/2, 1233:1/2, 1241:1/2, 1300:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 4 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 2/2 | - | - |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 2/2 | - | - |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 4 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 4 | 2/2 | - | - |
| `seaice_diffusion` | seaice_diffusion.F:6 | 48 | 15/20 | 67, 95-98 | 63:1/2, 66:1/2, 94:1/2 |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 4 | 63/206 | 158-168, 228-230, 245-250, 268-273, 310-317, 340, 415-733 | 136:1/2, 157:1/2, 227:1/2, 243:1/2, 244:1/2, 267:1/2, 285:1/2, 306:1/2, 309:1/2, 338:1/2, 344:1/2, 376:1/2, 414:1/2 |
| `seaice_freedrift` | seaice_freedrift.F:4 | 4 | 61/64 | 45, 100, 118 | 44:1/2, 99:1/2, 117:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 4 | 26/57 | 159-164, 173, 178, 221-232, 245-273 | 98:1/2, 146:1/2, 157:1/2, 172:1/2, 177:1/2, 244:1/2 |
| `seaice_growth` | seaice_growth.F:15 | 4 | 273/344 | 336-337, 344, 568-570, 749-759, 1033, 1042, 1464-1471, 1808, 1823, 1836, 1839, 2136, 2348, 2352, 2355, 2425-2435, 2483-2555, 2671-2677 | 335:1/2, 343:1/2, 567:1/2, 747:1/2, 790:1/2, 1030:1/2, 1040:1/2, 1299:1/2, 1462:1/2, 1528:1/2, 1698:1/2, 1807:1/2, 1822:1/2, 1833:1/2, 1837:1/2, 2134:1/2, 2135:1/2, 2346:1/2, 2350:1/2, 2353:1/2, 2356:1/2, 2424:1/2, 2482:1/2, 2667:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 42/79 | 60, 67, 80, 87-89, 229, 296-353, 365-370 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2, 82:1/2, 228:1/2, 278:1/2, 279:1/2, 362:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 109/160 | 54, 237-242, 254, 272, 274, 277-288, 293-299, 318-327, 346-352, 392-393, 448-451, 463-471 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 271:1/2, 273:1/2, 275:1/2, 292:1/2, 317:1/2, 345:1/2, 391:1/2, 447:1/2, 462:1/2 |
| `seaice_lsr` | seaice_lsr.F:24 | 4 | 170/227 | 185-186, 193, 234-241, 342-349, 597-601, 617-637, 651-685, 799-800, 1120-1133, 1152-1160 | 184:1/2, 192:1/2, 233:1/2, 300:1/2, 328:1/2, 595:1/2, 607:1/2, 615:1/2, 616:1/2, 639:1/2, 646:1/2, 798:1/2, 953:1/2, 976:1/2, 1116:1/2, 1139:1/2 |
| `seaice_lsr_calc_coeffs` | seaice_lsr.F:1294 | 8 | 74/85 | 1387-1390, 1402-1405, 1597-1600 | 1386:1/2, 1397:1/2, 1401:1/2, 1589:1/2 |
| `seaice_lsr_rhsu` | seaice_lsr.F:1616 | 32 | 27/31 | 1708-1723 | 1704:1/2, 1741:1/2 |
| `seaice_lsr_rhsv` | seaice_lsr.F:1771 | 32 | 27/31 | 1864-1879 | 1860:1/2, 1896:1/2 |
| `seaice_lsr_tridiagu` | seaice_lsr.F:1926 | 1128 | 29/29 | - | 1998:1/4 |
| `seaice_lsr_tridiagv` | seaice_lsr.F:2071 | 1928 | 28/28 | - | 2147:1/4 |
| `seaice_model` | seaice_model.F:13 | 4 | 38/72 | 92, 103-115, 130-133, 254-257, 281-286, 341, 369-390 | 82:1/2, 85:1/2, 102:1/2, 125:1/2, 128:1/2, 178:1/2, 222:1/2, 224:1/2, 252:1/2, 267:1/2, 275:1/2, 340:1/2, 366:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 5 | 39/48 | 65, 70-80 | 55:1/2, 58:1/2, 62:1/2, 69:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 4 | 18/29 | 56, 69-92 | 55:1/2, 64:1/2 |
| `seaice_oceandrag_coeffs` | seaice_oceandrag_coeffs.F:9 | 8 | 16/18 | 64, 95 | 63:1/2, 94:1/2 |
| `seaice_output` | seaice_output.F:10 | 5 | 26/54 | 66-105, 112, 155-159 | 57:1/2, 65:1/2, 108:1/2, 109:1/2, 127:1/2, 153:1/2, 174:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 446/832 | 233-238, 460-468, 752-757, 773-827, 837-840, 846-856, 866, 868, 912-913, 921-932, 936-942, 954-960, 965-971, 975-986, 992, 1000, 1006-1012, 1017-1025, 1032-1036, 1044, 1053, 1055, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1442-1444, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1518-1522, 1525-1529, 1532-1536, 1539-1543, 1546-1549 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 720:1/2, 721:2/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 763:1/2, 767:1/2, 769:1/2, 771:1/2, 834:1/2, 835:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 896:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 951:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1440:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1511:1/2, 1514:1/2, 1517:1/2, 1524:1/2, 1531:1/2, 1538:1/2, 1545:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 4 | 53/62 | 125-144, 394 | 124:1/2, 393:1/2 |
| `seaice_residual` | seaice_lsr.F:1173 | 16 | 20/20 | - | 1260:1/2, 1278:1/2, 1282:1/2, 1283:1/2 |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 112 | 106/148 | 191, 289-294, 310, 324, 392-404, 503-545, 553-561 | 190:1/2, 217:1/2, 288:1/2, 309:1/2, 321:1/2, 385:1/2, 445:1/2, 502:1/2, 552:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 240/261 | 109-111, 287-311, 357-359, 390, 400, 480-482 | 45:1/2, 108:1/2, 225:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 388:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 1 | 46/75 | 109-110, 160-168, 172-188, 194-200 | 78:1/2, 101:1/2, 103:1/2, 117:1/2, 121:1/2, 125:1/2, 133:1/2, 152:1/2, 157:1/2, 159:1/2, 171:1/2, 193:1/2 |

### Compiled but not executed (932 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_r_star, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, freesurf_rescale_g, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cartesian_grid, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_daysformonth, cal_dayspermonth, cal_getmonthsrec, cal_monthsforyear, cal_monthsperyear, cal_printdate, cal_printerror, cal_stepsforday, cal_stepsperday, cal_timestamp, cal_weekday
- pkg/cd_code: cd_code_read_pickup
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/down_slope: dwnslp_diagnostics_init
- pkg/ecco: cost_bp_read, cost_gencost_seaicev4, cost_gencost_sshv4, cost_gencost_sstv4, cost_gencost_transp, cost_sla_read, cost_sla_read_yd, ecco_add, ecco_cprsrl, ecco_diagnostics_init, ecco_div, ecco_error, ecco_multfield, ecco_read_pickup, ecco_subtract, get_exconc_deconc
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: kpp_calc_diff_ptr, kpp_calc_dummy, kpp_diagnostics_init, kpp_doublediff, kpp_transport_ptr, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mnc: mnc_chk_vtyp_r_ncvar, mnc_cw_add_gname, mnc_cw_add_vattr_any, mnc_cw_add_vattr_dbl, mnc_cw_add_vattr_int, mnc_cw_add_vattr_text, mnc_cw_add_vname, mnc_cw_append_vname, mnc_cw_citer_getg, mnc_cw_citer_setg, mnc_cw_del_gname, mnc_cw_del_vname, mnc_cw_dump, mnc_cw_file_aorc, mnc_cw_get_citer, mnc_cw_get_face_num, mnc_cw_get_tile_num, mnc_cw_get_udim, mnc_cw_get_xyfo, mnc_cw_i_r, mnc_cw_i_r_s, mnc_cw_i_r_tf, mnc_cw_i_w, mnc_cw_i_w_offset, mnc_cw_i_w_s, mnc_cw_init, mnc_cw_rl_r, mnc_cw_rl_r_s, mnc_cw_rl_r_tf, mnc_cw_rl_w, mnc_cw_rl_w_offset, mnc_cw_rl_w_s, mnc_cw_rs_r, mnc_cw_rs_r_s, mnc_cw_rs_r_tf, mnc_cw_rs_w, mnc_cw_rs_w_offset, mnc_cw_rs_w_s, mnc_cw_set_citer, mnc_cw_set_gattr, mnc_cw_set_udim, mnc_cw_vattr_missing, mnc_cw_write_cvar, mnc_cw_write_grid_coord, mnc_cw_write_grid_info, mnc_dim_init, mnc_dim_init_all, mnc_dim_init_all_cv, mnc_dim_unlim_size, mnc_dump, mnc_dump_all, mnc_file_add_attr_any, mnc_file_add_attr_dbl, mnc_file_add_attr_int, mnc_file_add_attr_real, mnc_file_add_attr_str, mnc_file_close, mnc_file_close_all, mnc_file_close_all_matching, mnc_file_create, mnc_file_enddef, mnc_file_open, mnc_file_readall, mnc_file_redef, mnc_file_try_read, mnc_get_fvinds, mnc_get_ind, mnc_get_next_empty_ind, mnc_grid_get_dimind, mnc_grid_init, mnc_grid_init_all, mnc_handle_err, mnc_init, mnc_psncm, mnc_set_outdir, mnc_update_time, mnc_var_add_attr_any, mnc_var_add_attr_dbl, mnc_var_add_attr_int, mnc_var_add_attr_real, mnc_var_add_attr_str, mnc_var_append_dbl, mnc_var_append_int, mnc_var_append_real, mnc_var_init_any, mnc_var_init_dbl, mnc_var_init_int, mnc_var_init_real, mnc_var_write_any, mnc_var_write_dbl, mnc_var_write_int, mnc_var_write_real
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_uv_boundary, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/salt_plume: salt_plume_apply, salt_plume_calc_depth, salt_plume_check, salt_plume_diagnostics_fill, salt_plume_diagnostics_init, salt_plume_do_exch, salt_plume_forcing_surf, salt_plume_frac, salt_plume_init_fixed, salt_plume_init_varia, salt_plume_mnc_init, salt_plume_tendency_apply_s, salt_plume_tendency_apply_t, salt_plume_volfrac
- pkg/seaice: adseaice_monitor, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advection, seaice_bottomdrag_coeffs, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_stress, seaice_calc_stressdiv, seaice_check_pickup, seaice_cost_export, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_growth_adx, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys

## lab_sea/input_ad.noseaice

Run `$MJX_REFERENCE/coverage/lab_sea/input_ad.noseaice/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/lab_sea/input_ad.noseaice/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| theta | 13 blocks, first 1.404768e+00, last 1.403050e+00, min 1.403050e+00, max 1.404768e+00 | yes |
| EXF bulk formulae | exf_bulkformulae (12 calls) | yes |
| KPP | kpp_calc (48 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (12 calls) | yes |
| cost function | global fc = 21.4674959933052; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

(parameter files could not be read by mitjax.config: ValueError: $MJX_UPSTREAM/verification/lab_sea/input_ad/data.err:1: text outside a namelist group: '0.25\n    0.5201    0.2676\n    '; all printed switches are listed)

- `.TRUE.`: calc_wVelocity, doAB_onGtGs, doSaltClimRelax, dumpInitAndLast, fluidIsWater, implicitDiffusion, implicitFreeSurface, implicitViscosity, inAdExact, KPP_ghatUseTotalDiffus, KPPwriteState, LimitHblStable, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, snapshot_mnc, staggerTimeStep, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCDscheme, useCoriolis, useCtrlCostContribution, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useKPPinAdMode, useMultiDimAdvec, usingGregorianCalendar, usingSphericalPolarGrid, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, diag_mnc, diags_opOceWeighted, doResetHFactors, doThetaClimRelax, dumpAtLast, exactConserv, fluidIsAir, globalFiles, GM_AdvForm, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, implicitIntGravWave, interDiffKr_pCell, interViscAr_pCell, KPPuseDoubleDiff, KPPuseSWfrac3D, linFSConserveTr, momImplVertAdv, monitor_mnc, no_slip_sides, noNegativeEvap, nonHydrostatic, pickup_read_mnc, pickup_write_mnc, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFREEDRIFTswitchInAd, snapshot_mdsio, stressIsOnCgrid, tempImplVertAdv, tempSOM_Advection, twoDigitYear, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useMin4hFacEdges, useMissingValue, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSEAICEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCurvilinearGrid, usingCylindricalGrid, usingJulianCalendar, usingModelCalendar, usingMPI, usingNoLeapYearCal, usingPCoords, vectorInvariantMomentum
- selectors: exf_adjMonSelect=3, monitorSelect=3, pCellMix_select=0, saltAdvScheme=30, saltVertAdvScheme=30, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempAdvScheme=30, tempVertAdvScheme=30

### Executed routines (574)

**eesupp/src** (78)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 13608 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 6366 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `diff_phase_multiple` | diff_phase_multiple.F:7 | 60 | 15/16 | 46 | 44:1/2, 45:1/2, 50:1/2 |
| `different_multiple` | different_multiple.F:7 | 171 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 56/77 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1284 | 27/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 240 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 12 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1524 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1284 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1284 | 38/90 | 289-324, 346-381 | 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1284 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1284 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 240 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 240 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 240 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 240 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 1054 | 11/12 | 60 | 56:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 12 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 12 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_dgrid_3d_rl` | exch_uv_dgrid_3d_rl.F:8 | 12 | 14/32 | 88-177 | 72:1/2, 74:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 29 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 105 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 180 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 41 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 32706 | 2/2 | - | - |
| `gather_2d_r8` | gather_2d_r8.F:7 | 1593 | 14/14 | - | 61:1/2, 63:1/2 |
| `global_max_r8` | global_max.F:97 | 1596 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 73 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 1236 | 12/12 | - | 154:1/2 |
| `global_sum_singlecpu_rl` | global_sum_singlecpu.F:15 | 1593 | 22/22 | - | 98:1/2, 107:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 1943 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 10730 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 1472330 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 27 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 2421 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1094 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 836 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 6096 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 394 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 14 | 30/40 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 55 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 130 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 234 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 5267 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `timer_control` | timers.F:74 | 692 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 692 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 692 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 347 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 345 | 4/4 | - | - |
| `ucase` | utils.F:311 | 1387 | 7/8 | 333 | 332:1/2 |
| `write_0d_c` | write_utils.F:579 | 4 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 47 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 130 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 189 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 44 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (107)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 2208 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 1104 | 13/25 | 838, 840, 842, 913-918, 925-929, 955 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2, 951:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 1104 | 19/55 | 467, 469, 471, 563-612, 627-632, 639-643, 723 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2, 719:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 1104 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 1104 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 192 | 26/32 | 164-166, 195-197 | 132:1/2, 180:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 2208 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 1104 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 1104 | 19/19 | - | - |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 48 | 8/8 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 48 | 7/81 | 76, 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 1104 | 36/162 | 149, 184, 212-240, 268-610, 636-648 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 48 | 12/16 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 12 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `config_check` | config_check.F:14 | 1 | 96/545 | 92-97, 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 986-989, 992-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 985:1/2, 991:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 443/541 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 533-538, 554-588, 756-762, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 48 | 26/57 | 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 96 | 6/6 | - | - |
| `diags_oceanic_surf_flux` | diags_oceanic_surf_flux.F:7 | 12 | 42/43 | 56 | 55:1/2, 81:1/2, 94:1/2, 115:1/2, 131:1/2, 161:1/2 |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 1104 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 1104 | 24/32 | 79-84, 123-125 | 61:1/2, 76:1/2, 121:1/2 |
| `diags_rho_g` | diags_rho.F:100 | 12 | 8/33 | 161-171, 178-188, 195-213 | 160:1/2, 177:1/2, 194:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 12 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 12 | 10/16 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 12 | 113/151 | 258, 260, 262, 264, 450-464, 471, 548, 557, 735-739, 771-784, 797-798, 830-833, 869-874, 882, 906, 934, 962, 1043, 1051-1058, 1123 | 248:1/2, 251:1/2, 256:1/2, 257:1/2, 259:1/2, 261:1/2, 263:1/2, 421:1/2, 467:1/2, 546:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 734:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 867:1/2, 878:1/2, 896:1/2, 903:1/2, 932:1/2, 951:1/2, 953:1/2, 1030:1/2, 1032:1/2, 1049:1/2, 1096:1/2, 1102:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1126:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 12 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_statevars_diags` | do_statevars_diags.F:8 | 36 | 14/16 | 64, 96 | 54:1/2, 60:1/2, 95:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 13 | 14/21 | 97-110, 194 | 96:1/2, 116:1/2, 144:1/2, 187:1/2, 193:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 12 | 23/26 | 84, 115-117 | 66:1/2, 83:1/2, 94:1/2, 99:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 12 | 70/84 | 358, 533, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 615:1/2, 682:1/2, 693:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 12 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 12 | 42/69 | 75, 151-153, 250, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 247:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_alpha` | find_alpha.F:14 | 1104 | 29/76 | 77-79, 85-99, 222-337 | 75:1/2, 83:1/2, 113:1/2 |
| `find_beta` | find_alpha.F:347 | 1104 | 27/74 | 410-412, 418-430, 539-641 | 408:1/2, 416:1/2, 444:1/2 |
| `find_bulkmod` | find_rho.F:412 | 7676 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 5468 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 48427 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 7676 | 16/16 | - | - |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 12 | 16/21 | 64, 77-88 | 63:1/2, 75:1/2, 242:1/2 |
| `forward_step` | forward_step.F:70 | 12 | 91/102 | 730-734, 767-769, 980, 1176, 1201-1202 | 424:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 812:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1169:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 1104 | 20/22 | 61, 74 | 59:1/2, 72:1/2 |
| `impldiff` | impldiff.F:7 | 288 | 76/102 | 289-298, 317, 324-382 | 199:1/2, 219:1/2, 288:1/2, 314:1/2, 319:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/71 | 57-63, 70-79, 106-112, 121-180, 191, 196-201 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 195:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/76 | 63-66, 94-98, 107-114, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 106:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 57/87 | 52, 60, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/121 | 131, 134-144, 192, 197-202 | 130:1/2, 132:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 171:1/2, 175:1/2, 185:1/2, 189:1/2, 196:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 177/193 | 210-212, 247-261, 435, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_mnc_vars` | ini_mnc_vars.F:10 | 1 | 203/204 | 223 | 36:1/2, 206:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 38/58 | 146-179, 191-195 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 206:1/2, 217:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 404/881 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 584, 586-589, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1351-1355, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 585:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1116:1/2, 1117:1/2, 1118:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 20/22 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 26/45 | 68-73, 100, 109-124, 130 | 65:1/2, 67:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_spherical_polar_grid` | ini_spherical_polar_grid.F:8 | 1 | 75/85 | 258-263, 277-280 | 119:1/2, 224:1/2, 237:1/2, 257:1/2, 276:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 27/54 | 69-74, 101, 110-125, 131-138, 151 | 66:1/2, 68:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 13 | 14/65 | 92-185, 211-221, 279-282, 329-331 | 90:1/2, 207:1/2, 277:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 1196 | 17/28 | 95-117 | 93:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 12 | 27/34 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 27/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161, 173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 12 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 12 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 110/124 | 242-261 | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 64/98 | 207, 212, 280, 289, 304, 321, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 252:1/2, 260:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 339:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 428:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 44/52 | 349-351, 367-369, 512-514, 521-523 | 144:1/2, 158:1/2, 160:1/2, 167:1/2, 170:1/2, 174:1/2, 193:1/2, 200:1/2, 202:1/2, 209:1/2, 211:1/2, 249:1/2, 251:1/2, 317:1/2, 319:1/2, 327:1/2, 329:1/2, 347:1/2, 358:1/2, 365:1/2, 374:1/2, 379:1/2, 510:1/2, 519:1/2, 651:1/2, 655:1/2, 673:1/2, 675:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 27/31 | 169, 286, 445, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 204:1/2, 206:1/2, 251:1/2, 253:1/2, 270:1/2, 285:1/2, 291:1/2, 293:1/2, 435:1/2, 444:1/2, 601:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 24 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 17/17 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 3 | 32/37 | 68-70, 82-83 | 60:1/2, 62:1/2, 64:1/2, 78:1/2, 97:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 2 | 14/15 | 186 | 103:1/2, 124:1/2, 184:1/2, 223:1/2, 237:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 7676 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 38 | 50/64 | 67-70, 78, 100-101, 168-174 | 66:1/2, 77:1/2, 99:1/2, 111:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 48 | 57/70 | 169, 192, 326, 367-369, 386-390, 448-461, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 444:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 96/166 | 56-57, 60-63, 67, 71, 76, 109-115, 120-126, 171-175, 186-189, 192-195, 202, 208, 214, 220, 226, 232, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 12 | 60/94 | 143-147, 152-157, 164, 170-176, 218-224, 265, 269, 319, 327, 342-344, 355-366 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1201 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 48 | 57/70 | 171, 194, 328, 369-371, 388-392, 450-463, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 446:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 53/53 | - | 292:1/2, 382:1/2, 454:1/2, 472:1/2, 664:1/2, 666:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/45 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 12 | 44/52 | 158, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 1104 | 73/94 | 118-121, 140-143, 220-223, 328-332, 400-405 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 229:1/2, 321:1/2, 391:1/2, 392:1/2, 412:1/2, 416:1/2, 421:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 96 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 12 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 19/20 | 107 | 55:1/2, 78:1/2, 88:1/2, 106:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 55/110 | 81-140, 198-199, 217-219 | 73:1/2, 78:1/2, 162:1/2, 164:1/2, 197:1/2, 214:1/2 |
| `write_pickup` | write_pickup.F:8 | 2 | 52/100 | 253-257, 261-265, 288-290, 335-338, 365-371, 391-438 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2, 390:1/2 |
| `write_state` | write_state.F:11 | 13 | 31/46 | 87, 123-154, 178 | 84:1/2, 95:1/2, 175:1/2, 177:1/2 |

**pkg/autodiff** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 87 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 81 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 6 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 43 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 40 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 3 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 3 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 12 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 12 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 135/145 | 69-83 | 66:1/2, 68:1/2, 121:1/2, 307:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 80/148 | 63-68, 127-132, 157, 159, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 12 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 12 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 26 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 13 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 12 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 12 | 2/2 | - | - |

**pkg/cal** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_addtime` | cal_addtime.F:3 | 401 | 47/107 | 66-75, 79-81, 89-106, 114-116, 127-140, 169-171, 179, 181-186, 192-199 | 65:1/2, 78:1/2, 88:1/2, 110:1/2, 124:1/2, 167:1/2, 168:1/2, 177:1/2, 180:1/2, 191:1/2, 200:2/2, 214:1/2, 218:1/2 |
| `cal_checkdate` | cal_checkdate.F:3 | 47 | 22/54 | 61-63, 65-70, 76-81, 89-93, 96, 100-102, 106-108, 111, 123-126, 129-132, 135-138 | 59:1/2, 64:1/2, 73:1/2, 88:1/2, 94:1/2, 98:1/2, 103:1/2, 109:1/2, 116:1/2, 122:1/2, 128:1/2, 134:1/2 |
| `cal_compdates` | cal_compdates.F:3 | 26 | 5/5 | - | - |
| `cal_convdate` | cal_convdate.F:3 | 1144 | 21/37 | 51-57, 66-68, 71-73, 87-89 | 50:1/2, 65:1/2, 70:1/2, 82:1/2 |
| `cal_copydate` | cal_copydate.F:3 | 169 | 6/6 | - | - |
| `cal_fulldate` | cal_fulldate.F:3 | 47 | 15/37 | 59-65, 71-74, 87-93, 98-101 | 58:1/2, 70:1/2, 77:1/2, 84:1/2 |
| `cal_getdate` | cal_getdate.F:3 | 758 | 16/23 | 57-63 | 55:1/2 |
| `cal_init_fixed` | cal_init_fixed.F:8 | 1 | 6/6 | - | 28:1/2, 42:1/2 |
| `cal_intdays` | cal_intdays.F:3 | 2 | 9/10 | 58 | 55:1/2 |
| `cal_intmonths` | cal_intmonths.F:3 | 2 | 9/11 | 62, 69 | 59:1/2, 67:1/2 |
| `cal_intyears` | cal_intyears.F:3 | 2 | 4/5 | 49 | 47:1/2 |
| `cal_isleap` | cal_isleap.F:3 | 19871 | 9/21 | 41-47, 60-67 | 40:1/2, 50:1/2 |
| `cal_numints` | cal_numints.F:3 | 3 | 7/10 | 61-63 | 53:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 19/23 | 70-76 | 65:1/2, 68:1/2, 79:1/2, 114:1/2 |
| `cal_set` | cal_set.F:3 | 1 | 60/93 | 104-114, 163-184, 202-204, 207-209, 212-214 | 82:1/2, 99:1/2, 119:1/2, 160:1/2, 201:1/2, 206:1/2, 211:1/2 |
| `cal_subdates` | cal_subdates.F:3 | 2 | 6/24 | 48-57, 65-69, 77-79 | 47:1/2, 60:1/2, 63:1/2 |
| `cal_summary` | cal_summary.F:3 | 1 | 43/43 | - | 49:1/2 |
| `cal_time2dump` | cal_time2dump.F:3 | 84 | 3/15 | 36-51 | 31:1/2 |
| `cal_timeinterval` | cal_timeinterval.F:3 | 413 | 15/36 | 60-66, 73-92 | 57:1/2, 59:1/2 |
| `cal_timepassed` | cal_timepassed.F:3 | 501 | 52/75 | 67-76, 98, 120-127, 148-149, 182-184 | 66:1/2, 87:1/2, 97:1/2, 119:1/2, 147:1/2 |
| `cal_toseconds` | cal_toseconds.F:3 | 455 | 15/26 | 57-63, 69, 92-94 | 56:1/2, 67:1/2, 73:1/2 |

**pkg/cd_code** (4)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cd_code_ini_vars` | cd_code_ini_vars.F:3 | 1 | 15/16 | 53 | 52:1/2 |
| `cd_code_init_fixed` | cd_code_init_fixed.F:3 | 1 | 23/23 | - | 22:1/2 |
| `cd_code_scheme` | cd_code_scheme.F:7 | 1104 | 47/49 | 82-83 | 78:1/2 |
| `cd_code_write_pickup` | cd_code_write_pickup.F:8 | 2 | 12/26 | 47-66 | 70:1/2, 85:1/2 |

**pkg/cost** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 18/18 | - | - |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 12/12 | - | 44:1/2, 48:1/2, 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 47/47 | - | 82:1/2, 97:1/2, 102:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 21/21 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 12 | 2/2 | - | - |

**pkg/ctrl** (33)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 4 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 20 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 2 | 11/13 | 51, 54 | 41:1/2, 50:1/2, 53:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 31/67 | 102-110, 124-128, 139-143, 182-186, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 121:1/2, 136:1/2, 179:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 32/37 | 74, 83-84, 113, 140 | 65:1/2, 73:1/2, 78:1/2, 82:1/2, 112:1/2, 120:1/2, 139:1/2, 147:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 83/83 | - | 66:1/2, 131:1/2, 146:1/2, 161:1/2, 176:1/2, 186:1/2, 200:1/2, 214:1/2 |
| `ctrl_cost_gen2d` | ctrl_cost_gen.F:12 | 11 | 41/42 | 158 | 121:1/2, 157:1/2, 162:1/2 |
| `ctrl_cost_gen3d` | ctrl_cost_gen.F:187 | 2 | 41/42 | 322 | 283:1/2, 319:1/2, 326:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 143 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 108 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 108 | 35/66 | 99, 104-108, 145, 174-197, 206-223 | 79:1/2, 92:1/2, 100:1/2, 144:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 139 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 4 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 31 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 64/73 | 272-273, 283, 285, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 18 | 21/36 | 72-76, 87-88, 90-91, 106-112, 118-124 | 86:1/2, 89:1/2, 93:1/2, 104:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 168:1/2, 174:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 12 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr2d` | ctrl_map_genarr.F:13 | 2 | 42/57 | 91-93, 97-99, 102, 108-109, 153-160, 168-170 | 71:1/2, 90:1/2, 95:1/2, 101:1/2, 104:1/2, 145:1/2, 149:1/2, 152:1/2, 167:1/2, 199:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 2 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 12 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 42/49 | 157, 159, 161, 201, 360, 362, 364 | 128:1/2, 154:1/2, 156:1/2, 158:1/2, 160:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 71/127 | 151-153, 157-159, 162, 183-190, 254-261, 276-393, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 253:1/2, 272:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 217/245 | 181-186, 565, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 539:1/2, 540:2/4, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 31 | 15/16 | 60 | 42:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 29 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 2 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 102/140 | 153-155, 169-171, 184-187, 218-222, 237-241, 267-292, 304-306 | 136:1/2, 146:1/2, 162:1/2, 178:1/2, 217:1/2, 236:1/2, 255:1/2, 259:1/4, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 9 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (47)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_addtolist` | diagnostics_addtolist.F:8 | 274 | 17/44 | 62, 68-92, 111-117 | 60:1/2, 67:1/2, 99:1/2, 103:1/2, 108:1/2, 125:1/2 |
| `diagnostics_check` | diagnostics_check.F:8 | 1 | 31/150 | 44-47, 51-54, 59-64, 85-89, 92-96, 104-113, 120-130, 145-213, 231-237, 244-250, 261-268, 277-280 | 37:1/2, 43:1/2, 50:1/2, 57:1/2, 84:1/2, 91:1/2, 103:1/2, 118:1/2, 119:2/2, 144:1/2, 224:1/2, 230:1/2, 243:1/2, 260:1/2, 275:1/2 |
| `diagnostics_clear` | diagnostics_clear.F:6 | 4 | 8/8 | - | 32:1/2 |
| `diagnostics_clrdiag` | diagnostics_clear.F:51 | 26 | 10/10 | - | - |
| `diagnostics_count` | diagnostics_utils.F:19 | 96 | 8/19 | 57-58, 69-81 | 56:1/2, 68:1/2 |
| `diagnostics_cumulate` | diagnostics_fill_field.F:413 | 3792 | 20/62 | 478-517, 521-556, 569-578, 602 | 477:1/2, 520:1/2, 561:1/2, 594:1/2 |
| `diagnostics_fill` | diagnostics_fill.F:6 | 37536 | 31/34 | 75, 130-131 | 73:1/2, 94:1/2, 119:1/2, 129:1/2 |
| `diagnostics_fill_field` | diagnostics_fill_field.F:13 | 156 | 49/101 | 114-116, 136-141, 146-151, 159-160, 170-177, 182-183, 188-189, 195, 203-216, 235-268 | 105:3/6, 107:1/2, 132:1/2, 145:1/2, 158:1/2, 167:1/2, 181:1/2, 184:1/2, 192:1/2, 194:1/2, 202:1/2, 220:1/2, 226:1/2 |
| `diagnostics_fill_rs` | diagnostics_fill_rs.F:6 | 48 | 14/34 | 75, 84-85, 93-103, 118-146 | 73:1/2, 80:1/2, 92:1/2, 117:1/2 |
| `diagnostics_fill_state` | diagnostics_fill_state.F:6 | 36 | 59/287 | 70-79, 94-95, 112-139, 145-166, 170-183, 187-203, 207-223, 229-241, 245-257, 261-274, 278-290, 294-306, 310-323, 327-339, 343-355, 359-380, 398-410, 423-435, 452, 502, 504, 537-550, 556-568, 572-584, 590-603, 607-620, 624-637, 641-654, 658-671, 675-688, 714-724, 739-749 | 69:1/2, 93:1/2, 111:1/2, 143:1/2, 169:1/2, 186:1/2, 206:1/2, 228:1/2, 244:1/2, 260:1/2, 277:1/2, 293:1/2, 309:1/2, 326:1/2, 342:1/2, 358:1/2, 393:1/2, 418:1/2, 451:1/2, 501:1/2, 503:1/2, 534:1/2, 555:1/2, 571:1/2, 589:1/2, 606:1/2, 623:1/2, 640:1/2, 657:1/2, 674:1/2, 709:1/2, 734:1/2 |
| `diagnostics_get_diag` | diagnostics_utils.F:97 | 632 | 16/26 | 141-142, 177-186 | 137:1/2, 139:1/2, 147:1/2, 149:1/2, 154:1/2 |
| `diagnostics_ini_io` | diagnostics_ini_io.F:6 | 1 | 4/12 | 45-54 | 39:1/2, 42:1/2 |
| `diagnostics_init_early` | diagnostics_init_early.F:8 | 1 | 107/107 | - | 71:1/2 |
| `diagnostics_init_fixed` | diagnostics_init_fixed.F:8 | 1 | 8/8 | - | - |
| `diagnostics_init_varia` | diagnostics_init_varia.F:8 | 1 | 24/24 | - | 30:1/2 |
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 5928 | 14/16 | 56-57 | 45:1/2, 55:1/2 |
| `diagnostics_main_init` | diagnostics_main_init.F:8 | 1 | 583/619 | 95-97, 104-110, 112-117, 128-129, 131-132, 285, 780-801 | 94:1/2, 103:1/2, 111:1/2, 127:1/2, 130:1/2, 201:1/2, 248:1/2, 284:1/2, 621:1/2, 633:1/2, 779:1/2 |
| `diagnostics_out` | diagnostics_out.F:8 | 4 | 83/164 | 95, 100-102, 112-113, 130, 159, 175, 177-184, 203-208, 211-244, 258-267, 277-278, 308-336, 347-357, 365, 370-382, 410, 416-425 | 92:1/2, 99:1/2, 107:1/2, 110:1/2, 129:1/2, 141:1/2, 155:1/2, 173:1/2, 176:1/2, 190:1/2, 195:1/2, 196:1/2, 199:1/2, 209:1/2, 255:1/2, 256:1/2, 275:1/2, 294:1/2, 307:1/2, 345:1/2, 360:1/2, 367:1/2, 392:1/2, 398:1/2, 399:1/2, 401:1/2, 415:1/2, 438:1/2 |
| `diagnostics_read_pickup` | diagnostics_read_pickup.F:7 | 1 | 2/2 | - | - |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 256/370 | 111-119, 280-282, 293, 295, 298-300, 302-309, 311-312, 322, 333-342, 357-366, 372-381, 421-424, 429-435, 446-447, 454-467, 471-478, 493-502, 508-517, 538-540, 582-583, 587-588, 593-596 | 109:1/2, 123:1/2, 168:1/2, 279:1/2, 292:1/2, 294:1/2, 297:1/2, 301:1/2, 310:1/2, 313:1/2, 321:1/2, 332:1/2, 356:1/2, 371:1/2, 420:1/2, 428:1/2, 444:1/2, 453:1/2, 469:1/2, 481:1/2, 492:1/2, 507:1/2, 536:1/2, 570:1/2, 586:1/2, 592:1/2, 600:1/2, 601:2/4, 603:1/4, 607:2/4, 609:1/4, 629:1/2, 631:1/2, 635:2/4, 638:1/4 |
| `diagnostics_scale_fill` | diagnostics_scale_fill.F:6 | 312 | 21/33 | 78, 120-146 | 76:1/2, 96:1/2, 119:1/2 |
| `diagnostics_scale_fill_rs` | diagnostics_scale_fill_rs.F:6 | 72 | 19/33 | 78, 86-87, 120-148 | 76:1/2, 82:1/2, 96:1/2, 119:1/2 |
| `diagnostics_set_calc` | diagnostics_set_calc.F:9 | 1 | 7/80 | 78-194 | 46:1/2 |
| `diagnostics_set_levels` | diagnostics_set_levels.F:7 | 1 | 75/133 | 88, 95-115, 119-122, 131-134, 138-141, 148-151, 156-160, 164-167, 218-220, 226-228, 244-253 | 73:1/2, 87:1/2, 93:1/2, 118:1/2, 130:1/2, 137:1/2, 147:1/2, 155:1/2, 163:1/2, 183:1/2, 217:1/2, 221:1/2, 241:1/2, 243:1/2 |
| `diagnostics_set_pointers` | diagnostics_set_pointers.F:6 | 1 | 88/160 | 59-63, 70-96, 101-109, 145-151, 166-186, 248-251, 254-260, 292-302 | 41:1/2, 57:1/2, 58:1/2, 69:1/2, 100:1/2, 142:1/2, 158:1/2, 194:1/2, 206:1/2, 246:1/2, 252:1/2, 271:1/2, 272:1/4, 274:1/4, 278:1/2, 286:1/2 |
| `diagnostics_setdiag` | diagnostics_setdiag.F:6 | 23 | 35/89 | 66-72, 92-94, 109-111, 118-126, 133-134, 142-145, 156, 159-160, 165-210 | 65:1/2, 90:1/2, 101:1/2, 107:1/2, 114:1/2, 115:1/2, 131:1/2, 155:1/2, 158:1/2, 163:1/2 |
| `diagnostics_summary` | diagnostics_summary.F:7 | 5 | 97/124 | 87, 118-120, 136-159, 229-232 | 55:1/2, 60:1/2, 69:1/2, 72:1/2, 90:1/2, 117:1/2, 123:2/4, 125:1/4, 135:1/2, 162:1/2, 164:2/4, 169:2/4, 207:1/2, 208:2/4, 221:1/2, 223:2/4, 228:1/2 |
| `diagnostics_switch_onoff` | diagnostics_switch_onoff.F:11 | 12 | 16/78 | 79, 101-135, 146-173, 188-235 | 77:1/2, 87:1/2, 98:1/2, 144:1/2, 185:1/2 |
| `diagnostics_write` | diagnostics_write.F:3 | 13 | 50/54 | 69-70, 117-118 | 62:1/2, 84:1/2, 91:1/2, 92:1/2, 110:1/2, 132:1/2, 139:1/2, 151:1/2, 161:1/2 |
| `diagnostics_write_pickup` | diagnostics_write_pickup.F:7 | 2 | 3/3 | - | - |
| `diags_mk_title` | diagnostics_utils.F:593 | 36 | 17/23 | 643-650 | 632:1/2, 635:1/2, 642:1/2 |
| `diags_mk_units` | diagnostics_utils.F:514 | 45 | 13/30 | 554-568, 575-582 | 547:1/2, 552:1/2, 574:1/2 |
| `diags_renamed` | diagnostics_utils.F:661 | 500 | 8/19 | 695-700, 708-714 | 694:1/2, 702:1/2, 703:1/2, 705:1/2 |
| `diags_track_diva` | diagnostics_utils.F:410 | 1 | 5/9 | 448-452 | 444:1/2 |
| `diagstats_ascii_out` | diagstats_ascii_out.F:8 | 20 | 19/19 | - | 43:1/2, 45:1/2, 55:1/2, 75:1/4 |
| `diagstats_calc` | diagstats_calc.F:6 | 4464 | 43/67 | 99-102, 116-121, 125-130, 153-155, 161-163, 167-170, 174-177 | 98:1/2, 115:1/2, 124:1/2, 152:1/2, 160:1/2, 166:1/2, 173:1/2 |
| `diagstats_clear` | diagstats_clear.F:8 | 4 | 7/7 | - | 30:1/2 |
| `diagstats_close_io` | diagstats_close_io.F:6 | 1 | 16/18 | 55, 71 | 40:1/2, 42:1/2, 52:1/2, 70:1/2 |
| `diagstats_clrdiag` | diagstats_clear.F:46 | 20 | 8/8 | - | - |
| `diagstats_fill` | diagstats_fill.F:13 | 60 | 34/78 | 121-122, 136-141, 149-150, 160-167, 173-175, 181-183, 188, 196-209, 234-246 | 120:1/2, 135:1/2, 148:1/2, 157:1/2, 172:1/2, 176:1/2, 187:1/2, 195:1/2, 216:1/2 |
| `diagstats_global` | diagstats_global.F:8 | 20 | 55/87 | 87-96, 115-116, 127-129, 151-159, 188-206 | 61:1/2, 63:1/2, 86:1/2, 113:1/2, 126:1/2, 150:1/2, 168:1/2, 181:1/2 |
| `diagstats_ini_io` | diagstats_ini_io.F:6 | 1 | 36/37 | 54 | 40:1/2, 42:1/2, 51:1/2, 76:1/2, 77:2/4, 83:2/4, 85:1/4, 87:2/4, 89:1/4 |
| `diagstats_local` | diagstats_local.F:6 | 4464 | 37/46 | 112-114, 172, 185, 201, 212, 242-243 | 106:1/2, 111:1/2, 121:1/2, 160:1/2, 173:1/2, 191:1/2, 202:1/2, 237:1/2 |
| `diagstats_output` | diagstats_output.F:8 | 4 | 25/47 | 72, 88-111, 123-125, 142 | 66:1/2, 69:1/2, 78:1/2, 87:1/2, 115:1/2, 116:1/2, 121:1/2, 133:1/2, 139:1/2 |
| `diagstats_set_pointers` | diagstats_set_pointers.F:6 | 1 | 37/84 | 72-78, 82-95, 98-105, 120-140, 150-156, 162-165 | 40:1/2, 68:1/2, 80:1/2, 97:1/2, 112:1/2, 148:1/2, 149:2/2, 161:1/2 |
| `diagstats_set_regions` | diagstats_set_regions.F:6 | 1 | 17/21 | 209-212 | 200:1/2, 203:1/2, 208:1/2 |
| `diagstats_setdiag` | diagstats_setdiag.F:6 | 5 | 20/51 | 70-72, 84-85, 95-97, 104-141 | 65:1/2, 68:1/2, 81:1/2, 82:1/2, 103:1/2 |

**pkg/down_slope** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `dwnslp_readparms` | dwnslp_readparms.F:6 | 1 | 5/32 | 53-117 | 42:1/2, 44:1/2 |

**pkg/ecco** (41)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_averagesfields` | cost_averagesfields.F:3 | 13 | 14/14 | - | 74:1/2, 127:1/2 |
| `cost_averagesflags` | cost_averagesflags.F:4 | 13 | 74/95 | 153, 174-183, 207, 228-239, 263, 285-289 | 152:1/2, 170:1/2, 206:1/2, 224:1/2, 262:1/2, 280:1/2 |
| `cost_averagesgeneric` | cost_averagesgeneric.F:3 | 78 | 38/50 | 113-134, 209 | 111:1/2, 193:1/2 |
| `cost_averagesinit` | cost_averagesinit.F:3 | 1 | 20/20 | - | - |
| `cost_gencal` | cost_gencal.F:7 | 8 | 17/35 | 75-77, 80-98 | 74:1/2, 78:1/2, 112:1/2 |
| `cost_gencost_all` | cost_gencost_all.F:3 | 1 | 21/23 | 54-56 | 53:1/2, 93:1/2, 94:1/2, 95:1/2, 96:1/2 |
| `cost_gencost_assignperiod` | cost_gencost_assignperiod.F:3 | 13 | 12/37 | 58-62, 70-93 | 56:1/2, 63:1/2 |
| `cost_gencost_boxmean` | cost_gencost_boxmean.F:3 | 1 | 42/45 | 125-128 | 121:1/2, 132:1/2 |
| `cost_gencost_bpv4` | cost_gencost_bpv4.F:3 | 1 | 7/104 | 95-361 | 80:1/2, 84:1/2 |
| `cost_gencost_customize` | cost_gencost_customize.F:20 | 13 | 62/97 | 131, 137, 140, 143, 147, 166, 169, 172, 175, 179, 182, 185, 190, 193, 197, 210, 213, 217, 220, 223, 247-250, 253-256, 259-261, 264-266, 269-271 | 129:1/2, 135:1/2, 138:1/2, 141:1/2, 144:1/2, 164:1/2, 167:1/2, 170:1/2, 173:1/2, 177:1/2, 180:1/2, 183:1/2, 188:1/2, 191:1/2, 195:1/2, 207:1/2, 211:1/2, 214:1/2, 218:1/2, 221:1/2, 246:1/2, 252:1/2, 258:1/2, 263:1/2, 268:1/2 |
| `cost_gencost_glbmean` | cost_gencost_glbmean.F:3 | 1 | 2/2 | - | - |
| `cost_gencost_moc` | cost_gencost_moc.F:3 | 1 | 14/89 | 95-270 | 92:1/2 |
| `cost_generic` | cost_generic.F:12 | 4 | 18/18 | - | 101:1/2 |
| `cost_genloop` | cost_generic.F:147 | 4 | 99/125 | 294-295, 298-299, 302, 321-342, 407, 458-464, 501, 511 | 285:1/2, 286:1/2, 287:1/2, 293:1/2, 297:1/2, 301:1/2, 312:1/2, 362:1/2, 370:1/2, 375:1/2, 393:1/2, 406:1/2, 448:1/2, 457:1/2, 495:1/2, 498:1/2, 500:1/2, 505:1/2, 508:1/2, 510:1/2 |
| `cost_genread` | cost_genread.F:7 | 4 | 7/27 | 64-94 | 108:1/2 |
| `ecco_addcost` | ecco_toolbox.F:238 | 4 | 18/20 | 276, 289 | 275:1/2, 286:1/2 |
| `ecco_addmask` | ecco_toolbox.F:414 | 1 | 13/14 | 445 | 444:1/2 |
| `ecco_check` | ecco_check.F:10 | 1 | 71/300 | 248-252, 261-266, 269-274, 277-282, 285-290, 297-301, 320-325, 330-335, 347, 352-358, 365-366, 369-370, 373-374, 380, 382, 384, 388-393, 397-402, 413-432, 446-450, 454-461, 470-476, 482-505, 521-534, 543-679, 698-703, 715-748, 759-763 | 50:1/2, 247:1/2, 260:1/2, 268:1/2, 276:1/2, 284:1/2, 296:1/2, 305:1/2, 318:1/2, 328:1/2, 344:1/2, 348:1/2, 351:1/2, 364:1/2, 368:1/2, 372:1/2, 376:1/2, 379:1/2, 381:1/2, 383:1/2, 386:1/2, 395:1/2, 412:1/2, 445:1/2, 453:1/2, 464:1/2, 469:1/2, 480:1/2, 511:1/2, 540:1/2, 688:1/2, 696:1/2, 714:1/2, 755:1/2, 758:1/2 |
| `ecco_check_files` | ecco_check.F:785 | 4 | 13/36 | 836-841, 852-874, 883-888 | 833:1/2, 845:1/2, 847:1/2, 851:1/2, 879:1/2, 880:1/2, 897:1/2 |
| `ecco_cost_final` | ecco_cost_final.F:6 | 1 | 34/36 | 139-141 | 111:1/2 |
| `ecco_cost_init_barfiles` | ecco_cost_init_barfiles.F:7 | 1 | 28/28 | - | 109:1/2 |
| `ecco_cost_init_fixed` | ecco_cost_init_fixed.F:7 | 1 | 36/48 | 132-133, 138-152, 171 | 83:1/2, 130:1/2, 134:1/2, 168:1/2, 238:1/2 |
| `ecco_cost_init_varia` | ecco_cost_init_varia.F:12 | 1 | 14/14 | - | 68:1/2 |
| `ecco_cp` | ecco_toolbox.F:138 | 8 | 11/12 | 166 | 165:1/2 |
| `ecco_diagnostics_init` | ecco_diagnostics_init.F:8 | 1 | 34/35 | 48 | 47:1/2 |
| `ecco_diffmsk` | ecco_toolbox.F:74 | 4 | 14/15 | 109 | 108:1/2 |
| `ecco_div` | ecco_toolbox.F:471 | 13 | 12/13 | 498 | 497:1/2, 499:1/2 |
| `ecco_divfield` | ecco_toolbox.F:523 | 8 | 11/12 | 547 | 546:1/2 |
| `ecco_init_fixed` | ecco_init_fixed.F:8 | 1 | 6/6 | - | 31:1/2, 36:1/2 |
| `ecco_init_varia` | ecco_init_varia.F:3 | 1 | 5/5 | - | - |
| `ecco_maskmindepth` | ecco_toolbox.F:669 | 1 | 11/12 | 696 | 695:1/2 |
| `ecco_mult` | ecco_toolbox.F:573 | 4 | 10/11 | 599 | 598:1/2 |
| `ecco_offset` | ecco_toolbox.F:720 | 1 | 34/36 | 754, 791 | 775:1/2, 795:1/2, 811:1/2 |
| `ecco_phys` | ecco_phys.F:17 | 13 | 98/126 | 365-389, 411-412, 454-455, 492-499, 555-557, 567, 577-598 | 112:1/2, 364:1/2, 406:1/2, 452:1/2, 491:1/2, 548:1/2, 563:1/2, 572:1/2 |
| `ecco_readbar` | ecco_toolbox.F:817 | 4 | 16/18 | 875-876 | 869:1/2 |
| `ecco_readparms` | ecco_readparms.F:3 | 1 | 402/497 | 204-209, 513-519, 522-525, 528-531, 534-537, 541-548, 552-558, 561-568, 692-693, 709-710, 715-721, 725-731, 738-739, 781-784, 793, 798, 807, 812, 834-836, 839-841, 844-846, 855, 868-875, 880-887, 891-898, 916-923 | 202:1/2, 212:1/2, 511:1/2, 521:1/2, 527:1/2, 533:1/2, 539:1/2, 550:1/2, 560:1/2, 576:1/2, 597:1/2, 690:1/2, 708:1/2, 714:1/2, 724:1/2, 736:1/2, 777:1/2, 791:1/2, 795:1/2, 805:1/2, 809:1/2, 819:1/2, 833:1/2, 838:1/2, 843:1/2, 854:1/2, 865:1/2, 878:1/2, 890:1/2, 902:1/2, 904:1/2, 913:1/2, 934:1/2 |
| `ecco_readwei` | ecco_toolbox.F:912 | 4 | 14/16 | 960, 967 | 959:1/2, 965:1/2 |
| `ecco_summary` | ecco_summary.F:3 | 1 | 65/89 | 76-82, 90-92, 98-101, 111-114, 133-135, 138-140 | 75:1/2, 88:1/2, 97:1/2, 110:1/2, 132:1/2, 137:1/2 |
| `ecco_write_pickup` | ecco_write_pickup.F:6 | 2 | 3/3 | - | - |
| `ecco_zero` | ecco_toolbox.F:30 | 605 | 8/8 | - | - |
| `stergloh_output` | stergloh_output.F:8 | 13 | 2/2 | - | - |

**pkg/exf** (27)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 36 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 12 | 90/101 | 236, 241-242, 325-333, 403, 502-506 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 30/140 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 333-342, 353-357, 399-402, 550-569, 575-578, 616-621, 635-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 625:1/2, 634:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 28/76 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 12 | 26/26 | - | 39:1/2, 41:1/2, 56:1/2 |
| `exf_diagnostics_init` | exf_diagnostics_init.F:6 | 1 | 193/197 | 102, 113, 232, 244 | 101:1/2, 112:1/2, 231:1/2, 243:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 16 | 12/22 | 66-78 | 38:1/2, 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 8 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 12 | 13/14 | 89 | 68:1/2, 88:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 8 | 12/55 | 65-76, 86-92, 106-141 | 80:1/2, 85:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 96 | 18/97 | 101-109, 121-132, 139, 155-190, 206-259 | 97:1/2, 112:1/2, 119:1/2, 138:1/2, 196:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 12 | 60/76 | 97, 147-152, 315-320, 507, 512, 525, 540 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 505:1/2, 510:1/2, 523:1/2, 538:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 12 | 46/53 | 175-181, 202-203, 329 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 12 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 16 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 79/133 | 64-65, 88-114, 149-155, 159-179, 185-191, 195-201, 207-213, 262-268, 282-289, 309-315, 322-329, 359-366, 387-394, 474-481, 489, 590-593 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 137:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 320:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 385:1/2, 472:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 617:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 19 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 41/49 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 12 | 69/95 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 12 | 69/86 | 74, 79-89, 114-116, 164, 186, 192, 204, 210, 222 | 64:1/2, 67:1/2, 71:1/2, 78:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 202:1/2, 208:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 12 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 228 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 12 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 170/208 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 402-408, 487-493, 500-501, 548-555, 571-581, 585-586, 625-626, 636-642, 684-690, 714-720, 761-767, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 633:1/2, 653:1/2, 666:1/2, 681:1/2, 711:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 8 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 12 | 40/83 | 104-115, 158-240 | 79:1/2, 102:1/2, 126:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (15)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 96 | 133/221 | 199-202, 253-266, 279-282, 335-337, 351-364, 394, 414, 418, 423-446, 461, 475-539, 615, 635, 639, 644-667, 682, 696-761, 824-827, 844-845, 848-849, 866, 991, 995, 1000-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 419:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 640:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 2208 | 92/206 | 181-184, 213-216, 236-244, 256-316, 329, 340, 385-445, 458, 469, 515-592, 614, 620, 684-685, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 669:1/2, 695:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diag_sufx` | gad_diagnostics_init.F:324 | 2404 | 6/7 | 368 | 356:1/2 |
| `gad_diagnostics_init` | gad_diagnostics_init.F:6 | 1 | 81/87 | 53, 60, 181-185 | 52:1/2, 59:1/2, 180:1/2 |
| `gad_diagnostics_state` | gad_diagnostics_state.F:8 | 12 | 2/2 | - | - |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 2112 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 2208 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 2208 | 17/17 | - | 82:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 97/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 2 | 3/3 | - | - |

**pkg/gmredi** (18)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 96 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 48 | 147/218 | 166-209, 301-303, 529, 620-627, 689-691, 713-716, 768, 816-837, 851-886, 908-911, 965, 1013-1034, 1048-1083 | 144:1/2, 164:1/2, 291:1/2, 526:1/2, 610:1/2, 688:1/2, 702:1/2, 765:1/2, 815:1/2, 850:1/2, 897:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_diagnostics_fill` | gmredi_diagnostics_fill.F:6 | 48 | 10/14 | 69-70, 78-79 | 52:1/2, 68:1/2, 77:1/2 |
| `gmredi_diagnostics_impl` | gmredi_diagnostics_impl.F:6 | 12 | 4/13 | 51-65 | 48:1/2, 50:1/2 |
| `gmredi_diagnostics_init` | gmredi_diagnostics_init.F:6 | 1 | 109/109 | - | - |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 12 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 21/27 | 63-64, 67-68, 82, 86 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 141:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_mnc_init` | gmredi_mnc_init.F:8 | 1 | 33/33 | - | 27:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 12 | 3/25 | 50-82 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 107/145 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 48 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 2208 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 3264 | 52/87 | 176, 234, 397, 474, 479, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 473:1/2, 478:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 2 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 2208 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 2208 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 49/71 | 96, 121-130, 190-199, 218-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 113:1/2, 115:1/2, 116:1/2, 189:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 87-89, 116-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 54/118 | 116-126, 134, 163, 192-202, 278-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 185:1/2, 186:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `bldepth` | kpp_routines.F:309 | 48 | 86/117 | 494-495, 523-524, 534-549, 691-697, 716-717, 730-744, 830-833, 853-854, 867-881 | 486:1/2, 491:1/2, 532:1/2, 639:1/2, 685:1/2, 690:1/2, 728:1/2, 781:1/2, 824:1/2, 829:1/2, 865:1/2 |
| `blmix` | kpp_routines.F:1395 | 48 | 72/72 | - | - |
| `enhance` | kpp_routines.F:1696 | 48 | 11/11 | - | 1741:1/2 |
| `kpp_calc` | kpp_calc.F:19 | 48 | 75/88 | 537, 644-649, 682-701 | 223:1/2, 495:1/2, 528:1/2, 636:1/2, 641:1/2, 680:1/2 |
| `kpp_calc_diff_s` | kpp_calc_diff_s.F:3 | 48 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_diff_t` | kpp_calc_diff_t.F:3 | 48 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_visc` | kpp_calc_visc.F:3 | 1104 | 8/8 | - | - |
| `kpp_check` | kpp_check.F:3 | 1 | 53/62 | 130-132, 137-139, 142-144 | 28:1/2, 128:1/2, 136:1/2, 141:1/2 |
| `kpp_diagnostics_init` | kpp_diagnostics_init.F:9 | 1 | 105/105 | - | - |
| `kpp_do_exch` | kpp_do_exch.F:6 | 12 | 3/3 | - | - |
| `kpp_forcing_surf` | kpp_forcing_surf.F:10 | 48 | 51/54 | 263-272 | 233:1/2, 247:1/2, 281:1/2, 507:1/2 |
| `kpp_init_fixed` | kpp_init_fixed.F:6 | 1 | 64/64 | - | 45:1/2, 116:1/2, 162:1/2, 187:1/2 |
| `kpp_init_varia` | kpp_init_varia.F:7 | 1 | 17/17 | - | - |
| `kpp_output` | kpp_output.F:11 | 13 | 15/50 | 118-181, 196-220 | 105:1/2, 114:1/2, 195:1/2 |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 73/112 | 61-66, 171-176, 196-199, 202-205, 208-211, 214-217, 220-226, 230-238 | 59:1/2, 69:1/2, 169:1/2, 195:1/2, 201:1/2, 207:1/2, 213:1/2, 219:1/2, 229:1/2 |
| `kpp_transport_s` | kpp_transport_s.F:9 | 1056 | 9/11 | 75, 89 | 73:1/2, 86:1/2 |
| `kpp_transport_t` | kpp_transport_t.F:6 | 1056 | 8/9 | 72 | 69:1/2 |
| `kppmix` | kpp_routines.F:28 | 48 | 17/17 | - | - |
| `ri_iwmix` | kpp_routines.F:1032 | 48 | 39/42 | 1199-1201 | 1198:1/2 |
| `smooth_horiz` | kpp_routines.F:1311 | 1056 | 15/15 | - | - |
| `statekpp` | kpp_routines.F:1766 | 48 | 32/32 | - | 1949:1/2 |
| `wscale` | kpp_routines.F:923 | 2256 | 28/28 | - | - |

**pkg/mdsio** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 49 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 1 | 13/36 | 60, 78-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 251 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_rd_rec_rl` | mdsio_rd_rec_rl.F:8 | 2 | 10/18 | 49-51, 55-60, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 138 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 487, 535-538, 543, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 530:1/2, 540:1/2, 541:1/2, 544:1/2 |
| `mds_readvec_loc` | mdsio_readvec_loc.F:6 | 2 | 34/79 | 94-99, 106-117, 122-123, 129-131, 151-165, 174-176, 183-198, 206, 212-214, 221 | 92:1/2, 104:1/2, 120:1/2, 124:1/2, 128:1/2, 141:1/2, 149:1/2, 171:1/2, 172:1/2, 203:1/2, 204:1/2, 207:1/2, 219:1/2, 220:1/2, 222:1/2, 231:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 6 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 2 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 167 | 86/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 376, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 374:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 519 | 53/64 | 108, 135-139, 146-147, 157-159, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 3 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mnc** (47)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mnc_cw_add_gname` | mnc_cwrapper.F:25 | 298 | 20/24 | 61-64 | 60:1/2, 70:2/4, 77:2/4 |
| `mnc_cw_add_vattr_any` | mnc_cwrapper.F:486 | 257 | 24/52 | 523-526, 536-543, 554-560, 574-580, 585-591 | 522:1/2, 529:1/2, 534:1/2, 548:2/4, 552:1/2, 567:1/2, 568:2/4, 572:1/2, 583:1/2 |
| `mnc_cw_add_vattr_text` | mnc_cwrapper.F:409 | 257 | 6/6 | - | - |
| `mnc_cw_add_vname` | mnc_cwrapper.F:290 | 129 | 19/27 | 332-335, 343-346 | 331:1/2, 342:1/2, 350:2/4 |
| `mnc_cw_citer_setg` | mnc_cw_citer.F:9 | 13 | 6/11 | 31-41 | 29:1/2 |
| `mnc_cw_file_aorc` | mnc_cwrapper.F:741 | 208 | 10/10 | - | 773:1/2 |
| `mnc_cw_get_tile_num` | mnc_cwrapper.F:602 | 208 | 7/7 | - | - |
| `mnc_cw_i_w` | MNC_CW_READWRITE_I.F:50 | 4 | 6/6 | - | - |
| `mnc_cw_i_w_offset` | MNC_CW_READWRITE_I.F:86 | 4 | 159/306 | 163-166, 177-180, 237, 240-251, 276-336, 352, 369, 372, 394-397, 409, 419, 421, 426-427, 447-454, 459-460, 463-464, 470-472, 497-512, 516-528, 587-591, 596-631, 637-642, 644-649 | 161:1/2, 176:1/2, 187:1/2, 193:1/2, 234:1/2, 236:1/2, 258:1/2, 261:2/4, 262:2/4, 273:1/2, 349:1/2, 367:1/2, 370:1/2, 373:1/2, 385:1/2, 408:1/2, 415:1/2, 418:1/2, 420:1/2, 425:1/2, 444:1/2, 458:1/2, 462:1/2, 469:1/2, 482:1/2, 486:1/2, 494:1/2, 515:1/2, 586:1/2, 594:1/2, 636:1/2, 643:1/2, 650:1/2, 682:1/2 |
| `mnc_cw_i_w_s` | MNC_CW_READWRITE_I.F:18 | 4 | 5/5 | - | - |
| `mnc_cw_init` | mnc_cw_init.F:8 | 1 | 106/111 | 212-216 | 99:2/4, 103:2/4, 111:2/4, 211:1/2 |
| `mnc_cw_rl_w` | MNC_CW_READWRITE_RL.F:50 | 18 | 6/6 | - | - |
| `mnc_cw_rl_w_offset` | MNC_CW_READWRITE_RL.F:86 | 18 | 191/306 | 163-166, 177-180, 237, 240-251, 301-336, 375, 394-397, 426-427, 447-454, 470-472, 497-512, 516-528, 587-591, 596-631, 650-656 | 161:1/2, 176:1/2, 187:1/2, 193:1/2, 234:1/2, 236:1/2, 258:1/2, 261:2/4, 262:2/4, 281:1/2, 284:1/2, 294:1/2, 298:1/2, 373:1/2, 385:1/2, 425:1/2, 444:1/2, 469:1/2, 482:1/2, 486:1/2, 494:1/2, 515:1/2, 586:1/2, 594:1/2, 643:1/2, 682:1/2 |
| `mnc_cw_rl_w_s` | MNC_CW_READWRITE_RL.F:18 | 4 | 5/5 | - | - |
| `mnc_cw_rs_w` | MNC_CW_READWRITE_RS.F:50 | 30 | 6/6 | - | - |
| `mnc_cw_rs_w_offset` | MNC_CW_READWRITE_RS.F:86 | 30 | 159/306 | 163-166, 177-180, 235-239, 247-251, 276-336, 350, 372, 375, 382, 394-397, 411, 426-427, 444-454, 470-472, 497-512, 516-528, 587-591, 596-631, 643-656 | 161:1/2, 176:1/2, 187:1/2, 193:1/2, 234:1/2, 240:1/2, 258:1/2, 261:2/4, 262:2/4, 273:1/2, 349:1/2, 367:1/2, 370:1/2, 373:1/2, 377:1/2, 385:1/2, 408:1/2, 425:1/2, 443:1/2, 469:1/2, 476:1/2, 482:1/2, 486:1/2, 494:1/2, 515:1/2, 586:1/2, 594:1/2, 636:1/2, 682:1/2 |
| `mnc_cw_set_citer` | mnc_cw_citer.F:88 | 1 | 11/20 | 121-124, 128, 134-137 | 118:1/2, 127:1/2, 132:1/2 |
| `mnc_cw_set_gattr` | mnc_cw_model_attr.F:8 | 16 | 27/28 | 46 | 44:1/2 |
| `mnc_cw_set_udim` | mnc_cw_udim.F:8 | 10 | 21/26 | 70-74 | 51:2/4, 64:1/2 |
| `mnc_cw_write_cvar` | mnc_cw_cvars.F:8 | 88 | 131/237 | 92-93, 102-103, 105-106, 110-115, 128-129, 138-139, 141-142, 146-151, 156-189, 202-203, 212-213, 215-216, 220-225, 238-239, 248-249, 251-252, 256-261, 266-298, 313-317, 332-336, 351-356, 371-376, 381-396 | 68:1/2, 91:1/2, 101:1/2, 104:1/2, 107:1/2, 127:1/2, 137:1/2, 140:1/2, 143:1/2, 154:1/2, 201:1/2, 211:1/2, 214:1/2, 217:1/2, 237:1/2, 247:1/2, 250:1/2, 253:1/2, 264:1/2, 310:1/2, 329:1/2, 347:1/2, 367:1/2, 379:1/2, 421:1/2, 429:1/2 |
| `mnc_dim_init_all_cv` | mnc_dim.F:61 | 472 | 31/42 | 100-103, 114-123 | 99:1/2, 113:1/2, 136:1/2, 149:1/2, 157:2/4 |
| `mnc_dim_unlim_size` | mnc_dim.F:171 | 16 | 17/21 | 202-205 | 201:1/2, 213:1/2 |
| `mnc_file_add_attr_any` | mnc_file.F:232 | 336 | 21/33 | 265-269, 292-295, 302-306 | 264:1/2, 278:2/4, 291:1/2, 296:1/2 |
| `mnc_file_add_attr_dbl` | mnc_file.F:142 | 16 | 5/5 | - | - |
| `mnc_file_add_attr_int` | mnc_file.F:202 | 224 | 5/5 | - | - |
| `mnc_file_add_attr_str` | mnc_file.F:113 | 96 | 5/5 | - | - |
| `mnc_file_enddef` | mnc_file.F:543 | 284 | 11/16 | 571-575 | 570:1/2, 580:1/2 |
| `mnc_file_open` | mnc_file.F:36 | 16 | 18/28 | 69-72, 82-92 | 68:1/2, 76:1/2, 98:2/4 |
| `mnc_file_redef` | mnc_file.F:490 | 1144 | 11/16 | 518-522 | 517:1/2 |
| `mnc_file_try_read` | mnc_file.F:635 | 16 | 9/60 | 676-751 | 671:1/2 |
| `mnc_get_fvinds` | mnc_utils.F:186 | 244 | 15/18 | 219-220, 236 | 218:1/2, 225:1/2 |
| `mnc_get_ind` | mnc_utils.F:71 | 4064 | 13/17 | 102-105 | 100:1/2 |
| `mnc_get_next_empty_ind` | mnc_utils.F:128 | 795 | 9/17 | 167-175 | 157:1/2 |
| `mnc_grid_init` | mnc_grid.F:8 | 208 | 5/5 | - | - |
| `mnc_grid_init_all` | mnc_grid.F:40 | 208 | 58/85 | 87-90, 106-109, 130-136, 144-150, 193-197 | 83:1/2, 84:2/4, 86:1/2, 105:1/2, 129:1/2, 143:1/2, 170:2/4, 182:1/2, 192:1/2 |
| `mnc_handle_err` | mnc_utils.F:17 | 13220 | 6/20 | 48-61 | 47:1/2 |
| `mnc_init` | mnc_init.F:8 | 1 | 62/62 | - | - |
| `mnc_psncm` | mnc_utils.F:328 | 208 | 13/13 | - | 353:1/2, 360:1/2, 361:2/4 |
| `mnc_readparms` | mnc_readparms.F:13 | 1 | 52/85 | 57-62, 69-82, 145-152, 182-186, 194-197, 200-203, 206-209 | 55:1/2, 68:1/2, 144:1/2, 156:1/2, 168:1/2, 170:1/2, 172:1/2, 175:1/2, 178:1/2, 179:1/2, 193:1/2, 199:1/2, 205:1/2 |
| `mnc_set_outdir` | mnc_readparms.F:220 | 1 | 31/33 | 253, 282 | 252:1/2, 261:1/2, 262:2/4, 265:1/2, 280:1/2, 281:1/2 |
| `mnc_update_time` | mnc_update_time.F:8 | 12 | 4/6 | 51-53 | 50:1/2 |
| `mnc_var_add_attr_any` | mnc_var.F:405 | 244 | 17/33 | 449-453, 462-473 | 448:1/2, 460:1/2 |
| `mnc_var_add_attr_str` | mnc_var.F:264 | 244 | 7/7 | - | - |
| `mnc_var_init_any` | mnc_var.F:114 | 208 | 46/81 | 156-160, 167-171, 177-180, 199-202, 212-216, 231-242 | 155:1/2, 166:1/2, 176:1/2, 182:1/2, 211:1/2, 230:1/2, 248:2/4 |
| `mnc_var_init_dbl` | mnc_var.F:27 | 136 | 4/4 | - | - |
| `mnc_var_init_int` | mnc_var.F:85 | 16 | 4/4 | - | - |
| `mnc_var_init_real` | mnc_var.F:56 | 56 | 4/4 | - | - |

**pkg/mom_common** (6)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 1104 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 1104 | 12/26 | 62-66, 77-84, 90-97, 115-137 | 61:1/2, 70:1/2, 88:1/2, 101:1/2 |
| `mom_diagnostics_init` | mom_diagnostics_init.F:6 | 1 | 361/362 | 520 | 519:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 39/41 | 51-52 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 1104 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 1104 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |

**pkg/mom_fluxform** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_rtrans` | mom_calc_rtrans.F:6 | 1152 | 11/11 | - | - |
| `mom_fluxform` | mom_fluxform.F:42 | 1104 | 186/276 | 265, 276, 331-363, 457, 514-523, 592-594, 607, 622-623, 651, 663-666, 738-748, 764-767, 774-776, 888-890, 902, 917-918, 946, 958-961, 1033-1042, 1058-1061, 1068-1070, 1092-1106, 1113-1124 | 257:1/2, 264:1/2, 270:1/2, 330:1/2, 422:1/2, 444:1/2, 453:1/2, 476:1/2, 510:1/2, 512:1/2, 546:1/2, 601:1/2, 605:1/2, 621:1/2, 647:1/2, 650:1/2, 656:1/2, 671:1/2, 689:1/2, 735:1/2, 752:1/2, 753:1/2, 762:1/2, 772:1/2, 773:1/2, 789:1/2, 822:1/2, 842:1/2, 897:1/2, 900:1/2, 916:1/2, 942:1/2, 945:1/2, 951:1/2, 966:1/2, 984:1/2, 1030:1/2, 1046:1/2, 1047:1/2, 1056:1/2, 1066:1/2, 1067:1/2, 1082:1/2, 1112:1/2, 1141:1/2 |
| `mom_u_adv_uu` | mom_u_adv_uu.F:7 | 1104 | 5/5 | - | - |
| `mom_u_adv_vu` | mom_u_adv_vu.F:7 | 1104 | 6/9 | 65-74 | 46:1/2 |
| `mom_u_adv_wu` | mom_u_adv_wu.F:7 | 1152 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_u_metric_sphere` | mom_u_metric_sphere.F:7 | 1104 | 6/9 | 59-73 | 46:1/2 |
| `mom_u_xviscflux` | mom_u_xviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_u_yviscflux` | mom_u_yviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_v_adv_uv` | mom_v_adv_uv.F:7 | 1104 | 5/5 | - | - |
| `mom_v_adv_vv` | mom_v_adv_vv.F:7 | 1104 | 5/5 | - | - |
| `mom_v_adv_wv` | mom_v_adv_wv.F:7 | 1152 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_v_metric_sphere` | mom_v_metric_sphere.F:7 | 1104 | 6/9 | 60-76 | 44:1/2 |
| `mom_v_xviscflux` | mom_v_xviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_v_yviscflux` | mom_v_yviscflux.F:10 | 1104 | 5/5 | - | - |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 26 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 13 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 13 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 12 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 48 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 270 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 65 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 13 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 2056 | 52/57 | 213-219 | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2, 211:1/2 |
| `mon_out_i` | mon_out.F:8 | 25 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 2031 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 131 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 13 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 13 | 39/50 | 95, 123-132, 178-179, 196-198 | 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 13 | 74/127 | 145-237, 244-260, 263-278 | 143:1/2, 242:1/2, 243:1/2, 262:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 270 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 65 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 13 | 65/76 | 57, 62-72, 119-120 | 48:1/2, 50:1/2, 54:1/2, 61:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 40 | 7/7 | - | - |
| `read_rec_lev_rl` | read_rec.F:445 | 8 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 2 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 90 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 26 | 7/7 | - | - |
| `write_rec_xy_rl` | write_rec.F:145 | 3 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 2 | 7/7 | - | - |

**pkg/salt_plume** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `salt_plume_diagnostics_fill` | salt_plume_diagnostics_fill.F:6 | 12 | 5/5 | - | 28:1/2 |
| `salt_plume_readparms` | salt_plume_readparms.F:6 | 1 | 5/32 | 56-124 | 46:1/2, 48:1/2 |

**pkg/seaice** (3)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 48/160 | 54, 171-471 | 53:1/2, 165:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 5/832 | 241-1559 | 231:1/2, 233:1/2 |

### Compiled but not executed (867 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_r_star, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_rho_l, diags_sound_speed, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, freesurf_rescale_g, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cartesian_grid, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_daysformonth, cal_dayspermonth, cal_getmonthsrec, cal_monthsforyear, cal_monthsperyear, cal_printdate, cal_printerror, cal_stepsforday, cal_stepsperday, cal_timestamp, cal_weekday
- pkg/cd_code: cd_code_read_pickup
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_calc_phivel, diagnostics_fract_fill, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_write_adj, diags_get_parms_i, diagstats_g_calc, diagstats_lm_calc, diagstats_mnc_out
- pkg/down_slope: dwnslp_apply, dwnslp_calc_flow, dwnslp_calc_rho, dwnslp_diagnostics_init, dwnslp_init_fixed, dwnslp_init_varia
- pkg/ecco: cost_bp_read, cost_gencost_seaicev4, cost_gencost_sshv4, cost_gencost_sstv4, cost_gencost_transp, cost_sla_read, cost_sla_read_yd, ecco_add, ecco_cprsrl, ecco_error, ecco_multfield, ecco_read_pickup, ecco_subtract, get_exconc_deconc
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: kpp_calc_diff_ptr, kpp_calc_dummy, kpp_doublediff, kpp_transport_ptr, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mnc: mnc_chk_vtyp_r_ncvar, mnc_cw_add_vattr_dbl, mnc_cw_add_vattr_int, mnc_cw_append_vname, mnc_cw_citer_getg, mnc_cw_del_gname, mnc_cw_del_vname, mnc_cw_dump, mnc_cw_get_citer, mnc_cw_get_face_num, mnc_cw_get_udim, mnc_cw_get_xyfo, mnc_cw_i_r, mnc_cw_i_r_s, mnc_cw_i_r_tf, mnc_cw_rl_r, mnc_cw_rl_r_s, mnc_cw_rl_r_tf, mnc_cw_rs_r, mnc_cw_rs_r_s, mnc_cw_rs_r_tf, mnc_cw_rs_w_s, mnc_cw_vattr_missing, mnc_cw_write_grid_coord, mnc_cw_write_grid_info, mnc_dim_init, mnc_dim_init_all, mnc_dump, mnc_dump_all, mnc_file_add_attr_real, mnc_file_close, mnc_file_close_all, mnc_file_close_all_matching, mnc_file_create, mnc_file_readall, mnc_grid_get_dimind, mnc_var_add_attr_dbl, mnc_var_add_attr_int, mnc_var_add_attr_real, mnc_var_append_dbl, mnc_var_append_int, mnc_var_append_real, mnc_var_write_any, mnc_var_write_dbl, mnc_var_write_int, mnc_var_write_real
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_uv_boundary, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_fld_xy_rl, write_fld_xy_rs, write_fld_xyz_rl, write_fld_xyz_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/salt_plume: salt_plume_apply, salt_plume_calc_depth, salt_plume_check, salt_plume_diagnostics_init, salt_plume_do_exch, salt_plume_forcing_surf, salt_plume_frac, salt_plume_init_fixed, salt_plume_init_varia, salt_plume_mnc_init, salt_plume_tendency_apply_s, salt_plume_tendency_apply_t, salt_plume_volfrac
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advdiff, seaice_advection, seaice_bottomdrag_coeffs, seaice_budget_ocean, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stress, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check, seaice_check_pickup, seaice_cost_accumulate_mean, seaice_cost_export, seaice_cost_final, seaice_cost_init_fixed, seaice_cost_sensi, seaice_cost_test, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_dynsolver, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_get_dynforcing, seaice_growth, seaice_growth_adx, seaice_init_fixed, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_lsr_calc_coeffs, seaice_lsr_rhsu, seaice_lsr_rhsv, seaice_lsr_tridiagu, seaice_lsr_tridiagv, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_model, seaice_mom_advection, seaice_monitor, seaice_obcs_output, seaice_ocean_stress, seaice_oceandrag_coeffs, seaice_output, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_reg_ridge, seaice_residual, seaice_scalprod, seaice_sidedrag_stress, seaice_solve4temp, seaice_summary, seaice_tracer_phys, seaice_turnoff_io, seaice_write_pickup

## lab_sea/input_ad.noseaicedyn

Run `$MJX_REFERENCE/coverage/lab_sea/input_ad.noseaicedyn/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/lab_sea/input_ad.noseaicedyn/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice volume changes | 13 blocks, first 1.000000e+00, last 1.779325e-01, min 1.779325e-01, max 1.000000e+00 | yes |
| sea-ice thermodynamics | seaice_growth (12 calls) | yes |
| salt plume | salt_plume_calc_depth (48 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (12 calls) | yes |
| cost function | global fc = 10238.9008218131; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

(parameter files could not be read by mitjax.config: ValueError: $MJX_UPSTREAM/verification/lab_sea/input_ad/data.err:1: text outside a namelist group: '0.25\n    0.5201    0.2676\n    '; all printed switches are listed)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, doSaltClimRelax, dumpInitAndLast, fluidIsWater, implicitDiffusion, implicitFreeSurface, implicitViscosity, inAdExact, KPP_ghatUseTotalDiffus, KPPwriteState, LimitHblStable, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICEadvArea, SEAICEadvHeff, SEAICEadvSnow, SEAICEupdateOceanStress, SEAICEuseFlooding, SEAICEwriteState, snapshot_mdsio, staggerTimeStep, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCDscheme, useCoriolis, useCtrlCostContribution, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useKPPinAdMode, useMaykutSatVapPoly, useMultiDimAdvec, usePW79thermodynamics, useSALT_PLUMEinAdMode, useSEAICEinAdMode, usingGregorianCalendar, usingSphericalPolarGrid, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, doResetHFactors, doThetaClimRelax, exactConserv, fluidIsAir, globalFiles, GM_AdvForm, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, implicitIntGravWave, interDiffKr_pCell, interViscAr_pCell, KPPuseDoubleDiff, KPPuseSWfrac3D, linFSConserveTr, momImplVertAdv, monitor_mnc, no_slip_sides, noNegativeEvap, nonHydrostatic, pickup_read_mnc, pickup_write_mnc, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICE_doOpenWaterGrowth, SEAICE_doOpenWaterMelt, SEAICE_dump_mnc, SEAICE_growMeltByConv, SEAICE_mcPheeStepFunc, SEAICE_mon_mnc, SEAICE_salinityTracer, SEAICE_useMultDimSnow, SEAICEadvSalt, SEAICEheatConsFix, SEAICEmomAdvection, SEAICEmultiDimAdvection, SEAICErestoreUnderIce, SEAICEuseBDF2, SEAICEuseDYNAMICS, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFluxForm, SEAICEuseFREEDRIFTswitchInAd, snapshot_mnc, stressIsOnCgrid, tempImplVertAdv, tempSOM_Advection, twoDigitYear, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCurvilinearGrid, usingCylindricalGrid, usingJulianCalendar, usingModelCalendar, usingMPI, usingNoLeapYearCal, usingPCoords, vectorInvariantMomentum
- selectors: exf_adjMonSelect=3, monitorSelect=3, pCellMix_select=0, saltAdvScheme=30, saltVertAdvScheme=30, SEAICEadvScheme=2, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempAdvScheme=30, tempVertAdvScheme=30

### Executed routines (504)

**eesupp/src** (78)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 11042 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 7422 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 209 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 56/77 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1478 | 27/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 312 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 12 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1790 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1478 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1478 | 38/90 | 289-324, 346-381 | 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1478 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1478 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 312 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 312 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 312 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 312 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 1054 | 11/12 | 60 | 56:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 12 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 24 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_dgrid_3d_rl` | exch_uv_dgrid_3d_rl.F:8 | 12 | 14/32 | 88-177 | 72:1/2, 74:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 1 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 41 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 273 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 228 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 41 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 33308 | 2/2 | - | - |
| `gather_2d_r8` | gather_2d_r8.F:7 | 1593 | 14/14 | - | 61:1/2, 63:1/2 |
| `global_max_r8` | global_max.F:97 | 1008 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 73 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 120 | 12/12 | - | 154:1/2 |
| `global_sum_singlecpu_rl` | global_sum_singlecpu.F:15 | 1593 | 22/22 | - | 98:1/2, 107:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 2316 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 9074 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 24995 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 26 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 2896 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1242 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 1068 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 7160 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 385 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 15 | 30/40 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 62 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 153 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 286 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 5907 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 668 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 668 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 668 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 335 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 333 | 4/4 | - | - |
| `ucase` | utils.F:311 | 1338 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 10 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 52 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 153 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 240 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 45 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (103)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 2208 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 1104 | 14/25 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2, 951:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 1104 | 20/55 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2, 719:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 1104 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 1104 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 192 | 26/32 | 164-166, 195-197 | 132:1/2, 180:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 2208 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 1104 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 1104 | 19/19 | - | - |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 48 | 8/8 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 48 | 6/81 | 76, 81, 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 1104 | 36/162 | 149, 184, 212-240, 268-610, 636-648 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 48 | 12/16 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 12 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `config_check` | config_check.F:14 | 1 | 96/545 | 92-97, 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 986-989, 992-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 985:1/2, 991:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 443/541 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 533-538, 554-588, 756-762, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 48 | 24/57 | 79, 83, 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 96 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 1104 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 1104 | 24/32 | 79-84, 123-125 | 61:1/2, 76:1/2, 121:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 12 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 12 | 10/16 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 12 | 114/151 | 257-264, 467-471, 557, 759-784, 797-798, 830-833, 869-874, 882, 934, 962, 1043, 1054, 1116, 1119, 1123, 1127 | 248:1/2, 251:1/2, 256:1/2, 421:1/2, 450:1/2, 546:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 734:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 867:1/2, 878:1/2, 896:1/2, 903:1/2, 932:1/2, 951:1/2, 953:1/2, 1030:1/2, 1032:1/2, 1049:1/2, 1051:1/2, 1096:1/2, 1102:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1126:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 12 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 13 | 14/21 | 97-110, 242 | 96:1/2, 116:1/2, 144:1/2, 187:1/2, 193:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 12 | 23/26 | 84, 115-117 | 66:1/2, 83:1/2, 94:1/2, 99:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 12 | 58/84 | 337-340, 358, 533, 684-700, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 615:1/2, 682:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 12 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 12 | 43/69 | 75, 151-153, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 247:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_alpha` | find_alpha.F:14 | 1104 | 29/76 | 77-79, 85-99, 222-337 | 75:1/2, 83:1/2, 113:1/2 |
| `find_beta` | find_alpha.F:347 | 1104 | 27/74 | 410-412, 418-430, 539-641 | 408:1/2, 416:1/2, 444:1/2 |
| `find_bulkmod` | find_rho.F:412 | 7676 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 5468 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 67 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 7676 | 16/16 | - | - |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 12 | 12/21 | 64, 93-104, 245-252 | 63:1/2, 75:1/2, 242:1/2 |
| `forward_step` | forward_step.F:70 | 12 | 80/102 | 508-512, 730-734, 767-769, 813, 980, 989-991, 1109-1111, 1176, 1201-1202 | 424:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 812:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1169:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 1104 | 20/22 | 61, 74 | 59:1/2, 72:1/2 |
| `impldiff` | impldiff.F:7 | 288 | 65/102 | 286-298, 304-382 | 199:1/2, 219:1/2, 284:1/2, 303:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/71 | 57-63, 70-79, 106-112, 121-180, 191, 196-201 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 195:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/76 | 63-66, 94-98, 107-114, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 106:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 57/87 | 52, 60, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/121 | 131, 134-144, 192, 197-202 | 130:1/2, 132:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 171:1/2, 175:1/2, 185:1/2, 189:1/2, 196:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 177/193 | 210-212, 247-261, 435, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 35/58 | 146-179, 191-195, 219-227 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 206:1/2, 217:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 404/881 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 584, 586-589, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1351-1355, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 585:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1116:1/2, 1117:1/2, 1118:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 20/22 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 26/45 | 68-73, 100, 109-124, 130 | 65:1/2, 67:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_spherical_polar_grid` | ini_spherical_polar_grid.F:8 | 1 | 75/85 | 258-263, 277-280 | 119:1/2, 224:1/2, 237:1/2, 257:1/2, 276:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 27/54 | 69-74, 101, 110-125, 131-138, 151 | 66:1/2, 68:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 13 | 14/65 | 92-185, 211-221, 279-282, 329-331 | 90:1/2, 207:1/2, 277:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 1196 | 17/28 | 95-117 | 93:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 12 | 26/34 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 27/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161, 173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 12 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 12 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 110/124 | 242-261 | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 64/98 | 207, 212, 280, 289, 304, 321, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 252:1/2, 260:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 339:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 428:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 46/52 | 170-176, 675-677 | 144:1/2, 158:1/2, 160:1/2, 167:1/2, 193:1/2, 200:1/2, 202:1/2, 209:1/2, 211:1/2, 249:1/2, 251:1/2, 317:1/2, 319:1/2, 327:1/2, 329:1/2, 347:1/2, 349:1/2, 358:1/2, 365:1/2, 367:1/2, 374:1/2, 379:1/2, 510:1/2, 512:1/2, 519:1/2, 521:1/2, 651:1/2, 655:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 28/31 | 169, 174, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 204:1/2, 206:1/2, 251:1/2, 253:1/2, 270:1/2, 285:1/2, 291:1/2, 293:1/2, 435:1/2, 444:1/2, 601:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 24 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 17/17 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 2 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 2 | 14/15 | 225 | 103:1/2, 124:1/2, 184:1/2, 223:1/2, 237:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 7676 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 38 | 50/64 | 67-70, 78, 100-101, 168-174 | 66:1/2, 77:1/2, 99:1/2, 111:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 48 | 58/70 | 167, 169, 192, 326, 367-369, 386-390, 454, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 444:1/2, 448:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 96/166 | 56-57, 60-63, 67, 71, 76, 109-115, 120-126, 171-175, 186-189, 192-195, 202, 208, 214, 220, 226, 232, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 12 | 59/94 | 143-147, 152-157, 164, 170-176, 218-224, 265, 269, 319, 327, 342-344, 355-366, 371 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1201 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 48 | 58/70 | 169, 171, 194, 328, 369-371, 388-392, 456, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 446:1/2, 450:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 53/53 | - | 292:1/2, 382:1/2, 454:1/2, 472:1/2, 664:1/2, 666:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/45 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 12 | 44/52 | 158, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 1104 | 58/94 | 118-121, 140-143, 189-190, 220-223, 328-332, 392-410, 413-414, 417-418, 422-423 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 229:1/2, 321:1/2, 391:1/2, 412:1/2, 416:1/2, 421:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 96 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 12 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/20 | - | 55:1/2, 78:1/2, 88:1/2, 106:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 47/110 | 74, 98-101, 106-111, 117, 119-121, 133-140, 164-208, 217-219 | 73:1/2, 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2, 162:1/2, 214:1/2 |
| `write_pickup` | write_pickup.F:8 | 2 | 52/100 | 253-257, 261-265, 288-290, 335-338, 365-371, 391-438 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2, 390:1/2 |
| `write_state` | write_state.F:11 | 13 | 19/46 | 87, 126, 177-209 | 84:1/2, 95:1/2, 123:1/2, 175:1/2 |

**pkg/autodiff** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 85 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 78 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 7 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 41 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 38 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 3 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 3 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 12 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 12 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 18/145 | 69-83, 123-403 | 66:1/2, 68:1/2, 121:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 81/148 | 63-68, 127-132, 157, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 12 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 12 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 26 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 13 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 12 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 12 | 2/2 | - | - |

**pkg/cal** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_addtime` | cal_addtime.F:3 | 477 | 47/107 | 66-75, 79-81, 89-106, 114-116, 127-140, 169-171, 179, 181-186, 192-199 | 65:1/2, 78:1/2, 88:1/2, 110:1/2, 124:1/2, 167:1/2, 168:1/2, 177:1/2, 180:1/2, 191:1/2, 200:2/2, 214:1/2, 218:1/2 |
| `cal_checkdate` | cal_checkdate.F:3 | 47 | 22/54 | 61-63, 65-70, 76-81, 89-93, 96, 100-102, 106-108, 111, 123-126, 129-132, 135-138 | 59:1/2, 64:1/2, 73:1/2, 88:1/2, 94:1/2, 98:1/2, 103:1/2, 109:1/2, 116:1/2, 122:1/2, 128:1/2, 134:1/2 |
| `cal_compdates` | cal_compdates.F:3 | 26 | 5/5 | - | - |
| `cal_convdate` | cal_convdate.F:3 | 1220 | 21/37 | 51-57, 66-68, 71-73, 87-89 | 50:1/2, 65:1/2, 70:1/2, 82:1/2 |
| `cal_copydate` | cal_copydate.F:3 | 169 | 6/6 | - | - |
| `cal_fulldate` | cal_fulldate.F:3 | 47 | 15/37 | 59-65, 71-74, 87-93, 98-101 | 58:1/2, 70:1/2, 77:1/2, 84:1/2 |
| `cal_getdate` | cal_getdate.F:3 | 874 | 16/23 | 57-63 | 55:1/2 |
| `cal_init_fixed` | cal_init_fixed.F:8 | 1 | 6/6 | - | 28:1/2, 42:1/2 |
| `cal_intdays` | cal_intdays.F:3 | 2 | 9/10 | 58 | 55:1/2 |
| `cal_intmonths` | cal_intmonths.F:3 | 2 | 9/11 | 62, 69 | 59:1/2, 67:1/2 |
| `cal_intyears` | cal_intyears.F:3 | 2 | 4/5 | 49 | 47:1/2 |
| `cal_isleap` | cal_isleap.F:3 | 20099 | 9/21 | 41-47, 60-67 | 40:1/2, 50:1/2 |
| `cal_numints` | cal_numints.F:3 | 3 | 7/10 | 61-63 | 53:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 19/23 | 70-76 | 65:1/2, 68:1/2, 79:1/2, 114:1/2 |
| `cal_set` | cal_set.F:3 | 1 | 60/93 | 104-114, 163-184, 202-204, 207-209, 212-214 | 82:1/2, 99:1/2, 119:1/2, 160:1/2, 201:1/2, 206:1/2, 211:1/2 |
| `cal_subdates` | cal_subdates.F:3 | 2 | 6/24 | 48-57, 65-69, 77-79 | 47:1/2, 60:1/2, 63:1/2 |
| `cal_summary` | cal_summary.F:3 | 1 | 43/43 | - | 49:1/2 |
| `cal_time2dump` | cal_time2dump.F:3 | 24 | 3/15 | 36-51 | 31:1/2 |
| `cal_timeinterval` | cal_timeinterval.F:3 | 489 | 15/36 | 60-66, 73-92 | 57:1/2, 59:1/2 |
| `cal_timepassed` | cal_timepassed.F:3 | 501 | 52/75 | 67-76, 98, 120-127, 148-149, 182-184 | 66:1/2, 87:1/2, 97:1/2, 119:1/2, 147:1/2 |
| `cal_toseconds` | cal_toseconds.F:3 | 455 | 15/26 | 57-63, 69, 92-94 | 56:1/2, 67:1/2, 73:1/2 |

**pkg/cd_code** (4)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cd_code_ini_vars` | cd_code_ini_vars.F:3 | 1 | 15/16 | 53 | 52:1/2 |
| `cd_code_init_fixed` | cd_code_init_fixed.F:3 | 1 | 3/23 | 25-67 | 22:1/2 |
| `cd_code_scheme` | cd_code_scheme.F:7 | 1104 | 47/49 | 82-83 | 78:1/2 |
| `cd_code_write_pickup` | cd_code_write_pickup.F:8 | 2 | 12/26 | 47-66 | 70:1/2, 85:1/2 |

**pkg/cost** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 18/18 | - | - |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 12/12 | - | 44:1/2, 48:1/2, 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 47/47 | - | 82:1/2, 97:1/2, 102:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 21/21 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 12 | 2/2 | - | - |

**pkg/ctrl** (33)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 4 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 20 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 2 | 11/13 | 51, 54 | 41:1/2, 50:1/2, 53:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 31/67 | 102-110, 124-128, 139-143, 182-186, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 121:1/2, 136:1/2, 179:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 32/37 | 74, 83-84, 113, 140 | 65:1/2, 73:1/2, 78:1/2, 82:1/2, 112:1/2, 120:1/2, 139:1/2, 147:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 83/83 | - | 66:1/2, 131:1/2, 146:1/2, 161:1/2, 176:1/2, 186:1/2, 200:1/2, 214:1/2 |
| `ctrl_cost_gen2d` | ctrl_cost_gen.F:12 | 11 | 41/42 | 158 | 121:1/2, 157:1/2, 162:1/2 |
| `ctrl_cost_gen3d` | ctrl_cost_gen.F:187 | 2 | 41/42 | 322 | 283:1/2, 319:1/2, 326:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 143 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 108 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 108 | 35/66 | 99, 104-108, 145, 174-197, 206-223 | 79:1/2, 92:1/2, 100:1/2, 144:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 139 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 4 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 31 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 64/73 | 272-273, 283, 285, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 18 | 21/36 | 72-76, 87-88, 90-91, 106-112, 118-124 | 86:1/2, 89:1/2, 93:1/2, 104:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 168:1/2, 174:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 12 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr2d` | ctrl_map_genarr.F:13 | 2 | 42/57 | 91-93, 97-99, 102, 108-109, 153-160, 168-170 | 71:1/2, 90:1/2, 95:1/2, 101:1/2, 104:1/2, 145:1/2, 149:1/2, 152:1/2, 167:1/2, 199:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 2 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 12 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 42/49 | 157, 159, 161, 201, 360, 362, 364 | 128:1/2, 154:1/2, 156:1/2, 158:1/2, 160:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 71/127 | 151-153, 157-159, 162, 183-190, 254-261, 276-393, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 253:1/2, 272:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 217/245 | 181-186, 565, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 539:1/2, 540:2/4, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 31 | 15/16 | 60 | 42:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 29 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 2 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 102/140 | 153-155, 169-171, 184-187, 218-222, 237-241, 267-292, 304-306 | 136:1/2, 146:1/2, 162:1/2, 178:1/2, 217:1/2, 236:1/2, 255:1/2, 259:1/4, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 9 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 4416 | 10/16 | 43-45, 55-57 | 41:1/2, 42:2/2, 50:1/2, 52:1/2, 53:2/2 |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/370 | 123-653 | 109:1/2, 111:1/2 |

**pkg/down_slope** (6)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `dwnslp_apply` | dwnslp_apply.F:6 | 96 | 30/49 | 104-115, 120-122, 128-132, 170-175, 183-184, 191 | 95:1/2, 100:1/2, 103:1/2, 119:1/2, 127:1/2, 147:4/6, 152:4/6, 169:1/2, 182:1/2, 190:1/2 |
| `dwnslp_calc_flow` | dwnslp_calc_flow.F:6 | 48 | 22/48 | 79-83, 123-128, 139-157, 163-164 | 74:1/2, 77:1/2, 109:4/8, 122:1/2, 138:1/2, 162:1/2 |
| `dwnslp_calc_rho` | dwnslp_calc_rho.F:6 | 1104 | 8/8 | - | - |
| `dwnslp_init_fixed` | dwnslp_init_fixed.F:6 | 1 | 94/138 | 75-121, 197-202, 216-218, 248-266, 313-314, 337 | 71:1/2, 140:1/2, 149:1/2, 168:1/2, 177:1/2, 195:1/2, 215:1/2, 230:4/6, 235:1/2, 280:1/2, 283:1/2, 286:1/2, 299:1/2, 312:1/2, 328:1/2, 336:1/2 |
| `dwnslp_init_varia` | dwnslp_init_varia.F:6 | 1 | 7/7 | - | - |
| `dwnslp_readparms` | dwnslp_readparms.F:6 | 1 | 23/32 | 44-50, 93-95, 99-101 | 42:1/2, 53:1/2, 91:1/2, 97:1/2, 105:1/2 |

**pkg/ecco** (39)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_averagesfields` | cost_averagesfields.F:3 | 13 | 14/14 | - | 74:1/2, 127:1/2 |
| `cost_averagesflags` | cost_averagesflags.F:4 | 13 | 74/95 | 153, 174-183, 207, 228-239, 263, 285-289 | 152:1/2, 170:1/2, 206:1/2, 224:1/2, 262:1/2, 280:1/2 |
| `cost_averagesgeneric` | cost_averagesgeneric.F:3 | 52 | 38/50 | 113-134, 209 | 111:1/2, 193:1/2 |
| `cost_averagesinit` | cost_averagesinit.F:3 | 1 | 20/20 | - | - |
| `cost_gencal` | cost_gencal.F:7 | 8 | 17/35 | 75-77, 80-98 | 74:1/2, 78:1/2, 112:1/2 |
| `cost_gencost_all` | cost_gencost_all.F:3 | 1 | 21/23 | 54-56 | 53:1/2, 93:1/2, 94:1/2, 95:1/2, 96:1/2 |
| `cost_gencost_assignperiod` | cost_gencost_assignperiod.F:3 | 13 | 12/37 | 58-62, 70-93 | 56:1/2, 63:1/2 |
| `cost_gencost_boxmean` | cost_gencost_boxmean.F:3 | 1 | 4/45 | 76-169 | 71:1/2 |
| `cost_gencost_bpv4` | cost_gencost_bpv4.F:3 | 1 | 7/104 | 95-361 | 80:1/2, 84:1/2 |
| `cost_gencost_customize` | cost_gencost_customize.F:20 | 13 | 61/97 | 128, 131, 137, 140, 143, 147, 166, 169, 172, 175, 179, 182, 185, 190, 193, 197, 210, 213, 217, 220, 223, 247-250, 253-256, 259-261, 264-266, 269-271 | 126:1/2, 129:1/2, 135:1/2, 138:1/2, 141:1/2, 144:1/2, 164:1/2, 167:1/2, 170:1/2, 173:1/2, 177:1/2, 180:1/2, 183:1/2, 188:1/2, 191:1/2, 195:1/2, 207:1/2, 211:1/2, 214:1/2, 218:1/2, 221:1/2, 246:1/2, 252:1/2, 258:1/2, 263:1/2, 268:1/2 |
| `cost_gencost_glbmean` | cost_gencost_glbmean.F:3 | 1 | 2/2 | - | - |
| `cost_gencost_moc` | cost_gencost_moc.F:3 | 1 | 14/89 | 95-270 | 92:1/2 |
| `cost_generic` | cost_generic.F:12 | 4 | 18/18 | - | 101:1/2 |
| `cost_genloop` | cost_generic.F:147 | 4 | 99/125 | 294-295, 298-299, 302, 321-342, 407, 458-464, 501, 511 | 285:1/2, 286:1/2, 287:1/2, 293:1/2, 297:1/2, 301:1/2, 312:1/2, 362:1/2, 370:1/2, 375:1/2, 393:1/2, 406:1/2, 448:1/2, 457:1/2, 495:1/2, 498:1/2, 500:1/2, 505:1/2, 508:1/2, 510:1/2 |
| `cost_genread` | cost_genread.F:7 | 4 | 7/27 | 64-94 | 108:1/2 |
| `ecco_addcost` | ecco_toolbox.F:238 | 4 | 18/20 | 276, 289 | 275:1/2, 286:1/2 |
| `ecco_addmask` | ecco_toolbox.F:414 | 1 | 13/14 | 445 | 444:1/2 |
| `ecco_check` | ecco_check.F:10 | 1 | 59/300 | 248-252, 261-266, 269-274, 277-282, 285-290, 297-301, 320-325, 330-335, 347, 352-358, 365-366, 369-370, 373-374, 380, 382, 384, 388-393, 397-402, 413-432, 440-534, 543-679, 698-703, 715-748, 759-763 | 50:1/2, 247:1/2, 260:1/2, 268:1/2, 276:1/2, 284:1/2, 296:1/2, 305:1/2, 318:1/2, 328:1/2, 344:1/2, 348:1/2, 351:1/2, 364:1/2, 368:1/2, 372:1/2, 376:1/2, 379:1/2, 381:1/2, 383:1/2, 386:1/2, 395:1/2, 412:1/2, 438:1/2, 540:1/2, 688:1/2, 696:1/2, 714:1/2, 755:1/2, 758:1/2 |
| `ecco_check_files` | ecco_check.F:785 | 4 | 13/36 | 836-841, 852-874, 883-888 | 833:1/2, 845:1/2, 847:1/2, 851:1/2, 879:1/2, 880:1/2, 897:1/2 |
| `ecco_cost_final` | ecco_cost_final.F:6 | 1 | 34/36 | 139-141 | 111:1/2 |
| `ecco_cost_init_barfiles` | ecco_cost_init_barfiles.F:7 | 1 | 28/28 | - | 109:1/2 |
| `ecco_cost_init_fixed` | ecco_cost_init_fixed.F:7 | 1 | 36/48 | 132-133, 138-152, 171 | 83:1/2, 130:1/2, 134:1/2, 168:1/2, 238:1/2 |
| `ecco_cost_init_varia` | ecco_cost_init_varia.F:12 | 1 | 14/14 | - | 68:1/2 |
| `ecco_cp` | ecco_toolbox.F:138 | 8 | 11/12 | 166 | 165:1/2 |
| `ecco_diffmsk` | ecco_toolbox.F:74 | 4 | 14/15 | 109 | 108:1/2 |
| `ecco_divfield` | ecco_toolbox.F:523 | 8 | 11/12 | 547 | 546:1/2 |
| `ecco_init_fixed` | ecco_init_fixed.F:8 | 1 | 5/6 | 37 | 31:1/2, 36:1/2 |
| `ecco_init_varia` | ecco_init_varia.F:3 | 1 | 5/5 | - | - |
| `ecco_maskmindepth` | ecco_toolbox.F:669 | 1 | 11/12 | 696 | 695:1/2 |
| `ecco_mult` | ecco_toolbox.F:573 | 4 | 10/11 | 599 | 598:1/2 |
| `ecco_offset` | ecco_toolbox.F:720 | 1 | 34/36 | 754, 791 | 775:1/2, 795:1/2, 811:1/2 |
| `ecco_phys` | ecco_phys.F:17 | 13 | 83/126 | 292-294, 365-389, 411-412, 425-435, 450-451, 454-455, 457-458, 492-499, 555-557, 567, 577-598, 620-628 | 112:1/2, 291:1/2, 364:1/2, 406:1/2, 417:1/2, 449:1/2, 452:1/2, 456:1/2, 491:1/2, 548:1/2, 563:1/2, 572:1/2, 618:1/2 |
| `ecco_readbar` | ecco_toolbox.F:817 | 4 | 16/18 | 875-876 | 869:1/2 |
| `ecco_readparms` | ecco_readparms.F:3 | 1 | 371/497 | 204-209, 513-519, 522-525, 528-531, 534-537, 541-548, 552-558, 561-568, 692-693, 700-710, 715-721, 725-731, 738-739, 774-822, 834-836, 839-841, 844-846, 855, 868-875, 880-887, 891-923 | 202:1/2, 212:1/2, 511:1/2, 521:1/2, 527:1/2, 533:1/2, 539:1/2, 550:1/2, 560:1/2, 576:1/2, 597:1/2, 690:1/2, 698:1/2, 714:1/2, 724:1/2, 736:1/2, 769:1/2, 833:1/2, 838:1/2, 843:1/2, 852:1/2, 854:1/2, 865:1/2, 878:1/2, 890:1/2, 934:1/2 |
| `ecco_readwei` | ecco_toolbox.F:912 | 4 | 14/16 | 960, 967 | 959:1/2, 965:1/2 |
| `ecco_summary` | ecco_summary.F:3 | 1 | 65/89 | 76-82, 90-92, 98-101, 111-114, 133-135, 138-140 | 69:1/2, 75:1/2, 88:1/2, 97:1/2, 110:1/2, 132:1/2, 137:1/2 |
| `ecco_write_pickup` | ecco_write_pickup.F:6 | 2 | 3/3 | - | - |
| `ecco_zero` | ecco_toolbox.F:30 | 605 | 8/8 | - | - |
| `stergloh_output` | stergloh_output.F:8 | 13 | 2/2 | - | - |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 36 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 12 | 90/101 | 236, 241-242, 325-333, 403, 502-506 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 30/140 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 333-342, 353-357, 399-402, 550-569, 575-578, 616-621, 635-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 625:1/2, 634:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 28/76 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 12 | 3/26 | 41-79 | 39:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 16 | 12/22 | 66-78 | 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 8 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 12 | 13/14 | 89 | 68:1/2, 88:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 8 | 12/55 | 65-76, 86-92, 106-141 | 80:1/2, 85:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 96 | 18/97 | 101-109, 121-132, 139, 155-190, 206-259 | 97:1/2, 112:1/2, 119:1/2, 138:1/2, 196:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 12 | 60/76 | 97, 147-152, 315-320, 507, 512, 525, 540 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 505:1/2, 510:1/2, 523:1/2, 538:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 12 | 45/53 | 175-181, 202-203, 329, 388 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 12 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 16 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 89/133 | 64-65, 149-155, 159-179, 185-191, 195-201, 207-213, 262-268, 282-289, 309-315, 322-329, 359-366, 387-394, 474-481, 489, 590-593, 617-619 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 137:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 320:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 385:1/2, 472:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 19 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 41/49 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 12 | 69/95 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 12 | 69/86 | 74, 79-89, 114-116, 164, 186, 192, 204, 210, 222 | 64:1/2, 67:1/2, 71:1/2, 78:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 202:1/2, 208:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 12 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 228 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 12 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 170/208 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 402-408, 487-493, 500-501, 548-555, 571-581, 585-586, 625-626, 636-642, 684-690, 714-720, 761-767, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 633:1/2, 653:1/2, 666:1/2, 681:1/2, 711:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 8 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 12 | 37/83 | 104-115, 138-140, 158-240 | 79:1/2, 102:1/2, 126:1/2, 133:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (12)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 96 | 126/221 | 199-202, 213-219, 253-266, 279-282, 335-337, 351-364, 394, 414, 418, 423-446, 461, 475-539, 615, 635, 639, 644-667, 682, 696-761, 824-827, 844-845, 848-849, 866, 991, 995, 1000-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 419:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 640:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 2208 | 82/206 | 181-184, 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 614, 620, 646-647, 684-685, 696-700, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 669:1/2, 695:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 2112 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 2208 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 2208 | 17/17 | - | 82:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 96/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 2 | 3/3 | - | - |

**pkg/gmredi** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 96 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 48 | 144/218 | 145-147, 166-209, 301-303, 529, 620-627, 689-691, 713-716, 768, 816-837, 851-886, 908-911, 965, 1013-1034, 1048-1083, 1122 | 144:1/2, 164:1/2, 291:1/2, 526:1/2, 610:1/2, 688:1/2, 702:1/2, 765:1/2, 815:1/2, 850:1/2, 897:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 12 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 19/27 | 63-64, 67-68, 82, 86, 142, 148 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 141:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 12 | 3/25 | 50-82 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 107/145 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 48 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 2208 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 3264 | 52/87 | 176, 234, 397, 474, 479, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 473:1/2, 478:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 2 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 2208 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 2208 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 50/71 | 96, 121-130, 202-211, 219-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 115:1/2, 116:1/2, 189:1/2, 201:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 60/118 | 116-126, 134, 163, 173, 192-202, 278, 282, 298-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 172:1/2, 183:1/2, 185:1/2, 186:1/2, 280:1/2, 281:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `bldepth` | kpp_routines.F:309 | 48 | 99/117 | 494-495, 523-524, 640-641, 691-697, 716-717, 830-833, 853-854 | 486:1/2, 491:1/2, 532:1/2, 639:1/2, 685:1/2, 690:1/2, 728:1/2, 781:1/2, 824:1/2, 829:1/2, 865:1/2 |
| `blmix` | kpp_routines.F:1395 | 48 | 72/72 | - | - |
| `enhance` | kpp_routines.F:1696 | 48 | 11/11 | - | 1741:1/2 |
| `kpp_calc` | kpp_calc.F:19 | 48 | 81/88 | 496, 537, 644-649 | 223:1/2, 495:1/2, 528:1/2, 636:1/2, 641:1/2, 680:1/2 |
| `kpp_calc_diff_s` | kpp_calc_diff_s.F:3 | 48 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_diff_t` | kpp_calc_diff_t.F:3 | 48 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_visc` | kpp_calc_visc.F:3 | 1104 | 8/8 | - | - |
| `kpp_check` | kpp_check.F:3 | 1 | 53/62 | 130-132, 137-139, 142-144 | 28:1/2, 128:1/2, 136:1/2, 141:1/2 |
| `kpp_do_exch` | kpp_do_exch.F:6 | 12 | 3/3 | - | - |
| `kpp_forcing_surf` | kpp_forcing_surf.F:10 | 48 | 50/54 | 282-285, 508 | 233:1/2, 247:1/2, 281:1/2, 507:1/2 |
| `kpp_init_fixed` | kpp_init_fixed.F:6 | 1 | 35/64 | 48-111, 188 | 45:1/2, 116:1/2, 162:1/2, 187:1/2 |
| `kpp_init_varia` | kpp_init_varia.F:7 | 1 | 17/17 | - | - |
| `kpp_output` | kpp_output.F:11 | 13 | 8/50 | 118-181, 192-227 | 105:1/2, 114:1/2, 191:1/2 |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 73/112 | 61-66, 171-176, 196-199, 202-205, 208-211, 214-217, 220-226, 230-238 | 59:1/2, 69:1/2, 169:1/2, 195:1/2, 201:1/2, 207:1/2, 213:1/2, 219:1/2, 229:1/2 |
| `kpp_transport_s` | kpp_transport_s.F:9 | 1056 | 9/11 | 80, 89 | 73:1/2, 86:1/2 |
| `kpp_transport_t` | kpp_transport_t.F:6 | 1056 | 8/9 | 72 | 69:1/2 |
| `kppmix` | kpp_routines.F:28 | 48 | 17/17 | - | - |
| `ri_iwmix` | kpp_routines.F:1032 | 48 | 39/42 | 1199-1201 | 1198:1/2 |
| `smooth_horiz` | kpp_routines.F:1311 | 1056 | 15/15 | - | - |
| `statekpp` | kpp_routines.F:1766 | 48 | 30/32 | 1950-1951 | 1949:1/2 |
| `wscale` | kpp_routines.F:923 | 2256 | 28/28 | - | - |

**pkg/mdsio** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 51 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 26 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 259 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 134 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 487, 535-538, 543, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 530:1/2, 540:1/2, 541:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 4 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 206 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 719 | 52/64 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mnc** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mnc_readparms` | mnc_readparms.F:13 | 1 | 5/85 | 68-212 | 55:1/2, 57:1/2 |

**pkg/mom_common** (5)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 1104 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 1104 | 12/26 | 62-66, 77-84, 90-97, 115-137 | 61:1/2, 70:1/2, 88:1/2, 101:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 38/41 | 51-52, 230 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 1104 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 1104 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |

**pkg/mom_fluxform** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_rtrans` | mom_calc_rtrans.F:6 | 1152 | 11/11 | - | - |
| `mom_fluxform` | mom_fluxform.F:42 | 1104 | 161/276 | 265, 276, 331-363, 457, 514-523, 547-594, 607, 622-623, 648-651, 663-666, 690-693, 738-748, 764-767, 773-780, 843-890, 902, 917-918, 943-946, 958-961, 985-988, 1033-1042, 1058-1061, 1067-1074, 1092-1106, 1113-1124, 1142-1146 | 257:1/2, 264:1/2, 270:1/2, 330:1/2, 422:1/2, 444:1/2, 453:1/2, 476:1/2, 510:1/2, 512:1/2, 546:1/2, 601:1/2, 605:1/2, 621:1/2, 647:1/2, 656:1/2, 671:1/2, 689:1/2, 735:1/2, 752:1/2, 753:1/2, 762:1/2, 772:1/2, 789:1/2, 822:1/2, 842:1/2, 897:1/2, 900:1/2, 916:1/2, 942:1/2, 951:1/2, 966:1/2, 984:1/2, 1030:1/2, 1046:1/2, 1047:1/2, 1056:1/2, 1066:1/2, 1082:1/2, 1112:1/2, 1141:1/2 |
| `mom_u_adv_uu` | mom_u_adv_uu.F:7 | 1104 | 5/5 | - | - |
| `mom_u_adv_vu` | mom_u_adv_vu.F:7 | 1104 | 6/9 | 65-74 | 46:1/2 |
| `mom_u_adv_wu` | mom_u_adv_wu.F:7 | 1152 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_u_metric_sphere` | mom_u_metric_sphere.F:7 | 1104 | 6/9 | 59-73 | 46:1/2 |
| `mom_u_xviscflux` | mom_u_xviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_u_yviscflux` | mom_u_yviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_v_adv_uv` | mom_v_adv_uv.F:7 | 1104 | 5/5 | - | - |
| `mom_v_adv_vv` | mom_v_adv_vv.F:7 | 1104 | 5/5 | - | - |
| `mom_v_adv_wv` | mom_v_adv_wv.F:7 | 1152 | 18/21 | 52-55 | 51:1/2, 94:1/2 |
| `mom_v_metric_sphere` | mom_v_metric_sphere.F:7 | 1104 | 6/9 | 60-76 | 44:1/2 |
| `mom_v_xviscflux` | mom_v_xviscflux.F:10 | 1104 | 5/5 | - | - |
| `mom_v_yviscflux` | mom_v_yviscflux.F:10 | 1104 | 5/5 | - | - |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 26 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 13 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 13 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 12 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 48 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 348 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 65 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 13 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 2472 | 52/57 | 213-219 | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2, 211:1/2 |
| `mon_out_i` | mon_out.F:8 | 38 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 2434 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 144 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 13 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 13 | 39/50 | 95, 123-132, 178-179, 196-198 | 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 13 | 74/127 | 145-237, 244-260, 263-278 | 143:1/2, 242:1/2, 243:1/2, 262:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 348 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 65 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 13 | 65/76 | 57, 62-72, 119-120 | 48:1/2, 50:1/2, 54:1/2, 61:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 38 | 7/7 | - | - |
| `read_rec_lev_rl` | read_rec.F:445 | 8 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 17 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 22 | 16/17 | 38 | 37:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 104 | 7/7 | - | - |
| `write_rec_xy_rl` | write_rec.F:145 | 3 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 2 | 7/7 | - | - |

**pkg/salt_plume** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `salt_plume_calc_depth` | salt_plume_calc_depth.F:9 | 48 | 35/54 | 86-127 | 84:1/2, 133:1/2 |
| `salt_plume_check` | salt_plume_check.F:6 | 1 | 6/10 | 51-54 | 28:1/2, 50:1/2 |
| `salt_plume_do_exch` | salt_plume_do_exch.F:6 | 12 | 4/4 | - | 66:1/2 |
| `salt_plume_forcing_surf` | salt_plume_forcing_surf.F:6 | 48 | 7/8 | 47 | 46:1/2 |
| `salt_plume_frac` | salt_plume_frac.F:6 | 6091 | 15/50 | 105, 118-212 | 96:1/2, 100:1/2, 103:1/2 |
| `salt_plume_init_fixed` | salt_plume_init_fixed.F:6 | 1 | 4/6 | 26, 32 | 25:1/2, 31:1/2 |
| `salt_plume_init_varia` | salt_plume_init_varia.F:6 | 1 | 9/9 | - | - |
| `salt_plume_readparms` | salt_plume_readparms.F:6 | 1 | 20/32 | 48-53, 104-113 | 46:1/2, 56:1/2, 103:1/2 |
| `salt_plume_tendency_apply_s` | salt_plume_tendency_apply_s.F:6 | 1104 | 17/19 | 159-161 | 157:1/2 |
| `salt_plume_tendency_apply_t` | salt_plume_tendency_apply_t.F:6 | 1104 | 2/2 | - | - |

**pkg/seaice** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `advect` | advect.F:9 | 36 | 54/63 | 132-149 | 104:1/2, 160:1/2 |
| `diffus` | diffus.F:9 | 72 | 16/30 | 59-99 | 56:1/2 |
| `seaice_advdiff` | seaice_advdiff.F:10 | 12 | 19/74 | 138-401, 586-593, 605-612, 624-631, 642-652 | 135:1/2, 579:1/2, 584:1/2, 598:1/2, 603:1/2, 617:1/2, 622:1/2, 638:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 48 | 6/6 | - | - |
| `seaice_check` | seaice_check.F:12 | 1 | 82/489 | 12, 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 231-237, 243-249, 253-259, 371-404, 434-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 567-569, 700-702, 725-730, 733-741, 744-752, 789-794, 818-925, 935-940, 947-952, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1119-1124, 1130-1135, 1140-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1205-1209, 1216-1221, 1228-1231, 1234-1239, 1242-1247, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 230:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 433:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 788:1/2, 798:1/2, 933:1/2, 945:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1204:1/2, 1214:1/2, 1226:1/2, 1233:1/2, 1241:1/2, 1300:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 12 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 2/2 | - | - |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 2/2 | - | - |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 12 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 12 | 2/2 | - | - |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 12 | 21/206 | 146-346, 415-733 | 136:1/2, 376:1/2, 414:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 12 | 26/57 | 159-164, 173, 178, 221-232, 245-273 | 98:1/2, 146:1/2, 157:1/2, 172:1/2, 177:1/2, 244:1/2 |
| `seaice_growth` | seaice_growth.F:15 | 12 | 273/344 | 336-337, 344, 568-570, 749-759, 1042, 1464-1471, 1808, 1823, 1836, 1839, 2135-2136, 2348, 2352, 2355, 2425-2435, 2483-2555, 2671-2677 | 335:1/2, 343:1/2, 567:1/2, 747:1/2, 790:1/2, 1040:1/2, 1299:1/2, 1462:1/2, 1528:1/2, 1698:1/2, 1807:1/2, 1822:1/2, 1833:1/2, 1837:1/2, 2134:1/2, 2346:1/2, 2350:1/2, 2353:1/2, 2356:1/2, 2424:1/2, 2482:1/2, 2667:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 42/79 | 60, 67, 80, 87-89, 229, 296-353, 365-370 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2, 82:1/2, 228:1/2, 278:1/2, 279:1/2, 362:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 109/160 | 54, 237-242, 254, 272, 274, 277-288, 293-299, 318-327, 346-352, 392-393, 448-451, 463-471 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 271:1/2, 273:1/2, 275:1/2, 292:1/2, 317:1/2, 345:1/2, 391:1/2, 447:1/2, 462:1/2 |
| `seaice_model` | seaice_model.F:13 | 12 | 38/72 | 92, 103-115, 130-133, 254-257, 281-286, 341, 369-390 | 82:1/2, 85:1/2, 102:1/2, 125:1/2, 128:1/2, 178:1/2, 222:1/2, 224:1/2, 252:1/2, 267:1/2, 275:1/2, 340:1/2, 366:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 13 | 39/48 | 65, 70-80 | 55:1/2, 58:1/2, 62:1/2, 69:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 12 | 18/29 | 56, 69-92 | 55:1/2, 64:1/2 |
| `seaice_output` | seaice_output.F:10 | 13 | 26/54 | 66-105, 112, 155-159 | 57:1/2, 65:1/2, 108:1/2, 109:1/2, 127:1/2, 153:1/2, 174:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 440/832 | 233-238, 460-468, 752-757, 773-827, 837-840, 846-856, 866, 868, 912-913, 921-932, 936-942, 954-960, 965-971, 975-986, 992, 1000, 1006-1012, 1017-1025, 1032-1036, 1044, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1442-1444, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1512-1549 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 720:1/2, 721:2/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 763:1/2, 767:1/2, 769:1/2, 771:1/2, 834:1/2, 835:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 896:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 951:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1440:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1511:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 12 | 53/62 | 125-144, 394 | 124:1/2, 393:1/2 |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 336 | 106/148 | 191, 289-294, 310, 324, 392-404, 503-545, 553-561 | 190:1/2, 217:1/2, 288:1/2, 309:1/2, 321:1/2, 385:1/2, 445:1/2, 502:1/2, 552:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 190/261 | 116-311, 357-359, 390, 400, 480-482 | 45:1/2, 108:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 388:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 2 | 46/75 | 109-110, 160-168, 172-188, 194-200 | 78:1/2, 101:1/2, 103:1/2, 117:1/2, 121:1/2, 125:1/2, 133:1/2, 152:1/2, 157:1/2, 159:1/2, 171:1/2, 193:1/2 |

### Compiled but not executed (937 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_r_star, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, freesurf_rescale_g, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cartesian_grid, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_daysformonth, cal_dayspermonth, cal_getmonthsrec, cal_monthsforyear, cal_monthsperyear, cal_printdate, cal_printerror, cal_stepsforday, cal_stepsperday, cal_timestamp, cal_weekday
- pkg/cd_code: cd_code_read_pickup
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/down_slope: dwnslp_diagnostics_init
- pkg/ecco: cost_bp_read, cost_gencost_seaicev4, cost_gencost_sshv4, cost_gencost_sstv4, cost_gencost_transp, cost_sla_read, cost_sla_read_yd, ecco_add, ecco_cprsrl, ecco_diagnostics_init, ecco_div, ecco_error, ecco_multfield, ecco_read_pickup, ecco_subtract, get_exconc_deconc
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: kpp_calc_diff_ptr, kpp_calc_dummy, kpp_diagnostics_init, kpp_doublediff, kpp_transport_ptr, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mnc: mnc_chk_vtyp_r_ncvar, mnc_cw_add_gname, mnc_cw_add_vattr_any, mnc_cw_add_vattr_dbl, mnc_cw_add_vattr_int, mnc_cw_add_vattr_text, mnc_cw_add_vname, mnc_cw_append_vname, mnc_cw_citer_getg, mnc_cw_citer_setg, mnc_cw_del_gname, mnc_cw_del_vname, mnc_cw_dump, mnc_cw_file_aorc, mnc_cw_get_citer, mnc_cw_get_face_num, mnc_cw_get_tile_num, mnc_cw_get_udim, mnc_cw_get_xyfo, mnc_cw_i_r, mnc_cw_i_r_s, mnc_cw_i_r_tf, mnc_cw_i_w, mnc_cw_i_w_offset, mnc_cw_i_w_s, mnc_cw_init, mnc_cw_rl_r, mnc_cw_rl_r_s, mnc_cw_rl_r_tf, mnc_cw_rl_w, mnc_cw_rl_w_offset, mnc_cw_rl_w_s, mnc_cw_rs_r, mnc_cw_rs_r_s, mnc_cw_rs_r_tf, mnc_cw_rs_w, mnc_cw_rs_w_offset, mnc_cw_rs_w_s, mnc_cw_set_citer, mnc_cw_set_gattr, mnc_cw_set_udim, mnc_cw_vattr_missing, mnc_cw_write_cvar, mnc_cw_write_grid_coord, mnc_cw_write_grid_info, mnc_dim_init, mnc_dim_init_all, mnc_dim_init_all_cv, mnc_dim_unlim_size, mnc_dump, mnc_dump_all, mnc_file_add_attr_any, mnc_file_add_attr_dbl, mnc_file_add_attr_int, mnc_file_add_attr_real, mnc_file_add_attr_str, mnc_file_close, mnc_file_close_all, mnc_file_close_all_matching, mnc_file_create, mnc_file_enddef, mnc_file_open, mnc_file_readall, mnc_file_redef, mnc_file_try_read, mnc_get_fvinds, mnc_get_ind, mnc_get_next_empty_ind, mnc_grid_get_dimind, mnc_grid_init, mnc_grid_init_all, mnc_handle_err, mnc_init, mnc_psncm, mnc_set_outdir, mnc_update_time, mnc_var_add_attr_any, mnc_var_add_attr_dbl, mnc_var_add_attr_int, mnc_var_add_attr_real, mnc_var_add_attr_str, mnc_var_append_dbl, mnc_var_append_int, mnc_var_append_real, mnc_var_init_any, mnc_var_init_dbl, mnc_var_init_int, mnc_var_init_real, mnc_var_write_any, mnc_var_write_dbl, mnc_var_write_int, mnc_var_write_real
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_uv_boundary, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/salt_plume: salt_plume_apply, salt_plume_diagnostics_fill, salt_plume_diagnostics_init, salt_plume_mnc_init, salt_plume_volfrac
- pkg/seaice: adseaice_monitor, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advection, seaice_bottomdrag_coeffs, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stress, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check_pickup, seaice_cost_export, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_growth_adx, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_lsr_calc_coeffs, seaice_lsr_rhsu, seaice_lsr_rhsv, seaice_lsr_tridiagu, seaice_lsr_tridiagv, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_oceandrag_coeffs, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_residual, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys

