# Coverage: global_ocean.cs32x15 (code_ad) — M4 porting worklist

Written by `tools/coverage.py` (mitjax fc1982a) from the gcov build `global_ocean.cs32x15-code_ad-63cdc0b-78ca390-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 869 files from the link farm, 205 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_ad.seaice` | `job27855996` | 3 | 0 / 0 | 504 | yes | identical (412 lines) |
| `input_ad.seaice_dynmix` | `job27855996` | 6 | 0 / 0 | 447 | yes | identical (730 lines) |
| `input_ad.thsice` | `job27855996` | 6 | 0 / 0 | 427 | yes | identical (784 lines) |

## global_ocean.cs32x15/input_ad.seaice

Run `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.seaice/job27855996` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.seaice/job27855996/gcov-8773796/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice thermodynamics | seaice_growth (2 calls) | yes |
| LSR momentum solver | seaice_lsr (2 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (2 calls) | yes |
| cost function | global fc = 110168.104101416; terms: objf_test sum 1.101681e+05 | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, doResetHFactors, dumpInitAndLast, fluidIsWater, hasWetCSCorners, inAdExact, momAdvection, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, no_slip_sides, pickup_read_mdsio, pickup_write_mdsio, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, SEAICE_clipVeloctities, SEAICE_doOpenWaterGrowth, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICEadvArea, SEAICEadvHeff, SEAICEadvSnow, SEAICEmultiDimAdvection, SEAICEupdateOceanStress, SEAICEuseDYNAMICS, SEAICEuseFlooding, SEAICEuseLSR, SEAICEuseTilt, snapshot_mdsio, stressIsOnCgrid, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useMultiDimAdvec, usePW79thermodynamics, useSEAICEinAdMode, usingZCoords, W2_useE2ioLayOut, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, balancePrintMean, balanceQnet, balanceSaltClimRelax, balanceThetaClimRelax, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, diag_mnc, doSaltClimRelax, doThetaClimRelax, dumpAtLast, fluidIsAir, globalFiles, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, highOrderVorticity, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momImplVertAdv, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICE_2ndOrderBC, SEAICE_doOpenWaterMelt, SEAICE_growMeltByConv, SEAICE_maskRHS, SEAICE_mcPheeStepFunc, SEAICE_no_slip, SEAICE_salinityTracer, SEAICEheatConsFix, SEAICEmomAdvection, SEAICErestoreUnderIce, SEAICEuseBDF2, SEAICEuseDYNAMICSswitchInAd, SEAICEuseEVP, SEAICEuseFREEDRIFT, SEAICEuseFREEDRIFTswitchInAd, SEAICEuseJFNK, SEAICEuseKrylov, SEAICEuseLSRflex, SEAICEuseMultiTileSolver, SEAICEusePicardAsPrecon, SEAICEuseStrImpCpl, SEAICEuseTEM, SEAICEwriteState, startFromPickupAB2, tempImplVertAdv, tempSOM_Advection, twoDigitYear, upwindShear, upwindVorticity, use3Dsolver, useAbsVorticity, useBiharmonicVisc, useCDscheme, useCoupler, useCtrlCostContribution, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useHB87stressCoupling, useJamartMomAdv, useKPPinAdMode, useMaykutSatVapPoly, useMin4hFacEdges, useMissingValue, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useSALT_PLUMEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid
- selectors: exf_adjMonSelect=1, monitorSelect=3, pCellMix_select=0, saltVertAdvScheme=33, SEAICEselectMetricTerms=2, select_ZenAlbedo=0, selectAddFluid=0, selectBalanceEmPmR=0, selectBotDragQuadr=-1, selectKEscheme=0, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempVertAdvScheme=33

### Executed routines (504)

**eesupp/src** (70)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 8108 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 185 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `diff_phase_multiple` | diff_phase_multiple.F:7 | 16 | 15/16 | 46 | 44:1/2, 45:1/2, 50:1/2 |
| `different_multiple` | different_multiple.F:7 | 387 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 29/29 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch_3d_rl` | exch_3d_rl.F:8 | 9 | 4/4 | - | - |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 384 | 4/4 | - | - |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 5 | 4/4 | - | - |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 4 | 4/4 | - | - |
| `exch_uv_agrid_3d_rs` | exch_uv_agrid_3d_rs.F:8 | 2 | 4/4 | - | - |
| `exch_uv_bgrid_3d_rs` | exch_uv_bgrid_3d_rs.F:8 | 1 | 4/4 | - | - |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 590 | 3/3 | - | - |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 14 | 3/3 | - | - |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 3/3 | - | - |
| `exch_xy_rl` | exch_xy_rl.F:9 | 36 | 3/3 | - | - |
| `exch_xy_rs` | exch_xy_rs.F:9 | 54 | 3/3 | - | - |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 8 | 3/3 | - | - |
| `exch_z_3d_rs` | exch_z_3d_rs.F:8 | 3 | 4/4 | - | - |
| `fill_cs_corner_tr_rl` | fill_cs_corner_tr_rl.F:9 | 1560 | 45/62 | 93-117, 263 | 69:1/2, 71:1/2, 90:1/2, 193:1/2 |
| `fill_cs_corner_uv_rs` | fill_cs_corner_uv_rs.F:9 | 792 | 38/38 | - | 65:1/2, 68:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 8663 | 2/2 | - | - |
| `get_periodic_interval` | get_periodic_interval.F:7 | 22 | 20/45 | 73-81, 87-93, 99-113 | 71:1/2, 86:1/2, 96:1/2 |
| `global_max_r8` | global_max.F:97 | 868 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 50 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 223 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 927 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 1620 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 13601 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 55/82 | 90-99, 114-119, 124-129, 134-139 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2 |
| `lcase` | utils.F:190 | 21 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 563 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1535 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 1242 | 11/19 | 44-50, 62-64 | 42:1/2, 60:1/2 |
| `nml_change_syntax` | nml_change_syntax.F:7 | 274 | 7/7 | - | - |
| `nml_set_terminator` | nml_set_terminator.F:7 | 5 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 11 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 65 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 168 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 311 | 43/49 | 648-650, 668-671 | 647:1/2, 662:1/2, 664:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 3860 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `timer_control` | timers.F:74 | 132 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 132 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 132 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 67 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 65 | 4/4 | - | - |
| `ucase` | utils.F:311 | 266 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 10 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 55 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 168 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 219 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 43 | 32/32 | - | 195:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (107)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth3` | adams_bashforth3.F:6 | 720 | 21/29 | 85-87, 103-109 | 84:1/2, 101:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 22/51 | 55-71, 113-133 | 52:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 360 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 360 | 18/53 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 360 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 360 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 48 | 21/24 | 164-166 | 83:1/2, 132:1/2, 203:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 720 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 360 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 360 | 36/81 | 70-73, 84-131, 170, 174-177, 205-261 | 63:1/2, 68:1/2, 158:1/2, 164:1/2, 165:1/2, 172:1/2 |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 24 | 8/8 | - | - |
| `calc_grid_angles` | calc_grid_angles.F:7 | 1 | 29/37 | 76-83 | 75:1/2 |
| `calc_ivdc` | calc_ivdc.F:5 | 336 | 7/7 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 24 | 8/81 | 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 360 | 37/155 | 149, 184, 194-197, 212-240, 268-610 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 193:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_r_star` | calc_r_star.F:10 | 4 | 66/119 | 68, 143-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 111:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 24 | 10/14 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 2 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `check_pickup` | check_pickup.F:8 | 1 | 42/124 | 67-73, 84-88, 92-97, 107-110, 117-120, 123-126, 129-133, 137-140, 143-146, 149-152, 161-164, 172-175, 181, 185-211, 216, 218-222, 242-243 | 55:1/2, 57:1/2, 66:1/2, 77:1/2, 83:1/2, 91:1/2, 101:1/2, 116:1/2, 122:1/2, 128:1/2, 136:1/2, 142:1/2, 148:1/2, 159:1/2, 170:1/2, 179:1/2, 182:1/2, 215:1/2, 217:1/2, 223:1/2, 229:1/2, 240:1/2 |
| `config_check` | config_check.F:14 | 1 | 86/482 | 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 253-259, 266-271, 275-280, 344-349, 366-371, 417-423, 442-448, 454-460, 476-479, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 947-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 252:1/2, 264:1/2, 273:1/2, 342:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 475:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 946:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 451/545 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 426, 470-486, 540-548, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 968-974, 992-994, 1002-1008, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 24 | 26/57 | 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 48 | 6/6 | - | - |
| `diags_oceanic_surf_flux` | diags_oceanic_surf_flux.F:7 | 2 | 50/51 | 56 | 55:1/2, 81:1/2, 94:1/2, 115:1/2, 131:1/2, 141:1/2, 161:1/2, 178:1/2 |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 360 | 15/24 | 78-86, 92-102 | 73:1/2, 76:1/2, 91:1/2, 121:1/2 |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 360 | 28/48 | 79-84, 123-125, 136-144, 150-157, 179-183 | 61:1/2, 76:1/2, 121:1/2, 132:1/2, 134:1/2, 147:1/2 |
| `diags_rho_g` | diags_rho.F:100 | 2 | 8/33 | 161-171, 178-188, 195-213 | 160:1/2, 177:1/2, 194:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 2 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 2 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 2 | 101/128 | 258, 260, 262, 264, 397-403, 467-471, 557, 771-784, 797-798, 830-833, 882, 934, 1043 | 248:1/2, 251:1/2, 256:1/2, 257:1/2, 259:1/2, 261:1/2, 263:1/2, 377:1/2, 421:1/2, 450:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 869:1/2, 878:1/2, 896:1/2, 932:1/2, 1030:1/2, 1032:1/2, 1096:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 2 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_statevars_diags` | do_statevars_diags.F:8 | 6 | 16/18 | 64, 102 | 54:1/2, 60:1/2, 101:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 3 | 12/19 | 97-110, 207 | 96:1/2, 116:1/2, 193:1/2, 206:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 2 | 20/26 | 69-72, 82, 84, 115-117 | 66:1/2, 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 2 | 64/75 | 358, 591-599, 708-715 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 682:1/2, 693:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 2 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 2 | 45/78 | 75, 100-107, 111-113, 176, 269-274, 296-343, 376-378 | 74:1/2, 82:1/2, 98:1/2, 110:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 268:1/2, 279:1/2, 364:1/2, 367:1/2 |
| `find_bulkmod` | find_rho.F:412 | 696 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 696 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 43 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 696 | 16/16 | - | - |
| `forward_step` | forward_step.F:70 | 2 | 105/127 | 483-485, 730-734, 767-769, 842-853, 952-960, 980, 1176, 1201-1202 | 424:1/2, 468:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 832:1/2, 866:1/2, 897:1/2, 924:1/2, 939:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 720 | 7/12 | 49-66 | 39:1/2, 40:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 360 | 22/22 | - | 59:1/2, 72:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 24/67 | 57-63, 70-79, 106-112, 121-180, 191 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_curvilinear_grid` | ini_curvilinear_grid.F:7 | 1 | 95/133 | 280, 353, 391-408, 430-447 | 259:1/2, 279:1/2, 344:1/2, 390:1/2, 429:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/68 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 32/32 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 50/50 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 5/10 | 29-33 | 28:1/2, 37:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 59/96 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 216-220, 228-229, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 214:1/2, 226:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 58/62 | 126-131, 136-138 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2, 176:1/2, 177:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/115 | 131, 133, 136-144, 192 | 130:1/2, 132:1/2, 134:1/2, 161:1/2, 163:1/2, 165:1/2, 167:1/2, 169:1/2, 175:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 183/198 | 210-212, 247-261, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 9/11 | 52-53 | 51:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 30/64 | 100-113, 146-179, 191-195, 206-209 | 99:1/2, 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 74/74 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 407/889 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 671-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1128-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1396, 1398, 1403-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1552-1555, 1558-1561, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 669:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1397:1/2, 1399:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1551:1/2, 1557:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 37/42 | 309-321, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 3 | 56/70 | 147-150, 164-169, 212-214, 279-282 | 90:1/2, 93:1/2, 95:1/2, 146:1/2, 163:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 540 | 18/36 | 95-117, 177-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 2 | 27/34 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 13/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 115-161, 168-173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 167:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 2 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 2 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 117/117 | - | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/119 | 132-140, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 252:1/2, 260:1/2, 264:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 380:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 36/40 | 160-162, 530-532 | 144:1/2, 158:1/2, 167:1/2, 170:1/2, 174:1/2, 193:1/2, 200:1/2, 202:1/2, 249:1/2, 251:1/2, 327:1/2, 329:1/2, 358:1/2, 365:1/2, 367:1/2, 510:1/2, 512:1/2, 528:1/2, 651:1/2, 655:1/2, 673:1/2, 675:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 19/23 | 169, 452-454, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 270:1/2, 291:1/2, 293:1/2, 435:1/2, 450:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 19 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 13/13 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 2 | 33/37 | 68-70, 76 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 97:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 12/13 | 193 | 124:1/2, 184:1/2, 191:1/2, 223:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 696 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `read_pickup` | read_pickup.F:8 | 1 | 59/147 | 86-89, 110-116, 124-152, 161-235, 298-304, 307-313, 318-324, 327-333, 409, 448, 474-477, 565, 567 | 82:1/2, 83:1/2, 97:1/2, 107:1/2, 109:1/2, 122:1/2, 160:1/2, 276:1/2, 278:1/2, 282:1/2, 287:1/2, 291:1/2, 297:1/2, 306:1/2, 317:1/2, 326:1/2, 407:1/2, 446:1/2, 456:1/2, 460:1/2, 473:1/2, 564:1/2, 566:1/2 |
| `reset_nlfs_vars` | reset_nlfs_vars.F:6 | 2 | 8/11 | 48-50 | 47:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 2 | 28/64 | 67-70, 78, 85-127, 168-174 | 66:1/2, 77:1/2, 83:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 24 | 60/74 | 169, 186, 326, 367-369, 380-390, 422-426, 493-501, 511 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 413:1/2, 479:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 99/165 | 56-57, 60-63, 71, 76, 109-115, 120-126, 171-175, 191-195, 202, 208, 214, 220, 226, 230, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 186:1/2, 188:1/2, 189:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 2 | 63/93 | 152-157, 164, 170-176, 228-234, 265, 269, 319, 327, 342-344, 355-366 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `solve_tridiagonal` | solve_tridiagonal.F:10 | 48 | 32/38 | 249-251, 266-268 | 244:1/2, 259:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 24 | 60/74 | 171, 188, 328, 369-371, 382-392, 424-428, 495-503, 513 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 481:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 43/43 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 2 | 46/63 | 158, 218-250, 395-399 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 206:1/2, 207:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 360 | 63/96 | 118-121, 130-133, 140-143, 277-312, 328-332, 400-405 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 276:1/2, 321:1/2, 391:1/2, 392:1/2, 412:1/2, 416:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 48 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 2 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/21 | 113 | 55:1/2, 78:1/2, 106:1/2, 112:1/2 |
| `update_cg2d` | update_cg2d.F:7 | 3 | 41/49 | 58, 132-140, 169, 175, 182 | 57:1/2, 61:1/2, 131:1/2, 159:1/2, 168:1/2, 174:1/2, 181:1/2 |
| `update_etah` | update_etah.F:7 | 3 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 5 | 29/29 | - | - |
| `write_grid` | write_grid.F:12 | 1 | 48/66 | 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 63/101 | 167-182, 188-203, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 137:1/2, 139:1/2, 143:1/2, 145:1/2, 149:1/2, 152:1/2, 156:1/2, 158:1/2, 162:1/2, 166:1/2, 187:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 3 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (19)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 20 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 16 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 4 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 9 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 8 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 2 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 2 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 86/152 | 63-68, 127-132, 157, 165-172, 175-176, 237-247, 270-273, 276-279, 302-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 294:1/2, 301:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 3 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 3 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 6 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 3 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 2 | 2/2 | - | - |

**pkg/cal** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_getdate` | cal_getdate.F:3 | 1 | 7/23 | 55-88 | 47:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 7/23 | 79-117 | 65:1/2, 68:1/2, 70:1/2 |

**pkg/cost** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_accumulate_mean` | cost_accumulate_mean.F:3 | 2 | 13/13 | - | - |
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 7/7 | - | 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 50/50 | - | 82:1/2, 102:1/2, 110:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 22/22 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 2 | 4/4 | - | 121:1/2 |

**pkg/ctrl** (30)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 3 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 4 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 3 | 12/13 | 51 | 50:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 34/72 | 104-110, 124-128, 139-143, 182-186, 226-230, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 103:1/2, 121:1/2, 136:1/2, 179:1/2, 225:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/37 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/83 | 70-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 15 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 8 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 8 | 12/66 | 81-162, 190-197, 206-223 | 79:1/2, 177:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 12 | 7/11 | 129-130, 132-133 | 126:1/2, 128:1/2, 131:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 3 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 16 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 66/73 | 233-238, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 8 | 15/36 | 72-76, 87-88, 93-112, 118-124 | 86:1/2, 89:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 99/104 | 192-219 | 168:1/2, 183:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 2 | 51/57 | 82, 109, 111, 113, 115, 118 | 81:1/2, 83:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 3 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 2 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 35/51 | 155-166, 201, 219, 221, 360, 362 | 128:1/2, 154:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 392:1/2, 394:1/2, 405:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 65/80 | 151-153, 157-159, 162, 183-190, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 207/245 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 16 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 12 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 4 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 89/140 | 153-155, 161-171, 184-187, 218-222, 229-241, 257-261, 267-292, 304-306 | 146:1/2, 160:1/2, 178:1/2, 217:1/2, 228:1/2, 255:1/2, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 4 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (42)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_addtolist` | diagnostics_addtolist.F:8 | 324 | 17/44 | 62, 68-92, 111-117 | 60:1/2, 67:1/2, 99:1/2, 103:1/2, 108:1/2, 125:1/2 |
| `diagnostics_check` | diagnostics_check.F:8 | 1 | 31/150 | 44-47, 51-54, 59-64, 85-89, 92-96, 104-113, 120-130, 145-213, 231-237, 244-250, 261-268, 277-280 | 37:1/2, 43:1/2, 50:1/2, 57:1/2, 84:1/2, 91:1/2, 103:1/2, 118:1/2, 119:2/2, 144:1/2, 224:1/2, 230:1/2, 243:1/2, 260:1/2, 275:1/2 |
| `diagnostics_cumulate` | diagnostics_fill_field.F:413 | 3480 | 20/62 | 478-517, 521-556, 569-578, 602 | 477:1/2, 520:1/2, 561:1/2, 594:1/2 |
| `diagnostics_fill` | diagnostics_fill.F:6 | 11386 | 31/34 | 75, 130-131 | 73:1/2, 94:1/2, 129:1/2 |
| `diagnostics_fill_field` | diagnostics_fill_field.F:13 | 38 | 49/101 | 114-116, 136-141, 146-151, 159-160, 170-177, 182-183, 188-189, 195, 203-216, 235-268 | 105:3/6, 107:1/2, 132:1/2, 145:1/2, 158:1/2, 167:1/2, 181:1/2, 184:1/2, 192:1/2, 194:1/2, 202:1/2, 220:1/2, 226:1/2 |
| `diagnostics_fill_rs` | diagnostics_fill_rs.F:6 | 20 | 14/34 | 75, 84-85, 93-103, 118-146 | 73:1/2, 80:1/2, 92:1/2, 117:1/2 |
| `diagnostics_fill_state` | diagnostics_fill_state.F:6 | 6 | 78/290 | 70-79, 112-139, 145-166, 170-183, 187-203, 207-223, 229-241, 245-257, 261-274, 278-290, 294-306, 310-323, 327-339, 343-355, 359-380, 398-410, 423-435, 492, 496, 499, 537-550, 590-603, 607-620, 624-637, 641-654, 658-671, 675-688, 714-724, 739-749 | 69:1/2, 93:1/2, 111:1/2, 143:1/2, 169:1/2, 186:1/2, 206:1/2, 228:1/2, 244:1/2, 260:1/2, 277:1/2, 293:1/2, 309:1/2, 326:1/2, 342:1/2, 358:1/2, 393:1/2, 418:1/2, 451:1/2, 487:1/2, 494:1/2, 497:1/2, 534:1/2, 555:1/2, 571:1/2, 589:1/2, 606:1/2, 623:1/2, 640:1/2, 657:1/2, 674:1/2, 709:1/2, 734:1/2 |
| `diagnostics_ini_io` | diagnostics_ini_io.F:6 | 1 | 4/12 | 45-54 | 39:1/2, 42:1/2 |
| `diagnostics_init_early` | diagnostics_init_early.F:8 | 1 | 107/107 | - | 71:1/2 |
| `diagnostics_init_fixed` | diagnostics_init_fixed.F:8 | 1 | 8/8 | - | - |
| `diagnostics_init_varia` | diagnostics_init_varia.F:8 | 1 | 24/24 | - | 30:1/2 |
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 2050 | 14/16 | 56-57 | 45:1/2, 55:1/2 |
| `diagnostics_main_init` | diagnostics_main_init.F:8 | 1 | 605/624 | 95-97, 104-110, 112-117, 128-129, 131-132, 285 | 94:1/2, 103:1/2, 111:1/2, 127:1/2, 130:1/2, 201:1/2, 248:1/2, 284:1/2, 621:1/2, 633:1/2, 779:1/2 |
| `diagnostics_read_pickup` | diagnostics_read_pickup.F:7 | 1 | 2/2 | - | - |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 258/372 | 111-119, 280-282, 293, 295, 298-300, 302-309, 311-312, 322, 333-342, 357-366, 372-381, 421-424, 429-435, 446-447, 454-467, 471-478, 493-502, 508-517, 538-540, 582-583, 587-588, 593-596 | 109:1/2, 123:1/2, 168:1/2, 279:1/2, 292:1/2, 294:1/2, 297:1/2, 301:1/2, 310:1/2, 313:1/2, 321:1/2, 332:1/2, 356:1/2, 371:1/2, 420:1/2, 428:1/2, 444:1/2, 453:1/2, 469:1/2, 481:1/2, 492:1/2, 507:1/2, 536:1/2, 570:1/2, 586:1/2, 592:1/2, 600:1/2, 601:2/4, 603:1/4, 607:1/4, 609:1/4, 631:1/2, 635:2/4, 638:1/4 |
| `diagnostics_scale_fill` | diagnostics_scale_fill.F:6 | 440 | 21/33 | 78, 120-146 | 76:1/2, 96:1/2, 119:1/2 |
| `diagnostics_scale_fill_rs` | diagnostics_scale_fill_rs.F:6 | 12 | 19/33 | 78, 86-87, 120-148 | 76:1/2, 82:1/2, 96:1/2, 119:1/2 |
| `diagnostics_set_calc` | diagnostics_set_calc.F:9 | 1 | 7/80 | 78-194 | 46:1/2 |
| `diagnostics_set_levels` | diagnostics_set_levels.F:7 | 1 | 75/133 | 88, 95-115, 119-122, 131-134, 138-141, 148-151, 156-160, 164-167, 218-220, 226-228, 244-253 | 73:1/2, 87:1/2, 93:1/2, 118:1/2, 130:1/2, 137:1/2, 147:1/2, 155:1/2, 163:1/2, 183:1/2, 217:1/2, 221:1/2, 241:1/2, 243:1/2 |
| `diagnostics_set_pointers` | diagnostics_set_pointers.F:6 | 1 | 88/160 | 59-63, 70-96, 101-109, 145-151, 166-186, 248-251, 254-260, 292-302 | 41:1/2, 57:1/2, 58:1/2, 69:1/2, 100:1/2, 142:1/2, 158:1/2, 194:1/2, 206:1/2, 246:1/2, 252:1/2, 271:1/2, 272:2/4, 274:1/4, 278:1/2, 286:1/2 |
| `diagnostics_setdiag` | diagnostics_setdiag.F:6 | 49 | 35/89 | 66-72, 92-94, 109-111, 118-126, 133-134, 142-145, 156, 159-160, 165-210 | 65:1/2, 87:1/2, 90:1/2, 101:1/2, 107:1/2, 114:1/2, 115:1/2, 131:1/2, 155:1/2, 158:1/2, 163:1/2 |
| `diagnostics_summary` | diagnostics_summary.F:7 | 1 | 7/124 | 61-83, 94-243 | 55:1/2, 60:1/2, 90:1/2 |
| `diagnostics_switch_onoff` | diagnostics_switch_onoff.F:11 | 2 | 37/78 | 81, 101-135, 146-173, 205, 220-221, 233-234 | 77:1/2, 87:1/2, 98:1/2, 144:1/2, 185:1/2, 202:1/2, 216:1/2, 218:1/2, 229:1/2, 231:1/2 |
| `diagnostics_write` | diagnostics_write.F:3 | 3 | 45/54 | 69-70, 87, 97-98, 120-121, 135, 152 | 62:1/2, 84:1/2, 91:1/2, 92:1/2, 96:1/2, 110:1/2, 132:1/2, 139:1/2, 151:1/2, 161:1/2, 171:1/2 |
| `diagnostics_write_pickup` | diagnostics_write_pickup.F:7 | 1 | 3/3 | - | - |
| `diags_mk_title` | diagnostics_utils.F:593 | 37 | 17/23 | 643-650 | 632:1/2, 635:1/2, 642:1/2 |
| `diags_mk_units` | diagnostics_utils.F:514 | 41 | 13/30 | 554-568, 575-582 | 547:1/2, 552:1/2, 574:1/2 |
| `diags_renamed` | diagnostics_utils.F:661 | 700 | 14/19 | 708-714 | 694:1/2, 695:1/2, 696:1/2, 697:1/2, 698:1/2, 699:1/2, 700:1/2, 702:1/2, 703:1/2, 705:1/2 |
| `diags_track_diva` | diagnostics_utils.F:410 | 1 | 5/9 | 448-452 | 444:1/2 |
| `diagstats_ascii_out` | diagstats_ascii_out.F:8 | 5 | 19/19 | - | 43:1/2, 45:1/2, 75:1/4 |
| `diagstats_calc` | diagstats_calc.F:6 | 732 | 43/67 | 99-102, 116-121, 125-130, 153-155, 161-163, 167-170, 174-177 | 98:1/2, 115:1/2, 124:1/2, 152:1/2, 160:1/2, 166:1/2, 173:1/2 |
| `diagstats_clear` | diagstats_clear.F:8 | 1 | 7/7 | - | 30:1/2 |
| `diagstats_close_io` | diagstats_close_io.F:6 | 1 | 16/18 | 55, 71 | 40:1/2, 42:1/2, 52:1/2, 70:1/2 |
| `diagstats_clrdiag` | diagstats_clear.F:46 | 5 | 8/8 | - | - |
| `diagstats_fill` | diagstats_fill.F:13 | 5 | 34/78 | 121-122, 136-141, 149-150, 160-167, 173-175, 181-183, 188, 196-209, 234-246 | 120:1/2, 135:1/2, 148:1/2, 157:1/2, 172:1/2, 176:1/2, 187:1/2, 195:1/2, 216:1/2 |
| `diagstats_global` | diagstats_global.F:8 | 5 | 53/87 | 87-96, 115-116, 127-129, 151-159, 171-172, 188-206 | 61:1/2, 63:1/2, 72:1/2, 86:1/2, 113:1/2, 126:1/2, 136:1/2, 150:1/2, 168:1/2, 170:1/2, 181:1/2 |
| `diagstats_ini_io` | diagstats_ini_io.F:6 | 1 | 36/37 | 54 | 40:1/2, 42:1/2, 51:1/2, 77:2/4, 83:2/4, 85:1/4, 87:2/4, 89:1/4 |
| `diagstats_local` | diagstats_local.F:6 | 732 | 35/46 | 112-114, 172, 185, 201, 212, 234-235, 242-243 | 111:1/2, 121:1/2, 160:1/2, 173:1/2, 191:1/2, 202:1/2, 229:1/2, 230:1/2, 237:1/2 |
| `diagstats_output` | diagstats_output.F:8 | 1 | 19/45 | 72, 88-111, 116-125 | 66:1/2, 69:1/2, 87:1/2, 115:1/2, 133:1/2 |
| `diagstats_set_pointers` | diagstats_set_pointers.F:6 | 1 | 37/84 | 72-78, 82-95, 98-105, 120-140, 150-156, 162-165 | 40:1/2, 68:1/2, 80:1/2, 97:1/2, 112:1/2, 149:1/2, 161:1/2 |
| `diagstats_set_regions` | diagstats_set_regions.F:6 | 1 | 31/71 | 70-73, 76-80, 86-105, 132-135, 138-142, 144, 170-173 | 65:1/2, 69:1/2, 75:1/2, 85:1/2, 118:1/2, 125:1/2, 130:1/2, 137:1/2, 143:1/2, 169:1/2 |
| `diagstats_setdiag` | diagstats_setdiag.F:6 | 5 | 20/51 | 70-72, 84-85, 95-97, 104-141 | 65:1/2, 68:1/2, 81:1/2, 82:1/2, 103:1/2 |

**pkg/exch2** (34)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exch2_3d_rl` | exch2_3d_rl.F:8 | 53 | 11/11 | - | - |
| `exch2_3d_rs` | exch2_3d_rs.F:8 | 54 | 11/11 | - | - |
| `exch2_check_depths` | exch2_check_depths.F:9 | 1 | 36/63 | 108-137, 149-156 | 85:1/2, 91:1/2, 97:1/2, 103:1/2, 106:1/2, 148:1/2 |
| `exch2_get_rl1` | exch2_get_rl1.F:8 | 27324 | 18/19 | 125 | 105:1/2, 129:4/8, 130:4/8, 131:4/6 |
| `exch2_get_rl2` | exch2_get_rl2.F:8 | 64260 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_rs1` | exch2_get_rs1.F:8 | 6804 | 18/19 | 125 | 105:1/2, 129:5/8, 130:4/8, 131:4/6 |
| `exch2_get_rs2` | exch2_get_rs2.F:8 | 1620 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_scal_bounds` | exch2_get_scal_bounds.F:7 | 68364 | 46/50 | 65, 82, 99, 116 | 62:1/2, 79:1/2, 96:1/2, 113:1/2 |
| `exch2_get_uv_bounds` | exch2_get_uv_bounds.F:7 | 131760 | 92/100 | 92, 110, 128, 146, 173, 184, 259-260 | 89:1/2, 107:1/2, 125:1/2, 143:1/2, 165:1/2, 172:1/2, 183:1/2, 233:1/2, 238:1/2 |
| `exch2_put_rl1` | exch2_put_rl1.F:8 | 27324 | 29/33 | 190-193 | 135:4/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rl2` | exch2_put_rl2.F:8 | 64260 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_put_rs1` | exch2_put_rs1.F:8 | 6804 | 29/33 | 190-193 | 135:5/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rs2` | exch2_put_rs2.F:8 | 1620 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_rl1_cube` | exch2_rl1_cube.F:7 | 506 | 34/34 | - | - |
| `exch2_rl2_cube` | exch2_rl2_cube.F:8 | 1190 | 40/40 | - | - |
| `exch2_rs1_cube` | exch2_rs1_cube.F:7 | 126 | 34/34 | - | - |
| `exch2_rs2_cube` | exch2_rs2_cube.F:8 | 30 | 40/40 | - | - |
| `exch2_s3d_rl` | exch2_s3d_rl.F:8 | 384 | 10/10 | - | - |
| `exch2_uv_3d_rl` | exch2_uv_3d_rl.F:9 | 595 | 42/42 | - | 79:1/2 |
| `exch2_uv_3d_rs` | exch2_uv_3d_rs.F:9 | 15 | 42/42 | - | 79:1/2 |
| `exch2_uv_agrid_3d_rl` | exch2_uv_agrid_3d_rl.F:8 | 4 | 46/46 | - | 68:1/2, 94:1/2, 148:1/2, 161:1/2 |
| `exch2_uv_agrid_3d_rs` | exch2_uv_agrid_3d_rs.F:8 | 2 | 46/46 | - | 94:1/2, 148:1/2, 161:1/2 |
| `exch2_uv_bgrid_3d_rs` | exch2_uv_bgrid_3d_rs.F:8 | 1 | 100/108 | 86-93 | 81:1/2, 83:1/2, 85:1/2, 134:1/2, 196:1/2, 211:1/2 |
| `exch2_z_3d_rs` | exch2_z_3d_rs.F:8 | 3 | 84/92 | 65-72 | 63:1/2, 64:1/2, 95:1/2, 106:1/2, 141:1/2, 158:1/2, 203:1/2, 220:1/2 |
| `w2_e2setup` | w2_e2setup.F:10 | 1 | 35/63 | 64-78, 95-101, 108-113, 119, 121, 123, 127 | 63:1/2, 94:1/2, 105:1/2, 106:2/2, 118:1/2, 120:1/2, 122:1/2, 124:1/2 |
| `w2_eeboot` | w2_eeboot.F:7 | 1 | 56/56 | - | 85:1/2, 106:1/2, 111:1/2 |
| `w2_map_procs` | w2_map_procs.F:7 | 1 | 55/69 | 101, 113-117, 121-125, 130, 152-155 | 75:1/2, 94:1/2, 99:1/2, 111:1/2, 119:1/2, 129:1/2, 149:1/2, 150:1/2, 157:1/2 |
| `w2_print_comm_sequence` | w2_print_comm_sequence.F:8 | 1 | 52/52 | - | - |
| `w2_readparms` | w2_readparms.F:9 | 1 | 63/89 | 69, 111-127, 133-134, 139-140, 166-172, 182-187, 190 | 66:1/2, 110:1/2, 132:1/2, 135:1/2, 161:1/2, 163:1/2, 165:1/2, 179:1/2, 181:1/2, 189:1/2 |
| `w2_set_cs6_facets` | w2_set_cs6_facets.F:9 | 1 | 67/99 | 54-55, 87-92, 97-98, 113-121, 125-126, 166-180 | 53:1/2, 86:1/2, 95:1/2, 99:1/2, 105:1/2, 124:1/2, 144:1/2, 152:1/2, 165:1/2 |
| `w2_set_f2f_index` | w2_set_f2f_index.F:9 | 1 | 92/142 | 56-59, 62-65, 72-85, 92-94, 111-117, 186-193, 206-208, 246-253, 260-262 | 54:1/2, 61:1/2, 70:1/2, 90:1/2, 107:1/2, 110:1/2, 176:1/2, 196:1/2, 200:1/2, 204:1/2, 221:1/2, 245:1/2, 258:1/2 |
| `w2_set_map_cumsum` | w2_set_map_cumsum.F:8 | 1 | 85/138 | 89, 122-123, 135-136, 142-165, 173-200, 220-226 | 83:1/2, 88:1/2, 105:1/2, 121:1/2, 134:1/2, 140:1/2, 171:1/2, 219:1/2, 228:1/2, 234:1/4, 241:1/2, 245:1/2, 256:1/2 |
| `w2_set_map_tiles` | w2_set_map_tiles.F:14 | 1 | 83/122 | 69-72, 75-78, 87-89, 94-99, 105-118, 139-144, 193-202 | 68:1/2, 74:1/2, 85:1/2, 92:1/2, 102:1/2, 136:1/2, 172:1/2, 173:2/2, 175:1/2, 189:1/2, 204:1/2 |
| `w2_set_tile2tiles` | w2_set_tile2tiles.F:9 | 1 | 152/193 | 122-123, 254-260, 282-290, 295-298, 305-307, 318-330, 336-338 | 71:1/2, 101:1/2, 107:1/2, 119:1/2, 125:1/2, 147:1/2, 177:1/2, 221:1/2, 252:1/2, 271:1/2, 279:1/2, 294:1/2, 303:1/2, 317:1/2, 334:1/2 |

**pkg/exf** (28)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 6 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 2 | 78/101 | 231, 241-242, 309-323, 394-401, 430-458, 502-506, 513-515 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 32/149 | 58-60, 66-68, 72-83, 87-100, 111-115, 122-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 348-357, 399-402, 553-555, 558-561, 566-578, 616-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 119:1/2, 120:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 550:1/2, 557:1/2, 563:1/2, 607:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 26/76 | 57-59, 66-68, 75-77, 84-86, 91-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 2 | 16/25 | 42-47, 57-60 | 39:1/2, 41:1/2, 56:1/2 |
| `exf_diagnostics_init` | exf_diagnostics_init.F:6 | 1 | 193/197 | 104, 115, 234, 246 | 101:1/2, 112:1/2, 231:1/2, 243:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 22 | 12/22 | 66-78 | 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 11 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 2 | 14/14 | - | 68:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 11 | 9/55 | 64, 67-76, 84-103, 111-128, 132-141 | 65:1/2, 80:1/2, 106:1/2, 109:1/2, 131:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 22 | 15/97 | 94-196, 210-217, 235-259 | 209:1/2, 233:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 2 | 49/72 | 99-104, 144, 315-320, 495, 498, 501, 504, 507, 512, 517, 520, 525, 535, 545 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 493:1/2, 496:1/2, 499:1/2, 502:1/2, 505:1/2, 510:1/2, 515:1/2, 518:1/2, 523:1/2, 533:1/2, 543:1/2, 546:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 2 | 47/53 | 175-181, 331, 388 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 202:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 2 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 22 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 95/125 | 67-68, 125-143, 162, 173, 185-191, 195-201, 207-213, 262-268, 282-289, 359-366, 475, 489, 590-593 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 147:1/2, 149:1/2, 158:1/2, 159:1/2, 161:1/2, 170:1/2, 172:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 309:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 472:1/2, 474:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 617:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 17 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 39/47 | 93-98, 120-131 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 2 | 61/87 | 144-193, 220, 230, 241-246, 258, 268, 279-284 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 2 | 4/73 | 67-258 | 64:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 2 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 34 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 2 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 176/202 | 256-258, 331-337, 344-350, 358-364, 372-378, 385-395, 487-493, 500-501, 548-555, 625-626, 684-690, 769-771, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 11 | 8/8 | - | - |
| `exf_weight_sfx_diags` | exf_weight_sfx_diags.F:6 | 4 | 18/89 | 55-56, 60-80, 84-104, 112-115, 120-130, 134-144, 149-159, 171-174, 179-189, 193-203, 207-217, 221-231 | 51:1/2, 54:1/2, 59:1/2, 83:1/2, 111:1/2, 119:1/2, 133:1/2, 148:1/2, 169:1/2, 178:1/2, 192:1/2, 206:1/2, 220:1/2 |
| `exf_wind` | exf_wind.F:9 | 2 | 39/83 | 104-115, 129-148, 168, 176-178, 191-222 | 79:1/2, 102:1/2, 126:1/2, 160:1/2, 170:1/2, 183:1/2, 252:1/2 |

**pkg/generic_advdiff** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 48 | 178/222 | 199-202, 269-274, 279-282, 368-369, 414, 418, 422, 427-446, 635, 639, 643, 648-667, 824-827, 844-845, 848-849, 866, 991, 995, 999, 1004-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 411:1/2, 415:1/2, 419:1/2, 423:1/2, 479:1/2, 480:1/2, 610:1/2, 632:1/2, 636:1/2, 640:1/2, 644:1/2, 703:1/2, 734:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1000:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 720 | 77/189 | 181-184, 213-216, 236-244, 256-316, 329, 340, 385-445, 458, 469, 515-592, 614, 620, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diag_sufx` | gad_diagnostics_init.F:324 | 818 | 6/7 | 368 | 356:1/2 |
| `gad_diagnostics_init` | gad_diagnostics_init.F:6 | 1 | 81/87 | 53, 60, 181-185 | 52:1/2, 59:1/2, 180:1/2 |
| `gad_diagnostics_state` | gad_diagnostics_state.F:8 | 2 | 2/2 | - | - |
| `gad_dst3fl_adv_r` | gad_dst3fl_adv_r.F:7 | 672 | 25/25 | - | - |
| `gad_dst3fl_adv_x` | gad_dst3fl_adv_x.F:3 | 1056 | 27/27 | - | 59:1/2 |
| `gad_dst3fl_adv_y` | gad_dst3fl_adv_y.F:3 | 1056 | 27/27 | - | 55:1/2 |
| `gad_implicit_r` | gad_implicit_r.F:6 | 48 | 39/132 | 165-250, 270-284, 305-439 | 121:1/2, 162:1/2, 261:1/2, 269:1/2, 289:1/2, 293:1/2, 299:1/2, 304:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 98/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/gmredi** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 48 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 24 | 144/206 | 166-209, 529, 620-627, 689-691, 768, 816-837, 851-886, 965, 1013-1034, 1048-1083 | 144:1/2, 164:1/2, 526:1/2, 610:1/2, 688:1/2, 765:1/2, 815:1/2, 850:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_diagnostics_fill` | gmredi_diagnostics_fill.F:6 | 24 | 10/14 | 69-70, 78-79 | 52:1/2, 68:1/2, 77:1/2 |
| `gmredi_diagnostics_impl` | gmredi_diagnostics_impl.F:6 | 2 | 4/13 | 51-65 | 48:1/2, 50:1/2 |
| `gmredi_diagnostics_init` | gmredi_diagnostics_init.F:6 | 1 | 109/109 | - | - |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 2 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 19/25 | 63-64, 67-68, 82, 86 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 2 | 3/12 | 50-61 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 108/146 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 24 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 720 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 1056 | 54/87 | 176, 234, 397, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 720 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 720 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (8)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/118 | 116-126, 134, 163, 192-202, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/mdsio** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_facef_read_rs` | mdsio_facef_read.F:13 | 216 | 24/33 | 83-90, 105-108 | 82:1/2, 93:1/2, 95:1/4 |
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 14 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 24 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 109 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 4 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 66 | 104/222 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 413-414, 417-418, 435, 460-462, 487, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 404:1/2, 411:1/2, 415:1/2, 434:1/2, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 526:1/2, 540:1/2, 544:1/2, 572:1/2 |
| `mds_read_meta` | mdsio_read_meta.F:6 | 2 | 86/168 | 135, 154-159, 164-166, 169-178, 183-186, 197-200, 212-216, 224-227, 248-257, 273, 277-280, 287-297, 304, 323-334, 338-344, 351-352, 360-363, 383-400 | 98:1/2, 99:2/4, 127:1/2, 132:1/2, 133:1/2, 141:1/2, 144:1/2, 150:1/2, 161:1/2, 168:1/2, 182:1/2, 196:1/2, 211:1/2, 223:1/2, 241:1/2, 243:1/2, 247:3/7, 260:1/2, 272:1/2, 274:1/2, 286:1/2, 303:1/2, 320:1/2, 337:1/2, 349:1/2, 359:1/2, 371:2/4, 373:3/7, 375:1/2, 419:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 51/74 | 120-139, 148-149, 176-177, 180-181 | 112:1/2, 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 169:1/2, 173:1/2, 174:1/2, 178:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 86 | 99/233 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 457-458, 461-462, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 448:1/2, 455:1/2, 459:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2, 605:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 859 | 41/64 | 108, 135-139, 146-147, 157-159, 182-198, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 178:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (12)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hdiv` | mom_calc_hdiv.F:3 | 360 | 10/14 | 39-48, 74 | 38:1/2, 55:1/2 |
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 360 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 360 | 10/26 | 62-66, 88-137 | 61:1/2, 70:1/2 |
| `mom_calc_relvort3` | mom_calc_relvort3.F:4 | 360 | 41/41 | - | 80:1/2 |
| `mom_diagnostics_init` | mom_diagnostics_init.F:6 | 1 | 305/306 | 520 | 519:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 39/41 | 51-52 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 126:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 360 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_u_rviscflux` | mom_u_rviscflux.F:7 | 360 | 9/9 | - | - |
| `mom_u_sidedrag` | mom_u_sidedrag.F:7 | 360 | 10/19 | 62-96 | 58:1/2, 148:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 360 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_rviscflux` | mom_v_rviscflux.F:7 | 360 | 9/9 | - | - |
| `mom_v_sidedrag` | mom_v_sidedrag.F:7 | 360 | 10/20 | 62-93 | 58:1/2, 135:1/2 |

**pkg/mom_vecinv** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_vecinv` | mom_vecinv.F:10 | 360 | 181/255 | 237, 246, 332-340, 381, 403-406, 426, 670, 686-689, 709-711, 729-732, 749, 754, 758, 774, 779, 783, 801-803, 860-862, 879-890, 902-913, 932, 937-956, 988-993 | 234:1/2, 240:1/2, 305:1/2, 315:1/2, 331:1/2, 373:1/2, 400:1/2, 419:1/2, 441:1/2, 468:1/2, 484:1/2, 502:1/2, 553:1/2, 578:1/2, 594:1/2, 612:1/2, 669:1/2, 680:1/2, 683:1/2, 708:1/2, 723:1/2, 742:1/2, 745:1/2, 750:1/2, 755:1/2, 770:1/2, 775:1/2, 780:1/2, 800:1/2, 816:1/2, 823:1/2, 839:1/2, 859:1/2, 878:1/2, 900:1/2, 930:1/2, 936:1/2, 981:1/2, 984:1/2, 987:1/2 |
| `mom_vi_coriolis` | mom_vi_coriolis.F:7 | 360 | 13/47 | 73-122, 140-189 | 58:1/2, 125:1/2 |
| `mom_vi_hdissip` | mom_vi_hdissip.F:3 | 360 | 20/76 | 50-69, 105-108, 116-232 | 45:1/2, 49:1/2, 99:1/2, 114:1/2 |
| `mom_vi_u_coriolis` | mom_vi_u_coriolis.F:6 | 360 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_u_grad_ke` | mom_vi_u_grad_ke.F:3 | 360 | 5/5 | - | - |
| `mom_vi_u_vertshear` | mom_vi_u_vertshear.F:7 | 360 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |
| `mom_vi_v_coriolis` | mom_vi_v_coriolis.F:6 | 360 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_v_grad_ke` | mom_vi_v_grad_ke.F:3 | 360 | 5/5 | - | - |
| `mom_vi_v_vertshear` | mom_vi_v_vertshear.F:7 | 360 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 6 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 3 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 3 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 2 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 24 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 33 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 15 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 3 | 52/128 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 393 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 6 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 387 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 32 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 3 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 3 | 57/69 | 95, 123-132, 148, 178-179, 196-198 | 93:1/2, 122:1/2, 140:1/2, 147:1/2, 158:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 3 | 110/133 | 226-237, 243-278 | 143:1/2, 224:1/2, 242:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 33 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 15 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 3 | 64/67 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (19)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 1 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_3d_rl` | read_mflds.F:238 | 16 | 32/43 | 292-296, 332-338 | 291:1/2, 302:1/2, 310:1/2, 315:1/2, 326:1/2 |
| `read_mflds_check` | read_mflds.F:622 | 2 | 41/51 | 668-672, 714-718 | 667:1/2, 689:1/2, 696:2/4, 698:1/4, 705:2/4, 707:1/4, 713:1/2, 730:1/2, 743:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_mflds_lev_rl` | read_mflds.F:364 | 1 | 22/43 | 421-425, 439-449, 461-467 | 420:1/2, 431:1/2, 437:1/2, 454:1/2, 455:1/2, 471:1/2 |
| `read_mflds_set` | read_mflds.F:59 | 2 | 52/71 | 124-129, 173-180, 190-191, 195-196, 220-221 | 123:1/2, 133:1/2, 164:1/2, 167:1/2, 170:1/2, 186:1/2, 189:1/2, 194:1/2, 205:1/4, 211:2/4, 213:1/4, 219:1/2 |
| `read_rec_3d_rl` | read_rec.F:318 | 29 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 3 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 21 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 35 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 1 | 7/7 | - | - |

**pkg/seaice** (43)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_advdiff` | seaice_advdiff.F:10 | 2 | 40/61 | 311, 338, 365, 579-631 | 135:1/2, 297:1/2, 304:1/2, 324:1/2, 331:1/2, 351:1/2, 358:1/2 |
| `seaice_advection` | seaice_advection.F:14 | 72 | 159/227 | 169-172, 182-183, 233-238, 247-250, 322-323, 326, 374-379, 384, 388, 393-416, 452-459, 474-481, 588-593, 598, 602, 607-630, 665-673, 687-695, 766-771, 790 | 167:1/2, 177:1/2, 190:1/2, 216:1/2, 246:1/2, 289:1/2, 305:1/2, 325:1/2, 345:1/2, 371:1/2, 381:1/2, 385:1/2, 389:1/2, 438:1/2, 439:1/2, 451:1/2, 473:1/2, 559:1/2, 585:1/2, 595:1/2, 599:1/2, 603:1/2, 655:1/2, 677:1/2, 700:1/2, 707:1/2, 765:1/2, 775:1/2, 788:1/2 |
| `seaice_bottomdrag_coeffs` | seaice_bottomdrag_coeffs.F:9 | 4 | 3/29 | 82-155 | 80:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 24 | 6/6 | - | - |
| `seaice_calc_ice_strength` | seaice_calc_ice_strength.F:9 | 24 | 15/19 | 107-108, 111-112 | 105:1/2, 109:1/2 |
| `seaice_calc_strainrates` | seaice_calc_strainrates.F:11 | 4 | 32/39 | 159-186 | 79:1/2, 158:1/2 |
| `seaice_calc_viscosities` | seaice_calc_viscosities.F:8 | 4 | 63/67 | 126-135 | 95:1/2, 98:1/2, 117:1/2, 467:1/2 |
| `seaice_check` | seaice_check.F:12 | 1 | 87/485 | 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 222-225, 243-249, 253-259, 371-404, 411-457, 462-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 567-569, 700-702, 725-730, 733-741, 744-752, 777-780, 789-794, 819-826, 836-845, 852-869, 889-902, 907-916, 923-925, 935-940, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1078-1083, 1119-1124, 1130-1135, 1141-1144, 1147-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1205-1209, 1228-1231, 1234-1239, 1242-1247, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 221:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 459:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 776:1/2, 788:1/2, 798:1/2, 818:1/2, 828:1/2, 850:1/2, 888:1/2, 904:1/2, 921:1/2, 933:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1076:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1140:1/2, 1146:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1204:1/2, 1226:1/2, 1233:1/2, 1241:1/2, 1300:1/2 |
| `seaice_check_pickup` | seaice_check_pickup.F:6 | 1 | 3/62 | 74-214 | 73:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 2 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 2/2 | - | - |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 2/2 | - | - |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 2 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 2 | 2/2 | - | - |
| `seaice_diag_sufx` | seaice_diagnostics_init.F:935 | 76 | 10/11 | 969 | 966:1/2 |
| `seaice_diagnostics_init` | seaice_diagnostics_init.F:12 | 1 | 449/449 | - | - |
| `seaice_diagnostics_state` | seaice_diagnostics_state.F:6 | 2 | 11/35 | 60-87, 117-133, 136-152 | 53:1/2, 59:1/2, 116:1/2, 135:1/2 |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 2 | 93/213 | 158-168, 228-230, 254-257, 268-273, 310-317, 340, 450-469, 488, 507-561, 565-576, 585-612, 615-621, 624-631, 634-643, 646-663, 667-712, 716-733 | 136:1/2, 157:1/2, 227:1/2, 243:1/2, 244:1/2, 267:1/2, 285:1/2, 306:1/2, 309:1/2, 338:1/2, 344:1/2, 376:1/2, 389:1/2, 414:1/2, 449:1/2, 484:1/2, 503:1/2, 564:1/2, 581:1/2, 614:1/2, 623:1/2, 633:1/2, 645:1/2, 665:1/2, 715:1/2 |
| `seaice_freedrift` | seaice_freedrift.F:4 | 2 | 63/64 | 45 | 44:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 2 | 21/57 | 151-203, 248-258, 263-273 | 98:1/2, 146:1/2, 244:1/2, 247:1/2, 262:1/2 |
| `seaice_growth` | seaice_growth.F:15 | 2 | 337/413 | 336-337, 344, 749-759, 1042, 1464-1471, 1825, 1836, 1842-1848, 2280, 2285-2289, 2348, 2353-2357, 2445-2449, 2457-2461, 2467-2471, 2583-2587, 2592, 2607-2639, 2644-2660 | 335:1/2, 343:1/2, 567:1/2, 747:1/2, 790:1/2, 1040:1/2, 1299:1/2, 1462:1/2, 1528:1/2, 1698:1/2, 1822:1/2, 1833:1/2, 1837:1/2, 2278:1/2, 2282:1/2, 2297:1/2, 2302:1/2, 2346:1/2, 2350:1/2, 2424:1/2, 2444:1/2, 2455:1/2, 2466:1/2, 2482:1/2, 2582:1/2, 2591:1/2, 2603:1/2, 2606:1/2, 2643:1/2, 2667:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 60/79 | 60, 67, 80, 82-89, 286-292, 365-370 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2, 228:1/2, 278:1/2, 279:1/2, 296:1/2, 362:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 84/152 | 54, 237-242, 258-352, 463-471 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 447:1/2, 462:1/2 |
| `seaice_lsr` | seaice_lsr.F:24 | 2 | 173/228 | 185-186, 193, 234-241, 342-349, 597-601, 617-637, 651-685, 1120-1133, 1152-1160 | 184:1/2, 192:1/2, 233:1/2, 300:1/2, 328:1/2, 595:1/2, 607:1/2, 615:1/2, 616:1/2, 639:1/2, 646:1/2, 798:1/2, 892:1/2, 900:1/2, 976:1/2, 1116:1/2, 1139:1/2 |
| `seaice_lsr_calc_coeffs` | seaice_lsr.F:1294 | 4 | 74/85 | 1387-1390, 1402-1405, 1597-1600 | 1386:1/2, 1397:1/2, 1401:1/2, 1589:1/2 |
| `seaice_lsr_rhsu` | seaice_lsr.F:1616 | 48 | 27/31 | 1708-1723 | 1704:1/2, 1741:1/2 |
| `seaice_lsr_rhsv` | seaice_lsr.F:1771 | 48 | 27/31 | 1864-1879 | 1860:1/2, 1896:1/2 |
| `seaice_lsr_tridiagu` | seaice_lsr.F:1926 | 6936 | 28/28 | - | - |
| `seaice_lsr_tridiagv` | seaice_lsr.F:2071 | 6936 | 27/27 | - | - |
| `seaice_model` | seaice_model.F:13 | 2 | 52/76 | 92, 98, 105-115, 211-213, 281-286, 374-390 | 82:1/2, 85:1/2, 95:1/2, 102:1/2, 104:1/2, 125:1/2, 128:1/2, 131:1/2, 178:1/2, 209:1/2, 222:1/2, 224:1/2, 252:1/2, 255:1/2, 267:1/2, 275:1/2, 340:1/2, 366:1/2, 373:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 3 | 37/38 | 65 | 55:1/2, 58:1/2, 62:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 2 | 18/29 | 56, 69-92 | 55:1/2, 64:1/2 |
| `seaice_oceandrag_coeffs` | seaice_oceandrag_coeffs.F:9 | 4 | 17/18 | 64 | 63:1/2 |
| `seaice_output` | seaice_output.F:10 | 3 | 5/28 | 60-159 | 57:1/2, 174:1/2 |
| `seaice_read_pickup` | seaice_read_pickup.F:6 | 1 | 43/120 | 67-71, 89-95, 103-130, 141-174, 189-194, 203, 262-264, 269-273, 287-290, 325-327 | 63:1/2, 64:1/2, 87:1/2, 88:1/2, 101:1/2, 138:1/2, 186:1/2, 187:1/2, 201:1/2, 260:1/2, 267:1/2, 286:1/2, 302:1/2, 324:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 439/829 | 233-238, 460-468, 752-757, 773-827, 837-840, 846-856, 866, 868, 889-896, 912-913, 921-932, 936-942, 954-960, 965-971, 975-986, 992, 997, 1006-1012, 1017-1025, 1032-1036, 1044, 1070, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1517-1549 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 721:1/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 763:1/2, 767:1/2, 769:1/2, 771:1/2, 834:1/2, 835:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 951:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1511:1/2, 1514:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 2 | 44/44 | - | 393:1/2 |
| `seaice_residual` | seaice_lsr.F:1173 | 8 | 20/20 | - | 1260:1/2, 1278:1/2, 1282:1/2, 1283:1/2 |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 24 | 130/148 | 191, 298-299, 315, 387-388, 449-451, 514, 552-561 | 190:1/2, 217:1/2, 297:1/2, 309:1/2, 385:1/2, 445:1/2, 502:1/2, 513:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 234/258 | 109-111, 155-159, 287-311, 357-359, 377, 400, 480-482 | 45:1/2, 108:1/2, 153:1/2, 225:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 1 | 41/71 | 102-106, 160-168, 172-188, 194-200 | 78:1/2, 101:1/2, 110:1/2, 117:1/2, 121:1/2, 125:1/2, 152:1/2, 157:1/2, 159:1/2, 171:1/2, 193:1/2 |

**pkg/thsice** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `thsice_cost_init_varia` | thsice_cost_init_varia.F:3 | 1 | 6/6 | - | - |
| `thsice_readparms` | thsice_readparms.F:6 | 1 | 5/199 | 96-383 | 86:1/2, 88:1/2 |

**verification/global_ocean.cs32x15/code_ad** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_test` | cost_test.F:3 | 1 | 22/25 | 46-48 | 41:1/2 |

### Compiled but not executed (883 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_cycle_ebl, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_rl_recv_get_x, exch_rl_recv_get_y, exch_rl_send_put_x, exch_rl_send_put_y, exch_rs_recv_get_x, exch_rs_recv_get_y, exch_rs_send_put_x, exch_rs_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, fill_cs_corner_ag_rl, fill_cs_corner_uv_rl, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, memsync, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth2, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_gw, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_rho_l, diags_sound_speed, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, forcing_surf_relax, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cartesian_grid, ini_cg3d, ini_cylinder_grid, ini_local_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_pressure, ini_psurf, ini_salt, ini_sigma_hfac, ini_spherical_polar_grid, ini_theta, ini_vel, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, remove_mean_rl, remove_mean_rs, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, autodiff_findunit, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dummy_in_dynamics, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_addtime, cal_checkdate, cal_compdates, cal_convdate, cal_copydate, cal_daysformonth, cal_dayspermonth, cal_fulldate, cal_getmonthsrec, cal_init_fixed, cal_intdays, cal_intmonths, cal_intyears, cal_isleap, cal_monthsforyear, cal_monthsperyear, cal_numints, cal_printdate, cal_printerror, cal_set, cal_stepsforday, cal_stepsperday, cal_subdates, cal_summary, cal_time2dump, cal_timeinterval, cal_timepassed, cal_timestamp, cal_toseconds, cal_weekday
- pkg/cost: cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_calc_phivel, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_write_adj, diags_get_parms_i, diagstats_g_calc, diagstats_lm_calc, diagstats_mnc_out
- pkg/exch2: exch2_3d_r4, exch2_3d_r8, exch2_ad_get_r41, exch2_ad_get_r42, exch2_ad_get_r81, exch2_ad_get_r82, exch2_ad_get_rl1, exch2_ad_get_rl2, exch2_ad_get_rs1, exch2_ad_get_rs2, exch2_ad_put_r41, exch2_ad_put_r42, exch2_ad_put_r81, exch2_ad_put_r82, exch2_ad_put_rl1, exch2_ad_put_rl2, exch2_ad_put_rs1, exch2_ad_put_rs2, exch2_get_r41, exch2_get_r42, exch2_get_r81, exch2_get_r82, exch2_put_r41, exch2_put_r42, exch2_put_r81, exch2_put_r82, exch2_r41_cube, exch2_r41_cube_ad, exch2_r42_cube, exch2_r42_cube_ad, exch2_r81_cube, exch2_r81_cube_ad, exch2_r82_cube, exch2_r82_cube_ad, exch2_recv_r41, exch2_recv_r42, exch2_recv_r81, exch2_recv_r82, exch2_recv_rl1, exch2_recv_rl2, exch2_recv_rs1, exch2_recv_rs2, exch2_rl1_cube_ad, exch2_rl1_cube_b, exch2_rl1_cube_d, exch2_rl2_cube_ad, exch2_rl2_cube_b, exch2_rl2_cube_d, exch2_rs1_cube_ad, exch2_rs1_cube_b, exch2_rs1_cube_d, exch2_rs2_cube_ad, exch2_rs2_cube_b, exch2_rs2_cube_d, exch2_s3d_r4, exch2_s3d_r8, exch2_s3d_rs, exch2_send_r41, exch2_send_r42, exch2_send_r81, exch2_send_r82, exch2_send_rl1, exch2_send_rl2, exch2_send_rs1, exch2_send_rs2, exch2_sm_3d_r4, exch2_sm_3d_r8, exch2_sm_3d_rl, exch2_sm_3d_rs, exch2_uv_3d_r4, exch2_uv_3d_r8, exch2_uv_agrid_3d_r4, exch2_uv_agrid_3d_r8, exch2_uv_bgrid_3d_r4, exch2_uv_bgrid_3d_r8, exch2_uv_bgrid_3d_rl, exch2_uv_cgrid_3d_r4, exch2_uv_cgrid_3d_r8, exch2_uv_cgrid_3d_rl, exch2_uv_cgrid_3d_rs, exch2_uv_dgrid_3d_r4, exch2_uv_dgrid_3d_r8, exch2_uv_dgrid_3d_rl, exch2_uv_dgrid_3d_rs, exch2_z_3d_r4, exch2_z_3d_r8, exch2_z_3d_rl, find_gcd_n, w2_cumulsum_z_tile_rl, w2_print_e2setup, w2_set_gen_facets, w2_set_myown_facets, w2_set_single_facet
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_mnc_init, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_get_position, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_vecinv: mom_vi_del2uv, mom_vi_u_coriolis_c4, mom_vi_v_coriolis_c4
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_lev_rs, read_mflds_rename, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_stress, seaice_calc_stressdiv, seaice_cost_export, seaice_diffusion, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_growth_adx, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_preconditioner, seaice_prepare_ridging, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys
- pkg/thsice: thsice_advdiff, thsice_advection, thsice_albedo, thsice_ave, thsice_balance_frw, thsice_calc_thickn, thsice_check, thsice_check_conserv, thsice_cost_driver, thsice_cost_final, thsice_cost_test, thsice_diag_sufx, thsice_diagnostics_init, thsice_diagnostics_state, thsice_diffusion, thsice_do_advect, thsice_do_exch, thsice_extend, thsice_get_bulkf, thsice_get_exf, thsice_get_ocean, thsice_get_precip, thsice_get_velocity, thsice_impl_temp, thsice_ini_vars, thsice_init_fixed, thsice_main, thsice_map_exf, thsice_mnc_init, thsice_monitor, thsice_output, thsice_read_pickup, thsice_reshape_layers, thsice_salt_plume, thsice_slab_ocean, thsice_solve4temp, thsice_step_fwd, thsice_step_temp, thsice_turnoff_io, thsice_write_pickup

## global_ocean.cs32x15/input_ad.seaice_dynmix

Run `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.seaice_dynmix/job27855996` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.seaice_dynmix/job27855996/gcov-8773796/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice thermodynamics | seaice_growth (5 calls) | yes |
| LSR momentum solver | seaice_lsr (5 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (5 calls) | yes |
| cost function | global fc = 110845.859772594; terms: objf_test sum 1.108459e+05 | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, doResetHFactors, dumpInitAndLast, fluidIsWater, hasWetCSCorners, inAdExact, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, no_slip_sides, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, SEAICE_clipVeloctities, SEAICE_doOpenWaterGrowth, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICEadvArea, SEAICEadvHeff, SEAICEadvSnow, SEAICEmultiDimAdvection, SEAICEupdateOceanStress, SEAICEuseFlooding, SEAICEuseLSR, SEAICEuseTilt, snapshot_mdsio, stressIsOnCgrid, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useMultiDimAdvec, usePW79thermodynamics, useSEAICEinAdMode, usingZCoords, W2_useE2ioLayOut, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, balancePrintMean, balanceQnet, balanceSaltClimRelax, balanceThetaClimRelax, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, doSaltClimRelax, doThetaClimRelax, fluidIsAir, globalFiles, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, highOrderVorticity, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momImplVertAdv, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICE_2ndOrderBC, SEAICE_doOpenWaterMelt, SEAICE_growMeltByConv, SEAICE_maskRHS, SEAICE_mcPheeStepFunc, SEAICE_no_slip, SEAICE_salinityTracer, SEAICEheatConsFix, SEAICEmomAdvection, SEAICErestoreUnderIce, SEAICEuseBDF2, SEAICEuseDYNAMICSswitchInAd, SEAICEuseEVP, SEAICEuseJFNK, SEAICEuseKrylov, SEAICEuseLSRflex, SEAICEuseMultiTileSolver, SEAICEusePicardAsPrecon, SEAICEuseStrImpCpl, SEAICEuseTEM, SEAICEwriteState, startFromPickupAB2, tempImplVertAdv, tempSOM_Advection, twoDigitYear, upwindShear, upwindVorticity, use3Dsolver, useAbsVorticity, useApproxAdvectionInAdMode, useBiharmonicVisc, useCDscheme, useCoupler, useCtrlCostContribution, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useHB87stressCoupling, useJamartMomAdv, useKPPinAdMode, useMaykutSatVapPoly, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useSALT_PLUMEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid
- selectors: exf_adjMonSelect=1, monitorSelect=3, pCellMix_select=0, saltVertAdvScheme=30, SEAICEselectMetricTerms=2, select_ZenAlbedo=0, selectAddFluid=0, selectBalanceEmPmR=0, selectBotDragQuadr=-1, selectKEscheme=0, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempVertAdvScheme=30

### Executed routines (447)

**eesupp/src** (69)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 17204 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 178 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 963 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 29/29 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch_3d_rl` | exch_3d_rl.F:8 | 12 | 4/4 | - | - |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 940 | 4/4 | - | - |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 8 | 4/4 | - | - |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 10 | 4/4 | - | - |
| `exch_uv_agrid_3d_rs` | exch_uv_agrid_3d_rs.F:8 | 2 | 4/4 | - | - |
| `exch_uv_bgrid_3d_rs` | exch_uv_bgrid_3d_rs.F:8 | 1 | 4/4 | - | - |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 1514 | 3/3 | - | - |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 23 | 3/3 | - | - |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 3/3 | - | - |
| `exch_xy_rl` | exch_xy_rl.F:9 | 69 | 3/3 | - | - |
| `exch_xy_rs` | exch_xy_rs.F:9 | 105 | 3/3 | - | - |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 14 | 3/3 | - | - |
| `exch_z_3d_rs` | exch_z_3d_rs.F:8 | 3 | 4/4 | - | - |
| `fill_cs_corner_tr_rl` | fill_cs_corner_tr_rl.F:9 | 3900 | 45/62 | 93-117, 263 | 69:1/2, 71:1/2, 90:1/2, 193:1/2 |
| `fill_cs_corner_uv_rs` | fill_cs_corner_uv_rs.F:9 | 1980 | 38/38 | - | 65:1/2, 68:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 17738 | 2/2 | - | - |
| `get_periodic_interval` | get_periodic_interval.F:7 | 55 | 20/45 | 73-81, 87-93, 99-113 | 71:1/2, 86:1/2, 96:1/2 |
| `global_max_r8` | global_max.F:97 | 1787 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 50 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 67 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 2046 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 2843 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 14463 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 55/82 | 90-99, 114-119, 124-129, 134-139 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2 |
| `lcase` | utils.F:190 | 22 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 882 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1513 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 1238 | 11/19 | 44-50, 62-64 | 42:1/2, 60:1/2 |
| `nml_change_syntax` | nml_change_syntax.F:7 | 227 | 7/7 | - | - |
| `nml_set_terminator` | nml_set_terminator.F:7 | 5 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 10 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 63 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 165 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 309 | 43/49 | 648-650, 668-671 | 647:1/2, 662:1/2, 664:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 3936 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `timer_control` | timers.F:74 | 266 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 266 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 266 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 134 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 132 | 4/4 | - | - |
| `ucase` | utils.F:311 | 535 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 10 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 53 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 165 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 217 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 43 | 32/32 | - | 195:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (103)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth3` | adams_bashforth3.F:6 | 1800 | 18/29 | 85-87, 90-92, 103-109 | 84:1/2, 89:1/2, 101:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 22/51 | 55-71, 113-133 | 52:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 900 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 900 | 18/53 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 900 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 900 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 120 | 21/24 | 164-166 | 83:1/2, 132:1/2, 203:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 1800 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 900 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 900 | 22/81 | 68-80, 87-119, 162-261 | 63:1/2, 84:1/2, 158:1/2 |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 60 | 8/8 | - | - |
| `calc_grid_angles` | calc_grid_angles.F:7 | 1 | 29/37 | 76-83 | 75:1/2 |
| `calc_ivdc` | calc_ivdc.F:5 | 840 | 7/7 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 60 | 7/81 | 81, 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 900 | 37/155 | 149, 184, 194-197, 212-240, 268-610 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 193:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_r_star` | calc_r_star.F:10 | 7 | 66/119 | 68, 143-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 111:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 60 | 10/14 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 5 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `check_pickup` | check_pickup.F:8 | 1 | 11/124 | 67-73, 78-243 | 55:1/2, 57:1/2, 66:1/2, 77:1/2 |
| `config_check` | config_check.F:14 | 1 | 86/482 | 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 253-259, 266-271, 275-280, 344-349, 366-371, 417-423, 442-448, 454-460, 476-479, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 947-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 252:1/2, 264:1/2, 273:1/2, 342:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 475:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 946:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 451/545 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 426, 470-486, 540-548, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 968-974, 992-994, 1002-1008, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 60 | 24/57 | 79, 83, 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 120 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 900 | 7/24 | 76-122 | 73:1/2 |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 900 | 25/48 | 79-84, 123-125, 134-170 | 61:1/2, 76:1/2, 121:1/2, 132:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 5 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 5 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 5 | 94/128 | 257-264, 397-403, 467-471, 557, 771-784, 797-798, 830-833, 882, 934, 1043, 1116, 1119, 1123 | 248:1/2, 251:1/2, 256:1/2, 377:1/2, 421:1/2, 450:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 869:1/2, 878:1/2, 896:1/2, 932:1/2, 1030:1/2, 1032:1/2, 1096:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 5 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 6 | 11/19 | 97-110, 207, 242 | 96:1/2, 116:1/2, 193:1/2, 206:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 5 | 20/26 | 69-72, 82, 84, 115-117 | 66:1/2, 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 5 | 52/75 | 337-340, 358, 591-599, 684-700, 708-715 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 682:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 5 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 5 | 45/78 | 75, 100-107, 111-113, 176, 269-274, 296-343, 376-378 | 74:1/2, 82:1/2, 98:1/2, 110:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 268:1/2, 279:1/2, 364:1/2, 367:1/2 |
| `find_bulkmod` | find_rho.F:412 | 1740 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 1740 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 43 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 1740 | 16/16 | - | - |
| `forward_step` | forward_step.F:70 | 5 | 92/127 | 483-485, 508-512, 730-734, 767-769, 842-853, 868-870, 952-960, 980, 989-991, 1109-1111, 1176, 1201-1202 | 424:1/2, 468:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 832:1/2, 866:1/2, 897:1/2, 924:1/2, 939:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 1800 | 7/12 | 49-66 | 39:1/2, 40:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 900 | 22/22 | - | 59:1/2, 72:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 24/67 | 57-63, 70-79, 106-112, 121-180, 191 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_curvilinear_grid` | ini_curvilinear_grid.F:7 | 1 | 95/133 | 280, 353, 391-408, 430-447 | 259:1/2, 279:1/2, 344:1/2, 390:1/2, 429:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/68 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 32/32 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 50/50 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 5/10 | 29-33 | 28:1/2, 37:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 59/96 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 216-220, 228-229, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 214:1/2, 226:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 58/62 | 126-131, 136-138 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2, 176:1/2, 177:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/115 | 131, 133, 136-144, 192 | 130:1/2, 132:1/2, 134:1/2, 161:1/2, 163:1/2, 165:1/2, 167:1/2, 169:1/2, 175:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 183/198 | 210-212, 247-261, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 9/11 | 52-53 | 51:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 30/64 | 100-113, 146-179, 191-195, 206-209 | 99:1/2, 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 74/74 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 407/889 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 671-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1128-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1396, 1398, 1403-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1552-1555, 1558-1561, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 669:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1397:1/2, 1399:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1551:1/2, 1557:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 36/42 | 309-321, 326, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 6 | 56/70 | 147-150, 164-169, 212-214, 279-282 | 90:1/2, 93:1/2, 95:1/2, 146:1/2, 163:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 1080 | 18/36 | 95-117, 177-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 5 | 26/34 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 13/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 115-161, 168-173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 167:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 5 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 5 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 117/117 | - | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/119 | 132-140, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 252:1/2, 260:1/2, 264:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 380:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 30/40 | 160-162, 170-176, 530-532, 675-677 | 144:1/2, 158:1/2, 167:1/2, 193:1/2, 200:1/2, 202:1/2, 249:1/2, 251:1/2, 327:1/2, 329:1/2, 358:1/2, 365:1/2, 367:1/2, 510:1/2, 512:1/2, 528:1/2, 651:1/2, 655:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 18/23 | 169, 174, 452-454, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 270:1/2, 291:1/2, 293:1/2, 435:1/2, 450:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 19 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 13/13 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 3 | 33/37 | 68-70, 76 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 97:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 11/13 | 193, 225 | 124:1/2, 184:1/2, 191:1/2, 223:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 1740 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `read_pickup` | read_pickup.F:8 | 1 | 57/147 | 86-89, 110-116, 124-152, 161-235, 284, 293, 298-304, 307-313, 318-324, 327-333, 409, 448, 474-477, 565, 567 | 82:1/2, 83:1/2, 97:1/2, 107:1/2, 109:1/2, 122:1/2, 160:1/2, 276:1/2, 278:1/2, 282:1/2, 287:1/2, 291:1/2, 297:1/2, 306:1/2, 317:1/2, 326:1/2, 407:1/2, 446:1/2, 456:1/2, 460:1/2, 473:1/2, 564:1/2, 566:1/2 |
| `reset_nlfs_vars` | reset_nlfs_vars.F:6 | 5 | 8/11 | 48-50 | 47:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 5 | 28/64 | 67-70, 78, 85-127, 168-174 | 66:1/2, 77:1/2, 83:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 60 | 59/74 | 167, 169, 186, 326, 367-369, 380-390, 422-426, 493-501, 511 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 413:1/2, 479:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 99/165 | 56-57, 60-63, 71, 76, 109-115, 120-126, 171-175, 191-195, 202, 208, 214, 220, 226, 230, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 186:1/2, 188:1/2, 189:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 5 | 62/93 | 152-157, 164, 170-176, 228-234, 265, 269, 319, 327, 342-344, 355-366, 371 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `solve_tridiagonal` | solve_tridiagonal.F:10 | 120 | 32/38 | 249-251, 266-268 | 244:1/2, 259:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 60 | 59/74 | 169, 171, 188, 328, 369-371, 382-392, 424-428, 495-503, 513 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 481:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 43/43 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 5 | 46/63 | 158, 218-250, 395-399 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 206:1/2, 207:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 900 | 50/96 | 118-121, 140-143, 189-190, 220-223, 277-312, 328-332, 392-410, 413-414, 417-418 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 276:1/2, 321:1/2, 391:1/2, 412:1/2, 416:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 120 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 5 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/21 | 113 | 55:1/2, 78:1/2, 106:1/2, 112:1/2 |
| `update_etah` | update_etah.F:7 | 6 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 11 | 29/29 | - | - |
| `write_grid` | write_grid.F:12 | 1 | 48/66 | 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 57/101 | 146-149, 159-162, 167-182, 188-203, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 137:1/2, 139:1/2, 143:1/2, 145:1/2, 152:1/2, 156:1/2, 158:1/2, 166:1/2, 187:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 6 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (19)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 20 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 16 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 4 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 9 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 8 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 5 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 5 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 81/152 | 63-68, 127-132, 157, 165-172, 175-176, 237-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 4 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 4 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 8 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 6 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 5 | 2/2 | - | - |

**pkg/cal** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_getdate` | cal_getdate.F:3 | 1 | 7/23 | 55-88 | 47:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 7/23 | 79-117 | 65:1/2, 68:1/2, 70:1/2 |

**pkg/cost** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_accumulate_mean` | cost_accumulate_mean.F:3 | 5 | 13/13 | - | - |
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 7/7 | - | 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 50/50 | - | 82:1/2, 102:1/2, 110:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 22/22 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 5 | 4/4 | - | 121:1/2 |

**pkg/ctrl** (30)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 3 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 4 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 3 | 12/13 | 51 | 50:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 34/72 | 104-110, 124-128, 139-143, 182-186, 226-230, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 103:1/2, 121:1/2, 136:1/2, 179:1/2, 225:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/37 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/83 | 70-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 27 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 20 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 20 | 12/66 | 81-162, 190-197, 206-223 | 79:1/2, 177:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 24 | 7/11 | 129-130, 132-133 | 126:1/2, 128:1/2, 131:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 3 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 16 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 66/73 | 233-238, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 8 | 15/36 | 72-76, 87-88, 93-112, 118-124 | 86:1/2, 89:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 99/104 | 192-219 | 168:1/2, 183:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 5 | 51/57 | 82, 109, 111, 113, 115, 118 | 81:1/2, 83:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 3 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 5 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 35/51 | 155-166, 201, 219, 221, 360, 362 | 128:1/2, 154:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 392:1/2, 394:1/2, 405:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 65/80 | 151-153, 157-159, 162, 183-190, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 207/245 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 16 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 12 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 4 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 89/140 | 153-155, 161-171, 184-187, 218-222, 229-241, 257-261, 267-292, 304-306 | 146:1/2, 160:1/2, 178:1/2, 217:1/2, 228:1/2, 255:1/2, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 4 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 3605 | 10/16 | 43-45, 55-57 | 41:1/2, 42:2/2, 50:1/2, 52:1/2, 53:2/2 |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/372 | 123-653 | 109:1/2, 111:1/2 |

**pkg/exch2** (34)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exch2_3d_rl` | exch2_3d_rl.F:8 | 95 | 11/11 | - | - |
| `exch2_3d_rs` | exch2_3d_rs.F:8 | 105 | 11/11 | - | - |
| `exch2_check_depths` | exch2_check_depths.F:9 | 1 | 36/63 | 108-137, 149-156 | 85:1/2, 91:1/2, 97:1/2, 103:1/2, 106:1/2, 148:1/2 |
| `exch2_get_rl1` | exch2_get_rl1.F:8 | 63180 | 18/19 | 125 | 105:1/2, 129:4/8, 130:4/8, 131:4/6 |
| `exch2_get_rl2` | exch2_get_rl2.F:8 | 164376 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_rs1` | exch2_get_rs1.F:8 | 12312 | 18/19 | 125 | 105:1/2, 129:5/8, 130:4/8, 131:4/6 |
| `exch2_get_rs2` | exch2_get_rs2.F:8 | 2592 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_scal_bounds` | exch2_get_scal_bounds.F:7 | 151092 | 46/50 | 65, 82, 99, 116 | 62:1/2, 79:1/2, 96:1/2, 113:1/2 |
| `exch2_get_uv_bounds` | exch2_get_uv_bounds.F:7 | 333936 | 92/100 | 92, 110, 128, 146, 173, 184, 259-260 | 89:1/2, 107:1/2, 125:1/2, 143:1/2, 165:1/2, 172:1/2, 183:1/2, 233:1/2, 238:1/2 |
| `exch2_put_rl1` | exch2_put_rl1.F:8 | 63180 | 29/33 | 190-193 | 135:4/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rl2` | exch2_put_rl2.F:8 | 164376 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_put_rs1` | exch2_put_rs1.F:8 | 12312 | 29/33 | 190-193 | 135:5/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rs2` | exch2_put_rs2.F:8 | 2592 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_rl1_cube` | exch2_rl1_cube.F:7 | 1170 | 34/34 | - | - |
| `exch2_rl2_cube` | exch2_rl2_cube.F:8 | 3044 | 40/40 | - | - |
| `exch2_rs1_cube` | exch2_rs1_cube.F:7 | 228 | 34/34 | - | - |
| `exch2_rs2_cube` | exch2_rs2_cube.F:8 | 48 | 40/40 | - | - |
| `exch2_s3d_rl` | exch2_s3d_rl.F:8 | 940 | 10/10 | - | - |
| `exch2_uv_3d_rl` | exch2_uv_3d_rl.F:9 | 1522 | 42/42 | - | 79:1/2 |
| `exch2_uv_3d_rs` | exch2_uv_3d_rs.F:9 | 24 | 42/42 | - | 79:1/2 |
| `exch2_uv_agrid_3d_rl` | exch2_uv_agrid_3d_rl.F:8 | 10 | 46/46 | - | 68:1/2, 94:1/2, 148:1/2, 161:1/2 |
| `exch2_uv_agrid_3d_rs` | exch2_uv_agrid_3d_rs.F:8 | 2 | 46/46 | - | 94:1/2, 148:1/2, 161:1/2 |
| `exch2_uv_bgrid_3d_rs` | exch2_uv_bgrid_3d_rs.F:8 | 1 | 100/108 | 86-93 | 81:1/2, 83:1/2, 85:1/2, 134:1/2, 196:1/2, 211:1/2 |
| `exch2_z_3d_rs` | exch2_z_3d_rs.F:8 | 3 | 84/92 | 65-72 | 63:1/2, 64:1/2, 95:1/2, 106:1/2, 141:1/2, 158:1/2, 203:1/2, 220:1/2 |
| `w2_e2setup` | w2_e2setup.F:10 | 1 | 35/63 | 64-78, 95-101, 108-113, 119, 121, 123, 127 | 63:1/2, 94:1/2, 105:1/2, 106:2/2, 118:1/2, 120:1/2, 122:1/2, 124:1/2 |
| `w2_eeboot` | w2_eeboot.F:7 | 1 | 56/56 | - | 85:1/2, 106:1/2, 111:1/2 |
| `w2_map_procs` | w2_map_procs.F:7 | 1 | 55/69 | 101, 113-117, 121-125, 130, 152-155 | 75:1/2, 94:1/2, 99:1/2, 111:1/2, 119:1/2, 129:1/2, 149:1/2, 150:1/2, 157:1/2 |
| `w2_print_comm_sequence` | w2_print_comm_sequence.F:8 | 1 | 52/52 | - | - |
| `w2_readparms` | w2_readparms.F:9 | 1 | 63/89 | 69, 111-127, 133-134, 139-140, 166-172, 182-187, 190 | 66:1/2, 110:1/2, 132:1/2, 135:1/2, 161:1/2, 163:1/2, 165:1/2, 179:1/2, 181:1/2, 189:1/2 |
| `w2_set_cs6_facets` | w2_set_cs6_facets.F:9 | 1 | 67/99 | 54-55, 87-92, 97-98, 113-121, 125-126, 166-180 | 53:1/2, 86:1/2, 95:1/2, 99:1/2, 105:1/2, 124:1/2, 144:1/2, 152:1/2, 165:1/2 |
| `w2_set_f2f_index` | w2_set_f2f_index.F:9 | 1 | 92/142 | 56-59, 62-65, 72-85, 92-94, 111-117, 186-193, 206-208, 246-253, 260-262 | 54:1/2, 61:1/2, 70:1/2, 90:1/2, 107:1/2, 110:1/2, 176:1/2, 196:1/2, 200:1/2, 204:1/2, 221:1/2, 245:1/2, 258:1/2 |
| `w2_set_map_cumsum` | w2_set_map_cumsum.F:8 | 1 | 85/138 | 89, 122-123, 135-136, 142-165, 173-200, 220-226 | 83:1/2, 88:1/2, 105:1/2, 121:1/2, 134:1/2, 140:1/2, 171:1/2, 219:1/2, 228:1/2, 234:1/4, 241:1/2, 245:1/2, 256:1/2 |
| `w2_set_map_tiles` | w2_set_map_tiles.F:14 | 1 | 83/122 | 69-72, 75-78, 87-89, 94-99, 105-118, 139-144, 193-202 | 68:1/2, 74:1/2, 85:1/2, 92:1/2, 102:1/2, 136:1/2, 172:1/2, 173:2/2, 175:1/2, 189:1/2, 204:1/2 |
| `w2_set_tile2tiles` | w2_set_tile2tiles.F:9 | 1 | 152/193 | 122-123, 254-260, 282-290, 295-298, 305-307, 318-330, 336-338 | 71:1/2, 101:1/2, 107:1/2, 119:1/2, 125:1/2, 147:1/2, 177:1/2, 221:1/2, 252:1/2, 271:1/2, 279:1/2, 294:1/2, 303:1/2, 317:1/2, 334:1/2 |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 15 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 5 | 78/101 | 231, 241-242, 309-323, 394-401, 430-458, 502-506, 513-515 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 32/149 | 58-60, 66-68, 72-83, 87-100, 111-115, 122-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 348-357, 399-402, 553-555, 558-561, 566-578, 616-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 119:1/2, 120:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 550:1/2, 557:1/2, 563:1/2, 607:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 26/76 | 57-59, 66-68, 75-77, 84-86, 91-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 5 | 3/25 | 41-73 | 39:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 22 | 12/22 | 66-78 | 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 11 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 5 | 14/14 | - | 68:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 11 | 9/55 | 64, 67-76, 84-103, 111-128, 132-141 | 65:1/2, 80:1/2, 106:1/2, 109:1/2, 131:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 55 | 15/97 | 94-196, 210-217, 235-259 | 209:1/2, 233:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 5 | 49/72 | 99-104, 144, 315-320, 495, 498, 501, 504, 507, 512, 517, 520, 525, 535, 545 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 493:1/2, 496:1/2, 499:1/2, 502:1/2, 505:1/2, 510:1/2, 515:1/2, 518:1/2, 523:1/2, 533:1/2, 543:1/2, 546:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 5 | 47/53 | 175-181, 331, 388 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 202:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 5 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 22 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 93/125 | 67-68, 125-143, 162, 173, 185-191, 195-201, 207-213, 262-268, 282-289, 359-366, 475, 489, 590-593, 617-619 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 147:1/2, 149:1/2, 158:1/2, 159:1/2, 161:1/2, 170:1/2, 172:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 309:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 472:1/2, 474:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 17 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 39/47 | 93-98, 120-131 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 5 | 61/87 | 144-193, 220, 230, 241-246, 258, 268, 279-284 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 5 | 4/73 | 67-258 | 64:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 5 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 85 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 5 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 176/202 | 256-258, 331-337, 344-350, 358-364, 372-378, 385-395, 487-493, 500-501, 548-555, 625-626, 684-690, 769-771, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 11 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 5 | 39/83 | 104-115, 129-148, 168, 176-178, 191-222 | 79:1/2, 102:1/2, 126:1/2, 160:1/2, 170:1/2, 183:1/2, 252:1/2 |

**pkg/generic_advdiff** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 120 | 168/222 | 199-202, 213-219, 269-274, 279-282, 368-369, 414, 418, 423-446, 635, 639, 644-667, 824-827, 844-845, 848-849, 866, 991, 995, 1000-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 411:1/2, 415:1/2, 419:1/2, 479:1/2, 480:1/2, 610:1/2, 632:1/2, 636:1/2, 640:1/2, 703:1/2, 734:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 1800 | 70/189 | 181-184, 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 614, 620, 646-647, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 1680 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 2640 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 2640 | 17/17 | - | 82:1/2 |
| `gad_implicit_r` | gad_implicit_r.F:6 | 120 | 31/132 | 165-250, 270-284, 290-439 | 121:1/2, 162:1/2, 261:1/2, 269:1/2, 289:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 97/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/gmredi** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 120 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 60 | 141/206 | 145-147, 166-209, 529, 620-627, 689-691, 768, 816-837, 851-886, 965, 1013-1034, 1048-1083, 1122 | 144:1/2, 164:1/2, 526:1/2, 610:1/2, 688:1/2, 765:1/2, 815:1/2, 850:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 5 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 18/25 | 63-64, 67-68, 82, 86, 148 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 5 | 3/12 | 50-61 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 108/146 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 60 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 1800 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 2640 | 54/87 | 176, 234, 397, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 1800 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 1800 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (8)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/118 | 116-126, 134, 163, 192-202, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/mdsio** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_facef_read_rs` | mdsio_facef_read.F:13 | 216 | 24/33 | 83-90, 105-108 | 82:1/2, 93:1/2, 95:1/4 |
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 14 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 24 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 107 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 4 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 66 | 104/222 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 413-414, 417-418, 435, 460-462, 487, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 404:1/2, 411:1/2, 415:1/2, 434:1/2, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 526:1/2, 540:1/2, 544:1/2, 572:1/2 |
| `mds_read_meta` | mdsio_read_meta.F:6 | 2 | 86/168 | 135, 154-159, 164-166, 169-178, 183-186, 197-200, 212-216, 224-227, 248-257, 273, 277-280, 287-297, 304, 323-334, 338-344, 351-352, 360-363, 383-400 | 98:1/2, 99:2/4, 127:1/2, 132:1/2, 133:1/2, 141:1/2, 144:1/2, 150:1/2, 161:1/2, 168:1/2, 182:1/2, 196:1/2, 211:1/2, 223:1/2, 241:1/2, 243:1/2, 247:3/7, 260:1/2, 272:1/2, 274:1/2, 286:1/2, 303:1/2, 320:1/2, 337:1/2, 349:1/2, 359:1/2, 371:2/4, 373:3/7, 375:1/2, 419:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 51/74 | 120-139, 148-149, 176-177, 180-181 | 112:1/2, 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 169:1/2, 173:1/2, 174:1/2, 178:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 84 | 99/233 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 457-458, 461-462, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 448:1/2, 455:1/2, 459:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2, 605:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 859 | 41/64 | 108, 135-139, 146-147, 157-159, 182-198, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 178:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hdiv` | mom_calc_hdiv.F:3 | 900 | 10/14 | 39-48, 74 | 38:1/2, 55:1/2 |
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 900 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 900 | 10/26 | 62-66, 88-137 | 61:1/2, 70:1/2 |
| `mom_calc_relvort3` | mom_calc_relvort3.F:4 | 900 | 41/41 | - | 80:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 38/41 | 51-52, 230 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 126:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 900 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_u_rviscflux` | mom_u_rviscflux.F:7 | 900 | 9/9 | - | - |
| `mom_u_sidedrag` | mom_u_sidedrag.F:7 | 900 | 9/19 | 62-96, 149 | 58:1/2, 148:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 900 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_rviscflux` | mom_v_rviscflux.F:7 | 900 | 9/9 | - | - |
| `mom_v_sidedrag` | mom_v_sidedrag.F:7 | 900 | 9/20 | 62-93, 136 | 58:1/2, 135:1/2 |

**pkg/mom_vecinv** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_vecinv` | mom_vecinv.F:10 | 900 | 162/255 | 237, 246, 332-340, 381, 403-406, 426, 503-506, 613-616, 670, 686-689, 709-711, 724-732, 749, 754, 758, 774, 779, 783, 801-803, 817-818, 840-841, 860-862, 879-890, 902-913, 932, 937-956, 982-999 | 234:1/2, 240:1/2, 305:1/2, 315:1/2, 331:1/2, 373:1/2, 400:1/2, 419:1/2, 441:1/2, 468:1/2, 484:1/2, 502:1/2, 553:1/2, 578:1/2, 594:1/2, 612:1/2, 669:1/2, 680:1/2, 683:1/2, 708:1/2, 723:1/2, 742:1/2, 745:1/2, 750:1/2, 755:1/2, 770:1/2, 775:1/2, 780:1/2, 800:1/2, 816:1/2, 823:1/2, 839:1/2, 859:1/2, 878:1/2, 900:1/2, 930:1/2, 936:1/2, 981:1/2 |
| `mom_vi_coriolis` | mom_vi_coriolis.F:7 | 900 | 13/47 | 73-122, 140-189 | 58:1/2, 125:1/2 |
| `mom_vi_hdissip` | mom_vi_hdissip.F:3 | 900 | 18/76 | 50-69, 100-108, 116-232 | 45:1/2, 49:1/2, 99:1/2, 114:1/2 |
| `mom_vi_u_coriolis` | mom_vi_u_coriolis.F:6 | 900 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_u_grad_ke` | mom_vi_u_grad_ke.F:3 | 900 | 5/5 | - | - |
| `mom_vi_u_vertshear` | mom_vi_u_vertshear.F:7 | 900 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |
| `mom_vi_v_coriolis` | mom_vi_v_coriolis.F:6 | 900 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_v_grad_ke` | mom_vi_v_grad_ke.F:3 | 900 | 5/5 | - | - |
| `mom_vi_v_vertshear` | mom_vi_v_vertshear.F:7 | 900 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 12 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 6 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 6 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 5 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 60 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 66 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 30 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 6 | 52/128 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 702 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 12 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 690 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 62 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 6 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 6 | 57/69 | 95, 123-132, 148, 178-179, 196-198 | 93:1/2, 122:1/2, 140:1/2, 147:1/2, 158:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 6 | 110/133 | 226-237, 243-278 | 143:1/2, 224:1/2, 242:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 66 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 30 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 6 | 64/67 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (19)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 1 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_3d_rl` | read_mflds.F:238 | 14 | 22/43 | 292-296, 310-320, 332-338 | 291:1/2, 302:1/2, 308:1/2, 326:1/2, 342:1/2 |
| `read_mflds_check` | read_mflds.F:622 | 2 | 19/51 | 668-672, 688-724 | 667:1/2, 685:1/2, 730:1/2, 743:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_mflds_lev_rl` | read_mflds.F:364 | 1 | 22/43 | 421-425, 439-449, 461-467 | 420:1/2, 431:1/2, 437:1/2, 454:1/2, 455:1/2, 471:1/2 |
| `read_mflds_set` | read_mflds.F:59 | 2 | 52/71 | 124-129, 173-180, 190-191, 195-196, 220-221 | 123:1/2, 133:1/2, 164:1/2, 167:1/2, 170:1/2, 186:1/2, 189:1/2, 194:1/2, 205:1/4, 211:2/4, 213:1/4, 219:1/2 |
| `read_rec_3d_rl` | read_rec.F:318 | 29 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 3 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 21 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 33 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 1 | 7/7 | - | - |

**pkg/seaice** (40)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_advdiff` | seaice_advdiff.F:10 | 5 | 40/61 | 311, 338, 365, 579-631 | 135:1/2, 297:1/2, 304:1/2, 324:1/2, 331:1/2, 351:1/2, 358:1/2 |
| `seaice_advection` | seaice_advection.F:14 | 180 | 152/227 | 169-172, 182-183, 191, 233-238, 247-250, 322-323, 326, 374-379, 384, 389-416, 452-459, 474-481, 588-593, 598, 603-630, 665-673, 687-695, 766-771, 776-779, 790 | 167:1/2, 177:1/2, 190:1/2, 216:1/2, 246:1/2, 289:1/2, 305:1/2, 325:1/2, 345:1/2, 371:1/2, 381:1/2, 385:1/2, 438:1/2, 439:1/2, 451:1/2, 473:1/2, 559:1/2, 585:1/2, 595:1/2, 599:1/2, 655:1/2, 677:1/2, 700:1/2, 707:1/2, 765:1/2, 775:1/2, 788:1/2 |
| `seaice_bottomdrag_coeffs` | seaice_bottomdrag_coeffs.F:9 | 10 | 3/29 | 82-155 | 80:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 60 | 6/6 | - | - |
| `seaice_calc_ice_strength` | seaice_calc_ice_strength.F:9 | 60 | 15/19 | 107-108, 111-112 | 105:1/2, 109:1/2 |
| `seaice_calc_strainrates` | seaice_calc_strainrates.F:11 | 10 | 32/39 | 159-186 | 79:1/2, 158:1/2 |
| `seaice_calc_viscosities` | seaice_calc_viscosities.F:8 | 10 | 63/67 | 126-135 | 95:1/2, 98:1/2, 117:1/2, 467:1/2 |
| `seaice_check` | seaice_check.F:12 | 1 | 87/485 | 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 222-225, 243-249, 253-259, 371-404, 411-457, 462-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 567-569, 700-702, 725-730, 733-741, 744-752, 777-780, 789-794, 819-826, 836-845, 852-869, 889-902, 907-916, 923-925, 935-940, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1078-1083, 1119-1124, 1130-1135, 1141-1144, 1147-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1205-1209, 1228-1231, 1234-1239, 1242-1247, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 221:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 459:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 776:1/2, 788:1/2, 798:1/2, 818:1/2, 828:1/2, 850:1/2, 888:1/2, 904:1/2, 921:1/2, 933:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1076:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1140:1/2, 1146:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1204:1/2, 1226:1/2, 1233:1/2, 1241:1/2, 1300:1/2 |
| `seaice_check_pickup` | seaice_check_pickup.F:6 | 1 | 3/62 | 74-214 | 73:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 5 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 2/2 | - | - |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 2/2 | - | - |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 5 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 5 | 2/2 | - | - |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 5 | 70/213 | 158-168, 228-230, 254-257, 268-273, 310-317, 340, 415-733 | 136:1/2, 157:1/2, 227:1/2, 243:1/2, 244:1/2, 267:1/2, 285:1/2, 306:1/2, 309:1/2, 338:1/2, 344:1/2, 376:1/2, 389:1/2, 414:1/2 |
| `seaice_freedrift` | seaice_freedrift.F:4 | 5 | 63/64 | 45 | 44:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 5 | 17/57 | 151-203, 245-273 | 98:1/2, 146:1/2, 244:1/2 |
| `seaice_growth` | seaice_growth.F:15 | 5 | 293/413 | 336-337, 344, 568-570, 749-759, 1042, 1464-1471, 1825, 1836, 1842-1848, 2280, 2285-2289, 2304, 2348, 2353-2357, 2445-2449, 2457-2461, 2467-2471, 2483-2555, 2583-2587, 2592, 2607-2639, 2644-2660, 2671-2677 | 335:1/2, 343:1/2, 567:1/2, 747:1/2, 790:1/2, 1040:1/2, 1299:1/2, 1462:1/2, 1528:1/2, 1698:1/2, 1822:1/2, 1833:1/2, 1837:1/2, 2278:1/2, 2282:1/2, 2297:1/2, 2302:1/2, 2346:1/2, 2350:1/2, 2424:1/2, 2444:1/2, 2455:1/2, 2466:1/2, 2482:1/2, 2582:1/2, 2591:1/2, 2603:1/2, 2606:1/2, 2643:1/2, 2667:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 59/79 | 60, 67, 80, 82-89, 229, 286-292, 365-370 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2, 228:1/2, 278:1/2, 279:1/2, 296:1/2, 362:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 84/152 | 54, 237-242, 258-352, 463-471 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 447:1/2, 462:1/2 |
| `seaice_lsr` | seaice_lsr.F:24 | 5 | 173/228 | 185-186, 193, 234-241, 342-349, 597-601, 617-637, 651-685, 1120-1133, 1152-1160 | 184:1/2, 192:1/2, 233:1/2, 300:1/2, 328:1/2, 595:1/2, 607:1/2, 615:1/2, 616:1/2, 639:1/2, 646:1/2, 798:1/2, 892:1/2, 900:1/2, 976:1/2, 1116:1/2, 1139:1/2 |
| `seaice_lsr_calc_coeffs` | seaice_lsr.F:1294 | 10 | 74/85 | 1387-1390, 1402-1405, 1597-1600 | 1386:1/2, 1397:1/2, 1401:1/2, 1589:1/2 |
| `seaice_lsr_rhsu` | seaice_lsr.F:1616 | 120 | 27/31 | 1708-1723 | 1704:1/2, 1741:1/2 |
| `seaice_lsr_rhsv` | seaice_lsr.F:1771 | 120 | 27/31 | 1864-1879 | 1860:1/2, 1896:1/2 |
| `seaice_lsr_tridiagu` | seaice_lsr.F:1926 | 17880 | 28/28 | - | - |
| `seaice_lsr_tridiagv` | seaice_lsr.F:2071 | 17880 | 27/27 | - | - |
| `seaice_model` | seaice_model.F:13 | 5 | 40/76 | 92, 98, 103-115, 130-133, 211-213, 254-257, 281-286, 369-390 | 82:1/2, 85:1/2, 95:1/2, 102:1/2, 125:1/2, 128:1/2, 178:1/2, 209:1/2, 222:1/2, 224:1/2, 252:1/2, 267:1/2, 275:1/2, 340:1/2, 366:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 6 | 37/38 | 65 | 55:1/2, 58:1/2, 62:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 5 | 18/29 | 56, 69-92 | 55:1/2, 64:1/2 |
| `seaice_oceandrag_coeffs` | seaice_oceandrag_coeffs.F:9 | 10 | 17/18 | 64 | 63:1/2 |
| `seaice_output` | seaice_output.F:10 | 6 | 5/28 | 60-159 | 57:1/2, 174:1/2 |
| `seaice_read_pickup` | seaice_read_pickup.F:6 | 1 | 43/120 | 67-71, 89-95, 103-130, 141-174, 189-194, 203, 262-264, 269-273, 287-290, 325-327 | 63:1/2, 64:1/2, 87:1/2, 88:1/2, 101:1/2, 138:1/2, 186:1/2, 187:1/2, 201:1/2, 260:1/2, 267:1/2, 286:1/2, 302:1/2, 324:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 441/829 | 233-238, 460-468, 752-757, 773-827, 837-840, 846-856, 866, 868, 912-913, 921-932, 936-942, 954-960, 965-971, 975-986, 992, 997, 1006-1012, 1017-1025, 1032-1036, 1044, 1070, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1517-1549 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 721:1/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 763:1/2, 767:1/2, 769:1/2, 771:1/2, 834:1/2, 835:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 896:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 951:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1511:1/2, 1514:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 5 | 43/44 | 394 | 393:1/2 |
| `seaice_residual` | seaice_lsr.F:1173 | 20 | 20/20 | - | 1260:1/2, 1278:1/2, 1282:1/2, 1283:1/2 |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 60 | 132/148 | 191, 315, 387-388, 449-451, 514, 552-561 | 190:1/2, 217:1/2, 309:1/2, 385:1/2, 445:1/2, 502:1/2, 513:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 234/258 | 109-111, 155-159, 287-311, 357-359, 377, 400, 480-482 | 45:1/2, 108:1/2, 153:1/2, 225:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 1 | 41/71 | 102-106, 160-168, 172-188, 194-200 | 78:1/2, 101:1/2, 110:1/2, 117:1/2, 121:1/2, 125:1/2, 152:1/2, 157:1/2, 159:1/2, 171:1/2, 193:1/2 |

**pkg/thsice** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `thsice_cost_init_varia` | thsice_cost_init_varia.F:3 | 1 | 6/6 | - | - |
| `thsice_readparms` | thsice_readparms.F:6 | 1 | 5/199 | 96-383 | 86:1/2, 88:1/2 |

**verification/global_ocean.cs32x15/code_ad** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_test` | cost_test.F:3 | 1 | 22/25 | 46-48 | 41:1/2 |

### Compiled but not executed (940 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_cycle_ebl, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_rl_recv_get_x, exch_rl_recv_get_y, exch_rl_send_put_x, exch_rl_send_put_y, exch_rs_recv_get_x, exch_rs_recv_get_y, exch_rs_send_put_x, exch_rs_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, fill_cs_corner_ag_rl, fill_cs_corner_uv_rl, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, memsync, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth2, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_gw, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, forcing_surf_relax, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cartesian_grid, ini_cg3d, ini_cylinder_grid, ini_local_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_pressure, ini_psurf, ini_salt, ini_sigma_hfac, ini_spherical_polar_grid, ini_theta, ini_vel, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, remove_mean_rl, remove_mean_rs, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, autodiff_findunit, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dummy_in_dynamics, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_addtime, cal_checkdate, cal_compdates, cal_convdate, cal_copydate, cal_daysformonth, cal_dayspermonth, cal_fulldate, cal_getmonthsrec, cal_init_fixed, cal_intdays, cal_intmonths, cal_intyears, cal_isleap, cal_monthsforyear, cal_monthsperyear, cal_numints, cal_printdate, cal_printerror, cal_set, cal_stepsforday, cal_stepsperday, cal_subdates, cal_summary, cal_time2dump, cal_timeinterval, cal_timepassed, cal_timestamp, cal_toseconds, cal_weekday
- pkg/cost: cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/exch2: exch2_3d_r4, exch2_3d_r8, exch2_ad_get_r41, exch2_ad_get_r42, exch2_ad_get_r81, exch2_ad_get_r82, exch2_ad_get_rl1, exch2_ad_get_rl2, exch2_ad_get_rs1, exch2_ad_get_rs2, exch2_ad_put_r41, exch2_ad_put_r42, exch2_ad_put_r81, exch2_ad_put_r82, exch2_ad_put_rl1, exch2_ad_put_rl2, exch2_ad_put_rs1, exch2_ad_put_rs2, exch2_get_r41, exch2_get_r42, exch2_get_r81, exch2_get_r82, exch2_put_r41, exch2_put_r42, exch2_put_r81, exch2_put_r82, exch2_r41_cube, exch2_r41_cube_ad, exch2_r42_cube, exch2_r42_cube_ad, exch2_r81_cube, exch2_r81_cube_ad, exch2_r82_cube, exch2_r82_cube_ad, exch2_recv_r41, exch2_recv_r42, exch2_recv_r81, exch2_recv_r82, exch2_recv_rl1, exch2_recv_rl2, exch2_recv_rs1, exch2_recv_rs2, exch2_rl1_cube_ad, exch2_rl1_cube_b, exch2_rl1_cube_d, exch2_rl2_cube_ad, exch2_rl2_cube_b, exch2_rl2_cube_d, exch2_rs1_cube_ad, exch2_rs1_cube_b, exch2_rs1_cube_d, exch2_rs2_cube_ad, exch2_rs2_cube_b, exch2_rs2_cube_d, exch2_s3d_r4, exch2_s3d_r8, exch2_s3d_rs, exch2_send_r41, exch2_send_r42, exch2_send_r81, exch2_send_r82, exch2_send_rl1, exch2_send_rl2, exch2_send_rs1, exch2_send_rs2, exch2_sm_3d_r4, exch2_sm_3d_r8, exch2_sm_3d_rl, exch2_sm_3d_rs, exch2_uv_3d_r4, exch2_uv_3d_r8, exch2_uv_agrid_3d_r4, exch2_uv_agrid_3d_r8, exch2_uv_bgrid_3d_r4, exch2_uv_bgrid_3d_r8, exch2_uv_bgrid_3d_rl, exch2_uv_cgrid_3d_r4, exch2_uv_cgrid_3d_r8, exch2_uv_cgrid_3d_rl, exch2_uv_cgrid_3d_rs, exch2_uv_dgrid_3d_r4, exch2_uv_dgrid_3d_r8, exch2_uv_dgrid_3d_rl, exch2_uv_dgrid_3d_rs, exch2_z_3d_r4, exch2_z_3d_r8, exch2_z_3d_rl, find_gcd_n, w2_cumulsum_z_tile_rl, w2_print_e2setup, w2_set_gen_facets, w2_set_myown_facets, w2_set_single_facet
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_get_position, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_vecinv: mom_vi_del2uv, mom_vi_u_coriolis_c4, mom_vi_v_coriolis_c4
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_lev_rs, read_mflds_rename, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_stress, seaice_calc_stressdiv, seaice_cost_export, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_growth_adx, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_preconditioner, seaice_prepare_ridging, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys
- pkg/thsice: thsice_advdiff, thsice_advection, thsice_albedo, thsice_ave, thsice_balance_frw, thsice_calc_thickn, thsice_check, thsice_check_conserv, thsice_cost_driver, thsice_cost_final, thsice_cost_test, thsice_diag_sufx, thsice_diagnostics_init, thsice_diagnostics_state, thsice_diffusion, thsice_do_advect, thsice_do_exch, thsice_extend, thsice_get_bulkf, thsice_get_exf, thsice_get_ocean, thsice_get_precip, thsice_get_velocity, thsice_impl_temp, thsice_ini_vars, thsice_init_fixed, thsice_main, thsice_map_exf, thsice_mnc_init, thsice_monitor, thsice_output, thsice_read_pickup, thsice_reshape_layers, thsice_salt_plume, thsice_slab_ocean, thsice_solve4temp, thsice_step_fwd, thsice_step_temp, thsice_turnoff_io, thsice_write_pickup

## global_ocean.cs32x15/input_ad.thsice

Run `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.thsice/job27855996` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/global_ocean.cs32x15/input_ad.thsice/job27855996/gcov-8773796/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| thermodynamic sea ice (pkg/thsice) | thsice_main (5 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (5 calls) | yes |
| cost function | global fc = 110812.395422576; terms: objf_test sum 1.108124e+05 | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, doResetHFactors, dumpInitAndLast, fluidIsWater, hasWetCSCorners, inAdExact, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, no_slip_bottom, no_slip_sides, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltStepping, snapshot_mdsio, stressIsOnCgrid, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useExfCheckRange, useGMRediInAdMode, useHarmonicVisc, useMultiDimAdvec, usingZCoords, W2_useE2ioLayOut, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, balancePrintMean, balanceQnet, balanceSaltClimRelax, balanceThetaClimRelax, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, doSaltClimRelax, doThetaClimRelax, fluidIsAir, globalFiles, GM_AdvSeparate, GM_ExtraDiag, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, highOrderVorticity, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momImplVertAdv, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rigidLid, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltSOM_Advection, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFREEDRIFTswitchInAd, startFromPickupAB2, tempImplVertAdv, tempSOM_Advection, twoDigitYear, upwindShear, upwindVorticity, use3Dsolver, useAbsVorticity, useApproxAdvectionInAdMode, useBiharmonicVisc, useCDscheme, useCoupler, useCtrlCostContribution, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useJamartMomAdv, useKPPinAdMode, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useSALT_PLUMEinAdMode, useSEAICEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid
- selectors: exf_adjMonSelect=1, monitorSelect=3, pCellMix_select=0, saltVertAdvScheme=30, select_ZenAlbedo=0, selectAddFluid=0, selectBalanceEmPmR=0, selectBotDragQuadr=-1, selectKEscheme=0, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempVertAdvScheme=30

### Executed routines (427)

**eesupp/src** (68)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 8110 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 170 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 959 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 29/29 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch_3d_rl` | exch_3d_rl.F:8 | 11 | 4/4 | - | - |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 894 | 4/4 | - | - |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 8 | 4/4 | - | - |
| `exch_uv_agrid_3d_rs` | exch_uv_agrid_3d_rs.F:8 | 2 | 4/4 | - | - |
| `exch_uv_bgrid_3d_rs` | exch_uv_bgrid_3d_rs.F:8 | 1 | 4/4 | - | - |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 17 | 3/3 | - | - |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 18 | 3/3 | - | - |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 3/3 | - | - |
| `exch_xy_rl` | exch_xy_rl.F:9 | 60 | 3/3 | - | - |
| `exch_xy_rs` | exch_xy_rs.F:9 | 85 | 3/3 | - | - |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 14 | 3/3 | - | - |
| `exch_z_3d_rs` | exch_z_3d_rs.F:8 | 3 | 4/4 | - | - |
| `fill_cs_corner_tr_rl` | fill_cs_corner_tr_rl.F:9 | 3600 | 45/62 | 93-117, 263 | 69:1/2, 71:1/2, 90:1/2, 193:1/2 |
| `fill_cs_corner_uv_rs` | fill_cs_corner_uv_rs.F:9 | 1800 | 38/38 | - | 65:1/2, 68:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 8620 | 2/2 | - | - |
| `get_periodic_interval` | get_periodic_interval.F:7 | 60 | 20/45 | 73-81, 87-93, 99-113 | 71:1/2, 86:1/2, 96:1/2 |
| `global_max_r8` | global_max.F:97 | 381 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 50 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 27 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 2067 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 3024 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 14573 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 55/82 | 90-99, 114-119, 124-129, 134-139 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2 |
| `lcase` | utils.F:190 | 22 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 963 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 1794 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 1601 | 11/19 | 44-50, 62-64 | 42:1/2, 60:1/2 |
| `nml_change_syntax` | nml_change_syntax.F:7 | 210 | 7/7 | - | - |
| `nml_set_terminator` | nml_set_terminator.F:7 | 5 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 10 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 44 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 125 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 235 | 43/49 | 648-650, 668-671 | 647:1/2, 662:1/2, 664:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 3541 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `timer_control` | timers.F:74 | 256 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 256 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 256 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 129 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 127 | 4/4 | - | - |
| `ucase` | utils.F:311 | 515 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 4 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 36 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 125 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 144 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 42 | 32/32 | - | 195:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (103)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth3` | adams_bashforth3.F:6 | 1800 | 18/29 | 85-87, 90-92, 103-109 | 84:1/2, 89:1/2, 101:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 22/51 | 55-71, 113-133 | 52:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 900 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 900 | 18/53 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 900 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 900 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 120 | 21/24 | 164-166 | 83:1/2, 132:1/2, 203:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 1800 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 900 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 900 | 22/81 | 68-80, 87-119, 162-261 | 63:1/2, 84:1/2, 158:1/2 |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 60 | 8/8 | - | - |
| `calc_grid_angles` | calc_grid_angles.F:7 | 1 | 29/37 | 76-83 | 75:1/2 |
| `calc_ivdc` | calc_ivdc.F:5 | 840 | 7/7 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 60 | 7/81 | 81, 86-251 | 74:1/2, 80:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 900 | 37/155 | 149, 184, 194-197, 212-240, 268-610 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 193:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_r_star` | calc_r_star.F:10 | 7 | 66/119 | 68, 143-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 111:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 60 | 10/14 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 5 | 98/131 | 150-152, 191-192, 330-334, 340-346, 355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 338:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `check_pickup` | check_pickup.F:8 | 1 | 11/124 | 67-73, 78-243 | 55:1/2, 57:1/2, 66:1/2, 77:1/2 |
| `config_check` | config_check.F:14 | 1 | 86/482 | 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 253-259, 266-271, 275-280, 344-349, 366-371, 417-423, 442-448, 454-460, 476-479, 510-516, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 947-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 252:1/2, 264:1/2, 273:1/2, 342:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 475:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 946:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 451/545 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 426, 470-486, 540-548, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 968-974, 992-994, 1002-1008, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 60 | 24/57 | 79, 83, 158-167, 180-188, 202-204, 241-285 | 77:1/2, 81:1/2, 156:1/2, 179:1/2, 200:1/2, 240:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 120 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 900 | 7/24 | 76-122 | 73:1/2 |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 900 | 25/48 | 79-84, 123-125, 134-170 | 61:1/2, 76:1/2, 121:1/2, 132:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 5 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 5 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 5 | 94/128 | 257-264, 450-464, 471, 557, 771-784, 797-798, 830-833, 882, 934, 1043, 1116, 1119, 1123 | 248:1/2, 251:1/2, 256:1/2, 377:1/2, 397:1/2, 421:1/2, 467:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 869:1/2, 878:1/2, 896:1/2, 932:1/2, 1030:1/2, 1032:1/2, 1096:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 5 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 6 | 11/19 | 97-110, 194, 242 | 96:1/2, 116:1/2, 193:1/2, 206:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 5 | 20/26 | 69-72, 82, 84, 115-117 | 66:1/2, 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 5 | 52/75 | 337-340, 358, 591-599, 684-700, 708-715 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 682:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 5 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 5 | 45/78 | 75, 100-107, 111-113, 176, 269-274, 296-343, 376-378 | 74:1/2, 82:1/2, 98:1/2, 110:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 268:1/2, 279:1/2, 364:1/2, 367:1/2 |
| `find_bulkmod` | find_rho.F:412 | 1740 | 19/19 | - | - |
| `find_rho_2d` | find_rho.F:25 | 1740 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 43 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 1740 | 16/16 | - | - |
| `forward_step` | forward_step.F:70 | 5 | 92/127 | 483-485, 508-512, 730-734, 767-769, 842-853, 868-870, 952-960, 980, 989-991, 1109-1111, 1176, 1201-1202 | 424:1/2, 468:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 832:1/2, 866:1/2, 897:1/2, 924:1/2, 939:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 1800 | 7/12 | 49-66 | 39:1/2, 40:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 900 | 22/22 | - | 59:1/2, 72:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 24/67 | 57-63, 70-79, 106-112, 121-180, 191 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_curvilinear_grid` | ini_curvilinear_grid.F:7 | 1 | 95/133 | 280, 353, 391-408, 430-447 | 259:1/2, 279:1/2, 344:1/2, 390:1/2, 429:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/68 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 32/32 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 50/50 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 5/10 | 29-33 | 28:1/2, 37:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 59/96 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 216-220, 228-229, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 214:1/2, 226:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 58/62 | 126-131, 136-138 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2, 176:1/2, 177:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 104/115 | 131, 133, 136-144, 192 | 130:1/2, 132:1/2, 134:1/2, 161:1/2, 163:1/2, 165:1/2, 167:1/2, 169:1/2, 175:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 183/198 | 210-212, 247-261, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 9/11 | 52-53 | 51:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 30/64 | 100-113, 146-179, 191-195, 206-209 | 99:1/2, 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 74/74 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 407/889 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 671-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1128-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1396, 1398, 1403-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1552-1555, 1558-1561, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 669:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1397:1/2, 1399:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1551:1/2, 1557:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 36/42 | 309-321, 326, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 6 | 56/70 | 147-150, 164-169, 212-214, 279-282 | 90:1/2, 93:1/2, 95:1/2, 146:1/2, 163:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 1080 | 18/36 | 95-117, 177-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 5 | 26/34 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 13/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 115-161, 168-173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 167:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 5 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 5 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 117/117 | - | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/119 | 132-140, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 252:1/2, 260:1/2, 264:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 380:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 28/40 | 160-162, 170-176, 367-369, 512-514, 675-677 | 144:1/2, 158:1/2, 167:1/2, 193:1/2, 200:1/2, 202:1/2, 249:1/2, 251:1/2, 327:1/2, 329:1/2, 358:1/2, 365:1/2, 510:1/2, 528:1/2, 530:1/2, 651:1/2, 655:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 20/23 | 169, 174, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 270:1/2, 291:1/2, 293:1/2, 435:1/2, 450:1/2, 452:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 19 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 13/13 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 3 | 31/37 | 68-70, 76, 82-83 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 11/13 | 186, 225 | 124:1/2, 184:1/2, 191:1/2, 223:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 1740 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `read_pickup` | read_pickup.F:8 | 1 | 57/147 | 86-89, 110-116, 124-152, 161-235, 284, 293, 298-304, 307-313, 318-324, 327-333, 409, 448, 474-477, 565, 567 | 82:1/2, 83:1/2, 97:1/2, 107:1/2, 109:1/2, 122:1/2, 160:1/2, 276:1/2, 278:1/2, 282:1/2, 287:1/2, 291:1/2, 297:1/2, 306:1/2, 317:1/2, 326:1/2, 407:1/2, 446:1/2, 456:1/2, 460:1/2, 473:1/2, 564:1/2, 566:1/2 |
| `reset_nlfs_vars` | reset_nlfs_vars.F:6 | 5 | 8/11 | 48-50 | 47:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 5 | 28/64 | 67-70, 78, 85-127, 168-174 | 66:1/2, 77:1/2, 83:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 60 | 59/74 | 167, 169, 186, 326, 367-369, 380-390, 422-426, 493-501, 511 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 270:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 413:1/2, 479:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 99/165 | 56-57, 60-63, 71, 76, 109-115, 120-126, 171-175, 191-195, 202, 208, 214, 220, 226, 230, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 186:1/2, 188:1/2, 189:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 5 | 62/93 | 152-157, 164, 170-176, 228-234, 265, 269, 319, 327, 342-344, 355-366, 371 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2, 354:1/2, 370:1/2 |
| `solve_tridiagonal` | solve_tridiagonal.F:10 | 120 | 32/38 | 249-251, 266-268 | 244:1/2, 259:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 60 | 59/74 | 169, 171, 188, 328, 369-371, 382-392, 424-428, 495-503, 513 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 481:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 43/43 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 5 | 46/63 | 158, 218-250, 395-399 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 206:1/2, 207:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 900 | 50/96 | 118-121, 140-143, 189-190, 220-223, 277-312, 328-332, 392-410, 413-414, 417-418 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 188:1/2, 209:1/2, 219:1/2, 276:1/2, 321:1/2, 391:1/2, 412:1/2, 416:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 120 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 5 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/21 | 107 | 55:1/2, 78:1/2, 106:1/2, 112:1/2 |
| `update_etah` | update_etah.F:7 | 6 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 11 | 29/29 | - | - |
| `write_grid` | write_grid.F:12 | 1 | 48/66 | 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 57/101 | 146-149, 159-162, 167-182, 188-203, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 137:1/2, 139:1/2, 143:1/2, 145:1/2, 152:1/2, 156:1/2, 158:1/2, 166:1/2, 187:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 6 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (19)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 20 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 16 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 4 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 9 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 8 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 5 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 5 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 80/152 | 63-68, 127-132, 157, 159, 165-172, 175-176, 237-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 4 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 4 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 8 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 6 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 5 | 2/2 | - | - |

**pkg/cal** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_readparms` | cal_readparms.F:3 | 1 | 7/23 | 79-117 | 65:1/2, 68:1/2, 70:1/2 |

**pkg/cost** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_accumulate_mean` | cost_accumulate_mean.F:3 | 5 | 13/13 | - | - |
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 7/7 | - | 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 50/50 | - | 82:1/2, 102:1/2, 110:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 22/22 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 5 | 4/4 | - | 121:1/2 |

**pkg/ctrl** (30)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 3 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 4 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 3 | 12/13 | 51 | 50:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 34/72 | 104-110, 124-128, 139-143, 182-186, 226-230, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 103:1/2, 121:1/2, 136:1/2, 179:1/2, 225:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/37 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/83 | 70-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 27 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 20 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 20 | 12/66 | 81-162, 190-197, 206-223 | 79:1/2, 177:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 24 | 7/11 | 129-130, 132-133 | 126:1/2, 128:1/2, 131:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 3 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 16 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 66/73 | 233-238, 301, 303, 312-314 | 231:1/2, 251:1/2, 271:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 8 | 15/36 | 72-76, 87-88, 93-112, 118-124 | 86:1/2, 89:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 23/23 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 99/104 | 192-219 | 168:1/2, 183:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 5 | 51/57 | 82, 109, 111, 113, 115, 118 | 81:1/2, 83:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 3 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 5 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 35/51 | 155-166, 201, 219, 221, 360, 362 | 128:1/2, 154:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 359:1/2, 361:1/2, 392:1/2, 394:1/2, 405:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 65/80 | 151-153, 157-159, 162, 183-190, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 207/245 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 16 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 12 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 4 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 89/140 | 153-155, 161-171, 184-187, 218-222, 229-241, 257-261, 267-292, 304-306 | 146:1/2, 160:1/2, 178:1/2, 217:1/2, 228:1/2, 255:1/2, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 4 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 3600 | 10/16 | 43-45, 55-57 | 41:1/2, 42:2/2, 50:1/2, 52:1/2, 53:2/2 |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/372 | 123-653 | 109:1/2, 111:1/2 |

**pkg/exch2** (33)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exch2_3d_rl` | exch2_3d_rl.F:8 | 85 | 11/11 | - | - |
| `exch2_3d_rs` | exch2_3d_rs.F:8 | 85 | 11/11 | - | - |
| `exch2_check_depths` | exch2_check_depths.F:9 | 1 | 36/63 | 108-137, 149-156 | 85:1/2, 91:1/2, 97:1/2, 103:1/2, 106:1/2, 148:1/2 |
| `exch2_get_rl1` | exch2_get_rl1.F:8 | 57456 | 18/19 | 125 | 105:1/2, 129:4/8, 130:4/8, 131:4/6 |
| `exch2_get_rl2` | exch2_get_rl2.F:8 | 2700 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_rs1` | exch2_get_rs1.F:8 | 10152 | 18/19 | 125 | 105:1/2, 129:5/8, 130:4/8, 131:4/6 |
| `exch2_get_rs2` | exch2_get_rs2.F:8 | 2052 | 25/26 | 142 | 121:1/2, 146:4/8, 147:4/8, 148:4/6, 157:4/8, 158:4/8, 159:4/6 |
| `exch2_get_scal_bounds` | exch2_get_scal_bounds.F:7 | 135324 | 46/50 | 65, 82, 99, 116 | 62:1/2, 79:1/2, 96:1/2, 113:1/2 |
| `exch2_get_uv_bounds` | exch2_get_uv_bounds.F:7 | 9504 | 92/100 | 92, 110, 128, 146, 173, 184, 259-260 | 89:1/2, 107:1/2, 125:1/2, 143:1/2, 165:1/2, 172:1/2, 183:1/2, 233:1/2, 238:1/2 |
| `exch2_put_rl1` | exch2_put_rl1.F:8 | 57456 | 29/33 | 190-193 | 135:4/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rl2` | exch2_put_rl2.F:8 | 2700 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_put_rs1` | exch2_put_rs1.F:8 | 10152 | 29/33 | 190-193 | 135:5/8, 136:4/8, 137:4/6, 188:1/2 |
| `exch2_put_rs2` | exch2_put_rs2.F:8 | 2052 | 54/62 | 237-240, 325-328 | 167:4/8, 168:4/8, 169:4/6, 235:1/2, 255:4/8, 256:4/8, 257:4/6, 323:1/2 |
| `exch2_rl1_cube` | exch2_rl1_cube.F:7 | 1064 | 34/34 | - | - |
| `exch2_rl2_cube` | exch2_rl2_cube.F:8 | 50 | 40/40 | - | - |
| `exch2_rs1_cube` | exch2_rs1_cube.F:7 | 188 | 34/34 | - | - |
| `exch2_rs2_cube` | exch2_rs2_cube.F:8 | 38 | 40/40 | - | - |
| `exch2_s3d_rl` | exch2_s3d_rl.F:8 | 894 | 10/10 | - | - |
| `exch2_uv_3d_rl` | exch2_uv_3d_rl.F:9 | 25 | 42/42 | - | 79:1/2 |
| `exch2_uv_3d_rs` | exch2_uv_3d_rs.F:9 | 19 | 42/42 | - | 79:1/2 |
| `exch2_uv_agrid_3d_rs` | exch2_uv_agrid_3d_rs.F:8 | 2 | 46/46 | - | 94:1/2, 148:1/2, 161:1/2 |
| `exch2_uv_bgrid_3d_rs` | exch2_uv_bgrid_3d_rs.F:8 | 1 | 100/108 | 86-93 | 81:1/2, 83:1/2, 85:1/2, 134:1/2, 196:1/2, 211:1/2 |
| `exch2_z_3d_rs` | exch2_z_3d_rs.F:8 | 3 | 84/92 | 65-72 | 63:1/2, 64:1/2, 95:1/2, 106:1/2, 141:1/2, 158:1/2, 203:1/2, 220:1/2 |
| `w2_e2setup` | w2_e2setup.F:10 | 1 | 35/63 | 64-78, 95-101, 108-113, 119, 121, 123, 127 | 63:1/2, 94:1/2, 105:1/2, 106:2/2, 118:1/2, 120:1/2, 122:1/2, 124:1/2 |
| `w2_eeboot` | w2_eeboot.F:7 | 1 | 56/56 | - | 85:1/2, 106:1/2, 111:1/2 |
| `w2_map_procs` | w2_map_procs.F:7 | 1 | 55/69 | 101, 113-117, 121-125, 130, 152-155 | 75:1/2, 94:1/2, 99:1/2, 111:1/2, 119:1/2, 129:1/2, 149:1/2, 150:1/2, 157:1/2 |
| `w2_print_comm_sequence` | w2_print_comm_sequence.F:8 | 1 | 52/52 | - | - |
| `w2_readparms` | w2_readparms.F:9 | 1 | 63/89 | 69, 111-127, 133-134, 139-140, 166-172, 182-187, 190 | 66:1/2, 110:1/2, 132:1/2, 135:1/2, 161:1/2, 163:1/2, 165:1/2, 179:1/2, 181:1/2, 189:1/2 |
| `w2_set_cs6_facets` | w2_set_cs6_facets.F:9 | 1 | 67/99 | 54-55, 87-92, 97-98, 113-121, 125-126, 166-180 | 53:1/2, 86:1/2, 95:1/2, 99:1/2, 105:1/2, 124:1/2, 144:1/2, 152:1/2, 165:1/2 |
| `w2_set_f2f_index` | w2_set_f2f_index.F:9 | 1 | 92/142 | 56-59, 62-65, 72-85, 92-94, 111-117, 186-193, 206-208, 246-253, 260-262 | 54:1/2, 61:1/2, 70:1/2, 90:1/2, 107:1/2, 110:1/2, 176:1/2, 196:1/2, 200:1/2, 204:1/2, 221:1/2, 245:1/2, 258:1/2 |
| `w2_set_map_cumsum` | w2_set_map_cumsum.F:8 | 1 | 85/138 | 89, 122-123, 135-136, 142-165, 173-200, 220-226 | 83:1/2, 88:1/2, 105:1/2, 121:1/2, 134:1/2, 140:1/2, 171:1/2, 219:1/2, 228:1/2, 234:1/4, 241:1/2, 245:1/2, 256:1/2 |
| `w2_set_map_tiles` | w2_set_map_tiles.F:14 | 1 | 83/122 | 69-72, 75-78, 87-89, 94-99, 105-118, 139-144, 193-202 | 68:1/2, 74:1/2, 85:1/2, 92:1/2, 102:1/2, 136:1/2, 172:1/2, 173:2/2, 175:1/2, 189:1/2, 204:1/2 |
| `w2_set_tile2tiles` | w2_set_tile2tiles.F:9 | 1 | 152/193 | 122-123, 254-260, 282-290, 295-298, 305-307, 318-330, 336-338 | 71:1/2, 101:1/2, 107:1/2, 119:1/2, 125:1/2, 147:1/2, 177:1/2, 221:1/2, 252:1/2, 271:1/2, 279:1/2, 294:1/2, 303:1/2, 317:1/2, 334:1/2 |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 15 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 5 | 78/101 | 231, 241-242, 309-323, 394-401, 430-458, 502-506, 513-515 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 32/149 | 58-60, 66-68, 72-83, 87-100, 111-115, 122-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 348-357, 399-402, 553-555, 558-561, 566-578, 616-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 119:1/2, 120:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 550:1/2, 557:1/2, 563:1/2, 607:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 26/76 | 57-59, 66-68, 75-77, 84-86, 91-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 5 | 3/25 | 41-73 | 39:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 24 | 22/22 | - | 38:1/2, 41:1/2, 74:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 12 | 26/26 | - | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 5 | 14/14 | - | 68:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 12 | 9/55 | 64, 67-76, 84-103, 111-128, 132-141 | 65:1/2, 80:1/2, 106:1/2, 109:1/2, 131:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 60 | 15/97 | 94-196, 210-217, 235-259 | 209:1/2, 233:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 5 | 54/72 | 99-104, 144, 495, 498, 501, 504, 507, 512, 517, 520, 525, 535, 545 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 493:1/2, 496:1/2, 499:1/2, 502:1/2, 505:1/2, 510:1/2, 515:1/2, 518:1/2, 523:1/2, 533:1/2, 543:1/2, 546:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 5 | 48/53 | 175-181, 331 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 202:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 5 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 24 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 83/125 | 67-68, 88-114, 125-143, 162, 173, 185-191, 195-201, 207-213, 262-268, 283, 359-366, 475, 489, 590-593, 617-619 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 147:1/2, 149:1/2, 158:1/2, 159:1/2, 161:1/2, 170:1/2, 172:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 282:1/2, 307:1/2, 309:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 472:1/2, 474:1/2, 486:1/2, 488:1/2, 588:1/2, 610:1/2, 615:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 17 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 39/47 | 93-98, 120-131 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 5 | 61/87 | 144-193, 220, 230, 241-246, 258, 268, 279-284 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 5 | 4/73 | 67-258 | 64:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 5 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 85 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 5 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 178/202 | 256-258, 331-337, 344-350, 358-364, 372-378, 385-395, 487-493, 500-501, 625-626, 684-690, 769-771, 803-805 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 12 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 5 | 42/83 | 104-115, 129-148, 168, 191-222 | 79:1/2, 102:1/2, 126:1/2, 160:1/2, 183:1/2, 252:1/2 |

**pkg/generic_advdiff** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 120 | 168/222 | 199-202, 213-219, 269-274, 279-282, 368-369, 414, 418, 423-446, 635, 639, 644-667, 824-827, 844-845, 848-849, 866, 991, 995, 1000-1018, 1081-1083 | 194:1/2, 212:1/2, 252:1/2, 278:1/2, 334:1/2, 349:1/2, 389:1/2, 411:1/2, 415:1/2, 419:1/2, 479:1/2, 480:1/2, 610:1/2, 632:1/2, 636:1/2, 640:1/2, 703:1/2, 734:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 988:1/2, 992:1/2, 996:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 1800 | 70/189 | 181-184, 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 614, 620, 646-647, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 345:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 474:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 625:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 1680 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 2400 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 2400 | 17/17 | - | 82:1/2 |
| `gad_implicit_r` | gad_implicit_r.F:6 | 120 | 31/132 | 165-250, 270-284, 290-439 | 121:1/2, 162:1/2, 261:1/2, 269:1/2, 289:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 97/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/gmredi** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 120 | 9/16 | 63-82 | 49:1/2, 54:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 60 | 141/206 | 145-147, 166-209, 529, 620-627, 689-691, 768, 816-837, 851-886, 965, 1013-1034, 1048-1083, 1122 | 144:1/2, 164:1/2, 526:1/2, 610:1/2, 688:1/2, 765:1/2, 815:1/2, 850:1/2, 962:1/2, 1012:1/2, 1047:1/2, 1121:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 50/165 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 5 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 18/25 | 63-64, 67-68, 82, 86, 148 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2, 147:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 5 | 3/12 | 50-61 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 108/146 | 99-104, 235-239, 243-247, 256, 259, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 254:1/2, 257:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 60 | 4/20 | 61-94 | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 1800 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 2640 | 54/87 | 176, 234, 397, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 1800 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 1800 | 13/43 | 97-100, 144-185, 200-233, 243-255 | 95:1/2, 104:1/2, 143:1/2, 196:1/2, 241:1/2 |

**pkg/grdchk** (8)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/118 | 116-126, 134, 163, 192-202, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/mdsio** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_facef_read_rs` | mdsio_facef_read.F:13 | 216 | 24/33 | 83-90, 105-108 | 82:1/2, 93:1/2, 95:1/4 |
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 34 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 24 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 115 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 4 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 71 | 104/222 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 413-414, 417-418, 435, 460-462, 487, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 404:1/2, 411:1/2, 415:1/2, 434:1/2, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 526:1/2, 540:1/2, 544:1/2, 572:1/2 |
| `mds_read_meta` | mdsio_read_meta.F:6 | 1 | 86/168 | 135, 154-159, 164-166, 169-178, 183-186, 197-200, 212-216, 224-227, 248-257, 273, 277-280, 287-297, 304, 323-334, 338-344, 351-352, 360-363, 383-400 | 98:1/2, 99:2/4, 127:1/2, 132:1/2, 133:1/2, 141:1/2, 144:1/2, 150:1/2, 161:1/2, 168:1/2, 182:1/2, 196:1/2, 211:1/2, 223:1/2, 241:1/2, 243:1/2, 247:3/7, 260:1/2, 272:1/2, 274:1/2, 286:1/2, 303:1/2, 320:1/2, 337:1/2, 349:1/2, 359:1/2, 371:2/4, 373:3/7, 375:1/2, 419:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 1 | 51/74 | 120-139, 148-149, 176-177, 180-181 | 112:1/2, 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 169:1/2, 173:1/2, 174:1/2, 178:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 107 | 99/233 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 457-458, 461-462, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 448:1/2, 455:1/2, 459:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2, 605:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 1195 | 41/64 | 108, 135-139, 146-147, 157-159, 182-198, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 178:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hdiv` | mom_calc_hdiv.F:3 | 900 | 10/14 | 39-48, 74 | 38:1/2, 55:1/2 |
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 900 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 900 | 10/26 | 62-66, 88-137 | 61:1/2, 70:1/2 |
| `mom_calc_relvort3` | mom_calc_relvort3.F:4 | 900 | 41/41 | - | 80:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 38/41 | 51-52, 230 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2, 126:1/2, 229:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 900 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_u_rviscflux` | mom_u_rviscflux.F:7 | 900 | 9/9 | - | - |
| `mom_u_sidedrag` | mom_u_sidedrag.F:7 | 900 | 9/19 | 62-96, 149 | 58:1/2, 148:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 900 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_rviscflux` | mom_v_rviscflux.F:7 | 900 | 9/9 | - | - |
| `mom_v_sidedrag` | mom_v_sidedrag.F:7 | 900 | 9/20 | 62-93, 136 | 58:1/2, 135:1/2 |

**pkg/mom_vecinv** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_vecinv` | mom_vecinv.F:10 | 900 | 162/255 | 237, 246, 332-340, 381, 403-406, 426, 503-506, 613-616, 670, 686-689, 709-711, 724-732, 749, 754, 758, 774, 779, 783, 801-803, 817-818, 840-841, 860-862, 879-890, 902-913, 932, 937-956, 982-999 | 234:1/2, 240:1/2, 305:1/2, 315:1/2, 331:1/2, 373:1/2, 400:1/2, 419:1/2, 441:1/2, 468:1/2, 484:1/2, 502:1/2, 553:1/2, 578:1/2, 594:1/2, 612:1/2, 669:1/2, 680:1/2, 683:1/2, 708:1/2, 723:1/2, 742:1/2, 745:1/2, 750:1/2, 755:1/2, 770:1/2, 775:1/2, 780:1/2, 800:1/2, 816:1/2, 823:1/2, 839:1/2, 859:1/2, 878:1/2, 900:1/2, 930:1/2, 936:1/2, 981:1/2 |
| `mom_vi_coriolis` | mom_vi_coriolis.F:7 | 900 | 13/47 | 73-122, 140-189 | 58:1/2, 125:1/2 |
| `mom_vi_hdissip` | mom_vi_hdissip.F:3 | 900 | 18/76 | 50-69, 100-108, 116-232 | 45:1/2, 49:1/2, 99:1/2, 114:1/2 |
| `mom_vi_u_coriolis` | mom_vi_u_coriolis.F:6 | 900 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_u_grad_ke` | mom_vi_u_grad_ke.F:3 | 900 | 5/5 | - | - |
| `mom_vi_u_vertshear` | mom_vi_u_vertshear.F:7 | 900 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |
| `mom_vi_v_coriolis` | mom_vi_v_coriolis.F:6 | 900 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_v_grad_ke` | mom_vi_v_grad_ke.F:3 | 900 | 5/5 | - | - |
| `mom_vi_v_vertshear` | mom_vi_v_vertshear.F:7 | 900 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |

**pkg/monitor** (24)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 12 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 6 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 6 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 5 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 60 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 48 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 30 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 6 | 52/128 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 756 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 6 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 750 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 62 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 6 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_latbnd_rl` | mon_stats_latbnd_rl.F:8 | 30 | 54/62 | 62-68, 72-73 | 60:1/2, 71:1/2, 128:1/2, 132:1/2, 136:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 6 | 57/69 | 95, 123-132, 148, 178-179, 196-198 | 93:1/2, 122:1/2, 140:1/2, 147:1/2, 158:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 6 | 110/133 | 226-237, 243-278 | 143:1/2, 224:1/2, 242:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 36 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 30 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 6 | 64/67 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |
| `nlatbnd` | mon_stats_latbnd_rl.F:149 | 184320 | 5/5 | - | - |

**pkg/rw** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 1 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_3d_rl` | read_mflds.F:238 | 9 | 22/43 | 292-296, 310-320, 332-338 | 291:1/2, 302:1/2, 308:1/2, 326:1/2, 342:1/2 |
| `read_mflds_check` | read_mflds.F:622 | 1 | 19/51 | 668-672, 688-724 | 667:1/2, 685:1/2, 730:1/2, 743:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_mflds_set` | read_mflds.F:59 | 1 | 52/71 | 124-129, 173-180, 190-191, 195-196, 220-221 | 123:1/2, 133:1/2, 164:1/2, 166:1/2, 167:1/2, 170:1/2, 186:1/2, 189:1/2, 194:1/2, 205:1/4, 211:2/4, 213:1/4, 219:1/2 |
| `read_rec_3d_rl` | read_rec.F:318 | 40 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 23 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 21 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 37 | 7/7 | - | - |

**pkg/seaice** (3)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 48/152 | 54, 171-471 | 53:1/2, 165:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 5/829 | 241-1559 | 231:1/2, 233:1/2 |

**pkg/thsice** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `thsice_albedo` | thsice_albedo.F:6 | 60 | 24/35 | 94-98, 134, 146-151, 159-161 | 92:1/2, 129:1/2, 142:1/2, 156:1/2 |
| `thsice_ave` | thsice_ave.F:6 | 60 | 3/9 | 65-74 | 62:1/2 |
| `thsice_calc_thickn` | thsice_calc_thickn.F:10 | 60 | 248/285 | 438, 561-562, 633-634, 656-663, 695-701, 725-726, 786-789, 832-841, 923-934, 1088 | 391:1/2, 437:1/2, 557:1/2, 629:1/2, 654:1/2, 692:1/2, 723:1/2, 739:1/2, 744:1/2, 785:1/2, 826:1/2, 830:1/2, 866:1/2, 917:1/2, 921:1/2, 1008:1/2, 1073:1/2, 1087:1/2 |
| `thsice_check` | thsice_check.F:10 | 1 | 12/39 | 51-56, 60-63, 81-85, 92-94, 111-119, 124-127 | 39:1/2, 47:1/2, 58:1/2, 74:1/2, 90:1/2, 107:1/2, 122:1/2 |
| `thsice_cost_final` | thsice_cost_final.F:6 | 1 | 9/9 | - | - |
| `thsice_cost_init_varia` | thsice_cost_init_varia.F:3 | 1 | 6/6 | - | - |
| `thsice_extend` | thsice_extend.F:9 | 60 | 36/48 | 165-166, 176-183, 223-227 | 163:1/2, 171:1/2, 185:1/2, 221:1/2 |
| `thsice_get_exf` | thsice_get_exf.F:12 | 120 | 59/122 | 267-431, 492-495 | 263:1/2, 489:1/2 |
| `thsice_get_ocean` | thsice_get_ocean.F:9 | 60 | 21/36 | 70-86, 124-133 | 61:1/2, 62:1/2, 111:1/2 |
| `thsice_ini_vars` | thsice_ini_vars.F:9 | 1 | 46/73 | 120-151, 172-178 | 112:1/2, 171:1/2 |
| `thsice_init_fixed` | thsice_init_fixed.F:6 | 1 | 3/4 | 41 | 40:1/2 |
| `thsice_main` | thsice_main.F:12 | 5 | 31/47 | 86-91, 226, 236, 242, 265-276 | 84:1/2, 156:1/2, 223:1/2, 235:1/2, 238:1/2, 243:1/2, 258:1/2, 264:1/2 |
| `thsice_map_exf` | thsice_map_exf.F:9 | 60 | 12/20 | 94-99, 110-120 | 85:1/2, 104:1/2 |
| `thsice_monitor` | thsice_monitor.F:6 | 6 | 114/116 | 78, 263 | 69:1/2, 71:1/2, 75:1/2, 97:1/2, 134:1/2, 157:1/2, 175:1/2, 207:1/2, 231:1/2, 262:1/2, 266:1/2, 270:1/2 |
| `thsice_output` | thsice_output.F:6 | 6 | 21/24 | 67, 96-98 | 59:1/2, 64:1/2, 94:1/2, 139:1/2 |
| `thsice_read_pickup` | thsice_read_pickup.F:6 | 1 | 17/22 | 45-49, 62-63 | 38:1/2, 41:1/2, 42:1/2, 61:1/2 |
| `thsice_readparms` | thsice_readparms.F:6 | 1 | 168/199 | 88-93, 226-233, 236-243, 284-292, 295-303 | 86:1/2, 96:1/2, 156:1/2, 225:1/2, 235:1/2, 248:1/2, 252:1/2, 255:1/2, 281:1/2, 294:1/2, 373:1/2 |
| `thsice_solve4temp` | thsice_solve4temp.F:13 | 60 | 101/106 | 232, 252-254, 388 | 231:1/2, 249:1/2, 265:1/2, 385:1/2 |
| `thsice_step_fwd` | thsice_step_fwd.F:12 | 60 | 76/78 | 178-180 | 160:1/2, 175:1/2, 388:1/2 |
| `thsice_step_temp` | thsice_step_temp.F:9 | 60 | 31/31 | - | - |
| `thsice_turnoff_io` | thsice_turnoff_io.F:6 | 1 | 5/5 | - | 42:1/2 |
| `thsice_write_pickup` | thsice_write_pickup.F:5 | 1 | 16/18 | 65-66 | 64:1/2, 106:1/2 |

**verification/global_ocean.cs32x15/code_ad** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_test` | cost_test.F:3 | 1 | 22/25 | 46-48 | 41:1/2 |

### Compiled but not executed (960 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_cycle_ebl, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_rl_recv_get_x, exch_rl_recv_get_y, exch_rl_send_put_x, exch_rl_send_put_y, exch_rs_recv_get_x, exch_rs_recv_get_y, exch_rs_send_put_x, exch_rs_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rl, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, fill_cs_corner_ag_rl, fill_cs_corner_uv_rl, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, memsync, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth2, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_gw, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, forcing_surf_relax, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cartesian_grid, ini_cg3d, ini_cylinder_grid, ini_local_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_pressure, ini_psurf, ini_salt, ini_sigma_hfac, ini_spherical_polar_grid, ini_theta, ini_vel, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, remove_mean_rl, remove_mean_rs, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, autodiff_findunit, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dummy_in_dynamics, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_addtime, cal_checkdate, cal_compdates, cal_convdate, cal_copydate, cal_daysformonth, cal_dayspermonth, cal_fulldate, cal_getdate, cal_getmonthsrec, cal_init_fixed, cal_intdays, cal_intmonths, cal_intyears, cal_isleap, cal_monthsforyear, cal_monthsperyear, cal_numints, cal_printdate, cal_printerror, cal_set, cal_stepsforday, cal_stepsperday, cal_subdates, cal_summary, cal_time2dump, cal_timeinterval, cal_timepassed, cal_timestamp, cal_toseconds, cal_weekday
- pkg/cost: cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/exch2: exch2_3d_r4, exch2_3d_r8, exch2_ad_get_r41, exch2_ad_get_r42, exch2_ad_get_r81, exch2_ad_get_r82, exch2_ad_get_rl1, exch2_ad_get_rl2, exch2_ad_get_rs1, exch2_ad_get_rs2, exch2_ad_put_r41, exch2_ad_put_r42, exch2_ad_put_r81, exch2_ad_put_r82, exch2_ad_put_rl1, exch2_ad_put_rl2, exch2_ad_put_rs1, exch2_ad_put_rs2, exch2_get_r41, exch2_get_r42, exch2_get_r81, exch2_get_r82, exch2_put_r41, exch2_put_r42, exch2_put_r81, exch2_put_r82, exch2_r41_cube, exch2_r41_cube_ad, exch2_r42_cube, exch2_r42_cube_ad, exch2_r81_cube, exch2_r81_cube_ad, exch2_r82_cube, exch2_r82_cube_ad, exch2_recv_r41, exch2_recv_r42, exch2_recv_r81, exch2_recv_r82, exch2_recv_rl1, exch2_recv_rl2, exch2_recv_rs1, exch2_recv_rs2, exch2_rl1_cube_ad, exch2_rl1_cube_b, exch2_rl1_cube_d, exch2_rl2_cube_ad, exch2_rl2_cube_b, exch2_rl2_cube_d, exch2_rs1_cube_ad, exch2_rs1_cube_b, exch2_rs1_cube_d, exch2_rs2_cube_ad, exch2_rs2_cube_b, exch2_rs2_cube_d, exch2_s3d_r4, exch2_s3d_r8, exch2_s3d_rs, exch2_send_r41, exch2_send_r42, exch2_send_r81, exch2_send_r82, exch2_send_rl1, exch2_send_rl2, exch2_send_rs1, exch2_send_rs2, exch2_sm_3d_r4, exch2_sm_3d_r8, exch2_sm_3d_rl, exch2_sm_3d_rs, exch2_uv_3d_r4, exch2_uv_3d_r8, exch2_uv_agrid_3d_r4, exch2_uv_agrid_3d_r8, exch2_uv_agrid_3d_rl, exch2_uv_bgrid_3d_r4, exch2_uv_bgrid_3d_r8, exch2_uv_bgrid_3d_rl, exch2_uv_cgrid_3d_r4, exch2_uv_cgrid_3d_r8, exch2_uv_cgrid_3d_rl, exch2_uv_cgrid_3d_rs, exch2_uv_dgrid_3d_r4, exch2_uv_dgrid_3d_r8, exch2_uv_dgrid_3d_rl, exch2_uv_dgrid_3d_rs, exch2_z_3d_r4, exch2_z_3d_r8, exch2_z_3d_rl, find_gcd_n, w2_cumulsum_z_tile_rl, w2_print_e2setup, w2_set_gen_facets, w2_set_myown_facets, w2_set_single_facet
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bolus, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, gmredi_slope_psi, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_get_position, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_vecinv: mom_vi_del2uv, mom_vi_u_coriolis_c4, mom_vi_v_coriolis_c4
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_rl
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advdiff, seaice_advection, seaice_bottomdrag_coeffs, seaice_budget_ocean, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stress, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check, seaice_check_pickup, seaice_cost_accumulate_mean, seaice_cost_export, seaice_cost_final, seaice_cost_init_fixed, seaice_cost_sensi, seaice_cost_test, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_dynsolver, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_get_dynforcing, seaice_growth, seaice_growth_adx, seaice_init_fixed, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_lsr_calc_coeffs, seaice_lsr_rhsu, seaice_lsr_rhsv, seaice_lsr_tridiagu, seaice_lsr_tridiagv, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_model, seaice_mom_advection, seaice_monitor, seaice_obcs_output, seaice_ocean_stress, seaice_oceandrag_coeffs, seaice_output, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_reg_ridge, seaice_residual, seaice_scalprod, seaice_sidedrag_stress, seaice_solve4temp, seaice_summary, seaice_tracer_phys, seaice_turnoff_io, seaice_write_pickup
- pkg/thsice: thsice_advdiff, thsice_advection, thsice_balance_frw, thsice_check_conserv, thsice_cost_driver, thsice_cost_test, thsice_diag_sufx, thsice_diagnostics_init, thsice_diagnostics_state, thsice_diffusion, thsice_do_advect, thsice_do_exch, thsice_get_bulkf, thsice_get_precip, thsice_get_velocity, thsice_impl_temp, thsice_mnc_init, thsice_reshape_layers, thsice_salt_plume, thsice_slab_ocean

