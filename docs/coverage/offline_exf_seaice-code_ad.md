# Coverage: offline_exf_seaice (code_ad) — M4 porting worklist

Written by `tools/coverage.py` (mitjax 7c620b3) from the gcov build `offline_exf_seaice-code_ad-63cdc0b-dad681f-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 858 files from the link farm, 105 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_ad` | `job27856010` | 3 | 0 / 0 | 409 | yes | identical (617 lines) |
| `input_ad.thsice` | `job27856010` | 1 | 0 / 0 | 346 | yes | identical (470 lines) |

## offline_exf_seaice/input_ad

Run `$MJX_REFERENCE/coverage/offline_exf_seaice/input_ad/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/offline_exf_seaice/input_ad/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice volume changes | 11 blocks, first 1.810244e-01, last 1.734334e-01, min 1.734334e-01, max 1.811847e-01 | yes |
| sea-ice thermodynamics (GROWTH_ADX) | seaice_growth_adx (120 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (120 calls) | yes |
| cost function | global fc = 1.46159765738472e-05; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: AdamsBashforthGt, diags_opOceWeighted, doAB_onGtGs, doThetaClimRelax, dumpInitAndLast, fluidIsWater, inAdExact, momDissip_In_AB, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, SEAICE_doOpenWaterGrowth, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICE_useMultDimSnow, SEAICEadvArea, SEAICEadvHeff, SEAICEadvSnow, SEAICEmultiDimAdvection, SEAICEupdateOceanStress, SEAICEuseFlooding, snapshot_mdsio, tempForcing, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, usePW79thermodynamics, useSEAICEinAdMode, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, applyExchUV_early, bottomVisc_pCell, calc_wVelocity, cg2dFullAdjoint, debugMode, deepAtmosphere, diag_mnc, doResetHFactors, doSaltClimRelax, exactConserv, fluidIsAir, globalFiles, implicitDiffusion, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momAdvection, momForcing, momImplVertAdv, momPressureForcing, momViscosity, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, rotateStressOnAgrid, saltAdvection, saltForcing, saltImplVertAdv, saltIsActiveTr, saltMultiDimAdvec, saltSOM_Advection, SEAICE_doOpenWaterMelt, SEAICE_growMeltByConv, SEAICE_mcPheeStepFunc, SEAICE_salinityTracer, SEAICEheatConsFix, SEAICEmomAdvection, SEAICEuseBDF2, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFREEDRIFTswitchInAd, stressIsOnCgrid, tempImplVertAdv, tempIsActiveTr, tempMultiDimAdvec, tempSOM_Advection, twoDigitYear, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCDscheme, useCoriolis, useCoupler, useCtrlCostContribution, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useGMRediInAdMode, useHarmonicVisc, useKPPinAdMode, useMaykutSatVapPoly, useMin4hFacEdges, useMissingValue, useMultiDimAdvec, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useObcsCostContribution, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid, vectorInvariantMomentum
- selectors: exf_adjMonSelect=1, pCellMix_select=0, saltAdvScheme=2, saltVertAdvScheme=2, SEAICEadvScheme=77, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=0, selectSigmaCoord=0, tempAdvScheme=2, tempVertAdvScheme=2

### Executed routines (409)

**eesupp/src** (75)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 3054 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 18136 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `diff_phase_multiple` | diff_phase_multiple.F:7 | 480 | 15/16 | 46 | 44:1/2, 45:1/2, 50:1/2 |
| `different_multiple` | different_multiple.F:7 | 1084 | 14/15 | 45 | 44:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1934 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 2544 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 120 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 4478 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1934 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1934 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1934 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1934 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 2544 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 2544 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 2544 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 2544 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 120 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 240 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 2 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 365 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 846 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 1812 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 242 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 57462 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 424 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 23 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 310 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 554 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 13130 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 22023 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 20 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 863 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 897 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 566 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 17912 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 196 | 7/7 | - | - |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 10 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 52 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 131 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 231 | 43/49 | 648-650, 668-671 | 647:1/2, 662:1/2, 664:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 2963 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 5536 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 5536 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 5536 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 2769 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 2767 | 4/4 | - | - |
| `ucase` | utils.F:311 | 11074 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 8 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 42 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 131 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 182 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 45 | 32/32 | - | - |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (85)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 480 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 480 | 13/26 | 467, 469, 471, 626-632, 639-643, 740 | 466:1/2, 468:1/2, 470:1/2, 618:1/2, 638:1/2, 736:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 480 | 16/94 | 132-166, 269-277, 287-406 | 83:1/2, 119:1/2, 266:1/2, 284:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 480 | 19/26 | 87-94, 124-128 | 80:1/2, 113:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 480 | 34/162 | 149, 184, 212-240, 256, 268-610, 629-648 | 100:1/2, 121:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 252:1/2, 253:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `config_check` | config_check.F:14 | 1 | 87/489 | 92-97, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 464-470, 484-490, 497-503, 510-516, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 463:1/2, 482:1/2, 495:1/2, 508:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 435/537 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 268-281, 307-322, 533-538, 554-588, 595, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 480 | 6/6 | - | - |
| `diags_oceanic_surf_flux` | diags_oceanic_surf_flux.F:7 | 120 | 26/39 | 56, 121-152, 167-190 | 55:1/2, 81:1/2, 94:1/2, 115:1/2, 161:1/2 |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 480 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 480 | 24/32 | 79-84, 123-125 | 61:1/2, 68:1/2, 76:1/2, 94:1/2, 95:1/2, 117:1/2, 121:1/2 |
| `diags_rho_g` | diags_rho.F:100 | 120 | 8/33 | 161-171, 178-188, 195-213 | 160:1/2, 177:1/2, 194:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 120 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 120 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 120 | 74/116 | 258, 260, 262, 264, 320-323, 397-403, 467-471, 557, 593-596, 771-784, 797-798, 811-845, 869-874, 882, 899, 934, 1123 | 248:1/2, 251:1/2, 256:1/2, 257:1/2, 259:1/2, 261:1/2, 263:1/2, 291:1/2, 377:1/2, 421:1/2, 450:1/2, 553:1/2, 577:1/2, 589:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 878:1/2, 896:1/2, 932:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 120 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_statevars_diags` | do_statevars_diags.F:8 | 360 | 14/16 | 64, 102 | 54:1/2, 60:1/2, 101:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 121 | 11/19 | 97-110, 133, 207 | 96:1/2, 116:1/2, 132:1/2, 193:1/2, 206:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 120 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 120 | 60/88 | 358, 389, 413, 492-562, 591-599, 609-610, 665, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 409:1/2, 490:1/2, 582:1/2, 604:1/2, 664:1/2, 682:1/2, 693:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 120 | 3/109 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 120 | 39/65 | 75, 151-153, 268-285, 301-306, 325-343, 368-372 | 74:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 299:1/2, 310:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 480 | 14/60 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 1 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 120 | 16/21 | 64, 77-88 | 63:1/2, 75:1/2, 242:1/2 |
| `forward_step` | forward_step.F:70 | 120 | 82/99 | 730-734, 767-769, 903-905, 927-929, 980, 1176, 1201-1202 | 424:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 78/89 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/65 | 57-63, 84-112, 121-180, 191 | 55:1/2, 68:1/2, 72:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/70 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274, 286-288 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2, 284:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 47/47 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 31/49 | 54, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 193 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 192:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 177/193 | 210-212, 247-261, 435, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 29/53 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 398/880 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 623-625, 649-651, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 653:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1446:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 23/38 | 76-80, 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 24/47 | 77-81, 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 22/22 | - | 55:1/2, 57:1/2, 60:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 124:1/2, 137:1/2, 140:1/2, 145:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 1 | 17/75 | 92-196, 211-221, 279-282, 291-293, 329-331 | 90:1/2, 207:1/2, 277:1/2, 290:1/2, 298:1/2, 321:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 4 | 14/28 | 95-117, 187-193 | 93:1/2, 177:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 120 | 27/34 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 120 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 120 | 7/19 | 65-90, 95, 128 | 63:1/2, 94:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 99/99 | - | 100:1/2, 232:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/111 | 132-140, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 193:1/2, 200:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 254:1/2, 260:1/2, 264:1/2, 269:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 380:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 31/37 | 202-204, 221-223, 530-532 | 144:1/2, 167:1/2, 170:1/2, 174:1/2, 193:1/2, 200:1/2, 219:1/2, 249:1/2, 251:1/2, 358:1/2, 365:1/2, 367:1/2, 510:1/2, 512:1/2, 528:1/2, 651:1/2, 655:1/2, 673:1/2, 675:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 18/24 | 169, 452-454, 629-631, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 291:1/2, 293:1/2, 435:1/2, 450:1/2, 617:1/2, 627:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 18 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 12/12 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 2 | 27/37 | 68-70, 76, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 11/13 | 112, 193 | 110:1/2, 184:1/2, 191:1/2, 223:1/2, 255:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 120 | 28/64 | 67-70, 78, 85-127, 168-174 | 66:1/2, 77:1/2, 83:1/2, 146:1/2, 160:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 100/166 | 60-63, 67, 71, 79, 109-115, 120-126, 171-175, 186-189, 192-195, 200, 206, 212, 218, 224, 230, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 49/195 | 101-133, 140-165, 180-187, 193-201, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 192:1/2, 202:2/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `temp_integrate` | temp_integrate.F:13 | 480 | 57/68 | 194, 275-282, 328, 369-371, 392, 494, 503, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 391:1/2, 398:1/2, 481:1/2, 495:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 43/43 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 120 | 41/52 | 158, 330-336, 359, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 317:1/2, 319:1/2, 328:1/2, 358:1/2, 394:1/2, 413:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 480 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 120 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 21/22 | 113 | 55:1/2, 78:1/2, 83:1/2, 106:1/2, 112:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 50/75 | 245-250, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 252:1/2, 255:1/2, 256:1/2, 257:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 121 | 18/20 | 87, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (25)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 10 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 9 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 1 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 5 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 4 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `active_write_xz` | active_file.F:482 | 8 | 7/7 | - | - |
| `active_write_xz_rl` | active_file_control_slice.F:701 | 8 | 10/19 | 781-799, 809 | 775:1/2, 804:1/2, 813:1/2 |
| `active_write_yz` | active_file.F:543 | 8 | 7/7 | - | - |
| `active_write_yz_rl` | active_file_control_slice.F:937 | 8 | 10/19 | 1017-1035, 1045 | 1011:1/2, 1040:1/2, 1049:1/2 |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 2 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 120 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 120 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 81/148 | 63-68, 127-132, 157, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 26 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 26 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 52 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 1 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 120 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 120 | 2/2 | - | - |

**pkg/cost** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 18/18 | - | - |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 8/12 | 35-39 | 33:1/2, 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 48/48 | - | 82:1/2, 102:1/2, 105:1/2, 110:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 22/22 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 120 | 4/4 | - | 121:1/2, 136:1/2 |

**pkg/ctrl** (34)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 1 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 2 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 1 | 3/13 | 43-54 | 41:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 34/72 | 104-110, 124-128, 139-143, 182-186, 226-230, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 103:1/2, 121:1/2, 136:1/2, 179:1/2, 225:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/37 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/83 | 70-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 243 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 240 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 240 | 12/29 | 190-197, 206-223 | 79:1/2, 177:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 242 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 1 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 11 | 31/50 | 80-87, 114-126, 171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 149:1/2, 158:1/2, 161:1/2, 167:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 77/86 | 233-238, 283, 285, 301, 303, 312-314 | 231:1/2, 251:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 8 | 15/26 | 72-76, 87-88, 111-112, 118-124 | 86:1/2, 89:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 24/24 | - | 81:1/2, 97:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 203/209 | 182, 184, 216-219 | 168:1/2, 181:1/2, 183:1/2, 192:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 120 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 1 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 120 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 30/49 | 155-166, 201, 219, 221, 358, 360, 362, 364, 395 | 128:1/2, 154:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 355:1/2, 357:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 65/80 | 151-153, 157-159, 162, 183-190, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_mask_set_xz` | ctrl_mask_set_xz.F:3 | 2 | 29/37 | 88-98 | 64:1/2, 86:1/2 |
| `ctrl_mask_set_yz` | ctrl_mask_set_yz.F:3 | 2 | 29/37 | 89-99 | 64:1/2, 87:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 208/247 | 181-186, 531, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 530:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 11 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 6 | 16/16 | - | 71:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 1 | 10/10 | - | - |
| `ctrl_set_globfld_xz` | ctrl_set_globfld_xz.F:3 | 2 | 17/18 | 69 | 66:1/2, 74:1/2 |
| `ctrl_set_globfld_yz` | ctrl_set_globfld_yz.F:3 | 2 | 17/18 | 69 | 66:1/2, 74:1/2 |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 91/145 | 153-155, 161-171, 184-187, 218-222, 229-241, 267-292, 304-306, 318-321, 324-327 | 146:1/2, 160:1/2, 178:1/2, 217:1/2, 228:1/2, 266:1/2, 302:1/2, 317:1/2, 323:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 2 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (47)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_addtolist` | diagnostics_addtolist.F:8 | 243 | 17/44 | 62, 68-92, 111-117 | 60:1/2, 67:1/2, 99:1/2, 103:1/2, 108:1/2, 125:1/2 |
| `diagnostics_check` | diagnostics_check.F:8 | 1 | 22/150 | 44-47, 51-54, 59-64, 85-89, 92-96, 104-113, 120-130, 145-213, 225-250, 261-268, 277-280 | 37:1/2, 43:1/2, 50:1/2, 57:1/2, 84:1/2, 91:1/2, 103:1/2, 118:1/2, 119:2/2, 144:1/2, 224:1/2, 260:1/2, 275:1/2 |
| `diagnostics_clear` | diagnostics_clear.F:6 | 10 | 8/8 | - | 32:1/2 |
| `diagnostics_clrdiag` | diagnostics_clear.F:51 | 80 | 10/10 | - | - |
| `diagnostics_cumulate` | diagnostics_fill_field.F:413 | 4000 | 21/62 | 478-517, 531-556, 561-578, 602 | 475:1/2, 477:1/2, 523:1/2, 559:1/2, 594:1/2 |
| `diagnostics_fill` | diagnostics_fill.F:6 | 9360 | 31/34 | 75, 130-131 | 73:1/2, 119:1/2, 129:1/2 |
| `diagnostics_fill_field` | diagnostics_fill_field.F:13 | 1000 | 42/101 | 114-116, 128-141, 146-151, 159-160, 170-177, 182-183, 188-189, 203-216, 235-268 | 105:3/6, 107:1/2, 124:1/2, 145:1/2, 158:1/2, 167:1/2, 181:1/2, 184:1/2, 192:1/2, 202:1/2, 220:1/2, 226:1/2 |
| `diagnostics_fill_rs` | diagnostics_fill_rs.F:6 | 840 | 29/34 | 75, 84-85, 130-131 | 73:1/2, 80:1/2, 119:1/2, 129:1/2 |
| `diagnostics_fill_state` | diagnostics_fill_state.F:6 | 360 | 60/287 | 70-79, 94-95, 112-139, 145-166, 170-183, 187-203, 207-223, 229-241, 245-257, 261-274, 278-290, 294-306, 310-323, 327-339, 343-355, 359-380, 398-410, 423-435, 452, 504, 537-550, 556-568, 572-584, 590-603, 607-620, 624-637, 641-654, 658-671, 675-688, 714-724, 739-749 | 69:1/2, 93:1/2, 111:1/2, 143:1/2, 169:1/2, 186:1/2, 206:1/2, 228:1/2, 244:1/2, 260:1/2, 277:1/2, 293:1/2, 309:1/2, 326:1/2, 342:1/2, 358:1/2, 393:1/2, 418:1/2, 451:1/2, 501:1/2, 503:1/2, 534:1/2, 555:1/2, 571:1/2, 589:1/2, 606:1/2, 623:1/2, 640:1/2, 657:1/2, 674:1/2, 709:1/2, 734:1/2 |
| `diagnostics_fract_fill` | diagnostics_fract_fill.F:6 | 120 | 37/56 | 90, 98-99, 116-122, 155-161, 174-175 | 88:1/2, 94:1/2, 114:1/2, 115:1/2, 147:1/2, 153:1/2, 154:1/2, 173:1/2 |
| `diagnostics_get_diag` | diagnostics_utils.F:97 | 320 | 24/26 | 141-142 | 137:1/2, 139:1/2, 147:1/2, 154:1/2 |
| `diagnostics_ini_io` | diagnostics_ini_io.F:6 | 1 | 4/12 | 45-54 | 39:1/2, 42:1/2 |
| `diagnostics_init_early` | diagnostics_init_early.F:8 | 1 | 107/107 | - | 71:1/2 |
| `diagnostics_init_fixed` | diagnostics_init_fixed.F:8 | 1 | 8/8 | - | - |
| `diagnostics_init_varia` | diagnostics_init_varia.F:8 | 1 | 24/24 | - | 30:1/2 |
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 7320 | 14/16 | 56-57 | 55:1/2 |
| `diagnostics_main_init` | diagnostics_main_init.F:8 | 1 | 583/619 | 95-97, 104-110, 112-117, 128-129, 131-132, 285, 780-801 | 94:1/2, 103:1/2, 111:1/2, 127:1/2, 130:1/2, 201:1/2, 248:1/2, 284:1/2, 621:1/2, 633:1/2, 779:1/2 |
| `diagnostics_out` | diagnostics_out.F:8 | 10 | 70/158 | 95, 100-102, 130, 177-184, 188, 203-208, 211-244, 256-289, 308-336, 347-357, 365, 370-382, 410 | 92:1/2, 99:1/2, 107:1/2, 129:1/2, 141:1/2, 176:1/2, 186:1/2, 190:1/2, 196:1/2, 197:1/2, 199:1/2, 209:1/2, 255:1/2, 294:1/2, 307:1/2, 345:1/2, 360:1/2, 367:1/2, 392:1/2, 398:1/2, 399:1/2, 401:1/2, 438:1/2 |
| `diagnostics_read_pickup` | diagnostics_read_pickup.F:7 | 1 | 2/2 | - | - |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 247/372 | 111-119, 280-282, 295, 298-300, 302-309, 311-312, 327-342, 357-366, 372-381, 421-424, 429-435, 446-447, 454-467, 471-478, 493-502, 508-517, 538-540, 582-583, 587-588, 592-604 | 109:1/2, 123:1/2, 168:1/2, 279:1/2, 294:1/2, 297:1/2, 301:1/2, 310:1/2, 321:1/2, 326:1/2, 356:1/2, 371:1/2, 420:1/2, 428:1/2, 444:1/2, 453:1/2, 469:1/2, 481:1/2, 492:1/2, 507:1/2, 536:1/2, 570:1/2, 586:1/2, 589:1/2, 607:2/4, 609:1/4, 629:1/2, 631:1/2, 635:2/4, 638:1/4 |
| `diagnostics_scale_fill` | diagnostics_scale_fill.F:6 | 3600 | 15/33 | 78, 95-105, 120-146 | 76:1/2, 94:1/2, 119:1/2 |
| `diagnostics_scale_fill_rs` | diagnostics_scale_fill_rs.F:6 | 720 | 28/33 | 78, 86-87, 132-133 | 76:1/2, 82:1/2, 121:1/2, 131:1/2 |
| `diagnostics_set_calc` | diagnostics_set_calc.F:9 | 1 | 7/89 | 57-194 | 46:1/2 |
| `diagnostics_set_levels` | diagnostics_set_levels.F:7 | 1 | 75/133 | 88, 95-115, 119-122, 131-134, 138-141, 148-151, 156-160, 164-167, 218-220, 226-228, 244-253 | 73:1/2, 87:1/2, 93:1/2, 118:1/2, 130:1/2, 137:1/2, 147:1/2, 155:1/2, 163:1/2, 183:1/2, 217:1/2, 221:1/2, 241:1/2, 243:1/2 |
| `diagnostics_set_pointers` | diagnostics_set_pointers.F:6 | 1 | 72/160 | 59-63, 70-96, 101-109, 145-151, 166-186, 200-207, 213-217, 248-251, 254-260, 278-302 | 41:1/2, 57:1/2, 58:1/2, 69:1/2, 100:1/2, 142:1/2, 158:1/2, 199:1/2, 212:1/2, 239:1/2, 246:1/2, 252:1/2, 271:1/2, 272:2/4, 274:1/4 |
| `diagnostics_setdiag` | diagnostics_setdiag.F:6 | 16 | 49/89 | 66-72, 92-94, 109-111, 118-126, 133-134, 142-145, 156, 186-205 | 65:1/2, 90:1/2, 101:1/2, 107:1/2, 114:1/2, 115:1/2, 131:1/2, 155:1/2, 174:1/2, 185:1/2 |
| `diagnostics_summary` | diagnostics_summary.F:7 | 1 | 7/124 | 61-83, 94-243 | 55:1/2, 60:1/2, 90:1/2 |
| `diagnostics_switch_onoff` | diagnostics_switch_onoff.F:11 | 120 | 28/74 | 81, 146-173, 188-235 | 77:1/2, 87:1/2, 127:1/2, 135:1/2, 144:1/2, 185:1/2 |
| `diagnostics_write` | diagnostics_write.F:3 | 121 | 45/50 | 93-94, 117-118, 152 | 92:1/2, 110:1/2, 151:1/2, 161:1/2 |
| `diagnostics_write_pickup` | diagnostics_write_pickup.F:7 | 1 | 3/3 | - | - |
| `diags_mk_title` | diagnostics_utils.F:593 | 36 | 17/23 | 643-650 | 632:1/2, 635:1/2, 642:1/2 |
| `diags_mk_units` | diagnostics_utils.F:514 | 41 | 13/30 | 554-568, 575-582 | 547:1/2, 552:1/2, 574:1/2 |
| `diags_renamed` | diagnostics_utils.F:661 | 300 | 14/19 | 708-714 | 694:1/2, 695:1/2, 696:1/2, 697:1/2, 698:1/2, 699:1/2, 700:1/2, 702:1/2, 703:1/2, 705:1/2 |
| `diags_track_diva` | diagnostics_utils.F:410 | 1 | 5/9 | 448-452 | 444:1/2 |
| `diagstats_ascii_out` | diagstats_ascii_out.F:8 | 88 | 19/19 | - | 43:1/2, 45:1/2, 51:1/2, 55:1/2, 75:1/4 |
| `diagstats_calc` | diagstats_calc.F:6 | 3840 | 47/67 | 99-102, 116-121, 153-155, 161-163, 167-170, 174-177 | 98:1/2, 115:1/2, 152:1/2, 160:1/2, 166:1/2, 173:1/2 |
| `diagstats_clear` | diagstats_clear.F:8 | 11 | 7/7 | - | 30:1/2 |
| `diagstats_close_io` | diagstats_close_io.F:6 | 1 | 16/18 | 55, 71 | 40:1/2, 42:1/2, 52:1/2, 70:1/2 |
| `diagstats_clrdiag` | diagstats_clear.F:46 | 88 | 8/8 | - | - |
| `diagstats_fill` | diagstats_fill.F:13 | 960 | 35/78 | 121-122, 136-141, 149-150, 160-167, 173-175, 181-183, 196-209, 234-246 | 120:1/2, 135:1/2, 148:1/2, 157:1/2, 172:1/2, 176:1/2, 195:1/2, 216:1/2 |
| `diagstats_global` | diagstats_global.F:8 | 88 | 73/87 | 115-116, 142-146, 158-159, 171-172, 191-192, 195-196 | 63:1/2, 72:1/2, 89:1/2, 113:1/2, 136:1/2, 137:1/2, 152:1/2, 153:1/2, 170:1/2, 190:1/2, 194:1/2 |
| `diagstats_ini_io` | diagstats_ini_io.F:6 | 1 | 36/37 | 54 | 40:1/2, 42:1/2, 51:1/2, 76:1/2, 77:2/4, 83:2/4, 85:1/4, 87:2/4, 89:1/4 |
| `diagstats_local` | diagstats_local.F:6 | 3840 | 35/46 | 112-114, 134, 146, 172, 185, 201, 212, 242-243 | 106:1/2, 111:1/2, 120:1/2, 121:1/2, 122:1/2, 124:1/2, 136:1/2, 160:1/2, 173:1/2, 191:1/2, 202:1/2, 229:1/2, 237:1/2 |
| `diagstats_output` | diagstats_output.F:8 | 11 | 20/45 | 88-111, 116-125 | 66:1/2, 78:1/2, 87:1/2, 115:1/2, 133:1/2 |
| `diagstats_set_pointers` | diagstats_set_pointers.F:6 | 1 | 41/84 | 72-78, 84-86, 89-91, 98-105, 120-140, 150-156, 162-165 | 40:1/2, 68:1/2, 83:1/2, 88:2/2, 95:1/2, 97:1/2, 112:1/2, 148:1/2, 149:2/2, 161:1/2 |
| `diagstats_set_regions` | diagstats_set_regions.F:6 | 1 | 17/21 | 209-212 | 200:1/2, 203:1/2, 208:1/2 |
| `diagstats_setdiag` | diagstats_setdiag.F:6 | 8 | 34/51 | 70-72, 84-85, 95-97, 124-135 | 65:1/2, 68:1/2, 81:1/2, 82:1/2, 110:1/2, 113:1/2, 123:1/2 |

**pkg/exf** (28)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 360 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 120 | 90/101 | 236, 241-242, 325-333, 403, 502-506 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 31/149 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 348-357, 399-402, 550-569, 575-578, 617-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 616:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 27/72 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 120 | 16/25 | 42-47, 57-60 | 39:1/2, 41:1/2, 56:1/2 |
| `exf_diagnostics_init` | exf_diagnostics_init.F:6 | 1 | 193/197 | 102, 113, 232, 244 | 101:1/2, 112:1/2, 231:1/2, 243:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 21 | 12/22 | 66-78 | 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 7 | 17/26 | 890-893, 901-909 | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 120 | 14/14 | - | 68:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 7 | 9/43 | 65-76, 111-128, 132-141 | 80:1/2, 106:1/2, 109:1/2, 131:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 840 | 11/38 | 220-259 | 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 120 | 51/72 | 97, 147-152, 315-320, 498, 501, 504, 507, 512, 520, 525, 535, 545 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 496:1/2, 499:1/2, 502:1/2, 505:1/2, 510:1/2, 518:1/2, 523:1/2, 533:1/2, 543:1/2, 546:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 120 | 40/44 | 202-203, 329, 388 | 164:1/2, 171:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 120 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 14 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 89/143 | 64-65, 137-143, 149-155, 159-179, 185-191, 195-201, 207-213, 262-268, 282-289, 309-315, 359-366, 401-431, 436-466, 475, 488-495, 590-593 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 399:1/2, 434:1/2, 472:1/2, 474:1/2, 486:1/2, 588:1/2, 610:1/2, 615:1/2, 617:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 17 | 20/26 | 101-107 | 100:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 39/47 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 120 | 60/86 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 120 | 66/73 | 74, 114-116, 164, 186, 192, 228 | 67:1/2, 71:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 120 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 442/463 | 285-290, 1000-1003, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 998:1/2, 1005:1/2, 1006:1/2, 1007:1/2, 1008:1/2, 1009:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1016:1/2, 1017:1/2, 1018:1/2, 1019:1/2, 1020:1/2, 1022:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 2040 | 27/63 | 124-129, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 120 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 166/202 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 385-395, 402-408, 487-493, 500-501, 548-555, 571-581, 585-586, 625-626, 684-690, 769-771, 795-801 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 7 | 8/8 | - | - |
| `exf_weight_sfx_diags` | exf_weight_sfx_diags.F:6 | 240 | 18/89 | 59-104, 112-115, 120-130, 134-144, 149-159, 171-174, 179-189, 193-203, 207-217, 221-231 | 51:1/2, 57:1/2, 111:1/2, 119:1/2, 133:1/2, 148:1/2, 169:1/2, 178:1/2, 192:1/2, 206:1/2, 220:1/2 |
| `exf_wind` | exf_wind.F:9 | 120 | 37/83 | 104-115, 138-140, 158-240 | 79:1/2, 102:1/2, 126:1/2, 133:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advscheme_get` | gad_advscheme.F:116 | 4 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 480 | 65/187 | 181-184, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 606-608, 620, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diag_sufx` | gad_diagnostics_init.F:324 | 482 | 6/7 | 368 | 356:1/2 |
| `gad_diagnostics_init` | gad_diagnostics_init.F:6 | 1 | 81/87 | 53, 60, 181-185 | 52:1/2, 59:1/2, 180:1/2 |
| `gad_diagnostics_state` | gad_diagnostics_state.F:8 | 120 | 2/2 | - | - |
| `gad_diff_r` | gad_diff_r.F:7 | 480 | 7/10 | 59-64 | 52:1/2 |
| `gad_fluxlimit_adv_x` | gad_fluxlimit_adv_x.F:7 | 1440 | 21/22 | 85 | 78:1/2, 84:1/2 |
| `gad_fluxlimit_adv_y` | gad_fluxlimit_adv_y.F:7 | 1440 | 21/22 | 85 | 78:1/2, 84:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 97/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 159-162, 171-172, 175-176, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 35/58 | 81-135 | 52:1/2, 73:1/2 |
| `grdchk_get_obcs_mask` | grdchk_get_obcs_mask.F:6 | 1 | 13/49 | 70-135 | 67:1/2, 69:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 13/25 | 87-89, 98-100, 107-109, 116-123 | 84:1/2, 95:1/2, 104:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/139 | 116-126, 132, 163, 192-251, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/mdsio** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 25 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 146 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 26 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 40 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 489, 529, 535-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 487:1/4, 493:1/2, 527:1/2, 530:1/2, 540:1/2, 544:1/2 |
| `mds_seg4torl_2d` | mdsio_segxtorx_2d.F:12 | 128 | 5/7 | 39-40 | 38:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 12 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 161 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 545:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 323 | 41/52 | 108, 135-139, 146-147, 157-159, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 222:1/4, 228:1/2 |
| `mds_write_sec_xz` | mdsio_write_section.F:13 | 16 | 43/81 | 109-115, 127, 139-149, 190-193, 203, 209-211, 217-246, 259-260 | 107:1/2, 124:1/2, 125:1/2, 138:1/2, 157:1/2, 174:1/2, 183:1/2, 200:1/2, 201:1/2, 204:1/2, 249:1/2, 258:1/2, 267:1/2 |
| `mds_write_sec_yz` | mdsio_write_section.F:274 | 16 | 43/81 | 369-375, 387, 399-409, 450-453, 463, 469-471, 477-506, 519-520 | 367:1/2, 384:1/2, 385:1/2, 398:1/2, 417:1/2, 434:1/2, 443:1/2, 460:1/2, 461:1/2, 464:1/2, 509:1/2, 518:1/2, 527:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 250:1/2 |

**pkg/monitor** (20)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 6 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 3 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 3 | 10/13 | 41-45 | 39:1/2, 40:2/2 |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 2 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 8 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 89 | 71/71 | - | 88:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 3 | 52/125 | 167-320 | 70:1/2, 74:1/2, 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 616 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 15 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 601 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 38 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 3 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 3 | 39/50 | 95, 123-132, 178-179, 196-198 | 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 3 | 72/127 | 145-237, 243-278 | 143:1/2, 242:1/2, 286:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 89 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 121 | 58/67 | 57, 119-120, 135-145 | 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/obcs** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `obcs_readparms` | obcs_readparms.F:6 | 1 | 5/382 | 222-850 | 212:1/2, 214:1/2 |

**pkg/rw** (18)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xy_rl` | read_fld_xy_rl.F:3 | 3 | 12/15 | 35-37 | 32:1/2 |
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 24 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 15 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 22 | 16/17 | 38 | 37:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 21 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 81 | 7/7 | - | - |
| `write_rec_xz_rl` | write_rec.F:661 | 8 | 7/7 | - | - |
| `write_rec_yz_rl` | write_rec.F:793 | 8 | 7/7 | - | - |

**pkg/seaice** (28)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_advdiff` | seaice_advdiff.F:10 | 120 | 40/61 | 311, 338, 365, 579-631 | 135:1/2, 297:1/2, 304:1/2, 324:1/2, 331:1/2, 351:1/2, 358:1/2 |
| `seaice_advection` | seaice_advection.F:14 | 1440 | 97/226 | 169-172, 182-183, 217-230, 247-250, 290-292, 307-318, 326, 359, 374-379, 385-416, 426, 434-496, 573, 588-593, 599-630, 640, 648-710, 766-771, 790 | 167:1/2, 177:1/2, 190:1/2, 216:1/2, 246:1/2, 289:1/2, 305:1/2, 325:1/2, 345:1/2, 356:1/2, 371:1/2, 381:1/2, 424:1/2, 433:1/2, 505:1/2, 506:1/2, 559:1/2, 570:1/2, 585:1/2, 595:1/2, 638:1/2, 647:1/2, 719:1/2, 720:1/2, 765:1/2, 775:1/2, 788:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 480 | 6/6 | - | - |
| `seaice_check` | seaice_check.F:12 | 1 | 76/477 | 12, 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 243-249, 253-259, 371-404, 411-457, 462-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 567-569, 700-702, 725-730, 733-741, 744-752, 777-780, 789-794, 818-925, 935-940, 947-952, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1078-1083, 1119-1124, 1130-1135, 1140-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1205-1209, 1216-1221, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 459:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 776:1/2, 788:1/2, 798:1/2, 933:1/2, 945:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1076:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1204:1/2, 1214:1/2, 1300:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 120 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 15/15 | - | 83:1/2 |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 2/2 | - | - |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 120 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 120 | 12/52 | 75, 101-221 | 74:1/2, 79:1/2, 88:1/2 |
| `seaice_diag_sufx` | seaice_diagnostics_init.F:935 | 1444 | 10/11 | 969 | 966:1/2 |
| `seaice_diagnostics_init` | seaice_diagnostics_init.F:12 | 1 | 461/461 | - | - |
| `seaice_diagnostics_state` | seaice_diagnostics_state.F:6 | 120 | 23/35 | 117-133, 136-152 | 53:1/2, 59:1/2, 116:1/2, 135:1/2 |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 120 | 21/222 | 146-346, 415-733 | 136:1/2, 376:1/2, 414:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 120 | 31/57 | 159-164, 173, 221-232, 248-258, 263-273 | 98:1/2, 146:1/2, 157:1/2, 172:1/2, 244:1/2, 247:1/2, 262:1/2 |
| `seaice_growth_adx` | seaice_growth_adx.F:17 | 120 | 257/274 | 295-296, 305, 454-455, 532, 798-802, 1012, 1273-1283, 1319-1323 | 294:1/2, 304:1/2, 453:1/2, 466:1/2, 528:1/2, 667:1/2, 784:1/2, 994:1/2, 1022:1/2, 1241:1/2, 1263:1/2, 1272:1/2, 1293:1/2, 1318:1/2, 1478:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 53/105 | 60, 67, 80, 84, 279-353, 426-446, 459-461 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2, 82:1/2, 228:1/2, 278:1/2, 362:1/2, 425:1/2, 457:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 128/153 | 54, 237-242, 254, 272, 274, 277-288, 448-451, 463-471 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 271:1/2, 273:1/2, 275:1/2, 292:1/2, 317:1/2, 345:1/2, 447:1/2, 462:1/2 |
| `seaice_model` | seaice_model.F:13 | 120 | 52/77 | 92, 98, 105-115, 211-213, 281-286, 341, 374-390 | 82:1/2, 85:1/2, 95:1/2, 102:1/2, 104:1/2, 125:1/2, 128:1/2, 131:1/2, 178:1/2, 193:1/2, 209:1/2, 222:1/2, 224:1/2, 252:1/2, 255:1/2, 267:1/2, 270:1/2, 312:1/2, 340:1/2, 366:1/2, 373:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 121 | 37/38 | 65 | 58:1/2, 62:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 120 | 18/29 | 56, 69-92 | 55:1/2, 64:1/2 |
| `seaice_output` | seaice_output.F:10 | 121 | 24/28 | 112, 155-159 | 57:1/2, 108:1/2, 109:1/2, 127:1/2, 153:1/2, 174:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 439/834 | 233-238, 460-468, 752-757, 773-827, 837-840, 846-856, 866, 868, 889-896, 912-913, 921-932, 936-942, 954-960, 965-971, 975-986, 992, 997, 1006-1012, 1017-1025, 1032-1036, 1044, 1070, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1502-1507, 1517-1549 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 720:1/2, 721:2/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 763:1/2, 767:1/2, 769:1/2, 771:1/2, 834:1/2, 835:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 951:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1501:1/2, 1511:1/2, 1514:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 120 | 44/44 | - | 393:1/2 |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 480 | 128/148 | 191, 262-265, 308, 310, 322, 387-388, 449-451, 514, 552-561 | 190:1/2, 217:1/2, 260:1/2, 307:1/2, 309:1/2, 321:1/2, 385:1/2, 445:1/2, 502:1/2, 513:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 189/263 | 116-311, 357-359, 377, 400, 480-482 | 45:1/2, 108:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 1 | 41/71 | 102-106, 160-168, 172-188, 194-200 | 78:1/2, 101:1/2, 110:1/2, 117:1/2, 121:1/2, 125:1/2, 152:1/2, 157:1/2, 159:1/2, 171:1/2, 193:1/2 |

**pkg/thsice** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `thsice_cost_init_varia` | thsice_cost_init_varia.F:3 | 1 | 6/6 | - | - |
| `thsice_readparms` | thsice_readparms.F:6 | 1 | 5/199 | 96-383 | 86:1/2, 88:1/2 |

### Compiled but not executed (879 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, apply_forcing_s, apply_forcing_u, apply_forcing_v, calc_div_ghat, calc_eddy_stress, calc_grad_phi_fv, calc_grad_phi_hyd, calc_grad_phi_surf, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_r_star, calc_surf_dr, calc_viscosity, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, correction_step, cycle_ab_tracer, diags_rho_l, diags_sound_speed, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, freesurf_rescale_g, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, salt_integrate, solve_for_pressure, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, swfrac, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz_loc, active_write_xz_rs, active_write_yz_loc, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_pack, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_calc_phivel, diagnostics_count, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_write_adj, diags_get_parms_i, diagstats_g_calc, diagstats_lm_calc, diagstats_mnc_out
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_advection, gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/grdchk: grdchk_get_position, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_hfacz, mom_calc_ke, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_init_fixed, mom_quasihydrostatic, mom_u_botdrag_coeff, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_botdrag_coeff, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_calc_rtrans, mom_fluxform, mom_u_adv_uu, mom_u_adv_vu, mom_u_adv_wu, mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_u_metric_sphere, mom_u_xviscflux, mom_u_yviscflux, mom_uv_boundary, mom_v_adv_uv, mom_v_adv_vv, mom_v_adv_wv, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder, mom_v_metric_sphere, mom_v_xviscflux, mom_v_yviscflux
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_calc_stats_rs, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, mon_writestats_rs, nlatbnd
- pkg/obcs: obcs_add_tides, obcs_adjust, obcs_adjust_uvice, obcs_apply_eta, obcs_apply_ptracer, obcs_apply_r_star, obcs_apply_seaice, obcs_apply_surf_dr, obcs_apply_ts, obcs_apply_uv, obcs_apply_uvice, obcs_apply_w, obcs_balance_flow, obcs_calc, obcs_calc_stevens, obcs_check, obcs_check_depths, obcs_copy_tracer, obcs_copy_uv_n, obcs_cost_ageos, obcs_cost_driver, obcs_cost_final, obcs_cost_ob_e, obcs_cost_ob_n, obcs_cost_ob_s, obcs_cost_ob_w, obcs_cost_vol, obcs_cost_weights, obcs_diag_balance, obcs_exchanges, obcs_exf_load, obcs_exf_read_xz, obcs_exf_read_yz, obcs_fields_load, obcs_init_fixed, obcs_init_variables, obcs_mon_stats_ew_rl, obcs_mon_stats_ns_rl, obcs_mon_writestats, obcs_monitor, obcs_output, obcs_prescribe_read, obcs_read_pickup, obcs_save_uv_n, obcs_seaice_sponge_a, obcs_seaice_sponge_h, obcs_seaice_sponge_sl, obcs_seaice_sponge_sn, obcs_set_connect, obcs_sponge_s, obcs_sponge_t, obcs_sponge_u, obcs_sponge_v, obcs_stevens_calc_tracer_east, obcs_stevens_calc_tracer_north, obcs_stevens_calc_tracer_south, obcs_stevens_calc_tracer_west, obcs_stevens_save_tracers, obcs_time_interp_xz, obcs_time_interp_yz, obcs_u1_adv_tracer, obcs_write_pickup, orlanski_east, orlanski_init, orlanski_north, orlanski_south, orlanski_west
- pkg/rw: get_write_global_fld, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rs, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_bottomdrag_coeffs, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stress, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check_pickup, seaice_cost_export, seaice_diffusion, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_growth, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_lsr_calc_coeffs, seaice_lsr_rhsu, seaice_lsr_rhsv, seaice_lsr_tridiagu, seaice_lsr_tridiagv, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_oceandrag_coeffs, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_residual, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys
- pkg/thsice: thsice_advdiff, thsice_advection, thsice_albedo, thsice_ave, thsice_balance_frw, thsice_calc_thickn, thsice_check, thsice_check_conserv, thsice_cost_driver, thsice_cost_final, thsice_cost_test, thsice_diag_sufx, thsice_diagnostics_init, thsice_diagnostics_state, thsice_diffusion, thsice_do_advect, thsice_do_exch, thsice_extend, thsice_get_bulkf, thsice_get_exf, thsice_get_ocean, thsice_get_precip, thsice_get_velocity, thsice_impl_temp, thsice_ini_vars, thsice_init_fixed, thsice_main, thsice_map_exf, thsice_mnc_init, thsice_monitor, thsice_output, thsice_read_pickup, thsice_reshape_layers, thsice_salt_plume, thsice_slab_ocean, thsice_solve4temp, thsice_step_fwd, thsice_step_temp, thsice_turnoff_io, thsice_write_pickup

## offline_exf_seaice/input_ad.thsice

Run `$MJX_REFERENCE/coverage/offline_exf_seaice/input_ad.thsice/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/offline_exf_seaice/input_ad.thsice/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| thermodynamic sea ice (pkg/thsice) | thsice_main (60 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (60 calls) | yes |
| cost function | global fc = 16033697113.2022; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: AdamsBashforthGt, diags_opOceWeighted, doAB_onGtGs, doThetaClimRelax, dumpInitAndLast, fluidIsWater, inAdExact, momDissip_In_AB, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, snapshot_mdsio, tempForcing, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, applyExchUV_early, bottomVisc_pCell, calc_wVelocity, cg2dFullAdjoint, debugMode, deepAtmosphere, doResetHFactors, doSaltClimRelax, exactConserv, fluidIsAir, globalFiles, implicitDiffusion, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momAdvection, momForcing, momImplVertAdv, momPressureForcing, momViscosity, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, rotateStressOnAgrid, saltAdvection, saltForcing, saltImplVertAdv, saltIsActiveTr, saltMultiDimAdvec, saltSOM_Advection, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFREEDRIFTswitchInAd, stressIsOnCgrid, tempImplVertAdv, tempIsActiveTr, tempMultiDimAdvec, tempSOM_Advection, twoDigitYear, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCDscheme, useCoriolis, useCoupler, useCtrlCostContribution, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useGMRediInAdMode, useHarmonicVisc, useKPPinAdMode, useMin4hFacEdges, useMultiDimAdvec, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useObcsCostContribution, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSEAICEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid, vectorInvariantMomentum
- selectors: exf_adjMonSelect=1, pCellMix_select=0, saltAdvScheme=2, saltVertAdvScheme=2, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=0, selectSigmaCoord=0, tempAdvScheme=2, tempVertAdvScheme=2

### Executed routines (346)

**eesupp/src** (73)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 1798 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 6789 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 484 | 14/15 | 45 | 44:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 676 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 984 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 60 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1660 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 676 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 676 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 676 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 676 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 984 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 984 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 984 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 984 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 60 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 60 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 125 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 252 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 732 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 122 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 22165 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 262 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 23 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 22 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 471 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 6190 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 12734 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 21 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 618 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 576 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 493 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 6640 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 156 | 7/7 | - | - |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 9 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 41 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 106 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 178 | 43/49 | 648-650, 668-671 | 647:1/2, 662:1/2, 664:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 2276 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `timer_control` | timers.F:74 | 2296 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 2296 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 2296 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 1149 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 1147 | 4/4 | - | - |
| `ucase` | utils.F:311 | 4595 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 3 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 33 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 106 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 130 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 44 | 32/32 | - | - |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (82)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 240 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 240 | 13/26 | 467, 469, 471, 626-632, 639-643, 740 | 466:1/2, 468:1/2, 470:1/2, 618:1/2, 638:1/2, 736:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 240 | 16/94 | 132-166, 269-277, 287-406 | 83:1/2, 119:1/2, 266:1/2, 284:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 240 | 19/26 | 87-94, 124-128 | 80:1/2, 113:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 240 | 34/162 | 149, 184, 212-240, 256, 268-610, 629-648 | 100:1/2, 121:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 252:1/2, 253:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `config_check` | config_check.F:14 | 1 | 87/489 | 92-97, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 464-470, 484-490, 497-503, 510-516, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 463:1/2, 482:1/2, 495:1/2, 508:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 435/537 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 268-281, 307-322, 533-538, 554-588, 595, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 240 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 240 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 240 | 24/32 | 79-84, 123-125 | 61:1/2, 68:1/2, 76:1/2, 94:1/2, 95:1/2, 117:1/2, 121:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 60 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 60 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 60 | 68/116 | 257-264, 320-323, 450-464, 471, 557, 593-596, 771-784, 797-798, 811-845, 869-874, 882, 899, 934, 1116, 1119, 1123 | 248:1/2, 251:1/2, 256:1/2, 291:1/2, 377:1/2, 397:1/2, 421:1/2, 467:1/2, 553:1/2, 577:1/2, 589:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 878:1/2, 896:1/2, 932:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 60 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 61 | 10/19 | 97-110, 133, 194, 242 | 96:1/2, 116:1/2, 132:1/2, 193:1/2, 206:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 60 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 60 | 48/88 | 337-340, 358, 389, 413, 492-562, 591-599, 609-610, 665, 684-700, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 409:1/2, 490:1/2, 582:1/2, 604:1/2, 664:1/2, 682:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 60 | 3/109 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 60 | 39/65 | 75, 151-153, 268-285, 301-306, 325-343, 368-372 | 74:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 299:1/2, 310:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 240 | 14/60 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 1 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 60 | 12/21 | 64, 77-88, 245-252 | 63:1/2, 75:1/2, 242:1/2 |
| `forward_step` | forward_step.F:70 | 60 | 72/99 | 508-512, 730-734, 767-769, 903-905, 927-929, 980, 989-991, 1109-1111, 1176, 1201-1202 | 424:1/2, 507:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 78/89 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/65 | 57-63, 84-112, 121-180, 191 | 55:1/2, 68:1/2, 72:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 33/70 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274, 286-288 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2, 284:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 47/47 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 31/49 | 54, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 193 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 192:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 177/193 | 210-212, 247-261, 435, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 29/53 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 398/880 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 623-625, 649-651, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 653:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1446:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 23/38 | 76-80, 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 24/47 | 77-81, 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 22/22 | - | 55:1/2, 57:1/2, 60:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 124:1/2, 137:1/2, 140:1/2, 145:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 1 | 17/75 | 92-196, 211-221, 279-282, 291-293, 329-331 | 90:1/2, 207:1/2, 277:1/2, 290:1/2, 298:1/2, 321:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 4 | 14/28 | 95-117, 187-193 | 93:1/2, 177:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 60 | 26/34 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 60 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 60 | 7/19 | 65-90, 95, 128 | 63:1/2, 94:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 99/99 | - | 100:1/2, 232:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/111 | 132-140, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 193:1/2, 200:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 254:1/2, 260:1/2, 264:1/2, 269:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 380:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 23/37 | 170-176, 202-204, 221-223, 367-369, 512-514, 675-677 | 144:1/2, 167:1/2, 193:1/2, 200:1/2, 219:1/2, 249:1/2, 251:1/2, 358:1/2, 365:1/2, 510:1/2, 528:1/2, 530:1/2, 651:1/2, 655:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 19/24 | 169, 174, 629-631, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 291:1/2, 293:1/2, 435:1/2, 450:1/2, 452:1/2, 617:1/2, 627:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 18 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 12/12 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 3 | 31/37 | 68-70, 76, 82-83 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 10/13 | 112, 186, 225 | 110:1/2, 184:1/2, 191:1/2, 223:1/2, 255:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 60 | 28/64 | 67-70, 78, 85-127, 168-174 | 66:1/2, 77:1/2, 83:1/2, 146:1/2, 160:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 100/166 | 60-63, 67, 71, 79, 109-115, 120-126, 171-175, 186-189, 192-195, 200, 206, 212, 218, 224, 230, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 49/195 | 101-133, 140-165, 180-187, 193-201, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 192:1/2, 202:2/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `temp_integrate` | temp_integrate.F:13 | 240 | 55/68 | 169, 171, 194, 275-282, 328, 369-371, 392, 494, 503, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 272:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 391:1/2, 398:1/2, 481:1/2, 495:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 43/43 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 60 | 39/52 | 158, 282, 330-336, 359, 389, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 358:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 240 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 60 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 21/22 | 107 | 55:1/2, 78:1/2, 83:1/2, 106:1/2, 112:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 50/75 | 245-250, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 252:1/2, 255:1/2, 256:1/2, 257:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 61 | 18/20 | 87, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (25)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 10 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 9 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 1 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 5 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 4 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `active_write_xz` | active_file.F:482 | 8 | 7/7 | - | - |
| `active_write_xz_rl` | active_file_control_slice.F:701 | 8 | 10/19 | 781-799, 809 | 775:1/2, 804:1/2, 813:1/2 |
| `active_write_yz` | active_file.F:543 | 8 | 7/7 | - | - |
| `active_write_yz_rl` | active_file_control_slice.F:937 | 8 | 10/19 | 1017-1035, 1045 | 1011:1/2, 1040:1/2, 1049:1/2 |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 2 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 60 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 60 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 80/148 | 63-68, 127-132, 157, 159, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 14 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 14 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 28 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 1 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 60 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 60 | 2/2 | - | - |

**pkg/cost** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 8/12 | 35-39 | 33:1/2, 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 48/48 | - | 82:1/2, 102:1/2, 105:1/2, 110:1/2, 113:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 22/22 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 60 | 4/4 | - | 121:1/2, 136:1/2 |

**pkg/ctrl** (34)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 1 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 2 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 1 | 3/13 | 43-54 | 41:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 34/72 | 104-110, 124-128, 139-143, 182-186, 226-230, 323-328, 376-381, 591-594 | 80:1/2, 101:1/2, 103:1/2, 121:1/2, 136:1/2, 179:1/2, 225:1/2, 320:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/37 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/83 | 70-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 123 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 120 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 120 | 12/29 | 190-197, 206-223 | 79:1/2, 177:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 122 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 1 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 11 | 31/50 | 80-87, 114-126, 171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 149:1/2, 158:1/2, 161:1/2, 167:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 77/86 | 233-238, 283, 285, 301, 303, 312-314 | 231:1/2, 251:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 8 | 15/26 | 72-76, 87-88, 111-112, 118-124 | 86:1/2, 89:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 24/24 | - | 81:1/2, 97:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 203/209 | 182, 184, 216-219 | 168:1/2, 181:1/2, 183:1/2, 192:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 60 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 1 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 60 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 30/49 | 155-166, 201, 219, 221, 358, 360, 362, 364, 395 | 128:1/2, 154:1/2, 200:1/2, 218:1/2, 220:1/2, 354:1/2, 355:1/2, 357:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 65/80 | 151-153, 157-159, 162, 183-190, 437 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 436:1/2, 459:1/2, 501:1/2 |
| `ctrl_mask_set_xz` | ctrl_mask_set_xz.F:3 | 2 | 29/37 | 88-98 | 64:1/2, 86:1/2 |
| `ctrl_mask_set_yz` | ctrl_mask_set_yz.F:3 | 2 | 29/37 | 89-99 | 64:1/2, 87:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 208/247 | 181-186, 531, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 486:1/2, 492:1/2, 498:1/2, 530:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 11 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 6 | 16/16 | - | 71:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 1 | 10/10 | - | - |
| `ctrl_set_globfld_xz` | ctrl_set_globfld_xz.F:3 | 2 | 17/18 | 69 | 66:1/2, 74:1/2 |
| `ctrl_set_globfld_yz` | ctrl_set_globfld_yz.F:3 | 2 | 17/18 | 69 | 66:1/2, 74:1/2 |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 91/145 | 153-155, 161-171, 184-187, 218-222, 229-241, 267-292, 304-306, 318-321, 324-327 | 146:1/2, 160:1/2, 178:1/2, 217:1/2, 228:1/2, 266:1/2, 302:1/2, 317:1/2, 323:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 2 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/diagnostics** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/372 | 123-653 | 109:1/2, 111:1/2 |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 180 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 60 | 90/101 | 236, 241-242, 325-333, 403, 502-506 | 230:1/2, 240:1/2, 298:1/2, 304:1/2, 388:1/2, 415:1/2, 501:1/2, 507:1/2, 520:1/2, 521:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 31/149 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 348-357, 399-402, 550-569, 575-578, 617-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 616:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 27/72 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 60 | 3/25 | 41-73 | 39:1/2 |
| `exf_filter_rl` | exf_filter_rl.F:3 | 21 | 12/22 | 66-78 | 38:1/2, 41:1/2, 58:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 7 | 17/26 | 890-893, 901-909 | 888:1/2, 900:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 60 | 14/14 | - | 68:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 7 | 9/43 | 65-76, 111-128, 132-141 | 80:1/2, 106:1/2, 109:1/2, 131:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 420 | 11/38 | 220-259 | 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 60 | 51/72 | 97, 147-152, 315-320, 498, 501, 504, 507, 512, 520, 525, 535, 545 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 496:1/2, 499:1/2, 502:1/2, 505:1/2, 510:1/2, 518:1/2, 523:1/2, 533:1/2, 543:1/2, 546:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 60 | 41/44 | 202-203, 329 | 164:1/2, 171:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 60 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 14 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 75/143 | 64-65, 88-114, 137-143, 149-155, 159-179, 185-191, 195-201, 207-213, 262-268, 282-289, 309-315, 359-366, 401-431, 436-466, 475, 488-495, 590-593, 617-619 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 230:1/2, 260:1/2, 270:1/2, 272:1/2, 280:1/2, 307:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 399:1/2, 434:1/2, 472:1/2, 474:1/2, 486:1/2, 588:1/2, 610:1/2, 615:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 17 | 20/26 | 101-107 | 100:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 39/47 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 60 | 60/86 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 60 | 66/73 | 74, 114-116, 164, 186, 192, 228 | 67:1/2, 71:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 60 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 442/463 | 285-290, 1000-1003, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 998:1/2, 1005:1/2, 1006:1/2, 1007:1/2, 1008:1/2, 1009:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1016:1/2, 1017:1/2, 1018:1/2, 1019:1/2, 1020:1/2, 1022:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 1020 | 27/63 | 124-129, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 60 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 166/202 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 385-395, 402-408, 487-493, 500-501, 548-555, 571-581, 585-586, 625-626, 684-690, 769-771, 795-801 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 7 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 60 | 40/83 | 104-115, 158-240 | 79:1/2, 102:1/2, 126:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advscheme_get` | gad_advscheme.F:116 | 4 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 240 | 62/187 | 181-184, 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 606-608, 620, 646-647, 793 | 176:1/2, 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diff_r` | gad_diff_r.F:7 | 240 | 7/10 | 59-64 | 52:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 96/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 159-162, 171-172, 175-176, 180, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 35/58 | 81-135 | 52:1/2, 73:1/2 |
| `grdchk_get_obcs_mask` | grdchk_get_obcs_mask.F:6 | 1 | 13/49 | 70-135 | 67:1/2, 69:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 13/25 | 87-89, 98-100, 107-109, 116-123 | 84:1/2, 95:1/2, 104:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/139 | 116-126, 132, 163, 192-251, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/mdsio** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 25 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 76 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 21 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 39 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 489, 529, 535-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 487:1/4, 493:1/2, 527:1/2, 530:1/2, 540:1/2, 544:1/2 |
| `mds_seg4torl_2d` | mdsio_segxtorx_2d.F:12 | 128 | 5/7 | 39-40 | 38:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 1 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 87 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 545:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 327 | 40/52 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_write_sec_xz` | mdsio_write_section.F:13 | 16 | 43/81 | 109-115, 127, 139-149, 190-193, 203, 209-211, 217-246, 259-260 | 107:1/2, 124:1/2, 125:1/2, 138:1/2, 157:1/2, 174:1/2, 183:1/2, 200:1/2, 201:1/2, 204:1/2, 249:1/2, 258:1/2, 267:1/2 |
| `mds_write_sec_yz` | mdsio_write_section.F:274 | 16 | 43/81 | 369-375, 387, 399-409, 450-453, 463, 469-471, 477-506, 519-520 | 367:1/2, 384:1/2, 385:1/2, 398:1/2, 417:1/2, 434:1/2, 443:1/2, 460:1/2, 461:1/2, 464:1/2, 509:1/2, 518:1/2, 527:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 250:1/2 |

**pkg/monitor** (20)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 2 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 1 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 1 | 10/13 | 41-45 | 39:1/2, 40:2/2 |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 36 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 1 | 52/125 | 167-320 | 70:1/2, 74:1/2, 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 469 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 2 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 467 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 18 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 1 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_latbnd_rl` | mon_stats_latbnd_rl.F:8 | 35 | 54/62 | 62-68, 72-73 | 60:1/2, 71:1/2, 128:1/2, 132:1/2, 136:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 1 | 39/50 | 95, 123-132, 178-179, 196-198 | 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 1 | 72/127 | 145-237, 243-278 | 143:1/2, 242:1/2, 286:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 22 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 61 | 58/67 | 57, 119-120, 135-145 | 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |
| `nlatbnd` | mon_stats_latbnd_rl.F:149 | 117600 | 5/5 | - | - |

**pkg/obcs** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `obcs_readparms` | obcs_readparms.F:6 | 1 | 5/382 | 222-850 | 212:1/2, 214:1/2 |

**pkg/rw** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 24 | 7/7 | - | - |
| `read_rec_xy_rl` | read_rec.F:78 | 2 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 23 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 11 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 25 | 7/7 | - | - |
| `write_rec_xz_rl` | write_rec.F:661 | 8 | 7/7 | - | - |
| `write_rec_yz_rl` | write_rec.F:793 | 8 | 7/7 | - | - |

**pkg/seaice** (3)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 49/153 | 54, 171-471 | 53:1/2, 165:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 5/834 | 241-1559 | 231:1/2, 233:1/2 |

**pkg/thsice** (23)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `thsice_albedo` | thsice_albedo.F:6 | 240 | 24/35 | 94-98, 134, 146-151, 159-161 | 92:1/2, 129:1/2, 142:1/2, 156:1/2 |
| `thsice_ave` | thsice_ave.F:6 | 240 | 3/9 | 65-74 | 62:1/2 |
| `thsice_calc_thickn` | thsice_calc_thickn.F:10 | 240 | 214/285 | 438, 526-533, 561-562, 633-634, 656-663, 695-701, 725-726, 745-750, 786-789, 832-841, 879-886, 923-934, 964-973, 1080-1085, 1088 | 391:1/2, 437:1/2, 519:1/2, 557:1/2, 629:1/2, 654:1/2, 692:1/2, 723:1/2, 739:1/2, 744:1/2, 785:1/2, 826:1/2, 830:1/2, 866:1/2, 875:1/2, 917:1/2, 921:1/2, 962:1/2, 1008:1/2, 1073:1/2, 1076:1/2, 1087:1/2 |
| `thsice_check` | thsice_check.F:10 | 1 | 12/43 | 51-56, 60-63, 76-85, 92-94, 111-119, 124-127 | 39:1/2, 47:1/2, 58:1/2, 74:1/2, 90:1/2, 107:1/2, 122:1/2 |
| `thsice_cost_driver` | thsice_cost_driver.F:9 | 60 | 3/3 | - | - |
| `thsice_cost_final` | thsice_cost_final.F:6 | 1 | 9/9 | - | - |
| `thsice_cost_init_varia` | thsice_cost_init_varia.F:3 | 1 | 6/6 | - | - |
| `thsice_cost_test` | thsice_cost_test.F:9 | 60 | 10/19 | 98-115 | 80:1/2, 85:1/2 |
| `thsice_extend` | thsice_extend.F:9 | 240 | 48/48 | - | 185:1/2 |
| `thsice_get_exf` | thsice_get_exf.F:12 | 5040 | 58/122 | 220, 267-431, 492-495 | 219:1/2, 263:1/2, 489:1/2 |
| `thsice_get_ocean` | thsice_get_ocean.F:9 | 240 | 19/26 | 124-133 | 111:1/2 |
| `thsice_ini_vars` | thsice_ini_vars.F:9 | 1 | 62/73 | 116, 127, 130, 133-134, 137, 172-178 | 112:1/2, 120:1/2, 123:1/2, 126:1/2, 129:1/2, 132:1/2, 136:1/2, 139:1/2, 171:1/2 |
| `thsice_init_fixed` | thsice_init_fixed.F:6 | 1 | 3/4 | 41 | 40:1/2 |
| `thsice_main` | thsice_main.F:12 | 60 | 29/45 | 86-91, 236, 242, 244, 265-276 | 84:1/2, 156:1/2, 235:1/2, 238:1/2, 243:1/2, 258:1/2, 264:1/2 |
| `thsice_map_exf` | thsice_map_exf.F:9 | 240 | 13/20 | 86-88, 110-120 | 85:1/2, 104:1/2 |
| `thsice_monitor` | thsice_monitor.F:6 | 61 | 114/116 | 78, 263 | 71:1/2, 75:1/2, 97:1/2, 134:1/2, 157:1/2, 175:1/2, 207:1/2, 231:1/2, 262:1/2, 266:1/2, 270:1/2 |
| `thsice_output` | thsice_output.F:6 | 61 | 21/24 | 67, 96-98 | 59:1/2, 64:1/2, 94:1/2, 139:1/2 |
| `thsice_readparms` | thsice_readparms.F:6 | 1 | 168/199 | 88-93, 226-233, 236-243, 284-292, 295-303 | 86:1/2, 96:1/2, 156:1/2, 225:1/2, 235:1/2, 248:1/2, 252:1/2, 255:1/2, 281:1/2, 294:1/2, 373:1/2 |
| `thsice_solve4temp` | thsice_solve4temp.F:13 | 240 | 100/106 | 234, 252-254, 262, 388 | 231:1/2, 249:1/2, 261:1/2, 265:1/2, 385:1/2 |
| `thsice_step_fwd` | thsice_step_fwd.F:12 | 240 | 75/78 | 162, 178-180 | 160:1/2, 175:1/2, 362:1/2, 388:1/2 |
| `thsice_step_temp` | thsice_step_temp.F:9 | 240 | 31/31 | - | - |
| `thsice_turnoff_io` | thsice_turnoff_io.F:6 | 1 | 5/5 | - | 42:1/2 |
| `thsice_write_pickup` | thsice_write_pickup.F:5 | 1 | 16/18 | 65-66 | 64:1/2, 106:1/2 |

### Compiled but not executed (942 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, apply_forcing_s, apply_forcing_u, apply_forcing_v, calc_div_ghat, calc_eddy_stress, calc_grad_phi_fv, calc_grad_phi_hyd, calc_grad_phi_surf, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_r_star, calc_surf_dr, calc_viscosity, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, correction_step, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, freesurf_rescale_g, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, salt_integrate, solve_for_pressure, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, swfrac, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz_loc, active_write_xz_rs, active_write_yz_loc, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_pack, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_is_on, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_advection, gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/grdchk: grdchk_get_position, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_hfacz, mom_calc_ke, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_init_fixed, mom_quasihydrostatic, mom_u_botdrag_coeff, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_botdrag_coeff, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_calc_rtrans, mom_fluxform, mom_u_adv_uu, mom_u_adv_vu, mom_u_adv_wu, mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_u_metric_sphere, mom_u_xviscflux, mom_u_yviscflux, mom_uv_boundary, mom_v_adv_uv, mom_v_adv_vv, mom_v_adv_wv, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder, mom_v_metric_sphere, mom_v_xviscflux, mom_v_yviscflux
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_calc_advcfl_glob, mon_calc_advcfl_tile, mon_calc_stats_rs, mon_out_rs, mon_printstats_rl, mon_stats_rl, mon_writestats_rs
- pkg/obcs: obcs_add_tides, obcs_adjust, obcs_adjust_uvice, obcs_apply_eta, obcs_apply_ptracer, obcs_apply_r_star, obcs_apply_seaice, obcs_apply_surf_dr, obcs_apply_ts, obcs_apply_uv, obcs_apply_uvice, obcs_apply_w, obcs_balance_flow, obcs_calc, obcs_calc_stevens, obcs_check, obcs_check_depths, obcs_copy_tracer, obcs_copy_uv_n, obcs_cost_ageos, obcs_cost_driver, obcs_cost_final, obcs_cost_ob_e, obcs_cost_ob_n, obcs_cost_ob_s, obcs_cost_ob_w, obcs_cost_vol, obcs_cost_weights, obcs_diag_balance, obcs_exchanges, obcs_exf_load, obcs_exf_read_xz, obcs_exf_read_yz, obcs_fields_load, obcs_init_fixed, obcs_init_variables, obcs_mon_stats_ew_rl, obcs_mon_stats_ns_rl, obcs_mon_writestats, obcs_monitor, obcs_output, obcs_prescribe_read, obcs_read_pickup, obcs_save_uv_n, obcs_seaice_sponge_a, obcs_seaice_sponge_h, obcs_seaice_sponge_sl, obcs_seaice_sponge_sn, obcs_set_connect, obcs_sponge_s, obcs_sponge_t, obcs_sponge_u, obcs_sponge_v, obcs_stevens_calc_tracer_east, obcs_stevens_calc_tracer_north, obcs_stevens_calc_tracer_south, obcs_stevens_calc_tracer_west, obcs_stevens_save_tracers, obcs_time_interp_xz, obcs_time_interp_yz, obcs_u1_adv_tracer, obcs_write_pickup, orlanski_east, orlanski_init, orlanski_north, orlanski_south, orlanski_west
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rs, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, advect, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advdiff, seaice_advection, seaice_bottomdrag_coeffs, seaice_budget_ocean, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stress, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check, seaice_check_pickup, seaice_cost_accumulate_mean, seaice_cost_export, seaice_cost_final, seaice_cost_init_fixed, seaice_cost_sensi, seaice_cost_test, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_dynsolver, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_get_dynforcing, seaice_growth, seaice_growth_adx, seaice_init_fixed, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_lsr_calc_coeffs, seaice_lsr_rhsu, seaice_lsr_rhsv, seaice_lsr_tridiagu, seaice_lsr_tridiagv, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_model, seaice_mom_advection, seaice_monitor, seaice_obcs_output, seaice_ocean_stress, seaice_oceandrag_coeffs, seaice_output, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_reg_ridge, seaice_residual, seaice_scalprod, seaice_sidedrag_stress, seaice_solve4temp, seaice_summary, seaice_tracer_phys, seaice_turnoff_io, seaice_write_pickup
- pkg/thsice: thsice_advdiff, thsice_advection, thsice_balance_frw, thsice_check_conserv, thsice_diag_sufx, thsice_diagnostics_init, thsice_diagnostics_state, thsice_diffusion, thsice_do_advect, thsice_do_exch, thsice_get_bulkf, thsice_get_precip, thsice_get_velocity, thsice_impl_temp, thsice_mnc_init, thsice_read_pickup, thsice_reshape_layers, thsice_salt_plume, thsice_slab_ocean

