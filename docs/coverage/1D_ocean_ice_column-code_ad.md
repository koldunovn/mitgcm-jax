# Coverage: 1D_ocean_ice_column (code_ad) — M4 porting worklist

Written by `tools/coverage.py` (mitjax 7c620b3) from the gcov build `1D_ocean_ice_column-code_ad-63cdc0b-dad681f-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 798 files from the link farm, 105 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_ad` | `job27856010` | 11 | 0 / 0 | 453 | yes | identical (1973 lines) |

## 1D_ocean_ice_column/input_ad

Run `$MJX_REFERENCE/coverage/1D_ocean_ice_column/input_ad/job27856010` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/1D_ocean_ice_column/input_ad/job27856010/gcov-dad681f/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| sea-ice volume changes | 11 blocks, first 0.000000e+00, last 8.405834e-03, min 0.000000e+00, max 8.405834e-03 | yes |
| theta | 11 blocks, first 8.785391e-01, last 8.804718e-01, min 8.782293e-01, max 8.804718e-01 | yes |
| sea-ice thermodynamics | seaice_growth (10 calls) | yes |
| EXF bulk formulae | exf_bulkformulae (10 calls) | yes |
| KPP | kpp_calc (10 calls) | yes |
| time-varying forcing controls | ctrl_map_gentim2d (10 calls) | yes |
| cost function | global fc = 691935.719996437; terms:  | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: calc_wVelocity, diags_opOceWeighted, doAB_onGtGs, dumpInitAndLast, fluidIsWater, inAdExact, LimitHblStable, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momTidalForcing, momViscosity, monitor_stdio, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltIsActiveTr, SEAICE_doOpenWaterGrowth, SEAICE_dump_mdsio, SEAICE_mon_stdio, SEAICE_useMultDimSnow, SEAICEadvArea, SEAICEadvHeff, SEAICEupdateOceanStress, SEAICEuseFluxForm, snapshot_mdsio, tempAdvection, tempForcing, tempIsActiveTr, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useCtrlCostContribution, useHarmonicVisc, useKPPinAdMode, usePW79thermodynamics, useSEAICEinAdMode, useSingleCpuInput, usingGregorianCalendar, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, doResetHFactors, doSaltClimRelax, doThetaClimRelax, exactConserv, fluidIsAir, globalFiles, highOrderVorticity, implicitIntGravWave, interDiffKr_pCell, interViscAr_pCell, KPPuseDoubleDiff, linFSConserveTr, momImplVertAdv, noNegativeEvap, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, rotateStressOnAgrid, saltImplVertAdv, saltMultiDimAdvec, saltSOM_Advection, SEAICE_doOpenWaterMelt, SEAICE_growMeltByConv, SEAICE_mcPheeStepFunc, SEAICE_salinityTracer, SEAICEheatConsFix, SEAICEmomAdvection, SEAICEmultiDimAdvection, SEAICErestoreUnderIce, SEAICEuseBDF2, SEAICEuseDYNAMICSswitchInAd, SEAICEuseFREEDRIFTswitchInAd, stressIsOnCgrid, tempImplVertAdv, tempMultiDimAdvec, tempSOM_Advection, twoDigitYear, upwindShear, upwindVorticity, use3Dsolver, useAbsVorticity, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useExfYearlyFields, useExfZenAlbedo, useExfZenIncoming, useGGL90inAdMode, useGMRediInAdMode, useJamartMomAdv, useMaykutSatVapPoly, useMin4hFacEdges, useMultiDimAdvec, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useSALT_PLUMEinAdMode, useSmag3D, useSRCGSolver, useStabilityFct_overIce, useStrainTensionVisc, useVariableVisc, usingCurvilinearGrid, usingCylindricalGrid, usingJulianCalendar, usingModelCalendar, usingMPI, usingNoLeapYearCal, usingPCoords, usingSphericalPolarGrid
- selectors: exf_adjMonSelect=1, monitorSelect=3, pCellMix_select=0, saltVertAdvScheme=30, select_rStar=0, select_ZenAlbedo=0, selectAddFluid=0, selectBotDragQuadr=-1, selectKEscheme=0, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempVertAdvScheme=30

### Executed routines (453)

**eesupp/src** (77)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 8264 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 2521 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 355 | 15/15 | - | 58:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 313 | 39/77 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 121-159, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 203:1/2, 236:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 264 | 38/77 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 121-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 10 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 577 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 313 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 313 | 38/90 | 289-324, 346-381 | 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 313 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 313 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 264 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 264 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 264 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 264 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_s3d_rl` | exch_s3d_rl.F:8 | 20 | 11/12 | 60 | 56:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 10 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 20 | 14/40 | 85-153 | 75:1/2, 77:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 35 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 190 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 192 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 33 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 15827 | 2/2 | - | - |
| `gather_2d_r4` | gather_2d_r4.F:7 | 407 | 14/14 | - | 61:1/2, 63:1/2 |
| `gather_2d_r8` | gather_2d_r8.F:7 | 468 | 14/14 | - | 61:1/2, 63:1/2 |
| `global_max_r8` | global_max.F:97 | 794 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 73 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 114 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 1848 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 7125 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 19409 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 115:1/2, 119:1/2, 142:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 18 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 2298 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 296 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 458 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 2308 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 301 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 12 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 62 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 144 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 262 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 4883 | 23/31 | 90, 96-99, 131, 135, 154-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 151:1/2 |
| `scatter_2d_r4` | scatter_2d_r4.F:7 | 104 | 14/14 | - | 61:1/2, 156:1/2 |
| `scatter_2d_r8` | scatter_2d_r8.F:7 | 274 | 14/14 | - | 61:1/2, 156:1/2 |
| `timer_control` | timers.F:74 | 500 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 500 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 500 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 251 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 249 | 4/4 | - | - |
| `ucase` | utils.F:311 | 1000 | 7/8 | 333 | 332:1/2 |
| `write_0d_c` | write_utils.F:579 | 9 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 52 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 144 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 217 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 45 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (99)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 460 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 230 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 230 | 18/50 | 467, 469, 471, 563-604, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 230 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 230 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 40 | 24/30 | 164-166, 195-197 | 132:1/2, 180:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 460 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 230 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 230 | 19/19 | - | - |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 10 | 8/8 | - | - |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 230 | 36/162 | 149, 184, 212-240, 268-610, 636-648 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 10 | 12/16 | 124-127 | 121:1/2 |
| `cg2d` | cg2d.F:13 | 10 | 96/131 | 150-152, 191-192, 330-334, 338-355, 360-364, 393-416 | 117:1/2, 121:1/2, 148:1/2, 190:1/2, 196:1/2, 197:1/2, 204:1/2, 207:1/2, 329:1/2, 337:1/2, 358:1/2, 371:1/2, 392:1/2 |
| `config_check` | config_check.F:14 | 1 | 98/560 | 92-97, 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 308-313, 320-325, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 510-516, 523-529, 535-541, 547-550, 572-575, 582-585, 590-593, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 947-950, 970-979, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 306:1/2, 318:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 508:1/2, 521:1/2, 534:1/2, 545:1/2, 567:1/2, 577:1/2, 588:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 946:1/2, 955:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 446/537 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 263-267, 270-278, 307-322, 470-486, 540-548, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 268:1/2, 279:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `correction_step` | correction_step.F:7 | 10 | 18/25 | 158-167, 180-188 | 156:1/2, 179:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 20 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 230 | 5/5 | - | - |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 230 | 24/32 | 79-84, 123-125 | 61:1/2, 76:1/2, 121:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 10 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 10 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 10 | 71/99 | 467-471, 557, 771-784, 797-798, 811-845, 869-874, 899, 962 | 248:1/2, 251:1/2, 421:1/2, 450:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 896:1/2, 951:1/2, 953:1/2, 1102:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 10 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 11 | 11/17 | 97-110 | 96:1/2, 116:1/2, 144:1/2, 187:1/2, 193:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 10 | 22/26 | 82, 84, 115-117 | 66:1/2, 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 10 | 54/72 | 337-340, 358, 523, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 10 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 10 | 43/67 | 75, 176, 268-285, 301-306, 325-343, 376-378 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 299:1/2, 310:1/2, 364:1/2, 367:1/2 |
| `find_alpha` | find_alpha.F:14 | 230 | 27/76 | 77-79, 85-99, 150-151, 222-337 | 75:1/2, 83:1/2, 113:1/2, 147:1/2 |
| `find_beta` | find_alpha.F:347 | 230 | 25/74 | 410-412, 418-430, 481-482, 539-641 | 408:1/2, 416:1/2, 444:1/2, 478:1/2 |
| `find_bulkmod` | find_rho.F:412 | 1383 | 17/19 | 503-504 | 500:1/2 |
| `find_rho_2d` | find_rho.F:25 | 923 | 17/60 | 98-108, 114-143, 185-265 | 92:1/2, 112:1/2, 148:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 67 | 34/72 | 935-938, 946, 954-961, 1098-1179 | 932:1/2, 941:1/2, 949:1/2, 964:1/2 |
| `find_rhop0` | find_rho.F:275 | 1383 | 14/16 | 350-351 | 347:1/2 |
| `forward_step` | forward_step.F:70 | 10 | 76/87 | 730-734, 767-769, 980, 1176, 1201-1202 | 424:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 897:1/2, 924:1/2, 976:1/2, 979:1/2, 1002:1/2, 1151:1/2, 1169:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `impldiff` | impldiff.F:7 | 40 | 63/63 | - | 133:1/2, 151:1/2, 199:1/2, 219:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 22/65 | 68-112, 121-180, 191 | 55:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 73/255 | 85-86, 88-103, 116-124, 176-365 | 45:1/2, 48:1/2, 84:1/2, 87:1/2, 107:1/2, 114:1/2, 145:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 57/87 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 1 | 28/28 | - | 127:1/2, 136:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 168/193 | 118, 159, 210-212, 228, 247-261, 435, 449-451, 464-465, 471-472, 478-479 | 65:1/2, 116:1/2, 158:1/2, 181:1/2, 188:1/2, 196:1/2, 206:1/2, 227:1/2, 243:1/2, 414:1/2, 415:1/2, 418:1/2, 420:1/2, 448:1/2, 460:1/2, 467:1/2, 474:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 32/52 | 146-179, 191-195 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 206:1/2, 232:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 11/11 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 407/881 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 584, 586-589, 609-612, 615-620, 628-632, 666, 671-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1100-1101, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1142, 1148, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1314-1317, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1397-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 585:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 669:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 23/38 | 76-80, 100, 109-124, 130 | 65:1/2, 88:1/2, 91:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 24/47 | 77-81, 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 92:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 28/28 | - | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 331:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 11 | 14/65 | 92-185, 211-221, 279-282, 329-331 | 90:1/2, 207:1/2, 277:1/2, 325:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 253 | 17/28 | 95-117 | 93:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 10 | 25/32 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 209:1/2, 211:1/2, 229:1/2, 231:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 10 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 10 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 104/118 | 242-261 | 100:1/2, 226:1/2, 227:1/2, 228:1/2, 232:1/2, 241:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 67/112 | 132-140, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 460, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 254:1/2, 260:1/2, 264:1/2, 273:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 333:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 428:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 459:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 32/32 | - | 144:1/2, 158:1/2, 160:1/2, 193:1/2, 200:1/2, 202:1/2, 249:1/2, 251:1/2, 317:1/2, 319:1/2, 358:1/2, 365:1/2, 367:1/2, 374:1/2, 379:1/2, 510:1/2, 512:1/2, 651:1/2, 655:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 18/20 | 169, 637 | 168:1/2, 192:1/2, 194:1/2, 251:1/2, 253:1/2, 291:1/2, 293:1/2, 435:1/2, 601:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 18 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 12/12 | - | - |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 9/9 | - | 184:1/2, 237:1/2, 255:1/2 |
| `pressure_for_eos` | pressure_for_eos.F:6 | 1383 | 8/18 | 80-84, 100-111 | 58:1/2, 73:1/2, 90:1/2 |
| `rotate_uv2en_rl` | rotate_uv2en.F:8 | 32 | 50/64 | 67-70, 78, 100-101, 168-174 | 66:1/2, 77:1/2, 99:1/2, 111:1/2, 146:1/2, 160:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 10 | 48/56 | 192, 273-280, 367-369, 386, 517 | 147:1/2, 177:1/2, 270:1/2, 319:1/2, 366:1/2, 374:1/2, 396:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 99/166 | 60-63, 67, 71, 74-79, 109-115, 120-126, 171-175, 191-195, 202, 208, 214, 220, 226, 230, 259, 261-262, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 186:1/2, 188:1/2, 189:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 53/195 | 101-133, 140-165, 180-187, 207-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 10 | 60/82 | 152-157, 164, 170-176, 218-224, 265, 269, 319, 327, 342-344 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 10 | 48/56 | 194, 275-282, 369-371, 388, 519 | 149:1/2, 179:1/2, 272:1/2, 321:1/2, 368:1/2, 376:1/2, 398:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 51/51 | - | 292:1/2, 382:1/2, 664:1/2, 666:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 24/42 | 639-641, 719-726, 736-797 | 606:1/2, 616:1/2, 633:1/2, 636:1/2, 638:1/2, 707:1/2, 718:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 10 | 42/50 | 158, 395-402 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 230 | 45/62 | 118-121, 210-213, 220-223, 328-332 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 209:1/2, 219:1/2, 321:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 20 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 10 | 5/6 | 117 | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 20/20 | - | 55:1/2, 78:1/2, 88:1/2, 106:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 51/75 | 253-257, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 11 | 18/20 | 87, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 79 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xy` | active_file.F:33 | 72 | 6/6 | - | - |
| `active_read_xyz` | active_file.F:100 | 7 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 39 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xy` | active_file.F:360 | 36 | 7/7 | - | - |
| `active_write_xyz` | active_file.F:421 | 3 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 71-81 | 69:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 2 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 10 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 10 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 81/148 | 63-68, 127-132, 157, 165-172, 175-176, 243-247, 270-273, 276-279, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 156:1/2, 158:1/2, 164:1/2, 174:1/2, 235:1/2, 269:1/2, 275:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 2 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 2 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 4 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 11 | 2/2 | - | - |
| `dummy_in_dynamics` | dummy_in_dynamics.F:6 | 10 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 10 | 2/2 | - | - |

**pkg/cal** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cal_addtime` | cal_addtime.F:3 | 300 | 47/107 | 66-75, 79-81, 89-106, 114-116, 127-140, 169-171, 179, 181-186, 192-199 | 65:1/2, 78:1/2, 88:1/2, 110:1/2, 124:1/2, 167:1/2, 168:1/2, 177:1/2, 180:1/2, 191:1/2, 200:2/2, 214:1/2, 218:1/2 |
| `cal_checkdate` | cal_checkdate.F:3 | 44 | 22/54 | 61-63, 65-70, 76-81, 89-93, 96, 100-102, 106-108, 111, 123-126, 129-132, 135-138 | 59:1/2, 64:1/2, 73:1/2, 88:1/2, 94:1/2, 98:1/2, 103:1/2, 109:1/2, 116:1/2, 122:1/2, 128:1/2, 134:1/2 |
| `cal_compdates` | cal_compdates.F:3 | 22 | 5/5 | - | - |
| `cal_convdate` | cal_convdate.F:3 | 924 | 21/37 | 51-57, 66-68, 71-73, 87-89 | 50:1/2, 65:1/2, 70:1/2, 82:1/2 |
| `cal_copydate` | cal_copydate.F:3 | 153 | 6/6 | - | - |
| `cal_fulldate` | cal_fulldate.F:3 | 44 | 15/37 | 59-65, 71-74, 87-93, 98-101 | 58:1/2, 70:1/2, 77:1/2, 84:1/2 |
| `cal_getdate` | cal_getdate.F:3 | 336 | 16/23 | 57-63 | 55:1/2 |
| `cal_init_fixed` | cal_init_fixed.F:8 | 1 | 6/6 | - | 28:1/2, 42:1/2 |
| `cal_intdays` | cal_intdays.F:3 | 2 | 9/10 | 58 | 55:1/2 |
| `cal_intmonths` | cal_intmonths.F:3 | 2 | 9/11 | 62, 69 | 59:1/2, 67:1/2 |
| `cal_intyears` | cal_intyears.F:3 | 2 | 4/5 | 49 | 47:1/2 |
| `cal_isleap` | cal_isleap.F:3 | 18464 | 9/21 | 41-47, 60-67 | 40:1/2, 50:1/2 |
| `cal_numints` | cal_numints.F:3 | 3 | 7/10 | 61-63 | 53:1/2 |
| `cal_readparms` | cal_readparms.F:3 | 1 | 19/23 | 70-76 | 65:1/2, 68:1/2, 79:1/2, 114:1/2 |
| `cal_set` | cal_set.F:3 | 1 | 60/93 | 104-114, 163-184, 202-204, 207-209, 212-214 | 82:1/2, 99:1/2, 119:1/2, 160:1/2, 201:1/2, 206:1/2, 211:1/2 |
| `cal_subdates` | cal_subdates.F:3 | 2 | 6/24 | 48-57, 65-69, 77-79 | 47:1/2, 60:1/2, 63:1/2 |
| `cal_summary` | cal_summary.F:3 | 1 | 43/43 | - | 49:1/2 |
| `cal_time2dump` | cal_time2dump.F:3 | 20 | 3/15 | 36-51 | 31:1/2 |
| `cal_timeinterval` | cal_timeinterval.F:3 | 312 | 15/36 | 60-66, 73-92 | 57:1/2, 59:1/2 |
| `cal_timepassed` | cal_timepassed.F:3 | 425 | 52/75 | 67-76, 98, 120-127, 148-149, 182-184 | 66:1/2, 87:1/2, 97:1/2, 119:1/2, 147:1/2 |
| `cal_toseconds` | cal_toseconds.F:3 | 382 | 15/26 | 57-63, 69, 92-94 | 56:1/2, 67:1/2, 73:1/2 |

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
| `cost_tile` | cost_tile.F:59 | 10 | 2/2 | - | - |

**pkg/ctrl** (32)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 2 | 8/8 | - | - |
| `ctrl_bound_2d` | ctrl_bound.F:74 | 18 | 3/12 | 103-113 | 101:1/2 |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 2 | 11/13 | 51, 54 | 41:1/2, 49:1/2, 50:1/2, 53:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 24/50 | 102-110, 139-143, 182-186, 376-381, 591-594 | 80:1/2, 101:1/2, 136:1/2, 179:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 24/28 | 74, 83-84, 140 | 65:1/2, 73:1/2, 78:1/2, 82:1/2, 139:1/2, 147:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 59/59 | - | 66:1/2, 131:1/2, 161:1/2, 176:1/2, 186:1/2, 214:1/2 |
| `ctrl_cost_gen2d` | ctrl_cost_gen.F:12 | 9 | 41/42 | 158 | 121:1/2, 154:1/2, 157:1/2, 162:1/2 |
| `ctrl_cost_gen3d` | ctrl_cost_gen.F:187 | 2 | 41/42 | 322 | 283:1/2, 317:1/2, 319:1/2, 326:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 121 | 9/9 | - | - |
| `ctrl_get_gen` | ctrl_get_gen.F:3 | 90 | 36/38 | 194, 200 | 96:1/2, 192:1/2, 198:1/2, 207:1/2 |
| `ctrl_get_gen_rec` | ctrl_get_gen_rec.F:3 | 90 | 35/66 | 99, 104-108, 145, 174-197, 206-223 | 79:1/2, 92:1/2, 100:1/2, 144:1/2, 204:1/2 |
| `ctrl_get_mask2d` | ctrl_get_mask.F:70 | 117 | 5/11 | 128-133 | 126:1/2, 141:1/2 |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 4 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 29 | 25/50 | 80-87, 114-126, 152-171 | 79:1/2, 113:1/2, 132:1/2, 140:1/2, 143:1/2, 149:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 59/68 | 272-273, 283, 285, 301, 303, 312-314 | 251:1/2, 271:1/2, 282:1/2, 284:1/2, 297:1/2, 300:1/2, 302:1/2, 304:1/2, 311:1/2, 380:1/2 |
| `ctrl_init_rec` | ctrl_init_rec.F:5 | 18 | 21/36 | 72-76, 87-88, 90-91, 106-112, 118-124 | 86:1/2, 89:1/2, 93:1/2, 104:1/2, 116:1/2, 128:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 18/18 | - | 81:1/2, 104:1/2, 110:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 113:1/2, 117:1/2, 121:1/2, 168:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 10 | 48/57 | 82, 105, 107, 109, 111, 113, 115, 118, 120 | 81:1/2, 83:1/2, 104:1/2, 106:1/2, 108:1/2, 110:1/2, 112:1/2, 114:1/2, 116:1/2, 119:1/2, 121:1/2 |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 2 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 10 | 18/40 | 80-85, 104-137 | 53:1/2, 79:1/2, 102:1/2 |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 22/25 | 360, 362, 364 | 128:1/2, 354:1/2, 359:1/2, 361:1/2, 363:1/2, 392:1/2, 394:1/2, 434:1/2 |
| `ctrl_map_ini_gentim2d` | ctrl_map_ini_gentim2d.F:9 | 1 | 70/127 | 151-153, 157-159, 162, 183-190, 254-261, 276-393, 437, 464 | 87:1/2, 118:1/2, 150:1/2, 155:1/2, 161:1/2, 182:1/2, 208:1/2, 253:1/2, 272:1/2, 436:1/2, 457:1/2, 459:1/2, 501:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 202/240 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 492:1/2, 498:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 29 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xy` | ctrl_set_globfld_xy.F:3 | 27 | 16/16 | - | 65:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 2 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 90/121 | 153-155, 184-187, 218-222, 267-292, 304-306 | 146:1/2, 178:1/2, 217:1/2, 255:1/2, 259:1/4, 266:1/2, 302:1/2 |
| `ctrl_swapffields` | ctrl_swapffields.F:12 | 9 | 12/12 | - | - |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/ecco** (36)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_averagesfields` | cost_averagesfields.F:3 | 11 | 13/14 | 107 | 74:1/2, 99:1/2, 127:1/2 |
| `cost_averagesflags` | cost_averagesflags.F:4 | 11 | 74/95 | 153, 174-183, 207, 228-239, 263, 285-289 | 152:1/2, 170:1/2, 206:1/2, 224:1/2, 262:1/2, 280:1/2 |
| `cost_averagesgeneric` | cost_averagesgeneric.F:3 | 22 | 37/50 | 113-134, 180, 209 | 111:1/2, 178:1/2, 193:1/2 |
| `cost_averagesinit` | cost_averagesinit.F:3 | 1 | 20/20 | - | - |
| `cost_gencal` | cost_gencal.F:7 | 4 | 17/35 | 75-77, 80-98 | 74:1/2, 78:1/2, 112:1/2 |
| `cost_gencost_all` | cost_gencost_all.F:3 | 1 | 19/23 | 54-56, 61-62 | 53:1/2, 60:1/2, 93:1/2, 94:1/2, 95:1/2, 96:1/2 |
| `cost_gencost_assignperiod` | cost_gencost_assignperiod.F:3 | 11 | 12/37 | 58-62, 70-93 | 56:1/2, 63:1/2 |
| `cost_gencost_boxmean` | cost_gencost_boxmean.F:3 | 1 | 4/45 | 76-169 | 71:1/2 |
| `cost_gencost_bpv4` | cost_gencost_bpv4.F:3 | 1 | 7/104 | 95-361 | 80:1/2, 84:1/2 |
| `cost_gencost_customize` | cost_gencost_customize.F:20 | 11 | 59/97 | 125, 128, 131, 134, 137, 140, 143, 147, 166, 169, 172, 175, 179, 182, 185, 190, 193, 197, 210, 213, 217, 220, 223, 247-250, 253-256, 259-261, 264-266, 269-271 | 122:1/2, 126:1/2, 129:1/2, 132:1/2, 135:1/2, 138:1/2, 141:1/2, 144:1/2, 164:1/2, 167:1/2, 170:1/2, 173:1/2, 177:1/2, 180:1/2, 183:1/2, 188:1/2, 191:1/2, 195:1/2, 207:1/2, 211:1/2, 214:1/2, 218:1/2, 221:1/2, 246:1/2, 252:1/2, 258:1/2, 263:1/2, 268:1/2 |
| `cost_gencost_glbmean` | cost_gencost_glbmean.F:3 | 1 | 2/2 | - | - |
| `cost_gencost_moc` | cost_gencost_moc.F:3 | 1 | 14/89 | 95-270 | 92:1/2 |
| `cost_generic` | cost_generic.F:12 | 2 | 17/18 | 118 | 100:1/2, 101:1/2, 106:1/2, 123:1/2 |
| `cost_genloop` | cost_generic.F:147 | 2 | 79/125 | 290-291, 294-295, 298-299, 302, 321-342, 404, 407, 411, 452, 458-464, 479-483, 492-501, 505-511 | 284:1/2, 285:1/2, 286:1/2, 287:1/2, 288:1/2, 289:1/2, 293:1/2, 297:1/2, 301:1/2, 312:1/2, 362:1/2, 370:1/2, 375:1/2, 393:1/2, 403:1/2, 406:1/2, 409:1/2, 413:1/2, 448:1/2, 451:1/2, 453:1/2, 457:1/2, 478:1/2, 486:1/2, 504:1/2 |
| `cost_genread` | cost_genread.F:7 | 2 | 7/27 | 64-94 | 108:1/2 |
| `ecco_addcost` | ecco_toolbox.F:238 | 2 | 18/20 | 276, 289 | 275:1/2, 286:1/2 |
| `ecco_check` | ecco_check.F:10 | 1 | 59/300 | 248-252, 261-266, 269-274, 277-282, 285-290, 297-301, 320-325, 330-335, 347, 352-358, 365-366, 369-370, 373-374, 380, 382, 384, 388-393, 397-402, 413-432, 440-534, 543-679, 698-703, 715-748, 759-763 | 50:1/2, 247:1/2, 260:1/2, 268:1/2, 276:1/2, 284:1/2, 296:1/2, 305:1/2, 318:1/2, 328:1/2, 344:1/2, 348:1/2, 351:1/2, 364:1/2, 368:1/2, 372:1/2, 376:1/2, 379:1/2, 381:1/2, 383:1/2, 386:1/2, 395:1/2, 412:1/2, 438:1/2, 540:1/2, 688:1/2, 692:1/2, 696:1/2, 714:1/2, 755:1/2, 758:1/2 |
| `ecco_check_files` | ecco_check.F:785 | 2 | 13/36 | 836-841, 852-874, 883-888 | 833:1/2, 845:1/2, 847:1/2, 851:1/2, 879:1/2, 880:1/2, 897:1/2 |
| `ecco_cost_final` | ecco_cost_final.F:6 | 1 | 34/36 | 139-141 | 111:1/2 |
| `ecco_cost_init_barfiles` | ecco_cost_init_barfiles.F:7 | 1 | 28/28 | - | 109:1/2 |
| `ecco_cost_init_fixed` | ecco_cost_init_fixed.F:7 | 1 | 35/48 | 132-133, 138-152, 160, 171 | 83:1/2, 130:1/2, 134:1/2, 157:1/2, 168:1/2, 238:1/2 |
| `ecco_cost_init_varia` | ecco_cost_init_varia.F:12 | 1 | 14/14 | - | 68:1/2 |
| `ecco_cp` | ecco_toolbox.F:138 | 4 | 11/12 | 166 | 165:1/2 |
| `ecco_diffmsk` | ecco_toolbox.F:74 | 2 | 14/15 | 109 | 108:1/2 |
| `ecco_divfield` | ecco_toolbox.F:523 | 4 | 10/12 | 547, 554 | 546:1/2, 553:1/2 |
| `ecco_init_fixed` | ecco_init_fixed.F:8 | 1 | 4/4 | - | 31:1/2 |
| `ecco_init_varia` | ecco_init_varia.F:3 | 1 | 5/5 | - | - |
| `ecco_mult` | ecco_toolbox.F:573 | 2 | 10/11 | 599 | 598:1/2 |
| `ecco_phys` | ecco_phys.F:17 | 11 | 82/123 | 365-389, 411-412, 425-435, 450-451, 454-455, 457-458, 492-499, 555-557, 567, 577-598, 620-628 | 112:1/2, 364:1/2, 406:1/2, 417:1/2, 449:1/2, 452:1/2, 456:1/2, 491:1/2, 548:1/2, 563:1/2, 572:1/2, 618:1/2 |
| `ecco_readbar` | ecco_toolbox.F:817 | 2 | 10/18 | 868, 875-876, 892-896 | 869:1/2, 891:1/2, 906:1/2 |
| `ecco_readparms` | ecco_readparms.F:3 | 1 | 371/497 | 204-209, 513-519, 522-525, 528-531, 534-537, 541-548, 552-558, 561-568, 692-693, 700-710, 715-721, 725-731, 738-739, 774-822, 834-836, 839-841, 844-846, 855, 868-875, 880-887, 891-923 | 202:1/2, 212:1/2, 511:1/2, 521:1/2, 527:1/2, 533:1/2, 539:1/2, 550:1/2, 560:1/2, 576:1/2, 597:1/2, 690:1/2, 698:1/2, 714:1/2, 724:1/2, 736:1/2, 769:1/2, 833:1/2, 838:1/2, 843:1/2, 852:1/2, 854:1/2, 865:1/2, 878:1/2, 890:1/2, 934:1/2 |
| `ecco_readwei` | ecco_toolbox.F:912 | 2 | 13/16 | 960, 965-967 | 959:1/2, 962:1/2 |
| `ecco_summary` | ecco_summary.F:3 | 1 | 61/89 | 76-82, 90-92, 98-101, 105-108, 111-114, 133-135, 138-140 | 69:1/2, 75:1/2, 88:1/2, 97:1/2, 104:1/2, 110:1/2, 127:1/2, 132:1/2, 137:1/2 |
| `ecco_write_pickup` | ecco_write_pickup.F:6 | 1 | 3/3 | - | - |
| `ecco_zero` | ecco_toolbox.F:30 | 505 | 8/8 | - | - |
| `stergloh_output` | stergloh_output.F:8 | 11 | 2/2 | - | - |

**pkg/exf** (26)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exf_adjoint_snapshots` | exf_adjoint_snapshots.F:6 | 30 | 2/2 | - | - |
| `exf_bulkformulae` | exf_bulkformulae.F:9 | 10 | 81/103 | 236, 241-242, 325-333, 346-350, 403, 502-506, 520-525 | 230:1/2, 240:1/2, 283:1/2, 298:1/2, 304:1/2, 357:1/2, 388:1/2, 415:1/2, 481:1/2, 501:1/2, 507:1/2 |
| `exf_check` | exf_check.F:13 | 1 | 30/145 | 58-60, 66-68, 72-83, 87-100, 113-115, 120-124, 141-144, 147-150, 200-203, 206-209, 215-218, 266-269, 274-277, 306-309, 315-318, 333-342, 353-357, 399-402, 550-569, 575-578, 616-621, 634-639, 647-650 | 45:1/2, 54:1/2, 63:1/2, 71:1/2, 86:1/2, 110:1/2, 111:1/2, 119:1/2, 140:1/2, 146:1/2, 199:1/2, 205:1/2, 214:1/2, 265:1/2, 273:1/2, 305:1/2, 314:1/2, 331:1/2, 346:1/2, 398:1/2, 548:1/2, 573:1/2, 607:1/2, 625:1/2, 645:1/2 |
| `exf_check_range` | exf_check_range.F:3 | 1 | 28/76 | 57-59, 66-68, 75-77, 84-86, 94-96, 103-105, 114-116, 125-128, 136-138, 146-148, 156-159, 169-171, 181-191, 213-216 | 36:1/2, 39:1/2, 54:1/2, 63:1/2, 72:1/2, 81:1/2, 89:1/2, 91:1/2, 100:1/2, 111:1/2, 122:1/2, 133:1/2, 143:1/2, 153:1/2, 166:1/2, 178:1/2, 211:1/2 |
| `exf_diagnostics_fill` | exf_diagnostics_fill.F:6 | 10 | 2/2 | - | - |
| `exf_filter_rl` | exf_filter_rl.F:3 | 10 | 11/22 | 62-78 | 41:1/2, 58:1/2, 61:1/2 |
| `exf_fld_summary` | exf_summary.F:840 | 5 | 25/26 | 905 | 888:1/2, 900:1/2, 902:1/2, 920:1/2 |
| `exf_getclim` | exf_getclim.F:9 | 10 | 13/14 | 89 | 68:1/2, 88:1/2 |
| `exf_getffield_start` | exf_getffield_start.F:8 | 5 | 12/55 | 65-76, 86-92, 106-141 | 80:1/2, 85:1/2, 148:1/2 |
| `exf_getffieldrec` | exf_getffieldrec.F:6 | 50 | 18/97 | 101-109, 121-132, 139, 155-190, 206-259 | 97:1/2, 112:1/2, 119:1/2, 138:1/2, 196:1/2, 269:1/2 |
| `exf_getffields` | exf_getffields.F:12 | 10 | 58/73 | 97, 147-152, 315-320, 507, 512, 525 | 78:1/2, 126:1/2, 314:1/2, 486:1/2, 505:1/2, 510:1/2, 523:1/2 |
| `exf_getforcing` | exf_getforcing.F:95 | 10 | 45/53 | 175-181, 202-203, 329, 388 | 164:1/2, 171:1/2, 174:1/2, 200:1/2, 328:1/2, 387:1/2 |
| `exf_getsurfacefluxes` | exf_getsurfacefluxes.F:12 | 10 | 13/16 | 114, 117, 128 | 105:1/2, 112:1/2, 115:1/2, 126:1/2, 129:1/2 |
| `exf_getyearlyfieldname` | exf_getyearlyfieldname.F:7 | 10 | 4/10 | 48-54 | 46:1/2 |
| `exf_init_fixed` | exf_init_fixed.F:6 | 1 | 81/126 | 64-65, 149-155, 159-179, 185-191, 195-201, 207-213, 230-236, 262-268, 272-278, 282-289, 309-315, 322-329, 359-366, 474-481, 488-495, 590-593 | 44:1/2, 47:1/2, 63:1/2, 85:1/2, 124:1/2, 125:1/2, 127:1/2, 135:1/2, 137:1/2, 147:1/2, 158:1/2, 183:1/2, 193:1/2, 205:1/2, 218:1/2, 220:1/2, 228:1/2, 260:1/2, 270:1/2, 280:1/2, 307:1/2, 320:1/2, 334:1/2, 336:1/2, 344:1/2, 346:1/2, 357:1/2, 472:1/2, 486:1/2, 588:1/2, 610:1/2, 624:1/2 |
| `exf_init_fld` | exf_init_fld.F:8 | 18 | 11/26 | 99-139 | 98:1/2 |
| `exf_init_varia` | exf_init_varia.F:8 | 1 | 40/48 | 79-90, 134-139 | 44:1/2, 64:1/2, 105:1/2 |
| `exf_mapfields` | exf_mapfields.F:10 | 10 | 64/90 | 144-193, 220, 230, 235-237, 258, 268, 273-275 | 90:1/2, 105:1/2, 122:1/2, 134:1/2, 219:1/2, 229:1/2, 234:1/2, 257:1/2, 267:1/2, 272:1/2 |
| `exf_monitor` | exf_monitor.F:11 | 10 | 64/75 | 74, 114-116, 148, 160, 164, 186, 192, 204, 222, 228 | 64:1/2, 67:1/2, 71:1/2, 93:1/2, 112:1/2, 123:1/2, 127:1/2, 131:1/2, 137:1/2, 142:1/2, 146:1/2, 150:1/2, 154:1/2, 158:1/2, 162:1/2, 168:1/2, 174:1/2, 178:1/2, 184:1/2, 190:1/2, 202:1/2, 220:1/2, 226:1/2, 242:1/2, 246:1/2 |
| `exf_radiation` | exf_radiation.F:7 | 10 | 16/17 | 57 | 55:1/2, 60:1/2, 107:1/2 |
| `exf_readparms` | exf_readparms.F:3 | 1 | 424/442 | 285-290, 1038, 1068-1074, 1082-1088 | 283:1/2, 293:1/2, 1037:1/2, 1042:1/2, 1043:1/2, 1047:1/2, 1067:1/2, 1081:1/2 |
| `exf_set_fld` | exf_set_fld.F:10 | 180 | 27/65 | 124-129, 140, 152, 155-160, 172-180, 191-201, 251-261 | 123:1/2, 133:1/2, 142:1/2, 154:1/2, 171:1/2, 190:1/2, 250:1/2 |
| `exf_set_uv` | exf_set_uv.F:7 | 10 | 6/18 | 566-585 | 565:1/2 |
| `exf_summary` | exf_summary.F:14 | 1 | 163/205 | 261-263, 301-307, 314-324, 331-337, 344-350, 358-364, 402-408, 429-435, 487-493, 500-501, 535-541, 548-555, 562-563, 571-581, 585-586, 625-626, 636-642, 684-690, 761-767, 795-801 | 68:1/2, 254:1/2, 298:1/2, 311:1/2, 328:1/2, 341:1/2, 355:1/2, 369:1/2, 382:1/2, 399:1/2, 413:1/2, 426:1/2, 439:1/2, 484:1/2, 498:1/2, 532:1/2, 545:1/2, 560:1/2, 570:1/2, 583:1/2, 623:1/2, 633:1/2, 653:1/2, 666:1/2, 681:1/2, 758:1/2, 792:1/2 |
| `exf_swapffields` | exf_swapffields.F:12 | 5 | 8/8 | - | - |
| `exf_wind` | exf_wind.F:9 | 10 | 37/80 | 104-109, 138-140, 158-240 | 79:1/2, 102:1/2, 126:1/2, 133:1/2, 144:1/2, 252:1/2 |

**pkg/generic_advdiff** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 460 | 108/175 | 181-184, 213-216, 236-244, 257, 262, 276, 279, 282, 287-296, 329, 340, 386, 391, 405, 408, 411, 416-425, 458, 469, 517-545, 550, 555, 559, 561, 563, 568-577, 614, 620, 684-685, 793 | 176:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 256:1/2, 259:1/2, 273:1/2, 277:1/2, 280:1/2, 283:1/2, 328:1/2, 339:1/2, 384:1/2, 385:1/2, 388:1/2, 402:1/2, 406:1/2, 409:1/2, 412:1/2, 457:1/2, 468:1/2, 515:1/2, 549:1/2, 552:1/2, 556:1/2, 560:1/2, 562:1/2, 564:1/2, 605:1/2, 617:1/2, 669:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/76 | 70-76, 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 69:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_dst3_adv_r` | gad_dst3_adv_r.F:7 | 440 | 15/15 | - | - |
| `gad_dst3_adv_x` | gad_dst3_adv_x.F:7 | 460 | 17/17 | - | 83:1/2 |
| `gad_dst3_adv_y` | gad_dst3_adv_y.F:7 | 460 | 17/17 | - | 82:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 95/123 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 2/2 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 3/3 | - | - |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 47/71 | 96, 121-130, 201-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 113:1/2, 114:1/2, 115:1/2, 116:1/2, 117:1/2, 186:1/2, 189:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 53/118 | 116-126, 134, 163, 192-202, 276-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 162:1/2, 172:1/2, 183:1/2, 184:1/2, 185:1/2, 186:1/2, 187:1/2, 261:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `bldepth` | kpp_routines.F:309 | 10 | 76/91 | 505-511, 523-524, 700-706, 716-717, 836-842, 853-854 | 464:1/2, 486:1/2, 491:1/2, 666:1/2, 685:1/2, 690:1/2, 781:1/2, 824:1/2, 829:1/2 |
| `blmix` | kpp_routines.F:1395 | 10 | 72/72 | - | - |
| `enhance` | kpp_routines.F:1696 | 10 | 11/11 | - | 1741:1/2 |
| `kpp_calc` | kpp_calc.F:19 | 10 | 66/71 | 537, 655-663 | 223:1/2, 528:1/2, 636:1/2, 641:1/2 |
| `kpp_calc_diff_s` | kpp_calc_diff_s.F:3 | 10 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_diff_t` | kpp_calc_diff_t.F:3 | 10 | 8/12 | 56-59 | 45:1/2 |
| `kpp_calc_visc` | kpp_calc_visc.F:3 | 230 | 8/8 | - | - |
| `kpp_check` | kpp_check.F:3 | 1 | 53/62 | 130-132, 137-139, 142-144 | 28:1/2, 128:1/2, 136:1/2, 141:1/2 |
| `kpp_do_exch` | kpp_do_exch.F:6 | 10 | 3/3 | - | - |
| `kpp_forcing_surf` | kpp_forcing_surf.F:10 | 10 | 39/40 | 204 | 203:1/2, 233:1/2 |
| `kpp_init_fixed` | kpp_init_fixed.F:6 | 1 | 33/33 | - | 116:1/2, 162:1/2 |
| `kpp_init_varia` | kpp_init_varia.F:7 | 1 | 17/17 | - | - |
| `kpp_output` | kpp_output.F:11 | 11 | 7/16 | 118-157 | 105:1/2, 114:1/2 |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 73/112 | 61-66, 171-176, 196-199, 202-205, 208-211, 214-217, 220-226, 230-238 | 59:1/2, 69:1/2, 169:1/2, 195:1/2, 201:1/2, 207:1/2, 213:1/2, 219:1/2, 229:1/2 |
| `kpp_transport_s` | kpp_transport_s.F:9 | 220 | 5/5 | - | - |
| `kpp_transport_t` | kpp_transport_t.F:6 | 220 | 6/6 | - | - |
| `kppmix` | kpp_routines.F:28 | 10 | 17/17 | - | - |
| `ri_iwmix` | kpp_routines.F:1032 | 10 | 37/42 | 1120-1121, 1199-1201 | 1119:1/2, 1198:1/2 |
| `smooth_horiz` | kpp_routines.F:1311 | 220 | 14/15 | 1376 | 1363:1/2 |
| `statekpp` | kpp_routines.F:1766 | 10 | 29/29 | - | - |
| `wscale` | kpp_routines.F:923 | 470 | 28/28 | - | - |

**pkg/mdsio** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 419 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 92 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 673 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 69 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 115 | 67/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 249-251, 327-330, 342, 348-358, 377-565 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 233:1/2, 247:1/2, 262:1/2, 265:1/2, 291:1/2, 294:1/4, 299:1/4, 322:1/2, 332:1/2, 340:1/2, 343:1/2, 364:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 25/58 | 112, 144-208 | 113:1/2, 116:1/2, 118:1/2, 133:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 171 | 73/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 297-300, 318-321, 332-335, 370-561 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 272:1/2, 292:1/2, 309:1/2, 313:1/2, 341:1/2, 347:1/4, 352:1/4, 360:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 164 | 52/64 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (7)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hdiv` | mom_calc_hdiv.F:3 | 230 | 10/14 | 39-48, 74 | 38:1/2, 55:1/2 |
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 230 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 230 | 10/26 | 62-66, 88-137 | 61:1/2, 70:1/2 |
| `mom_calc_relvort3` | mom_calc_relvort3.F:4 | 230 | 9/40 | 93-282 | 80:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 37/39 | 51-52 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 230 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 230 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |

**pkg/mom_vecinv** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_vecinv` | mom_vecinv.F:10 | 230 | 132/227 | 235, 246, 332-340, 381, 403-406, 426, 442-462, 475-478, 503-506, 554-572, 585-588, 613-616, 670, 686-689, 709-732, 749, 754, 758, 774, 779, 783, 801-803, 860-862, 879-890, 902-907, 932, 937-956 | 234:1/2, 240:1/2, 305:1/2, 331:1/2, 373:1/2, 400:1/2, 419:1/2, 441:1/2, 468:1/2, 484:1/2, 502:1/2, 553:1/2, 578:1/2, 594:1/2, 612:1/2, 669:1/2, 680:1/2, 683:1/2, 708:1/2, 742:1/2, 745:1/2, 750:1/2, 755:1/2, 770:1/2, 775:1/2, 780:1/2, 800:1/2, 823:1/2, 859:1/2, 878:1/2, 900:1/2, 930:1/2, 936:1/2 |
| `mom_vi_coriolis` | mom_vi_coriolis.F:7 | 230 | 13/47 | 73-122, 140-189 | 58:1/2, 125:1/2 |
| `mom_vi_hdissip` | mom_vi_hdissip.F:3 | 230 | 17/70 | 50-69, 105-108, 116-226 | 45:1/2, 49:1/2, 114:1/2 |
| `mom_vi_u_coriolis` | mom_vi_u_coriolis.F:6 | 230 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_u_grad_ke` | mom_vi_u_grad_ke.F:3 | 230 | 5/5 | - | - |
| `mom_vi_u_vertshear` | mom_vi_u_vertshear.F:7 | 230 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |
| `mom_vi_v_coriolis` | mom_vi_v_coriolis.F:6 | 230 | 13/47 | 60-71, 92-175, 180-187 | 57:1/2, 75:1/2, 179:1/2 |
| `mom_vi_v_grad_ke` | mom_vi_v_grad_ke.F:3 | 230 | 5/5 | - | - |
| `mom_vi_v_vertshear` | mom_vi_v_vertshear.F:7 | 230 | 20/24 | 47, 88-94, 131 | 46:1/2, 69:1/2, 127:1/2 |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 22 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 11 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 11 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 10 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 10 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 262 | 71/71 | - | 75:1/2, 88:1/2, 93:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2, 140:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 55 | 71/71 | - | 70:1/2, 75:1/2, 88:1/2, 93:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2, 140:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 11 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 1942 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 32 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 1910 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 122 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 11 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 55:1/2, 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 11 | 39/50 | 95, 123-132, 178-179, 196-198 | 83:1/2, 93:1/2, 122:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 11 | 72/127 | 145-237, 243-278 | 143:1/2, 242:1/2, 286:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 262 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 55 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 11 | 69/72 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 123:1/2, 126:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (17)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xy_rl` | read_fld_xy_rl.F:3 | 1 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 30 | 7/7 | - | - |
| `read_rec_lev_rl` | read_rec.F:445 | 4 | 7/7 | - | - |
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
| `write_rec_3d_rl` | write_rec.F:399 | 73 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 1 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 2 | 7/7 | - | - |

**pkg/seaice** (25)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `advect` | advect.F:9 | 30 | 30/63 | 105-125, 164-226 | 104:1/2, 160:1/2 |
| `seaice_advdiff` | seaice_advdiff.F:10 | 10 | 19/74 | 138-401, 586-593, 605-612, 624-631, 642-652 | 135:1/2, 579:1/2, 584:1/2, 598:1/2, 603:1/2, 617:1/2, 622:1/2, 638:1/2 |
| `seaice_budget_ocean` | seaice_budget_ocean.F:7 | 10 | 6/6 | - | - |
| `seaice_check` | seaice_check.F:12 | 1 | 86/448 | 67, 84-87, 97-100, 104-107, 113-119, 126, 129-135, 140-146, 152-155, 160-166, 171-178, 181-188, 191-199, 202-210, 231-237, 243-249, 253-259, 371-404, 434-468, 478-484, 488-491, 496-502, 509-516, 519-526, 529-535, 540-542, 557-560, 567-569, 700-702, 725-730, 733-741, 744-752, 769-772, 777-780, 789-794, 935-940, 947-952, 960-966, 972-975, 980-983, 988-991, 996-999, 1002-1005, 1016-1022, 1063-1065, 1069-1071, 1119-1124, 1130-1135, 1140-1150, 1154-1157, 1162-1167, 1173-1177, 1182-1185, 1189-1200, 1216-1221, 1228-1231, 1234-1239, 1242-1247, 1301-1309 | 66:1/2, 73:1/2, 83:1/2, 92:1/2, 95:1/2, 102:1/2, 111:1/2, 125:1/2, 127:1/2, 138:1/2, 149:1/2, 158:1/2, 170:1/2, 180:1/2, 190:1/2, 201:1/2, 230:1/2, 242:1/2, 252:1/2, 370:1/2, 406:1/2, 433:1/2, 472:1/2, 487:1/2, 495:1/2, 507:1/2, 508:1/2, 518:1/2, 528:1/2, 539:1/2, 555:1/2, 565:1/2, 691:1/2, 698:1/2, 723:1/2, 732:1/2, 743:1/2, 768:1/2, 776:1/2, 788:1/2, 933:1/2, 945:1/2, 957:1/2, 971:1/2, 979:1/2, 987:1/2, 995:1/2, 1001:1/2, 1010:1/2, 1011:1/2, 1012:1/2, 1013:1/2, 1014:1/2, 1015:1/2, 1061:1/2, 1067:1/2, 1117:1/2, 1128:1/2, 1139:1/2, 1153:1/2, 1160:1/2, 1170:1/2, 1181:1/2, 1188:1/2, 1214:1/2, 1226:1/2, 1233:1/2, 1241:1/2, 1300:1/2 |
| `seaice_cost_accumulate_mean` | seaice_cost_accumulate_mean.F:8 | 10 | 2/2 | - | - |
| `seaice_cost_final` | seaice_cost_final.F:12 | 1 | 15/15 | - | 83:1/2 |
| `seaice_cost_init_fixed` | seaice_cost_init_fixed.F:3 | 1 | 14/16 | 53-54 | 40:1/2, 42:1/2, 51:1/2 |
| `seaice_cost_init_varia` | seaice_cost_init_varia.F:3 | 1 | 7/7 | - | - |
| `seaice_cost_sensi` | seaice_cost_sensi.F:3 | 10 | 4/4 | - | - |
| `seaice_cost_test` | seaice_cost_test.F:3 | 10 | 13/52 | 75, 90-95, 135-221 | 74:1/2, 79:1/2, 88:1/2, 101:1/2 |
| `seaice_dynsolver` | seaice_dynsolver.F:9 | 10 | 11/11 | - | 376:1/2 |
| `seaice_get_dynforcing` | seaice_get_dynforcing.F:9 | 10 | 26/38 | 159-164, 173, 221-232 | 98:1/2, 146:1/2, 157:1/2, 172:1/2 |
| `seaice_growth` | seaice_growth.F:15 | 10 | 245/276 | 336-337, 344, 749-759, 1033, 1042, 1464-1471, 1490-1491, 1808, 1825, 1837-1848, 1859, 2099, 2348, 2352, 2355 | 335:1/2, 343:1/2, 747:1/2, 790:1/2, 1030:1/2, 1040:1/2, 1299:1/2, 1462:1/2, 1482:1/2, 1698:1/2, 1807:1/2, 1822:1/2, 1833:1/2, 1852:1/2, 2065:1/2, 2346:1/2, 2350:1/2, 2353:1/2, 2356:1/2, 2424:1/2 |
| `seaice_init_fixed` | seaice_init_fixed.F:7 | 1 | 18/25 | 60, 67, 80, 82-89 | 59:1/2, 66:1/2, 71:1/2, 75:1/2, 79:1/2, 81:1/2 |
| `seaice_init_varia` | seaice_init_varia.F:7 | 1 | 72/97 | 54, 237, 254, 272, 274, 288, 293-299, 318-327, 392-393 | 53:1/2, 165:1/2, 235:1/2, 251:1/2, 271:1/2, 273:1/2, 275:1/2, 292:1/2, 310:1/2, 317:1/2, 345:1/2, 391:1/2, 447:1/2 |
| `seaice_model` | seaice_model.F:13 | 10 | 37/49 | 92, 130-133, 254-257, 281-286 | 82:1/2, 85:1/2, 125:1/2, 128:1/2, 178:1/2, 222:1/2, 224:1/2, 252:1/2, 267:1/2, 275:1/2, 340:1/2, 411:1/2 |
| `seaice_monitor` | seaice_monitor.F:8 | 11 | 36/37 | 65 | 55:1/2, 58:1/2, 62:1/2, 84:1/2, 115:1/2, 136:1/2, 140:1/2 |
| `seaice_ocean_stress` | seaice_ocean_stress.F:6 | 10 | 18/29 | 56, 69-90 | 55:1/2, 64:1/2 |
| `seaice_output` | seaice_output.F:10 | 11 | 24/25 | 112 | 57:1/2, 108:1/2, 109:1/2, 127:1/2, 174:1/2 |
| `seaice_readparms` | seaice_readparms.F:9 | 1 | 454/788 | 233-238, 460-468, 752-757, 846-856, 866, 868, 912-913, 921-932, 936-942, 951-954, 957-960, 965-971, 975-986, 992, 997, 1000, 1006-1012, 1017-1025, 1032-1036, 1044, 1078-1085, 1088-1095, 1098-1105, 1108-1115, 1118-1125, 1128-1136, 1139-1147, 1150-1158, 1161-1166, 1169-1175, 1178-1184, 1187-1193, 1196-1204, 1207-1215, 1218-1227, 1230-1238, 1241-1249, 1252-1258, 1261-1265, 1268-1276, 1279-1287, 1290-1298, 1301-1309, 1315-1323, 1326-1331, 1334-1338, 1341-1345, 1348-1352, 1364-1368, 1371-1375, 1378-1382, 1386-1392, 1395-1401, 1405-1410, 1414-1419, 1423-1424, 1457-1468, 1475-1479, 1482-1486, 1489-1497, 1539-1543 | 231:1/2, 241:1/2, 445:1/2, 711:1/2, 713:1/2, 718:1/2, 721:1/2, 724:1/2, 725:1/2, 727:1/2, 729:1/2, 731:1/2, 733:1/2, 735:1/2, 737:1/2, 740:1/2, 748:1/2, 845:1/2, 865:1/2, 867:1/2, 871:1/2, 874:1/2, 875:1/2, 878:1/2, 882:1/2, 885:1/2, 896:1/2, 901:1/2, 906:1/2, 907:1/2, 911:1/2, 916:1/2, 917:1/2, 934:1/2, 945:1/2, 950:1/2, 956:1/2, 964:1/2, 974:1/2, 990:1/2, 995:1/2, 999:1/2, 1005:1/2, 1015:1/2, 1028:1/2, 1039:1/2, 1041:1/2, 1043:1/2, 1045:1/2, 1047:1/2, 1049:1/2, 1052:1/2, 1054:1/2, 1056:1/2, 1058:1/2, 1060:1/2, 1062:1/2, 1069:1/2, 1077:1/2, 1087:1/2, 1097:1/2, 1107:1/2, 1117:1/2, 1127:1/2, 1138:1/2, 1149:1/2, 1160:1/2, 1168:1/2, 1177:1/2, 1186:1/2, 1195:1/2, 1206:1/2, 1217:1/2, 1229:1/2, 1240:1/2, 1251:1/2, 1260:1/2, 1267:1/2, 1278:1/2, 1289:1/2, 1300:1/2, 1313:1/2, 1325:1/2, 1333:1/2, 1340:1/2, 1347:1/2, 1362:1/2, 1370:1/2, 1377:1/2, 1385:1/2, 1394:1/2, 1404:1/2, 1413:1/2, 1422:1/2, 1433:1/2, 1434:1/2, 1455:1/2, 1474:1/2, 1481:1/2, 1488:1/2, 1511:1/2, 1514:1/2, 1517:1/2, 1524:1/2, 1531:1/2, 1538:1/2, 1545:1/2 |
| `seaice_reg_ridge` | seaice_reg_ridge.F:12 | 10 | 48/48 | - | - |
| `seaice_solve4temp` | seaice_solve4temp.F:12 | 10 | 121/148 | 191, 262-267, 289-294, 308, 315, 322, 387-388, 449-451, 514, 544-561 | 190:1/2, 217:1/2, 260:1/2, 288:1/2, 307:1/2, 309:1/2, 321:1/2, 385:1/2, 445:1/2, 502:1/2, 513:1/2, 539:1/2 |
| `seaice_summary` | seaice_summary.F:5 | 1 | 187/244 | 119-251, 357-359, 390, 400, 480-482 | 45:1/2, 108:1/2, 355:1/2, 375:1/2, 378:1/2, 381:1/2, 384:1/2, 388:1/2, 398:1/2, 479:1/2 |
| `seaice_turnoff_io` | seaice_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `seaice_write_pickup` | seaice_write_pickup.F:6 | 1 | 44/62 | 102-106, 160-168, 194-200 | 78:1/2, 101:1/2, 110:1/2, 117:1/2, 121:1/2, 125:1/2, 133:1/2, 152:1/2, 157:1/2, 159:1/2, 193:1/2 |

### Compiled but not executed (765 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_r_star, calc_surf_dr, calc_wsurf_tr, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhoteos, forcing_surf_relax, freesurf_rescale_g, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, packages_error_msg, packages_unused_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cal: cal_daysformonth, cal_dayspermonth, cal_getmonthsrec, cal_monthsforyear, cal_monthsperyear, cal_printdate, cal_printerror, cal_stepsforday, cal_stepsperday, cal_timestamp, cal_weekday
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_tracer, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_map_genarr2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/ecco: cost_bp_read, cost_gencost_seaicev4, cost_gencost_sshv4, cost_gencost_sstv4, cost_gencost_transp, cost_sla_read, cost_sla_read_yd, ecco_add, ecco_addmask, ecco_cprsrl, ecco_diagnostics_init, ecco_div, ecco_error, ecco_maskmindepth, ecco_multfield, ecco_offset, ecco_read_pickup, ecco_subtract, get_exconc_deconc
- pkg/exf: adexf_adjoint_snapshots, adexf_monitor, exf_check_interp, exf_diagnostics_init, exf_getfield_start, exf_getmonthsrec, exf_init_gen, exf_init_interp, exf_interp, exf_interp_read, exf_interp_uv, exf_interpolate, exf_set_gen, exf_set_obcs_x, exf_set_obcs_xz, exf_set_obcs_y, exf_set_obcs_yz, exf_swapffields_3d, exf_swapffields_xz, exf_swapffields_yz, exf_weight_sfx_diags, exf_zenithangle, exf_zenithangle_table, g_exf_adjoint_snapshots, lagran
- pkg/generic_advdiff: gad_advection, gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: kpp_calc_diff_ptr, kpp_calc_dummy, kpp_diagnostics_init, kpp_doublediff, kpp_transport_ptr, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_rviscflux, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_rviscflux, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_calc_rtrans, mom_fluxform, mom_u_adv_uu, mom_u_adv_vu, mom_u_adv_wu, mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_u_metric_sphere, mom_u_xviscflux, mom_u_yviscflux, mom_uv_boundary, mom_v_adv_uv, mom_v_adv_vv, mom_v_adv_wv, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder, mom_v_metric_sphere, mom_v_xviscflux, mom_v_yviscflux
- pkg/mom_vecinv: mom_vi_del2uv, mom_vi_u_coriolis_c4, mom_vi_v_coriolis_c4
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rs, read_fld_xyz_rl, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs
- pkg/seaice: adseaice_monitor, diffus, dynsolver, lsr, ostres, seaice_ad_dump, seaice_advection, seaice_bottomdrag_coeffs, seaice_calc_ice_strength, seaice_calc_lhs, seaice_calc_residual, seaice_calc_rhs, seaice_calc_strainrates, seaice_calc_stressdiv, seaice_calc_viscosities, seaice_check_pickup, seaice_cost_export, seaice_diag_sufx, seaice_diagnostics_init, seaice_diagnostics_state, seaice_diffusion, seaice_do_ridging, seaice_evp, seaice_fake, seaice_fgmres, seaice_freedrift, seaice_growth_adx, seaice_itd_pickup, seaice_itd_redist, seaice_itd_remap, seaice_itd_sum, seaice_jacvec, seaice_jfnk, seaice_krylov, seaice_lsr, seaice_map2vec, seaice_map_rs2vec, seaice_map_thsice, seaice_mnc_init, seaice_mom_advection, seaice_obcs_output, seaice_oceandrag_coeffs, seaice_preconditioner, seaice_prepare_ridging, seaice_read_pickup, seaice_scalprod, seaice_sidedrag_stress, seaice_tracer_phys

