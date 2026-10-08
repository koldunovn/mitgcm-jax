# Coverage: tutorial_tracer_adjsens (code_ad) — M2 porting worklist

Written by `tools/coverage.py` (mitjax d117407) from the gcov build `tutorial_tracer_adjsens-code_ad-63cdc0b-8316e95-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 682 files from the link farm, 105 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_ad` | `job27832235` | 5 | 0 / 0 | 361 | yes | identical (506 lines) |
| `input_ad.som81` | `job27832235` | 5 | 0 / 0 | 371 | yes | identical (506 lines) |

## tutorial_tracer_adjsens/input_ad

Run `$MJX_REFERENCE/coverage/tutorial_tracer_adjsens/input_ad/job27832235` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/tutorial_tracer_adjsens/input_ad/job27832235/gcov-8316e95/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| theta | 5 blocks, first 4.251363e+00, last 4.250485e+00, min 4.250485e+00, max 4.251363e+00 | yes |
| passive tracer step | ptracers_integrate (16 calls) | yes |
| GM/Redi tensor | gmredi_calc_tensor (16 calls) | yes |
| cost function | global fc = 184102723380947.0; terms: objf_tracer sum 1.841027e+14 | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: AdamsBashforthGs, AdamsBashforthGt, calc_wVelocity, doAB_onGtGs, doResetHFactors, doSaltClimRelax, doThetaClimRelax, dumpInitAndLast, fluidIsWater, GM_ExtraDiag, inAdExact, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, PTRACERS_AdamsBashGtr, PTRACERS_doAB_onGpTr, PTRACERS_startAllTrc, PTRACERS_useGMRedi, saltAdvection, saltForcing, saltIsActiveTr, saltStepping, snapshot_mdsio, tempAdvection, tempForcing, tempIsActiveTr, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useGMRediInAdMode, useHarmonicVisc, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, fluidIsAir, globalFiles, GM_AdvSeparate, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momImplVertAdv, nonHydrostatic, printMapIncludesZeros, PTRACERS_AdamsBash_Tr, PTRACERS_addSrelax2EmP, PTRACERS_ImplVertAdv, PTRACERS_MultiDimAdv, PTRACERS_pickup_read_mnc, PTRACERS_pickup_write_mnc, PTRACERS_snapshot_mnc, PTRACERS_SOM_Advection, PTRACERS_useDWNSLP, PTRACERS_useKPP, PTRACERS_useRecords, quasiHydrostatic, rotateGrid, saltImplVertAdv, saltMultiDimAdvec, saltSOM_Advection, staggerTimeStep, tempImplVertAdv, tempMultiDimAdvec, tempSOM_Advection, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useCtrlCostContribution, useGGL90inAdMode, useKPPinAdMode, useMin4hFacEdges, useMultiDimAdvec, useNest2W_child, useNest2W_parent, useNHMTerms, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSEAICEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, vectorInvariantMomentum
- selectors: pCellMix_select=0, saltAdvScheme=2, saltVertAdvScheme=2, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempAdvScheme=2, tempVertAdvScheme=2

### Executed routines (361)

**eesupp/src** (74)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 6182 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 6633 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 64 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1582 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 29 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 4 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1611 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1582 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1582 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1582 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1582 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 29 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 29 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 29 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 29 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_uv_dgrid_3d_rl` | exch_uv_dgrid_3d_rl.F:8 | 4 | 14/32 | 88-177 | 72:1/2, 74:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 6 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 5 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 4 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 1534 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 17 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 16 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 26081 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 210 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 64 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 26 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 2706 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 2108 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 8765 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 18 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 612 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 369 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 416 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 6444 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 144 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 9 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 40 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 117 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 164 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 2611 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 200 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 200 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 200 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 101 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 99 | 4/4 | - | - |
| `ucase` | utils.F:311 | 401 | 7/8 | 333 | 332:1/2 |
| `write_0d_c` | write_utils.F:579 | 7 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 32 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 117 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 112 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 47 | 32/32 | - | - |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (106)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 1600 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 16/51 | 55-71, 87-93, 113-133 | 52:1/2, 86:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 320 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 320 | 18/50 | 467, 469, 471, 563-604, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 320 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 320 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 80 | 28/41 | 164-166, 175-197 | 83:1/2, 146:1/2, 173:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 960 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 320 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 320 | 22/81 | 68-80, 87-119, 162-261 | 63:1/2, 84:1/2, 158:1/2 |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 16 | 8/8 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 16 | 6/77 | 86-243 | 74:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 320 | 37/155 | 149, 184, 194-197, 212-240, 268-610 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 193:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_r_star` | calc_r_star.F:10 | 6 | 66/119 | 68, 143-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 111:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 16 | 11/16 | 81, 124-127 | 77:1/2, 121:1/2 |
| `cg2d_nsa` | cg2d_nsa.F:15 | 4 | 89/102 | 120-122, 125-130, 222, 358-362, 392 | 118:1/2, 124:1/2, 219:1/2, 226:1/2, 227:1/2, 288:1/2, 332:1/2, 357:1/2, 389:1/2 |
| `config_check` | config_check.F:14 | 1 | 93/525 | 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 344-349, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 986-989, 992-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 342:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 985:1/2, 991:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 439/537 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 268-281, 307-322, 533-538, 554-588, 756-762, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `convective_adjustment` | convective_adjustment.F:10 | 16 | 30/34 | 103-106 | 67:1/2, 95:1/2, 110:4/8, 162:1/2 |
| `convective_adjustment_ini` | convective_adjustment_ini.F:10 | 4 | 29/33 | 112-115 | 104:1/2, 119:4/8, 171:1/2 |
| `convective_weights` | convective_weights.F:6 | 380 | 15/15 | - | - |
| `convectively_mixtracer` | convectively_mixtracer.F:6 | 1140 | 7/7 | - | - |
| `correction_step` | correction_step.F:7 | 16 | 18/25 | 158-167, 180-188 | 156:1/2, 179:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 48 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 320 | 7/22 | 76-114 | 73:1/2 |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 320 | 25/48 | 79-84, 123-125, 134-170 | 61:1/2, 76:1/2, 121:1/2, 132:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 4 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 4 | 16/18 | 79, 86 | 49:1/2, 52:1/2, 53:1/2, 55:1/2, 60:1/2, 78:1/2, 85:1/2, 96:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 4 | 87/109 | 557, 771-784, 797-798, 830-833, 869-874, 953-958, 1043, 1103 | 248:1/2, 251:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 867:1/2, 896:1/2, 951:1/2, 1030:1/2, 1032:1/2, 1096:1/2, 1102:1/2, 1132:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 5 | 10/17 | 97-110, 145 | 96:1/2, 116:1/2, 144:1/2, 212:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 4 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 4 | 55/77 | 337-340, 358, 533, 591-599, 623-631, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 615:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 4 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 4 | 42/71 | 75, 151-153, 161-163, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 188:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 1384 | 14/60 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 58 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 4 | 11/23 | 64, 116-151 | 63:1/2, 115:1/2 |
| `forward_step` | forward_step.F:70 | 4 | 94/120 | 483-485, 767-769, 842-853, 952-960, 979-1006, 1176, 1201-1202 | 424:1/2, 468:1/2, 500:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 730:1/2, 766:1/2, 789:1/2, 832:1/2, 866:1/2, 897:1/2, 924:1/2, 939:1/2, 976:1/2, 1151:1/2, 1175:1/2, 1187:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 1920 | 7/12 | 49-66 | 39:1/2, 40:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 320 | 20/22 | 61, 74 | 59:1/2, 72:1/2 |
| `impldiff` | impldiff.F:7 | 48 | 64/66 | 101-102 | 93:1/2, 199:1/2, 219:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 81/87 | 120, 151, 161, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 152:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 24/65 | 57-63, 70-79, 106-112, 121-180, 191 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 28/28 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 61/87 | 54, 60, 78, 80, 83-90, 98, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 103/115 | 131, 134-144, 192 | 130:1/2, 132:1/2, 155:1/2, 161:1/2, 163:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 183/198 | 210-212, 247-261, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 9/11 | 52-53 | 51:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 28/52 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 74/74 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 406/889 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1109-1112, 1123-1124, 1127, 1131-1133, 1139, 1148, 1151, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1552-1555, 1558-1561, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1116:1/2, 1117:1/2, 1118:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1551:1/2, 1557:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 20/22 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 25/38 | 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_spherical_polar_grid` | ini_spherical_polar_grid.F:8 | 1 | 75/85 | 258-263, 277-280 | 257:1/2, 276:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 26/47 | 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 42/47 | 309-321, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 282:1/2, 284:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 5 | 56/70 | 146-160, 212-214, 279-282 | 90:1/2, 93:1/2, 95:1/2, 142:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 400 | 18/36 | 95-117, 177-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 4 | 20/27 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 229:1/2, 231:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 27/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161, 173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 4 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 4 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 98/98 | - | 100:1/2, 232:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 66/109 | 132-140, 207, 212, 265, 280, 289, 321, 342, 349, 356, 369, 376, 403, 412, 437, 460, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 252:1/2, 260:1/2, 264:1/2, 269:1/2, 275:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 301:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 335:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 459:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 24/26 | 319-321 | 144:1/2, 193:1/2, 200:1/2, 202:1/2, 209:1/2, 211:1/2, 317:1/2, 327:1/2, 329:1/2, 358:1/2, 425:1/2, 427:1/2, 651:1/2, 655:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 17/21 | 169, 253-255, 637 | 168:1/2, 192:1/2, 194:1/2, 204:1/2, 206:1/2, 251:1/2, 270:1/2, 332:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 17 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 10/10 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 1 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 11/11 | - | 103:1/2, 124:1/2, 155:1/2, 255:1/2 |
| `reset_nlfs_vars` | reset_nlfs_vars.F:6 | 4 | 8/11 | 48-50 | 47:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 16 | 54/63 | 192, 259-267, 273-280, 397-399, 517 | 147:1/2, 177:1/2, 257:1/2, 268:1/2, 319:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 413:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 97/165 | 56-57, 60-63, 67, 71, 76, 109-115, 120-126, 171-175, 186-189, 192-195, 202, 208, 214, 220, 226, 232, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 54/195 | 101-133, 140-165, 180-187, 212-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 4 | 58/85 | 143-147, 152-157, 164, 170-176, 228-234, 265, 269, 312, 319, 327, 342-344 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 298:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 16 | 54/63 | 194, 261-269, 275-282, 399-401, 519 | 149:1/2, 179:1/2, 259:1/2, 270:1/2, 321:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 36/36 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 19/27 | 736-797 | 606:1/2, 616:1/2, 707:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 4 | 52/74 | 158, 177, 218-250, 395-406 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 173:1/2, 206:1/2, 207:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 344:1/2, 346:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 320 | 60/83 | 210-213, 220-223, 286-312, 322-325, 329-332 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 209:1/2, 219:1/2, 229:1/2, 276:1/2, 277:1/2, 321:1/2, 328:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 48 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 4 | 6/6 | - | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 18/18 | - | 55:1/2, 118:1/2 |
| `update_cg2d` | update_cg2d.F:7 | 5 | 41/49 | 58, 132-140, 169, 175, 182 | 57:1/2, 61:1/2, 131:1/2, 159:1/2, 168:1/2, 174:1/2, 181:1/2 |
| `update_etah` | update_etah.F:7 | 5 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 9 | 29/29 | - | - |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 61/75 | 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 255:1/2, 256:1/2, 257:1/2, 260:1/2, 263:1/2, 264:1/2, 265:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 5 | 18/20 | 85, 126 | 81:1/2, 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (18)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 2 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xyz` | active_file.F:100 | 2 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 1 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 39-49 | 37:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 1 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 48:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 4 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 4 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 75/119 | 63-68, 127-132, 165-172, 175-176, 237-247, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 164:1/2, 174:1/2, 235:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 1 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 1 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 2 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 5 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 4 | 2/2 | - | - |

**pkg/cd_code** (4)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cd_code_ini_vars` | cd_code_ini_vars.F:3 | 1 | 15/16 | 53 | 52:1/2 |
| `cd_code_init_fixed` | cd_code_init_fixed.F:3 | 1 | 2/2 | - | - |
| `cd_code_scheme` | cd_code_scheme.F:7 | 320 | 47/49 | 79-80 | 78:1/2 |
| `cd_code_write_pickup` | cd_code_write_pickup.F:8 | 1 | 11/11 | - | 85:1/2 |

**pkg/cost** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 7/7 | - | 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 47/47 | - | 82:1/2, 102:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 20/20 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 4 | 5/5 | - | - |
| `cost_tracer` | cost_tracer.F:3 | 16 | 8/8 | - | - |

**pkg/ctrl** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 1 | 8/8 | - | - |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 1 | 3/13 | 43-54 | 41:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 31/52 | 139-143, 162-168, 376-381, 591-594 | 80:1/2, 136:1/2, 148:1/2, 151:1/2, 155:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/13 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/35 | 82-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 1 | 9/9 | - | - |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 1 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 1 | 22/50 | 80-87, 114-126, 143-171 | 79:1/2, 88:1/2, 113:1/2, 132:1/2, 134:1/2, 140:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 36/36 | - | 251:1/2, 380:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 11/11 | - | 81:1/2, 104:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 168:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 4 | 2/2 | - | - |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 1 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 4 | 2/2 | - | - |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 29/36 | 356, 358, 360, 362, 364, 393, 395 | 128:1/2, 354:1/2, 355:1/2, 357:1/2, 359:1/2, 361:1/2, 363:1/2, 379:1/2, 381:1/2, 384:1/2, 392:1/2, 394:1/2, 416:1/2, 434:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 197/235 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 492:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 1 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 1 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 73/83 | 153-155, 218-222, 304-306 | 144:1/2, 146:1/2, 209:1/2, 217:1/2, 302:1/2 |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/generic_advdiff** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advscheme_get` | gad_advscheme.F:116 | 7 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_c2_adv_r` | gad_c2_adv_r.F:7 | 912 | 7/10 | 52-54 | 51:1/2 |
| `gad_c2_adv_x` | gad_c2_adv_x.F:7 | 960 | 7/7 | - | - |
| `gad_c2_adv_y` | gad_c2_adv_y.F:7 | 960 | 7/7 | - | - |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 960 | 79/183 | 181-184, 213-216, 236-244, 259-296, 331-333, 340, 388-425, 460-462, 469, 517-545, 552-577, 614, 620, 659-690, 793 | 176:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 256:1/2, 328:1/2, 339:1/2, 345:1/2, 384:1/2, 385:1/2, 457:1/2, 468:1/2, 474:1/2, 515:1/2, 549:1/2, 605:1/2, 617:1/2, 625:1/2, 658:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 16/68 | 82-88, 99-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 81:1/2, 95:1/2, 97:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diff_x` | gad_diff_x.F:7 | 960 | 6/6 | - | - |
| `gad_diff_y` | gad_diff_y.F:7 | 960 | 7/7 | - | - |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 99/123 | 63-65, 90-93, 97-100, 104-107, 111-114, 159-162, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 11/14 | 60-69 | 58:1/2 |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 6/20 | 72-83, 89-100 | 70:1/2, 87:1/2 |

**pkg/gmredi** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 48 | 10/16 | 70-82 | 49:1/2 |
| `gmredi_calc_psi_bolus` | gmredi_calc_psi_bolus.F:16 | 16 | 32/33 | 127 | 87:1/2, 124:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 16 | 146/188 | 166-209, 292-294, 529, 703-706, 768, 898-901, 965 | 164:1/2, 291:1/2, 526:1/2, 610:1/2, 620:1/2, 702:1/2, 765:1/2, 815:1/2, 897:1/2, 962:1/2, 1012:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 55/178 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 506-509, 513-516, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 502:1/2, 505:1/2, 512:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 4 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 17/23 | 63-64, 67-68, 82, 86 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 4 | 3/12 | 50-61 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 106/146 | 99-104, 235-239, 245, 253-266, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 244:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 16 | 20/20 | - | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 960 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 944 | 54/87 | 176, 234, 397, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_slope_psi` | gmredi_slope_psi.F:9 | 304 | 48/100 | 100, 183, 273-286, 294-310, 330-418 | 95:1/2, 181:1/2, 218:1/2, 252:1/2, 270:1/2, 290:1/2, 314:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 960 | 22/35 | 97-100, 200-233 | 95:1/2, 104:1/2, 143:1/2, 196:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 960 | 22/35 | 97-100, 200-233 | 95:1/2, 104:1/2, 143:1/2, 196:1/2 |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 50/71 | 96, 121-130, 202-211, 219-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 115:1/2, 116:1/2, 189:1/2, 201:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 60/118 | 116-126, 134, 163, 173, 192-202, 278, 282, 298-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 172:1/2, 183:1/2, 185:1/2, 186:1/2, 280:1/2, 281:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `kpp_calc_dummy` | kpp_calc.F:726 | 16 | 11/11 | - | - |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 5/112 | 69-248 | 59:1/2, 61:1/2 |

**pkg/mdsio** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 45 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 25 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 23 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 10 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 487, 535-538, 543, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 220:1/2, 230:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 530:1/2, 540:1/2, 541:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 87 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 311 | 40/52 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (7)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 320 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 320 | 12/26 | 62-66, 77-84, 90-97, 115-137 | 61:1/2, 70:1/2, 88:1/2, 101:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 37/39 | 51-52 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 320 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_u_rviscflux` | mom_u_rviscflux.F:7 | 640 | 9/9 | - | - |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 320 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_rviscflux` | mom_v_rviscflux.F:7 | 640 | 9/9 | - | - |

**pkg/mom_fluxform** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_rtrans` | mom_calc_rtrans.F:6 | 336 | 33/37 | 72-79 | 69:1/2, 111:1/2 |
| `mom_fluxform` | mom_fluxform.F:42 | 320 | 167/248 | 258-259, 265, 276, 299-303, 331-350, 457, 514-523, 566-594, 607, 663-666, 690-693, 738-742, 764-767, 862-890, 902, 958-961, 985-988, 1033-1037, 1058-1061, 1092-1100, 1113-1124 | 257:1/2, 264:1/2, 270:1/2, 298:1/2, 330:1/2, 422:1/2, 444:1/2, 453:1/2, 476:1/2, 510:1/2, 512:1/2, 556:1/2, 565:1/2, 601:1/2, 605:1/2, 621:1/2, 656:1/2, 671:1/2, 689:1/2, 735:1/2, 752:1/2, 753:1/2, 762:1/2, 789:1/2, 822:1/2, 852:1/2, 861:1/2, 897:1/2, 900:1/2, 916:1/2, 951:1/2, 966:1/2, 984:1/2, 1030:1/2, 1046:1/2, 1047:1/2, 1056:1/2, 1082:1/2, 1112:1/2 |
| `mom_u_adv_uu` | mom_u_adv_uu.F:7 | 320 | 5/5 | - | - |
| `mom_u_adv_vu` | mom_u_adv_vu.F:7 | 320 | 6/9 | 65-74 | 46:1/2 |
| `mom_u_adv_wu` | mom_u_adv_wu.F:7 | 336 | 15/21 | 52-55, 97-106 | 51:1/2, 94:1/2 |
| `mom_u_metric_sphere` | mom_u_metric_sphere.F:7 | 320 | 6/9 | 59-73 | 46:1/2 |
| `mom_u_xviscflux` | mom_u_xviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_u_yviscflux` | mom_u_yviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_v_adv_uv` | mom_v_adv_uv.F:7 | 320 | 5/5 | - | - |
| `mom_v_adv_vv` | mom_v_adv_vv.F:7 | 320 | 5/5 | - | - |
| `mom_v_adv_wv` | mom_v_adv_wv.F:7 | 336 | 15/21 | 52-55, 97-106 | 51:1/2, 94:1/2 |
| `mom_v_metric_sphere` | mom_v_metric_sphere.F:7 | 320 | 6/9 | 60-76 | 44:1/2 |
| `mom_v_xviscflux` | mom_v_xviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_v_yviscflux` | mom_v_yviscflux.F:10 | 320 | 5/5 | - | - |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 10 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 5 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 5 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 4 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 16 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 35 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 25 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 5 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 489 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 5 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 484 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 52 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 5 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 5 | 57/69 | 95, 123-132, 148, 178-179, 196-198 | 93:1/2, 122:1/2, 140:1/2, 147:1/2, 158:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 5 | 74/127 | 145-237, 244-260, 263-278 | 143:1/2, 242:1/2, 243:1/2, 262:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 35 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 25 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 5 | 64/67 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/ptracers** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ptracers_apply_forcing` | ptracers_apply_forcing.F:7 | 320 | 12/23 | 58, 60, 62, 91-96, 103-108 | 57:1/2, 59:1/2, 61:1/2, 90:1/2, 102:1/2 |
| `ptracers_check` | ptracers_check.F:9 | 1 | 63/130 | 102, 149-156, 158-164, 171-176, 182-187, 192-198, 201-207, 210-216, 220-229, 234-241, 248-251 | 39:1/2, 100:1/2, 148:1/2, 157:1/2, 169:1/2, 180:1/2, 191:1/2, 200:1/2, 209:1/2, 219:1/2, 233:1/2, 246:1/2 |
| `ptracers_convect` | ptracers_convect.F:10 | 380 | 8/8 | - | 53:1/2 |
| `ptracers_fields_blocking_exch` | ptracers_fields_blocking_exch.F:8 | 4 | 5/5 | - | 44:1/2 |
| `ptracers_init_fixed` | ptracers_init_fixed.F:8 | 1 | 28/57 | 62-71, 89-90, 98-107, 120-121, 124-127, 142-145 | 43:1/2, 59:1/2, 61:1/2, 79:1/2, 88:1/2, 97:1/2, 118:1/2, 123:1/2, 140:1/2 |
| `ptracers_init_varia` | ptracers_init_varia.F:9 | 1 | 31/34 | 101-102, 128 | 43:1/2, 96:1/2, 99:1/2, 124:1/2 |
| `ptracers_integrate` | ptracers_integrate.F:10 | 16 | 56/63 | 196, 254-264, 369-371, 502 | 136:1/2, 155:1/2, 189:1/2, 251:1/2, 338:1/2, 351:1/2, 368:1/2, 381:1/2, 386:1/2, 467:1/2, 496:1/2 |
| `ptracers_monitor` | ptracers_monitor.F:7 | 5 | 33/37 | 65, 102-104 | 52:1/2, 58:1/2, 62:1/2, 81:1/2, 98:1/2, 117:1/2, 121:1/2 |
| `ptracers_output` | ptracers_output.F:7 | 5 | 4/4 | - | - |
| `ptracers_readparms` | ptracers_readparms.F:9 | 1 | 80/114 | 91-96, 192-198, 201-208, 221, 230, 236-240, 261-266, 306-309 | 89:1/2, 99:1/2, 191:1/2, 200:1/2, 212:1/2, 229:1/2, 234:1/2, 245:1/2, 255:1/2, 259:1/2, 274:1/2, 304:1/2 |
| `ptracers_reset` | ptracers_reset.F:9 | 4 | 5/31 | 58-133 | 53:1/2 |
| `ptracers_set_iolabel` | ptracers_set_iolabel.F:24 | 1 | 33/37 | 116-117, 129-130 | 115:1/2, 128:1/2 |
| `ptracers_switch_onoff` | ptracers_switch_onoff.F:7 | 4 | 3/4 | 40 | 37:1/2 |
| `ptracers_turnoff_io` | ptracers_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `ptracers_write_pickup` | ptracers_write_pickup.F:8 | 1 | 27/36 | 120, 151, 159-165 | 117:1/2, 119:1/2, 136:1/2, 142:1/2, 148:1/2, 150:1/2, 158:1/2 |
| `ptracers_write_state` | ptracers_write_state.F:7 | 5 | 13/16 | 101, 105-106 | 52:1/2, 78:1/2, 98:1/2, 103:1/2 |

**pkg/rw** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xy_rs` | read_fld_xy_rs.F:3 | 4 | 12/15 | 35-37 | 32:1/2 |
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 1 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 9 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 29 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 20 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 5 | 7/7 | - | - |

**verification/tutorial_tracer_adjsens/code_ad** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ptracers_forcing_surf` | ptracers_forcing_surf.F:7 | 16 | 17/45 | 56, 78-96, 117-131, 147-159, 175-181 | 55:1/2, 74:1/2, 115:1/2, 144:1/2, 171:1/2 |

### Compiled but not executed (709 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rl, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rl, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_surf_dr, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_stagger_fields_exchanges, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cartesian_grid, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, rotate_spherical_polar_grid, rotate_uv2en_rl, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dummy_in_dynamics, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cd_code: cd_code_read_pickup
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_get_gen, ctrl_get_gen_rec, ctrl_get_mask2d, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_init_rec, ctrl_map_genarr2d, ctrl_map_ini_gentim2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xy, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/generic_advdiff: gad_advection, gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: bldepth, blmix, enhance, kpp_calc, kpp_calc_diff_ptr, kpp_calc_diff_s, kpp_calc_diff_t, kpp_calc_visc, kpp_check, kpp_diagnostics_init, kpp_do_exch, kpp_doublediff, kpp_forcing_surf, kpp_init_fixed, kpp_init_varia, kpp_output, kpp_transport_ptr, kpp_transport_s, kpp_transport_t, kppmix, ri_iwmix, smooth_horiz, statekpp, wscale, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_uv_boundary, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/ptracers: adptracers_monitor, ptracers_ad_dump, ptracers_calc_wsurf_tr, ptracers_check_pickup, ptracers_debug, ptracers_diagnostics_init, ptracers_diagnostics_state, ptracers_dyn_state_data_dummy, ptracers_dyn_state_mod_dummy, ptracers_mnc_init, ptracers_read_pickup, ptracers_zonal_filt_apply
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs

## tutorial_tracer_adjsens/input_ad.som81

Run `$MJX_REFERENCE/coverage/tutorial_tracer_adjsens/input_ad.som81/job27832235` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/tutorial_tracer_adjsens/input_ad.som81/job27832235/gcov-8316e95/gcov.json.gz`.

The run did not end normally (no `PROGRAM MAIN: Execution ended Normally`): counts cover the code executed up to the stop (see the run's output.txt).

### Physics-active check

| check | measured | active |
|---|---|---|
| theta | 5 blocks, first 4.251363e+00, last 4.247103e+00, min 4.247103e+00, max 4.251363e+00 | yes |
| passive tracer step | ptracers_integrate (16 calls) | yes |
| GM/Redi tensor | gmredi_calc_tensor (16 calls) | yes |
| cost function | global fc = 184096178634598.0; terms: objf_tracer sum 1.840962e+14 | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: calc_wVelocity, doAB_onGtGs, doResetHFactors, doSaltClimRelax, doThetaClimRelax, dumpInitAndLast, fluidIsWater, GM_ExtraDiag, inAdExact, momAdvection, momDissip_In_AB, momForcing, momPressureForcing, momStepping, momTidalForcing, momViscosity, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, PTRACERS_AdamsBashGtr, PTRACERS_doAB_onGpTr, PTRACERS_startAllTrc, PTRACERS_useGMRedi, saltAdvection, saltForcing, saltIsActiveTr, saltMultiDimAdvec, saltSOM_Advection, saltStepping, snapshot_mdsio, tempAdvection, tempForcing, tempIsActiveTr, tempMultiDimAdvec, tempSOM_Advection, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useCoriolis, useGMRediInAdMode, useHarmonicVisc, useMultiDimAdvec, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, bottomVisc_pCell, cg2dFullAdjoint, debugMode, deepAtmosphere, fluidIsAir, globalFiles, GM_AdvSeparate, GM_InMomAsStress, GM_UseBVP, GM_useGEOM, GM_useLeithQG, GM_useSubMeso, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momImplVertAdv, nonHydrostatic, printMapIncludesZeros, PTRACERS_AdamsBash_Tr, PTRACERS_addSrelax2EmP, PTRACERS_ImplVertAdv, PTRACERS_MultiDimAdv, PTRACERS_pickup_read_mnc, PTRACERS_pickup_write_mnc, PTRACERS_snapshot_mnc, PTRACERS_SOM_Advection, PTRACERS_useDWNSLP, PTRACERS_useKPP, PTRACERS_useRecords, quasiHydrostatic, rotateGrid, saltImplVertAdv, tempImplVertAdv, use3Dsolver, useApproxAdvectionInAdMode, useBiharmonicVisc, useCoupler, useCtrlCostContribution, useGGL90inAdMode, useKPPinAdMode, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useRealFreshWaterFlux, useSALT_PLUMEinAdMode, useSEAICEinAdMode, useSingleCpuInput, useSingleCpuIO, useSmag3D, useSRCGSolver, useStrainTensionVisc, useVariableVisc, usingCartesianGrid, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, vectorInvariantMomentum
- selectors: monitorSelect=3, pCellMix_select=0, saltVertAdvScheme=81, selectAddFluid=0, selectBotDragQuadr=-1, selectNHfreeSurf=0, selectPenetratingSW=1, selectSigmaCoord=0, tempVertAdvScheme=80

### Executed routines (371)

**eesupp/src** (76)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 12/28 | 138-139, 144, 150-151, 194-220 | 137:1/2, 143:1/2, 149:1/2, 171:1/2, 178:1/2, 183:1/2 |
| `bar2` | bar2.F:58 | 6214 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 4 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 6921 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `different_multiple` | different_multiple.F:7 | 64 | 15/15 | - | - |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1654 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 29 | 26/44 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 24 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1683 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1654 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1654 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1654 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1654 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 29 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 29 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 29 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 29 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_sm_3d_rl` | exch_sm_3d_rl.F:8 | 8 | 12/31 | 80-130 | 72:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 4 | 12/13 | 76 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 24 | 14/40 | 85-153 | 77:1/2 |
| `exch_uv_dgrid_3d_rl` | exch_uv_dgrid_3d_rl.F:8 | 4 | 14/32 | 88-177 | 72:1/2, 74:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 6 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 5 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 1534 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 17 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 12 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 26977 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 210 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_int` | global_sum.F:188 | 64 | 12/12 | - | 240:1/2 |
| `global_sum_r8` | global_sum.F:97 | 26 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 2706 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 2124 | 9/10 | 112 | 108:1/2 |
| `ilnblnk` | utils.F:123 | 8984 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 173:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 18 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 628 | 6/6 | - | 33:1/2, 34:1/2 |
| `master_cpu_thread` | master_cpu_thread.F:7 | 1 | 5/5 | - | 41:1/2 |
| `mds_reclen` | mds_reclen.F:3 | 433 | 6/11 | 34-39 | 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 504 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 6732 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 146 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 9 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_error` | print.F:167 | 2 | 18/28 | 228-232, 260-261, 270, 274, 283-284 | 226:1/2, 242:1/2, 245:1/2, 258:1/2, 265:1/2, 269:1/2, 279:1/2, 280:1/2 |
| `print_list_i` | print.F:305 | 40 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 117 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 164 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 689:1/2, 691:1/2 |
| `print_message` | print.F:23 | 2614 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 208 | 45/159 | 232, 485-587, 595-730 | 227:1/2, 228:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 285:1/2, 286:1/2, 294:1/2 |
| `timer_get_time` | timers.F:743 | 208 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 208 | 11/12 | 57 | 67:1/2 |
| `timer_start` | timers.F:884 | 105 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 103 | 4/4 | - | - |
| `ucase` | utils.F:311 | 417 | 7/8 | 333 | 332:1/2 |
| `write_0d_c` | write_utils.F:579 | 7 | 19/20 | 629 | 633:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 32 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 117 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 112 | 9/9 | - | - |
| `write_0d_rs` | write_utils.F:465 | 1 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 47 | 32/32 | - | - |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (107)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 960 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 16/51 | 55-71, 87-93, 113-133 | 52:1/2, 86:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 320 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 320 | 18/50 | 467, 469, 471, 563-604, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2, 682:1/2 |
| `apply_forcing_u` | apply_forcing.F:15 | 320 | 14/20 | 97, 99, 150-155 | 96:1/2, 98:1/2, 127:1/2, 149:1/2 |
| `apply_forcing_v` | apply_forcing.F:205 | 320 | 14/20 | 287, 289, 340-345 | 286:1/2, 288:1/2, 317:1/2, 339:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 80 | 28/41 | 164-166, 175-197 | 83:1/2, 146:1/2, 173:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 960 | 26/26 | - | - |
| `calc_div_ghat` | calc_div_ghat.F:6 | 320 | 18/18 | - | - |
| `calc_grad_phi_hyd` | calc_grad_phi_hyd.F:7 | 320 | 22/81 | 68-80, 87-119, 162-261 | 63:1/2, 84:1/2, 158:1/2 |
| `calc_grad_phi_surf` | calc_grad_phi_surf.F:6 | 16 | 8/8 | - | - |
| `calc_oce_mxlayer` | calc_oce_mxlayer.F:7 | 16 | 6/77 | 86-243 | 74:1/2, 84:1/2 |
| `calc_phi_hyd` | calc_phi_hyd.F:10 | 320 | 37/155 | 149, 184, 194-197, 212-240, 268-610 | 100:1/2, 130:1/2, 138:1/2, 181:1/2, 193:1/2, 205:1/2, 258:1/2, 621:1/2, 624:1/2, 660:1/2 |
| `calc_r_star` | calc_r_star.F:10 | 6 | 66/119 | 68, 143-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 111:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `calc_viscosity` | calc_viscosity.F:10 | 16 | 11/16 | 81, 124-127 | 77:1/2, 121:1/2 |
| `cg2d_nsa` | cg2d_nsa.F:15 | 4 | 89/102 | 120-122, 125-130, 222, 358-362, 392 | 118:1/2, 124:1/2, 219:1/2, 226:1/2, 227:1/2, 288:1/2, 332:1/2, 357:1/2, 389:1/2 |
| `config_check` | config_check.F:14 | 1 | 93/525 | 104-122, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 204-209, 213-218, 222-227, 233-238, 241-247, 253-259, 266-271, 275-280, 344-349, 366-371, 417-423, 442-448, 454-460, 484-490, 497-503, 523-529, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 970-979, 986-989, 992-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 103:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 202:1/2, 211:1/2, 220:1/2, 230:1/2, 240:1/2, 252:1/2, 264:1/2, 273:1/2, 342:1/2, 364:1/2, 404:1/2, 441:1/2, 453:1/2, 482:1/2, 495:1/2, 521:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 955:1/2, 984:1/2, 985:1/2, 991:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 439/537 | 135, 139, 144, 147, 150, 153-168, 174, 177, 180, 183-191, 233, 240, 268-281, 307-322, 533-538, 554-588, 756-762, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 133:1/2, 136:1/2, 142:1/2, 145:1/2, 148:1/2, 151:1/2, 172:1/2, 175:1/2, 178:1/2, 181:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `convective_adjustment` | convective_adjustment.F:10 | 16 | 30/34 | 103-106 | 67:1/2, 95:1/2, 110:4/8, 162:1/2 |
| `convective_adjustment_ini` | convective_adjustment_ini.F:10 | 4 | 29/33 | 112-115 | 104:1/2, 119:4/8, 171:1/2 |
| `convective_weights` | convective_weights.F:6 | 380 | 15/15 | - | - |
| `convectively_mixtracer` | convectively_mixtracer.F:6 | 1140 | 7/7 | - | - |
| `correction_step` | correction_step.F:7 | 16 | 18/25 | 158-167, 180-188 | 156:1/2, 179:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 48 | 6/6 | - | - |
| `diags_phi_hyd` | diags_phi_hyd.F:7 | 320 | 7/22 | 76-114 | 73:1/2 |
| `diags_phi_rlow` | diags_phi_rlow.F:6 | 320 | 25/48 | 79-84, 123-125, 134-170 | 61:1/2, 76:1/2, 121:1/2, 132:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 4 | 11/20 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 4 | 13/18 | 53-56, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2, 96:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 4 | 87/109 | 557, 771-784, 797-798, 830-833, 869-874, 953-958, 1043, 1103 | 248:1/2, 251:1/2, 553:1/2, 577:1/2, 728:1/2, 731:1/2, 758:1/2, 795:1/2, 810:1/2, 812:1/2, 839:1/2, 867:1/2, 896:1/2, 951:1/2, 1030:1/2, 1032:1/2, 1096:1/2, 1102:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 4 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 5 | 10/17 | 97-110, 145 | 96:1/2, 116:1/2, 144:1/2, 212:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 4 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `dynamics` | dynamics.F:21 | 4 | 55/77 | 337-340, 358, 533, 591-599, 623-631, 708-720 | 250:1/2, 336:1/2, 353:1/2, 385:1/2, 490:1/2, 515:1/2, 582:1/2, 615:1/2, 707:1/2, 735:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 4 | 3/125 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 4 | 45/71 | 75, 151-153, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 82:1/2, 149:1/2, 160:1/2, 173:1/2, 188:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 1384 | 14/60 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 58 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forcing_surf_relax` | forcing_surf_relax.F:7 | 4 | 16/23 | 64, 127-151 | 63:1/2, 115:1/2, 116:1/2 |
| `forward_step` | forward_step.F:70 | 4 | 98/120 | 483-485, 730-734, 767-769, 842-853, 952-960, 980, 1176, 1201-1202 | 424:1/2, 468:1/2, 500:1/2, 530:1/2, 571:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 789:1/2, 832:1/2, 866:1/2, 897:1/2, 924:1/2, 939:1/2, 976:1/2, 979:1/2, 1002:1/2, 1151:1/2, 1175:1/2, 1187:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 1280 | 7/12 | 49-66 | 39:1/2, 40:1/2 |
| `grad_sigma` | grad_sigma.F:6 | 320 | 20/22 | 61, 74 | 59:1/2, 72:1/2 |
| `impldiff` | impldiff.F:7 | 48 | 64/66 | 101-102 | 93:1/2, 199:1/2, 219:1/2 |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 81/87 | 120, 151, 161, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 152:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 24/65 | 57-63, 70-79, 106-112, 121-180, 191 | 55:1/2, 68:1/2, 84:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 28/28 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 49/49 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 8/10 | 37-38 | 28:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 61/87 | 54, 60, 78, 80, 83-90, 98, 108, 112, 116-123, 139, 158, 176-177, 193, 243 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 115:1/2, 136:1/2, 149:1/2, 153:1/2, 172:1/2, 192:1/2, 242:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 103/115 | 131, 134-144, 192 | 130:1/2, 132:1/2, 155:1/2, 161:1/2, 163:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 24/70 | 90-182, 192, 211, 218 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 4 | 28/28 | - | - |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 183/198 | 210-212, 247-261, 449-451 | 65:1/2, 206:1/2, 243:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 9/11 | 52-53 | 51:1/2 |
| `ini_model_io` | ini_model_io.F:8 | 1 | 28/52 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 232:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 74/74 | - | 47:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 408/889 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 536, 538-541, 551, 553, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1109-1112, 1123-1124, 1127, 1131-1133, 1139, 1148, 1151, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1304-1305, 1314-1317, 1320-1321, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1393-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1552-1555, 1558-1561, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 537:1/2, 543:1/2, 550:1/2, 552:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 656:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1116:1/2, 1117:1/2, 1118:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1200:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1551:1/2, 1557:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/19 | - | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 20/22 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 25/38 | 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_spherical_polar_grid` | ini_spherical_polar_grid.F:8 | 1 | 75/85 | 258-263, 277-280 | 257:1/2, 276:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 26/47 | 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 17/22 | 57-63 | 55:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 42/47 | 309-321, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 246:1/2, 251:1/2, 282:1/2, 284:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 5 | 56/70 | 146-160, 212-214, 279-282 | 90:1/2, 93:1/2, 95:1/2, 142:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 400 | 18/36 | 95-117, 177-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 4 | 20/27 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 187:1/2, 189:1/2, 229:1/2, 231:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 27/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161, 173 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 4 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `momentum_correction_step` | momentum_correction_step.F:7 | 4 | 16/17 | 128 | 63:1/2, 127:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 98/98 | - | 100:1/2, 232:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 66/109 | 132-140, 207, 212, 265, 280, 289, 321, 342, 349, 356, 369, 376, 403, 412, 437, 460, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 246:1/2, 252:1/2, 260:1/2, 264:1/2, 269:1/2, 275:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 301:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 335:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 416:1/2, 420:1/2, 432:1/2, 436:1/2, 441:1/2, 445:1/2, 453:1/2, 459:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 24/26 | 319-321 | 144:1/2, 193:1/2, 200:1/2, 202:1/2, 209:1/2, 211:1/2, 317:1/2, 327:1/2, 329:1/2, 358:1/2, 425:1/2, 427:1/2, 651:1/2, 655:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 17/21 | 169, 253-255, 637 | 168:1/2, 192:1/2, 194:1/2, 204:1/2, 206:1/2, 251:1/2, 270:1/2, 332:1/2, 617:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 17 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 10/10 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 1 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 11/11 | - | 103:1/2, 124:1/2, 155:1/2, 255:1/2 |
| `reset_nlfs_vars` | reset_nlfs_vars.F:6 | 4 | 8/11 | 48-50 | 47:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 16 | 53/63 | 192, 268-280, 386, 397-399, 434, 517 | 147:1/2, 177:1/2, 257:1/2, 259:1/2, 319:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 413:1/2, 495:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 313/314 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 97/165 | 56-57, 60-63, 67, 71, 76, 109-115, 120-126, 171-175, 186-189, 192-195, 202, 208, 214, 220, 226, 232, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 54/195 | 101-133, 140-165, 180-187, 212-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_for_pressure` | solve_for_pressure.F:7 | 4 | 58/85 | 143-147, 152-157, 164, 170-176, 228-234, 265, 269, 312, 319, 327, 342-344 | 107:1/2, 142:1/2, 151:1/2, 163:1/2, 169:1/2, 216:1/2, 263:1/2, 268:1/2, 288:1/2, 298:1/2, 318:1/2, 325:1/2, 332:1/2, 334:1/2, 335:1/2, 341:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `swfrac` | swfrac.F:7 | 1 | 9/9 | - | - |
| `temp_integrate` | temp_integrate.F:13 | 16 | 53/63 | 194, 270-282, 388, 399-401, 436, 519 | 149:1/2, 179:1/2, 259:1/2, 261:1/2, 321:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 497:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 36/36 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 19/27 | 736-797 | 606:1/2, 616:1/2, 707:1/2, 733:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 4 | 52/74 | 158, 177, 218-250, 395-406 | 127:1/2, 132:1/2, 134:1/2, 153:1/2, 173:1/2, 206:1/2, 207:1/2, 271:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 344:1/2, 346:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep` | timestep.F:10 | 320 | 59/83 | 118-121, 210-213, 220-223, 286-312, 328-332 | 104:1/2, 116:1/2, 129:1/2, 139:1/2, 209:1/2, 219:1/2, 229:1/2, 276:1/2, 277:1/2, 321:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 48 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 4 | 6/6 | - | 115:1/2 |
| `turnoff_model_io` | turnoff_model_io.F:7 | 1 | 18/18 | - | 55:1/2, 118:1/2 |
| `update_cg2d` | update_cg2d.F:7 | 5 | 41/49 | 58, 132-140, 169, 175, 182 | 57:1/2, 61:1/2, 131:1/2, 159:1/2, 168:1/2, 174:1/2, 181:1/2 |
| `update_etah` | update_etah.F:7 | 5 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 9 | 29/29 | - | - |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 51/75 | 253-257, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 247:1/2, 250:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 5 | 18/20 | 85, 126 | 81:1/2, 84:1/2, 95:1/2, 123:1/2 |

**pkg/autodiff** (18)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `active_read_3d_rl` | active_file_control.F:29 | 2 | 11/36 | 118-136, 149-179, 195 | 114:1/2, 143:1/2, 189:1/2, 199:1/2 |
| `active_read_xyz` | active_file.F:100 | 2 | 6/6 | - | - |
| `active_write_3d_rl` | active_file_control.F:714 | 1 | 10/20 | 796-816, 826 | 790:1/2, 821:1/2, 830:1/2 |
| `active_write_3d_rs` | active_file_control.F:836 | 3 | 10/20 | 918-938, 948 | 912:1/2, 943:1/2, 952:1/2 |
| `active_write_gen_rs` | active_file_gen.F:288 | 3 | 6/11 | 348-364 | 368:1/2 |
| `active_write_xyz` | active_file.F:421 | 1 | 7/7 | - | - |
| `autodiff_check` | autodiff_check.F:7 | 1 | 3/12 | 39-49 | 37:1/2 |
| `autodiff_findunit` | autodiff_findunit.F:3 | 1 | 11/19 | 40-46, 58-60 | 35:1/2, 38:1/2, 48:1/2, 56:1/2 |
| `autodiff_inadmode_set` | autodiff_inadmode_set.F:3 | 4 | 2/2 | - | - |
| `autodiff_inadmode_unset` | autodiff_inadmode_unset.F:3 | 4 | 2/2 | - | - |
| `autodiff_ini_model_io` | autodiff_ini_model_io.F:18 | 1 | 17/27 | 69-83 | 66:1/2, 68:1/2 |
| `autodiff_init_varia` | autodiff_init_varia.F:9 | 1 | 6/6 | - | - |
| `autodiff_readparms` | autodiff_readparms.F:11 | 1 | 75/119 | 63-68, 127-132, 165-172, 175-176, 237-247, 289-335, 347-350 | 61:1/2, 71:1/2, 125:1/2, 164:1/2, 174:1/2, 235:1/2, 286:1/2, 345:1/2 |
| `autodiff_restore` | autodiff_restore.F:15 | 1 | 4/4 | - | 83:1/2, 452:1/2 |
| `autodiff_store` | autodiff_store.F:15 | 1 | 4/4 | - | 84:1/2, 538:1/2 |
| `autodiff_whtapeio_sync` | autodiff_whtapeio_sync.F:11 | 2 | 32/44 | 80, 86, 107-108, 161-170 | 79:1/2, 83:1/2, 85:1/2, 93:1/2, 94:1/2, 99:1/2, 101:1/2, 160:1/2 |
| `dummy_for_etan` | dummy_for_etan.F:6 | 5 | 2/2 | - | - |
| `dummy_in_stepping` | dummy_in_stepping.F:6 | 4 | 2/2 | - | - |

**pkg/cd_code** (4)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cd_code_ini_vars` | cd_code_ini_vars.F:3 | 1 | 15/16 | 53 | 52:1/2 |
| `cd_code_init_fixed` | cd_code_init_fixed.F:3 | 1 | 2/2 | - | - |
| `cd_code_scheme` | cd_code_scheme.F:7 | 320 | 47/49 | 82-83 | 78:1/2 |
| `cd_code_write_pickup` | cd_code_write_pickup.F:8 | 1 | 11/11 | - | 85:1/2 |

**pkg/cost** (10)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `cost_check` | cost_check.F:3 | 1 | 7/12 | 53-57 | 26:1/2, 52:1/2 |
| `cost_copy_file` | cost_copy_file.F:8 | 1 | 10/18 | 63-76 | 61:1/2 |
| `cost_dependent_init` | cost_dependent_init.F:3 | 1 | 7/7 | - | 58:1/2 |
| `cost_driver` | cost_driver.F:7 | 1 | 7/7 | - | 57:1/2, 59:1/2 |
| `cost_final` | cost_final.F:11 | 1 | 47/47 | - | 82:1/2, 102:1/2, 216:1/2, 228:1/2 |
| `cost_init_fixed` | cost_init_fixed.F:8 | 1 | 3/3 | - | - |
| `cost_init_varia` | cost_init_varia.F:3 | 1 | 20/20 | - | 89:1/2 |
| `cost_readparms` | cost_readparms.F:3 | 1 | 40/47 | 92, 103-109 | 47:1/2, 90:1/2, 101:1/2 |
| `cost_tile` | cost_tile.F:59 | 4 | 5/5 | - | - |
| `cost_tracer` | cost_tracer.F:3 | 16 | 8/8 | - | - |

**pkg/ctrl** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ctrl_assign` | ctrl_toolbox.F:16 | 1 | 8/8 | - | - |
| `ctrl_bound_3d` | ctrl_bound.F:14 | 1 | 3/13 | 43-54 | 41:1/2 |
| `ctrl_check` | ctrl_check.F:22 | 1 | 31/52 | 139-143, 162-168, 376-381, 591-594 | 80:1/2, 136:1/2, 148:1/2, 151:1/2, 155:1/2, 373:1/2, 589:1/2 |
| `ctrl_check_retired_parms` | ctrl_readparms.F:864 | 34 | 4/11 | 911-919 | 923:1/2 |
| `ctrl_cost_driver` | ctrl_cost_driver.F:7 | 1 | 3/13 | 66-153 | 65:1/2 |
| `ctrl_cost_final` | ctrl_cost_final.F:9 | 1 | 3/35 | 82-225 | 66:1/2 |
| `ctrl_cprsrs` | ctrl_toolbox.F:154 | 1 | 9/9 | - | - |
| `ctrl_get_mask3d` | ctrl_get_mask.F:16 | 1 | 3/3 | - | - |
| `ctrl_init_ctrlvar` | ctrl_init_ctrlvar.F:9 | 1 | 22/50 | 80-87, 114-126, 143-171 | 79:1/2, 88:1/2, 113:1/2, 132:1/2, 134:1/2, 140:1/2 |
| `ctrl_init_fixed` | ctrl_init_fixed.F:6 | 1 | 36/36 | - | 251:1/2, 380:1/2 |
| `ctrl_init_variables` | ctrl_init_variables.F:11 | 1 | 11/11 | - | 81:1/2, 104:1/2, 149:1/2 |
| `ctrl_init_wet` | ctrl_init_wet.F:3 | 1 | 95/104 | 181-219 | 168:1/2, 179:1/2, 358:1/4 |
| `ctrl_map_forcing` | ctrl_map_forcing.F:9 | 4 | 2/2 | - | - |
| `ctrl_map_genarr3d` | ctrl_map_genarr.F:211 | 1 | 45/61 | 290-292, 296-298, 301, 307-308, 353-360, 370-373 | 272:1/2, 289:1/2, 294:1/2, 300:1/2, 303:1/2, 344:1/2, 349:1/2, 352:1/2, 369:1/2, 397:1/2, 411:1/2 |
| `ctrl_map_gentim2d` | ctrl_map_gentim2d.F:6 | 4 | 2/2 | - | - |
| `ctrl_map_ini_genarr` | ctrl_map_ini_genarr.F:24 | 1 | 29/36 | 356, 358, 360, 362, 364, 393, 395 | 128:1/2, 354:1/2, 355:1/2, 357:1/2, 359:1/2, 361:1/2, 363:1/2, 379:1/2, 381:1/2, 384:1/2, 392:1/2, 394:1/2, 416:1/2, 434:1/2 |
| `ctrl_readparms` | ctrl_readparms.F:18 | 1 | 197/235 | 181-186, 537-551, 562, 573-584, 587-598, 602-605, 799-806 | 179:1/2, 189:1/2, 483:1/2, 492:1/2, 536:1/2, 560:1/2, 572:1/2, 586:1/2, 600:1/2, 798:1/2 |
| `ctrl_set_fname` | ctrl_set_fname.F:6 | 1 | 15/16 | 60 | 42:1/2, 46:1/2 |
| `ctrl_set_globfld_xyz` | ctrl_set_globfld_xyz.F:3 | 1 | 10/10 | - | - |
| `ctrl_set_retired_parms` | ctrl_readparms.F:820 | 20 | 10/10 | - | 854:1/2, 855:2/4, 858:1/2 |
| `ctrl_summary` | ctrl_summary.F:3 | 1 | 73/83 | 153-155, 218-222, 304-306 | 144:1/2, 146:1/2, 209:1/2, 217:1/2, 302:1/2 |
| `optim_readparms` | optim_readparms.F:3 | 1 | 24/27 | 67-73 | 65:1/2, 76:1/2 |

**pkg/generic_advdiff** (20)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advscheme_get` | gad_advscheme.F:116 | 7 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_c2_adv_r` | gad_c2_adv_r.F:7 | 304 | 7/10 | 52-54 | 51:1/2 |
| `gad_c2_adv_x` | gad_c2_adv_x.F:7 | 320 | 7/7 | - | - |
| `gad_c2_adv_y` | gad_c2_adv_y.F:7 | 320 | 7/7 | - | - |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 960 | 79/183 | 181-184, 213-216, 236-244, 259-296, 331-333, 340, 388-425, 460-462, 469, 517-545, 552-577, 614, 620, 659-690, 793 | 176:1/2, 199:1/2, 212:1/2, 229:1/2, 256:1/2, 328:1/2, 339:1/2, 345:1/2, 385:1/2, 457:1/2, 468:1/2, 474:1/2, 515:1/2, 549:1/2, 605:1/2, 617:1/2, 625:1/2, 658:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 15/68 | 82-88, 97-106, 119-129, 137-142, 147-152, 157-162, 167-172, 178-181 | 39:1/2, 81:1/2, 95:1/2, 118:1/2, 135:1/2, 145:1/2, 155:1/2, 165:1/2, 176:1/2 |
| `gad_diff_x` | gad_diff_x.F:7 | 960 | 6/6 | - | - |
| `gad_diff_y` | gad_diff_y.F:7 | 960 | 7/7 | - | - |
| `gad_exch_som` | gad_exch_som.F:6 | 8 | 9/9 | - | - |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 95/123 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 12/14 | 68-69 | 58:1/2, 60:1/2 |
| `gad_som_adv_r` | gad_som_adv_r.F:13 | 640 | 123/137 | 268-285 | 158:1/2, 261:1/2 |
| `gad_som_adv_x` | gad_som_adv_x.F:13 | 640 | 94/103 | 144-153 | 134:1/2, 142:1/2, 157:1/2, 158:1/2 |
| `gad_som_adv_y` | gad_som_adv_y.F:13 | 640 | 94/103 | 144-153 | 134:1/2, 142:1/2, 157:1/2, 158:1/2 |
| `gad_som_advect` | gad_som_advect.F:14 | 32 | 102/147 | 216-219, 222-225, 230-243, 258-261, 318-331, 361, 386, 408, 433, 443-455, 484, 597-602, 633 | 215:1/2, 221:1/2, 229:1/2, 257:1/2, 316:1/2, 356:1/2, 366:1/2, 403:1/2, 413:1/2, 441:1/2, 482:1/2, 496:1/2, 584:1/2, 611:1/2 |
| `gad_som_exchanges` | gad_som_exchanges.F:6 | 4 | 6/6 | - | 35:1/2, 40:1/2 |
| `gad_som_lim_r` | gad_som_lim_r.F:7 | 32 | 15/15 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 18/20 | 73, 90 | 70:1/2, 72:1/2, 87:1/2, 89:1/2 |

**pkg/gmredi** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gmredi_calc_diff` | gmredi_calc_diff.F:3 | 48 | 10/16 | 70-82 | 49:1/2 |
| `gmredi_calc_psi_bolus` | gmredi_calc_psi_bolus.F:16 | 16 | 32/33 | 127 | 87:1/2, 124:1/2 |
| `gmredi_calc_tensor` | gmredi_calc_tensor.F:24 | 16 | 146/188 | 166-209, 292-294, 529, 703-706, 768, 898-901, 965 | 164:1/2, 291:1/2, 526:1/2, 610:1/2, 620:1/2, 702:1/2, 765:1/2, 815:1/2, 897:1/2, 962:1/2, 1012:1/2 |
| `gmredi_check` | gmredi_check.F:9 | 1 | 55/178 | 138-140, 146-151, 157-162, 170-175, 221-226, 298-303, 360-365, 369-372, 377-383, 388-391, 405-408, 413-415, 420-456, 465-495, 506-509, 513-516, 531-535, 541-544 | 54:1/2, 137:1/2, 144:1/2, 155:1/2, 168:1/2, 219:1/2, 296:1/2, 358:1/2, 368:1/2, 376:1/2, 387:1/2, 394:1/2, 411:1/2, 418:1/2, 460:1/2, 502:1/2, 505:1/2, 512:1/2, 525:1/2, 539:1/2 |
| `gmredi_do_exch` | gmredi_do_exch.F:6 | 4 | 3/5 | 52-54 | 49:1/2 |
| `gmredi_init_fixed` | gmredi_init_fixed.F:6 | 1 | 17/23 | 63-64, 67-68, 82, 86 | 62:1/2, 66:1/2, 72:1/2, 80:1/2, 84:1/2 |
| `gmredi_init_varia` | gmredi_init_varia.F:9 | 1 | 23/27 | 154, 158, 161, 164 | 75:1/2, 152:1/2, 156:1/2, 160:1/2, 163:1/2 |
| `gmredi_output` | gmredi_output.F:7 | 4 | 3/12 | 50-61 | 47:1/2 |
| `gmredi_readparms` | gmredi_readparms.F:9 | 1 | 106/146 | 99-104, 235-239, 245, 253-266, 271, 275-276, 289-295, 298-304, 308-315 | 97:1/2, 107:1/2, 223:1/2, 226:1/2, 231:1/2, 234:1/2, 242:1/2, 244:1/2, 268:1/2, 274:1/2, 288:1/2, 297:1/2, 307:1/2 |
| `gmredi_residual_flow` | gmredi_residual_flow.F:6 | 16 | 20/20 | - | 59:1/2 |
| `gmredi_rtransport` | gmredi_rtransport.F:8 | 960 | 15/25 | 83-86, 178-200 | 81:1/2, 160:1/2 |
| `gmredi_slope_limit` | gmredi_slope_limit.F:9 | 944 | 54/87 | 176, 234, 397, 525-533, 542-549, 570-641 | 171:1/2, 229:1/2, 393:1/2, 522:1/2, 539:1/2, 555:1/2 |
| `gmredi_slope_psi` | gmredi_slope_psi.F:9 | 304 | 48/100 | 100, 183, 273-286, 294-310, 330-418 | 95:1/2, 181:1/2, 218:1/2, 252:1/2, 270:1/2, 290:1/2, 314:1/2 |
| `gmredi_write_pickup` | gmredi_write_pickup.F:7 | 1 | 3/3 | - | - |
| `gmredi_xtransport` | gmredi_xtransport.F:9 | 960 | 22/35 | 97-100, 200-233 | 95:1/2, 104:1/2, 143:1/2, 196:1/2 |
| `gmredi_ytransport` | gmredi_ytransport.F:9 | 960 | 22/35 | 97-100, 200-233 | 95:1/2, 104:1/2, 143:1/2, 196:1/2 |

**pkg/grdchk** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `grdchk_check` | grdchk_check.F:6 | 1 | 9/13 | 57-60 | 47:1/2, 55:1/2 |
| `grdchk_ctrl_fname` | grdchk_ctrl_fname.F:11 | 1 | 11/31 | 64-94 | 50:1/2, 52:1/2, 57:1/2, 63:1/2 |
| `grdchk_get_mask` | grdchk_get_mask.F:6 | 1 | 34/44 | 81-93 | 52:1/2, 73:1/2 |
| `grdchk_get_position` | grdchk_get_position.F:9 | 1 | 50/71 | 96, 121-130, 202-211, 219-222 | 72:1/2, 77:1/2, 92:1/2, 103:1/2, 107:1/2, 112:1/2, 115:1/2, 116:1/2, 189:1/2, 201:1/2 |
| `grdchk_getadxx` | grdchk_getadxx.F:6 | 1 | 11/17 | 88-123 | 84:1/2 |
| `grdchk_loc` | grdchk_loc.F:11 | 1 | 60/118 | 116-126, 134, 163, 173, 192-202, 278, 282, 298-357 | 90:1/2, 101:1/2, 102:1/2, 104:1/2, 130:1/2, 145:1/2, 153:1/2, 172:1/2, 183:1/2, 185:1/2, 186:1/2, 280:1/2, 281:1/2 |
| `grdchk_main` | grdchk_main.F:53 | 1 | 49/138 | 205, 247-255, 280-549 | 202:1/2, 227:1/2, 229:6/8, 234:1/2, 241:1/2 |
| `grdchk_readparms` | grdchk_readparms.F:6 | 1 | 36/49 | 70-75, 123-125, 128-130, 133-136 | 68:1/2, 78:1/2, 122:1/2, 127:1/2, 132:1/2, 143:1/2 |
| `grdchk_summary` | grdchk_summary.F:8 | 1 | 41/41 | - | 37:1/2 |

**pkg/kpp** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `kpp_calc_dummy` | kpp_calc.F:726 | 16 | 11/11 | - | - |
| `kpp_readparms` | kpp_readparms.F:8 | 1 | 5/112 | 69-248 | 59:1/2, 61:1/2 |

**pkg/mdsio** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r4torl` | mdsio_pass_r4torl.F:12 | 45 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r4tors` | mdsio_pass_r4tors.F:12 | 25 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 39 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 3 | 13/36 | 60, 65-71, 92-115 | 59:1/2, 63:1/2, 64:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 10 | 96/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 235-237, 247-253, 265-365, 460-462, 487, 535-538, 543, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 220:1/2, 230:1/2, 233:1/2, 246:1/2, 262:1/2, 377:1/2, 435:1/4, 437:1/4, 458:1/2, 486:1/2, 489:1/4, 493:1/2, 530:1/2, 540:1/2, 541:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 2 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 9/18 | 49-51, 62-74 | 47:1/2, 54:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 103 | 87/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 382-385, 396-406, 425-434, 474-484, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 253:1/2, 269:1/2, 377:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 512:1/4, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 383 | 40/52 | 108, 135-139, 146-147, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/mom_common** (7)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_hfacz` | mom_calc_hfacz.F:13 | 320 | 17/17 | - | - |
| `mom_calc_ke` | mom_calc_ke.F:7 | 320 | 12/26 | 62-66, 77-84, 90-97, 115-137 | 61:1/2, 70:1/2, 88:1/2, 101:1/2 |
| `mom_init_fixed` | mom_init_fixed.F:8 | 1 | 37/39 | 51-52 | 43:1/2, 45:1/2, 50:1/2, 99:1/2, 102:1/2, 123:1/2 |
| `mom_u_botdrag_coeff` | mom_u_botdrag_coeff.F:10 | 320 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_u_rviscflux` | mom_u_rviscflux.F:7 | 640 | 9/9 | - | - |
| `mom_v_botdrag_coeff` | mom_v_botdrag_coeff.F:10 | 320 | 34/63 | 80-83, 115-119, 134-157, 163-179, 185-207, 212 | 68:1/2, 72:1/2, 113:1/2, 122:1/2, 133:1/2, 161:1/2, 183:1/2, 211:1/2 |
| `mom_v_rviscflux` | mom_v_rviscflux.F:7 | 640 | 9/9 | - | - |

**pkg/mom_fluxform** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mom_calc_rtrans` | mom_calc_rtrans.F:6 | 336 | 33/37 | 72-79 | 69:1/2, 111:1/2 |
| `mom_fluxform` | mom_fluxform.F:42 | 320 | 167/248 | 258-259, 265, 276, 299-303, 331-350, 457, 514-523, 566-594, 607, 663-666, 690-693, 738-742, 764-767, 862-890, 902, 958-961, 985-988, 1033-1037, 1058-1061, 1092-1100, 1113-1124 | 257:1/2, 264:1/2, 270:1/2, 298:1/2, 330:1/2, 422:1/2, 444:1/2, 453:1/2, 476:1/2, 510:1/2, 512:1/2, 556:1/2, 565:1/2, 601:1/2, 605:1/2, 621:1/2, 656:1/2, 671:1/2, 689:1/2, 735:1/2, 752:1/2, 753:1/2, 762:1/2, 789:1/2, 822:1/2, 852:1/2, 861:1/2, 897:1/2, 900:1/2, 916:1/2, 951:1/2, 966:1/2, 984:1/2, 1030:1/2, 1046:1/2, 1047:1/2, 1056:1/2, 1082:1/2, 1112:1/2 |
| `mom_u_adv_uu` | mom_u_adv_uu.F:7 | 320 | 5/5 | - | - |
| `mom_u_adv_vu` | mom_u_adv_vu.F:7 | 320 | 6/9 | 65-74 | 46:1/2 |
| `mom_u_adv_wu` | mom_u_adv_wu.F:7 | 336 | 15/21 | 52-55, 97-106 | 51:1/2, 94:1/2 |
| `mom_u_metric_sphere` | mom_u_metric_sphere.F:7 | 320 | 6/9 | 59-73 | 46:1/2 |
| `mom_u_xviscflux` | mom_u_xviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_u_yviscflux` | mom_u_yviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_v_adv_uv` | mom_v_adv_uv.F:7 | 320 | 5/5 | - | - |
| `mom_v_adv_vv` | mom_v_adv_vv.F:7 | 320 | 5/5 | - | - |
| `mom_v_adv_wv` | mom_v_adv_wv.F:7 | 336 | 15/21 | 52-55, 97-106 | 51:1/2, 94:1/2 |
| `mom_v_metric_sphere` | mom_v_metric_sphere.F:7 | 320 | 6/9 | 60-76 | 44:1/2 |
| `mom_v_xviscflux` | mom_v_xviscflux.F:10 | 320 | 5/5 | - | - |
| `mom_v_yviscflux` | mom_v_yviscflux.F:10 | 320 | 5/5 | - | - |

**pkg/monitor** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 10 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 5 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 5 | 13/13 | - | - |
| `mon_calc_advcfl_glob` | mon_calc_advcfl.F:127 | 4 | 17/17 | - | 169:1/2 |
| `mon_calc_advcfl_tile` | mon_calc_advcfl.F:13 | 16 | 28/28 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 35 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_calc_stats_rs` | mon_calc_stats_rs.F:8 | 25 | 71/71 | - | 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 5 | 52/125 | 167-320 | 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 489 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 5 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 484 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 52 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 5 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_surfcor` | mon_surfcor.F:8 | 5 | 57/69 | 95, 123-132, 148, 178-179, 196-198 | 93:1/2, 122:1/2, 140:1/2, 147:1/2, 158:1/2, 177:1/2, 181:1/2, 195:1/2 |
| `mon_vort3` | mon_vort3.F:8 | 5 | 74/127 | 145-237, 244-260, 263-278 | 143:1/2, 242:1/2, 243:1/2, 262:1/2, 333:1/2, 338:1/2, 340:1/2, 344:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 35 | 17/17 | - | - |
| `mon_writestats_rs` | mon_writestats_rs.F:8 | 25 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 5 | 64/67 | 57, 119-120 | 48:1/2, 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/ptracers** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ptracers_apply_forcing` | ptracers_apply_forcing.F:7 | 320 | 12/23 | 58, 60, 62, 91-96, 103-108 | 57:1/2, 59:1/2, 61:1/2, 90:1/2, 102:1/2 |
| `ptracers_check` | ptracers_check.F:9 | 1 | 63/130 | 102, 149-156, 158-164, 171-176, 182-187, 192-198, 201-207, 210-216, 220-229, 234-241, 248-251 | 39:1/2, 100:1/2, 148:1/2, 157:1/2, 169:1/2, 180:1/2, 191:1/2, 200:1/2, 209:1/2, 219:1/2, 233:1/2, 246:1/2 |
| `ptracers_convect` | ptracers_convect.F:10 | 380 | 8/8 | - | 53:1/2 |
| `ptracers_fields_blocking_exch` | ptracers_fields_blocking_exch.F:8 | 4 | 5/5 | - | 44:1/2 |
| `ptracers_init_fixed` | ptracers_init_fixed.F:8 | 1 | 28/57 | 62-71, 89-90, 98-107, 120-121, 124-127, 142-145 | 43:1/2, 59:1/2, 61:1/2, 79:1/2, 88:1/2, 97:1/2, 118:1/2, 123:1/2, 140:1/2 |
| `ptracers_init_varia` | ptracers_init_varia.F:9 | 1 | 31/34 | 101-102, 128 | 43:1/2, 96:1/2, 99:1/2, 124:1/2 |
| `ptracers_integrate` | ptracers_integrate.F:10 | 16 | 56/63 | 196, 254-264, 369-371, 502 | 136:1/2, 155:1/2, 189:1/2, 251:1/2, 338:1/2, 351:1/2, 368:1/2, 381:1/2, 386:1/2, 467:1/2, 496:1/2 |
| `ptracers_monitor` | ptracers_monitor.F:7 | 5 | 33/37 | 65, 102-104 | 52:1/2, 58:1/2, 62:1/2, 81:1/2, 98:1/2, 117:1/2, 121:1/2 |
| `ptracers_output` | ptracers_output.F:7 | 5 | 4/4 | - | - |
| `ptracers_readparms` | ptracers_readparms.F:9 | 1 | 80/114 | 91-96, 192-198, 201-208, 221, 230, 236-240, 261-266, 306-309 | 89:1/2, 99:1/2, 191:1/2, 200:1/2, 212:1/2, 229:1/2, 234:1/2, 245:1/2, 255:1/2, 259:1/2, 274:1/2, 304:1/2 |
| `ptracers_reset` | ptracers_reset.F:9 | 4 | 5/31 | 58-133 | 53:1/2 |
| `ptracers_set_iolabel` | ptracers_set_iolabel.F:24 | 1 | 33/37 | 116-117, 129-130 | 115:1/2, 128:1/2 |
| `ptracers_switch_onoff` | ptracers_switch_onoff.F:7 | 4 | 3/4 | 40 | 37:1/2 |
| `ptracers_turnoff_io` | ptracers_turnoff_io.F:6 | 1 | 5/5 | - | 43:1/2 |
| `ptracers_write_pickup` | ptracers_write_pickup.F:8 | 1 | 27/36 | 120, 151, 159-165 | 117:1/2, 119:1/2, 136:1/2, 142:1/2, 148:1/2, 150:1/2, 158:1/2 |
| `ptracers_write_state` | ptracers_write_state.F:7 | 5 | 13/16 | 101, 105-106 | 52:1/2, 78:1/2, 98:1/2, 103:1/2 |

**pkg/rw** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xy_rs` | read_fld_xy_rs.F:3 | 4 | 12/15 | 35-37 | 32:1/2 |
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 2 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_3d_rl` | read_rec.F:318 | 1 | 7/7 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 9 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 29 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 36 | 7/7 | - | - |
| `write_rec_xyz_rl` | write_rec.F:271 | 5 | 7/7 | - | - |

**verification/tutorial_tracer_adjsens/code_ad** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `ptracers_forcing_surf` | ptracers_forcing_surf.F:7 | 16 | 17/45 | 56, 78-96, 117-131, 147-159, 175-181 | 55:1/2, 74:1/2, 115:1/2, 144:1/2, 171:1/2 |

### Compiled but not executed (699 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, eedie, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_flush, print_maprl, print_maprs, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_printall, write_0d_r4, write_0d_r8, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, calc_eddy_stress, calc_grad_phi_fv, calc_grid_angles, calc_gw, calc_ivdc, calc_surf_dr, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convert_ct2pt, convert_pt2ct, cycle_ab_tracer, diags_oceanic_surf_flux, diags_rho_g, diags_rho_l, diags_sound_speed, do_statevars_diags, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, freeze_surface, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, ini_cartesian_grid, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, look_for_neg_salinity, packages_error_msg, plot_field_xyrl, plot_field_xyrs, plot_field_xyzrl, plot_field_xyzrs, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, rotate_spherical_polar_grid, rotate_uv2en_rl, rotate_uv2en_rs, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep_wvel, tracers_iigw_correction, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/autodiff: active_read_1d, active_read_1d_rl, active_read_1d_rs, active_read_3d_rs, active_read_gen_rl, active_read_gen_rs, active_read_xy, active_read_xy_loc, active_read_xyz_loc, active_read_xz, active_read_xz_loc, active_read_xz_rl, active_read_xz_rs, active_read_yz, active_read_yz_loc, active_read_yz_rl, active_read_yz_rs, active_write_1d, active_write_1d_rl, active_write_1d_rs, active_write_gen_rl, active_write_xy, active_write_xy_loc, active_write_xyz_loc, active_write_xz, active_write_xz_loc, active_write_xz_rl, active_write_xz_rs, active_write_yz, active_write_yz_loc, active_write_yz_rl, active_write_yz_rs, adactive_read_1d, adactive_read_gen_rl, adactive_read_gen_rs, adactive_read_xy, adactive_read_xy_loc, adactive_read_xyz, adactive_read_xyz_loc, adactive_read_xz, adactive_read_xz_loc, adactive_read_yz, adactive_read_yz_loc, adactive_write_1d, adactive_write_gen_rl, adactive_write_gen_rs, adactive_write_xy, adactive_write_xy_loc, adactive_write_xyz, adactive_write_xyz_loc, adactive_write_xz, adactive_write_xz_loc, adactive_write_yz, adactive_write_yz_loc, adautodiff_inadmode_set, adautodiff_inadmode_unset, adautodiff_whtapeio_sync, adclose, add_prefix, addamp_adj, addummy_for_etan, addummy_in_dynamics, addummy_in_stepping, admyactivefunction, adopen, adread, adread_i, adwrite, adwrite_i, adzero_adj, adzero_adj_1d, adzero_adj_loc, cg2d_mad, cg2d_store, copy_ad_uv_outp, copy_advar_outp, damp_adj, dummy_in_dynamics, dump_adj_xy, dump_adj_xy_uv, dump_adj_xyz, dump_adj_xyz_uv, g_active_read_1d, g_active_read_gen_rl, g_active_read_gen_rs, g_active_read_xy, g_active_read_xy_loc, g_active_read_xyz, g_active_read_xyz_loc, g_active_read_xz, g_active_read_xz_loc, g_active_read_yz, g_active_read_yz_loc, g_active_write_1d, g_active_write_gen_rl, g_active_write_gen_rs, g_active_write_xy, g_active_write_xy_loc, g_active_write_xyz, g_active_write_xyz_loc, g_active_write_xz, g_active_write_xz_loc, g_active_write_yz, g_active_write_yz_loc, g_autodiff_inadmode_set, g_autodiff_inadmode_unset, g_dummy_for_etan, g_dummy_in_dynamics, g_dummy_in_stepping, g_zero_adj, g_zero_adj_1d, g_zero_adj_loc, global_admax_r4, global_admax_r8, global_adsum_r4, global_adsum_r8, global_adsum_tile_rl, myactivefunction, zero_adj, zero_adj_1d, zero_adj_loc
- pkg/cd_code: cd_code_read_pickup
- pkg/cost: cost_accumulate_mean, cost_atlantic_heat, cost_final_restore, cost_final_store, cost_state_final, cost_test, cost_vector
- pkg/ctrl: adctrl_bound_2d, adctrl_bound_3d, ctrl_bound_2d, ctrl_bound_2d_tl, ctrl_bound_3d_tl, ctrl_convert_header, ctrl_cost_gen2d, ctrl_cost_gen3d, ctrl_cprlrl, ctrl_cprsrl, ctrl_depth_ini, ctrl_get_gen, ctrl_get_gen_rec, ctrl_get_mask2d, ctrl_getobcse, ctrl_getobcsn, ctrl_getobcss, ctrl_getobcsw, ctrl_init_obcs_variables, ctrl_init_rec, ctrl_map_genarr2d, ctrl_map_ini_gentim2d, ctrl_mask_set_xz, ctrl_mask_set_yz, ctrl_pack, ctrl_set_globfld_xy, ctrl_set_globfld_xz, ctrl_set_globfld_yz, ctrl_set_pack_xy, ctrl_set_pack_xyz, ctrl_set_pack_xz, ctrl_set_pack_yz, ctrl_set_unpack_xy, ctrl_set_unpack_xyz, ctrl_set_unpack_xz, ctrl_set_unpack_yz, ctrl_swapffields, ctrl_swapffields_3d, ctrl_swapffields_xz, ctrl_swapffields_yz, ctrl_unpack
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/generic_advdiff: gad_advection, gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_r, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_fill_cs_corner, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/gmredi: gmredi_calc_bates_k, gmredi_calc_eigs, gmredi_calc_geom, gmredi_calc_psi_bvp, gmredi_calc_qgleith, gmredi_calc_tensor_dummy, gmredi_calc_urms, gmredi_diagnostics_fill, gmredi_diagnostics_impl, gmredi_diagnostics_init, gmredi_mnc_init, gmredi_read_pickup, submeso_calc_psi
- pkg/grdchk: grdchk_get_obcs_mask, grdchk_getxx, grdchk_print, grdchk_setxx
- pkg/kpp: bldepth, blmix, enhance, kpp_calc, kpp_calc_diff_ptr, kpp_calc_diff_s, kpp_calc_diff_t, kpp_calc_visc, kpp_check, kpp_diagnostics_init, kpp_do_exch, kpp_doublediff, kpp_forcing_surf, kpp_init_fixed, kpp_init_varia, kpp_output, kpp_transport_ptr, kpp_transport_s, kpp_transport_t, kppmix, ri_iwmix, smooth_horiz, statekpp, wscale, z121
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/mom_common: mom_calc_3d_strain, mom_calc_absvort3, mom_calc_hdiv, mom_calc_relvort3, mom_calc_smag_3d, mom_calc_strain, mom_calc_tension, mom_calc_visc, mom_diagnostics_init, mom_hdissip, mom_quasihydrostatic, mom_u_coriolis_nh, mom_u_implicit_r, mom_u_metric_nh, mom_u_sidedrag, mom_uv_smag_3d, mom_v_coriolis_nh, mom_v_implicit_r, mom_v_metric_nh, mom_v_sidedrag, mom_visc_qgl_limit, mom_visc_qgl_stretch, mom_w_coriolis_nh, mom_w_metric_nh, mom_w_sidedrag, mom_w_smag_3d
- pkg/mom_fluxform: mom_u_coriolis, mom_u_del2u, mom_u_metric_cylinder, mom_uv_boundary, mom_v_coriolis, mom_v_del2v, mom_v_metric_cylinder
- pkg/mom_vecinv: mom_vecinv, mom_vi_coriolis, mom_vi_del2uv, mom_vi_hdissip, mom_vi_u_coriolis, mom_vi_u_coriolis_c4, mom_vi_u_grad_ke, mom_vi_u_vertshear, mom_vi_v_coriolis, mom_vi_v_coriolis_c4, mom_vi_v_grad_ke, mom_vi_v_vertshear
- pkg/monitor: admonitor, g_monitor, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, nlatbnd
- pkg/ptracers: adptracers_monitor, ptracers_ad_dump, ptracers_calc_wsurf_tr, ptracers_check_pickup, ptracers_debug, ptracers_diagnostics_init, ptracers_diagnostics_state, ptracers_dyn_state_data_dummy, ptracers_dyn_state_mod_dummy, ptracers_mnc_init, ptracers_read_pickup, ptracers_zonal_filt_apply
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs

