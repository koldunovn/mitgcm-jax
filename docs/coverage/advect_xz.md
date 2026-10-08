# Coverage: advect_xz (code) — M1 porting worklist

Written by `tools/coverage.py` (mitjax ab7e71b) from the gcov build `advect_xz-code-63cdc0b-7fd68e0-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 455 files from the link farm, 105 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input` | `job27827383` | 21 | 0 / 1 | 226 | yes | identical (924 lines) |
| `input.nlfs` | `job27827383` | 21 | 0 / 1 | 269 | yes | identical (924 lines) |
| `input.pqm` | `job27827383` | 21 | 0 / 1 | 217 | yes | identical (924 lines) |

## advect_xz/input

Run `$MJX_REFERENCE/coverage/advect_xz/input/job27827383` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/advect_xz/input/job27827383/gcov-9c7ed2d/gcov.json.gz`.

### Physics-active check

| check | measured | active |
|---|---|---|
| theta moved | 21 blocks, first 5.544981e-04, last 5.760638e-04, min 2.644319e-04, max 1.225871e-03 | yes |
| salt moved | 21 blocks, first 5.544981e-04, last 1.513123e-03, min 3.483437e-04, max 2.134593e-03 | yes |
| vertical advection routine (explicit or implicit) | gad_ppm_adv_r (400 calls), gad_som_adv_r (8000 calls) | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: doAB_onGtGs, dumpInitAndLast, fluidIsWater, momDissip_In_AB, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltMultiDimAdvec, saltSOM_Advection, saltStepping, snapshot_mdsio, tempAdvection, tempForcing, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useMultiDimAdvec, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, calc_wVelocity, debugMode, deepAtmosphere, doResetHFactors, doSaltClimRelax, doThetaClimRelax, exactConserv, fluidIsAir, globalFiles, implicitDiffusion, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momAdvection, momForcing, momImplVertAdv, momPressureForcing, momViscosity, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, saltImplVertAdv, saltIsActiveTr, staggerTimeStep, tempImplVertAdv, tempIsActiveTr, tempSOM_Advection, use3Dsolver, useCDscheme, useCoriolis, useCoupler, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSingleCpuInput, useSingleCpuIO, useSRCGSolver, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid, vectorInvariantMomentum
- selectors: pCellMix_select=0, saltVertAdvScheme=81, select_rStar=0, selectAddFluid=0, selectNHfreeSurf=0, selectPenetratingSW=0, selectSigmaCoord=0, tempVertAdvScheme=42

### Executed routines (226)

**eesupp/src** (75)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 19/28 | 138-139, 144, 150-151, 212-216 | 137:1/2, 143:1/2, 149:1/2, 178:1/2, 183:1/2, 195:1/2, 211:1/2 |
| `bar2` | bar2.F:58 | 2914 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 5 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 11361 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `comm_stats` | comm_stats.F:7 | 1 | 53/59 | 71-73, 100-102, 143-144 | 45:1/2, 69:1/2, 98:1/2, 115:1/2, 137:1/2 |
| `different_multiple` | different_multiple.F:7 | 802 | 14/15 | 45 | 44:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eedie` | eedie.F:7 | 1 | 9/18 | 40-42, 58-64 | 37:1/2, 54:1/2, 56:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 2806 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 24 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 400 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 2830 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 2806 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 2806 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 2806 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 2806 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 24 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 24 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 24 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 24 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_sm_3d_rl` | exch_sm_3d_rl.F:8 | 200 | 12/31 | 80-130 | 72:1/2 |
| `exch_uv_agrid_3d_rl` | exch_uv_agrid_3d_rl.F:8 | 600 | 14/40 | 85-153 | 77:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 5 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 201 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 1 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 12 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 603 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 36997 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 400 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_r8` | global_sum.F:97 | 42 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 781 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 13575 | 10/10 | - | - |
| `ilnblnk` | utils.F:123 | 19198 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 7 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 1097 | 6/6 | - | 33:1/2, 34:1/2 |
| `mds_flush` | mds_flush.F:7 | 2 | 3/3 | - | - |
| `mds_reclen` | mds_reclen.F:3 | 255 | 5/11 | 29, 34-39 | 28:1/2, 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 372 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 11320 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 45 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 2 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_list_i` | print.F:305 | 33 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 76 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 121 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_maprs` | print.F:704 | 5 | 139/210 | 864, 960-1017, 1023-1026, 1048-1051, 1085-1088, 1095, 1100-1101 | 835:4/6, 836:4/8, 837:5/8, 838:4/8, 839:4/8, 861:1/2, 886:1/4, 931:1/2, 1021:1/2, 1031:5/8, 1032:5/8, 1039:4/8, 1040:4/8, 1046:1/2, 1061:4/8, 1062:4/8, 1075:5/8, 1076:4/8, 1080:4/8, 1081:4/8, 1083:1/2, 1090:1/2, 1097:1/2, 1099:1/2, 1131:1/2 |
| `print_message` | print.F:23 | 2275 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 5211 | 69/159 | 486-512, 587, 595-730 | 227:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 286:1/2, 294:1/2, 485:1/2, 536:1/2 |
| `timer_get_time` | timers.F:743 | 5210 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 5211 | 12/12 | - | - |
| `timer_printall` | timers.F:857 | 1 | 3/3 | - | - |
| `timer_start` | timers.F:884 | 2605 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 2605 | 4/4 | - | - |
| `ucase` | utils.F:311 | 10422 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 2 | 18/20 | 629, 643 | 633:1/2, 639:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 25 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 76 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 78 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 43 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (77)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 8000 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 8000 | 14/49 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 800 | 21/87 | 164-166, 269-277, 287-406 | 83:1/2, 132:1/2, 266:1/2, 284:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 16000 | 26/26 | - | - |
| `config_check` | config_check.F:14 | 1 | 82/455 | 92-97, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 253-259, 366-371, 417-423, 442-448, 464-470, 484-490, 497-503, 510-516, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 252:1/2, 364:1/2, 404:1/2, 441:1/2, 463:1/2, 482:1/2, 495:1/2, 508:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 410/491 | 233, 240, 268-281, 307-322, 533-538, 554-588, 595, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 800 | 6/6 | - | - |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 200 | 5/14 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 200 | 14/15 | 86 | 49:1/2, 52:1/2, 53:1/2, 55:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 200 | 45/76 | 257-264, 557, 797-798, 811-845, 869-874, 882, 899, 934, 1116, 1119, 1123 | 248:1/2, 251:1/2, 256:1/2, 553:1/2, 574:1/2, 577:1/2, 728:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 878:1/2, 896:1/2, 932:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 201 | 6/13 | 97-110, 242 | 96:1/2, 116:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 200 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 200 | 3/109 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 200 | 34/64 | 75, 151-153, 161-163, 176, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 8000 | 9/55 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 58 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forward_step` | forward_step.F:70 | 200 | 59/119 | 467-485, 508-512, 767-769, 789-793, 838-840, 844, 868-870, 903-905, 913-915, 927-929, 948-950, 958-960, 979-1006, 1109-1111, 1176, 1201-1202 | 424:1/2, 465:1/2, 507:1/2, 530:1/2, 624:1/2, 654:1/2, 725:1/2, 730:1/2, 766:1/2, 786:1/2, 832:1/2, 842:1/2, 866:1/2, 897:1/2, 911:1/2, 924:1/2, 939:1/2, 952:1/2, 976:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/65 | 57-63, 84-112, 121-180, 191 | 55:1/2, 68:1/2, 72:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 47/47 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 9/12 | 37-38, 50 | 28:1/2, 49:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 31/49 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 193 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 192:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 18/68 | 90-182, 192, 211, 228-235 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 2 | 28/28 | - | 136:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 187/199 | 228, 247-261, 435 | 65:1/2, 196:1/2, 206:1/2, 227:1/2, 243:1/2, 418:1/2, 420:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 28/52 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 59/59 | - | 47:1/2, 157:1/2, 159:1/2, 161:1/2, 163:1/2, 165:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 405/876 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 537-541, 557-565, 577-580, 585-591, 609-612, 615-620, 623-625, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1148, 1151, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1314-1317, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1397-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 543:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/72 | 81-180 | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 25/38 | 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 26/47 | 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 21/22 | 61 | 55:1/2, 57:1/2, 60:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 30/40 | 302, 305-321, 326, 341, 345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 251:1/2, 301:1/2, 304:1/2, 325:1/2, 331:1/2, 338:1/2, 343:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 1 | 15/73 | 92-185, 211-221, 249-254, 279-282, 318, 329-331, 337-338 | 90:1/2, 207:1/2, 245:1/2, 277:1/2, 317:1/2, 321:1/2, 325:1/2, 336:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 40 | 19/44 | 95-117, 126-144, 152-168 | 93:1/2, 123:1/2, 150:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 200 | 19/27 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 172:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 200 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 82/82 | - | 100:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 66/126 | 132-140, 148-155, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 146:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 254:1/2, 260:1/2, 264:1/2, 269:1/2, 275:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 335:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 432:1/2, 436:1/2, 441:1/2, 447:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 8/14 | 170-176, 675-677 | 144:1/2, 167:1/2, 193:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 8/11 | 169, 174, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 6 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 3/3 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 1 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 6/7 | 225 | 223:1/2, 255:1/2 |
| `plot_field_xyrs` | plot_field.F:10 | 2 | 24/26 | 58-59 | 54:1/2, 56:1/2 |
| `plot_field_xyzrs` | plot_field.F:176 | 3 | 26/27 | 232 | 225:1/2, 227:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 400 | 50/69 | 167, 169, 192, 268-280, 326, 386-390, 397-399, 412-434, 492, 501, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 257:1/2, 259:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 479:1/2, 493:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 312/313 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 92/158 | 60-63, 67, 71, 79, 109-115, 120-126, 171-175, 186-189, 192-195, 200, 206, 212, 218, 224, 230, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 54/195 | 101-133, 140-165, 180-187, 212-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `temp_integrate` | temp_integrate.F:13 | 400 | 51/69 | 169, 171, 194, 261-269, 328, 388-392, 399-401, 414-436, 494, 503, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 259:1/2, 270:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 481:1/2, 495:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 15/15 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 22/22 | - | 606:1/2, 616:1/2, 743:1/2, 785:1/2, 794:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 200 | 34/63 | 133-136, 158, 207-239, 282, 389, 395-402 | 127:1/2, 132:1/2, 153:1/2, 206:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 800 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 200 | 5/6 | 117 | 115:1/2 |
| `update_surf_dr` | update_surf_dr.F:6 | 200 | 13/49 | 52-79, 88-115 | 49:1/2, 84:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 45/75 | 245-250, 253-257, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 201 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/diagnostics** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/372 | 123-653 | 109:1/2, 111:1/2 |

**pkg/generic_advdiff** (35)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 400 | 143/232 | 213-219, 253-266, 335-337, 351-364, 394, 414, 418, 422, 426, 430, 439-446, 461, 475-539, 615, 635, 639, 643, 647, 651, 660-667, 682, 696-761, 824-827, 844-845, 848-849, 866, 915, 991, 995, 999, 1003, 1007, 1018, 1081-1083 | 212:1/2, 252:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 419:1/2, 423:1/2, 427:1/2, 433:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 640:1/2, 644:1/2, 648:1/2, 654:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 884:1/2, 905:1/2, 911:1/2, 988:1/2, 992:1/2, 996:1/2, 1000:1/2, 1004:1/2, 1009:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 16000 | 60/177 | 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 606-608, 620, 646-647, 793 | 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 12/40 | 82-88, 99-106, 119-129, 178-181 | 39:1/2, 81:1/2, 95:1/2, 97:1/2, 118:1/2, 176:1/2 |
| `gad_diff_r` | gad_diff_r.F:7 | 16000 | 10/10 | - | - |
| `gad_exch_som` | gad_exch_som.F:6 | 200 | 9/9 | - | - |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 96/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 12/14 | 68-69 | 58:1/2, 60:1/2 |
| `gad_osc_hat_r` | gad_osc_hat_r.F:95 | 64800 | 4/4 | - | - |
| `gad_osc_hat_x` | gad_osc_hat_x.F:103 | 72000 | 4/4 | - | - |
| `gad_osc_loc_r` | gad_osc_hat_r.F:10 | 1684800 | 18/18 | - | - |
| `gad_osc_loc_x` | gad_osc_hat_x.F:10 | 1296000 | 20/20 | - | - |
| `gad_osc_mul_r` | gad_osc_mul_r.F:3 | 691380 | 23/23 | - | - |
| `gad_osc_mul_x` | gad_osc_mul_x.F:3 | 503514 | 23/23 | - | - |
| `gad_plm_fun_u` | gad_plm_fun.F:10 | 2304000 | 14/14 | - | - |
| `gad_ppm_adv_r` | gad_ppm_adv_r.F:3 | 400 | 31/33 | 148-150 | 104:1/2, 122:1/2 |
| `gad_ppm_adv_x` | gad_ppm_adv_x.F:3 | 8000 | 23/25 | 136-138 | 99:1/2, 109:1/2 |
| `gad_ppm_adv_y` | gad_ppm_adv_y.F:3 | 8000 | 17/25 | 101-132 | 99:1/2 |
| `gad_ppm_flx_r` | gad_ppm_flx_r.F:3 | 64800 | 22/22 | - | - |
| `gad_ppm_flx_x` | gad_ppm_flx_x.F:3 | 72000 | 24/26 | 79, 106 | 74:1/2, 101:1/2 |
| `gad_ppm_fun_mono` | gad_ppm_fun.F:46 | 2304000 | 30/30 | - | - |
| `gad_ppm_fun_null` | gad_ppm_fun.F:10 | 2304000 | 6/6 | - | - |
| `gad_ppm_hat_r` | gad_ppm_hat_r.F:3 | 64800 | 23/26 | 80, 85-88 | 77:1/2, 83:1/2, 91:1/2 |
| `gad_ppm_hat_x` | gad_ppm_hat_x.F:3 | 72000 | 23/26 | 81, 86-89 | 78:1/2, 84:1/2, 92:1/2 |
| `gad_ppm_p3e_r` | gad_ppm_p3e_r.F:3 | 64800 | 14/14 | - | - |
| `gad_ppm_p3e_x` | gad_ppm_p3e_x.F:3 | 72000 | 14/14 | - | - |
| `gad_som_adv_r` | gad_som_adv_r.F:13 | 8000 | 99/113 | 268-285 | 158:1/2, 261:1/2 |
| `gad_som_adv_x` | gad_som_adv_x.F:13 | 8000 | 94/103 | 144-153 | 134:1/2, 142:1/2, 157:1/2, 158:1/2, 164:1/2 |
| `gad_som_adv_y` | gad_som_adv_y.F:13 | 8000 | 94/103 | 144-153 | 134:1/2, 142:1/2, 157:1/2, 158:1/2, 164:1/2 |
| `gad_som_advect` | gad_som_advect.F:14 | 400 | 104/161 | 176-182, 216-219, 222-225, 230-243, 318-331, 361, 386, 408, 433, 443-455, 462-463, 466-467, 484, 577-580, 586-591, 633, 660-662 | 175:1/2, 215:1/2, 221:1/2, 229:1/2, 316:1/2, 356:1/2, 366:1/2, 403:1/2, 413:1/2, 441:1/2, 461:1/2, 465:1/2, 482:1/2, 496:1/2, 575:1/2, 584:1/2, 611:1/2, 659:1/2 |
| `gad_som_exchanges` | gad_som_exchanges.F:6 | 200 | 5/6 | 36 | 35:1/2, 40:1/2 |
| `gad_som_lim_r` | gad_som_lim_r.F:7 | 400 | 15/15 | - | 60:1/2 |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 12/20 | 72-83, 90 | 70:1/2, 87:1/2, 89:1/2 |

**pkg/mdsio** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 105 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 21 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 4 | 65/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 244-253, 265-365, 435, 450-495, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 220:1/2, 230:1/2, 232:1/2, 233:1/2, 243:1/2, 262:1/2, 377:1/2, 380:1/2, 392:1/2, 434:1/2, 437:1/4, 506:1/2, 526:1/2, 540:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 1 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 122 | 82/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 374-385, 396-406, 425-434, 474-484, 512, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 248:1/2, 253:1/2, 269:1/2, 373:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 511:1/2, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 239 | 39/52 | 108, 135-139, 146-147, 152, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 151:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/monitor** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 42 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 21 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 21 | 13/13 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 126 | 71/71 | - | 93:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 21 | 52/125 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 924 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 21 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 903 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 107 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 21 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 126 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 201 | 54/67 | 57, 119-120, 135-145, 150-153 | 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 3 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 21 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 65 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 16 | 7/7 | - | - |

### Compiled but not executed (523 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rl, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_int, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, master_cpu_thread, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, print_error, print_maprl, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth2, adams_bashforth3, analylic_theta, apply_forcing_u, apply_forcing_v, calc_div_ghat, calc_eddy_stress, calc_grad_phi_fv, calc_grad_phi_hyd, calc_grad_phi_surf, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_phi_hyd, calc_r_star, calc_surf_dr, calc_viscosity, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, correction_step, cycle_ab_tracer, diags_oceanic_surf_flux, diags_phi_hyd, diags_phi_rlow, diags_rho_g, diags_rho_l, diags_sound_speed, do_stagger_fields_exchanges, do_statevars_diags, dynamics, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, forcing_surf_relax, freesurf_rescale_g, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, momentum_correction_step, packages_error_msg, plot_field_xyrl, plot_field_xyzrl, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rl, rotate_uv2en_rs, solve_for_pressure, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, swfrac, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep, timestep_wvel, tracers_iigw_correction, turnoff_model_io, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_is_on, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_y, gad_osc_loc_y, gad_osc_mul_y, gad_plm_fun_v, gad_ppm_flx_y, gad_ppm_hat_y, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_fill_cs_corner, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, quadroot, salt_fill
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_pass_r4torl, mds_pass_r4tors, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/monitor: admonitor, g_monitor, mon_calc_advcfl_glob, mon_calc_advcfl_tile, mon_calc_stats_rs, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, mon_surfcor, mon_vort3, mon_writestats_rs, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rl, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs

## advect_xz/input.nlfs

Run `$MJX_REFERENCE/coverage/advect_xz/input.nlfs/job27827383` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/advect_xz/input.nlfs/job27827383/gcov-9c7ed2d/gcov.json.gz`.

### Physics-active check

| check | measured | active |
|---|---|---|
| theta moved | 21 blocks, first 5.544981e-04, last 3.823900e-04, min 2.892337e-04, max 1.054369e-03 | yes |
| salt moved | 21 blocks, first 5.544981e-04, last 3.440128e-04, min 2.146531e-04, max 7.383391e-04 | yes |
| vertical advection routine (explicit or implicit) | gad_fluxlimit_impl_r (7600 calls), gad_u3c4_impl_r (7600 calls) | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: AdamsBashforthGs, calc_wVelocity, doAB_onGtGs, dumpInitAndLast, fluidIsWater, momDissip_In_AB, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltStepping, snapshot_mdsio, tempAdvection, tempForcing, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useMultiDimAdvec, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGt, applyExchUV_early, debugMode, deepAtmosphere, diag_mnc, doResetHFactors, doSaltClimRelax, doThetaClimRelax, dumpAtLast, fluidIsAir, globalFiles, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momAdvection, momForcing, momImplVertAdv, momPressureForcing, momViscosity, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, saltIsActiveTr, saltMultiDimAdvec, saltSOM_Advection, tempIsActiveTr, tempSOM_Advection, use3Dsolver, useCDscheme, useCoriolis, useCoupler, useMin4hFacEdges, useMissingValue, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSingleCpuInput, useSingleCpuIO, useSRCGSolver, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid, vectorInvariantMomentum
- selectors: pCellMix_select=0, saltVertAdvScheme=3, selectAddFluid=0, selectNHfreeSurf=0, selectPenetratingSW=0, selectSigmaCoord=0, tempVertAdvScheme=77

### Executed routines (269)

**eesupp/src** (76)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 19/28 | 138-139, 144, 150-151, 212-216 | 137:1/2, 143:1/2, 149:1/2, 178:1/2, 183:1/2, 195:1/2, 211:1/2 |
| `bar2` | bar2.F:58 | 75778 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 5 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 7449 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `comm_stats` | comm_stats.F:7 | 1 | 53/59 | 71-73, 100-102, 143-144 | 45:1/2, 69:1/2, 98:1/2, 115:1/2, 137:1/2 |
| `diff_phase_multiple` | diff_phase_multiple.F:7 | 600 | 15/16 | 46 | 44:1/2, 45:1/2, 50:1/2 |
| `different_multiple` | different_multiple.F:7 | 802 | 14/15 | 45 | 44:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eedie` | eedie.F:7 | 1 | 9/18 | 40-42, 58-64 | 37:1/2, 54:1/2, 56:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1812 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 25 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch_3d_rl` | exch_3d_rl.F:8 | 200 | 11/12 | 59 | 55:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1837 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1812 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1812 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1812 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1812 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 25 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 25 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 25 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 25 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_uv_3d_rl` | exch_uv_3d_rl.F:11 | 200 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xy_rl` | exch_uv_xy_rl.F:11 | 202 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 5 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 403 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 13 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 403 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 98125 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 14960 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_r8` | global_sum.F:97 | 21882 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 781 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 18378 | 10/10 | - | - |
| `ilnblnk` | utils.F:123 | 24799 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 6 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 1139 | 6/6 | - | 33:1/2, 34:1/2 |
| `mds_flush` | mds_flush.F:7 | 2 | 3/3 | - | - |
| `mds_reclen` | mds_reclen.F:3 | 319 | 5/11 | 29, 34-39 | 28:1/2, 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 408 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 7348 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 76 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 3 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_list_i` | print.F:305 | 35 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 79 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 123 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_maprs` | print.F:704 | 5 | 139/210 | 864, 960-1017, 1023-1026, 1048-1051, 1085-1088, 1095, 1100-1101 | 835:4/6, 836:4/8, 837:5/8, 838:4/8, 839:4/8, 861:1/2, 886:1/4, 931:1/2, 1021:1/2, 1031:5/8, 1032:5/8, 1039:4/8, 1040:4/8, 1046:1/2, 1061:4/8, 1062:4/8, 1075:5/8, 1076:4/8, 1080:4/8, 1081:4/8, 1083:1/2, 1090:1/2, 1097:1/2, 1099:1/2, 1131:1/2 |
| `print_message` | print.F:23 | 2447 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 7611 | 69/159 | 486-512, 587, 595-730 | 227:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 286:1/2, 294:1/2, 485:1/2, 536:1/2 |
| `timer_get_time` | timers.F:743 | 7610 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 7611 | 12/12 | - | - |
| `timer_printall` | timers.F:857 | 1 | 3/3 | - | - |
| `timer_start` | timers.F:884 | 3805 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 3805 | 4/4 | - | - |
| `ucase` | utils.F:311 | 15221 | 7/8 | 333 | 332:1/2 |
| `write_0d_c` | write_utils.F:579 | 2 | 18/20 | 629, 643 | 633:1/2, 639:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 27 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 79 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 80 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 43 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (87)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `adams_bashforth2` | adams_bashforth2.F:6 | 8000 | 13/18 | 71-75 | 69:1/2 |
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 8000 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 8000 | 14/49 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 800 | 21/87 | 164-166, 269-277, 287-406 | 83:1/2, 132:1/2, 266:1/2, 284:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 16000 | 26/26 | - | - |
| `calc_r_star` | calc_r_star.F:10 | 202 | 60/116 | 68, 107, 123, 137-163, 186, 189, 192, 195-196, 203-242, 247-252, 316-318 | 66:1/2, 73:1/2, 102:1/2, 111:1/2, 115:1/2, 129:1/2, 185:1/2, 188:1/2, 191:1/2, 194:1/2, 201:1/2, 246:1/2, 315:1/2, 336:1/2 |
| `config_check` | config_check.F:14 | 1 | 82/455 | 92-97, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 253-259, 366-371, 417-423, 442-448, 464-470, 484-490, 497-503, 510-516, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 252:1/2, 364:1/2, 404:1/2, 441:1/2, 463:1/2, 482:1/2, 495:1/2, 508:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 410/491 | 233, 240, 268-281, 307-322, 533-538, 554-588, 595, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 800 | 6/6 | - | - |
| `diags_oceanic_surf_flux` | diags_oceanic_surf_flux.F:7 | 200 | 26/47 | 56, 121-152, 167-190 | 55:1/2, 81:1/2, 94:1/2, 115:1/2, 161:1/2 |
| `diags_rho_g` | diags_rho.F:100 | 200 | 8/33 | 161-171, 178-188, 195-213 | 160:1/2, 177:1/2, 194:1/2 |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 200 | 5/14 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 200 | 9/15 | 53-56, 79, 86 | 49:1/2, 52:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 200 | 51/76 | 258, 260, 262, 264, 557, 797-798, 811-845, 869-874, 882, 899, 934, 1123 | 248:1/2, 251:1/2, 256:1/2, 257:1/2, 259:1/2, 261:1/2, 263:1/2, 553:1/2, 574:1/2, 577:1/2, 728:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 878:1/2, 896:1/2, 932:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_stagger_fields_exchanges` | do_stagger_fields_exchanges.F:6 | 200 | 9/11 | 49-50 | 32:1/2, 37:1/2, 38:1/2, 41:1/2, 46:1/2 |
| `do_statevars_diags` | do_statevars_diags.F:8 | 600 | 11/12 | 64 | 54:1/2, 60:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 201 | 7/13 | 97-110 | 96:1/2, 116:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 200 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 200 | 3/109 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 200 | 37/64 | 75, 151-153, 176, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 8000 | 9/55 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 58 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forward_step` | forward_step.F:70 | 200 | 78/119 | 467-485, 730-734, 767-769, 789-793, 842-853, 868-870, 903-905, 913-915, 952-960, 980, 1176, 1201-1202 | 424:1/2, 465:1/2, 507:1/2, 530:1/2, 624:1/2, 654:1/2, 725:1/2, 766:1/2, 786:1/2, 832:1/2, 866:1/2, 897:1/2, 911:1/2, 924:1/2, 939:1/2, 976:1/2, 979:1/2, 988:1/2, 1002:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `freesurf_rescale_g` | freesurf_rescale_g.F:6 | 24000 | 7/15 | 49-66 | 39:1/2, 40:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/65 | 57-63, 84-112, 121-180, 191 | 55:1/2, 68:1/2, 72:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 47/47 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 9/12 | 37-38, 50 | 28:1/2, 49:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 31/49 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 193 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 192:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 18/68 | 90-182, 192, 211, 228-235 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 2 | 28/28 | - | 136:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 187/199 | 228, 247-261, 435 | 65:1/2, 196:1/2, 206:1/2, 227:1/2, 243:1/2, 418:1/2, 420:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 28/52 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 59/59 | - | 47:1/2, 157:1/2, 159:1/2, 161:1/2, 163:1/2, 165:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 404/876 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 537-541, 557-565, 577-580, 585-591, 609-612, 615-620, 628-632, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1148, 1151, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1314-1317, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1397-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 543:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/72 | 81-180 | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 25/38 | 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 26/47 | 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 21/22 | 61 | 55:1/2, 57:1/2, 60:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 34/40 | 309-321, 343-345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 251:1/2, 301:1/2, 304:1/2, 305:1/2, 325:1/2, 331:1/2, 338:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 201 | 57/73 | 146-160, 212-214, 279-282, 337-338 | 90:1/2, 93:1/2, 95:1/2, 142:1/2, 211:1/2, 245:1/2, 277:1/2, 325:1/2, 329:1/2, 336:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 8040 | 18/44 | 95-117, 150-193 | 93:1/2, 123:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 200 | 20/27 | 149, 154-164 | 105:1/2, 148:1/2, 153:1/2, 172:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 200 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 82/82 | - | 100:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 66/126 | 132-140, 148-155, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 146:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 254:1/2, 260:1/2, 264:1/2, 269:1/2, 275:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 335:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 432:1/2, 436:1/2, 441:1/2, 447:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 14/14 | - | 144:1/2, 167:1/2, 170:1/2, 174:1/2, 193:1/2, 673:1/2, 675:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 9/11 | 169, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 6 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 3/3 | - | - |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 7/7 | - | 223:1/2, 255:1/2 |
| `plot_field_xyrs` | plot_field.F:10 | 2 | 24/26 | 58-59 | 54:1/2, 56:1/2 |
| `plot_field_xyzrs` | plot_field.F:176 | 3 | 26/27 | 232 | 225:1/2, 227:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 400 | 56/69 | 192, 259-267, 273-280, 326, 390, 397-399, 493-501, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 257:1/2, 268:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 389:1/2, 396:1/2, 408:1/2, 413:1/2, 479:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 312/313 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 92/158 | 60-63, 67, 71, 79, 109-115, 120-126, 171-175, 186-189, 192-195, 200, 206, 212, 218, 224, 230, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 54/195 | 101-133, 140-165, 180-187, 212-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `solve_pentadiagonal` | solve_pentadiagonal.F:10 | 400 | 54/58 | 387-390 | 381:1/2 |
| `solve_tridiagonal` | solve_tridiagonal.F:10 | 400 | 32/38 | 249-251, 266-268 | 244:1/2, 259:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `temp_integrate` | temp_integrate.F:13 | 400 | 54/69 | 171, 194, 261-269, 328, 388-392, 399-401, 436, 495-503, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 259:1/2, 270:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 415:1/2, 481:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 15/15 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 22/22 | - | 606:1/2, 616:1/2, 743:1/2, 785:1/2, 794:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 200 | 35/63 | 133-136, 158, 218-250, 282, 389, 395-402 | 127:1/2, 132:1/2, 153:1/2, 206:1/2, 207:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 800 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 200 | 5/6 | 117 | 115:1/2 |
| `update_cg2d` | update_cg2d.F:7 | 1 | 40/48 | 58, 132-140, 169, 175, 182 | 57:1/2, 61:1/2, 131:1/2, 159:1/2, 168:1/2, 174:1/2, 181:1/2 |
| `update_etah` | update_etah.F:7 | 201 | 12/16 | 62-66, 86 | 54:1/2, 84:1/2 |
| `update_r_star` | update_r_star.F:6 | 201 | 17/29 | 84-111 | 48:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 50/75 | 245-250, 253-257, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 252:1/2, 260:1/2, 263:1/2, 264:1/2, 265:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 201 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/diagnostics** (46)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_addtolist` | diagnostics_addtolist.F:8 | 127 | 17/44 | 62, 68-92, 111-117 | 60:1/2, 67:1/2, 99:1/2, 103:1/2, 108:1/2, 125:1/2 |
| `diagnostics_check` | diagnostics_check.F:8 | 1 | 21/150 | 44-47, 51-54, 59-64, 85-89, 92-96, 103-113, 120-130, 145-213, 225-250, 261-268, 277-280 | 37:1/2, 43:1/2, 50:1/2, 57:1/2, 84:1/2, 91:1/2, 101:1/2, 102:2/2, 118:1/2, 119:2/2, 144:1/2, 224:1/2, 260:1/2, 275:1/2 |
| `diagnostics_clear` | diagnostics_clear.F:6 | 10 | 8/8 | - | 32:1/2 |
| `diagnostics_clrdiag` | diagnostics_clear.F:51 | 40 | 10/10 | - | - |
| `diagnostics_cumulate` | diagnostics_fill_field.F:413 | 32000 | 12/62 | 478-517, 521-556, 561-578, 594-602 | 475:1/2, 477:1/2, 520:1/2, 559:1/2, 583:1/2 |
| `diagnostics_fill` | diagnostics_fill.F:6 | 61640 | 31/34 | 75, 130-131 | 73:1/2, 94:1/2, 129:1/2 |
| `diagnostics_fill_field` | diagnostics_fill_field.F:13 | 32000 | 46/101 | 108-110, 132-141, 146-151, 162-163, 168-169, 173-177, 182-183, 185-186, 195, 203-216, 222-242, 261-268 | 105:1/6, 107:1/2, 128:1/2, 145:1/2, 158:1/2, 167:1/2, 170:1/2, 181:1/2, 184:1/2, 192:1/2, 194:1/2, 202:1/2, 220:1/2, 252:1/2 |
| `diagnostics_fill_rs` | diagnostics_fill_rs.F:6 | 1000 | 14/34 | 75, 84-85, 93-103, 118-146 | 73:1/2, 80:1/2, 92:1/2, 117:1/2 |
| `diagnostics_fill_state` | diagnostics_fill_state.F:6 | 600 | 69/287 | 112-139, 145-166, 170-183, 187-203, 207-223, 229-241, 245-257, 261-274, 278-290, 294-306, 310-323, 327-339, 343-355, 359-380, 398-410, 423-435, 502, 537-550, 556-568, 572-584, 590-603, 607-620, 624-637, 641-654, 658-671, 675-688, 714-724, 739-749 | 93:1/2, 111:1/2, 143:1/2, 169:1/2, 186:1/2, 206:1/2, 228:1/2, 244:1/2, 260:1/2, 277:1/2, 293:1/2, 309:1/2, 326:1/2, 342:1/2, 358:1/2, 393:1/2, 418:1/2, 451:1/2, 501:1/2, 503:1/2, 534:1/2, 555:1/2, 571:1/2, 589:1/2, 606:1/2, 623:1/2, 640:1/2, 657:1/2, 674:1/2, 709:1/2, 734:1/2 |
| `diagnostics_get_diag` | diagnostics_utils.F:97 | 1600 | 16/26 | 141-142, 177-186 | 137:1/2, 139:1/2, 147:1/2, 149:1/2, 154:1/2 |
| `diagnostics_ini_io` | diagnostics_ini_io.F:6 | 1 | 4/12 | 45-54 | 39:1/2, 42:1/2 |
| `diagnostics_init_early` | diagnostics_init_early.F:8 | 1 | 107/107 | - | 71:1/2 |
| `diagnostics_init_fixed` | diagnostics_init_fixed.F:8 | 1 | 8/8 | - | - |
| `diagnostics_init_varia` | diagnostics_init_varia.F:8 | 1 | 24/24 | - | 30:1/2 |
| `diagnostics_is_on` | diagnostics_is_on.F:9 | 12400 | 16/16 | - | 45:1/2 |
| `diagnostics_main_init` | diagnostics_main_init.F:8 | 1 | 514/533 | 95-97, 104-110, 112-117, 128-129, 131-132, 285 | 94:1/2, 103:1/2, 111:1/2, 127:1/2, 130:1/2, 201:1/2, 248:1/2, 284:1/2, 621:1/2, 633:1/2, 779:1/2 |
| `diagnostics_out` | diagnostics_out.F:8 | 10 | 78/158 | 95, 100-102, 112-113, 130, 175, 177-184, 203-208, 211-244, 258-267, 277-278, 282-284, 308-336, 347-357, 365, 370-382, 410 | 92:1/2, 99:1/2, 107:1/2, 110:1/2, 129:1/2, 141:1/2, 173:1/2, 176:1/2, 190:1/2, 195:1/2, 196:1/2, 199:1/2, 209:1/2, 255:1/2, 256:1/2, 275:1/2, 280:1/2, 294:1/2, 307:1/2, 345:1/2, 360:1/2, 367:1/2, 392:1/2, 398:1/2, 399:1/2, 401:1/2, 438:1/2 |
| `diagnostics_read_pickup` | diagnostics_read_pickup.F:7 | 1 | 2/2 | - | - |
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 245/372 | 111-119, 280-282, 293, 295, 298-300, 302-309, 311-312, 322, 327-342, 357-366, 372-381, 421-424, 429-435, 446-447, 454-467, 471-478, 493-502, 508-517, 538-540, 582-583, 587-588, 592-604 | 109:1/2, 123:1/2, 168:1/2, 279:1/2, 292:1/2, 294:1/2, 297:1/2, 301:1/2, 310:1/2, 313:1/2, 321:1/2, 326:1/2, 356:1/2, 371:1/2, 420:1/2, 428:1/2, 444:1/2, 453:1/2, 469:1/2, 481:1/2, 492:1/2, 507:1/2, 536:1/2, 570:1/2, 586:1/2, 589:1/2, 607:2/4, 609:1/4, 629:1/2, 631:1/2, 635:1/4, 638:1/4 |
| `diagnostics_scale_fill` | diagnostics_scale_fill.F:6 | 3400 | 15/33 | 78, 95-105, 120-146 | 76:1/2, 94:1/2, 119:1/2 |
| `diagnostics_scale_fill_rs` | diagnostics_scale_fill_rs.F:6 | 1200 | 13/33 | 78, 86-87, 95-105, 120-148 | 76:1/2, 82:1/2, 94:1/2, 119:1/2 |
| `diagnostics_set_calc` | diagnostics_set_calc.F:9 | 1 | 7/80 | 78-194 | 46:1/2 |
| `diagnostics_set_levels` | diagnostics_set_levels.F:7 | 1 | 75/133 | 88, 95-115, 119-122, 131-134, 138-141, 148-151, 156-160, 164-167, 218-220, 226-228, 244-253 | 73:1/2, 87:1/2, 93:1/2, 118:1/2, 130:1/2, 137:1/2, 147:1/2, 155:1/2, 163:1/2, 183:1/2, 217:1/2, 221:1/2, 241:1/2, 243:1/2 |
| `diagnostics_set_pointers` | diagnostics_set_pointers.F:6 | 1 | 75/160 | 59-63, 70-96, 101-109, 145-151, 166-186, 206-207, 213-217, 248-251, 254-260, 278-302 | 41:1/2, 57:1/2, 58:1/2, 69:1/2, 100:1/2, 142:1/2, 158:1/2, 194:1/2, 202:1/2, 212:1/2, 239:1/2, 246:1/2, 252:1/2, 271:1/2, 272:2/4, 274:1/4 |
| `diagnostics_setdiag` | diagnostics_setdiag.F:6 | 4 | 35/89 | 66-72, 92-94, 109-111, 118-126, 133-134, 142-145, 156, 159-160, 165-210 | 65:1/2, 87:1/2, 90:1/2, 101:1/2, 107:1/2, 114:1/2, 115:1/2, 131:1/2, 155:1/2, 158:1/2, 163:1/2 |
| `diagnostics_summary` | diagnostics_summary.F:7 | 1 | 83/121 | 68-87, 118-120, 136-159, 168-170, 229-232 | 55:1/2, 60:1/2, 61:1/2, 90:1/2, 117:1/2, 123:2/4, 125:1/4, 135:1/2, 162:1/2, 164:2/4, 167:1/2, 207:1/2, 208:2/4, 221:1/2, 223:2/4, 228:1/2, 243:1/2 |
| `diagnostics_switch_onoff` | diagnostics_switch_onoff.F:11 | 200 | 36/74 | 81, 101-135, 146-173, 220-221, 233-234 | 77:1/2, 87:1/2, 98:1/2, 144:1/2, 185:1/2, 216:1/2, 218:1/2, 229:1/2, 231:1/2 |
| `diagnostics_write` | diagnostics_write.F:3 | 201 | 43/50 | 69-70, 93-94, 120-121, 152 | 62:1/2, 91:1/2, 92:1/2, 110:1/2, 139:1/2, 151:1/2, 161:1/2 |
| `diagnostics_write_pickup` | diagnostics_write_pickup.F:7 | 1 | 3/3 | - | - |
| `diags_mk_title` | diagnostics_utils.F:593 | 37 | 17/23 | 643-650 | 632:1/2, 635:1/2, 642:1/2 |
| `diags_mk_units` | diagnostics_utils.F:514 | 43 | 13/30 | 554-568, 575-582 | 547:1/2, 552:1/2, 574:1/2 |
| `diags_renamed` | diagnostics_utils.F:661 | 200 | 8/19 | 695-700, 708-714 | 694:1/2, 702:1/2, 703:1/2, 705:1/2 |
| `diags_track_diva` | diagnostics_utils.F:410 | 1 | 5/9 | 448-452 | 444:1/2 |
| `diagstats_ascii_out` | diagstats_ascii_out.F:8 | 440 | 19/19 | - | 43:1/2, 45:1/2, 55:1/2, 75:1/4 |
| `diagstats_calc` | diagstats_calc.F:6 | 14560 | 43/67 | 99-102, 116-121, 125-130, 153-155, 161-163, 167-170, 174-177 | 98:1/2, 115:1/2, 124:1/2, 152:1/2, 160:1/2, 166:1/2, 173:1/2 |
| `diagstats_clear` | diagstats_clear.F:8 | 40 | 7/7 | - | 30:1/2 |
| `diagstats_close_io` | diagstats_close_io.F:6 | 1 | 16/18 | 55, 71 | 40:1/2, 42:1/2, 52:1/2, 70:1/2 |
| `diagstats_clrdiag` | diagstats_clear.F:46 | 440 | 8/8 | - | - |
| `diagstats_fill` | diagstats_fill.F:13 | 6680 | 48/78 | 121-122, 136-141, 163-167, 173-175, 188, 196-209 | 120:1/2, 135:1/2, 160:1/2, 172:1/2, 187:1/2, 195:1/2 |
| `diagstats_global` | diagstats_global.F:8 | 440 | 53/87 | 87-96, 115-116, 127-129, 151-159, 171-172, 188-206 | 61:1/2, 63:1/2, 86:1/2, 113:1/2, 126:1/2, 136:1/2, 150:1/2, 168:1/2, 170:1/2 |
| `diagstats_ini_io` | diagstats_ini_io.F:6 | 1 | 36/37 | 54 | 40:1/2, 42:1/2, 51:1/2, 76:1/2, 77:2/4, 83:1/4, 85:1/4, 87:2/4, 89:1/4 |
| `diagstats_local` | diagstats_local.F:6 | 14560 | 34/46 | 112-114, 146, 172, 185, 201, 212, 234-235, 242-243 | 106:1/2, 111:1/2, 121:1/2, 136:1/2, 160:1/2, 173:1/2, 191:1/2, 202:1/2, 230:1/2, 237:1/2 |
| `diagstats_output` | diagstats_output.F:8 | 40 | 24/45 | 72, 88-111, 123-125 | 66:1/2, 69:1/2, 78:1/2, 87:1/2, 115:1/2, 116:1/2, 121:1/2, 133:1/2 |
| `diagstats_set_pointers` | diagstats_set_pointers.F:6 | 1 | 37/84 | 72-78, 82-95, 98-105, 120-140, 150-156, 162-165 | 40:1/2, 68:1/2, 80:1/2, 97:1/2, 112:1/2, 148:1/2, 149:2/2, 161:1/2 |
| `diagstats_set_regions` | diagstats_set_regions.F:6 | 1 | 17/21 | 209-212 | 200:1/2, 203:1/2, 208:1/2 |
| `diagstats_setdiag` | diagstats_setdiag.F:6 | 11 | 20/51 | 70-72, 84-85, 95-97, 104-141 | 65:1/2, 68:1/2, 81:1/2, 82:1/2, 103:1/2 |

**pkg/generic_advdiff** (21)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 400 | 100/232 | 253-266, 335-337, 351-364, 394, 414, 419-446, 461, 475-539, 615, 635, 640-667, 682, 696-761, 832-837, 848-849, 866, 880-1083 | 212:1/2, 252:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 16000 | 97/177 | 213-216, 236-244, 257, 262, 276, 280-296, 340, 386, 391, 405, 409-425, 469, 515-592, 614, 620, 646-647, 793 | 191:1/2, 199:1/2, 212:1/2, 229:1/2, 256:1/2, 259:1/2, 273:1/2, 277:1/2, 314:1/2, 339:1/2, 385:1/2, 388:1/2, 402:1/2, 406:1/2, 443:1/2, 468:1/2, 513:1/2, 605:1/2, 617:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 11/40 | 82-88, 97-106, 119-129, 178-181 | 39:1/2, 81:1/2, 95:1/2, 118:1/2, 176:1/2 |
| `gad_diag_sufx` | gad_diagnostics_init.F:324 | 17602 | 6/7 | 368 | 356:1/2 |
| `gad_diagnostics_init` | gad_diagnostics_init.F:6 | 1 | 177/183 | 53, 60, 181-185 | 52:1/2, 59:1/2, 180:1/2 |
| `gad_diagnostics_state` | gad_diagnostics_state.F:8 | 200 | 17/33 | 64-93, 109-137 | 63:1/2, 108:1/2 |
| `gad_diff_x` | gad_diff_x.F:7 | 8000 | 6/6 | - | - |
| `gad_diff_y` | gad_diff_y.F:7 | 8000 | 7/7 | - | - |
| `gad_fluxlimit_adv_x` | gad_fluxlimit_adv_x.F:7 | 8000 | 22/22 | - | 78:1/2 |
| `gad_fluxlimit_adv_y` | gad_fluxlimit_adv_y.F:7 | 8000 | 20/22 | 85, 92 | 78:1/2, 84:1/2, 89:1/2 |
| `gad_fluxlimit_impl_r` | gad_fluxlimit_impl_r.F:6 | 7600 | 29/29 | - | 81:1/2 |
| `gad_implicit_r` | gad_implicit_r.F:6 | 800 | 101/132 | 176-179, 215-219, 222-227, 243-250, 270, 281-284, 339-341, 395-422 | 121:1/2, 162:1/2, 167:1/2, 214:1/2, 221:1/2, 236:1/2, 269:1/2, 272:1/2, 280:1/2, 289:1/2, 293:1/2, 299:1/2, 304:1/2, 338:1/2, 350:1/2, 386:1/2, 388:1/2 |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 99/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 151, 159-162, 180 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 11/14 | 60-69 | 58:1/2 |
| `gad_u3_adv_x` | gad_u3_adv_x.F:7 | 8000 | 14/14 | - | - |
| `gad_u3_adv_y` | gad_u3_adv_y.F:7 | 8000 | 14/14 | - | - |
| `gad_u3c4_impl_r` | gad_u3c4_impl_r.F:6 | 7600 | 29/37 | 136-138, 142-147 | 83:1/2, 135:1/2, 139:1/2 |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 6/20 | 72-83, 89-100 | 70:1/2, 87:1/2 |

**pkg/mdsio** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 137 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 21 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 4 | 65/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 244-253, 265-365, 435, 450-495, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 220:1/2, 230:1/2, 232:1/2, 233:1/2, 243:1/2, 262:1/2, 377:1/2, 380:1/2, 392:1/2, 434:1/2, 437:1/4, 506:1/2, 526:1/2, 540:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 11 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 154 | 82/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 374-385, 396-406, 425-434, 474-484, 512, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 248:1/2, 253:1/2, 269:1/2, 373:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 511:1/2, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 241 | 40/52 | 108, 135-139, 146-147, 152, 157-159, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 151:1/2, 153:1/2, 207:1/4, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/monitor** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 42 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 21 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 21 | 13/13 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 126 | 71/71 | - | 93:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 21 | 52/125 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 924 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 21 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 903 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 107 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 21 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 126 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 201 | 54/67 | 57, 119-120, 135-145, 150-153 | 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (14)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 3 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 21 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 65 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 8 | 7/7 | - | - |
| `write_rec_lev_rl` | write_rec.F:530 | 40 | 7/7 | - | - |

### Compiled but not executed (480 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, cumulsum_z_tile_rl, date, eedata_example, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rl, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_int, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, master_cpu_thread, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, print_error, print_maprl, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth3, analylic_theta, apply_forcing_u, apply_forcing_v, calc_div_ghat, calc_eddy_stress, calc_grad_phi_fv, calc_grad_phi_hyd, calc_grad_phi_surf, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_phi_hyd, calc_surf_dr, calc_viscosity, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, correction_step, cycle_ab_tracer, diags_phi_hyd, diags_phi_rlow, diags_rho_l, diags_sound_speed, dynamics, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, forcing_surf_relax, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, momentum_correction_step, packages_error_msg, packages_unused_msg, plot_field_xyrl, plot_field_xyzrl, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rl, rotate_uv2en_rs, solve_for_pressure, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, swfrac, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep, timestep_wvel, tracers_iigw_correction, turnoff_model_io, update_etaws, update_masks_etc, update_sigma, update_surf_dr
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_calc_phivel, diagnostics_count, diagnostics_fract_fill, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_list_check, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_write_adj, diags_get_parms_i, diagstats_g_calc, diagstats_lm_calc, diagstats_mnc_out
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diff_r, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_grad_x, gad_grad_y, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_r, gad_osc_hat_x, gad_osc_hat_y, gad_osc_loc_r, gad_osc_loc_x, gad_osc_loc_y, gad_osc_mul_r, gad_osc_mul_x, gad_osc_mul_y, gad_plm_fun_u, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_adv_r, gad_pqm_adv_x, gad_pqm_adv_y, gad_pqm_flx_r, gad_pqm_flx_x, gad_pqm_flx_y, gad_pqm_fun_mono, gad_pqm_fun_null, gad_pqm_hat_r, gad_pqm_hat_x, gad_pqm_hat_y, gad_pqm_p5e_r, gad_pqm_p5e_x, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, quadroot, salt_fill
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_pass_r4torl, mds_pass_r4tors, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/monitor: admonitor, g_monitor, mon_calc_advcfl_glob, mon_calc_advcfl_tile, mon_calc_stats_rs, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, mon_surfcor, mon_vort3, mon_writestats_rs, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rl, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs

## advect_xz/input.pqm

Run `$MJX_REFERENCE/coverage/advect_xz/input.pqm/job27827383` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/advect_xz/input.pqm/job27827383/gcov-9c7ed2d/gcov.json.gz`.

### Physics-active check

| check | measured | active |
|---|---|---|
| theta moved | 21 blocks, first 5.544981e-04, last 4.877036e-04, min 2.702811e-04, max 1.102017e-03 | yes |
| salt moved | 21 blocks, first 5.544981e-04, last 6.981151e-04, min 2.677623e-04, max 1.475325e-03 | yes |
| vertical advection routine (explicit or implicit) | gad_pqm_adv_r (800 calls) | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: doAB_onGtGs, dumpInitAndLast, fluidIsWater, momDissip_In_AB, monitor_stdio, multiDimAdvection, pickup_read_mdsio, pickup_write_mdsio, pickupStrictlyMatch, saltAdvection, saltForcing, saltMultiDimAdvec, saltStepping, snapshot_mdsio, tempAdvection, tempForcing, tempMultiDimAdvec, tempStepping, uniformFreeSurfLev, uniformLin_PhiSurf, useMultiDimAdvec, usingZCoords, writePickupAtEnd
- `.FALSE.`: AdamsBashforth_S, AdamsBashforth_T, AdamsBashforthGs, AdamsBashforthGt, applyExchUV_early, calc_wVelocity, debugMode, deepAtmosphere, doResetHFactors, doSaltClimRelax, doThetaClimRelax, exactConserv, fluidIsAir, globalFiles, implicitDiffusion, implicitIntGravWave, implicitViscosity, interDiffKr_pCell, interViscAr_pCell, linFSConserveTr, momAdvection, momForcing, momImplVertAdv, momPressureForcing, momViscosity, nonHydrostatic, printMapIncludesZeros, quasiHydrostatic, rotateGrid, saltImplVertAdv, saltIsActiveTr, saltSOM_Advection, staggerTimeStep, tempImplVertAdv, tempIsActiveTr, tempSOM_Advection, use3Dsolver, useCDscheme, useCoriolis, useCoupler, useMin4hFacEdges, useNest2W_child, useNest2W_parent, useNHMTerms, useNSACGSolver, useRealFreshWaterFlux, useSingleCpuInput, useSingleCpuIO, useSRCGSolver, usingCurvilinearGrid, usingCylindricalGrid, usingMPI, usingPCoords, usingSphericalPolarGrid, vectorInvariantMomentum
- selectors: pCellMix_select=0, saltVertAdvScheme=52, select_rStar=0, selectAddFluid=0, selectNHfreeSurf=0, selectPenetratingSW=0, selectSigmaCoord=0, tempVertAdvScheme=51

### Executed routines (217)

**eesupp/src** (72)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 19/28 | 138-139, 144, 150-151, 212-216 | 137:1/2, 143:1/2, 149:1/2, 178:1/2, 183:1/2, 195:1/2, 211:1/2 |
| `bar2` | bar2.F:58 | 2896 | 16/22 | 123-129 | 121:1/2, 138:1/2, 139:2/2, 142:1/2 |
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 5 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 4161 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `comm_stats` | comm_stats.F:7 | 1 | 53/59 | 71-73, 100-102, 143-144 | 45:1/2, 69:1/2, 98:1/2, 115:1/2, 137:1/2 |
| `different_multiple` | different_multiple.F:7 | 802 | 14/15 | 45 | 44:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 28/28 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eedie` | eedie.F:7 | 1 | 9/18 | 40-42, 58-64 | 37:1/2, 54:1/2, 56:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch1_rl` | exch1_rl.F:8 | 1006 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch1_rs` | exch1_rs.F:8 | 24 | 32/61 | 91, 93, 95, 97, 99, 101, 103, 105, 107, 110, 145-159, 195, 208-233 | 90:1/2, 92:1/2, 94:1/2, 96:1/2, 98:1/2, 100:1/2, 102:1/2, 104:1/2, 106:1/2, 109:1/2, 115:1/2, 170:1/2, 178:1/2, 191:1/2, 203:1/2, 236:1/2 |
| `exch_cycle_ebl` | exch_cycle_ebl.F:7 | 1030 | 7/7 | - | 54:1/2 |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `exch_rl_recv_get_x` | exch_rl_recv_get_x.F:8 | 1006 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rl_recv_get_y` | exch_rl_recv_get_y.F:8 | 1006 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rl_send_put_x` | exch_rl_send_put_x.F:8 | 1006 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rl_send_put_y` | exch_rl_send_put_y.F:7 | 1006 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_rs_recv_get_x` | exch_rs_recv_get_x.F:8 | 24 | 33/85 | 279-314, 336-371 | 264:1/2, 269:1/2, 321:1/2, 326:1/2 |
| `exch_rs_recv_get_y` | exch_rs_recv_get_y.F:8 | 24 | 36/90 | 271-272, 289-324, 346-381 | 267:1/2, 274:1/2, 279:1/2, 331:1/2, 336:1/2 |
| `exch_rs_send_put_x` | exch_rs_send_put_x.F:8 | 24 | 49/106 | 151-156, 171-172, 179-184, 199-263 | 145:1/2, 150:1/2, 160:1/2, 178:1/2, 188:1/2, 279:1/2, 290:1/2, 291:1/2, 292:1/2, 293:1/2, 307:1/2 |
| `exch_rs_send_put_y` | exch_rs_send_put_y.F:7 | 24 | 54/111 | 162-167, 182-183, 190-195, 210-275 | 149:1/2, 156:1/2, 161:1/2, 171:1/2, 189:1/2, 199:1/2, 291:1/2, 302:1/2, 303:1/2, 304:1/2, 305:1/2, 319:1/2 |
| `exch_uv_xy_rs` | exch_uv_xy_rs.F:11 | 5 | 12/13 | 75 | 71:1/2 |
| `exch_uv_xyz_rl` | exch_uv_xyz_rl.F:12 | 201 | 12/13 | 76 | 72:1/2 |
| `exch_uv_xyz_rs` | exch_uv_xyz_rs.F:12 | 1 | 12/13 | 76 | 72:1/2 |
| `exch_xy_rl` | exch_xy_rl.F:9 | 1 | 11/12 | 60 | 56:1/2 |
| `exch_xy_rs` | exch_xy_rs.F:9 | 12 | 11/12 | 60 | 56:1/2 |
| `exch_xyz_rl` | exch_xyz_rl.F:8 | 603 | 11/12 | 59 | 55:1/2 |
| `fool_the_compiler` | fool_the_compiler.F:15 | 15379 | 2/2 | - | - |
| `global_max_r8` | global_max.F:97 | 400 | 12/12 | - | 151:1/2, 153:1/2 |
| `global_sum_r8` | global_sum.F:97 | 42 | 12/12 | - | 154:1/2 |
| `global_sum_tile_rl` | global_sum_tile.F:14 | 781 | 15/15 | - | 83:1/2 |
| `ifnblnk` | utils.F:88 | 13575 | 10/10 | - | - |
| `ilnblnk` | utils.F:123 | 19135 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 111/163 | 90-99, 114-119, 124-129, 134-139, 219-220, 237-238, 255-256, 273-274, 287-290, 295-298, 301-316 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2, 205:1/2, 223:1/2, 241:1/2, 259:1/2, 284:1/2, 292:1/2, 300:1/2 |
| `lcase` | utils.F:190 | 7 | 8/8 | - | - |
| `main` | main.F:223 | 1 | 1/1 | - | - |
| `master_cpu_io` | master_cpu_io.F:7 | 1088 | 6/6 | - | 33:1/2, 34:1/2 |
| `mds_flush` | mds_flush.F:7 | 2 | 3/3 | - | - |
| `mds_reclen` | mds_reclen.F:3 | 237 | 5/11 | 29, 34-39 | 28:1/2, 30:1/2 |
| `mdsfindunit` | mdsfindunit.F:3 | 345 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `memsync` | memsync.F:7 | 4120 | 2/2 | - | - |
| `nml_change_syntax` | nml_change_syntax.F:7 | 45 | 7/7 | - | 74:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 4 | 6/6 | - | 44:1/2 |
| `open_copy_data_file` | open_copy_data_file.F:6 | 2 | 31/41 | 72-76, 112-116 | 65:1/2, 110:1/2, 177:1/2 |
| `print_list_i` | print.F:305 | 33 | 24/49 | 370, 372, 374, 382-384, 393, 395-412, 422-427 | 369:1/2, 371:1/2, 373:1/2, 381:1/2, 392:1/2, 394:2/2, 416:1/2, 418:1/2, 420:1/2 |
| `print_list_l` | print.F:438 | 76 | 24/49 | 503, 505, 507, 515-517, 526, 528-545, 555-560 | 502:1/2, 504:1/2, 506:1/2, 514:1/2, 525:1/2, 527:2/2, 549:1/2, 551:1/2, 553:1/2 |
| `print_list_rl` | print.F:571 | 121 | 46/49 | 648-650 | 647:1/2, 664:1/2, 669:1/2, 682:1/2, 689:1/2, 691:1/2 |
| `print_maprs` | print.F:704 | 5 | 139/210 | 864, 960-1017, 1023-1026, 1048-1051, 1085-1088, 1095, 1100-1101 | 835:4/6, 836:4/8, 837:5/8, 838:4/8, 839:4/8, 861:1/2, 886:1/4, 931:1/2, 1021:1/2, 1031:5/8, 1032:5/8, 1039:4/8, 1040:4/8, 1046:1/2, 1061:4/8, 1062:4/8, 1075:5/8, 1076:4/8, 1080:4/8, 1081:4/8, 1083:1/2, 1090:1/2, 1097:1/2, 1099:1/2, 1131:1/2 |
| `print_message` | print.F:23 | 2275 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |
| `timer_control` | timers.F:74 | 5211 | 69/159 | 486-512, 587, 595-730 | 227:1/2, 229:1/2, 236:1/2, 237:1/2, 238:1/2, 246:1/2, 259:1/2, 286:1/2, 294:1/2, 485:1/2, 536:1/2 |
| `timer_get_time` | timers.F:743 | 5210 | 7/7 | - | - |
| `timer_index` | timers.F:25 | 5211 | 12/12 | - | - |
| `timer_printall` | timers.F:857 | 1 | 3/3 | - | - |
| `timer_start` | timers.F:884 | 2605 | 4/4 | - | - |
| `timer_stop` | timers.F:907 | 2605 | 4/4 | - | - |
| `ucase` | utils.F:311 | 10422 | 8/8 | - | - |
| `write_0d_c` | write_utils.F:579 | 2 | 18/20 | 629, 643 | 633:1/2, 639:1/2, 652:1/2 |
| `write_0d_i` | write_utils.F:240 | 25 | 9/9 | - | - |
| `write_0d_l` | write_utils.F:296 | 76 | 9/9 | - | - |
| `write_0d_rl` | write_utils.F:522 | 78 | 9/9 | - | - |
| `write_1d_rl` | write_utils.F:138 | 43 | 13/32 | 189-195, 208-226 | 202:1/2, 233:1/2 |
| `write_copy1d_rs` | write_utils.F:771 | 4 | 8/8 | - | - |
| `write_xy_xline_rs` | write_utils.F:827 | 12 | 20/20 | - | - |
| `write_xy_yline_rs` | write_utils.F:908 | 12 | 20/20 | - | - |

**model/src** (77)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `add_walls2masks` | add_walls2masks.F:7 | 1 | 10/51 | 55-71, 87-93, 96-102, 113-133 | 52:1/2, 86:1/2, 95:1/2, 112:1/2 |
| `apply_forcing_s` | apply_forcing.F:769 | 8000 | 12/23 | 838, 840, 842, 913-918, 925-929 | 837:1/2, 839:1/2, 841:1/2, 912:1/2, 924:1/2 |
| `apply_forcing_t` | apply_forcing.F:394 | 8000 | 14/49 | 467, 469, 471, 563-612, 627-632, 639-643 | 466:1/2, 468:1/2, 470:1/2, 554:1/2, 626:1/2, 638:1/2 |
| `calc_3d_diffusivity` | calc_3d_diffusivity.F:10 | 800 | 21/87 | 164-166, 269-277, 287-406 | 83:1/2, 132:1/2, 266:1/2, 284:1/2 |
| `calc_adv_flow` | calc_adv_flow.F:7 | 16000 | 26/26 | - | - |
| `config_check` | config_check.F:14 | 1 | 82/455 | 92-97, 130-136, 142-148, 153-159, 166-173, 179-185, 191-197, 253-259, 366-371, 417-423, 442-448, 464-470, 484-490, 497-503, 510-516, 535-541, 547-550, 600-603, 640-643, 646-649, 656-659, 665-668, 674-677, 682-685, 690-695, 700-705, 710-715, 719-722, 727-732, 737-742, 746-752, 758-761, 765-786, 796-803, 809-814, 820-825, 829-835, 839-844, 847-854, 867-870, 876-881, 886-892, 896-902, 906-913, 917-920, 924-931, 934-941, 946-950, 985-995, 999-1002, 1005-1009, 1015-1018, 1028-1045, 1051-1053, 1056-1059, 1062-1065, 1070-1073, 1076-1082, 1085-1091, 1094-1097, 1100-1103, 1108-1119, 1126-1128, 1133-1136, 1141-1144, 1149-1152 | 46:1/2, 58:1/2, 90:1/2, 129:1/2, 141:1/2, 152:1/2, 164:1/2, 178:1/2, 190:1/2, 252:1/2, 364:1/2, 404:1/2, 441:1/2, 463:1/2, 482:1/2, 495:1/2, 508:1/2, 534:1/2, 545:1/2, 598:1/2, 639:1/2, 645:1/2, 654:1/2, 661:1/2, 672:1/2, 680:1/2, 688:1/2, 698:1/2, 708:1/2, 718:1/2, 725:1/2, 735:1/2, 745:1/2, 754:1/2, 764:1/2, 794:1/2, 807:1/2, 818:1/2, 828:1/2, 837:1/2, 846:1/2, 866:1/2, 874:1/2, 885:1/2, 895:1/2, 904:1/2, 916:1/2, 923:1/2, 933:1/2, 945:1/2, 984:1/2, 998:1/2, 1004:1/2, 1011:1/2, 1022:1/2, 1049:1/2, 1055:1/2, 1061:1/2, 1069:1/2, 1075:1/2, 1084:1/2, 1093:1/2, 1099:1/2, 1107:1/2, 1124:1/2, 1131:1/2, 1139:1/2, 1147:1/2, 1158:1/2 |
| `config_summary` | config_summary.F:15 | 1 | 410/491 | 233, 240, 268-281, 307-322, 533-538, 554-588, 595, 756-762, 806-810, 815, 914-923, 946-950, 958-960, 965, 992-994, 1002-1008, 1077, 1083 | 72:1/2, 77:1/2, 230:1/2, 237:1/2, 261:1/2, 283:1/2, 298:1/2, 305:1/2, 423:1/2, 469:1/2, 532:1/2, 551:1/2, 593:1/2, 754:1/2, 804:1/2, 813:1/2, 912:1/2, 936:1/2, 944:1/2, 956:1/2, 962:1/2, 990:1/2, 1000:1/2, 1075:1/2, 1081:1/2 |
| `cycle_tracer` | cycle_tracer.F:6 | 800 | 6/6 | - | - |
| `do_atmospheric_phys` | do_atmospheric_phys.F:10 | 200 | 5/14 | 76-94 | 66:1/2, 69:1/2, 152:1/2 |
| `do_fields_blocking_exchanges` | do_fields_blocking_exchanges.F:7 | 200 | 13/15 | 79, 86 | 49:1/2, 52:1/2, 53:1/2, 55:1/2, 60:1/2, 78:1/2, 85:1/2 |
| `do_oceanic_phys` | do_oceanic_phys.F:43 | 200 | 45/76 | 257-264, 557, 797-798, 811-845, 869-874, 882, 899, 934, 1116, 1119, 1123 | 248:1/2, 251:1/2, 256:1/2, 553:1/2, 574:1/2, 577:1/2, 728:1/2, 758:1/2, 795:1/2, 810:1/2, 867:1/2, 878:1/2, 896:1/2, 932:1/2, 1113:1/2, 1118:1/2, 1121:1/2, 1132:1/2 |
| `do_the_model_io` | do_the_model_io.F:9 | 201 | 6/13 | 97-110, 242 | 96:1/2, 116:1/2, 241:1/2 |
| `do_write_pickup` | do_write_pickup.F:8 | 200 | 19/23 | 82, 84, 115-117 | 81:1/2, 83:1/2, 94:1/2, 99:1/2, 108:1/2, 113:1/2 |
| `eos_check` | ini_eos.F:382 | 1 | 2/2 | - | - |
| `external_fields_load` | external_fields_load.F:7 | 200 | 3/109 | 72-350 | 63:1/2 |
| `external_forcing_surf` | external_forcing_surf.F:14 | 200 | 34/64 | 75, 151-153, 161-163, 176, 268-285, 299-317, 327-332, 368-372 | 74:1/2, 149:1/2, 160:1/2, 173:1/2, 262:1/2, 296:1/2, 325:1/2, 336:1/2, 364:1/2, 367:1/2 |
| `find_rho_2d` | find_rho.F:25 | 8000 | 9/55 | 112-265 | 92:1/2 |
| `find_rho_scalar` | find_rho.F:834 | 58 | 22/72 | 935-938, 949-1179 | 932:1/2, 941:1/2 |
| `forward_step` | forward_step.F:70 | 200 | 59/119 | 467-485, 508-512, 767-769, 789-793, 838-840, 844, 868-870, 903-905, 913-915, 927-929, 948-950, 958-960, 979-1006, 1109-1111, 1176, 1201-1202 | 424:1/2, 465:1/2, 507:1/2, 530:1/2, 624:1/2, 654:1/2, 725:1/2, 730:1/2, 766:1/2, 786:1/2, 832:1/2, 842:1/2, 866:1/2, 897:1/2, 911:1/2, 924:1/2, 939:1/2, 952:1/2, 976:1/2, 1108:1/2, 1151:1/2, 1175:1/2, 1200:1/2, 1218:1/2 |
| `ini_cartesian_grid` | ini_cartesian_grid.F:6 | 1 | 37/37 | - | - |
| `ini_cg2d` | ini_cg2d.F:7 | 1 | 76/87 | 120, 152-161, 172-175, 216, 221, 227 | 67:1/2, 117:1/2, 144:1/2, 149:1/2, 167:1/2, 171:1/2, 215:1/2, 220:1/2, 226:1/2 |
| `ini_cori` | ini_cori.F:8 | 1 | 25/65 | 57-63, 84-112, 121-180, 191 | 55:1/2, 68:1/2, 72:1/2, 119:1/2, 184:1/2, 188:1/2, 212:1/2 |
| `ini_depths` | ini_depths.F:7 | 1 | 32/67 | 63-66, 94-98, 140, 153, 173-205, 225-241, 271-274 | 61:1/2, 91:1/2, 137:1/2, 150:1/2, 155:1/2, 224:1/2, 261:1/2, 270:1/2 |
| `ini_dynvars` | ini_dynvars.F:13 | 1 | 27/27 | - | - |
| `ini_eos` | ini_eos.F:13 | 1 | 30/255 | 87-365 | 45:1/2, 48:1/2, 84:1/2, 85:1/2, 86:1/2 |
| `ini_ffields` | ini_ffields.F:6 | 1 | 47/47 | - | - |
| `ini_fields` | ini_fields.F:9 | 1 | 9/12 | 37-38, 50 | 28:1/2, 49:1/2 |
| `ini_forcing` | ini_forcing.F:7 | 1 | 31/49 | 52, 58, 72, 75, 78, 80, 83-90, 98, 101, 104, 108, 112, 193 | 50:1/2, 56:1/2, 71:1/2, 74:1/2, 77:1/2, 79:1/2, 82:1/2, 97:1/2, 100:1/2, 103:1/2, 106:1/2, 110:1/2, 192:1/2 |
| `ini_global_domain` | ini_global_domain.F:7 | 1 | 36/61 | 128-131, 136-138, 146-181 | 113:1/2, 118:1/2, 123:1/2, 135:1/2, 145:1/2 |
| `ini_grid` | ini_grid.F:8 | 1 | 102/115 | 132-144, 192 | 130:1/2, 153:1/2, 155:1/2, 161:1/2, 163:1/2, 169:1/2, 185:1/2, 189:1/2, 228:1/2 |
| `ini_linear_phisurf` | ini_linear_phisurf.F:7 | 1 | 18/68 | 90-182, 192, 211, 228-235 | 78:1/2, 191:1/2, 210:1/2, 215:1/2 |
| `ini_local_grid` | ini_local_grid.F:10 | 2 | 28/28 | - | 136:1/2 |
| `ini_masks_etc` | ini_masks_etc.F:7 | 1 | 187/199 | 228, 247-261, 435 | 65:1/2, 196:1/2, 206:1/2, 227:1/2, 243:1/2, 418:1/2, 420:1/2, 448:1/2 |
| `ini_mixing` | ini_mixing.F:6 | 1 | 2/2 | - | - |
| `ini_model_io` | ini_model_io.F:8 | 1 | 28/52 | 146-179, 191-195, 206-209 | 120:1/2, 130:1/2, 145:1/2, 190:1/2, 205:1/2, 244:1/2 |
| `ini_nlfs_vars` | ini_nlfs_vars.F:7 | 1 | 59/59 | - | 47:1/2, 157:1/2, 159:1/2, 161:1/2, 163:1/2, 165:1/2, 186:1/2 |
| `ini_parms` | ini_parms.F:11 | 1 | 405/876 | 434-438, 452-453, 455-456, 461-464, 471-478, 491-494, 499, 504-506, 530-533, 537-541, 557-565, 577-580, 585-591, 609-612, 615-620, 623-625, 666, 669-674, 680-683, 688-691, 696-702, 709-714, 720-725, 730-735, 740-743, 752-754, 758-761, 767-769, 773-775, 779-782, 798-804, 807-813, 816-822, 825-833, 836-840, 843-846, 849-852, 855-861, 864-870, 873-880, 883-890, 893-897, 900-904, 907-911, 914-918, 921-924, 927-930, 940-944, 952-955, 958-961, 976-980, 988-994, 997-1003, 1006-1009, 1012-1015, 1018-1020, 1023-1029, 1037-1039, 1041, 1048-1055, 1070-1091, 1105, 1109-1112, 1116-1118, 1123-1124, 1127, 1131-1133, 1139, 1148, 1151, 1154, 1159-1164, 1168-1173, 1178-1183, 1188-1196, 1199-1200, 1230-1234, 1244-1261, 1264-1268, 1273-1275, 1278-1282, 1285-1289, 1298-1301, 1314-1317, 1333-1335, 1340-1347, 1353, 1364, 1372-1384, 1394, 1397-1405, 1415-1425, 1439-1443, 1449-1451, 1462-1464, 1468-1474, 1477-1483, 1493-1497, 1505-1509, 1512-1515, 1520-1525, 1532-1537, 1542-1547, 1570-1571, 1605-1610, 1615-1618 | 334:1/2, 432:1/2, 451:1/2, 454:1/2, 457:1/2, 467:1/2, 480:1/2, 481:1/2, 482:1/2, 483:1/2, 484:1/2, 485:1/2, 487:1/2, 489:1/2, 496:1/2, 502:1/2, 509:1/2, 510:1/2, 512:1/2, 513:1/2, 514:1/2, 515:1/2, 517:1/2, 518:1/2, 520:1/2, 521:1/2, 522:1/2, 523:1/2, 524:1/2, 527:1/2, 529:1/2, 535:1/2, 543:1/2, 556:1/2, 567:1/2, 568:1/2, 569:1/2, 570:1/2, 571:1/2, 574:1/2, 576:1/2, 583:1/2, 593:1/2, 599:1/2, 600:1/2, 601:1/2, 602:1/2, 603:1/2, 606:1/2, 608:1/2, 614:1/2, 622:1/2, 636:1/2, 637:1/2, 638:1/2, 639:1/2, 641:1/2, 642:1/2, 643:1/2, 644:1/2, 645:1/2, 646:1/2, 648:1/2, 650:1/2, 651:1/2, 653:1/2, 661:1/2, 663:1/2, 664:1/2, 668:1/2, 678:1/2, 686:1/2, 694:1/2, 705:1/2, 708:1/2, 716:1/2, 719:1/2, 727:1/2, 729:1/2, 738:1/2, 747:1/2, 748:1/2, 749:1/2, 750:1/2, 756:1/2, 765:1/2, 771:1/2, 777:1/2, 789:1/2, 790:1/2, 793:1/2, 797:1/2, 806:1/2, 815:1/2, 824:1/2, 835:1/2, 842:1/2, 848:1/2, 854:1/2, 863:1/2, 872:1/2, 882:1/2, 892:1/2, 899:1/2, 906:1/2, 913:1/2, 920:1/2, 926:1/2, 938:1/2, 951:1/2, 957:1/2, 974:1/2, 987:1/2, 996:1/2, 1005:1/2, 1011:1/2, 1017:1/2, 1022:1/2, 1035:1/2, 1040:1/2, 1043:1/2, 1044:1/2, 1045:1/2, 1046:1/2, 1047:1/2, 1057:1/2, 1058:1/2, 1059:1/2, 1061:1/2, 1068:1/2, 1069:1/2, 1095:1/2, 1097:1/2, 1099:1/2, 1101:1/2, 1104:1/2, 1107:1/2, 1114:1/2, 1121:1/2, 1125:1/2, 1128:1/2, 1138:1/2, 1141:1/2, 1144:1/2, 1147:1/2, 1150:1/2, 1153:1/2, 1157:1/2, 1166:1/2, 1175:1/2, 1187:1/2, 1198:1/2, 1228:1/2, 1242:1/2, 1263:1/2, 1270:1/2, 1277:1/2, 1284:1/2, 1294:1/2, 1295:1/2, 1296:1/2, 1297:1/2, 1303:1/2, 1310:1/2, 1311:1/2, 1312:1/2, 1313:1/2, 1319:1/2, 1327:1/2, 1328:1/2, 1329:1/2, 1330:1/2, 1331:1/2, 1337:1/2, 1350:1/2, 1351:1/2, 1358:1/2, 1361:1/2, 1368:1/2, 1371:1/2, 1388:1/2, 1389:1/2, 1391:1/2, 1392:1/2, 1393:1/2, 1395:1/2, 1412:1/2, 1413:1/2, 1429:1/2, 1433:1/2, 1434:1/2, 1435:1/2, 1436:1/2, 1437:1/2, 1438:1/2, 1447:1/2, 1457:1/2, 1458:1/2, 1459:1/2, 1460:1/2, 1467:1/2, 1476:1/2, 1491:1/2, 1504:1/2, 1511:1/2, 1518:1/2, 1529:1/2, 1539:1/2, 1569:1/2, 1579:1/2, 1580:1/2, 1585:1/2, 1588:1/2, 1603:1/2, 1613:1/2 |
| `ini_pressure` | ini_pressure.F:7 | 1 | 19/72 | 81-180 | 55:1/2, 74:1/2, 188:1/2, 198:1/2 |
| `ini_psurf` | ini_psurf.F:7 | 1 | 15/17 | 60-62 | 59:1/2 |
| `ini_salt` | ini_salt.F:7 | 1 | 25/38 | 100, 109-124, 130 | 65:1/2, 88:1/2, 95:1/2, 99:1/2, 108:1/2, 128:1/2 |
| `ini_theta` | ini_theta.F:7 | 1 | 26/47 | 101, 110-125, 131-138, 151 | 66:1/2, 89:1/2, 96:1/2, 100:1/2, 109:1/2, 130:1/2, 149:1/2 |
| `ini_vel` | ini_vel.F:6 | 1 | 21/22 | 61 | 55:1/2, 57:1/2, 60:1/2 |
| `ini_vertical_grid` | ini_vertical_grid.F:6 | 1 | 52/180 | 56, 61-66, 80-100, 105-117, 156-166, 183-190, 217-405 | 40:1/2, 55:1/2, 59:1/2, 71:1/2, 78:1/2, 103:1/2, 137:1/2, 140:1/2, 175:1/2, 177:1/2, 203:1/2 |
| `initialise_fixed` | initialise_fixed.F:7 | 1 | 50/50 | - | 93:1/2, 109:1/2, 115:1/2, 131:1/2, 138:1/2, 144:1/2, 151:1/2, 161:1/2, 167:1/2, 173:1/2, 179:1/2, 185:1/2, 196:1/2, 209:1/2, 215:1/2, 221:1/2, 231:1/2, 241:1/2, 259:1/2, 265:1/2, 271:1/2, 276:1/2, 278:1/2, 298:1/2 |
| `initialise_varia` | initialise_varia.F:36 | 1 | 30/40 | 302, 305-321, 326, 341, 345 | 180:1/2, 203:1/2, 220:1/2, 229:1/2, 240:1/2, 251:1/2, 301:1/2, 304:1/2, 325:1/2, 331:1/2, 338:1/2, 343:1/2, 374:1/2, 380:1/2, 388:1/2, 393:1/2 |
| `integr_continuity` | integr_continuity.F:13 | 1 | 15/73 | 92-185, 211-221, 249-254, 279-282, 318, 329-331, 337-338 | 90:1/2, 207:1/2, 245:1/2, 277:1/2, 317:1/2, 321:1/2, 325:1/2, 336:1/2 |
| `integrate_for_w` | integrate_for_w.F:6 | 40 | 19/44 | 95-117, 126-144, 152-168 | 93:1/2, 123:1/2, 150:1/2 |
| `load_fields_driver` | load_fields_driver.F:19 | 200 | 19/27 | 149, 154-164, 263 | 105:1/2, 148:1/2, 153:1/2, 172:1/2, 229:1/2, 231:1/2, 261:1/2, 268:1/2 |
| `load_grid_spacing` | load_grid_spacing.F:11 | 1 | 28/78 | 60-65, 70-75, 80-85, 89-94, 99-105, 119-122, 126-129, 133-136, 144-147, 151-154, 158-161 | 56:1/2, 59:1/2, 69:1/2, 79:1/2, 88:1/2, 98:1/2, 109:1/2, 118:1/2, 125:1/2, 132:1/2, 143:1/2, 150:1/2, 157:1/2, 167:1/2, 172:1/2 |
| `load_ref_files` | load_ref_files.F:7 | 1 | 28/70 | 56-61, 67-72, 87-92, 98-103, 108-114, 121-152 | 42:1/2, 45:1/2, 48:1/2, 49:1/2, 51:1/2, 66:1/2, 74:1/2, 77:1/2, 80:1/2, 82:1/2, 97:1/2, 107:1/2, 120:1/2 |
| `main_do_loop` | main_do_loop.F:62 | 200 | 8/8 | - | 197:1/2, 211:1/2, 245:1/2 |
| `packages_boot` | packages_boot.F:7 | 1 | 82/82 | - | 100:1/2 |
| `packages_check` | packages_check.F:7 | 1 | 66/126 | 132-140, 148-155, 161-168, 194, 207, 212, 265, 280, 289, 304, 321, 342, 349, 356, 369, 376, 403, 412, 437, 479, 486, 499, 504, 511, 522-529, 534-541 | 118:1/2, 131:1/2, 146:1/2, 159:1/2, 193:1/2, 202:1/2, 206:1/2, 211:1/2, 218:1/2, 224:1/2, 230:1/2, 236:1/2, 242:1/2, 248:1/2, 254:1/2, 260:1/2, 264:1/2, 269:1/2, 275:1/2, 279:1/2, 284:1/2, 288:1/2, 293:1/2, 303:1/2, 310:1/2, 314:1/2, 320:1/2, 325:1/2, 329:1/2, 335:1/2, 341:1/2, 348:1/2, 355:1/2, 362:1/2, 368:1/2, 375:1/2, 382:1/2, 388:1/2, 392:1/2, 396:1/2, 402:1/2, 407:1/2, 411:1/2, 432:1/2, 436:1/2, 441:1/2, 447:1/2, 453:1/2, 457:1/2, 466:1/2, 472:1/2, 478:1/2, 485:1/2, 492:1/2, 498:1/2, 503:1/2, 510:1/2, 517:1/2, 521:1/2, 533:1/2 |
| `packages_init_fixed` | packages_init_fixed.F:7 | 1 | 8/14 | 170-176, 675-677 | 144:1/2, 167:1/2, 193:1/2, 673:1/2, 682:1/2 |
| `packages_init_variables` | packages_init_variables.F:21 | 1 | 8/11 | 169, 174, 637 | 168:1/2, 173:1/2, 192:1/2, 194:1/2, 636:1/2 |
| `packages_print_msg` | packages_print_msg.F:7 | 6 | 22/24 | 54-55 | 49:2/4, 52:1/2, 63:2/4, 66:2/4 |
| `packages_readparms` | packages_readparms.F:8 | 1 | 3/3 | - | - |
| `packages_unused_msg` | packages_unused_msg.F:7 | 1 | 25/37 | 68-70, 76, 82-83, 100-107 | 60:1/2, 62:1/2, 64:1/2, 72:1/2, 78:1/2, 97:1/2, 99:1/2 |
| `packages_write_pickup` | packages_write_pickup.F:11 | 1 | 6/7 | 225 | 223:1/2, 255:1/2 |
| `plot_field_xyrs` | plot_field.F:10 | 2 | 24/26 | 58-59 | 54:1/2, 56:1/2 |
| `plot_field_xyzrs` | plot_field.F:176 | 3 | 26/27 | 232 | 225:1/2, 227:1/2 |
| `salt_integrate` | salt_integrate.F:13 | 400 | 51/69 | 167, 169, 192, 259-267, 326, 386-390, 397-399, 412-434, 492, 501, 517 | 147:1/2, 166:1/2, 168:1/2, 177:1/2, 257:1/2, 268:1/2, 273:1/2, 319:1/2, 325:1/2, 366:1/2, 374:1/2, 396:1/2, 408:1/2, 479:1/2, 493:1/2, 504:1/2 |
| `set_defaults` | set_defaults.F:8 | 1 | 312/313 | 241 | 240:1/2 |
| `set_grid_factors` | set_grid_factors.F:6 | 1 | 15/34 | 62-90 | 37:1/2, 60:1/2 |
| `set_parms` | set_parms.F:11 | 1 | 92/158 | 60-63, 67, 71, 79, 109-115, 120-126, 171-175, 186-189, 192-195, 200, 206, 212, 218, 224, 230, 259, 279, 284-290, 294-300, 314-328, 348-351 | 51:1/2, 55:1/2, 59:1/2, 65:1/2, 69:1/2, 73:1/2, 74:1/2, 82:1/2, 85:1/2, 95:1/2, 101:1/2, 108:1/2, 119:1/2, 159:1/2, 167:1/2, 185:1/2, 191:1/2, 199:1/2, 205:1/2, 211:1/2, 217:1/2, 223:1/2, 229:1/2, 258:1/2, 260:1/2, 261:1/2, 262:1/2, 267:1/2, 268:1/2, 269:1/2, 272:1/2, 275:1/2, 277:1/2, 293:1/2, 313:1/2, 346:1/2 |
| `set_ref_state` | set_ref_state.F:6 | 1 | 54/195 | 101-133, 140-165, 180-187, 212-353, 362-382, 391-392, 396, 400-412, 420-421 | 48:1/2, 84:1/2, 87:1/2, 139:1/2, 168:1/2, 178:1/2, 202:1/2, 359:1/2, 389:1/2, 394:1/2, 399:1/2, 418:1/2 |
| `state_summary` | state_summary.F:6 | 1 | 11/11 | - | 43:1/2 |
| `temp_integrate` | temp_integrate.F:13 | 400 | 51/69 | 169, 171, 194, 261-269, 328, 388-392, 399-401, 414-436, 494, 503, 519 | 149:1/2, 168:1/2, 170:1/2, 179:1/2, 259:1/2, 270:1/2, 275:1/2, 321:1/2, 327:1/2, 368:1/2, 376:1/2, 398:1/2, 410:1/2, 481:1/2, 495:1/2, 506:1/2 |
| `the_main_loop` | the_main_loop.F:64 | 1 | 15/15 | - | 292:1/2, 382:1/2, 792:1/2 |
| `the_model_main` | the_model_main.F:516 | 1 | 22/22 | - | 606:1/2, 616:1/2, 743:1/2, 785:1/2, 794:1/2 |
| `thermodynamics` | thermodynamics.F:28 | 200 | 34/63 | 133-136, 158, 207-239, 282, 389, 395-402 | 127:1/2, 132:1/2, 153:1/2, 206:1/2, 278:1/2, 317:1/2, 319:1/2, 328:1/2, 330:1/2, 387:1/2, 394:1/2, 413:1/2 |
| `timestep_tracer` | timestep_tracer.F:7 | 800 | 6/6 | - | - |
| `tracers_correction_step` | tracers_correction_step.F:7 | 200 | 5/6 | 117 | 115:1/2 |
| `update_surf_dr` | update_surf_dr.F:6 | 200 | 13/49 | 52-79, 88-115 | 49:1/2, 84:1/2 |
| `write_grid` | write_grid.F:12 | 1 | 44/66 | 98-101, 106-111, 117, 119-121, 133-140 | 78:1/2, 97:1/2, 105:1/2, 116:1/2, 118:1/2, 132:1/2 |
| `write_pickup` | write_pickup.F:8 | 1 | 45/75 | 245-250, 253-257, 261-265, 288-290, 335-338, 365-371 | 98:1/2, 108:1/2, 111:1/2, 128:1/2, 131:1/2, 244:1/2, 252:1/2, 260:1/2, 287:1/2, 333:1/2, 334:1/2, 352:1/2, 360:1/2, 364:1/2 |
| `write_state` | write_state.F:11 | 201 | 18/20 | 85, 126 | 84:1/2, 95:1/2, 123:1/2 |

**pkg/diagnostics** (1)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `diagnostics_readparms` | diagnostics_readparms.F:8 | 1 | 8/372 | 123-653 | 109:1/2, 111:1/2 |

**pkg/generic_advdiff** (29)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `gad_advection` | gad_advection.F:11 | 800 | 145/232 | 213-219, 253-266, 335-337, 351-364, 394, 414, 418, 422, 426, 430, 437, 446, 461, 475-539, 615, 635, 639, 643, 647, 651, 658, 667, 682, 696-761, 824-827, 844-845, 848-849, 866, 909, 991, 995, 999, 1003, 1007, 1018, 1081-1083 | 212:1/2, 252:1/2, 334:1/2, 349:1/2, 389:1/2, 392:1/2, 411:1/2, 415:1/2, 419:1/2, 423:1/2, 427:1/2, 433:1/2, 439:1/2, 459:1/2, 474:1/2, 552:1/2, 553:1/2, 610:1/2, 613:1/2, 632:1/2, 636:1/2, 640:1/2, 644:1/2, 648:1/2, 654:1/2, 660:1/2, 680:1/2, 695:1/2, 775:1/2, 776:1/2, 819:1/2, 843:1/2, 847:1/2, 864:1/2, 875:1/2, 884:1/2, 905:1/2, 911:1/2, 988:1/2, 992:1/2, 996:1/2, 1000:1/2, 1004:1/2, 1009:1/2, 1080:1/2 |
| `gad_advscheme_get` | gad_advscheme.F:116 | 6 | 4/5 | 146 | 143:1/2 |
| `gad_advscheme_init` | gad_advscheme.F:14 | 1 | 5/5 | - | 41:1/2 |
| `gad_advscheme_set` | gad_advscheme.F:55 | 17 | 5/12 | 99-105 | 94:1/2, 95:1/2 |
| `gad_calc_rhs` | gad_calc_rhs.F:10 | 16000 | 60/177 | 192, 213-216, 236-244, 256-316, 329, 340, 365-366, 385-445, 458, 469, 494-495, 515-592, 606-608, 620, 646-647, 793 | 191:1/2, 197:1/2, 199:1/2, 212:1/2, 229:1/2, 255:1/2, 328:1/2, 339:1/2, 363:1/2, 384:1/2, 457:1/2, 468:1/2, 492:1/2, 513:1/2, 605:1/2, 617:1/2, 643:1/2, 791:1/2 |
| `gad_check` | gad_check.F:6 | 1 | 12/40 | 82-88, 99-106, 119-129, 178-181 | 39:1/2, 81:1/2, 95:1/2, 97:1/2, 118:1/2, 176:1/2 |
| `gad_diff_r` | gad_diff_r.F:7 | 16000 | 10/10 | - | - |
| `gad_init_fixed` | gad_init_fixed.F:6 | 1 | 96/125 | 63-65, 90-93, 97-100, 104-107, 111-114, 131, 136, 151, 156, 159-162, 180, 193 | 37:1/2, 61:1/2, 89:1/2, 96:1/2, 103:1/2, 110:1/2, 130:1/2, 135:1/2, 150:1/2, 155:1/2, 158:1/2, 170:1/2, 174:1/2, 178:1/2, 191:1/2, 200:1/2 |
| `gad_init_varia` | gad_init_varia.F:6 | 1 | 11/14 | 60-69 | 58:1/2 |
| `gad_osc_hat_r` | gad_osc_hat_r.F:95 | 64800 | 4/4 | - | - |
| `gad_osc_hat_x` | gad_osc_hat_x.F:103 | 72000 | 4/4 | - | - |
| `gad_osc_loc_r` | gad_osc_hat_r.F:10 | 1684800 | 18/18 | - | - |
| `gad_osc_loc_x` | gad_osc_hat_x.F:10 | 1296000 | 20/20 | - | - |
| `gad_osc_mul_r` | gad_osc_mul_r.F:3 | 686817 | 23/23 | - | - |
| `gad_osc_mul_x` | gad_osc_mul_x.F:3 | 393696 | 23/23 | - | - |
| `gad_plm_fun_u` | gad_plm_fun.F:10 | 3931200 | 14/14 | - | - |
| `gad_pqm_adv_r` | gad_pqm_adv_r.F:3 | 800 | 30/32 | 145-147 | 108:1/2 |
| `gad_pqm_adv_x` | gad_pqm_adv_x.F:3 | 16000 | 22/24 | 134-136 | 104:1/2 |
| `gad_pqm_adv_y` | gad_pqm_adv_y.F:3 | 16000 | 19/24 | 107-130 | 104:1/2 |
| `gad_pqm_flx_r` | gad_pqm_flx_r.F:3 | 129600 | 26/26 | - | - |
| `gad_pqm_flx_x` | gad_pqm_flx_x.F:3 | 144000 | 28/30 | 79, 112 | 74:1/2, 107:1/2 |
| `gad_pqm_fun_mono` | gad_pqm_fun.F:123 | 3931200 | 75/75 | - | - |
| `gad_pqm_fun_null` | gad_pqm_fun.F:76 | 1965600 | 8/8 | - | - |
| `gad_pqm_hat_r` | gad_pqm_hat_r.F:3 | 129600 | 30/31 | 98 | 95:1/2, 110:1/2 |
| `gad_pqm_hat_x` | gad_pqm_hat_x.F:3 | 144000 | 33/34 | 100 | 97:1/2, 112:1/2 |
| `gad_pqm_p5e_r` | gad_pqm_p5e_r.F:3 | 129600 | 21/21 | - | - |
| `gad_pqm_p5e_x` | gad_pqm_p5e_x.F:3 | 144000 | 21/21 | - | - |
| `gad_write_pickup` | gad_write_pickup.F:7 | 1 | 6/20 | 72-83, 89-100 | 70:1/2, 87:1/2 |
| `quadroot` | gad_pqm_fun.F:11 | 2377503 | 15/20 | 54-63 | 30:1/2 |

**pkg/mdsio** (9)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mds_pass_r8torl` | mdsio_pass_r8torl.F:12 | 96 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_pass_r8tors` | mdsio_pass_r8tors.F:12 | 21 | 20/36 | 60, 92-115 | 59:1/2, 63:1/2 |
| `mds_read_field` | mdsio_read_field.F:6 | 4 | 65/206 | 145-150, 152-158, 163-174, 179-191, 196-212, 222, 244-253, 265-365, 435, 450-495, 527-538, 549-559 | 144:1/2, 151:1/2, 161:1/2, 177:1/2, 194:1/2, 216:1/2, 219:1/2, 220:1/2, 230:1/2, 232:1/2, 233:1/2, 243:1/2, 262:1/2, 377:1/2, 380:1/2, 392:1/2, 434:1/2, 437:1/4, 506:1/2, 526:1/2, 540:1/2, 544:1/2 |
| `mds_wr_metafiles` | mdsio_wr_metafiles.F:7 | 1 | 38/58 | 112, 120-139, 148-149 | 113:1/2, 116:1/2, 118:1/2, 145:1/2, 146:1/2, 200:1/2, 201:1/2, 202:1/2, 203:1/2, 204:1/2, 222:1/2 |
| `mds_wr_rec_rl` | mdsio_wr_rec_rl.F:8 | 1 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_wr_rec_rs` | mdsio_wr_rec_rs.F:8 | 6 | 10/18 | 49-51, 55-61, 72-74 | 47:1/2, 54:1/2, 62:1/2 |
| `mds_write_field` | mdsio_write_field.F:6 | 113 | 82/217 | 173-178, 180-186, 191-202, 207-219, 224-240, 250, 255-258, 272-361, 374-385, 396-406, 425-434, 474-484, 512, 560-561, 577-597 | 172:1/2, 179:1/2, 189:1/2, 205:1/2, 222:1/2, 244:1/2, 247:1/2, 248:1/2, 253:1/2, 269:1/2, 373:1/2, 387:1/2, 391:1/2, 413:1/2, 424:1/2, 471:1/2, 511:1/2, 514:1/4, 518:1/2, 559:1/2, 575:1/2 |
| `mds_write_meta` | mdsio_write_meta.F:6 | 221 | 39/52 | 108, 135-139, 146-147, 152, 157-159, 214, 229 | 107:1/2, 124:1/2, 128:1/4, 130:1/4, 145:1/2, 151:1/2, 153:1/2, 207:1/4, 212:1/2, 222:1/4, 228:1/2 |
| `mds_writevec_loc` | mdsio_writevec_loc.F:6 | 7 | 47/88 | 106-111, 118-129, 134-136, 144-145, 158-161, 172-173, 177-179, 193-201, 224-233 | 96:1/2, 104:1/2, 116:1/2, 133:1/2, 142:1/2, 153:1/2, 166:1/2, 175:1/2, 184:1/2, 188:1/2, 205:1/2, 209:1/2, 211:1/2, 213:1/2, 239:1/2, 250:1/2 |

**pkg/monitor** (16)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `mon_advcfl` | mon_advcfl.F:8 | 42 | 13/13 | - | - |
| `mon_advcflw` | mon_advcflw.F:8 | 21 | 13/13 | - | - |
| `mon_advcflw2` | mon_advcflw2.F:8 | 21 | 13/13 | - | - |
| `mon_calc_stats_rl` | mon_calc_stats_rl.F:8 | 126 | 71/71 | - | 93:1/2, 117:1/2, 122:1/2, 125:1/2, 129:1/2 |
| `mon_init` | mon_init.F:8 | 1 | 12/12 | - | 36:1/2, 42:1/2 |
| `mon_ke` | mon_ke.F:8 | 21 | 52/125 | 167-320 | 159:1/2, 160:1/2, 166:1/2 |
| `mon_out_all` | mon_out.F:84 | 924 | 51/51 | - | 127:1/2, 142:1/2, 143:2/4, 151:2/4, 153:2/4, 162:1/2, 166:2/4, 168:2/4, 176:1/2, 180:2/4, 182:2/4, 193:1/2 |
| `mon_out_i` | mon_out.F:8 | 21 | 4/4 | - | - |
| `mon_out_rl` | mon_out.F:58 | 903 | 5/5 | - | - |
| `mon_printstats_rs` | mon_printstats_rs.F:8 | 21 | 9/9 | - | - |
| `mon_set_iounit` | mon_set_iounit.F:8 | 1 | 6/6 | - | 30:1/2 |
| `mon_set_pref` | mon_set_pref.F:8 | 107 | 13/13 | - | 38:1/2, 42:1/2, 45:2/4 |
| `mon_solution` | mon_solution.F:8 | 21 | 6/17 | 44, 48-62 | 35:1/2, 47:1/2 |
| `mon_stats_rs` | mon_stats_rs.F:8 | 21 | 52/52 | - | 83:1/2, 87:1/2, 91:1/2 |
| `mon_writestats_rl` | mon_writestats_rl.F:8 | 126 | 17/17 | - | - |
| `monitor` | monitor.F:8 | 201 | 54/67 | 57, 119-120, 135-145, 150-153 | 50:1/2, 54:1/2, 77:1/2, 103:1/2, 134:1/2, 149:1/2, 169:1/2, 172:1/2, 178:1/2, 182:1/2 |

**pkg/rw** (13)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `read_fld_xyz_rl` | read_fld_xyz_rl.F:3 | 3 | 12/15 | 35-37 | 32:1/2 |
| `read_mflds_init` | read_mflds.F:17 | 1 | 10/10 | - | - |
| `read_rec_xy_rs` | read_rec.F:22 | 1 | 7/7 | - | - |
| `set_write_global_fld` | set_write_global_fld.F:3 | 1 | 3/3 | - | - |
| `set_write_global_rec` | write_rec.F:24 | 1 | 3/3 | - | - |
| `set_write_global_sec` | write_rec.F:53 | 1 | 3/3 | - | - |
| `write_fld_xy_rl` | write_fld_xy_rl.F:3 | 21 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xy_rs` | write_fld_xy_rs.F:3 | 17 | 12/17 | 37-42 | 35:1/2 |
| `write_fld_xyz_rl` | write_fld_xyz_rl.F:3 | 65 | 15/17 | 36, 38 | 35:1/2, 37:1/2 |
| `write_fld_xyz_rs` | write_fld_xyz_rs.F:3 | 3 | 12/17 | 37-42 | 35:1/2 |
| `write_glvec_rl` | write_glvec_rl.F:6 | 1 | 14/17 | 49-51 | 46:1/2 |
| `write_glvec_rs` | write_glvec_rs.F:6 | 6 | 14/17 | 49-51 | 46:1/2 |
| `write_rec_3d_rl` | write_rec.F:399 | 7 | 7/7 | - | - |

### Compiled but not executed (532 routines)

- eesupp/src: all_proc_die, barrier_ms, barrier_mu, cumulsum_z_tile_rl, date, diff_phase_multiple, eedata_example, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rl, exch_3d_rs, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rl, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rl, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_xy_r4, exch_xy_r8, exch_xyz_r4, exch_xyz_r8, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_sum_int, global_sum_r4, global_sum_singlecpu_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lef_zero, machine, master_cpu_thread, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, print_error, print_maprl, ptreentry, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, write_0d_r4, write_0d_r8, write_0d_rs, write_1d_i, write_1d_l, write_copy1d_r4, write_copy1d_r8
- model/src: adams_bashforth2, adams_bashforth3, analylic_theta, apply_forcing_u, apply_forcing_v, calc_div_ghat, calc_eddy_stress, calc_grad_phi_fv, calc_grad_phi_hyd, calc_grad_phi_surf, calc_grid_angles, calc_gw, calc_ivdc, calc_oce_mxlayer, calc_phi_hyd, calc_r_star, calc_surf_dr, calc_viscosity, calc_wsurf_tr, cg2d, cg2d_ex0, cg2d_nsa, cg2d_sr, cg3d, cg3d_ex0, check_pickup, convective_adjustment, convective_adjustment_ini, convective_weights, convectively_mixtracer, convert_ct2pt, convert_pt2ct, correction_step, cycle_ab_tracer, diags_oceanic_surf_flux, diags_phi_hyd, diags_phi_rlow, diags_rho_g, diags_rho_l, diags_sound_speed, do_stagger_fields_exchanges, do_statevars_diags, dynamics, external_forcing_s, external_forcing_t, external_forcing_u, external_forcing_v, find_alpha, find_beta, find_bulkmod, find_hyd_press_1d, find_rhoden, find_rhonum, find_rhop0, find_rhoteos, forcing_surf_relax, freesurf_rescale_g, freeze_surface, grad_sigma, gsw_ct_from_pt, gsw_gibbs_pt0_pt0, gsw_pt_from_ct, impldiff, ini_cg3d, ini_curvilinear_grid, ini_cylinder_grid, ini_mnc_vars, ini_nh_fields, ini_nh_vars, ini_p_ground, ini_sigma_hfac, ini_spherical_polar_grid, look_for_neg_salinity, momentum_correction_step, packages_error_msg, plot_field_xyrl, plot_field_xyzrl, plot_field_xzrl, plot_field_xzrs, plot_field_yzrl, plot_field_yzrs, port_ranarr, port_rand, port_rand_norm, post_cg3d, pre_cg3d, pressure_for_eos, read_pickup, remove_mean_rl, remove_mean_rs, reset_nlfs_vars, rotate_spherical_polar_grid, rotate_uv2en_rl, rotate_uv2en_rs, solve_for_pressure, solve_pentadiagonal, solve_tridiagonal, solve_uv_tridiago, sw_adtg, sw_ptmp, sw_temp, swfrac, taueddy_init_varia, taueddy_tendency_apply_u, taueddy_tendency_apply_v, timestep, timestep_wvel, tracers_iigw_correction, turnoff_model_io, update_cg2d, update_etah, update_etaws, update_masks_etc, update_r_star, update_sigma
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/diagnostics: diag_calc_psivel, diag_cg2d, diag_vegtile_fill, diagnostics_addtolist, diagnostics_calc_phivel, diagnostics_check, diagnostics_clear, diagnostics_clrdiag, diagnostics_count, diagnostics_cumulate, diagnostics_fill, diagnostics_fill_field, diagnostics_fill_rs, diagnostics_fill_state, diagnostics_fract_fill, diagnostics_get_diag, diagnostics_get_pointers, diagnostics_hf_cumul, diagnostics_ini_io, diagnostics_init_early, diagnostics_init_fixed, diagnostics_init_varia, diagnostics_interp_p2p, diagnostics_interp_vert, diagnostics_is_on, diagnostics_list_check, diagnostics_main_init, diagnostics_mnc_out, diagnostics_mnc_set, diagnostics_out, diagnostics_read_pickup, diagnostics_scale_fill, diagnostics_scale_fill_rs, diagnostics_set_calc, diagnostics_set_levels, diagnostics_set_pointers, diagnostics_setdiag, diagnostics_setklev, diagnostics_status_error, diagnostics_sum_levels, diagnostics_summary, diagnostics_switch_onoff, diagnostics_write, diagnostics_write_adj, diagnostics_write_pickup, diags_get_parms_i, diags_mk_title, diags_mk_units, diags_renamed, diags_track_diva, diagstats_ascii_out, diagstats_calc, diagstats_clear, diagstats_close_io, diagstats_clrdiag, diagstats_fill, diagstats_g_calc, diagstats_global, diagstats_ini_io, diagstats_lm_calc, diagstats_local, diagstats_mnc_out, diagstats_output, diagstats_set_pointers, diagstats_set_regions, diagstats_setdiag
- pkg/generic_advdiff: gad_biharm_r, gad_biharm_x, gad_biharm_y, gad_c2_adv_r, gad_c2_adv_x, gad_c2_adv_y, gad_c2_impl_r, gad_c4_adv_r, gad_c4_adv_x, gad_c4_adv_y, gad_del2, gad_diag_sufx, gad_diagnostics_init, gad_diagnostics_state, gad_diff_x, gad_diff_y, gad_dst2u1_adv_r, gad_dst2u1_adv_x, gad_dst2u1_adv_y, gad_dst2u1_impl_r, gad_dst3_adv_r, gad_dst3_adv_x, gad_dst3_adv_y, gad_dst3fl_adv_r, gad_dst3fl_adv_x, gad_dst3fl_adv_y, gad_dst3fl_impl_r, gad_exch_som, gad_fluxlimit_adv_r, gad_fluxlimit_adv_x, gad_fluxlimit_adv_y, gad_fluxlimit_impl_r, gad_grad_x, gad_grad_y, gad_implicit_r, gad_os7mp_adv_r, gad_os7mp_adv_x, gad_os7mp_adv_y, gad_osc_hat_y, gad_osc_loc_y, gad_osc_mul_y, gad_plm_fun_v, gad_ppm_adv_r, gad_ppm_adv_x, gad_ppm_adv_y, gad_ppm_flx_r, gad_ppm_flx_x, gad_ppm_flx_y, gad_ppm_fun_mono, gad_ppm_fun_null, gad_ppm_hat_r, gad_ppm_hat_x, gad_ppm_hat_y, gad_ppm_p3e_r, gad_ppm_p3e_x, gad_ppm_p3e_y, gad_pqm_flx_y, gad_pqm_hat_y, gad_pqm_p5e_y, gad_read_pickup, gad_som_adv_r, gad_som_adv_x, gad_som_adv_y, gad_som_advect, gad_som_exchanges, gad_som_fill_cs_corner, gad_som_lim_r, gad_som_prep_cs_corner, gad_u3_adv_r, gad_u3_adv_x, gad_u3_adv_y, gad_u3c4_impl_r, salt_fill
- pkg/mdsio: mds_buffertorl, mds_buffertors, mds_check4file, mds_facef_read_rs, mds_pass_r4torl, mds_pass_r4tors, mds_rd_rec_rl, mds_rd_rec_rs, mds_read_meta, mds_read_sec_xz, mds_read_sec_yz, mds_read_tape, mds_read_whalos, mds_readvec_loc, mds_seg4torl, mds_seg4torl_2d, mds_seg4tors, mds_seg4tors_2d, mds_seg8torl, mds_seg8torl_2d, mds_seg8tors, mds_seg8tors_2d, mds_write_sec_xz, mds_write_sec_yz, mds_write_tape, mds_write_whalos, mds_writelocal, mdsreadfield, mdsreadfield_2d_gl, mdsreadfield_3d_gl, mdsreadfield_loc, mdsreadfield_xz_gl, mdsreadfield_yz_gl, mdsreadfieldxz, mdsreadfieldxz_loc, mdsreadfieldyz, mdsreadfieldyz_loc, mdswritefield, mdswritefield_2d_gl, mdswritefield_3d_gl, mdswritefield_loc, mdswritefield_xz_gl, mdswritefield_yz_gl, mdswritefieldxz, mdswritefieldxz_loc, mdswritefieldyz, mdswritefieldyz_loc
- pkg/monitor: admonitor, g_monitor, mon_calc_advcfl_glob, mon_calc_advcfl_tile, mon_calc_stats_rs, mon_out_rs, mon_printstats_rl, mon_stats_latbnd_rl, mon_stats_rl, mon_surfcor, mon_vort3, mon_writestats_rs, nlatbnd
- pkg/rw: get_write_global_fld, read_fld_xy_rl, read_fld_xy_rs, read_fld_xyz_rs, read_glvec_rl, read_glvec_rs, read_mflds_3d_rl, read_mflds_check, read_mflds_lev_rl, read_mflds_lev_rs, read_mflds_rename, read_mflds_set, read_rec_3d_rl, read_rec_3d_rs, read_rec_lev_rl, read_rec_lev_rs, read_rec_xy_rl, read_rec_xyz_rl, read_rec_xyz_rs, read_rec_xz_rl, read_rec_xz_rs, read_rec_yz_rl, read_rec_yz_rs, rw_get_suffix, write_fld_3d_rl, write_fld_3d_rs, write_fld_s3d_rl, write_fld_s3d_rs, write_local_rl, write_local_rs, write_rec_3d_rs, write_rec_lev_rl, write_rec_lev_rs, write_rec_xy_rl, write_rec_xy_rs, write_rec_xyz_rl, write_rec_xyz_rs, write_rec_xz_rl, write_rec_xz_rs, write_rec_yz_rl, write_rec_yz_rs

