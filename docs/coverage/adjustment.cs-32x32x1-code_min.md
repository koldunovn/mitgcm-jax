# Coverage: adjustment.cs-32x32x1 (code_min) — M2 porting worklist

Written by `tools/coverage.py` (mitjax d117407) from the gcov build `adjustment.cs-32x32x1-code_min-63cdc0b-78ca390-gcov` (oracle flags + `--coverage`, `reference/coverage/`) and its runs; do not edit by hand. Line numbers are `file.F:line` at upstream 63cdc0b. Every compiled `.f` was regenerated with the project's CPP module and matched byte for byte, then mapped to `.F` lines through `cpp` line markers (`tools/cpp_live.py`): 93 files from the link farm, 205 from the build directory (genmake2-generated sources the farm does not hold).

Columns: *calls* = gcov function execution count; *lines run* = instrumented lines executed / instrumented lines; *unexecuted* = runs of instrumented lines with count 0 inside an executed routine (`.F` lines of the routine's file unless prefixed); *never taken* = executed lines with a branch arc never taken (`line:zero arcs/arcs`; e.g. an IF whose THEN never ran).

| variant | run | %MON blocks | exit / normal end | executed routines | physics-active | gcov run vs plain oracle (%MON, cg2d, %SBO, cost lines) |
|---|---|---|---|---|---|---|
| `input_min` | `job27832650` | 0 | 0 / 1 | 35 | yes | identical (0 lines) |

## adjustment.cs-32x32x1/input_min

Run `$MJX_REFERENCE/coverage/adjustment.cs-32x32x1/input_min/job27832650` (STDOUT `rundir/output.txt`); counts `$MJX_REFERENCE/coverage/adjustment.cs-32x32x1/input_min/job27832650/gcov-d117407/gcov.json.gz`.

### Physics-active check

| check | measured | active |
|---|---|---|
| W2 topology set-up (EEBOOT) | w2_eeboot (1 calls) | yes |
| exch2 tile connectivity | w2_e2setup (1 calls) | yes |

### Run-time switches from STDOUT not set in the data files (defaults or derived by the code)

- `.TRUE.`: W2_useE2ioLayOut
- `.FALSE.`: debugMode, printMapIncludesZeros, useCoupler, useNest2W_child, useNest2W_parent, usingMPI
- selectors: -

### Executed routines (35)

**eesupp/src** (22)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `bar2_init` | bar2.F:7 | 4 | 10/10 | - | - |
| `bar_check` | bar_check.F:7 | 1 | 13/18 | 62-68 | 43:1/2, 61:1/2 |
| `barrier` | barrier.F:50 | 3 | 39/61 | 140-143, 148-151, 166-167, 194-197, 212-213, 230-233, 248-249 | 139:1/2, 147:1/2, 155:1/2, 159:1/2, 162:1/2, 182:1/2, 193:1/2, 201:1/2, 205:1/2, 208:1/2, 223:1/2, 229:1/2, 237:1/2, 241:1/2, 244:1/2, 259:1/2 |
| `barrier_init` | barrier.F:7 | 1 | 11/11 | - | - |
| `check_threads` | check_threads.F:7 | 1 | 13/23 | 136-151, 155-157 | 132:1/2, 135:1/2, 153:1/2, 164:1/2 |
| `eeboot` | eeboot.F:8 | 1 | 29/29 | - | 124:1/2, 132:1/2, 144:1/2, 150:1/2 |
| `eeboot_minimal` | eeboot_minimal.F:8 | 1 | 13/18 | 280-285 | 102:1/2, 279:1/2 |
| `eedie` | eedie.F:7 | 1 | 9/18 | 40-42, 58-64 | 37:1/2, 54:1/2, 56:1/2 |
| `eeintro_msg` | eeintro_msg.F:7 | 1 | 30/30 | - | - |
| `eeset_parms` | eeset_parms.F:7 | 1 | 57/78 | 175-185, 247-252, 278-283, 327-328 | 172:1/2, 173:1/2, 191:1/2, 211:1/2, 218:1/2, 243:1/2, 244:1/2, 277:1/2, 317:1/2, 325:1/2 |
| `eewrite_eeenv` | eewrite_eeenv.F:7 | 1 | 99/99 | - | - |
| `exch_init` | exch_init.F:7 | 1 | 11/11 | - | - |
| `fool_the_compiler` | fool_the_compiler.F:15 | 9 | 2/2 | - | - |
| `ifnblnk` | utils.F:88 | 3 | 9/10 | 112 | 108:1/2, 109:1/2 |
| `ilnblnk` | utils.F:123 | 1326 | 10/10 | - | - |
| `ini_communication_patterns` | ini_communication_patterns.F:8 | 1 | 118/126 | 269, 274, 279, 284, 289, 294, 299, 304 | 119:1/2, 146:1/2, 169:1/2, 173:1/2, 191:1/2, 195:1/2, 268:1/2, 271:1/2, 278:1/2, 281:1/2, 288:1/2, 291:1/2, 298:1/2, 301:1/2 |
| `ini_procs` | ini_procs.F:7 | 1 | 13/13 | - | - |
| `ini_threading_environment` | ini_threading_environment.F:8 | 1 | 55/82 | 90-99, 114-119, 124-129, 134-139 | 88:1/2, 112:1/2, 122:1/2, 132:1/2, 141:1/2 |
| `mds_flush` | mds_flush.F:7 | 2 | 3/3 | - | - |
| `mdsfindunit` | mdsfindunit.F:3 | 1 | 11/19 | 44-50, 62-64 | 39:1/2, 42:1/2, 52:1/2, 60:1/2 |
| `nml_set_terminator` | nml_set_terminator.F:7 | 3 | 6/6 | - | 44:1/2 |
| `print_message` | print.F:23 | 350 | 21/31 | 90, 96-99, 131, 135, 151-155 | 86:1/2, 94:1/2, 101:1/2, 109:1/2, 124:1/2, 128:1/2, 130:1/2, 150:1/2 |

**pkg/exch2** (11)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `exch2_get_scal_bounds` | exch2_get_scal_bounds.F:7 | 408 | 38/50 | 65, 71-72, 82, 88-89, 99, 105-106, 116, 122-123 | 62:1/2, 67:1/2, 79:1/2, 84:1/2, 96:1/2, 101:1/2, 113:1/2, 118:1/2 |
| `w2_e2setup` | w2_e2setup.F:10 | 1 | 35/63 | 64-78, 95-101, 108-113, 119, 121, 123, 127 | 63:1/2, 94:1/2, 105:1/2, 106:2/2, 118:1/2, 120:1/2, 122:1/2, 124:1/2 |
| `w2_eeboot` | w2_eeboot.F:7 | 1 | 56/56 | - | 85:1/2, 106:1/2, 111:1/2 |
| `w2_map_procs` | w2_map_procs.F:7 | 1 | 55/69 | 101, 113-117, 121-125, 130, 152-155 | 75:1/2, 94:1/2, 99:1/2, 111:1/2, 119:1/2, 129:1/2, 149:1/2, 150:1/2, 157:1/2 |
| `w2_print_comm_sequence` | w2_print_comm_sequence.F:8 | 1 | 52/52 | - | - |
| `w2_readparms` | w2_readparms.F:9 | 1 | 63/89 | 69, 111-127, 133-134, 139-140, 166-172, 182-187, 190 | 66:1/2, 110:1/2, 132:1/2, 135:1/2, 161:1/2, 163:1/2, 165:1/2, 179:1/2, 181:1/2, 189:1/2 |
| `w2_set_cs6_facets` | w2_set_cs6_facets.F:9 | 1 | 67/99 | 54-55, 87-92, 97-98, 113-121, 125-126, 166-180 | 53:1/2, 86:1/2, 95:1/2, 99:1/2, 105:1/2, 124:1/2, 144:1/2, 152:1/2, 165:1/2 |
| `w2_set_f2f_index` | w2_set_f2f_index.F:9 | 1 | 92/142 | 56-59, 62-65, 72-85, 92-94, 111-117, 186-193, 206-208, 246-253, 260-262 | 54:1/2, 61:1/2, 70:1/2, 90:1/2, 107:1/2, 110:1/2, 176:1/2, 196:1/2, 200:1/2, 204:1/2, 221:1/2, 245:1/2, 258:1/2 |
| `w2_set_map_cumsum` | w2_set_map_cumsum.F:8 | 1 | 85/138 | 89, 122-123, 135-136, 142-165, 173-200, 220-226 | 83:1/2, 88:1/2, 105:1/2, 121:1/2, 134:1/2, 140:1/2, 171:1/2, 219:1/2, 228:1/2, 234:1/4, 241:1/2, 245:1/2, 256:1/2 |
| `w2_set_map_tiles` | w2_set_map_tiles.F:14 | 1 | 83/122 | 69-72, 75-78, 87-89, 94-99, 105-118, 139-144, 193-202 | 68:1/2, 74:1/2, 85:1/2, 92:1/2, 102:1/2, 136:1/2, 172:1/2, 173:2/2, 175:1/2, 189:1/2, 204:1/2 |
| `w2_set_tile2tiles` | w2_set_tile2tiles.F:9 | 1 | 154/193 | 254-260, 282-290, 295-298, 305-307, 318-330, 336-338 | 71:1/2, 125:1/2, 147:1/2, 177:1/2, 221:1/2, 252:1/2, 271:1/2, 279:1/2, 294:1/2, 303:1/2, 317:1/2, 334:1/2 |

**verification/adjustment.cs-32x32x1/code_min** (2)

| routine | file:line | calls | lines run | unexecuted | never taken |
|---|---|---|---|---|---|
| `MAIN_` | main.F:61 | 1 | 20/29 | 133-134, 139, 145-146, 210-214 | 132:1/2, 138:1/2, 144:1/2, 173:1/2, 178:1/2, 193:1/2, 209:1/2 |
| `main` | main.F:221 | 1 | 1/1 | - | - |

### Compiled but not executed (341 routines)

- eesupp/src: all_proc_die, bar2, barrier_ms, barrier_mu, comm_stats, cumulsum_z_tile_rl, date, diff_phase_multiple, different_multiple, eedata_example, exch0_r4, exch0_r8, exch0_rl, exch0_rs, exch1_bg_r4_cube, exch1_bg_r8_cube, exch1_bg_rl_cube, exch1_bg_rl_cube_ad, exch1_bg_rs_cube, exch1_bg_rs_cube_ad, exch1_r4, exch1_r4_cube, exch1_r8, exch1_r8_cube, exch1_rl, exch1_rl_ad, exch1_rl_b, exch1_rl_bwd, exch1_rl_cube, exch1_rl_cube_ad, exch1_rl_cube_b, exch1_rl_cube_bwd, exch1_rl_cube_d, exch1_rl_cube_fwd, exch1_rl_d, exch1_rl_fwd, exch1_rs, exch1_rs_ad, exch1_rs_b, exch1_rs_bwd, exch1_rs_cube, exch1_rs_cube_ad, exch1_rs_cube_b, exch1_rs_cube_d, exch1_rs_d, exch1_rs_fwd, exch1_uv_r4_cube, exch1_uv_r8_cube, exch1_uv_rl_cube, exch1_uv_rl_cube_ad, exch1_uv_rl_cube_b, exch1_uv_rl_cube_d, exch1_uv_rs_cube, exch1_uv_rs_cube_ad, exch1_uv_rs_cube_b, exch1_uv_rs_cube_d, exch1_z_r4_cube, exch1_z_r8_cube, exch1_z_rl_cube, exch1_z_rl_cube_ad, exch1_z_rs_cube, exch1_z_rs_cube_ad, exch_3d_r4, exch_3d_r8, exch_3d_rl, exch_3d_rs, exch_cycle_ebl, exch_r4_recv_get_x, exch_r4_recv_get_y, exch_r4_send_put_x, exch_r4_send_put_y, exch_r8_recv_get_x, exch_r8_recv_get_y, exch_r8_send_put_x, exch_r8_send_put_y, exch_rl_recv_get_x, exch_rl_recv_get_y, exch_rl_send_put_x, exch_rl_send_put_y, exch_rs_recv_get_x, exch_rs_recv_get_y, exch_rs_send_put_x, exch_rs_send_put_y, exch_s3d_r4, exch_s3d_r8, exch_s3d_rl, exch_s3d_rs, exch_sm_3d_r4, exch_sm_3d_r8, exch_sm_3d_rl, exch_sm_3d_rs, exch_uv_3d_r4, exch_uv_3d_r8, exch_uv_3d_rl, exch_uv_3d_rs, exch_uv_agrid_3d_r4, exch_uv_agrid_3d_r8, exch_uv_agrid_3d_rl, exch_uv_agrid_3d_rs, exch_uv_bgrid_3d_r4, exch_uv_bgrid_3d_r8, exch_uv_bgrid_3d_rl, exch_uv_bgrid_3d_rs, exch_uv_dgrid_3d_r4, exch_uv_dgrid_3d_r8, exch_uv_dgrid_3d_rl, exch_uv_dgrid_3d_rs, exch_uv_xy_r4, exch_uv_xy_r8, exch_uv_xy_rl, exch_uv_xy_rs, exch_uv_xyz_r4, exch_uv_xyz_r8, exch_uv_xyz_rl, exch_uv_xyz_rs, exch_xy_r4, exch_xy_r8, exch_xy_rl, exch_xy_rs, exch_xyz_r4, exch_xyz_r8, exch_xyz_rl, exch_xyz_rs, exch_z_3d_r4, exch_z_3d_r8, exch_z_3d_rl, exch_z_3d_rs, fill_cs_corner_ag_rl, fill_cs_corner_tr_rl, fill_cs_corner_uv_rl, fill_cs_corner_uv_rs, fill_halo_local_r4, fill_halo_local_r8, fill_halo_local_rl, fill_halo_local_rs, fool_the_compiler_r8, fool_the_compiler_rl, gather_2d_r4, gather_2d_r8, gather_2d_wh_r4, gather_2d_wh_r8, gather_vec_r4, gather_vec_r8, gather_xz, gather_yz, get_periodic_interval, glb_sum_vec, global_max_r4, global_max_r8, global_sum_int, global_sum_r4, global_sum_r8, global_sum_singlecpu_rl, global_sum_tile_rl, global_sum_vec_alt_rl, global_sum_vec_alt_rs, global_sum_vector_int, global_sum_vector_rl, global_sum_vector_rs, io_errcount, lcase, lef_zero, machine, master_cpu_io, master_cpu_thread, mds_byteswapi4, mds_byteswapr4, mds_byteswapr8, mds_reclen, memsync, nml_change_syntax, open_copy_data_file, print_error, print_list_i, print_list_l, print_list_rl, print_maprl, print_maprs, reset_halo_rl, reset_halo_rs, scatter_2d_r4, scatter_2d_r8, scatter_2d_wh_r4, scatter_2d_wh_r8, scatter_vec_r4, scatter_vec_r8, scatter_xz, scatter_yz, stop_if_error, timer_control, timer_get_time, timer_index, timer_printall, timer_start, timer_stop, ucase, write_0d_c, write_0d_i, write_0d_l, write_0d_r4, write_0d_r8, write_0d_rl, write_0d_rs, write_1d_i, write_1d_l, write_1d_rl, write_copy1d_r4, write_copy1d_r8, write_copy1d_rs, write_xy_xline_rs, write_xy_yline_rs
- pkg/debug: chksum_tiled, debug_call, debug_cs_corner_uv, debug_enter, debug_fld_stats_rl, debug_fld_stats_rs, debug_leave, debug_msg, debug_stats_rl, debug_stats_rs, fill_in_corners_rl, write_fullarray_rl, write_fullarray_rs
- pkg/exch2: exch2_3d_r4, exch2_3d_r8, exch2_3d_rl, exch2_3d_rs, exch2_ad_get_r41, exch2_ad_get_r42, exch2_ad_get_r81, exch2_ad_get_r82, exch2_ad_get_rl1, exch2_ad_get_rl2, exch2_ad_get_rs1, exch2_ad_get_rs2, exch2_ad_put_r41, exch2_ad_put_r42, exch2_ad_put_r81, exch2_ad_put_r82, exch2_ad_put_rl1, exch2_ad_put_rl2, exch2_ad_put_rs1, exch2_ad_put_rs2, exch2_check_depths, exch2_get_r41, exch2_get_r42, exch2_get_r81, exch2_get_r82, exch2_get_rl1, exch2_get_rl2, exch2_get_rs1, exch2_get_rs2, exch2_get_uv_bounds, exch2_put_r41, exch2_put_r42, exch2_put_r81, exch2_put_r82, exch2_put_rl1, exch2_put_rl2, exch2_put_rs1, exch2_put_rs2, exch2_r41_cube, exch2_r41_cube_ad, exch2_r42_cube, exch2_r42_cube_ad, exch2_r81_cube, exch2_r81_cube_ad, exch2_r82_cube, exch2_r82_cube_ad, exch2_recv_r41, exch2_recv_r42, exch2_recv_r81, exch2_recv_r82, exch2_recv_rl1, exch2_recv_rl2, exch2_recv_rs1, exch2_recv_rs2, exch2_rl1_cube, exch2_rl1_cube_ad, exch2_rl1_cube_b, exch2_rl1_cube_d, exch2_rl2_cube, exch2_rl2_cube_ad, exch2_rl2_cube_b, exch2_rl2_cube_d, exch2_rs1_cube, exch2_rs1_cube_ad, exch2_rs1_cube_b, exch2_rs1_cube_d, exch2_rs2_cube, exch2_rs2_cube_ad, exch2_rs2_cube_b, exch2_rs2_cube_d, exch2_s3d_r4, exch2_s3d_r8, exch2_s3d_rl, exch2_s3d_rs, exch2_send_r41, exch2_send_r42, exch2_send_r81, exch2_send_r82, exch2_send_rl1, exch2_send_rl2, exch2_send_rs1, exch2_send_rs2, exch2_sm_3d_r4, exch2_sm_3d_r8, exch2_sm_3d_rl, exch2_sm_3d_rs, exch2_uv_3d_r4, exch2_uv_3d_r8, exch2_uv_3d_rl, exch2_uv_3d_rs, exch2_uv_agrid_3d_r4, exch2_uv_agrid_3d_r8, exch2_uv_agrid_3d_rl, exch2_uv_agrid_3d_rs, exch2_uv_bgrid_3d_r4, exch2_uv_bgrid_3d_r8, exch2_uv_bgrid_3d_rl, exch2_uv_bgrid_3d_rs, exch2_uv_cgrid_3d_r4, exch2_uv_cgrid_3d_r8, exch2_uv_cgrid_3d_rl, exch2_uv_cgrid_3d_rs, exch2_uv_dgrid_3d_r4, exch2_uv_dgrid_3d_r8, exch2_uv_dgrid_3d_rl, exch2_uv_dgrid_3d_rs, exch2_z_3d_r4, exch2_z_3d_r8, exch2_z_3d_rl, exch2_z_3d_rs, find_gcd_n, w2_cumulsum_z_tile_rl, w2_print_e2setup, w2_set_gen_facets, w2_set_myown_facets, w2_set_single_facet
- verification/adjustment.cs-32x32x1/code_min: ptreentry

