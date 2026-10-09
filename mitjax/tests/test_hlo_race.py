"""mitjax/tests/hlo_race.py on small hand-written HLO modules (pure Python, seconds): the race of 2026-10-09 in
miniature (a multi-output fusion writes a parameter in place while a slice of it reaches a gather and another output)
is found; the pattern XLA keeps in place in the fixed programs (a slice of the written region, elementwise into the
update, plus a dynamic-slice at the update's own indices) is not; planted: that safe pattern with the elementwise
value also sent to a second output, and with the slice transposed, is found again. On the GPU dumps of 2026-10-09 the
scanner finds exactly the racing output in the 3-step programs compiled with multi-output fusion (jobs 27994893,
modules 0602 and 0644) and nothing in the programs compiled without it (job 28005413); the GPU test is
test_gpu_race.py."""

from mitjax.tests import hlo_race as H

RACE = """HloModule race

%fused_race (p0: f64[4,8], p1: f64[4,8], p2: s32[6,1]) -> (f64[4,8], f64[6,1]) {
  %p0 = f64[4,8]{1,0} parameter(0)
  %p1 = f64[4,8]{1,0} parameter(1)
  %p2 = s32[6,1]{1,0} parameter(2)
  %c0 = s32[] constant(0)
  %dus = f64[4,8]{1,0} dynamic-update-slice(%p0, %p1, %c0, %c0)
  %sl = f64[4,6]{1,0} slice(%p0), slice={[0:4], [1:7]}
  %bc = f64[24]{0} bitcast(%sl)
  %g = f64[6,1]{1,0} gather(%bc, %p2), offset_dims={1}, collapsed_slice_dims={}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}
  ROOT %t = (f64[4,8]{1,0}, f64[6,1]{1,0}) tuple(%dus, %g)
}

ENTRY %main (a: f64[4,8], b: f64[4,8], i: s32[6,1]) -> (f64[4,8], f64[6,1]) {
  %a = f64[4,8]{1,0} parameter(0)
  %b = f64[4,8]{1,0} parameter(1)
  %i = s32[6,1]{1,0} parameter(2)
  ROOT %f = (f64[4,8]{1,0}, f64[6,1]{1,0}) fusion(%a, %b, %i), kind=kLoop, calls=%fused_race
}
"""

SAFE = """HloModule safe

%fused_safe (p0: f64[6,8], p1: f64[4,6], p2: s32[]) -> f64[6,8] {
  %p0 = f64[6,8]{1,0} parameter(0)
  %p1 = f64[4,6]{1,0} parameter(1)
  %p2 = s32[] parameter(2)
  %b0 = f64[6,8]{1,0} bitcast(%p0)
  %one = s32[] constant(1)
  %ds = f64[4,6]{1,0} dynamic-slice(%b0, %p2, %one), dynamic_slice_sizes={4,6}
  %sl = f64[4,6]{1,0} slice(%b0), slice={[1:5], [1:7]}
  %add = f64[4,6]{1,0} add(%sl, %p1)
  %cmp = pred[4,6]{1,0} compare(%add, %p1), direction=GT
  %sel = f64[4,6]{1,0} select(%cmp, %add, %ds)
  ROOT %dus = f64[6,8]{1,0} dynamic-update-slice(%b0, %sel, %p2, %one)
}

ENTRY %main (a: f64[6,8], b: f64[4,6], i: s32[]) -> f64[6,8] {
  %a = f64[6,8]{1,0} parameter(0)
  %b = f64[4,6]{1,0} parameter(1)
  %i = s32[] parameter(2)
  ROOT %f = f64[6,8]{1,0} fusion(%a, %b, %i), kind=kLoop, calls=%fused_safe
}
"""


def test_hlo_race():
    # the race in miniature: output 0 written into p0 while p0's slice reaches the gather of output 1
    races = H.find(RACE)
    assert [(r.computation, r.output, r.op, r.parameter) for r in races] == \
        [("fused_race", 0, "dynamic-update-slice", "p0")], H.report(races)
    assert races[0].reads == ["slice sl f64[4,6]"] and races[0].fusion == "f in main"
    # XLA's in-place pattern of the fixed programs: no race
    assert H.find(SAFE) == [], H.report(H.find(SAFE))
    # planted 1: the elementwise value of the read also leaves the kernel as a second output
    two = SAFE.replace("-> f64[6,8] {\n  %p0", "-> (f64[6,8], f64[4,6]) {\n  %p0", 1).replace(
        "  ROOT %dus = f64[6,8]{1,0} dynamic-update-slice(%b0, %sel, %p2, %one)",
        "  %dus = f64[6,8]{1,0} dynamic-update-slice(%b0, %sel, %p2, %one)\n"
        "  ROOT %t = (f64[6,8]{1,0}, f64[4,6]{1,0}) tuple(%dus, %add)")
    assert two != SAFE and [r.output for r in H.find(two)] == [0], H.report(H.find(two))
    # planted 2: the read goes through a transpose (another point than the one written)
    tr = SAFE.replace("  %add = f64[4,6]{1,0} add(%sl, %p1)",
                      "  %tp = f64[4,6]{0,1} transpose(%sl), dimensions={0,1}\n"
                      "  %add = f64[4,6]{1,0} add(%tp, %p1)")
    assert tr != SAFE and [r.reads for r in H.find(tr)] == [["slice sl f64[4,6]"]], H.report(H.find(tr))
    # only computations that a fusion calls are scanned: the same computation, uncalled, has no race
    assert H.find(RACE.split("ENTRY")[0]) == []
