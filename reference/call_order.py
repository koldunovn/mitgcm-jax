#!/usr/bin/env python3
"""Call order of a debugMode run of the Fortran oracle (plan Task 3b item 1, lesson [E§1]: the step order is per
experiment and must be read from the run, not assumed).

    call_order.py RUN_TOP OUT_FILE

RUN_TOP is a run directory made with `make_rundir.py --set eedata:EEPARMS:debugMode=.TRUE.` (debugMode is an EEPARMS
variable, eeset_parms.F:72, default .FALSE. at :125) and run by reference/run.sh. With debugMode the model prints,
through pkg/debug (DEBUG_ENTER / DEBUG_CALL / DEBUG_LEAVE -> DEBUG_MSG, pkg/debug/debug_msg.F: `'DEBUG_MSG: '` + text,
A11,A60), one line per instrumented call: `ENTERED S/R X`, `CALLING S/R X`, `LEAVING S/R X` (and a few
package-specific `ENTERING ...` lines). This script writes OUT_FILE with
  * the initialisation sequence (every message before the first `ENTERED S/R FORWARD_STEP`), indented by the
    ENTERED/LEAVING nesting;
  * the first time step (from the first `ENTERED S/R FORWARD_STEP` to the message before the second one), indented;
  * for every later time step, whether its message sequence equals the first step's (else the first differing
    position, both messages), and the tail after the last step.
The text of each message is kept as printed (A60: truncated at 60 characters by the Fortran format). The debug run's
other output (STDOUT numbers, files) is never used as a yardstick: its overlay makes it a different run.
OUT_FILE must not exist. Stdlib only.
"""

import argparse
import re
import sys
from pathlib import Path

MSG = re.compile(r"^\(PID\.TID \d+\.\d+\) DEBUG_MSG: (.*?)\s*$")
STEP = "ENTERED S/R FORWARD_STEP"


def messages(stdout_lines):
    return [m[1] for m in (MSG.match(ln) for ln in stdout_lines) if m]


def indented(msgs, depth0=0):
    out, depth = [], depth0
    for m in msgs:
        if m.startswith("LEAVING S/R"):
            depth = max(depth - 1, 0)
        out.append("  " * depth + m)
        if m.startswith("ENTERED S/R"):
            depth += 1
    return out


def split_steps(msgs):
    """(init, [step 1, step 2, ...], tail): steps start at each `ENTERED S/R FORWARD_STEP`; the last step ends at
    its matching `LEAVING S/R FORWARD_STEP` (or the end), the rest is the tail."""
    starts = [i for i, m in enumerate(msgs) if m == STEP]
    if not starts:
        return msgs, [], []
    steps = [msgs[a:b] for a, b in zip(starts, starts[1:])]
    last = msgs[starts[-1]:]
    end = next((i for i, m in enumerate(last) if m == "LEAVING S/R FORWARD_STEP"), None)
    if end is None:
        steps.append(last)
        tail = []
    else:
        # the message after FORWARD_STEP's LEAVING still belongs to the step loop (e.g. CALLING S/R ... in
        # main_do_loop) until the next step; for the last step everything after LEAVING is the tail
        steps.append(last[:end + 1])
        tail = last[end + 1:]
    # earlier steps: keep their own text up to the next ENTERED FORWARD_STEP (includes the loop's calls)
    return msgs[:starts[0]], steps, tail


def report(msgs, label):
    init, steps, tail = split_steps(msgs)
    lines = [f"# call order (debugMode) of {label}", f"# {len(msgs)} DEBUG_MSG lines, {len(steps)} time steps", "",
             f"## initialisation ({len(init)} messages)"] + indented(init)
    if steps:
        first = steps[0]
        lines += ["", f"## time step 1 ({len(first)} messages)"] + indented(first)
        lines += ["", "## later time steps compared with step 1"]
        for n, s in enumerate(steps[1:], start=2):
            cmp_len = min(len(s), len(first))
            # the last step has no loop calls after LEAVING FORWARD_STEP: compare the common prefix up to it
            body = s if n < len(steps) else s
            ref = first if n < len(steps) else first[:len(s)]
            if body == ref:
                lines.append(f"step {n}: identical to step 1 ({len(s)} messages)")
            else:
                k = next((i for i in range(cmp_len) if s[i] != first[i]), cmp_len)
                a = first[k] if k < len(first) else "<end>"
                b = s[k] if k < len(s) else "<end>"
                lines.append(f"step {n}: differs from step 1 at message {k + 1}: step 1 {a!r}, step {n} {b!r} "
                             f"({len(s)} vs {len(first)} messages)")
    lines += ["", f"## after the last time step ({len(tail)} messages)"] + indented(tail)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_top")
    ap.add_argument("out_file")
    a = ap.parse_args(argv)
    stdout = Path(a.run_top) / "rundir" / "output.txt"
    msgs = messages(stdout.read_text(errors="replace").split("\n"))
    if not msgs:
        raise SystemExit(f"no DEBUG_MSG lines in {stdout} (not a debugMode run?)")
    out = Path(a.out_file)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    label = "/".join(Path(a.run_top).resolve().parts[-3:])
    out.write_text(report(msgs, label))
    print(f"WROTE {out}: {len(msgs)} messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
