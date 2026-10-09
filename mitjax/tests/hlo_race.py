"""Find XLA kernels that overwrite a buffer in place while reading it elsewhere: the GPU race of 2026-10-09
(mitjax/xla_flags.py docstring, `multi_output_fusion`).

A fusion whose output is a dynamic-update-slice (or scatter) of one of its parameters writes that output into the
parameter's buffer: the threads of the kernel overwrite the parameter while others may still read it. That is safe
only when every read of the parameter is at the index the same thread writes. The race found: one multi-output fusion
wrote UPDATE_ETAH's etaH := etaN into etaH's buffer (output 3) while it read etaH through a slice, an add and a gather
(the halo exchange of etaN) into another output (4); a thread read etaH at a point another thread had overwritten.

`find(text)` reads an optimized HLO module (`jax.jit(f).lower(...).compile().as_text()`, or XLA's
`*after_optimizations.txt` dump). An output of a fused computation is written in place when it is (through bitcast,
copy, reshape) a dynamic-update-slice or scatter whose operand 0 is (through those and nested updates) a parameter:
XLA's in-place input/output pairs of a fusion. Every other use of that buffer (the parameter, its bitcasts, the nested
updates) is a read, and is accepted when it is
  - a dynamic-slice with the same start indices as an update of the chain and that update's shape (XLA's own rule);
  - a slice or dynamic-slice of an update's shape, or an elementwise op, whose uses reach nothing but the update
    operand of an update of the chain, through elementwise ops (`ELEMENTWISE`) and whole-array slices.
Anything else, e.g. a gather, a transpose, a reduce or a path into another output, makes the output a Race. The check
does not prove that a static slice starts where a dynamic update starts. Stdlib only.
"""

import re
from collections import namedtuple

Race = namedtuple("Race", "computation fusion output op shape parameter reads")

_HEAD = re.compile(r"^\s*(?:ENTRY\s+)?%([\w.\-]+)\s*\(.*\{\s*$")
_INSTR = re.compile(r"^\s*(ROOT\s+)?%([\w.\-]+)\s*=\s*(.+?)\s+([a-z][\w\-]*)\((.*)$")
_SHAPE = re.compile(r"[a-z]+\d*\[[^\]]*\]")
_CALLS = re.compile(r"calls=%?([\w.\-]+)")

CHAIN = ("bitcast", "copy", "reshape")
UPDATES = ("dynamic-update-slice", "scatter")
ELEMENTWISE = frozenset((
    "abs", "add", "and", "atan2", "bitcast", "cbrt", "ceil", "clamp", "compare", "convert", "copy", "cosine",
    "divide", "exponential", "exponential-minus-one", "floor", "is-finite", "log", "log-plus-one", "map", "maximum",
    "minimum", "multiply", "negate", "not", "or", "power", "remainder", "reshape", "round-nearest-afz",
    "round-nearest-even", "rsqrt", "select", "sign", "sine", "sqrt", "subtract", "tan", "tanh", "xor"))


def _operands(rest):
    """The %names inside the first balanced parentheses of an instruction's text after `op(` (jax's
    `compiled.as_text()` and XLA's dumps print every name with its % sigil)."""
    depth, args = 1, []
    for ch in rest:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        args.append(ch)
    return re.findall(r"%([\w.\-]+)", "".join(args))


def parse(text):
    """{computation name: [instruction dict(name, shape, op, ops, root, line)]} of an HLO module's text."""
    comps, cur = {}, None
    for line in text.splitlines():
        if cur is None:
            h = _HEAD.match(line)
            if h and "=" not in line.split("(", 1)[0]:
                cur = h.group(1)
                comps[cur] = []
            continue
        if line.strip() == "}":
            cur = None
            continue
        m = _INSTR.match(line)
        if m:
            root, name, typ, op, rest = m.groups()
            shape = _SHAPE.search(typ)
            comps[cur].append(dict(name=name, shape=shape.group(0) if shape else typ, op=op, ops=_operands(rest),
                                   root=bool(root), line=line.strip()))
    return comps


def _outputs(by, root):
    """The output instructions of a fused computation (the tuple's operands, or the root)."""
    return [by[o] for o in root["ops"] if o in by] if root["op"] == "tuple" else [root]


def _in_place(by, out):
    """(update, parameter, buffer names) when the output `out` is written into a parameter's buffer: `out` is, through
    CHAIN ops, a dynamic-update-slice or scatter whose operand 0 is, through CHAIN ops and nested updates, a parameter
    of the fusion (XLA's in-place input/output pairs of a fusion). None otherwise."""
    buf, n = [], out["name"]
    while by.get(n) and by[n]["op"] in CHAIN:
        buf.append(n)
        n = by[n]["ops"][0]
    if not by.get(n) or by[n]["op"] not in UPDATES:
        return None
    update = by[n]
    while by.get(n) and by[n]["op"] in CHAIN + UPDATES:
        buf.append(n)
        n = by[n]["ops"][0]
    if not by.get(n) or by[n]["op"] != "parameter":
        return None
    buf.append(n)
    return update, n, set(buf)


def _identity(by, u):
    """A slice or dynamic-slice that returns its whole operand (start 0 in every dimension)."""
    return u["op"] in ("slice", "dynamic-slice") and u["shape"] == by.get(u["ops"][0], {}).get("shape")


def _cone_ok(by, users, start, buf):
    """True when every use of `start`, followed forward, reaches only the buffer chain `buf`, as the update operand of
    one of its updates (index 1 of a dynamic-update-slice, 2 of a scatter), through ELEMENTWISE ops and whole-array
    slices."""
    seen, todo = set(), [start]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        for u, k in users.get(n, ()):
            if u["name"] in buf:
                if not ((u["op"] == "dynamic-update-slice" and k == 1) or (u["op"] == "scatter" and k == 2)):
                    return False
                continue
            if (u["op"] not in ELEMENTWISE and not _identity(by, u)) or u["root"]:
                return False
            todo.append(u["name"])
    return True


def _safe_read(by, users, u, buf):
    """A read `u` of the buffer is safe (module docstring)."""
    updates = [by[n] for n in buf if by[n]["op"] == "dynamic-update-slice"]
    if u["op"] == "dynamic-slice" and any(u["ops"][1:] == d["ops"][2:] and u["shape"] == by[d["ops"][1]]["shape"]
                                          for d in updates if d["ops"][1] in by):
        return True
    shapes = {by[d["ops"][1]]["shape"] for d in updates if d["ops"][1] in by}
    if (u["op"] in ("slice", "dynamic-slice") and u["shape"] in shapes) or u["op"] in ELEMENTWISE:
        return _cone_ok(by, users, u["name"], buf)
    return False


def find(text):
    """Every Race of an optimized HLO module (module docstring)."""
    comps = parse(text)
    callers = {}
    for c, ins in comps.items():
        for i in ins:
            m = _CALLS.search(i["line"])
            if m and i["op"] == "fusion":
                callers.setdefault(m.group(1), []).append((c, i["name"]))
    races = []
    for c, ins in comps.items():
        if c not in callers or not ins:
            continue
        by = {i["name"]: i for i in ins}
        users = {}
        for i in ins:
            for k, o in enumerate(i["ops"]):
                users.setdefault(o, []).append((i, k))
        root = next((i for i in ins if i["root"]), ins[-1])
        for n_out, out in enumerate(_outputs(by, root)):
            hit = _in_place(by, out)
            if hit is None:
                continue
            update, param, buf = hit
            bad = []
            for a in sorted(buf):
                for u, k in users.get(a, ()):
                    if u["name"] in buf and k == 0 or (u["op"] == "tuple" and a == out["name"]):
                        continue
                    if not _safe_read(by, users, u, buf):
                        bad.append(f"{u['op']} {u['name']} {u['shape']}")
            if bad:
                caller = callers[c][0]
                races.append(Race(c, f"{caller[1]} in {caller[0]}", n_out, update["op"], out["shape"], param, bad))
    return races


def report(races, top=10):
    """One line per Race (the first `top`)."""
    return "\n".join(f"{r.computation} ({r.fusion}), output {r.output}: {r.op} {r.shape} in place on "
                     f"{r.parameter}; also read by {'; '.join(r.reads[:3])}" for r in races[:top])


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        with open(f) as fh:
            rs = find(fh.read())
        print(f"{f}: {len(rs)} race(s)")
        if rs:
            print(report(rs))
