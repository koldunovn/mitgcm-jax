"""JAX transforms never appear in the physics code (`mitjax/model/`, `mitjax/pkg/`), and `ragged_all_to_all` appears
nowhere in the package.

The physics code reads like the Fortran it ports: it uses jax.numpy and the allow-listed helpers (`scan_k`, `where`,
safe ops); `jit`, `grad`, `vjp`, `jvp`, `custom_jvp/vjp`, `custom_linear_solve`, `checkpoint`/`remat`, `shard_map` and
raw `lax.scan` live in `drivers/`, `ad/`, `eesupp/` and `ops/` (plan, Development Approach); so does
`optimization_barrier` (no barriers in physics code: Nikolay 2026-10-01). Host callbacks (`pure_callback`,
`io_callback`, `jax.debug.callback`) are banned in the physics code too: they leave the traced program, so they have no
derivative rule (AD breaks) and run per device under sharding (REVIEW_M0 #18). `ragged_all_to_all` has a wrong
transpose (ECCO port rule; fesom_jax lessons §8: still wrong in jax 0.11.1).

Not banned, deliberately: `lax.while_loop`, `lax.fori_loop`, `lax.cond`. The CG2D forward kernel (M1 Task 12) is a
loop that stops on the Fortran residual test (cg2d.F), which needs a while_loop to be literal. The solver rule
("never differentiate solver iterations", PORTING_RULES §3) is enforced by the derivative rule instead: the solve is
differentiated only through mitjax/ad/cg2d_rule.py (implicit derivative), whose tests check that the iterations are
never differentiated.

The scan is an AST walk that resolves import aliases (`import jax.lax as L; L.scan`, `from jax import grad as g`), so
renaming does not hide a use. A planted use of each form must be caught (negative controls below).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_DIRS = ("mitjax/model", "mitjax/pkg")

# Last component of a jax.* dotted name that is banned in the physics code.
BANNED_IN_PHYSICS = {"jit", "grad", "value_and_grad", "vjp", "jvp", "linearize", "linear_transpose", "custom_jvp",
                     "custom_vjp", "custom_linear_solve", "checkpoint", "remat", "shard_map", "pmap", "scan",
                     # no barriers in physics code (Nikolay 2026-10-01, docs/LESSONS_CARRIED.md L-CONF-7)
                     "optimization_barrier",
                     # host callbacks: no derivative, per-device under sharding (REVIEW_M0 #18); `callback` is
                     # jax.debug.callback
                     "pure_callback", "io_callback", "callback"}
# Banned in the whole package.
BANNED_EVERYWHERE = {"ragged_all_to_all"}


def _dotted(node):
    """'a.b.c' for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def jax_names(source):
    """Every fully qualified jax.* name a module uses: (line, 'jax.lax.scan'), with import aliases resolved."""
    tree = ast.parse(source)
    alias = {}
    used = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "jax" or a.name.startswith("jax."):
                    if a.asname:
                        alias[a.asname] = a.name
                    else:
                        alias["jax"] = "jax"
        elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "jax"
                                                                    or node.module.startswith("jax.")):
            for a in node.names:
                full = f"{node.module}.{a.name}"
                alias[a.asname or a.name] = full
                used.append((node.lineno, full))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            d = _dotted(node)
            if d is None:
                continue
            head, _, rest = d.partition(".")
            if head in alias:
                used.append((node.lineno, alias[head] + ("." + rest if rest else "")))
    return used


def violations(root, physics_dirs=PHYSICS_DIRS, package="mitjax"):
    """[(file, line, name)] of banned names: transforms in the physics dirs, BANNED_EVERYWHERE in the package."""
    root = Path(root)
    out = set()
    for p in sorted((root / package).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel == "mitjax/tests/test_banned_transforms.py":
            continue
        physics = any(rel.startswith(d + "/") for d in physics_dirs)
        for line, name in jax_names(p.read_text()):
            last = name.rsplit(".", 1)[-1]
            if last in BANNED_EVERYWHERE or (physics and last in BANNED_IN_PHYSICS):
                out.add((rel, line, name))
    return sorted(out)


def test_no_banned_transforms():
    assert violations(REPO_ROOT) == []


def test_negative_controls(tmp_path):
    pkg = tmp_path / "mitjax/pkg/gad"
    pkg.mkdir(parents=True)
    (tmp_path / "mitjax/model/src").mkdir(parents=True)
    (tmp_path / "mitjax/drivers").mkdir(parents=True)
    planted = {
        "mitjax/pkg/gad/a.py": "import jax\ndef f(x):\n    return jax.grad(x)\n",
        "mitjax/pkg/gad/b.py": "from jax import grad as g\n",
        "mitjax/pkg/gad/c.py": "import jax\n@jax.jit\ndef f(x):\n    return x\n",
        "mitjax/pkg/gad/d.py": "from jax import lax\ndef f(b, c, x):\n    return lax.scan(b, c, x)\n",
        "mitjax/model/src/e.py": "import jax.lax as L\ndef f(b, c, x):\n    return L.scan(b, c, x)\n",
        "mitjax/model/src/f.py": "from jax.experimental.shard_map import shard_map\n",
        "mitjax/model/src/g.py": "import jax\ndef f(x):\n    return jax.lax.custom_linear_solve(x)\n",
        "mitjax/drivers/h.py": "import jax\nr = jax.lax.ragged_all_to_all\n",
        "mitjax/pkg/gad/i.py": "from jax import lax\ndef f(x):\n    return lax.optimization_barrier(x)\n",
        "mitjax/pkg/gad/j.py": "import jax\ndef f(cb, s, x):\n    return jax.pure_callback(cb, s, x)\n",
        "mitjax/model/src/k.py": "from jax.experimental import io_callback\n",
        "mitjax/model/src/l.py": "import jax\ndef f(cb, x):\n    jax.debug.callback(cb, x)\n",
        # allowed: the loops (the solver rule is enforced by the derivative rule, see the module docstring)
        "mitjax/model/src/loops.py": "from jax import lax\ndef f(c, b, x):\n    return lax.while_loop(c, b, x)\n",
        # allowed: jax.numpy in physics, transforms in drivers, a local name that merely looks like one
        "mitjax/pkg/gad/ok.py": "import jax.numpy as jnp\nfrom mitjax.ops import scan_k\n"
                                "def f(x, grad):\n    return jnp.where(x > 0, x, grad)\n",
        "mitjax/drivers/ok.py": "import jax\nrun = jax.jit(lambda x: jax.grad(x))\n",
    }
    for rel, text in planted.items():
        (tmp_path / rel).write_text(text)
    hits = {(f, n) for f, _, n in violations(tmp_path)}
    assert hits == {
        ("mitjax/pkg/gad/a.py", "jax.grad"),
        ("mitjax/pkg/gad/b.py", "jax.grad"),
        ("mitjax/pkg/gad/c.py", "jax.jit"),
        ("mitjax/pkg/gad/d.py", "jax.lax.scan"),
        ("mitjax/model/src/e.py", "jax.lax.scan"),
        ("mitjax/model/src/f.py", "jax.experimental.shard_map.shard_map"),
        ("mitjax/model/src/g.py", "jax.lax.custom_linear_solve"),
        ("mitjax/drivers/h.py", "jax.lax.ragged_all_to_all"),
        ("mitjax/pkg/gad/i.py", "jax.lax.optimization_barrier"),
        ("mitjax/pkg/gad/j.py", "jax.pure_callback"),
        ("mitjax/model/src/k.py", "jax.experimental.io_callback"),
        ("mitjax/model/src/l.py", "jax.debug.callback"),
    }, hits
