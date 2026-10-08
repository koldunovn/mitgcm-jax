"""Execute one notebook with nbclient and measure it (helper of mitjax/tests/test_notebooks_laptop.py, docs plan
20261006 Task 8).

    python -m mitjax.tests.nb_exec <notebook.ipynb> <executed.ipynb> <measures.json> [--timeout S]

Runs every cell in a fresh `python3` kernel with the notebook's own directory as working directory (the notebooks
write their runs to relative paths), stops at the first failing cell (a failing `assert` in a cell is a failing
check), writes the executed notebook (also when a cell failed, with the error in its output) and a JSON record:
wall time, the peak resident memory of the kernel (`ru_maxrss` of this process's waited-for children: the kernel
is the only child), the failing cell's error name and value. Exit code 0 only when every cell ran.
Needs nbclient / nbformat / ipykernel (pyproject extra `notebooks`; on Levante the overlay venv of docs/ENV.md).
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path


def run(nb_in, nb_out, timeout):
    import nbclient
    import nbformat
    nb = nbformat.read(nb_in, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=timeout, kernel_name="python3", record_timing=False,
                                     resources={"metadata": {"path": str(Path(nb_in).parent)}})
    t0 = time.monotonic()
    err = None
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as e:
        err = {"ename": e.ename, "evalue": e.evalue}
    except nbclient.exceptions.DeadKernelError as e:      # the kernel process died (lane NBC): record it, keep outputs
        err = {"ename": "DeadKernelError", "evalue": str(e)[-2000:]}
    wall = time.monotonic() - t0
    nbformat.write(nb, nb_out)
    kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rss = kb * (1 if sys.platform == "darwin" else 1024)              # Linux: KiB, macOS: bytes
    return {"notebook": Path(nb_in).name, "wall_s": round(wall, 1), "peak_rss_bytes": rss,
            "peak_rss_gib": round(rss / 2**30, 2), "error": err}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook")
    ap.add_argument("executed")
    ap.add_argument("measures")
    ap.add_argument("--timeout", type=int, default=3600, help="per cell, seconds")
    a = ap.parse_args(argv)
    rec = run(a.notebook, a.executed, a.timeout)
    Path(a.measures).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    return 0 if rec["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
