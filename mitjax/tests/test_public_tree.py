"""tools/make_public_tree.py (docs plan 20261006 Task 11, decision 13): the content list covers this tree, and on a
small throw-away repository the build exports exactly the publish set into a new directory, commits it once with no
remote, refuses an existing directory, and its scan catches every forbidden class in a file and in the commit message.

One test (tier 1 counts tests against a budget of 100). Negative controls, each must bite: a sample of every scan
class planted into a published file (the build then does not commit), a marker in the commit message, a string from
the extra-strings file (reported without the string), an unclassified tracked file (refused), an existing
destination (refused), a test-only tree (its scan fails), and the levante.env allowance (the two allowed classes
pass there; a third class there, and the same two in any other file, are hits). The samples are assembled at run
time so that this file holds none of them. Stdlib + git; seconds.
"""

import importlib.util
import os
import subprocess

import pytest

from mitjax import paths

REPO = paths.REPO


def _tool():
    spec = importlib.util.spec_from_file_location("_mjx_make_public_tree", REPO / "tools" / "make_public_tree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.hooksPath=/dev/null", *args], check=True, capture_output=True, text=True)


def _samples():
    """One planted line per scan class (assembled here, never written literally in this file)."""
    u = "-".join(("0123abcd", "4567", "89ab", "cdef", "0123456789ab"))
    return {
        "assistant name": "made with " + "Cl" + "aude",
        "co-author trailer": "Co" + "-Authored" + "-By: someone",
        "generated-with marker": "Gener" + "ated with a tool",
        "session directory": "/tmp/" + "cl" + "aude-24253/x",
        "scratch path": "/scr" + "atch/a/b",
        "UUID": "id " + u,
        "account id": "user " + "a27" + "0088",
        "project work path": "/work/" + "ab0" + "995/x",
        "home path": "/home/" + "a/" + "x",
        "draft marker": "Copyright [" + "DRAFT: holder]",
        "test-only tree": "TEST ONLY - NOT FOR " + "PUBLICATION",
    }


def _repo_files():
    """Files this tree holds: `git ls-files` in a checkout; in a tier runner's snapshot (git archive, no .git) every
    file but byte-code."""
    if (REPO / ".git").exists():
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], check=True, capture_output=True).stdout
        return sorted(p for p in out.decode().split("\0") if p)
    return sorted(p.relative_to(REPO).as_posix() for p in REPO.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)


def test_public_tree_build_scan_and_controls(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.invalid")

    # 1. the content list classifies every file of this tree (a new top-level path cannot slip through)
    unclassified = [f for f in _repo_files() if tool.classify(f) is None]
    assert unclassified == [], unclassified
    # kept-out names are split into adjacent strings: they are test cases of the rules, not citations (the dangling
    # reference report of `list` would count them)
    for keep in ("docs/pl" "ans/x.md", "docs/hand" "off/lane_x.md", "docs/HAND" "OFF-20261008.md", "CL" "AUDE.md",
                 ".cl" "aude/agents/x.md", "docs/PUBLIC_" "CONTENT.md", "docs/brain" "storm-x.md",
                 "docs/PORTING_RULES_" "INTERNAL.md"):
        assert tool.classify(keep)[0] == tool.KEEP, keep
    for pub in ("mitjax/api.py", "LICENSE", "docs/install.md", "notebooks/01_quickstart.ipynb", "levante.env",
                "docs/PORTING_RULES.md"):
        assert tool.classify(pub)[0] == tool.PUBLISH, pub

    # 2. a throw-away repository: two published files, one kept out
    src = tmp_path / "src"
    (src / "mitjax").mkdir(parents=True)
    (src / "docs" / "plans").mkdir(parents=True)
    (src / "README.md").write_text("# public\n")
    (src / "mitjax" / "__init__.py").write_text("X = 1\n")
    (src / "docs" / "plans" / "p.md").write_text("internal\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "c1")
    dest = tmp_path / "out" / "tree"
    hits = tool.build("HEAD", dest, "public commit\n", repo=src)
    assert hits == [], hits
    assert _git(dest, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert _git(dest, "ls-files").stdout.split() == ["README.md", "mitjax/__init__.py"]
    assert _git(dest, "remote").stdout.strip() == ""
    assert _git(dest, "log", "-1", "--format=%B").stdout.strip() == "public commit"
    assert tool.scan_tree(dest) == []
    with pytest.raises(FileExistsError):                       # control: an existing destination is refused
        tool.build("HEAD", dest, "again\n", repo=src)

    # 3. controls: every class planted into a published file is found, and the build then does not commit
    samples = _samples()
    for cls, line in samples.items():
        found = {h[2] for h in tool.scan_text("f", "clean line\n" + line + "\n")}
        assert cls in found, (cls, found)
    (src / "README.md").write_text("# public\n" + "\n".join(samples.values()) + "\n")
    _git(src, "commit", "-q", "-am", "c2")
    dest2 = tmp_path / "out" / "tree2"
    hits = tool.build("HEAD", dest2, "public commit\n", repo=src)
    assert {h[2] for h in hits} == set(samples), hits
    assert all(h[0] == "README.md" for h in hits) and {h[1] for h in hits} == set(range(2, 2 + len(samples)))
    assert not (dest2 / ".git").exists()

    # 3c. dangling references: a published text naming a kept-out file (its path or, for docs/ and scripts/, its
    # name) is reported; the same name inside another project's path (a lessons source tag) is not
    kept = ["docs/PORTING_" "LESSONS.md", "docs/hand" "off/lane_x.md", "scripts/x_" "gate.sbatch"]
    texts = {"a.md": "\n".join([
        "see `docs/PORTING_" "LESSONS.md` entry",                    # 1: path
        "see PORTING_" "LESSONS.md and (lane_x.md section 2)",       # 2: two names
        "run x_" "gate.sbatch",                                      # 3: script name
        "in docs/hand" "off/ notes",                                 # 4: the handoff directory
        "the docs/HAND" "OFF-20261008.md handoff",                   # 5: session handoffs
        "source P2/PORTING_" "LESSONS.md:12; notes/HAND" "OFF-20260923.md:3; R/docs/PORTING_" "LESSONS.md",
        "clean line"])}
    assert sorted((ln, t) for _, ln, t in tool.dangling(texts, kept)) == [
        (1, kept[0]), (2, kept[0]), (2, kept[1]), (3, kept[2]), (4, "docs/hand" "off/"), (5, "docs/HAND" "OFF-*")]

    # 4. controls: a marker in the commit message; an extra string, reported without the string itself
    # 3b. the one per-file allowance (Nikolay 2026-10-08): levante.env, published as the site example, may hold the
    # account id and the project work path; any other class there, and the same two anywhere else, stay hits
    site = "\n".join((samples["account id"], samples["project work path"], samples["home path"])) + "\n"
    assert [h[2] for h in tool.scan_text("levante.env", site)] == ["home path"]
    assert sorted(h[2] for h in tool.scan_text("docs/levante.env", site)) == ["account id", "home path",
                                                                             "project work path"]
    assert {c for v in tool.ALLOW.values() for c in v} <= set(tool.SCAN) and set(tool.ALLOW) == {"levante.env"}
    msg_hits = tool.scan_text("<commit message>", "public\n\n" + samples["co-author trailer"] + "\n")
    assert [h[2] for h in msg_hits] == ["co-author trailer"]
    secret = "zq" + "-private-token-" + "7731"
    extra = tmp_path / "extra.txt"
    extra.write_text("# one per line\n" + secret + "\n")
    xh = tool.scan_text("f", "a\n" + secret.upper() + " b\n", tool.load_extra(extra))
    assert [(h[1], h[2], h[3]) for h in xh] == [(2, "extra string #2", "(not shown)")]
    assert secret not in repr(xh)

    # 5. a test-only build commits despite the hits, and the scan of that tree (files + git log) then always fails
    dest_t = tmp_path / "out" / "tree_t"
    assert tool.build("HEAD", dest_t, "m\n", repo=src, test_only=True)
    assert _git(dest_t, "log", "-1", "--format=%s").stdout.startswith(tool.TEST_ONLY)
    assert tool.main(["scan", str(dest_t)]) == 1

    # 6. control: an unclassified tracked file is refused before anything is written
    (src / "stray.txt").write_text("x\n")
    _git(src, "add", "stray.txt")
    _git(src, "commit", "-q", "-m", "c3")
    with pytest.raises(ValueError, match="stray.txt"):
        tool.build("HEAD", tmp_path / "out" / "tree3", "m\n", repo=src)
    assert sorted(os.listdir(tmp_path / "out")) == ["tree", "tree2", "tree_t"]   # tree3 was never created
