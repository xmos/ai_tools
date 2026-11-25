# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import re
import subprocess
from pathlib import Path

# This test that deps.cmake is up to date with ai_tools submodule

REPO = Path(__file__).resolve().parents[1]
LIB_NN = REPO / "third_party" / "lib_nn"
DEPS_FILE = REPO / "third_party/lib_tflite_micro/lib_tflite_micro/deps.cmake"
LIB_NN_TAG_RE = re.compile(r'^set\(LIB_NN_TAG\s+"([^"]+)"\)', re.MULTILINE)


def _git(*args, cwd, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _rev_parse(ref, cwd):
    """Return the commit for ref, or None if it does not exist locally."""
    res = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=cwd, check=False)
    return res.stdout.strip() if res.returncode == 0 else None


def _read_lib_nn_tag():
    match = LIB_NN_TAG_RE.search(DEPS_FILE.read_text())
    assert match is not None, f"LIB_NN_TAG not found in {DEPS_FILE}"
    return match.group(1)


def _resolve_tag(tag, cwd):
    """Resolve a tag, fetching it from origin if it is not available locally."""
    commit = _rev_parse(f"refs/tags/{tag}", cwd)
    if commit is None:
        _git("fetch", "origin", "tag", tag, "--no-tags", cwd=cwd, check=False)
        commit = _rev_parse(f"refs/tags/{tag}", cwd)
    return commit


def _resolve_branch(branch, cwd):
    return _rev_parse(f"refs/remotes/origin/{branch}", cwd)


def _resolve_dependency(ref, cwd):
    """LIB_NN_TAG may be a tag (release) or a branch (development)."""
    commit = _resolve_tag(ref, cwd) or _resolve_branch(ref, cwd)
    assert commit is not None, f"Cannot resolve '{ref}' as a tag or origin branch in {cwd}"
    return commit


def test_lib_nn_versions_match():
    dependency_commit = _resolve_dependency(_read_lib_nn_tag(), LIB_NN)
    checkout_commit = _git("rev-parse", "HEAD", cwd=LIB_NN).stdout.strip()
    assert dependency_commit == checkout_commit
