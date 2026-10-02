# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import re
import subprocess
from pathlib import Path
from string import Template

# This test that deps.cmake is up to date with ai_tools submodule

REPO = Path(__file__).resolve().parents[1]
LIB_NN = REPO / "third_party" / "lib_nn"
DEPS_FILE = REPO / "third_party/lib_tflite_micro/cmakefiles/deps.cmake"
GIT_REV_PARSE = ("git", "rev-parse")
DEPENDENCY_REF = Template("origin/${tag}^{commit}")

def _run(cmds, cwd):
    return subprocess.run(
        cmds,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

def test_lib_nn_versions_match():
    deps = DEPS_FILE.read_text()
    tag = re.search(r'^set\(LIB_NN_TAG\s+"([^"]+)"\)', deps, re.MULTILINE)
    assert tag is not None
    cmd_dep = [*GIT_REV_PARSE, DEPENDENCY_REF.substitute(tag=tag.group(1))]
    dependency_commit = _run(cmd_dep, LIB_NN)
    cmd_repo = [*GIT_REV_PARSE, "HEAD"]
    checkout_commit = _run(cmd_repo, LIB_NN)
    assert dependency_commit == checkout_commit
