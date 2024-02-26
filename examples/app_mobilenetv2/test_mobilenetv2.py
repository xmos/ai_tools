import subprocess
import pytest
import os
from obtain_and_optimize_mobilenetv2 import get_mobilenetv2, optimize_mobilenetv2

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_ARENA_SIZE = 345328

@pytest.fixture
def build_and_run_example():
    os.chdir(TEST_DIR)
    get_mobilenetv2()
    arena_size = optimize_mobilenetv2()
    assert arena_size <= MAX_ARENA_SIZE, f"Optimized model size is too large: {arena_size} bytes"
    subprocess.run(["xmake"], check=True, cwd=TEST_DIR)
    subprocess.run(["xflash", "--target", "XCORE-AI-EXPLORER", "--data", "xcore_flash_binary.out"], check=True, cwd=TEST_DIR)
    result = subprocess.run(["xrun", "--xscope", "bin/app_mobilenetv2.xe"], check=True, cwd=TEST_DIR, stderr=subprocess.PIPE, text=True)
    return result.stderr

def test_example_output(build_and_run_example):
    expected_keyword = "LION"
    assert expected_keyword in build_and_run_example, f"Output did not contain expected keyword: '{expected_keyword}'"
