import csv
import shutil
import subprocess

from pathlib import Path
from obtain_and_optimize_mobilenetv2 import get_mobilenetv2, optimize_mobilenetv2

CWD = Path(__file__).parent
TEST_DIR = CWD
CSV_FILE_PATH = CWD / 'arena_sizes.csv'  # Define path for CSV file
BUILD_PATH = TEST_DIR / 'build'

MAX_ARENA_SIZE = 345328
expected_keyword = "LION"

CMAKE_CMD = ["cmake", "-G", "Unix Makefiles", "-B", "build"]
XMAKE_CMD = ["cmake", "--build", "build"]
FLASH_CMD = ["xflash", "--target", "XK-EVK-XU316", "--id", "0", "--data", "xcore_flash_binary.out"]
RUN_CMD = ["xrun", "--xscope", "--id", "0", "bin/app_mobilenetv2.xe"]

def write_csv(arena_size):
    with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([arena_size])

def test_example_output():
    shutil.rmtree(BUILD_PATH, ignore_errors=True)  # Clean build directory before test
    get_mobilenetv2()
    arena_size = optimize_mobilenetv2()
    assert arena_size <= MAX_ARENA_SIZE, f"Optimized model size is too large: {arena_size} bytes"
    write_csv(arena_size)
    subprocess.run(CMAKE_CMD, check=True, cwd=TEST_DIR)
    subprocess.run(XMAKE_CMD, check=True, cwd=TEST_DIR)
    subprocess.run(FLASH_CMD, check=True, cwd=TEST_DIR)
    result = subprocess.run(RUN_CMD, check=True, cwd=TEST_DIR, capture_output=True, text=True)
    assert expected_keyword in result.stdout, f"Output did not contain expected keyword: '{expected_keyword}'"
