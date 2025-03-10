#!/bin/bash

set -e
OS=$(uname)

if [ "$OS" = "Linux" ]; then
  NUM_PROCS=$(nproc)
elif [ "$OS" = "Darwin" ]; then
  NUM_PROCS=$(sysctl -n hw.ncpu)
else
  echo "Unsupported operating system."
  exit 1
fi

ACTION="--build"
TARGET=""
DEBUG="false"
LSP="false"
SCRIPT_DIR="$(
  cd "$(dirname "$0")"
  pwd -P
)"
ARCH=$(uname -m)
MACHINE_ARCH=""
if [ "$ARCH" = "x86_64" ]; then
  MACHINE_ARCH="x86"
elif [[ "$ARCH" == *"arm"* ]]; then
  MACHINE_ARCH="arm"
else
  echo "Unknown architecture"
  exit 1
fi

help() {
  echo "Usage: $(basename "$0") [ACTIONS]..."
  echo "  -b                Build (default)"
  echo "  -c                Clean build"
  echo "  -t                Test build"
  echo "  -d                Enable debug"
  echo "  -j [NUM_PROCS]    Set number of jobs (default: nproc)"
  echo "  -T [TARGET]       Set target:"
  echo "       init         Initialise repository (update submodules and patch ltflm)"
  echo "       all          Build everything"
  echo "       xinterpreter Build interpreter only"
  echo "       xformer      Build compiler only"
  echo "  -h                Show this help message"
  exit 1
}

while getopts "cbtdj:T:hl" opt; do
  case $opt in
  c)
    ACTION="--clean"
    ;;
  b)
    ACTION="--build"
    ;;
  t)
    ACTION="--test"
    ;;
  d)
    DEBUG="true"
    ;;
  j)
    NUM_PROCS="$OPTARG"
    ;;
  T)
    TARGET="$OPTARG"
    ;;
  h)
    help
    ;;
  l)
    LSP="true"
    ;;
  *)
    echo "Invalid option: -$OPTARG" >&2
    help
    ;;
  esac
done

if [ -z "$TARGET" ]; then
  echo "No target specified."
  help
fi

bazel_compile_commands() {
  cd xformer
  bazel run @hedron_compile_commands//:refresh_all
  cd "$SCRIPT_DIR"
}

build_xformer() {
  if [ "$LSP" = "true" ]; then
    bazel_compile_commands
  fi
  cd xformer
  bazel_cmd="bazel build --jobs $NUM_PROCS //:xcore-opt"
  if [ "$MACHINE_ARCH" = "arm" ]; then
    bazel_cmd+=" --cpu=darwin_arm64"
  fi
  if [ "$DEBUG" = "true" ]; then
    bazel_cmd+=" -c dbg --spawn_strategy=local --javacopt=\"-g\" --copt=\"-g\" --strip=\"never\" --verbose_failures --sandbox_debug"
  fi
  eval "$bazel_cmd"
  cd "$SCRIPT_DIR"
}

version_check() {
  cd xformer
  ./version_check.sh
  cd "$SCRIPT_DIR"
}

submodule_update() {
  git submodule update --init --recursive --jobs "$NUM_PROCS"
}

patch() {
  make -C third_party/lib_tflite_micro patch
}

unsupported_action() {
  echo "Action $ACTION not supported for target $TARGET"
  exit 1
}

create_zip() {
  cd third_party/lib_tflite_micro
  mkdir -p build
  cd build
  if [ "$1" = "xcore" ]; then
    cmake .. --toolchain=../lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake
  else
    cmake .. -DLIB_NAME=x86tflitemicro
  fi
  make create_zip "-j$NUM_PROCS"
  cd "$SCRIPT_DIR"
  mv third_party/lib_tflite_micro/build/release_archive.zip python/xmos_ai_tools/runtime/release_archive.zip
  cd python/xmos_ai_tools/runtime
  rm -rf include
  unzip -o release_archive.zip
  rm release_archive.zip
  cd "$SCRIPT_DIR"
}

build_xinterpreter() {
  cd "$SCRIPT_DIR"
  make -C python/xmos_ai_tools/xinterpreters install "-j$NUM_PROCS"
}

xformer_integration_test() {
  pytest integration_tests/runner.py --models_path integration_tests/models/8x8 -n "$NUM_PROCS"
  pytest integration_tests/runner.py --models_path integration_tests/models/16x8 -n "$NUM_PROCS"
  pytest integration_tests/runner.py --models_path integration_tests/models/float32 -n "$NUM_PROCS"
  pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n"$NUM_PROCS"
}

clean_xinterpreter() {
  make -C python/xmos_ai_tools/xinterpreters clean
}

clean_runtime() {
  rm -rf third_party/lib_tflite_micro/build
}

test_xinterpreter() {
  echo "Not implemented yet"
  exit 1
}

# we want this script to build the repository it's in, no matter where we call it from
cd "$SCRIPT_DIR"

case $TARGET in
init)
  submodule_update
  patch
  ;;
xformer)
  build_xformer
  ;;
xinterpreter)
  case $ACTION in
  --build)
    version_check
    # create_zip "xcore"
    clean_runtime
    create_zip "x86"
    build_xinterpreter
    ;;
  --clean)
    clean_xinterpreter
    clean_runtime
    ;;
  --test)
    test_xinterpreter
    ;;
  *)
    unsupported_action
    ;;
  esac
  ;;
# this is a mess: xinterpreter-nozip only used for CI
xinterpreter-nozip)
  case $ACTION in
  --build)
    version_check
    build_xinterpreter
    ;;
  --clean)
    clean_xinterpreter
    ;;
  --test)
    test_xinterpreter
    ;;
  esac
  ;;
all)
  case $ACTION in
  --build)
    version_check
    build_xformer
    create_zip "xcore"
    build_xinterpreter
    ;;
  --clean)
    clean_xinterpreter
    ;;
  --test)
    xformer_integration_test
    ;;
  *)
    unsupported_action
    ;;
  esac
  ;;
*)
  echo "Unknown target: $TARGET"
  help
  ;;
esac
