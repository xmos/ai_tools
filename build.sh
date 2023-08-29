NUM_PROCS=$(nproc)
ACTION="--build"
TARGET=""
DEBUG="false"
LSP="false"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ARCH=$(uname -m)
MACHINE_ARCH=""
if [ "$ARCH" = "x86_64" ] ; then
  MACHINE_ARCH="x86"
elif [[ "$ARCH" == *"arm"* ]] ; then
  MACHINE_ARCH="arm"
else
  echo "Unknown architecture"
  exit 1
fi

TEMP=$(getopt -o "c,b,t,d,j:T:h,l" --long "clean,build,test,debug,jobs:,target:,help,lsp" -n "$(basename "$0")" -- "$@")

help() {
    echo "Usage: $(basename "$0") [ACTIONS]..."
    echo "  -b, --build       Build (default)"
    echo "  -c, --clean       Clean build"
    echo "  -t, --test        Test build"
    echo "  -d, --debug       Enable debug"
    echo "  -j, --jobs [N]    Set number of jobs (default: nproc)"
    echo "  -T, --target [T]  Set target:"
    echo "                      init         Initialise repository (update submodules and patch ltflm)"
    echo "                      all          Build everything"
    echo "                      xinterpreter Build interpreter only"
    echo "                      xformer      Build compiler only"
    echo "  -h, --help        Show this help message"
    exit 1
}

bazel_compile_commands() {
    cd xformer
    bazel run @hedron_compile_commands//:refresh_all
    cd $SCRIPT_DIR
}

build_xformer() {
    if [ "$LSP" = "true" ] ; then
        bazel_compile_commands
    fi
    cd xformer
    if [ "$MACHINE_ARCH" = "x86" ] ; then
        bazel build --jobs $NUM_PROCS //:xcore-opt
    else
        bazel build --jobs $NUM_PROCS //:xcore-opt --cpu=darwin_arm64
    fi
    cd $SCRIPT_DIR
}

version_check() {
    cd xformer
    ./version_check.sh
    cd $SCRIPT_DIR
}

submodule_update() {
    git submodule update --init --recursive --jobs $NUM_PROCS
}

patch() {
    make -C third_party/lib_tflite_micro patch
}

unsupported_action() {
    echo "Action $ACTION not supported for target $TARGET"
    exit 1
}

build_xinterpreter() {
    cd third_party/lib_tflite_micro
    mkdir -p build
    cd build
    cmake .. --toolchain=../lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake
    make create_zip -j$NUM_PROCS
    cd $SCRIPT_DIR
    mv third_party/lib_tflite_micro/build/release_archive.zip python/xmos_ai_tools/runtime/release_archive.zip
    cd python/xmos_ai_tools/runtime
    rm -rf lib include
    unzip release_archive.zip
    rm release_archive.zip
    cd $SCRIPT_DIR
    if [ "$LSP" = "true" ] ; then
        bear make -C python/xmos_ai_tools/xinterpreters/host install -j$NUM_PROCS
    else
        make -C python/xmos_ai_tools/xinterpreters/host install -j$NUM_PROCS
    fi
}

xformer_integration_test() {
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n $NUM_PROCS --junitxml=integration_tests/integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n $NUM_PROCS --junitxml=integration_tests/integration_bnns_junit.xml
}

clean_xinterpreter() {
    make -C python/xmos_ai_tools/xinterpreters/host clean
}

test_xinterpreter() {
    echo "Not implemented yet"
    exit 1
}

if [ $? -ne 0 ] || [ "$TEMP" = " -- " ] ; then
    echo "Error parsing options."
    help
fi

eval set -- "$TEMP"
while true; do
  case "$1" in
    --help|-h)
      help;;
    --clean|-c)
      ACTION="--clean"; shift;;
    --build|-b)
      ACTION="--build"; shift;;
    --test|-t)
      ACTION="--test"; shift;;
    --debug|-d)
      DEBUG="true"; shift;;
    --lsp|-l)
      LSP="true"; shift;;
    --jobs|-j)
      NUM_PROCS="$2"; shift 2;;
    --target|-T)
      TARGET="$2"; shift 2;;
    --)
      shift; break;;
    *)
  esac
done

if [ -z "$TARGET" ]; then
    echo "No target specified."
    help
fi

# we want this script to build the repository it's in, no matter where we call it from
cd $SCRIPT_DIR

case $TARGET in
  init)
    submodule_update
    patch
    ;;
  xformer)
    case $ACTION in
      --build)
        build_xformer
        ;;
      *)
        unsupported_action
        ;;
    esac
    ;;
  xinterpreter)
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
      *)
        unsupported_action
        ;;
    esac
    ;;
  all)
    case $ACTION in
      --build)
        build_xformer
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
