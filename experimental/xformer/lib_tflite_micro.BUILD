package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "shared_headers",
    srcs = [
        "lib_tflite_micro/api/version.h",
        "lib_tflite_micro/api/xcore_shared_config.h",
    ],
)

filegroup(
    name = "XTFLIB_SOURCES",
    srcs = [
        #"lib_tflite_micro/src/inference_engine.cc",
        #"lib_tflite_micro/src/thread_call_host.c",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_dispatcher.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_error_reporter.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_interpreter.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_profiler.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_ops.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_utils.cc",
    ],
)

filegroup(
    name = "XTFLIB_KERNEL_SOURCES",
    srcs = [
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_common.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_custom_options.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_bsign.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_conv2d_v2.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_detection_post.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_load_from_flash.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_lookup.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_strided_slice.cc",
    ],
)

# link error on Linux
#list(APPEND ALL_SOURCES  "${TOP_DIR}/lib_tflite_micro/submodules/flatbuffers/src/util.cpp")