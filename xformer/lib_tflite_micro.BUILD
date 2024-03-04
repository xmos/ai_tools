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
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_maxpool2d.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_detection_post.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_load_from_flash.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_lookup.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_softmax.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_add.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_pad.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_concat.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_3_to_4.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_slice.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_broadcast.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_mul.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_binaryi16.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_blob_unaryi16.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_unaryi16.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_beta_activationf32.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_beta_convf32.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_beta_concatf32.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_beta_transposeconvf32.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/xcore_beta_fcf32.cc",
        "lib_tflite_micro/src/tflite-xcore-kernels/conv2d_float.c",
    ],
)
