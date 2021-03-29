load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

cc_library(
    name = "xcore_passes",
    srcs = glob(["transforms/*.cc"]),
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
) 

cc_library(
    name = "utils",
    srcs = glob(["utils/*.cc"]),
    hdrs = glob(["utils/*.h"]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
    ],
    alwayslink = 1,
) 

cc_binary(
    name = "xcore-opt",
    srcs = ["xcore_opt.cc"],
    deps = [
        ":xcore_passes",
        ":utils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
