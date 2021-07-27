load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
)

gentbl(
    name = "xc_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "IR/XCoreOps.h.inc"),
        ("-gen-op-defs", "IR/XCoreOps.cpp.inc"),
        ("-gen-dialect-decls -dialect=xc", "IR/XCoreDialect.h.inc"),
        ("-gen-dialect-doc", "IR/XCoreOps.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/XCoreOps.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "patterns_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "Transforms/GeneratedPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/Patterns.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        "IR/XCoreOps.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

cc_library(
    name = "xcore_passes",
    srcs = glob(["Transforms/*.cpp"]) + [
        "IR/XCoreOps.cpp",
        "IR/XCoreDialect.h.inc",
        "IR/XCoreOps.cpp.inc",
        "IR/XCoreOps.h.inc",
        "Transforms/GeneratedPatterns.inc"
    ],
    hdrs = [
        "Transforms/Passes.h",
        "IR/XCoreOps.h",
        "IR/XCoreOps.md",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@lib_nn//:libnn_lib",
    ],
    copts = ["-Iexternal/lib_nn/lib_nn/api"],
    alwayslink = 1,
) 

cc_library(
    name = "utils",
    srcs = glob(["Utils/*.cpp"]),
    hdrs = glob(["Utils/*.h"]),
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
    srcs = ["XCoreOptMain.cpp"],
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
