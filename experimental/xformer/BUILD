load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
)

gentbl(
    name = "xc_ops_inc_gen",
    tbl_outs = [
        ("-gen-enum-decls", "IR/XCoreEnumOps.h.inc"),
        ("-gen-enum-defs", "IR/XCoreEnumOps.cpp.inc"),
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
    name = "xc_patterns_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "Transforms/GeneratedXCPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/XCPatterns.td",
    td_includes = ["external/org_tensorflow", "external/lce"],
    td_srcs = [
        "IR/XCoreOps.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@lce//larq_compute_engine/mlir:lce_ops_td_file",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "tfl_patterns_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "Transforms/GeneratedTFLPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/TFLPatterns.td",
    td_includes = ["external/org_tensorflow"],
    td_srcs = [
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "load_constant_op_patterns_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "Transforms/GeneratedLoadConstantOpPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/LoadConstantOpPatterns.td",
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
        "IR/XCoreEnumOps.h.inc",
        "IR/XCoreEnumOps.cpp.inc",
        "IR/XCoreOps.cpp",
        "IR/XCoreDialect.h.inc",
        "IR/XCoreOps.cpp.inc",
        "IR/XCoreOps.h.inc",
        "Transforms/GeneratedXCPatterns.inc",
        "Transforms/GeneratedTFLPatterns.inc",
        "Transforms/GeneratedLoadConstantOpPatterns.inc"
    ],
    hdrs = [
        "Transforms/ConvPatterns.h",
        "Transforms/Options.h",
        "Transforms/Passes.h",
        "Utils/FileIO.h",
        "IR/XCoreOps.h",
    ],
    deps = [
        ":xc_ops_inc_gen",
        ":xc_patterns_inc_gen",
        ":tfl_patterns_inc_gen",
        ":load_constant_op_patterns_inc_gen",
        "@lce//larq_compute_engine/mlir:lce_tfl_passes",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@lib_nn//:libnn_lib",
    ],
    copts = ["-DNN_USE_REF -Iexternal/lib_nn/lib_nn/api"],
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
