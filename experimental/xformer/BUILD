load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "xc_dialect_td",
    srcs = [
        "IR/XCoreOps.td"
    ],
    includes = ["external/org_tensorflow"],
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

td_library(
    name = "tfl_patterns_td",
    srcs = [
        "Transforms/TFLPatterns.td",
    ],
    includes = ["external/org_tensorflow", "external/lce"],
    deps = [
        "@lce//larq_compute_engine/mlir:lce_ops_td_file",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

td_library(
    name = "xc_patterns_td",
    srcs = [
        "Transforms/XCPatterns.td",
    ],
    includes = ["external/org_tensorflow", "external/lce"],
    deps = [
        ":xc_dialect_td",
        "@lce//larq_compute_engine/mlir:lce_ops_td_file",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
    ],
)

td_library(
    name = "load_constant_op_patterns_td",
    srcs = [
        "Transforms/LoadConstantOpPatterns.td",
    ],
    includes = ["external/org_tensorflow"],
    deps = [
        ":xc_dialect_td",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
    ],
)

td_library(
    name = "conv_patterns_td",
    srcs = [
        "Transforms/ConvPatterns.td",
    ],
    includes = ["external/org_tensorflow"],
    deps = [
        ":xc_dialect_td",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
    ],
)

td_library(
    name = "conv_revert_patterns_td",
    srcs = [
        "Transforms/ConvRevertPatterns.td",
    ],
    includes = ["external/org_tensorflow"],
    deps = [
        ":xc_dialect_td",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "xc_ops_inc_gen",
    tbl_outs = [
        (["-gen-enum-decls"], "IR/XCoreEnumOps.h.inc"),
        (["-gen-enum-defs"], "IR/XCoreEnumOps.cpp.inc"),
        (["-gen-op-decls"], "IR/XCoreOps.h.inc"),
        (["-gen-op-defs"], "IR/XCoreOps.cpp.inc"),
        (["-gen-dialect-decls", "-dialect=xc"], "IR/XCoreDialect.h.inc"),
        (["-gen-dialect-defs", "-dialect=xc"], "IR/XCoreDialect.cpp.inc"),
        (["-gen-dialect-doc", "-dialect=xc"], "IR/XCoreOps.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/XCoreOps.td",
    deps = [
        ":xc_dialect_td",
    ],
)

gentbl_cc_library(
    name = "xc_patterns_inc_gen",
    tbl_outs = [
        (["-gen-rewriters"], "Transforms/GeneratedXCPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/XCPatterns.td",
    deps = [
        ":xc_patterns_td",
    ],
)

gentbl_cc_library(
    name = "tfl_patterns_inc_gen",
    tbl_outs = [
        (["-gen-rewriters"], "Transforms/GeneratedTFLPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/TFLPatterns.td",
    deps = [
        ":tfl_patterns_td",
    ],
)

gentbl_cc_library(
    name = "load_constant_op_patterns_inc_gen",
    tbl_outs = [
        (["-gen-rewriters"], "Transforms/GeneratedLoadConstantOpPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/LoadConstantOpPatterns.td",
    deps = [
        ":load_constant_op_patterns_td",
    ],
)

gentbl_cc_library(
    name = "conv_patterns_inc_gen",
    tbl_outs = [
        (["-gen-rewriters"], "Transforms/GeneratedConvPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/ConvPatterns.td",
    deps = [
        ":conv_patterns_td",
    ],
)

gentbl_cc_library(
    name = "conv_revert_patterns_inc_gen",
    tbl_outs = [
        (["-gen-rewriters"], "Transforms/GeneratedConvRevertPatterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Transforms/ConvRevertPatterns.td",
    deps = [
        ":conv_revert_patterns_td",
    ],
)

cc_library(
    name = "xcore_passes",
    srcs = glob(["Transforms/*.cpp"]) + [
        "IR/XCoreEnumOps.h.inc",
        "IR/XCoreEnumOps.cpp.inc",
        "IR/XCoreOps.cpp",
        "IR/XCoreDialect.h.inc",
        "IR/XCoreDialect.cpp.inc",
        "IR/XCoreOps.cpp.inc",
        "IR/XCoreOps.h.inc",
        "Transforms/GeneratedXCPatterns.inc",
        "Transforms/GeneratedTFLPatterns.inc",
        "Transforms/GeneratedLoadConstantOpPatterns.inc",
        "Transforms/GeneratedConvPatterns.inc",
        "Transforms/GeneratedConvRevertPatterns.inc"
    ],
    hdrs = [
        "Transforms/ConvPatterns.h",
        "Transforms/Options.h",
        "Transforms/Passes.h",
        "Utils/FileIO.h",
        "Utils/ThreadSupport.h",
        "IR/XCoreOps.h",
    ],
    deps = [
        ":xc_ops_inc_gen",
        ":xc_patterns_inc_gen",
        ":tfl_patterns_inc_gen",
        ":load_constant_op_patterns_inc_gen",
        ":conv_patterns_inc_gen",
        ":conv_revert_patterns_inc_gen",
        "@lce//larq_compute_engine/mlir:larq_compute_engine_translate_tflite",
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
    srcs = ["XCoreOptMain.cpp", "Version.h"],
    deps = [
        ":xcore_passes",
        ":utils",
        "@lib_tflite_micro//:shared_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
