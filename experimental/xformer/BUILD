load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "dialect_td_files",
    srcs = [
        "IR/XCoreOps.td"
    ],
    includes = ["external/org_tensorflow"],
    deps = [
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
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
        (["-gen-dialect-doc"], "IR/XCoreOps.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IR/XCoreOps.td",
    deps = [
        ":dialect_td_files"
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
        ":dialect_td_files"
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
        ":dialect_td_files"
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
        ":dialect_td_files"
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
        "Transforms/GeneratedLoadConstantOpPatterns.inc"
    ],
    hdrs = [
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
