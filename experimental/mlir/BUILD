load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")
#load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "xc_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/xc_ops.h.inc"),
        ("-gen-op-defs", "ir/xc_ops.cc.inc"),
        ("-gen-dialect-decls -dialect=xc", "ir/xc_dialect.h.inc"),
        ("-gen-dialect-doc", "ir/xc_ops.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/xc_ops.td",
    td_srcs = [
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "patterns_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_patterns.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/patterns.td",
    td_srcs = [
        "ir/xc_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

'''
gentbl(
    name = "prepare_lce_target_arm_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare_target_arm.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_target_arm.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/op_removal_patterns.td",
        "transforms/prepare_patterns_common.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "prepare_lce_target_other_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_prepare_target_other.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_patterns_common.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/op_removal_patterns.td",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

gentbl(
    name = "optimize_lce_target_arm_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize_target_arm.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_target_arm.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/optimize_patterns_common.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "optimize_lce_target_other_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_optimize_target_other.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize_patterns_common.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "bitpack_weights_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_bitpack_weights.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/bitpack_weights_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "transforms/bitpack_weights_patterns.td",
        "transforms/op_removal_patterns.td",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "quantize_lce_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_quantize.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/quantize_patterns.td",
    td_srcs = [
        "ir/lce_ops.td",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)
'''

cc_library(
    name = "xcore_passes",
    srcs = [
        "ir/xc_ops.cc",
        "ir/xc_dialect.h.inc",
        "ir/xc_ops.cc.inc",
        "ir/xc_ops.h.inc",
        "transforms/generated_patterns.inc",
        "transforms/passes.cc",
        "transforms/optimize_fully_connected.cc",
        "transforms/print_nesting.cc",
    ],
    hdrs = [
        "transforms/passes.h",
        "ir/xc_ops.h",
        "ir/xc_ops.md",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        #"@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
    ],
    alwayslink = 1,
) 

cc_library(
    name = "utils",
    hdrs = [
        "utils/file_io.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
        #"@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite",
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