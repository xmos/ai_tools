load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

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
        "@llvm-project//mlir:StdOpsTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

cc_library(
    name = "xcore_passes",
    srcs = glob(["transforms/*.cc"]) + [
        "ir/xc_ops.cc",
        "ir/xc_dialect.h.inc",
        "ir/xc_ops.cc.inc",
        "ir/xc_ops.h.inc",
    ],
    hdrs = [
        "transforms/passes.h",
        "ir/xc_ops.h",
        "ir/xc_ops.md",
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
