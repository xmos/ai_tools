load("//Test:lit_test.bzl", "xcore_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
)

xcore_lit_test_suite(
    name = "lit",
    srcs = glob(["*.mlir"]),
    data = [
        "//:xcore-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
    driver = "//Test:run_lit.sh",
)
