load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")


cc_library(
    name = "lib-test",
    srcs = ["lib-test.cc"],
    hdrs = ["lib-test.h"],
    deps = ["@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",],
    linkstatic = 1
)
cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = ["lib-test"],

)
