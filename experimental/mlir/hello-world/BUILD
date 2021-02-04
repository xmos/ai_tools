load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "filegroup", "get_compatible_with_cloud")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")
load("@org_tensorflow//third_party/mlir:tblgen.bzl","gentbl",)


cc_library(
    name = "lib-test",
    srcs = ["lib-test.cc"],
    hdrs = ["lib-test.h"],
    deps = [ 
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
           ],
    linkstatic = 1,
)
tf_cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = ["lib-test"],

)
