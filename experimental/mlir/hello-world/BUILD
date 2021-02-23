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
        "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
           ],
    linkstatic = 1,
)
tf_cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
            #"lib-test",
            "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_import",
            "@org_tensorflow//tensorflow/compiler/mlir/lite:flatbuffer_export",
            "@org_tensorflow//tensorflow/lite/schema:schema_fbs_with_reflection",
            "@flatbuffers",
            "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_tfl_passes",
            # "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:Pass",
            "@llvm-project//mlir:Transforms",
            "@llvm-project//mlir:Parser",
            ],

)
