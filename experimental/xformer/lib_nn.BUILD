package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "libnn_lib2",
    hdrs = glob(["lib_nn/api/*.h*"]) + 
           glob(["lib_nn/api/geom/*.h*"]) + 
           glob(["lib_nn/src/asm/*.h*"]) + 
           glob(["lib_nn/src/*.h*"]),
    srcs = ["lib_nn.dylib"],
    deps = [],
)


cc_library(
    name = "libnn_lib",
    hdrs = glob(["lib_nn/api/*.h*"]) + 
           glob(["lib_nn/api/geom/*.h*"]) + 
           glob(["lib_nn/src/asm/*.h*"]) + 
           glob(["lib_nn/src/*.h*"]),
    srcs = glob(["lib_nn/src/c/*.c"]) + 
           glob(["lib_nn/src/asm/*.c"]) +
           #glob(["lib_nn/src/asm/*.S"]) + 
           glob(["lib_nn/src/cpp/*.cpp"]) + 
           glob(["lib_nn/src/cpp/filt2d/*.cpp"]) + 
           glob(["lib_nn/src/cpp/filt2d/geom/*.cpp"]),
    copts = ["-DNN_USE_REF -Iexternal/lib_nn/lib_nn/api"],
    deps = [],
    alwayslink = 1,
)