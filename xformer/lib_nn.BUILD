package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "nn_lib",
    hdrs = glob([
        "lib_nn/api/**/*.h*",
        "lib_nn/src/**/*.h*",
    ]),
    srcs = glob([
        "lib_nn/src/**/*.c",
        "lib_nn/src/**/*.cpp",
    ]),
    includes = ["../../external/lib_nn/lib_nn/api"],
    local_defines = ["NN_USE_REF"],
    deps = [],
    alwayslink = 1,
)
