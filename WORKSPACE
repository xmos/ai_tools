workspace(name = "xcore_mlir_experiment")

XMOS_PATH = "./" # TODO use this to move workspace into 'experimental' if desired

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# Skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()
###############################################################################

load("@bazel_skylib//lib:paths.bzl", "paths")

########################### Configure TensorFlow ###############################
# To update TensorFlow to a new revision.
# 1. Update the git hash in the urls and the 'strip_prefix' parameter.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | shasum -a 256
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
http_archive(
    name = "org_tensorflow",
    sha256 = "f681331f8fc0800883761c7709d13cda11942d4ad5ff9f44ad855e9dc78387e0",
    strip_prefix = "tensorflow-2.4.1",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.4.1.tar.gz",
    ],
)

# io_bazel_rules_closure
# This is copied from https://github.com/tensorflow/tensorflow/blob/v2.0.0-alpha0/WORKSPACE.
# Dependency of:
#   TensorFlow (boilerplate for tf_workspace(), apparently)
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Import all of the tensorflow dependencies. Note that we are deliberately
# letting TensorFlow take control of all the dependencies it sets up, whereas
# ours are initialized with `maybe`. Actually tracking this with Bazel is PITA
# and for now this gets TF stuff building. This includes, for instance,
# @llvm-project and @com_google_absl.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")

tf_repositories()

# TF depends on tf_toolchains.
http_archive(
    name = "tf_toolchains",
    sha256 = "d60f9637c64829e92dac3f4477a2c45cdddb9946c5da0dd46db97765eb9de08e",
    strip_prefix = "toolchains-1.1.5",
    urls = [
        "http://mirror.tensorflow.org/github.com/tensorflow/toolchains/archive/v1.1.5.tar.gz",
        "https://github.com/tensorflow/toolchains/archive/v1.1.5.tar.gz",
    ],
)

###############################################################################

##### gRPC Rules for Bazel
##### See https://github.com/grpc/grpc/blob/master/src/cpp/README.md#make
http_archive(
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/de893acb6aef88484a427e64b96727e4926fdcfd.tar.gz",
    ],
    strip_prefix = "grpc-de893acb6aef88484a427e64b96727e4926fdcfd",
    sha256 = "61272ea6d541f60bdc3752ddef9fd4ca87ff5ab18dd21afc30270faad90c8a34"
)
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
# Not mentioned in official docs... mentioned here https://github.com/grpc/grpc/issues/20511
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()