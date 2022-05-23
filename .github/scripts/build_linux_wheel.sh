#!/usr/bin/env bash
set -e -x

pip install cmake
# Have to add this option due to https://github.com/actions/checkout/issues/760
# and https://github.blog/2022-04-12-git-security-vulnerability-announced/
# This was preventing setuptools-scm from detecting the version as it uses git
git config --global --add safe.directory /ai_tools
git describe --tags
cd third_party/lib_tflite_micro
# Use gcc7 toolchain from the docker file to build tflm_interpreter
CC=/dt7/usr/bin/gcc CXX=/dt7/usr/bin/g++ make build

# Build xcore-opt with an older gcc7 toolchain
# Have to explicitly link with -lrt to prevent the following linking errors

#/usr/bin/ld: bazel-out/k8-fastbuild/bin/external/org_tensorflow/tensorflow/core/platform/default/libenv_time.a(env_time.pic.o): in function `tensorflow::EnvTime::NowNanos()':
#env_time.cc:(.text+0x15): undefined reference to `clock_gettime'
#/usr/bin/ld: bazel-out/k8-fastbuild/bin/external/com_google_absl/absl/time/libtime.a(clock.pic.o): in function `absl::lts_20210324::time_internal::GetCurrentTimeNanosFromSystem()':
#clock.cc:(.text+0xda): undefined reference to `clock_gettime'
#/usr/bin/ld: bazel-out/k8-fastbuild/bin/external/com_google_absl/absl/base/libbase.a(sysinfo.pic.o): in function `absl::lts_20210324::base_internal::ReadMonotonicClockNanos()':
#sysinfo.cc:(.text+0x156): undefined reference to `clock_gettime'
cd ../../experimental/xformer
bazel build //:xcore-opt --linkopt=-lrt --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda11.2:toolchain

# Build python wheel
cd ../../python
python setup.py bdist_wheel
