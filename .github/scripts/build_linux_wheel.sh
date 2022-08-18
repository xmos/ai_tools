#!/usr/bin/env bash
set -e -x

pip install cmake
# Have to add this option due to https://github.com/actions/checkout/issues/760
# and https://github.blog/2022-04-12-git-security-vulnerability-announced/
# This was preventing setuptools-scm from detecting the version as it uses git
git config --global --add safe.directory /ai_tools
git describe --tags
cd third_party/lib_tflite_micro
# Use gcc7 toolchain from the docker file and apply patch
CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ make build
# Use gcc7 toolchain from the docker file to build xinterpreters
cd ../..
CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ make build


# Build xcore-opt
# Crosstool toolchain info is mentioned here, "--crosstool_top"
# https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
cd experimental/xformer
bazel build //:xcore-opt --linkopt=-lrt --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain


# Build python wheel
cd ../../python
python setup.py bdist_wheel
