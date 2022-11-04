#!/usr/bin/env bash
set -e -x

# get latest pip
pip uninstall pip --yes
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install cmake

# Have to add this option due to https://github.com/actions/checkout/issues/760
# and https://github.blog/2022-04-12-git-security-vulnerability-announced/
# This was preventing setuptools-scm from detecting the version as it uses git
git config --global --add safe.directory /ai_tools
git config --global --add safe.directory /ai_tools/third_party/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro
git describe --tags

cd third_party/lib_tflite_micro
# Use gcc9 toolchain from the docker file and apply patch
CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ make build
# Use gcc9 toolchain from the docker file to build xinterpreters
cd ../..
CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ make build


# Build xcore-opt
# Crosstool toolchain info is mentioned here, "--crosstool_top"
# https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
cd experimental/xformer
bazel build //:xcore-opt --linkopt=-lrt --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain --execution_log_binary_file=../../logs/exec.log


# Build python wheel
cd ../../python
python setup.py bdist_wheel
