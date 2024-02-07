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
git config --global --add safe.directory /ai_tools/third_party/lib_nn
git config --global --add safe.directory /ai_tools/third_party/lib_tflite_micro
git config --global --add safe.directory /ai_tools/third_party/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro
git describe --tags

CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ make -C third_party/lib_tflite_micro patch 
CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ ./build.sh -T xinterpreter-nozip -b


# Build xcore-opt
# Crosstool toolchain info is mentioned here, "--crosstool_top"
# https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
cd xformer
bazel build //:xcore-opt --verbose_failures --linkopt=-lrt --crosstool_top="@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain" --//:disable_version_check -c dbg --spawn_strategy=local --javacopt="-g" --copt="-g" --strip="never" --sandbox_debug


# Build python wheel
cd ../python
python setup.py bdist_wheel
