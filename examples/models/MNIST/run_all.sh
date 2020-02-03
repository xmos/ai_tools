#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

rm -rf ./debug
set -e

./generate_lenet5.py -v --train_model -ep 1 -bs 128
./generate_lenet5.py -v
./generate_lenet5.py -v --train_model -ep 1 -bs 128 --xcore_tuned --classifier
./generate_lenet5.py -v --xcore_tuned --classifier
./generate_lenet5.py -v --train_model -ep 1 -bs 128 --xcore_tuned
./generate_lenet5.py -v --xcore_tuned

./generate_mlp.py -v --train_model -ep 1 -bs 128
./generate_mlp.py -v
./generate_mlp.py -v --train_model -ep 1 -bs 128 --xcore_tuned --classifier
./generate_mlp.py -v --xcore_tuned --classifier
./generate_mlp.py -v --train_model -ep 1 -bs 128 --xcore_tuned
./generate_mlp.py -v --xcore_tuned

./generate_simard.py -v --train_model -ep 1 -bs 128
./generate_simard.py -v
./generate_simard.py -v --train_model -ep 1 -bs 128 --xcore_tuned --classifier
./generate_simard.py -v --xcore_tuned --classifier
./generate_simard.py -v --train_model -ep 1 -bs 128 --xcore_tuned
./generate_simard.py -v --xcore_tuned

./generate_logistic_regression.py -v --train_model -ep 1 -bs 128
./generate_logistic_regression.py -v
./generate_logistic_regression.py -v --train_model -ep 1 -bs 128 --classifier
./generate_logistic_regression.py -v --classifier
