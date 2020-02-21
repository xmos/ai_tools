#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

rm -rf ./debug
set -e

./generate_argmax_16.py -v

# pooling
./generate_avgpool2d.py -v
./generate_avgpool2d_global.py -v
./generate_maxpool2d.py -v

# convolutions
./generate_conv2d_deepin_deepout_relu.py -v
./generate_conv2d_shallowin_deepout_relu.py -v

# fully connected
./generate_fully_connected.py -v --train_model -ep 1
./generate_fully_connected.py -v
./generate_fully_connected_requantized.py -v --train_model -ep 1
./generate_fully_connected_requantized.py -v

# activations
./generate_relu.py -v
./generate_relu6.py -v
