#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

rm -rf ./debug
set -e

./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif
./generate_conv2d_deepin_deepout_relu.py -v --bias_init const
./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif 0 1
./generate_conv2d_deepin_deepout_relu.py -v --bias_init const 1
./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif 2 3 --seed 42

./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 5 8
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 13
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 21 34 --seed 42

./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init unif
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init const
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init const 1
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init unif 1 2
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init unif
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init const
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init const 3
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init unif 5 8
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 55 89 --bias_init const 13
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 14.4 23.3 --bias_init unif 21 34
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 37.7 --bias_init const 55
./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 610 --bias_init unif 8.9 14.4
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 98.7 159.7 --bias_init const 23.3 --seed 42
./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 25.84 41.81 --bias_init unif 37.7 61.0 --seed 42

./generate_avgpool2d.py -v --input_init unif
./generate_avgpool2d.py -v --input_init const
./generate_avgpool2d.py -v --input_init unif 0 1
./generate_avgpool2d.py -v --input_init const 1
./generate_avgpool2d.py -v --input_init unif 2 3 --seed 42

./generate_avgpool2d_global.py -v --input_init unif
./generate_avgpool2d_global.py -v --input_init const
./generate_avgpool2d_global.py -v --input_init unif 0 1
./generate_avgpool2d_global.py -v --input_init const 1
./generate_avgpool2d_global.py -v --input_init unif 2 3 --seed 42

./generate_conv2d_deepin_deepout_relu.py -v --input_init unif
# broken
#./generate_conv2d_deepin_deepout_relu.py -v --input_init const
./generate_conv2d_deepin_deepout_relu.py -v --input_init unif 0 1
./generate_conv2d_deepin_deepout_relu.py -v --input_init const 1
./generate_conv2d_deepin_deepout_relu.py -v --input_init unif 2 3 --seed 42

./generate_conv2d_shallowin_deepout_relu.py -v --input_init unif
# broken
#./generate_conv2d_shallowin_deepout_relu.py -v --input_init const
./generate_conv2d_shallowin_deepout_relu.py -v --input_init unif 0 1
./generate_conv2d_shallowin_deepout_relu.py -v --input_init const 1
./generate_conv2d_shallowin_deepout_relu.py -v --input_init unif 2 3 --seed 42

./generate_fully_connected.py -v --input_init unif --train_model -ep 1
./generate_fully_connected.py -v --input_init const --train_model -ep 1
./generate_fully_connected.py -v --input_init unif 0 1 --train_model -ep 1
./generate_fully_connected.py -v --input_init const 1 --train_model -ep 1
./generate_fully_connected.py -v --input_init unif 2 3 --seed 42 --train_model -ep 1

./generate_fully_connected_requantized.py -v --input_init unif --train_model -ep 1
./generate_fully_connected_requantized.py -v --input_init const --train_model -ep 1
./generate_fully_connected_requantized.py -v --input_init unif 0 1 --train_model -ep 1
./generate_fully_connected_requantized.py -v --input_init const 1 --train_model -ep 1
./generate_fully_connected_requantized.py -v --input_init unif 2 3 --seed 42 --train_model -ep 1

./generate_lookup_8.py -v --input_init unif
./generate_lookup_8.py -v --input_init const
./generate_lookup_8.py -v --input_init unif 0 1
./generate_lookup_8.py -v --input_init const 1
./generate_lookup_8.py -v --input_init unif 2 3 --seed 42

./generate_maxpool2d.py -v --input_init unif
./generate_maxpool2d.py -v --input_init const
./generate_maxpool2d.py -v --input_init unif 0 1
./generate_maxpool2d.py -v --input_init const 1
./generate_maxpool2d.py -v --input_init unif 2 3 --seed 42

# Broken
#./generate_prelu.py -v --input_init unif
#./generate_prelu.py -v --input_init const
#./generate_prelu.py -v --input_init unif 0 1
#./generate_prelu.py -v --input_init const 1
#./generate_prelu.py -v --input_init unif 2 3 --seed 42
