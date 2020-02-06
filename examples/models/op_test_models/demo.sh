#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
# Script to test the integration of common_initializers in demo.py
rm -rf ./debug
set -e

./demo.py -v --bias_init
./demo.py -v --bias_init unif
./demo.py -v --bias_init const
./demo.py -v --bias_init unif 0 1
./demo.py -v --bias_init const 1
./demo.py -v --bias_init unif 2 3 --seed_init 42

./demo.py -v --weight_init
./demo.py -v --weight_init unif
./demo.py -v --weight_init const
./demo.py -v --weight_init unif 5 8
./demo.py -v --weight_init const 13
./demo.py -v --weight_init unif 21 34 --seed_init 42

./demo.py -v --weight_init --bias_init
./demo.py -v --weight_init unif --bias_init
./demo.py -v --weight_init unif --bias_init unif
./demo.py -v --weight_init unif --bias_init const
./demo.py -v --weight_init unif --bias_init const 1
./demo.py -v --weight_init unif --bias_init unif 1 2
./demo.py -v --weight_init const --bias_init
./demo.py -v --weight_init const --bias_init unif
./demo.py -v --weight_init const --bias_init const
./demo.py -v --weight_init const --bias_init const 3
./demo.py -v --weight_init const --bias_init unif 5 8
./demo.py -v --weight_init unif 55 89 --bias_init const 13
./demo.py -v --weight_init unif 14.4 23.3 --bias_init unif 21 34
./demo.py -v --weight_init const 37.7 --bias_init const 55
./demo.py -v --weight_init const 610 --bias_init unif 8.9 14.4
./demo.py -v --weight_init unif 98.7 159.7 --bias_init const 23.3 --seed_init 42
./demo.py -v --weight_init unif 25.84 41.81 --bias_init unif 37.7 61.0 --seed_init 42

./generate_conv2d_shallowin_deepout_relu.py -v --bias_init
./generate_conv2d_shallowin_deepout_relu.py -v --bias_init unif
./generate_conv2d_shallowin_deepout_relu.py -v --bias_init const
./generate_conv2d_shallowin_deepout_relu.py -v --bias_init unif 0 1
./generate_conv2d_shallowin_deepout_relu.py -v --bias_init const 1
./generate_conv2d_shallowin_deepout_relu.py -v --bias_init unif 2 3 --seed_init 42

./generate_conv2d_shallowin_deepout_relu.py -v --weight_init
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 5 8
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const 13
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 21 34 --seed_init 42

./generate_conv2d_shallowin_deepout_relu.py -v --weight_init --bias_init
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif --bias_init
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif --bias_init unif
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif --bias_init const
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif --bias_init const 1
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif --bias_init unif 1 2
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const --bias_init
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const --bias_init unif
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const --bias_init const
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const --bias_init const 3
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const --bias_init unif 5 8
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 55 89 --bias_init const 13
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 14.4 23.3 --bias_init unif 21 34
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const 37.7 --bias_init const 55
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init const 610 --bias_init unif 8.9 14.4
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 98.7 159.7 --bias_init const 23.3 --seed_init 42
./generate_conv2d_shallowin_deepout_relu.py -v --weight_init unif 25.84 41.81 --bias_init unif 37.7 61.0 --seed_init 42

./generate_fc_deepin_anyout.py -v --bias_init --train_model
./generate_fc_deepin_anyout.py -v --bias_init unif --train_model
./generate_fc_deepin_anyout.py -v --bias_init const --train_model
./generate_fc_deepin_anyout.py -v --bias_init unif 0 1 --train_model
./generate_fc_deepin_anyout.py -v --bias_init const 1 --train_model
./generate_fc_deepin_anyout.py -v --bias_init unif 2 3 --seed_init 42 --train_model

./generate_fc_deepin_anyout.py -v --weight_init --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 5 8 --train_model
./generate_fc_deepin_anyout.py -v --weight_init const 13 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 21 34 --seed_init 42 --train_model

./generate_fc_deepin_anyout.py -v --weight_init --bias_init --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --bias_init --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --bias_init unif --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --bias_init const --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --bias_init const 1 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif --bias_init unif 1 2 --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --bias_init --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --bias_init unif --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --bias_init const --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --bias_init const 3 --train_model
./generate_fc_deepin_anyout.py -v --weight_init const --bias_init unif 5 8 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 55 89 --bias_init const 13 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 14.4 23.3 --bias_init unif 21 34 --train_model
./generate_fc_deepin_anyout.py -v --weight_init const 37.7 --bias_init const 55 --train_model
./generate_fc_deepin_anyout.py -v --weight_init const 610 --bias_init unif 8.9 14.4 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 98.7 159.7 --bias_init const 23.3 --seed_init 42 --train_model
./generate_fc_deepin_anyout.py -v --weight_init unif 25.84 41.81 --bias_init unif 37.7 61.0 --seed_init 42 --train_model
