#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

rm -rf ./debug
set -e

## ARG MISC
v="-v"
seed=""--seed" 42"
train_flag=""--train_model" -ep 1"

## INITIALIZERS NAMES
init_names=(
    "--bias_init"
    "--input_init"
    "--weight_init"
)
input="--input_init"
bias="--bias_init"
weight="--weight_init"

## INITIALIZERS TYPE
init_types=(
    "unif"
    "const"
)

## INITIALIZERS VALUES
unif_val=( ""0" "1"" ""2" "3"" ""-2" "3"" )
const_val=( "1" "-1" "0" )

## SCRIPTS NAMES
scripts_with_input=(
    "generate_avgpool2d.py"
    "generate_avgpool2d_global.py"
    "generate_conv2d_deepin_deepout_relu.py"
    "generate_conv2d_shallowin_deepout_relu.py"
    "generate_lookup_8.py"
    "generate_maxpool2d.py"
) ## generate prelu is broken ##

scripts_bias_weights=(
    "generate_conv2d_deepin_deepout_relu.py"
    "generate_conv2d_shallowin_deepout_relu.py"
    "generate_fully_connected.py"
    "generate_fully_connected_requantized.py"
)

scripts_trainable=(
    "generate_fully_connected_requantized.py"
    "generate_fully_connected.py"
)
## EXECUTING INITIALIZERS
for script_name in "${scripts_with_input[@]}"; do       ## generate_script.py
    for init in "${init_names[@]}"; do                  ## bias input weight
        if [[ "${scripts_bias_weights[@]}" != *"${script_name}"* ]] && [ "$init" != "$input" ];
        then
            continue
        fi
        for type in "${init_types[@]}"; do              ## unif  const
            script_base="$script_name $v $seed $init $type"
            if [ "$type" == "unif" ]; then              ## unif
                for val in "${unif_val[@]}"; do         ## unif values
                    script="$script_base $val"
                    if [[ "${scripts_trainable[@]}" == *"${script_name}"* ]]; then
                        script="$script $train_flag"    ## train
                    fi
                    echo Executing: $script
                    ./$script
                done
            else                                        ## const
                for val in "${const_val[@]}"; do        ## const values
                    if [[ "${script_name}" == *"generate_conv2d"* ]] && [ "$val" == "0" ];
                    then # skipping 0
                        echo "Skipping because it's broken"
                        continue
                    fi
                    script="$script_base $val"
                    if [[ "${scripts_trainable[@]}" == *"${script_name}"* ]]; then
                        script="$script $train_flag"    ## train
                    fi
                    echo Executing: $script
                    ./$script
                done
            fi
        done
    done
done

exit 0
## COMBINATIONS
# ./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif
# ./generate_conv2d_deepin_deepout_relu.py -v --bias_init const
# ./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif 0 1
# ./generate_conv2d_deepin_deepout_relu.py -v --bias_init const 1
# ./generate_conv2d_deepin_deepout_relu.py -v --bias_init unif 2 3 --seed 42

# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 5 8
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 13
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 21 34 --seed 42

# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init unif
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init const
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init const 1
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif --bias_init unif 1 2
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init unif
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init const
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init const 3
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const --bias_init unif 5 8
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 55 89 --bias_init const 13
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 14.4 23.3 --bias_init unif 21 34
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 37.7 --bias_init const 55
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init const 610 --bias_init unif 8.9 14.4
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 98.7 159.7 --bias_init const 23.3 --seed 42
# ./generate_conv2d_deepin_deepout_relu.py -v --weight_init unif 25.84 41.81 --bias_init unif 37.7 61.0 --seed 42