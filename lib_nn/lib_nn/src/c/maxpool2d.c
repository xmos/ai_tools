

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>




WEAK_FUNC
void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_window_op_plan_t* plan)
{
    X = &X[plan->start_stride.x];
    Y = &Y[plan->start_stride.y];

    const unsigned chan_groups = plan->output.channels >> VPU_INT8_EPV_LOG2;

    for(unsigned chan_grp = 0; chan_grp <= chan_groups; chan_grp++){

        unsigned cur_chans = VPU_INT8_EPV;

        if(chan_grp == chan_groups)
            cur_chans = plan->output.channels - (chan_groups << VPU_INT8_EPV_LOG2);

        if(cur_chans == 0)
            break;

        for(unsigned y_row = 0; y_row < plan->output.rows; y_row++){
            for(unsigned y_col = 0; y_col < plan->output.cols; y_col++){

                int8_t maxes[VPU_INT8_EPV];
                memset(maxes, -128, sizeof(maxes));

                for(unsigned pool_row = 0; pool_row < plan->window.rows; pool_row++){
                    for(unsigned pool_col = 0; pool_col < plan->window.cols; pool_col++){
                        
                        for(int k = 0; k < VPU_INT8_EPV; k++){
                            maxes[k] = (X[k] > maxes[k])? X[k] : maxes[k];
                        }

                        X = &X[plan->inner_stride.horizontal.x];
                    }

                    X = &X[plan->inner_stride.vertical.x];
                }

                for(int k = 0; k < cur_chans; k++){
                    Y[k] = maxes[k];
                }

                Y = &Y[plan->outer_stride.horizontal.y];
                X = &X[plan->outer_stride.horizontal.x];
            }
            
            X = &X[plan->outer_stride.vertical.x];
            Y = &Y[plan->outer_stride.vertical.y];
        }

        X = &X[plan->chan_grp_stride.x];
        Y = &Y[plan->chan_grp_stride.y];
    }
}



void maxpool2d_init(
    nn_window_op_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_op_config_t* config)
{
    nn_window_op_init(plan, x, y, config, VPU_INT8_EPV);
}


