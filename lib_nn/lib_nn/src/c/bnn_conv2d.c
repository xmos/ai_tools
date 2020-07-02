
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../nn_op_helper.h"
#include "nn_op_structs.h"
#include "nn_operator.h"
#include "xs3_vpu.h"

typedef struct {
  unsigned stride[2]; //stride_height, stride_width
  unsigned y_dims[3]; // out_height, out_width, out_channels
  unsigned x_dims[3]; // in_height, in_width, in_channels
  unsigned k_dims[2]; // kernel_height, kernel_width
} nn_bnn_conv2d_plan_t;

WEAK_FUNC
void bnn_conv2d_1x1(int8_t* Y_p, const int8_t* X_p, const int8_t* K_p,
                    const nn_bnn_conv2d_plan_t* plan) {

  int8_t(*Y)[plan->y_dims[0]][plan->y_dims[1]] = 
    (int8_t(*)[plan->y_dims[0]][plan->y_dims[1]])Y_p;

  int8_t(*X)[plan->x_dims[0]][plan->x_dims[1]] = 
    (int8_t(*)[plan->x_dims[0]][plan->x_dims[1]])X_p;

  int8_t(*K)[plan->k_dims[0]][plan->k_dims[1]][plan->x_dims[2]] = 
    (int8_t(*)[plan->k_dims[0]][plan->k_dims[1]][plan->x_dims[2]])K_p;

  for (unsigned h = 0; h < plan->x_dims[0];
       h += plan->stride[0] - plan->k_dims[0] + 1) {
           
    for (unsigned w = 0; w < plan->x_dims[1];
         w += plan->stride[1] - plan->k_dims[1] + 1) {

      for (unsigned oc = 0; oc < plan->y_dims[2]; oc += 1) {
        int sum = 0;
        for (unsigned kh = 0; kh < plan->k_dims[0]; kh += 1) {
          for (unsigned kw = 0; kw < plan->k_dims[1]; kw += 1) {
            for (unsigned ic = 0; ic < plan->x_dims[2]; ic += 1) {
              sum += X[h][w][ic] * K[kh][kw][ic][oc];
            }
          }
        }
        Y[h][w][oc] = sum;
      }
    }
  }
}

void conv2d_1x1_init(nn_conv2d_1x1_plan_t* plan, const nn_image_params_t* x,
                     const nn_image_params_t* y, const unsigned start_row,
                     const unsigned start_col, const unsigned out_pixels) {
  assert(x->height == y->height);
  assert(x->width == y->width);
}
