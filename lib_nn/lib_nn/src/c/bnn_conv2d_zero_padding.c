

// #include <stdlib.h>
// #include <stdint.h>
// #include <string.h>
// #include <stdio.h>
// #include <assert.h>

// #include "nn_operator.h"
// #include "../nn_op_helper.h"
// #include "nn_op_structs.h"

// #include "xs3_vpu.h"

// // WIP
// void padded_kernel(bnn_b32_t* Y_p, bnn_b256_t* X_p, bnn_b256_t* K_p,
//                    int32_t* thresholds_p, padding_values_t padding_values,
//                    nn_image_params_t x, nn_image_params_t y,
//                    nn_window_params_t k) {
                     
//   // These will be parameters to the function
//   unsigned x_full_width = x.width;
//   unsigned x_full_height = x.height;

//   const unsigned k_full_width = k.shape.width;
//   const unsigned k_full_height = k.shape.height;

//   unsigned h_stride = k.stride.horizontal;
//   unsigned v_stride = k.stride.vertical;

//   unsigned h_dilation = 1;
//   unsigned v_dilation = 1;

//   unsigned y_full_width =
//       CONV2D_OUTPUT_LENGTH(x_full_width, k_full_width, h_dilation, h_stride);
//   unsigned y_full_height =
//       CONV2D_OUTPUT_LENGTH(x_full_height, k_full_height, v_dilation, v_stride);

//   // Edges are given in the order of: Top, Right, Bottom, Left
//   enum {
//     TOP = 0,
//     RIGHT,
//     BOTTOM,
//     LEFT,
//   };

//   unsigned edge_padding[4];
//   edge_padding[TOP] = padding_values.height;
//   edge_padding[RIGHT] = padding_values.width + padding_values.width_offset;
//   edge_padding[BOTTOM] = padding_values.height + padding_values.height_offset;
//   edge_padding[LEFT] = padding_values.width;

//   // Kernel side lengths: height, width
//   unsigned k_edge_length[2] = {k_full_height, k_full_width};

//   unsigned edge_height[4];
//   unsigned edge_pad_only_height[4];
//   for (unsigned edge = 0; edge < 4; ++edge) {
//     if (edge_padding[edge] > k_edge_length[(edge * 2) & 1] - 1) {
//       edge_height[edge] = k_edge_length[(edge * 2) & 1] - 1;
//     } else {
//       edge_height[edge] = edge_padding[edge];
//     }
//     edge_pad_only_height[edge] = edge_padding[edge] - edge_height[edge];
//   }

//   // top_pad_only

//   if (edge_pad_only_height[TOP] > 0) {
//     printf("TOP padding needed of %u\n", edge_pad_only_height[TOP]);
//   }
//   if (edge_pad_only_height[BOTTOM] > 0) {
//     printf("BOTTOM padding needed of %u\n", edge_pad_only_height[BOTTOM]);
//   }
//   if (edge_pad_only_height[LEFT] > 0) {
//     printf("LEFT padding needed of %u\n", edge_pad_only_height[LEFT]);
//   }
//   if (edge_pad_only_height[RIGHT] > 0) {
//     printf("RIGHT padding needed of %u\n", edge_pad_only_height[RIGHT]);
//   }

//   nn_image_params_t tl_x;
//   tl_x.channels = x.channels;

//   nn_image_params_t tl_y;
//   tl_y.channels = y.channels;
//   tl_y.height = 1;
//   tl_y.width = 1;

//   nn_window_params_t tl_k;
//   tl_k.start.column = 0;
//   tl_k.start.row = 0;
//   tl_k.stride.horizontal = h_stride;
//   tl_k.stride.vertical = v_stride;

//   // top left
//   for (unsigned loc_y = 0; loc_y < edge_height[TOP]; ++loc_y) {
//     unsigned y_loc_y = loc_y + edge_pad_only_height[TOP];
//     for (unsigned loc_x = 0; loc_x < edge_height[LEFT]; ++loc_x) {
//       unsigned y_loc_x = loc_x + edge_pad_only_height[LEFT];

//       tl_k.shape.height = loc_y + 1;
//       tl_k.shape.width = loc_x + 1;
//       tl_x.height =
//           tl_k.shape.height;  // TODO: This will need to incorporate the stride
//       tl_x.width = tl_k.shape.width;  // TODO: This will need to incorporate the
//                                       // stride

//       // nn_bconv2d_bin_DI_impl_plan_t plan;
//       unsigned x_loc_x = 0;
//       unsigned x_loc_y = 0;
//       unsigned k_loc_x = k_full_width - k.shape.width;
//       unsigned k_loc_y = k_full_height - k.shape.height;

//       //   printf("x_loc_x %u x_loc_x: %u, x_w %u x_h: %u\n", x_loc_x, x_loc_y,
//       //          tl_x.height, tl_x.width);
//       //   printf("y_loc_x %u y_loc_x: %u, y_w %u y_h: %u\n", y_loc_x, y_loc_y,
//       //          tl_y.height, tl_y.width);
//       //   printf("k_loc_x %u k_loc_x: %u, k_w %u k_h: %u\n", k_loc_x, k_loc_y,
//       //          tl_k.shape.height, tl_k.shape.width);
//       //   printf("\n");

//       // bconv2d_bin_DI_prepare(
//       //     &plan, (bnn_b32_t*)Y_p, (bnn_b256_t*)X_p, (bnn_b256_t*)K_p,
//       //     thresholds_p, &x, &y, &k, y_loc_x, y_loc_y,  // The output Y coord
//       //     y_loc_x, y_loc_y,                            // the input X coord

//       //     k_full_width - k.shape.width,
//       //     k_full_height -
//       //         k.shape.height,  // the top left corner of the kernel in use
//       //     y_full_width, x_full_width, k_width);

//       // bconv2d_bin_DI_impl(&plan);

//       // printf();
//     }
//   }
//   /*
//     // top
//     for (unsigned loc_y = 0; loc_y < edge_height[TOP]; ++loc_y) {
//       unsigned y_loc_y = loc_y + edge_pad_only_height[TOP];
//       unsigned loc_x = edge_height[LEFT];
//       unsigned y_loc_x = loc_x + edge_pad_only_height[LEFT];

//       k.shape.height = loc_y + 1;
//       k.shape.width = loc_x + 1;
//       x.height =
//           k.shape.height;       // TODO: This will need to incorporate the
//           stride
//       x.width = k.shape.width;  // TODO: This will need to incorporate the
//                                 // stride

//       nn_bconv2d_bin_DI_impl_plan_t plan;
//       bconv2d_bin_DI_prepare(&plan, (bnn_b32_t*)Y_p, (bnn_b256_t*)X_p,
//                                      (bnn_b256_t*)K_p, thresholds_p, &x, &y,
//                                      &k, y_loc_x, y_loc_y,  // The output Y
//                                      coord y_loc_x, y_loc_y,  // the input X
//                                      coord 0, 0,  // the region of the kernel
//                                      in use y_full_width, x_full_width,
//                                      k_width);

//       bconv2d_bin_DI_impl(&plan);

//       // printf();
//     }
//   */
//   // top right

//   // right

//   // bottom right

//   // bottom

//   // bottom left

//   // left

//   // center
// }