
#include <assert.h>
// #include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tst_common.h"
#include "unity.h"
#include "helpers.h"

void impl_pad_param_space(
  const unsigned channels_per_input_word,
  const unsigned sizeof_input_word,
  const unsigned max_x_height,
  const unsigned max_x_width,
  const unsigned x_channel_inc,
  const unsigned max_x_chan_words) {

  const unsigned max_pad_top = 3;
  const unsigned max_pad_bottom = 3;
  const unsigned max_pad_left = 3;
  const unsigned max_pad_right = 3;
            
  int seed = 0;
  for (unsigned pad_val_idx = 0; pad_val_idx < 8; pad_val_idx++) {

    //pick a pad value
    uint32_t pad_value = (uint32_t)pseudo_rand(&seed);

    for (unsigned x_height = 1; x_height <= max_x_height; ++x_height) {
      for (unsigned x_width = 1; x_width <= max_x_width; ++x_width) {
        for (unsigned x_chan_words = x_channel_inc; x_chan_words <= max_x_chan_words; 
          x_chan_words+=x_channel_inc) {

          size_t X_bytes = sizeof_input_word * x_height * x_width * x_chan_words;
          void * X =  malloc(X_bytes);

          for (unsigned pad_top = 0; pad_top <= max_pad_top; ++pad_top) {
            for (unsigned pad_bottom = 0; pad_bottom <= max_pad_bottom;
                 ++pad_bottom) {
              for (unsigned pad_right = 0; pad_right <= max_pad_right;
                   ++pad_right) {
                for (unsigned pad_left = 0; pad_left <= max_pad_left;
                     ++pad_left) {

                  unsigned y_height = x_height + pad_top + pad_bottom;
                  unsigned y_width = x_width + pad_left + pad_right;
                  
                  size_t Y_bytes = sizeof_input_word * y_height * y_width * x_chan_words;
                  void * Y_ref = malloc(Y_bytes);
                  void * Y     = malloc(Y_bytes);    
                  
                  for(unsigned b=0;b< X_bytes/sizeof(int);b++)
                    ((int*)X)[b] = pseudo_rand(&seed);
                  
                  memset(Y, 0, Y_bytes);
                  memset(Y_ref, 0, Y_bytes);

                  padding_sizes_t p;
                  p.top = pad_top;
                  p.bottom = pad_bottom;
                  p.left = pad_left;
                  p.right = pad_right;

                  unsigned bytes_per_pixel = sizeof_input_word * x_chan_words;

                  nn_image_params_t xp;
                  xp.height = x_height;
                  xp.width = x_width;
                  xp.channels = x_chan_words * channels_per_input_word;

                  pad_ref(Y_ref, X, &p, &xp, bytes_per_pixel,
                          pad_value);

                  nn_pad_plan_t plan;
                  pad_prepare(&plan, &p, &xp, bytes_per_pixel);
                  unsigned total_output =
                      plan.top_pad_bytes +
                      (plan.left_pad_bytes + plan.mid_copy_bytes +
                       plan.right_pad_bytes) *
                          (plan.mid_loop_count) +
                      plan.bottom_pad_bytes;

                  pad_run(Y, X, &plan, pad_value);

                  assert(total_output == Y_bytes);
                  TEST_ASSERT_EQUAL_INT8_ARRAY(Y, Y_ref, Y_bytes);

                  free(Y);
                  free(Y_ref);
                }
              }
            }
          }
          free(X);
        }
      }
    }
  }
}

void test_pad_param_space_b256(){
  impl_pad_param_space(256, sizeof(bnn_b256_t), 3, 3, 256, 4);
}
void test_pad_param_space_int8(){
  impl_pad_param_space(1, sizeof(int8_t), 3, 3, 4, 4);
}

void test_pad() {
  UNITY_SET_FILE();
  RUN_TEST(test_pad_param_space_b256);
  RUN_TEST(test_pad_param_space_int8);
}