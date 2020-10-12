
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "unity.h"

#include "helpers.h"

//These are used for collecting int8 output stats
int output_error_g[256] = {0};
unsigned abs_output_error_g[256] = {0};
unsigned error_counter_g[256] = {0};

/*
X_ref and K_ref must be initialised before running this.
This function test whole images, i.e. it wont work on a sub image.
*/
void run_int8_config(int8_t* Y_p, int8_t* Y_ref_p, bnn_b256_t* X_ref,
               bnn_b256_t* K_p, bnn_b256_t* K_ref_p, 
               float* post_activation_multiplier,
               float* post_activation_bias, 

               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q, 
               int16_t * post_activation_multiplier_q_reordered,
               int16_t* post_activation_bias_q_reordered, 

               unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride) {
                  
  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned X_bytes = (x_height * x_width * chans_in) / 8;
  unsigned K_bytes = (k_width * k_height * chans_in * chans_out) / 8;
  unsigned Y_bytes = (y_width * y_height * chans_out);

  int32_t receptive_field = k_width * k_height * chans_in;

  //Pick some random test values
  for (unsigned ch=0;ch < chans_out;ch++){
    float input_range = receptive_field;

    unsigned range =  rand()%10;
    float r =  (float)rand() / (float)RAND_MAX;
    float output_range = (range/2) + (float)(1<<range)*r;

    post_activation_multiplier[ch] = output_range/input_range;

    int32_t max_bias = 255;
    r =  (float)rand() / (float)RAND_MAX;
    post_activation_bias[ch] = (r * max_bias*2.0) - max_bias;

  }

  int accu_shr, final_shr;
  int32_t clamp_low = 0;
  int32_t clamp_high = receptive_field*2;

  quantise_activation(
               post_activation_multiplier_q, post_activation_bias_q,
               post_activation_multiplier, post_activation_bias, 
               chans_out,
               clamp_low, clamp_high,
               &accu_shr, &final_shr, receptive_field);

#if !defined(__XS3A__)
  error_stats_t e;
  measure_quantisation(
               post_activation_multiplier_q, post_activation_bias_q,
               post_activation_multiplier, post_activation_bias, 
               chans_out,
               clamp_low, clamp_high,
               accu_shr, final_shr, receptive_field, &e);

  for(unsigned b=0;b<256;b++){
    output_error_g[b] += e.output_error[b];
    abs_output_error_g[b] += e.abs_output_error[b];
    error_counter_g[b] += e.error_counter[b];
  }
#endif

  nn_image_params_t x;
  x.height = x_height;
  x.width = x_width;
  x.channels = chans_in;
  nn_image_params_t y;
  y.height = y_height;
  y.width = y_width;
  y.channels = chans_out;
  nn_window_params_t k;
  k.shape.height = k_height;
  k.shape.width = k_width;
  k.stride.horizontal = h_stride;
  k.stride.vertical = v_stride;
  k.dilation.horizontal = 1;
  k.dilation.vertical = 1;

  larq_ref_bconv2d_int8_out(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref_p,
                   (int8_t*)Y_ref_p, post_activation_multiplier, post_activation_bias);

#if defined(__XS3A__)

  bnn_reorder_multiplier_and_bias_tensors(
                                  post_activation_multiplier_q_reordered,
                                  post_activation_multiplier_q,
                                  post_activation_bias_q_reordered,
                                  post_activation_bias_q,
                                  chans_out);

  bnn_reorder_int8_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
                            chans_out);

  bnn_conv2d_int8_out((int8_t*)Y_p, (const bnn_b256_t*)X_ref,
    (const bnn_b256_t*)K_p, post_activation_multiplier_q_reordered, 
    post_activation_bias_q_reordered, accu_shr, final_shr,
    &x, &y, &k,
    0, 0, y_width, y_height,
    0, 0, 
    0, 0, k_width, k_height);
#else
  bnn_conv2d_int8_out((int8_t*)Y_p, (const bnn_b256_t*)X_ref,
    (const bnn_b256_t*)K_ref_p, 
    post_activation_multiplier_q, post_activation_bias_q, 
    accu_shr, final_shr,
    &x, &y, &k,
    0, 0, y_width, y_height,
    0, 0, 
    0, 0, k_width, k_height);
#endif

  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    TEST_ASSERT_INT8_WITHIN(1, Y_ref_p[e], Y_p[e]);

  //FIXME - why wont this link? The above is a workaround
  // TEST_ASSERT_INT8_ARRAY_WITHIN(1, Y_ref_p, Y_p, y_height * y_width * chans_out);
}

void test_bnn_conv2d_int8_out_pseudo_directed() {
#define X_V_DILATION 1
#define X_H_DILATION 1

#define X_HEIGHT 1
#define X_WIDTH 1
#define K_HEIGHT X_HEIGHT
#define K_WIDTH X_WIDTH
#define CHANS_IN XS3_VPU_VREG_WIDTH_BITS
#define CHANS_OUT 16
#define H_STRIDE 1
#define V_STRIDE 1

#define Y_HEIGHT \
  CONV2D_OUTPUT_LENGTH(X_HEIGHT, K_HEIGHT, X_V_DILATION, V_STRIDE)
#define Y_WIDTH CONV2D_OUTPUT_LENGTH(X_WIDTH, K_WIDTH, X_H_DILATION, H_STRIDE)

#define CHAN_WORDS_IN DIV_BY_AND_ROUND_UP(CHANS_IN, XS3_VPU_VREG_WIDTH_BITS)

  bnn_b256_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT / 16][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  int8_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
  int8_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

  float WORD_ALIGNED post_activation_multiplier[CHANS_OUT];
  float WORD_ALIGNED post_activation_bias[CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q[CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q[CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q_reordered[CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q_reordered[CHANS_OUT];

  srand(42);
  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  memset(K, 0, sizeof(K));
  memset(Y, 0, sizeof(Y));
  memset(Y_ref, 0, sizeof(Y_ref));

  run_int8_config((int8_t *)Y, (int8_t*)Y_ref, (bnn_b256_t*)X_ref,
              (bnn_b256_t*)K, (bnn_b256_t*)K_ref, (float*)post_activation_multiplier,
              (float*)post_activation_bias, (int16_t*)post_activation_multiplier_q,
              (int16_t*)post_activation_bias_q, (int16_t*)post_activation_multiplier_q_reordered,
              (int16_t*)post_activation_bias_q_reordered, X_HEIGHT, X_WIDTH, K_HEIGHT, K_WIDTH,
              CHANS_IN, CHANS_OUT, H_STRIDE, V_STRIDE);

#undef X_V_DILATION 
#undef X_H_DILATION
#undef X_HEIGHT 
#undef X_WIDTH 
#undef K_HEIGHT
#undef K_WIDTH
#undef CHANS_IN 
#undef CHANS_OUT 
#undef H_STRIDE
#undef V_STRIDE
#undef Y_HEIGHT
#undef Y_WIDTH 
#undef CHAN_WORDS_IN
}

void test_bnn_conv2d_int8_out_pseudo_random() {
#define MIN_H_STRIDE 1
#define MIN_V_STRIDE 1
#define MAX_H_STRIDE 4
#define MAX_V_STRIDE 4

#define MIN_K_HEIGHT 1
#define MIN_K_WIDTH 1
#define MAX_K_HEIGHT 5
#define MAX_K_WIDTH 5

#define MIN_CHANS_IN XS3_VPU_VREG_WIDTH_BITS
#define MAX_CHANS_IN (XS3_VPU_VREG_WIDTH_BITS*2)

#define MIN_CHANS_OUT 16
#define MAX_CHANS_OUT 48

#define MIN_X_HEIGHT MIN_K_HEIGHT
#define MIN_X_WIDTH MIN_K_WIDTH
#define MAX_X_HEIGHT MAX_K_HEIGHT
#define MAX_X_WIDTH MAX_K_WIDTH

#define MAX_CHAN_WORDS_IN DIV_BY_AND_ROUND_UP(MAX_CHANS_IN, XS3_VPU_VREG_WIDTH_BITS)

#define MAX_Y_HEIGHT (((MAX_X_HEIGHT - MIN_K_HEIGHT + 1) / MIN_V_STRIDE))
#define MAX_Y_WIDTH (((MAX_X_WIDTH - MIN_K_WIDTH + 1) / MIN_H_STRIDE))

  bnn_b256_t WORD_ALIGNED
      K_ref[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_CHAN_WORDS_IN];
  int8_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHANS_OUT];
  int8_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHANS_OUT];

  float WORD_ALIGNED post_activation_multiplier[MAX_CHANS_OUT];
  float WORD_ALIGNED post_activation_bias[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q_reordered[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q_reordered[MAX_CHANS_OUT];

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  srand(42);
  for (unsigned h_stride = MIN_H_STRIDE; h_stride <= MAX_H_STRIDE; ++h_stride) {
    for (unsigned v_stride = MIN_V_STRIDE; v_stride <= MAX_V_STRIDE;
         ++v_stride) {
      for (unsigned k_height = MIN_K_HEIGHT; k_height <= MAX_K_HEIGHT;
           ++k_height) {
        for (unsigned k_width = MIN_K_WIDTH; k_width <= MAX_K_WIDTH;
             ++k_width) {
          for (unsigned x_height = k_height; x_height <= MAX_X_HEIGHT;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= MAX_X_WIDTH;
                 ++x_width) {
              for (unsigned chans_in = MIN_CHANS_IN; chans_in <= MAX_CHANS_IN;
                   chans_in += 256) {
                for (unsigned chans_out = MIN_CHANS_OUT;
                     chans_out <= MAX_CHANS_OUT; chans_out += 16) {
                  run_int8_config(
                      (int8_t*)Y, (int8_t*)Y_ref, (bnn_b256_t*)X_ref,
                      (bnn_b256_t*)K, (bnn_b256_t*)K_ref,
                      (float*)post_activation_multiplier,
                      (float*)post_activation_bias, 
                      (int16_t*)post_activation_multiplier_q,
                      (int16_t*)post_activation_bias_q,  
                      (int16_t*)post_activation_multiplier_q_reordered,
                      (int16_t*)post_activation_bias_q_reordered, 
                      x_height,
                      x_width, k_height, k_width, chans_in, chans_out, h_stride,
                      v_stride);
                }
              }
            }
          }
        }
      }
    }
  }
#undef MIN_H_STRIDE 
#undef MIN_V_STRIDE 
#undef MAX_H_STRIDE 
#undef MAX_V_STRIDE 
#undef MIN_K_HEIGHT 
#undef MIN_K_WIDTH 
#undef MAX_K_HEIGHT 
#undef MAX_K_WIDTH 
#undef MIN_CHANS_IN 
#undef MAX_CHANS_IN 
#undef MIN_CHANS_OUT 
#undef MAX_CHANS_OUT 
#undef MIN_X_HEIGHT 
#undef MIN_X_WIDTH 
#undef MAX_X_HEIGHT 
#undef MAX_X_WIDTH 
#undef MAX_CHAN_WORDS_IN 
#undef MAX_CHAN_WORDS_OUT 
#undef MAX_Y_HEIGHT
#undef MAX_Y_WIDTH
}

void run_int8_sub_image(
              int8_t* Y_p, 
              const int8_t* Y_ref_p, 
              const bnn_b256_t* X_p,
              const bnn_b256_t* K_p, 
              
              int16_t * post_activation_multiplier_q_ordered,
              int16_t * post_activation_bias_q_ordered,
              const int accu_shr,
              const int final_shr,
              
              const nn_image_params_t* x,
              const nn_image_params_t* y,
              const nn_window_params_t* k,
              unsigned y_loc_x, unsigned y_loc_y, 
              unsigned y_sub_width, unsigned y_sub_height){

  bnn_conv2d_int8_out_valid(Y_p, X_p,
                      K_p, post_activation_multiplier_q_ordered,
                      post_activation_bias_q_ordered, accu_shr, final_shr, x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);


  int8_t(*Y)[y->width][y->channels] =
      (int8_t(*)[y->width][y->channels])Y_p;

  int8_t(*Y_ref)[y->width][y->channels] =
      (int8_t(*)[y->width][y->channels])Y_ref_p;

  for (unsigned h = 0; h < y->height; h++) {
    for (unsigned w = 0; w < y->width; w++) {
      if((h >= y_loc_y) && (h < (y_loc_y + y_sub_height)) && (w >= y_loc_x) && (w < (y_loc_x + y_sub_width))){
        //If the result should have been computed then check it against the reference
        for (unsigned c = 0; c < y->channels; c++) {
          TEST_ASSERT_INT8_WITHIN(1, Y_ref[h][w][c], Y[h][w][c]);
        }
      } else {
        //Otherwise check thet is hasn't been written to
        for (unsigned c = 0; c < y->channels; c++) {
          TEST_ASSERT_EQUAL_INT8(0, Y[h][w][c]);
        }
      }
    }
  }
}

void test_bnn_conv2d_int8_out_sub_image(){

  #define FULL_X_HEIGHT 7
  #define FULL_X_WIDTH 7
  #define FULL_K_HEIGHT 3
  #define FULL_K_WIDTH 3

  #define MIN_CHANS_IN XS3_VPU_VREG_WIDTH_BITS
  #define MAX_CHANS_IN (XS3_VPU_VREG_WIDTH_BITS*4)
  #define MIN_CHANS_OUT (16)
  #define MAX_CHANS_OUT (16*4)
  
  #define X_V_DILATION 1
  #define V_STRIDE 1
  #define X_H_DILATION 1
  #define H_STRIDE 1

  #define MAX_CHAN_WORDS_IN DIV_BY_AND_ROUND_UP(MAX_CHANS_IN, XS3_VPU_VREG_WIDTH_BITS)

  #define FULL_Y_HEIGHT \
    CONV2D_OUTPUT_LENGTH(FULL_X_HEIGHT, FULL_K_HEIGHT, X_V_DILATION, V_STRIDE)
  #define FULL_Y_WIDTH CONV2D_OUTPUT_LENGTH(FULL_X_WIDTH, FULL_K_WIDTH, X_H_DILATION, H_STRIDE)

  bnn_b256_t WORD_ALIGNED
      K_ref[MAX_CHANS_OUT][FULL_K_HEIGHT][FULL_K_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K    [MAX_CHANS_OUT][FULL_K_HEIGHT][FULL_K_WIDTH][MAX_CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[FULL_X_HEIGHT][FULL_X_WIDTH][MAX_CHAN_WORDS_IN];
  int8_t WORD_ALIGNED Y_ref[FULL_Y_HEIGHT][FULL_Y_WIDTH][MAX_CHANS_OUT];
  int8_t WORD_ALIGNED Y[FULL_Y_HEIGHT][FULL_Y_WIDTH][MAX_CHANS_OUT];

  float WORD_ALIGNED post_activation_multiplier[MAX_CHANS_OUT];
  float WORD_ALIGNED post_activation_bias[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_multiplier_q_ordered[MAX_CHANS_OUT];
  int16_t WORD_ALIGNED post_activation_bias_q_ordered[MAX_CHANS_OUT];

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);
  assert(sizeof(K) == sizeof(K_ref));
  
  srand(42);

  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  for(unsigned chans_out = MIN_CHANS_OUT; chans_out <= MAX_CHANS_OUT; chans_out += 16){
    for(unsigned chans_in = MIN_CHANS_IN; chans_in <= MAX_CHANS_IN; chans_in += XS3_VPU_VREG_WIDTH_BITS){

      for (unsigned h_stride = 1; h_stride < 5; h_stride++){

        for (unsigned v_stride=1; v_stride < 5; v_stride++){
            
          nn_image_params_t x;
          x.height = FULL_X_HEIGHT;
          x.width = FULL_X_WIDTH;
          x.channels = chans_in;
          nn_image_params_t y;
          y.height = CONV2D_OUTPUT_LENGTH(FULL_X_HEIGHT, FULL_K_HEIGHT, X_V_DILATION, v_stride);
          y.width = CONV2D_OUTPUT_LENGTH(FULL_X_WIDTH, FULL_K_WIDTH, X_H_DILATION, h_stride);
          y.channels = chans_out;
          nn_window_params_t k;
          k.shape.height = FULL_K_HEIGHT;
          k.shape.width = FULL_K_WIDTH;
          k.stride.horizontal = h_stride;
          k.stride.vertical = v_stride;
          k.dilation.horizontal = X_H_DILATION;
          k.dilation.vertical = X_V_DILATION;

          int32_t backtransform_add = k.shape.width * k.shape.height * x.channels;

          //Pick some random test values
          for (unsigned ch=0;ch < y.channels; ch++){
            float input_range = backtransform_add;

            unsigned range =  rand()%10;
            float r =  (float)rand() / (float)RAND_MAX;
            float output_range = (range/2) + (float)(1<<range)*r;

            post_activation_multiplier[ch] = output_range/input_range;

            int32_t max_bias = 255;
            r =  (float)rand() / (float)RAND_MAX;
            post_activation_bias[ch] = (r * max_bias*2.0) - max_bias;

          }

          larq_ref_bconv2d_int8_out(&x, &y, &k, (const int32_t*)X_ref, (const int32_t*)K_ref,
                      (int8_t*)Y_ref, post_activation_multiplier, post_activation_bias);

          int accu_shr, final_shr;
          int accu_min_post_clamp = -backtransform_add;
          int accu_max_post_clamp = backtransform_add;

          quantise_activation(
                      post_activation_multiplier_q, post_activation_bias_q,
                      post_activation_multiplier, post_activation_bias, 
                      y.channels,
                      accu_min_post_clamp, accu_max_post_clamp,
                      &accu_shr, &final_shr, backtransform_add);

    #if defined(__XS3A__)

          bnn_reorder_multiplier_and_bias_tensors(
                                          post_activation_multiplier_q_ordered,
                                          post_activation_multiplier_q,
                                          post_activation_bias_q_ordered,
                                          post_activation_bias_q,
                                          y.channels);

          bnn_reorder_int8_kernel_tensor((bnn_b256_t *)K, (const bnn_b256_t *)K_ref, k.shape.height, 
            k.shape.width, x.channels, y.channels);
          memcpy(K_ref, K, sizeof(K));
                                          
    #else
          memcpy(post_activation_multiplier_q_ordered, post_activation_multiplier_q, 
            sizeof(post_activation_multiplier_q));
          memcpy(post_activation_bias_q_ordered, post_activation_bias_q, 
            sizeof(post_activation_bias_q));
    #endif

          //Calculate the entire reference image

          for (unsigned y_loc_x = 0; y_loc_x<y.width; ++y_loc_x){
            for (unsigned y_loc_y = 0; y_loc_y<y.height; ++y_loc_y){
              for (unsigned y_sub_width = 1; y_sub_width<y.width-y_loc_x; ++y_sub_width){
                for (unsigned y_sub_height = 1; y_sub_height<y.height-y_loc_y; ++y_sub_height){

                    memset(Y, 0, sizeof(Y));

                    run_int8_sub_image((int8_t*)Y, 
                      (const int8_t*)Y_ref, 
                      (const bnn_b256_t*) X_ref, 
                      (const bnn_b256_t*) K_ref, 
                      
                      post_activation_multiplier_q_ordered,
                      post_activation_bias_q_ordered,
                      accu_shr,
                      final_shr,
                      
                      &x, &y, &k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height
                    );
                  }
                }
              } 
            }
        }
      }
    }
  }

  #undef FULL_X_HEIGHT
  #undef FULL_X_WIDTH 
  #undef FULL_K_HEIGHT 
  #undef FULL_K_WIDTH 
  #undef CHANS_IN 
  #undef CHANS_OUT 
  #undef X_V_DILATION 
  #undef V_STRIDE 
  #undef X_H_DILATION 
  #undef H_STRIDE 
  #undef CHAN_WORDS_IN 
  #undef FULL_Y_HEIGHT
  #undef FULL_Y_WIDTH 

}

void test_int8_stats(){
  // This checks the output stats of the int8 kernels
  double avg_abs_error = 0.0;
  double avg_error = 0.0;
  for (unsigned i=0;i<256;i++){
    double abs_error = (double)abs_output_error_g[i] / (double)error_counter_g[i];
    double error = (double)output_error_g[i]/(double)error_counter_g[i];
    avg_abs_error += abs_error;
    avg_error += error;
  }

  TEST_ASSERT_GREATER_OR_EQUAL(64, 1./(avg_abs_error/256));
  TEST_ASSERT_GREATER_OR_EQUAL(64, 1./(avg_error/256));
}

void test_bnn_conv2d_int8() {
  UNITY_SET_FILE();
  RUN_TEST(test_bnn_conv2d_int8_out_pseudo_directed);
  RUN_TEST(test_bnn_conv2d_int8_out_pseudo_random);
  RUN_TEST(test_bnn_conv2d_int8_out_sub_image);

#if !defined(__XS3A__)
  RUN_TEST(test_int8_stats);
#endif
}