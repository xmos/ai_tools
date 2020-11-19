#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "unity.h"

#include "helpers.h"

#define X_REF_OVERREAD_WORDS (7)
#define K_OVERREAD_WORDS (8*12)
#define DATA_SCRATCH_OVERREADWRITE_WORDS (8)

static const char undef_sentinal = 0x55;

/*
X_ref and K_ref must be initialised before running this.
This function test whole images, i.e. it wont work on a sub image.
*/
static void run_int8_config(int8_t* Y_p, int8_t* Y_ref_p, bnn_b32_t* X_ref,
               bnn_b32_t* K_p, bnn_b32_t* K_ref_p, 
               float* post_activation_multiplier,
               float* post_activation_bias, 

               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q, 

               int * chan_overlaps,
               
               unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride, int seed,
               void (*foo)()) {
                  
  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u seed:%d\n", 
  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out, seed);

  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned receptive_volume = k_width * k_height * chans_in;

  pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    Y_ref_p[e]=0;
  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    Y_p[e]=0;

  int32_t clamp_low = 0;
  int32_t clamp_high = receptive_volume*2;

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

  bnn_reorder_int8_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
                            chans_out, chan_overlaps);

  int16_t bias_multipler;
  int accu_shr, final_shr;

  bnn_quantise_activation(
      post_activation_multiplier_q,
      post_activation_bias_q,

      post_activation_multiplier,
      post_activation_bias, 

      chans_out,

      clamp_low, clamp_high,
      &accu_shr, &bias_multipler, &final_shr, receptive_volume, chan_overlaps
  );

  foo((int8_t*)Y_p, (const bnn_b32_t*)X_ref,
    (const bnn_b32_t*)K_p, post_activation_multiplier_q, 
    post_activation_bias_q, accu_shr, bias_multipler, final_shr,
    &x, &y, &k);

  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    TEST_ASSERT_INT8_WITHIN(1, Y_ref_p[e], Y_p[e]);

  //FIXME - why wont this link? The above is a workaround
  // TEST_ASSERT_INT8_ARRAY_WITHIN(1, Y_ref_p, Y_p, y_height * y_width * chans_out);
}

void impl_bnn_conv2d_int8_out_pseudo_random(
  const unsigned min_k_height, const unsigned max_k_height, 
  const unsigned min_k_width, const unsigned max_k_width,  
  
  const unsigned min_chans_in, const unsigned max_chans_in,    
  const unsigned min_chans_out, const unsigned max_chans_out,  

  const unsigned chans_in_inc, const unsigned chans_out_inc,

  const unsigned min_v_stride, const unsigned max_v_stride, 
  const unsigned min_h_stride, const unsigned max_h_stride,
  void (* valid_impl)()) {

  for (unsigned h_stride = min_h_stride; h_stride <= max_h_stride; ++h_stride) {
    for (unsigned v_stride = min_v_stride; v_stride <= max_v_stride;
         ++v_stride) {
      for (unsigned k_height = min_k_height; k_height <= max_k_height;
           ++k_height) {
        unsigned max_x_height = k_height;
        for (unsigned k_width = min_k_width; k_width <= max_k_width;
             ++k_width) {
          unsigned max_x_width = k_width;

          for (unsigned x_height = k_height; x_height <= max_x_height;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= max_x_width;
                 ++x_width) {
              unsigned y_height =  CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
              unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);
              
              for (unsigned chans_in = min_chans_in; chans_in <= max_chans_in;
                   chans_in += chans_in_inc) {
                for (unsigned chans_out = min_chans_out;
                     chans_out <= max_chans_out; chans_out += chans_out_inc) {

                  unsigned chan_words_in = chans_in/32;

                  size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*k_height*k_width*chan_words_in);
                  bnn_b32_t * K_ref = (bnn_b32_t *) malloc(K_ref_bytes);
                  bnn_b32_t * K     = (bnn_b32_t *) malloc(K_ref_bytes + sizeof(bnn_b32_t)*K_OVERREAD_WORDS);

                  size_t X_ref_bytes = sizeof(bnn_b32_t)*(x_height*x_width*chan_words_in+X_REF_OVERREAD_WORDS);
                  bnn_b32_t * X_ref =(bnn_b32_t *)malloc(X_ref_bytes);
                  int16_t *post_activation_multiplier_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
                  int16_t *post_activation_bias_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
                  
                  float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
                  float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
                  int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

                  int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
                  int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
      
                  assert(X_ref);
                  assert(Y);
                  assert(Y_ref);
                  assert(post_activation_multiplier_q);
                  assert(post_activation_bias_q);
                  assert(K);
                  assert(K_ref);

                  assert(post_activation_multiplier);
                  assert(post_activation_bias);
                  assert(chan_overlaps);
                  
                  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
                  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out);

                    for(unsigned c=0;c<1<<1;c++){
                      int seed = c;

                      for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
                        ((int*)X_ref)[b] = pseudo_rand(&seed);

                      
                      for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
                        ((int*)K_ref)[b] = pseudo_rand(&seed);

                      run_int8_config(
                          (int8_t*)Y, (int8_t*)Y_ref, (bnn_b32_t*)X_ref,
                          (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
                          (float*)post_activation_multiplier,
                          (float*)post_activation_bias, 
                          (int16_t*)post_activation_multiplier_q,
                          (int16_t*)post_activation_bias_q,  
                          (int*) chan_overlaps,
                          x_height,
                          x_width, k_height, k_width, chans_in, chans_out, h_stride,
                          v_stride, seed, valid_impl);
                    }

                    free(X_ref);
                    free(Y);
                    free(Y_ref);
                    free(post_activation_multiplier_q);
                    free(post_activation_bias_q);
                    free(K);
                    free(K_ref);

                    free(post_activation_multiplier);
                    free(post_activation_bias);
                    free(chan_overlaps);
                }
              }
            }
          }
        }
      }
    }
  }
}


void impl_bnn_conv2d_int8_out_pseudo_random2(
  const unsigned max_x_height, const unsigned max_x_width,  
  
  const unsigned chans_in,

  const unsigned min_chans_out,
  const unsigned max_chans_out,

  const unsigned chans_in_inc, 
  const unsigned chans_out_inc,

  void (* valid_impl)()) {

  for (unsigned x_height = 1; x_height <= max_x_height;
        ++x_height) {
    for (unsigned x_width = 1; x_width <= max_x_width;
          ++x_width) {
      unsigned k_height = x_height;
      unsigned k_width = x_width;
      unsigned y_height =  CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, 1);
      unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, 1);

        for (unsigned chans_out = min_chans_out;
              chans_out <= max_chans_out; chans_out += chans_out_inc) {

          unsigned chan_words_in = chans_in/32;

          size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*k_height*k_width*chan_words_in);
          bnn_b32_t * K_ref = (bnn_b32_t *) malloc(K_ref_bytes);
          bnn_b32_t * K     = (bnn_b32_t *) malloc(K_ref_bytes + sizeof(bnn_b32_t)*K_OVERREAD_WORDS);

          size_t X_ref_bytes = sizeof(bnn_b32_t)*(x_height*x_width*chan_words_in+X_REF_OVERREAD_WORDS);
          bnn_b32_t * X_ref =(bnn_b32_t *)malloc(X_ref_bytes);
          int16_t *post_activation_multiplier_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          int16_t *post_activation_bias_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          bnn_b32_t *data_scratch = (bnn_b32_t *)malloc(sizeof(bnn_b32_t)*(k_height * k_width * chan_words_in + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
          
          float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
          float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
          int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

          int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
          int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);

          assert(X_ref);
          assert(Y);
          assert(Y_ref);
          assert(post_activation_multiplier_q);
          assert(post_activation_bias_q);
          assert(K);
          assert(K_ref);

          assert(post_activation_multiplier);
          assert(post_activation_bias);
          assert(chan_overlaps);

          // printf("k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
          //    k_height, k_width, x_height, x_width, chans_in, chans_out);

          int seed = 42;

          for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
            ((int*)X_ref)[b] = pseudo_rand(&seed);

          
          for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
            ((int*)K_ref)[b] = pseudo_rand(&seed);

          run_int8_config(
              (int8_t*)Y, (int8_t*)Y_ref, (bnn_b32_t*)X_ref,
              (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
              (float*)post_activation_multiplier,
              (float*)post_activation_bias, 
              (int16_t*)post_activation_multiplier_q,
              (int16_t*)post_activation_bias_q,  
              (int*) chan_overlaps,
              x_height,
              x_width, k_height, k_width, chans_in, chans_out, 1,
              1, seed, valid_impl);

        free(X_ref);
        free(Y);
        free(Y_ref);
        free(post_activation_multiplier_q);
        free(post_activation_bias_q);
        free(K);
        free(K_ref);

        free(post_activation_multiplier);
        free(post_activation_bias);
        free(chan_overlaps);
      }
    }
  }
}

static void run_int8_sub_image(
              int8_t* Y_p, 
              const int8_t* Y_ref_p, 
              const bnn_b32_t* X_p,
              const bnn_b32_t* K_p, 

              int16_t * post_activation_multiplier_q,
              int16_t * post_activation_bias_q,
              const int accu_shr,
              const int16_t bias_multiplier,
              const int final_shr,

              const nn_image_params_t* x,
              const nn_image_params_t* y,
              const nn_window_params_t* k,
              unsigned y_loc_x, unsigned y_loc_y, 
              unsigned y_sub_width, unsigned y_sub_height,
              void (* valid_impl)()){

  valid_impl(Y_p, X_p,
      K_p, post_activation_multiplier_q,
      post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
      x, y, k,
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
          TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
        }
      }
    }
  }
}

/*
This test check for a fixed x_height, x_width, k_height and k_width a sub-region of the output
is correctly computed. It check this for MIN_CHANS_IN and MAX_CHANS_IN input channels and 
MIN_CHANS_OUT to MAX_CHANS_OUT output channels. Stride are tested, dilations are untested currently.
*/
void impl_bnn_conv2d_int8_out_sub_image(
  const unsigned full_x_height, const unsigned full_x_width,  
  const unsigned full_k_height, const unsigned full_k_width,
  
  const unsigned min_chans_in, const unsigned max_chans_in,    
  const unsigned min_chans_out, const unsigned max_chans_out,  

  const unsigned chans_in_inc, const unsigned chans_out_inc,

  const unsigned min_v_stride, const unsigned max_v_stride, 
  const unsigned min_h_stride, const unsigned max_h_stride,
  void (* valid_impl)()){

  #define X_V_DILATION 1
  #define X_H_DILATION 1

  int seed = 42;

  for(unsigned chans_out = min_chans_out; chans_out <= max_chans_out; chans_out += chans_out_inc){
    for(unsigned chans_in = min_chans_in; chans_in <= max_chans_in; chans_in += chans_in_inc){

      unsigned chan_words_in = chans_in/32;

      size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*full_k_height*full_k_width*chan_words_in);
      bnn_b32_t * K_ref = (bnn_b32_t * ) malloc(K_ref_bytes);

      size_t X_ref_bytes = sizeof(bnn_b32_t)*(full_x_height*full_x_width*chan_words_in+X_REF_OVERREAD_WORDS);
      bnn_b32_t * X_ref = (bnn_b32_t *) malloc(X_ref_bytes);
      int16_t * post_activation_multiplier_q = (int16_t *) malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
      int16_t * post_activation_bias_q = (int16_t *) malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
      bnn_b32_t * K = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(chans_out*full_k_height*full_k_width*chan_words_in + K_OVERREAD_WORDS));

      float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
      float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
      int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

      for (unsigned h_stride = min_h_stride; h_stride < max_h_stride; h_stride++){
        for (unsigned v_stride = min_v_stride; v_stride < max_v_stride; v_stride++){
            
          nn_image_params_t x;
          x.height = full_x_height;
          x.width = full_x_width;
          x.channels = chans_in;
          nn_image_params_t y;
          y.height = CONV2D_OUTPUT_LENGTH(full_x_height, full_k_height, X_V_DILATION, v_stride);
          y.width = CONV2D_OUTPUT_LENGTH(full_x_width, full_k_width, X_H_DILATION, h_stride);
          y.channels = chans_out;
          nn_window_params_t k;
          k.shape.height = full_k_height;
          k.shape.width = full_k_width;
          k.stride.horizontal = h_stride;
          k.stride.vertical = v_stride;
          k.dilation.horizontal = X_H_DILATION;
          k.dilation.vertical = X_V_DILATION;

          int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y.height * y.width * y.channels);
          int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y.height * y.width * y.channels);    

          if(y.height == 0 || y.width == 0)
            continue;

          for(unsigned i=0;i<1<<6;i++){

            for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
              ((int*)X_ref)[b] = pseudo_rand(&seed);
            
            for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
              ((int*)K_ref)[b] = pseudo_rand(&seed);

            unsigned receptive_volume = k.shape.width * k.shape.height * x.channels;

            pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

            //Calculate the entire reference image
            larq_ref_bconv2d_int8_out(&x, &y, &k, (const int32_t*)X_ref, (const int32_t*)K_ref,
                        (int8_t*)Y_ref, post_activation_multiplier, post_activation_bias);

            int32_t clamp_low = 0;     
            int32_t clamp_high = receptive_volume*2;

            bnn_reorder_int8_kernel_tensor((bnn_b32_t *)K, (const bnn_b32_t *)K_ref, k.shape.height, 
              k.shape.width, x.channels, y.channels, chan_overlaps);

            int accu_shr, final_shr;
            int16_t bias_multiplier;
            bnn_quantise_activation(
                post_activation_multiplier_q,
                post_activation_bias_q,

                post_activation_multiplier,
                post_activation_bias, 

                chans_out,

                clamp_low, clamp_high,
                &accu_shr, &bias_multiplier, &final_shr, receptive_volume, chan_overlaps
            );

            for (unsigned y_loc_x = 0; y_loc_x<y.width; ++y_loc_x){
              for (unsigned y_loc_y = 0; y_loc_y<y.height; ++y_loc_y){
                for (unsigned y_sub_width = 1; y_sub_width<y.width-y_loc_x; ++y_sub_width){
                  for (unsigned y_sub_height = 1; y_sub_height<y.height-y_loc_y; ++y_sub_height){

                      size_t addressable_Y_bytes = y.height * y.width * y.channels;
                      memset(Y, undef_sentinal, addressable_Y_bytes);

                      run_int8_sub_image(
                        (int8_t*)Y, 
                        (const int8_t*)Y_ref,
                        (const bnn_b32_t*) X_ref,
                        (const bnn_b32_t*) K, 

                        (int16_t * )post_activation_multiplier_q,
                        (int16_t *) post_activation_bias_q,
                        (const int )accu_shr,
                        (const int16_t) bias_multiplier,
                        (const int )final_shr,

                        &x, &y, &k,
                        y_loc_x, y_loc_y, y_sub_width, y_sub_height, valid_impl);
                    }
                  }
                } 
              }
            }
            free(Y_ref);
            free(Y);
        }
      }
      free(K_ref);
      free(K);
      free(X_ref);
      free(post_activation_multiplier);
      free(post_activation_bias);
      free(chan_overlaps);
      free(post_activation_multiplier_q);
      free(post_activation_bias_q);

    }
  }
}

static void SISO_valid(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,
      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
  x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bnn_conv2d_int8_out_SISO_valid(Y_p, X_p,
                      K_p, post_activation_multiplier_q,
                      post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
                      data_scratch, x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);
  free(data_scratch);
}

static void DI_valid(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,
      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bnn_conv2d_int8_out_valid(Y_p, (const bnn_b256_t*)X_p,
        (const bnn_b256_t*)K_p, post_activation_multiplier_q,
        post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
        x, y, k,
        y_loc_x, y_loc_y, y_sub_width, y_sub_height);
}


static void SISO_full(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,
      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
    x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bnn_conv2d_int8_out_SISO(Y_p, X_p,
                      K_p, post_activation_multiplier_q,
                      post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
                      data_scratch, x, y, k,
                      0, 0, y->width, y->height, 0, 0);
  free(data_scratch);
}

static void DI_full(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,
      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bnn_conv2d_int8_out(Y_p, (const bnn_b256_t*)X_p,
                      (const bnn_b256_t*)K_p, post_activation_multiplier_q,
                      post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
                      x, y, k,
                      0, 0, y->width, y->height, 0, 0);
}

void test_bnn_conv2d_int8_out_SISO_sub_image(){
  impl_bnn_conv2d_int8_out_sub_image(5, 5, 3, 3, 32*1, 32*9, 4*1, 4*3, 32, 4, 1, 1, 3, 3, (void*)&SISO_valid);
}

void test_bnn_conv2d_int8_out_DI_sub_image(){
  impl_bnn_conv2d_int8_out_sub_image(5, 5, 3, 3, 256*1, 256*2, 16*1, 16*3, 256, 32, 1, 1, 3, 3, (void*)&DI_valid);
}

void test_bnn_conv2d_int8_out_SISO_pseudo_random(){
  impl_bnn_conv2d_int8_out_pseudo_random(1, 5,1, 5, 32*1, 32*9, 4*1, 4*3, 32, 4, 1, 3, 1, 3, (void*)&SISO_full);
}

void test_bnn_conv2d_int8_out_DI_pseudo_random(){
  impl_bnn_conv2d_int8_out_pseudo_random(1, 4, 1, 4, 256*1, 256*2, 32*1, 32*3, 256, 32, 1, 3, 1, 3, (void*)&DI_full);
}

void test_bnn_conv2d_int8_out_SISO_pseudo_random2(){
  impl_bnn_conv2d_int8_out_pseudo_random2(1, 32, 32, 4, 4, 32, 4, (void*)&SISO_full);
}

void test_bnn_conv2d_int8_out_DI_pseudo_random2(){
  impl_bnn_conv2d_int8_out_pseudo_random2(1, 32, 256, 32, 32, 256, 32, (void*)&DI_full);
}

void test_bnn_conv2d_int8() {
  UNITY_SET_FILE();

  RUN_TEST(test_bnn_conv2d_int8_out_SISO_pseudo_random);
  RUN_TEST(test_bnn_conv2d_int8_out_DI_pseudo_random);

  RUN_TEST(test_bnn_conv2d_int8_out_SISO_pseudo_random2);
  RUN_TEST(test_bnn_conv2d_int8_out_DI_pseudo_random2);

  RUN_TEST(test_bnn_conv2d_int8_out_SISO_sub_image);
  RUN_TEST(test_bnn_conv2d_int8_out_DI_sub_image);
}
