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
static void run_bin_config(bnn_b32_t* Y_p, bnn_b32_t* Y_ref_p, bnn_b32_t* X_ref,
               bnn_b32_t* K_p, bnn_b32_t* K_ref_p, 
               int32_t* thresholds_ref,
               int32_t* thresholds_p,

               int * chan_overlaps,
               
               unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride, int seed,
               void (*impl_fn)()) {
                  
  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u seed:%d\n", 
  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out, seed);

  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);
  assert(thresholds_p != thresholds_ref);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned receptive_volume = k_width * k_height * chans_in;

  pick_threshold_params(thresholds_ref, chans_out, chans_in * k_height * k_width);

  for (unsigned e=0;e<y_height * y_width * chans_out/32;++e)
    Y_ref_p[e]=0;
  for (unsigned e=0;e<y_height * y_width * chans_out/32;++e)
    Y_p[e]=0;

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

  larq_ref_bconv2d_bin_out(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref_p,
                   (int32_t*)Y_ref_p, (const int32_t *)thresholds_ref);

  // bnn_reorder_int8_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
  //                           chans_out, chan_overlaps);

  bnn_reorder_kernel_tensor((bnn_b32_t* )K_p, (bnn_b32_t* )K_ref_p, k_height, k_width, chans_in,
                            chans_out, chan_overlaps);

  bnn_reorder_threshold_tensor(thresholds_p, thresholds_ref, chans_out,
                              k_width * k_height * chans_in, chan_overlaps);

  impl_fn(Y_p, X_ref,
    K_p, thresholds_p, 
    &x, &y, &k);

  unsigned chan_b32_out = DIV_BY_AND_ROUND_UP(chans_out, 32);
  TEST_ASSERT_EQUAL_INT_ARRAY(Y_p, Y_ref_p, y_height*y_width*chan_b32_out); 

}

void impl_bconv2d_bin_DI_pseudo_random(
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
                  int32_t *thresholds = (int32_t *)malloc(sizeof(int32_t)*(chans_out+(16 - chans_out%16)));
                  
                  int32_t * thresholds_ref = (int32_t *)malloc(sizeof(int32_t)*chans_out);
                  int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

                  bnn_b32_t * Y     = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y_height * y_width * chans_out/32);
                  bnn_b32_t * Y_ref = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y_height * y_width * chans_out/32);
      
                  assert(X_ref);
                  assert(Y);
                  assert(Y_ref);
                  assert(thresholds_ref);
                  assert(K);
                  assert(K_ref);

                  assert(thresholds);
                  assert(chan_overlaps);
                  
                  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
                  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out);

                    for(unsigned c=0;c<1<<1;c++){
                      int seed = c;

                      for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
                        ((int*)X_ref)[b] = pseudo_rand(&seed);

                      
                      for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
                        ((int*)K_ref)[b] = pseudo_rand(&seed);

                      run_bin_config(
                          (bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b32_t*)X_ref,
                          (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
                          (int32_t*)thresholds_ref,
                          (int32_t*)thresholds,  
                          (int*) chan_overlaps,
                          x_height,
                          x_width, k_height, k_width, chans_in, chans_out, h_stride,
                          v_stride, seed, valid_impl);
                    }

                    free(X_ref);
                    free(Y);
                    free(Y_ref);
                    free(thresholds_ref);
                    free(thresholds);
                    free(K);
                    free(K_ref);
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


void impl_bconv2d_bin_DI_pseudo_random2(
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
          int32_t *thresholds_ref = (int32_t *)malloc(sizeof(int32_t)*(chans_out+(16 - chans_out%16)));
          bnn_b32_t *data_scratch = (bnn_b32_t *)malloc(sizeof(bnn_b32_t)*(k_height * k_width * chan_words_in + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
          
          int32_t * thresholds = (int32_t *)malloc(sizeof(int32_t)*chans_out);
          int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

          bnn_b32_t * Y     = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y_height * y_width * chans_out/32);
          bnn_b32_t * Y_ref = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y_height * y_width * chans_out/32);

          assert(X_ref);
          assert(Y);
          assert(Y_ref);
          assert(thresholds_ref);
          assert(K);
          assert(K_ref);

          assert(thresholds);
          assert(chan_overlaps);

          // printf("k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
          //    k_height, k_width, x_height, x_width, chans_in, chans_out);

          int seed = 42;

          for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
            ((int*)X_ref)[b] = pseudo_rand(&seed);

          
          for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
            ((int*)K_ref)[b] = pseudo_rand(&seed);

          run_bin_config(
              (bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b32_t*)X_ref,
              (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
              (int32_t*)thresholds_ref,
              (int32_t*)thresholds, 
              (int*) chan_overlaps,
              x_height,
              x_width, k_height, k_width, chans_in, chans_out, 1,
              1, seed, valid_impl);

        free(X_ref);
        free(Y);
        free(Y_ref);
        free(thresholds_ref);
        free(thresholds);
        free(K);
        free(K_ref);

        free(chan_overlaps);
      }
    }
  }
}

static void run_bin_sub_image(
              bnn_b32_t* Y_p, 
              const bnn_b32_t* Y_ref_p, 
              const bnn_b32_t* X_p,
              const bnn_b32_t* K_p, 

              int32_t * thresholds,

              const nn_image_params_t* x,
              const nn_image_params_t* y,
              const nn_window_params_t* k,
              unsigned y_loc_x, unsigned y_loc_y, 
              unsigned y_sub_width, unsigned y_sub_height,
              void (* valid_impl)()){

  valid_impl(Y_p, X_p,
      K_p, thresholds, 
      x, y, k, y_loc_x, y_loc_y, y_sub_width, y_sub_height);

  bnn_b32_t(*Y)[y->width][y->channels/32] =
      (bnn_b32_t(*)[y->width][y->channels/32])Y_p;

  bnn_b32_t(*Y_ref)[y->width][y->channels/32] =
      (bnn_b32_t(*)[y->width][y->channels/32])Y_ref_p;

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
void impl_bconv2d_bin_DI_sub_image(
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
      bnn_b32_t * K = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(chans_out*full_k_height*full_k_width*chan_words_in + K_OVERREAD_WORDS));

      int32_t * thresholds = (int32_t *)malloc(sizeof(int32_t)*chans_out);
      int32_t * thresholds_ref = (int32_t *)malloc(sizeof(int32_t)*chans_out);
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

          bnn_b32_t * Y_ref = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y.height * y.width * y.channels/32);
          bnn_b32_t * Y     = (bnn_b32_t *) malloc(sizeof(bnn_b32_t) * y.height * y.width * y.channels/32);    

          if(y.height == 0 || y.width == 0)
            continue;

          for(unsigned i=0;i<1<<6;i++){

            for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
              ((int*)X_ref)[b] = pseudo_rand(&seed);
            
            for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
              ((int*)K_ref)[b] = pseudo_rand(&seed);

            unsigned receptive_volume = k.shape.width * k.shape.height * x.channels;

            //FIXME get the thresholds
            pick_threshold_params(thresholds_ref, chans_out, chans_in * full_k_height * full_k_width);


            larq_ref_bconv2d_bin_out(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref,
                            (int32_t*)Y_ref, (const int32_t *)thresholds_ref);

            // bnn_reorder_int8_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
            //                           chans_out, chan_overlaps);

            bnn_reorder_kernel_tensor((bnn_b32_t* )K, (bnn_b32_t* )K_ref, full_k_height, full_k_width, chans_in,
                                      chans_out, chan_overlaps);

            bnn_reorder_threshold_tensor(thresholds, thresholds_ref, chans_out,
                                        full_k_width * full_k_height * chans_in, chan_overlaps);


            for (unsigned y_loc_x = 0; y_loc_x<y.width; ++y_loc_x){
              for (unsigned y_loc_y = 0; y_loc_y<y.height; ++y_loc_y){
                for (unsigned y_sub_width = 1; y_sub_width<y.width-y_loc_x; ++y_sub_width){
                  for (unsigned y_sub_height = 1; y_sub_height<y.height-y_loc_y; ++y_sub_height){

                      size_t addressable_Y_bytes = y.height * y.width * y.channels;
                      memset(Y, undef_sentinal, addressable_Y_bytes);

                      run_bin_sub_image(
                        (bnn_b32_t*)Y, 
                        (const bnn_b32_t*)Y_ref,
                        (const bnn_b32_t*) X_ref,
                        (const bnn_b32_t*) K, 

                        thresholds,

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
      free(thresholds);
      free(thresholds_ref);
      free(chan_overlaps);
    }
  }
}

static void SISO_valid(   
      bnn_b32_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int32_t * thresholds,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
    x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bconv2d_bin_valid(Y_p, X_p,
                      K_p, thresholds,
                      data_scratch, x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);

  free(data_scratch);
}

static void DI_valid(   
      bnn_b32_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int32_t * thresholds,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bconv2d_bin_DI_valid(Y_p, (const bnn_b256_t*)X_p,
                      (const bnn_b256_t*)K_p, thresholds,
                      x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);
}


static void SISO_full(   
      bnn_b32_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int32_t * thresholds,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
    x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bconv2d_bin(Y_p, X_p,
                      K_p, thresholds, 
                      data_scratch,
                      x, y, k,
                      0, 0, y->width, y->height, 0, 0);
  free(data_scratch);
}

static void DI_full(   
      bnn_b32_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int32_t * thresholds,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bconv2d_bin_DI(Y_p, (const bnn_b256_t*)X_p,
                      (const bnn_b256_t*)K_p, thresholds, 
                      x, y, k,
                      0, 0, y->width, y->height, 0, 0);
}

void test_bconv2d_bin_pseudo_random(){
  impl_bconv2d_bin_DI_pseudo_random(1, 5, 1, 5, 32*1, 32*9, 32*1, 32*3, 32, 32, 1, 3, 1, 3, (void*)&SISO_full);
}

void test_bconv2d_bin_DI_DI_pseudo_random(){
  impl_bconv2d_bin_DI_pseudo_random(1, 4, 1, 4, 256*1, 256*2, 32*1, 32*3, 256, 32, 1, 3, 1, 3, (void*)&DI_full);
}

void test_bconv2d_bin_pseudo_random2(){
  impl_bconv2d_bin_DI_pseudo_random2(1, 32, 32, 32, 256, 32, 32, (void*)&SISO_full);
}

void test_bconv2d_bin_DI_DI_pseudo_random2(){
  impl_bconv2d_bin_DI_pseudo_random2(1, 32, 256, 32, 32, 256, 32, (void*)&DI_full);
}

void test_bconv2d_bin_sub_image(){
  impl_bconv2d_bin_DI_sub_image(5, 5, 3, 3, 32*1, 32*9, 32*1, 32*3, 32, 32, 1, 1, 3, 3, (void*)&SISO_valid);
}

void test_bconv2d_bin_DI_DI_sub_image(){
  impl_bconv2d_bin_DI_sub_image(5, 5, 3, 3, 256*1, 256*2, 32*1, 32*3, 256, 32, 1, 1, 3, 3, (void*)&DI_valid);
}
//TODO define channel strides in lib_nn

void test_bnn_conv2d_bin() {
  UNITY_SET_FILE();

  RUN_TEST(test_bconv2d_bin_pseudo_random);
  RUN_TEST(test_bconv2d_bin_DI_DI_pseudo_random);

  RUN_TEST(test_bconv2d_bin_pseudo_random2);
  RUN_TEST(test_bconv2d_bin_DI_DI_pseudo_random2);

  RUN_TEST(test_bconv2d_bin_sub_image);
  RUN_TEST(test_bconv2d_bin_DI_DI_sub_image);
}
