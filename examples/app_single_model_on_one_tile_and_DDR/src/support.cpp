#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <xcore/channel.h>
#include "model.tflite.h"
#include "image.h"

// Simple checksum calc
unsigned char checksum_calc(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}

// Quantize float to int8
int quantize_input(float n) {
  return n/model_input_scale(0) + model_input_zeropoint(0);
}

// Dequantize int8 to float
float dequantize_output(int n) {
  return (n - model_output_zeropoint(0)) * model_output_scale(0);
}

void init(unsigned flash_data) { model_init((void *)flash_data); }

void run() {
  printf("\nBefore model init");


  // Set input
  int8_t *in = model_input(0)->data.int8;
  int size = model_input_size(0);
  for (int i=0;i<size;++i) {
    in[i] = lion[i] - 128;
  }


  printf("\nBefore model invoke");

  // Run inference
  model_invoke();

  printf("\nDone model invoke");

  // Find top three classes
  int maxIndex1 = -1;
  int max1 = -128;
  int maxIndex2 = -1;
  int max2 = -128;
  int maxIndex3 = -1;
  int max3 = -128;
  int8_t *out = model_output(0)->data.int8;
  for (int i = 0; i < model_output_size(0); ++i) {
    if (out[i] > max1) {
      max3 = max2;
      maxIndex3 = maxIndex2;
      max2 = max1;
      maxIndex2 = maxIndex1;
      max1 = out[i];
      maxIndex1 = i;
    }
  }

  printf("\nClass with max1 value = %d and probability = %f", maxIndex1, dequantize_output(max1));
  printf("\nClass with max2 value = %d and probability = %f", maxIndex2, dequantize_output(max2));
  printf("\nClass with max3 value = %d and probability = %f", maxIndex3, dequantize_output(max3));
}

extern "C" {
void model_init(unsigned flash_data) { init(flash_data); }

void inference() { run(); }

//__attribute__ ((section(".ExtMem.data")))
int8_t src[38944];
int8_t dst[38944];
int time_t0, time_t1;
extern void OptMemCpy2(void* d, void* s, int n);
extern void vpu_memcpy_ext(void *, const void *, unsigned int);
extern void vpu_memcpy_vector_ext(void *, const void *, int);

#include<string.h>

void test_ddr(){
  for(int i = 0; i < 38944; i ++){
    src[i] = i;
  }
  int time = 0;
  char *d = (char*)((((unsigned)&dst[0]) + 31) & 0xffffffe0);
  int8_t *s = (int8_t *)(((unsigned)src + 31) &0xffffffe0);

  for (int i = 0; i < 82; i++) {
    #ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t0));
    #endif
    //memcpy(dst, src, 304*32*4);
    //OptMemCpy2(d, s, 304 * 32);
    //vpu_memcpy_ext(d, s, 304 * 32 * 4);
    vpu_memcpy_vector_ext(dst, src, 304);
    #ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t1));
    #endif
    time += time_t1 - time_t0;
  }
  printf("\ntotal time = %.2fms\nMBps = %.2f", time/100000.0, (304 * 32 * 4 * 82)/(time/100.0));
}

}
