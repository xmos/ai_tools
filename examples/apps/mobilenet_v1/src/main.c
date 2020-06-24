// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stdio.h>
#include <string.h>

#include "ai_engine.h"

#ifdef XCORE

#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))

static char swmem_handler_stack[1024];

#include "lib_ops/api/device_memory.h"

// void app_main(unsigned char **input, unsigned *input_size,
//               unsigned char **output, unsigned *output_size) {
void app_main() {
#if defined(USE_SWMEM) || defined(USE_EXTMEM)
  // start SW_Mem handler
  swmem_setup();
  size_t stack_words;
  GET_STACKWORDS(stack_words, swmem_handler);
  printf("app_main 111\n");
  printf("stack_words = %d\n", stack_words);
  run_async(swmem_handler, NULL, stack_base(swmem_handler_stack, stack_words));
  printf("app_main 222\n");
#endif

  // ai_initialize(input, input_size, output, output_size);
}

#else  // must be x86

unsigned input_size;
unsigned char *input;
unsigned output_size;
unsigned char *output;

static int load_input(const char *filename) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  if (fsize != input_size) {
    printf("Incorrect input file size. Expected %d bytes.\n", input_size);
    return 0;
  }

  fseek(fd, 0, SEEK_SET);
  fread(input, 1, input_size, fd);
  fclose(fd);

  return 1;
}

void print_output() {
  for (int i = 0; i < output_size; i++) {
    printf("Output index=%u, value=%i\n", i, (signed char)output[i]);
  }
}

int main(int argc, char *argv[]) {
  ai_initialize(&input, &input_size, &output, &output_size);
  printf("111\n");
  if (argc > 1) {
    printf("input filename=%s\n", argv[1]);
    // Load input tensor
    if (!load_input(argv[1])) return -1;
  } else {
    printf("no input file\n");
    memset(input, 0, input_size);
  }

  ai_invoke();

  print_output();

  return 0;
}

#endif