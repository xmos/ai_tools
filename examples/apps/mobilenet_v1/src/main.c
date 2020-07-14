// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stdio.h>
#include <string.h>

#include "ai_engine.h"

static int input_bytes = 0;
static int input_size;
static unsigned char *input_buffer;
static int output_size;
static unsigned char *output_buffer;

void print_output() {
  for (int i = 0; i < output_size; i++) {
    printf("Output index=%u, value=%i\n", i, (signed char)output_buffer[i]);
  }
  printf("DONE!\n");
}

#ifdef XCORE

#include "lib_ops/api/device_memory.h"

#define STRINGIFY(NAME) #NAME
#define GET_STACKWORDS(DEST, NAME) \
  asm("ldc %[__dest], " STRINGIFY(NAME) ".nstackwords" : [ __dest ] "=r"(DEST))

__attribute__((aligned(8))) static char swmem_handler_stack[1024];

void app_main() {
#if defined(USE_SWMEM) || defined(USE_EXTMEM)
  // start SW_Mem handler
  swmem_setup();
  size_t stack_words;
  GET_STACKWORDS(stack_words, swmem_handler);
  run_async(swmem_handler, NULL,
            stack_base(swmem_handler_stack, stack_words + 2));
#endif

  ai_initialize(&input_buffer, &input_size, &output_buffer, &output_size);
}

void app_data(void *data, size_t size) {
  memcpy(input_buffer + input_bytes, data, size - 1);
  input_bytes += size - 1;
  if (input_bytes == input_size) {
    ai_invoke();
    print_output();
    input_bytes = 0;
  }
}

#else  // must be x86

static int load_input(const char *filename) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  if (fsize != input_size) {
    printf("Incorrect input file size. Expected %d bytes.\n", input_size);
    return 0;
  }

  fseek(fd, 0, SEEK_SET);
  fread(input_buffer, 1, input_size, fd);
  fclose(fd);

  return 1;
}

int main(int argc, char *argv[]) {
  ai_initialize(&input_buffer, &input_size, &output_buffer, &output_size);

  if (argc > 1) {
    printf("input filename=%s\n", argv[1]);
    if (!load_input(argv[1])) return -1;
  } else {
    printf("no input file\n");
    memset(input_buffer, 0, input_size);
  }

  ai_invoke();

  print_output();

  return 0;
}

#endif