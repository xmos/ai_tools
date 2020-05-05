// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <cstdio>
#include <cstring>

#include "ai_engine.h"

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
  printf("Output tensor:\n");
  for (int i = 0; i < output_size; i++) {
    printf("   index=%u   value=%i\n", i, (signed char)output[i]);
  }
}

int main(int argc, char *argv[]) {
  ai_initialize(&input, &input_size, &output, &output_size);

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
