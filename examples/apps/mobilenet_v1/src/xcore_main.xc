// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include <platform.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xscope.h>

#include "ai_engine.h"

unsigned input_size;
unsigned char *unsafe input;
unsigned output_size;
unsigned char *unsafe output;

unsafe {
  void print_output() {
    for (int i = 0; i < output_size; i++) {
      printf("Output index=%u, value=%i\n", i, (signed char)output[i]);
    }
    printf("DONE!\n");
  }

  void process_xscope(chanend xscope_data_in) {
    int input_bytes = 0;
    int bytes_read = 0;
    unsigned char buffer[256];

    xscope_connect_data_from_host(xscope_data_in);
    xscope_mode_lossless();
    while (1) {
      select {
        case xscope_data_from_host(xscope_data_in, buffer, bytes_read):
          memcpy(input + input_bytes, buffer, bytes_read - 1);
          input_bytes += bytes_read - 1;
          if (input_bytes == input_size) {
            ai_invoke();
            print_output();
            input_bytes = 0;
          }
          break;
      }
    }
  }
}

int main(void) {
  chan xscope_data_in;

  par {
    xscope_host_data(xscope_data_in);
    on tile[0] : {
      ai_initialize(&input, &input_size, &output, &output_size);
      process_xscope(xscope_data_in);
    }
  }

  return 0;
}