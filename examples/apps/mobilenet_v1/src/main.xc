#include <platform.h>
#include <xscope.h>

#include "app.h"

// #define INPUT_SIZE_BYTES 128 * 128 * 3

unsigned char *unsafe input;
unsigned char *unsafe output;

unsafe {
  void process_xscope(chanend xscope_data_in) {
    int bytes_read = 0;

    xscope_connect_data_from_host(xscope_data_in);
    while (1) {
      select {
      case xscope_data_from_host(xscope_data_in, (unsigned char *alias)input,
                                 bytes_read):
        if (bytes_read) {
          invoke_tflite();
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
      setup_tflite(input, output);
      process_xscope(xscope_data_in);
    }
  }

  return 0;
}