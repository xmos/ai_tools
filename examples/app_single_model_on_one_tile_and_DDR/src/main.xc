#include <stdint.h>
#include <stdio.h>
#include <platform.h>

extern void model_init(unsigned);
extern void inference();

// Hack for xc to be happy with EXTMEM
// Have to specify the extern array size
#include "model_weights.h"
extern const int8_t weights[WEIGHTS_SIZE];
int8_t* weights_ptr = (int8_t*)&weights;

int main(void) {

    par {

        on tile[0]: {
            unsafe {
            model_init((unsigned)(weights_ptr));
            inference();
            }
        }
    }
    return 0;
}
