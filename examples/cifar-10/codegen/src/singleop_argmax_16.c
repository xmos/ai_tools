#include "nn_operator.h"
#include "singleop_argmax_16.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif


void singleop_argmax_16(const x_int16_t *x_int16, identity_t *Identity)
{

     argmax_16(x_int16, (int32_t *) Identity, 10);
}
