#ifndef SINGLEOP_ARGMAX_16_H
#define SINGLEOP_ARGMAX_16_H

typedef int16_t argmax_16_x_int16_t[1 * 10];
typedef int32_t argmax_16_identity_t[1];

void singleop_argmax_16(const argmax_16_x_int16_t *x_int16, argmax_16_identity_t *Identity);

#endif /* SINGLEOP_ARGMAX_16_H */