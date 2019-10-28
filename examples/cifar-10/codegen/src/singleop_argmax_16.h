#ifndef SINGLEOP_ARGMAX_16_H
#define SINGLEOP_ARGMAX_16_H

typedef int16_t x_int16_t[1 * 10];
typedef int32_t identity_t[1];

void singleop_argmax_16(const x_int16_t *x_int16, identity_t *Identity);

#endif /* SINGLEOP_ARGMAX_16_H */