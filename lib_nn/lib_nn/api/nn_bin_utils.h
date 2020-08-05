#include "nn_bin_types.h"

void set_bit_b32(bnn_b32_t* x, unsigned ch, bnn_bool_t val);
void set_bit_b256(bnn_b256_t* x, unsigned ch, bnn_bool_t val);

bnn_bool_t get_bit_b32(bnn_b32_t* x, unsigned ch);
bnn_bool_t get_bit_b256(bnn_b256_t* x, unsigned ch);

void pack_bits_b32(bnn_bool_t* unpacked_p, bnn_b32_t* packed_p, unsigned count,
                   unsigned channels);
void pack_bits_b256(bnn_bool_t* unpacked_p, bnn_b256_t* packed_p,
                    unsigned count, unsigned channels);
