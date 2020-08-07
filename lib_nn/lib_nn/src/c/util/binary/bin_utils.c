#include "nn_bin_utils.h"

void set_bit_b32(bnn_b32_t* x, unsigned ch, bnn_bool_t val) {
  *x &= (~(1 << ch));
  unsigned bit = ((1 - val) / 2) & 1;
  *x |= (bit << ch);
}

void set_bit_b256(bnn_b256_t* x, unsigned ch, bnn_bool_t val) {
  unsigned bytes_per_vector_subword = sizeof(((bnn_b256_t*)0)->d[0]);
  unsigned vector_bytes = sizeof(((bnn_b256_t*)0)->d);
  unsigned vector_subword_count = vector_bytes / bytes_per_vector_subword;

  unsigned vector_word = ch / (vector_bytes * 8);
  unsigned vector_subword = ch / (bytes_per_vector_subword * 8);
  unsigned vector_subword_bit = ch % (bytes_per_vector_subword * 8);

  x[vector_word].d[vector_subword] &= (~(1 << vector_subword_bit));

  unsigned bit = ((1 - val) / 2) & 1;
  x[vector_word].d[vector_subword] |= (bit << vector_subword_bit);
}

bnn_bool_t get_bit_b32(bnn_b32_t* x, unsigned ch) {
  bnn_b32_t t = x[ch / 32];
  unsigned bit = (t >> ch) & 1;
  bnn_bool_t val = (1 - (bit)*2);
  return val;
}

bnn_bool_t get_bit_b256(bnn_b256_t* x, unsigned ch) {
  unsigned bytes_per_vector_subword = sizeof(((bnn_b256_t*)0)->d[0]);
  unsigned vector_bytes = sizeof(((bnn_b256_t*)0)->d);
  unsigned vector_subword_count = vector_bytes / bytes_per_vector_subword;

  unsigned vector_word = ch / (vector_bytes * 8);
  unsigned vector_subword = ch / (bytes_per_vector_subword * 8);
  unsigned vector_subword_bit = ch % (bytes_per_vector_subword * 8);

  unsigned bit = (x[vector_word].d[vector_subword] >> vector_subword_bit) & 1;

  bnn_bool_t val = (1 - (bit)*2);
  return val;
}

void pack_bits_b32(bnn_bool_t* unpacked_p, bnn_b32_t* packed_p, unsigned count,
                   unsigned channels) {
  unsigned vector_bytes = sizeof(bnn_b32_t);
  unsigned vector_words =
      (channels + (vector_bytes * 8 - 1)) / (vector_bytes * 8);

  bnn_bool_t(*unpacked)[channels] = (bnn_bool_t(*)[channels])unpacked_p;
  bnn_b32_t(*packed)[vector_words] = (bnn_b32_t(*)[vector_words])packed_p;

  for (unsigned ch = 0; ch < channels; ch++) {
    for (unsigned element = 0; element < count; element += 1) {
      set_bit_b32(packed[element], ch, unpacked[element][ch]);
    }
  }
}

void pack_bits_b256(bnn_bool_t* unpacked_p, bnn_b256_t* packed_p,
                    unsigned count, unsigned channels) {
  unsigned vector_bytes = sizeof(((bnn_b256_t*)0)->d);

  unsigned vector_words =
      (channels + (vector_bytes * 8 - 1)) / (vector_bytes * 8);

  bnn_bool_t(*unpacked)[channels] = (bnn_bool_t(*)[channels])unpacked_p;
  bnn_b256_t(*packed)[vector_words] = (bnn_b256_t(*)[vector_words])packed_p;

  for (unsigned ch = 0; ch < channels; ch++) {
    for (unsigned element = 0; element < count; element += 1) {
      set_bit_b256(packed[element], ch, unpacked[element][ch]);
    }
  }
}
