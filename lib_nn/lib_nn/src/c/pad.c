#include <assert.h>
#include <string.h>

#include "nn_operator.h"
// #include "nn_op_structs.h"

void pad_prepare(nn_pad_plan_t* plan, const padding_sizes_t* p,
                 const nn_image_params_t* x, const unsigned bytes_per_pixel) {
  assert(bytes_per_pixel % 4 == 0);

  unsigned padded_row_bytes = bytes_per_pixel * (p->left + x->width + p->right);
  plan->top_pad_bytes = padded_row_bytes * p->top;
  plan->bottom_pad_bytes = padded_row_bytes * p->bottom;

  plan->mid_loop_count = x->height;
  plan->left_pad_bytes = bytes_per_pixel * p->left;
  plan->mid_copy_bytes = bytes_per_pixel * x->width;
  plan->right_pad_bytes = bytes_per_pixel * p->right;
}

void memset32(void* str, uint32_t c, size_t n) {
  for (unsigned i = 0; i < n / sizeof(c); i++) {
    ((uint32_t*)str)[i] = c;
  }
}

void pad_run(void* y, void* x, const nn_pad_plan_t* p, uint32_t pad_value) {
  memset32(y, pad_value, p->top_pad_bytes);
  y += p->top_pad_bytes;
  for (unsigned i = 0; i < p->mid_loop_count; i++) {
    memset32(y, pad_value, p->left_pad_bytes);
    y += p->left_pad_bytes;

    memcpy(y, x, p->mid_copy_bytes);
    y += p->mid_copy_bytes;
    x += p->mid_copy_bytes;

    memset32(y, pad_value, p->right_pad_bytes);
    y += p->right_pad_bytes;
  }
  memset32(y, pad_value, p->bottom_pad_bytes);
}

void pad_ref(void* y, void* x, const padding_sizes_t* p,
             const nn_image_params_t* xp, const unsigned bytes_per_pixel,
             uint32_t pad_value) {
  unsigned top_pad = p->top;
  unsigned left_pad = p->left;
  unsigned right_pad = p->right;
  unsigned bottom_pad = p->bottom;

  char(*Y)[xp->width + left_pad + right_pad][bytes_per_pixel] =
      (char(*)[xp->width + left_pad + right_pad][bytes_per_pixel]) y;
  char(*X)[xp->width][bytes_per_pixel] = (char(*)[xp->width][bytes_per_pixel])x;

  memset32(y, pad_value,
           (xp->width + left_pad + right_pad) *
               (top_pad + bottom_pad + xp->height) * bytes_per_pixel);

  for (unsigned h = 0; h < xp->height; h++) {
    for (unsigned w = 0; w < xp->width; w++) {
      memcpy(Y[h + top_pad][w + left_pad], X[h][w], bytes_per_pixel);
    }
  }
}