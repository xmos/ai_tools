#include <string.h>

#include "nn_operator.h"
#include "nn_op_structs.h"

void pad_perpare(nn_pad_plan_t* plan, const PaddingValues* p,
                 const nn_image_params_t* x, const unsigned bytes_per_pixel) {
  plan->top_pad_bytes =
      bytes_per_pixel *
      (p->height * ((2 * p->width) + p->width_offset + x->width) + p->width);

  plan->mid_loop_count = x->height - 1;
  plan->mid_copy_bytes = bytes_per_pixel * (x->width);
  plan->mid_pad_bytes = bytes_per_pixel * ((2 * p->width) + p->width_offset);

  plan->bottom_pad_bytes =
      bytes_per_pixel * ((p->height + p->height_offset) *
                             ((2 * p->width) + p->width_offset + x->width) +
                         p->width);
}

void pad_run(void* y, void* x, const nn_pad_plan_t* p) {
  unsigned pad_value = 0;
  memset(y, pad_value, p->top_pad_bytes);
  y += p->top_pad_bytes;
  for (unsigned i = 0; i < p->mid_loop_count + 1; i++) {
    memcpy(y, x, p->mid_copy_bytes);
    y += p->mid_copy_bytes;
    x += p->mid_copy_bytes;
    memset(y, pad_value, p->mid_pad_bytes);
    y += p->mid_pad_bytes;
  }
  memset(y, pad_value, p->bottom_pad_bytes);
}

void pad_ref(void* y, void* x, const PaddingValues* p,
             const nn_image_params_t* xp, const unsigned bytes_per_pixel) {
  unsigned pad_value = 0;
  unsigned top_pad = p->height;
  unsigned left_pad = p->width;
  unsigned right_pad = p->width + p->width_offset;
  unsigned bottom_pad = p->height + p->height_offset;

  char(*Y)[xp->width + left_pad + right_pad][bytes_per_pixel] =
      (char(*)[xp->width + left_pad + right_pad][bytes_per_pixel]) y;
  char(*X)[xp->width][bytes_per_pixel] = (char(*)[xp->width][bytes_per_pixel])x;

  memset(y, pad_value,
         (xp->width + left_pad + right_pad) *
             (top_pad + bottom_pad + xp->height) * bytes_per_pixel);

  for (unsigned h = 0; h < xp->height; h++) {
    for (unsigned w = 0; w < xp->width; w++) {
      memcpy(Y[h + top_pad][w + left_pad], X[h][w], bytes_per_pixel);
    }
  }
}