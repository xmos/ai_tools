#ifndef FAKE_KERNELS_H
#define FAKE_KERNELS_H

#include <stdint.h>


void conv2d_deepin_deepout_relu();
void conv2d_shallowin_deepout_relu();
void maxpool2d_deep();
void fc_deepin_shallowout_lin();

#endif /* FAKE_KERNELS_H */