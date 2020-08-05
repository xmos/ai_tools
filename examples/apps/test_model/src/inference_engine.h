// Copyright (c) 2019, XMOS Ltd, All rights reserved
#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void initialize(unsigned char *model_content, unsigned char *tensor_arena,
                size_t tensor_arena_size, unsigned char **input,
                unsigned *input_size, unsigned char **output,
                unsigned *output_size);
void invoke();

#ifdef __cplusplus
};
#endif

#endif  // INFERENCE_ENGINE_H_