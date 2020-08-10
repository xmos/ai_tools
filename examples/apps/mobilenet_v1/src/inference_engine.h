// Copyright (c) 2019, XMOS Ltd, All rights reserved
#ifndef MOBILENET_V1_APP_H
#define MOBILENET_V1_APP_H_

#ifdef __cplusplus
extern "C" {
#endif

void initialize(unsigned char **input, unsigned *input_size,
                unsigned char **output, unsigned *output_size);
void invoke();

#ifdef __cplusplus
};
#endif

#endif  // MOBILENET_V1_APP_H_
