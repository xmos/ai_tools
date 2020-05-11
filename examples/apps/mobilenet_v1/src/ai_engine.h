#ifndef MOBILENET_V1_APP_H_
#define MOBILENET_V1_APP_H_

extern "C" {

void ai_initialize(unsigned char **input, unsigned *input_size,
                   unsigned char **output, unsigned *output_size);
void ai_invoke();
};

#endif  // MOBILENET_V1_APP_H_
