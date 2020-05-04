#ifndef MOBILENET_V1_APP_H_
#define MOBILENET_V1_APP_H_

extern "C" {

void setup_tflite(unsigned char *input, unsigned char *output);
void invoke_tflite();
};

#endif // MOBILENET_V1_APP_H_
