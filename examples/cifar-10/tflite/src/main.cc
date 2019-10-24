
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
//#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "cifar10_model.h"

#define TEST_INPUT_SIZE = 32 * 32 * 4

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 50000;  //TODO: How can this be determined?
uint8_t tensor_arena[kTensorArenaSize];

static int load_test_input(const char *filename, char *input, size_t esize)
{
    FILE *fd = fopen(filename, "rb");
    fseek(fd, 0, SEEK_END);
    size_t fsize = ftell(fd);

    if (fsize != esize) {
        printf("Incorrect input file size. Expected %d bytes.\n", esize);
        return 0;
    }

    fseek(fd, 0, SEEK_SET);
    fread(input, 1, fsize, fd);
    fclose(fd);

    return 1;
}

// static int save_test_output(const char *filename, const char *output, size_t osize)
// {
//     FILE *fd = fopen(filename, "wb");
//     fwrite(output , sizeof(int8_t), osize, fd);
//     fclose(fd);

//     return 1;
// }

static void setup() {
    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(cifar10_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // This pulls in all the operation implementations we need.
    static tflite::ops::micro::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
    return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

int main(int argc, char *argv[])
{
    setup();

    if (argc > 1)
    {
        printf("starting: input filename=%s\n", argv[1]);
        // Load input tensor
        if (!load_test_input(argv[1], input->data.raw, input->bytes))
            return -1;
    } else {
        printf("starting: no input file\n");
        memset(input->data.raw, 0, input->bytes);
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed on input filename=%s\n", argv[1]);
        return -1;
    }

    // size_t output_size = output->bytes;
    // for (int i=0; i<output_size/2; i++)
    // {
    //     printf("%04X\n", output->data.i16[i]);
    // }

    // save_test_output("fc_test_0.out", output->data.raw, output_size);

    char classification[12] = { 0 };
    
    switch (output->data.i32[0])
    {
        case 0:
            snprintf(classification, 9, "airplane"); 
            break;
        case 1: 
            snprintf(classification, 11, "automobile"); 
            break;
        case 2:
            snprintf(classification, 5, "bird"); 
            break;
        case 3:
            snprintf(classification, 4, "cat"); 
            break;
        case 4:
            snprintf(classification, 5, "deer"); 
            break;
        case 5:
            snprintf(classification, 4, "dog"); 
            break;
        case 6:
            snprintf(classification, 5, "frog"); 
            break;
        case 7:
            snprintf(classification, 6, "horse"); 
            break;
        case 8:
            snprintf(classification, 5, "ship"); 
            break;
        case 9:
            snprintf(classification, 6, "truck"); 
            break;
        default:
            break;
    }
    printf("finished: classification=%s\n", classification);
 
    return 0;
}
