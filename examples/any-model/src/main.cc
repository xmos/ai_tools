
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 400000;  // Hopefully this is big enough for all tests
uint8_t tensor_arena[kTensorArenaSize];

static int load_model(const char *filename, char **buffer, size_t *size)
{
    FILE *fd = fopen(filename, "rb");
    fseek(fd, 0, SEEK_END);
    size_t fsize = ftell(fd);

    *buffer = (char *)malloc(fsize);

    fseek(fd, 0, SEEK_SET);
    fread(*buffer, 1, fsize, fd);
    fclose(fd);

    *size = fsize;

    return 1;
}

static int load_input(const char *filename, char *input, size_t esize)
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

static int save_output(const char *filename, const char *output, size_t osize)
{
    FILE *fd = fopen(filename, "wb");
    fwrite(output , sizeof(int8_t), osize, fd);
    fclose(fd);

    return 1;
}

static void setup_tflite(const char *model_buffer) {
    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(model_buffer);
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

    if (argc < 4)
    {
        printf("Three arguments expected: mode.tflite input-file output-file\n");
        return -1;
    }

    char *model_filename = argv[1];
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    // load model
    char *model_buffer = nullptr;
    size_t model_size;
    if (!load_model(model_filename, &model_buffer, &model_size))
    {
        printf("error loading model filename=%s\n", model_filename);
        return -1;
    }

    // setup runtime
    setup_tflite(model_buffer);

    // Load input tensor
    if (!load_input(input_filename, input->data.raw, input->bytes))
    {
        printf("error loading input filename=%s\n", input_filename);
        return -1;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed\n");
        return -1;
    }

    // save output
    if (!save_output(output_filename, output->data.raw, output->bytes))
    {
        printf("error saving output filename=%s\n", output_filename);
        return -1;
    }
    return 0;
}
