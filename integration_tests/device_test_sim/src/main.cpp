#include <stdio.h>
#include <assert.h>

#include "model.tflite.h"
#include "tensorflow/lite/micro/memory_helpers.h"

// Self-check of the generated *_dims_len() / *_dims() helpers.
static
void check_dims(TfLiteTensor *tensor,
                int dims_len, const int *dims, size_t tensor_bytes)
{
    size_t element_count = 1;
    size_t element_size = 0;

    for (auto i = 0; i < dims_len; i++) {
        element_count *= dims[i];
    }
    tflite_micro::TfLiteTypeSizeOf(tensor->type, &element_size);
    assert(element_count * element_size == tensor_bytes);
}

static
void get_input(int8_t *inputs, unsigned input_size)
{
    FILE * input_file = fopen("input.bin", "rb");
    assert(input_file != NULL);

    // check size of file should be exact to input
    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
    assert(file_size == input_size);

    fread(inputs, 1, input_size, input_file);
    fclose(input_file);
}

static
void set_output(int8_t *outputs, unsigned output_size)
{
    FILE * output_file = fopen("sim_output.bin", "wb");
    assert(output_file != NULL);
    fwrite(outputs, 1, output_size, output_file);
    fclose(output_file);
}

void generic_run_model(void)
{
    model_init(NULL);
    int8_t *inputs = (int8_t *)model_input_ptr(0);
    int8_t *outputs = (int8_t *)model_output_ptr(0);
    unsigned input_size = model_input_size(0);
    unsigned output_size = model_output_size(0);
    TfLiteTensor *input = model_input(0);
    TfLiteTensor *output = model_output(0);
    int input_dims_len = model_input_dims_len(0);
    int output_dims_len = model_output_dims_len(0);
    int *input_dims = model_input_dims(0);
    int *output_dims = model_output_dims(0);

    check_dims(input, input_dims_len, input_dims, input_size);
    check_dims(output, output_dims_len, output_dims, output_size);

    get_input(inputs, input_size);
    model_invoke();
    set_output(outputs, output_size);
    return;
}


int main(void) {
    generic_run_model();
    return 0;
}
