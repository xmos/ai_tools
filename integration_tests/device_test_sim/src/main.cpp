#include <stdio.h>
#include <assert.h>

#include "model.tflite.h"

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
    
    get_input(inputs, input_size);
    model_invoke();
    set_output(outputs, output_size);
    return;
}


int main(void) {
    generic_run_model();
    return 0;
}
