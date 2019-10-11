
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "argmax_16.h"
#include "cifar-10.h"

#define TEST_INPUT_SIZE = 32 * 32 * 4

static int load_test_input(const char *filename, xc_conv2d_shallowin_deepout_relu_input_t *input)
{
    FILE *fd = fopen(filename, "rb");
    fseek(fd, 0, SEEK_END);
    size_t fsize = ftell(fd);
    size_t esize = sizeof(xc_conv2d_shallowin_deepout_relu_input_t);

    if (fsize != esize) {
        printf("Incorrect input file size. Expected %d bytes.\n", esize);
        return 0;
    }

    fseek(fd, 0, SEEK_SET);
    fread(input, 1, fsize, fd);
    fclose(fd);

    return 1;
}

int main(int argc, char *argv[])
{
    xc_conv2d_shallowin_deepout_relu_input_t input = { 0 };
    xc_argmax_16_output_t output = { 0 };
    char classification[12] = { 0 };

    if (argc > 1)
    {
        printf("starting: input filename=%s\n", argv[1]);
        if (!load_test_input(argv[1], &input))
            return -1;
    } else {
        printf("starting: no input file\n");
    }

    xcore_model_quant(&input, &output);
    switch (output[0])
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
            snprintf(classification, 8, "unknown"); 
    }
    printf("finished: classification=%s\n", classification);
 
    return 0;
}
