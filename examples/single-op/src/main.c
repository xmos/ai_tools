
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "singleop_conv2d_shallowin_deepout.h"
#include "singleop_conv2d_deepin_deepout.h"
#include "singleop_fc_deepin_shallowout_final.h"
#include "singleop_argmax_16.h"

static int load_test_input(const char *filename, int8_t *input, size_t esize)
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

static int save_test_input(const char *filename, const int8_t *output, size_t osize)
{
    FILE *fd = fopen(filename, "wb");
    fwrite(output , sizeof(int8_t), osize, fd);
    fclose(fd);

    return 1;
}

int test_argmax(char* input_filename, char* output_filename)
{
    argmax_16_x_int16_t input = { 0 };
    argmax_16_identity_t output = { 0 };

    printf("test_argmax: input filename=%s\n", input_filename);
    if (!load_test_input(input_filename, (int8_t *) &input, sizeof(input)))
        return -1;

    singleop_argmax_16(&input, &output);

    printf("outputs:\n");
    size_t output_len = sizeof(output) / sizeof(output[0]);
    for (int i=0; i<output_len; i++)
    {
        printf("%04X   %ld\n", (unsigned int) output[i], (long) output[i]);
    }

    save_test_input(output_filename, (int8_t *)&output, sizeof(output));

    return 0;
}

int test_fc_deepin_shallowout(char* input_filename, char* output_filename)
{
    flatten_input_int8_t input = { 0 };
    xc_fc_deepin_shallowout_final_output_t output = { 0 };

    printf("test_fc_deepin_shallowout: input filename=%s\n", input_filename);
    if (!load_test_input(input_filename, (int8_t *) &input, sizeof(input)))
        return -1;

    singleop_fc_deepin_shallowout_final(&input, &output);

    printf("outputs:\n");
    size_t output_len = sizeof(output) / sizeof(output[0]);
    for (int i=0; i<output_len; i++)
    {
        printf("%04X   %d\n", output[i], output[i]);
    }

    save_test_input(output_filename, (int8_t *)&output, sizeof(output));

    return 0;
}

int test_conv2d_deepin_deepout(char* input_filename, char* output_filename)
{
    conv2d_deepin_deepout_input_t input = { 0 };
    conv2d_deepin_deepout_identity_t output = { 0 };

    printf("test_conv2d_deepin_deepout: input filename=%s\n", input_filename);
    if (!load_test_input(input_filename, (int8_t *) &input, sizeof(input)))
        return -1;

    singleop_conv2d_deepin_deepout(&input, &output);

    printf("outputs:\n");
    size_t output_len = sizeof(output) / sizeof(output[0]);
    for (int i=0; i<output_len; i++)
    {
        printf("%04X   %d\n", output[i], output[i]);
    }

    save_test_input(output_filename, (int8_t *)&output, sizeof(output));

    return 0;
}

int test_conv2d_shallowin_deepout(char* input_filename, char* output_filename)
{
    conv2d_shallowin_deepout_input_t input = { 0 };
    conv2d_shallowin_deepout_identity_t output = { 0 };

    printf("test_conv2d_shallowin_deepout: input filename=%s\n", input_filename);
    if (!load_test_input(input_filename, (int8_t *) &input, sizeof(input)))
        return -1;

    singleop_conv2d_shallowin_deepout(&input, &output);

    printf("outputs:\n");
    size_t output_len = sizeof(output) / sizeof(output[0]);
    for (int i=0; i<output_len; i++)
    {
        printf("%04X   %d\n", output[i], output[i]);
    }

    save_test_input(output_filename, (int8_t *)&output, sizeof(output));

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("exiting: not enough input arguments\n");
        return -1;
    }

    int ret = -1;
    if (strcmp(argv[1], "fc_deepin_shallowout") == 0)
    {
        ret = test_fc_deepin_shallowout(argv[2], argv[3]);
    }
    else if (strcmp(argv[1], "conv2d_deepin_deepout") == 0)
    {
        ret = test_conv2d_deepin_deepout(argv[2], argv[3]);
    }
    else if (strcmp(argv[1], "conv2d_shallowin_deepout") == 0)
    {
        ret = test_conv2d_shallowin_deepout(argv[2], argv[3]);
    }
    else if (strcmp(argv[1], "argmax") == 0)
    {
        ret = test_argmax(argv[2], argv[3]);
    }
    else
    {
        printf("exiting: unknown mode %s\n", argv[1]);
        return -1;
    }

    return ret;
}
