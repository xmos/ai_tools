
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "fc_deepin_shallowout_final.h"

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


int test_fc_deepin_shallowout(char* input_filename, char* output_filename)
{
    flatten_input_int8_t input = { 0 };
    xc_fc_deepin_shallowout_final_output_t output = { 0 };

    printf("test_fc_deepin_shallowout: input filename=%s\n", input_filename);
    if (!load_test_input(input_filename, (int8_t *) &input, sizeof(input)))
        return -1;

    fc_deepin_shallowout_final(&input, &output);

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

    int fc = test_fc_deepin_shallowout(argv[1], argv[2]);

    return fc;
}
