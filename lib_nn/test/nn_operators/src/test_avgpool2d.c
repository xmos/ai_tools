
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"


#if USE_ASM(avgpool2d)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)


#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
#if TEST_C
    int8_t WORD_ALIGNED  Y_c[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;    uint32_t width;     } X;
        struct {    uint32_t height;    uint32_t width;     } Y;
        struct {    uint32_t height;    uint32_t width;
                    int32_t vstride;    int32_t hstride;    } W;
        unsigned channels;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               // Y            // W                      //Chans
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        16          },  //0
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        32          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         8          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        28          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         4          },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        MAX_CHANS   },
        
        {   { 1,  2},       { 1,  1},       {  1,  2,  1,  1},        16          },  //6
        {   { 2,  1},       { 1,  1},       {  2,  1,  1,  1},        16          },
        {   { 2,  2},       { 1,  1},       {  2,  2,  1,  1},        16          },
        {   { 1,  4},       { 1,  1},       {  1,  4,  1,  1},        16          },
        {   { 4,  1},       { 1,  1},       {  4,  1,  1,  1},        16          },
        {   { 4,  4},       { 1,  1},       {  4,  4,  1,  1},        16          },

        {   { 1,  3},       { 1,  1},       {  1,  3,  1,  1},        16          },  //12
        {   { 3,  1},       { 1,  1},       {  3,  1,  1,  1},        16          },
        {   { 3,  3},       { 1,  1},       {  3,  3,  1,  1},        16          },
        {   { 5,  3},       { 1,  1},       {  5,  3,  1,  1},        16          },  
        {   { 9,  1},       { 1,  1},       {  9,  1,  1,  1},        16          },
        {   { 3, 13},       { 1,  1},       {  3, 13,  1,  1},        16          },

        {   { 1,  2},       { 1,  2},       {  1,  1,  1,  1},        16          },  //18
        {   { 2,  1},       { 2,  1},       {  1,  1,  1,  1},        16          },
        {   { 2,  2},       { 2,  2},       {  1,  1,  1,  1},        16          },
        {   { 1,  3},       { 1,  3},       {  1,  1,  1,  1},        16          },
        {   { 3,  3},       { 3,  3},       {  1,  1,  1,  1},        16          },
        {   { 4,  1},       { 4,  1},       {  1,  1,  1,  1},        16          },
        {   { 5,  7},       { 5,  7},       {  1,  1,  1,  1},        16          },
        
        {   { 1,  1},       { 1,  1},       {  1,  1,  2,  2},        16          },  //25
        {   { 4,  2},       { 2,  2},       {  1,  1,  2,  1},        16          },
        {   { 2,  4},       { 2,  2},       {  1,  1,  1,  2},        16          },
        {   { 4,  4},       { 2,  2},       {  1,  1,  2,  2},        16          },
        {   { 9,  9},       { 3,  3},       {  1,  1,  3,  3},        16          },
        
        {   { 4,  4},       { 2,  2},       {  2,  2,  2,  2},        16          },  //30
        {   { 4,  4},       { 3,  3},       {  2,  2,  1,  1},        16          },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        16          },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32          },
        {   {16, 16},       { 4,  4},       {  4,  4,  4,  4},        MAX_CHANS   },
        {   {25, 25},       { 5,  5},       {  5,  5,  5,  5},         8          },
        {   {32, 32},       { 4,  8},       {  8,  4,  8,  4},        24          },

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    memset(X, 120, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->X.height, casse->X.width, casse->channels };
        nn_image_params_t y_params = { casse->Y.height, casse->Y.width, casse->channels };

        nn_window_op_config_t window_config;
        nn_window_op_config_simple(&window_config, &x_params, &y_params, 
                                    casse->W.height, casse->W.width, 
                                    casse->W.vstride, casse->W.hstride);

        window_config.output.shape.height = y_params.height;
        window_config.output.shape.width = y_params.width;
        window_config.output.shape.channels = y_params.channels;

        nn_avgpool2d_plan_t plan;

        avgpool2d_init(&plan, &x_params, &y_params, &window_config);

        for(int r = 0; r < x_params.height; r++){
            for(int c = 0; c < x_params.width; c++){
                for(int ch = 0; ch < x_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, r, c, ch)] = (ch&1)? 120 : -120;
                }
            }
        }

#if (DEBUG_ON || 0)
    PRINTF("plan.window.output.rows                 = %d\n", plan.window.output.rows              );
    PRINTF("plan.window.output.cols                 = %d\n", plan.window.output.cols              );
    PRINTF("plan.window.output.channels             = %d\n", plan.window.output.channels          );
    PRINTF("plan.window.window.rows                 = %d\n", plan.window.window.rows              );
    PRINTF("plan.window.window.cols                 = %d\n\n", plan.window.window.cols              );
    PRINTF("plan.window.start_stride.x              = %d\n", plan.window.start_stride.x           );
    PRINTF("plan.window.inner_stride.horizontal.x   = %d\n", plan.window.inner_stride.horizontal.x);
    PRINTF("plan.window.inner_stride.vertical.x     = %d\n", plan.window.inner_stride.vertical.x  );
    PRINTF("plan.window.outer_stride.horizontal.x   = %d\n", plan.window.outer_stride.horizontal.x);
    PRINTF("plan.window.outer_stride.vertical.x     = %d\n", plan.window.outer_stride.vertical.x  );
    PRINTF("plan.window.chan_grp_stride.x           = %d\n\n", plan.window.chan_grp_stride.x        );
    PRINTF("plan.window.start_stride.y              = %d\n", plan.window.start_stride.y           );
    PRINTF("plan.window.outer_stride.horizontal.y   = %d\n", plan.window.outer_stride.horizontal.y);
    PRINTF("plan.window.outer_stride.vertical.y     = %d\n", plan.window.outer_stride.vertical.y  );
    PRINTF("plan.window.chan_grp_stride.y           = %d\n\n", plan.window.chan_grp_stride.y        );
    PRINTF("plan.window.scale                       = 0x%08X\n", plan.scale);
    PRINTF("plan.shift                              = 0x%08X\n", plan.shift);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, casse->Y.height * casse->Y.width * casse->channels);    //too expensive to write the whole image, so just do the part that's in play
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  casse->Y.height * casse->Y.width * casse->channels);
        avgpool2d_asm((int8_t*)Y_asm, (int8_t*)X, &plan);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                // PRINTF("y_base = %ld\n", y_base);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (chn&1)? 120 : -120;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = ((int8_t*)Y_c)[y_base + chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = ((int8_t*)Y_asm)[y_base + chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS
#undef DEBUG_ON






#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define CHANS       (32)
#define HEIGHT      (24)
#define WIDTH       (24)
void test_avgpool2d_case2()
{
    unsigned seed = 34524666;
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED Y_exp[HEIGHT][WIDTH][CHANS];

#if TEST_C
    int8_t WORD_ALIGNED  Y_c[HEIGHT][WIDTH][CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[HEIGHT][WIDTH][CHANS];
#endif

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;        uint32_t width;     } window;
        struct {    uint32_t row;           uint32_t col;       } x_start;
        struct {    uint32_t row;           uint32_t col;       } y_start;
    } test_case_t;

    const test_case_t casses[] = {
        {       {   1,  1 },        {   0,  0 },        {   0,  0 }         }, // 0
        {       {   2,  2 },        {   0,  0 },        {   0,  0 }         },
        {       {   3,  3 },        {   0,  0 },        {   0,  0 }         },
        {       {   4,  4 },        {   0,  0 },        {   0,  0 }         },
        {       {   5,  5 },        {   0,  0 },        {   0,  0 }         },
        {       {   6,  6 },        {   0,  0 },        {   0,  0 }         }, // 5
        {       {   8,  8 },        {   0,  0 },        {   0,  0 }         },
        {       {  12, 12 },        {   0,  0 },        {   0,  0 }         },
        {       {  24, 24 },        {   0,  0 },        {   0,  0 }         },
        {       {   1,  2 },        {   0,  0 },        {   0,  0 }         },
        {       {   2,  1 },        {   0,  0 },        {   0,  0 }         }, // 10
        {       {   3,  8 },        {   0,  0 },        {   0,  0 }         },
        {       {  24,  4 },        {   0,  0 },        {   0,  0 }         },
        {       {   1,  1 },        {   1,  1 },        {   0,  0 }         },
        {       {   1,  1 },        {   0,  0 },        {   1,  1 }         },
        {       {   2,  2 },        {   4,  4 },        {   0,  0 }         }, // 15
        {       {   2,  2 },        {   0,  0 },        {   8,  8 }         },
        {       {   3,  3 },        {   5,  1 },        {   8,  9 }         },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, CHANS };
        nn_image_params_t y_params = { HEIGHT, WIDTH, CHANS };

        nn_window_op_config_t window_config;
        nn_window_op_config_simple(&window_config, &x_params, &y_params, 
                                    casse->window.height, casse->window.width, 
                                    casse->window.height, casse->window.width);

        window_config.output.shape.height = (HEIGHT - casse->x_start.row) / casse->window.height;
        window_config.output.shape.width  = (WIDTH - casse->x_start.col)  / casse->window.width;

        if((HEIGHT - casse->y_start.row) < window_config.output.shape.height) 
            window_config.output.shape.height = (HEIGHT - casse->y_start.row);
        if((WIDTH  - casse->y_start.col) < window_config.output.shape.width) 
            window_config.output.shape.width = (WIDTH  - casse->y_start.col);

        window_config.window.start.rows = casse->x_start.row;
        window_config.window.start.cols = casse->x_start.col;
        window_config.window.start.channels = 0;

        window_config.output.start.rows = casse->y_start.row;
        window_config.output.start.cols = casse->y_start.col;
        window_config.output.start.channels = 0;
        
        nn_avgpool2d_plan_t params;

        avgpool2d_init(&params, &x_params, &y_params, &window_config);

        PRINTF("\t\tSetting X...\n");
        memset(Y_exp, 0xCC, sizeof(Y_exp));
        for(int vpos = 0; vpos < window_config.output.shape.height; vpos++){
            for(int hpos = 0; hpos < window_config.output.shape.width; hpos++){

                int x_top  = casse->x_start.row + vpos * casse->window.height;
                int x_left = casse->x_start.col + hpos * casse->window.width;

                for(int chn = 0; chn < y_params.channels;chn++){

                    // const unsigned pix = casse->window.height * casse->window.width;
                    // const unsigned pix_mod2 = pix & 0x01;
                    int8_t avg = pseudo_rand_uint32(&seed) & 0xFF;

                    for(int xr = 0; xr < casse->window.height; xr++){
                        for(int xc = 0; xc < casse->window.width; xc++){
                            X[x_top+xr][x_left+xc][chn] = avg;
                        }
                    }

                    Y_exp[vpos + casse->y_start.row][hpos + casse->y_start.col][chn] = avg;
                }
            }
        }

#if (DEBUG_ON || 0)
    PRINTF("plan.window.output.rows                 = %d\n", plan.window.output.rows              );
    PRINTF("plan.window.output.cols                 = %d\n", plan.window.output.cols              );
    PRINTF("plan.window.output.channels             = %d\n", plan.window.output.channels          );
    PRINTF("plan.window.window.rows                 = %d\n", plan.window.window.rows              );
    PRINTF("plan.window.window.cols                 = %d\n\n", plan.window.window.cols              );
    PRINTF("plan.window.start_stride.x              = %d\n", plan.window.start_stride.x           );
    PRINTF("plan.window.inner_stride.horizontal.x   = %d\n", plan.window.inner_stride.horizontal.x);
    PRINTF("plan.window.inner_stride.vertical.x     = %d\n", plan.window.inner_stride.vertical.x  );
    PRINTF("plan.window.outer_stride.horizontal.x   = %d\n", plan.window.outer_stride.horizontal.x);
    PRINTF("plan.window.outer_stride.vertical.x     = %d\n", plan.window.outer_stride.vertical.x  );
    PRINTF("plan.window.chan_grp_stride.x           = %d\n\n", plan.window.chan_grp_stride.x        );
    PRINTF("plan.window.start_stride.y              = %d\n", plan.window.start_stride.y           );
    PRINTF("plan.window.outer_stride.horizontal.y   = %d\n", plan.window.outer_stride.horizontal.y);
    PRINTF("plan.window.outer_stride.vertical.y     = %d\n", plan.window.outer_stride.vertical.y  );
    PRINTF("plan.window.chan_grp_stride.y           = %d\n\n", plan.window.chan_grp_stride.y        );
    PRINTF("plan.window.scale                       = 0x%08X\n", plan.scale);
    PRINTF("plan.shift                              = 0x%08X\n", plan.shift);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));    //too expensive to write the whole image, so just do the part that's in play
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &params);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(Y_asm));
        avgpool2d_asm((int8_t*)Y_asm, (int8_t*)X, &params);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = Y_exp[row][col][chn];
                    if(y_exp == -128)   y_exp = -127;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = Y_c[row][col][chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = Y_asm[row][col][chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }

    }

}
#undef WIDTH
#undef HEIGHT
#undef CHANS
#undef DEBUG_ON


#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define CHANS       (2*VPU_INT8_ACC_PERIOD)
#define X_HEIGHT    (12)
#define X_WIDTH     (12)
#define Y_HEIGHT    (6)
#define Y_WIDTH     (6)
void test_avgpool2d_case3()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y_exp[Y_HEIGHT][Y_WIDTH][CHANS] = {{{0}}};
#if TEST_C
    int8_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS];
#endif

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {        nn_image_vect_t X;  nn_image_vect_t Y;  } start;
        struct {        nn_image_vect_t shape;                  } output;
        struct {
            struct {    uint32_t height;    uint32_t width;     } shape;
            struct {    int32_t vertical;   int32_t horizontal; } stride;
        } window;

    } test_case_t;

    const test_case_t casses[] = {
        //  X                   Y                   Out Shape          Win shape    stride               
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 6, 6, 32}},     {{ 2, 2},    { 2, 2}}},  // 0
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 3, 3, 32}},     {{ 4, 4},    { 4, 4}}},
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 3, 3, 32}},     {{ 2, 2},    { 2, 2}}},
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {  {{  0,  0, 16},      {  0,  0,  0}},     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {  {{  0,  0,  0},      {  0,  0, 16}},     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {  {{  0,  0, 16},      {  0,  0, 16}},     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}}},
        {  {{  2,  2,  0},      {  0,  0,  0}},     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}}},
        {  {{  0,  0,  0},      {  1,  1,  0}},     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}}}, // 8 
        {  {{  3,  6,  8},      {  5,  4,  8}},     {{ 1, 2,  8}},     {{ 3, 2},    { 3, 2}}},

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\tTest vector %u...\n", v);
            
        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS };

        nn_window_op_config_t window_config;
        memset(&window_config, 0, sizeof(window_config));
        window_config.output.start = casse->start.Y;
        window_config.output.shape.height = casse->output.shape.rows;
        window_config.output.shape.width = casse->output.shape.cols;
        window_config.output.shape.channels = casse->output.shape.channels;
        window_config.output.stride.vertical.rows = 1;
        window_config.output.stride.horizontal.cols = 1;

        window_config.window.start = casse->start.X;
        window_config.window.shape.height = casse->window.shape.height;
        window_config.window.shape.width  = casse->window.shape.width;
        window_config.window.outer_stride.vertical.rows = casse->window.stride.vertical;
        window_config.window.outer_stride.horizontal.cols = casse->window.stride.horizontal;
        window_config.window.inner_stride.vertical.rows = 1;
        window_config.window.inner_stride.horizontal.cols = 1;
        
        nn_avgpool2d_plan_t plan;

        avgpool2d_init(&plan, &x_params, &y_params, &window_config);

        memset(Y_exp, 0xCC, sizeof(Y_exp));
        memset(X, 0xAA, sizeof(X));
        PRINTF("\t\tSetting X...\n");
        for(int wh = 0; wh < window_config.output.shape.height; wh++) {
            for(int ww = 0; ww < window_config.output.shape.width; ww++) {
                for(int ch = 0; ch < window_config.output.shape.channels; ch++) {
                    int8_t y_exp = wh + 2 * ww + ch;
                    unsigned yr = window_config.output.start.rows + wh;
                    unsigned yc = window_config.output.start.cols + ww;
                    unsigned yd = window_config.output.start.channels + ch;

                    Y_exp[yr][yc][yd] = y_exp;

                    for(int r = 0; r < window_config.window.shape.height; r++) {
                        for(int c = 0; c < window_config.window.shape.width; c++) {
                            unsigned xr = window_config.window.start.rows     + wh * window_config.window.outer_stride.vertical.rows   + r;
                            unsigned xc = window_config.window.start.cols     + ww * window_config.window.outer_stride.horizontal.cols + c;
                            unsigned xd = window_config.window.start.channels + ch;
                            X[xr][xc][xd] = y_exp;
                        }
                    }
                }
            }
        }

#if (DEBUG_ON || 0)
    PRINTF("plan.window.output.rows                 = %d\n", plan.window.output.rows              );
    PRINTF("plan.window.output.cols                 = %d\n", plan.window.output.cols              );
    PRINTF("plan.window.output.channels             = %d\n", plan.window.output.channels          );
    PRINTF("plan.window.window.rows                 = %d\n", plan.window.window.rows              );
    PRINTF("plan.window.window.cols                 = %d\n\n", plan.window.window.cols              );
    PRINTF("plan.window.start_stride.x              = %d\n", plan.window.start_stride.x           );
    PRINTF("plan.window.inner_stride.horizontal.x   = %d\n", plan.window.inner_stride.horizontal.x);
    PRINTF("plan.window.inner_stride.vertical.x     = %d\n", plan.window.inner_stride.vertical.x  );
    PRINTF("plan.window.outer_stride.horizontal.x   = %d\n", plan.window.outer_stride.horizontal.x);
    PRINTF("plan.window.outer_stride.vertical.x     = %d\n", plan.window.outer_stride.vertical.x  );
    PRINTF("plan.window.chan_grp_stride.x           = %d\n\n", plan.window.chan_grp_stride.x        );
    PRINTF("plan.window.start_stride.y              = %d\n", plan.window.start_stride.y           );
    PRINTF("plan.window.outer_stride.horizontal.y   = %d\n", plan.window.outer_stride.horizontal.y);
    PRINTF("plan.window.outer_stride.vertical.y     = %d\n", plan.window.outer_stride.vertical.y  );
    PRINTF("plan.window.chan_grp_stride.y           = %d\n\n", plan.window.chan_grp_stride.y        );
    PRINTF("plan.window.scale                       = 0x%08X\n", plan.scale);
    PRINTF("plan.shift                              = 0x%08X\n", plan.shift);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(Y_asm));
        avgpool2d_asm((int8_t*)Y_asm, (int8_t*)X, &plan);
#endif


        unsigned hadsomething = 0;
        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < Y_HEIGHT; row++){
            for(unsigned col = 0; col < Y_WIDTH; col++){
                for(unsigned y_chn = 0; y_chn < y_params.channels; y_chn++){
                    
                    int8_t y_exp = Y_exp[row][col][y_chn];
                    if(y_exp != (int8_t)0xCC)
                        hadsomething = 1;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = Y_c[row][col][y_chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = Y_asm[row][col][y_chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, y_chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }
        TEST_ASSERT(hadsomething);
    }

}
#undef CHANS
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT
#undef DEBUG_ON







#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define MAX_CHANS   (4*VPU_INT8_ACC_PERIOD)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
void test_avgpool2d_2x2_case1()
{
    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
#if TEST_C
    int8_t WORD_ALIGNED  Y_c[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];
#endif

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        uint32_t height;    
        uint32_t width;
        uint32_t channels;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               //Chans
        {   2,  2,          16          },  // 0
        {   4,  4,          16          },
        {   6,  6,          16          },
        {   8,  8,          16          },
        {   2,  4,          16          },
        {   4,  2,          16          },  // 5
        {   8,  6,          16          },
        {   2, 16,          16          },
        {  32, 16,          16          },
        {  32, 32,          16          },
        
        {   2,  2,          32          },  // 10
        {   2,  2,          48          },
        {   2,  2,          64          },
        {   2,  2,           4          },
        {   2,  2,           8          },
        {   2,  2,          12          },  // 15
        {   2,  2,          20          },
        {   2,  2,          24          },
        {   2,  2,          28          },
        {   2,  2,          36          },

        {   4,  8,          40          },  // 20
        {  12,  6,          12          },
        {  16,  2,          40          },
        {   4, 24,          36          },
        {  32, 32,          60          },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    memset(X, 120, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\ttest vector %u...\n", v);

#if (DEBUG_ON || 0)
    PRINTF("\t\t(X_height, X_width, X_channels) = (%u, %u, %u)\n", casse->height, casse->width, casse->channels);
#endif
            
        nn_image_params_t x_params = { casse->height, casse->width, casse->channels };
        nn_image_params_t y_params = { casse->height/2, casse->width/2, casse->channels };

        for(int r = 0; r < y_params.height; r++){
            for(int c = 0; c < y_params.width; c++){
                for(int ch = 0; ch < y_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+0, ch)] = (ch&1)? 110 : -110;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+0, ch)] = (ch&1)?  90 : -90;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+0, 2*c+1, ch)] = (ch&1)? 120 : -120;
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, 2*r+1, 2*c+1, ch)] = (ch&1)?  80 : -80;
                }
            }
        }

        nn_image_vect_t start_pos = {0};

        nn_avgpool2d_plan_t plan;
        avgpool2d_2x2_init(&plan, &x_params, &y_params, 
                            &start_pos, &start_pos, 
                            y_params.height, y_params.width, y_params.channels);


#if (DEBUG_ON || 0)
    PRINTF("plan.window.output.rows                 = %d\n",        plan.window.output.rows              );
    PRINTF("plan.window.output.cols                 = %d\n",        plan.window.output.cols              );
    PRINTF("plan.window.output.channels             = %d\n",        plan.window.output.channels          );
    PRINTF("plan.window.window.rows                 = %d\n",        plan.window.window.rows              );
    PRINTF("plan.window.window.cols                 = %d\n\n",      plan.window.window.cols              );
    PRINTF("plan.window.start_stride.x              = %d\n",        plan.window.start_stride.x           );
    PRINTF("plan.window.inner_stride.horizontal.x   = %d\n",        plan.window.inner_stride.horizontal.x);
    PRINTF("plan.window.inner_stride.vertical.x     = %d\n",        plan.window.inner_stride.vertical.x  );
    PRINTF("plan.window.outer_stride.horizontal.x   = %d\n",        plan.window.outer_stride.horizontal.x);
    PRINTF("plan.window.outer_stride.vertical.x     = %d\n",        plan.window.outer_stride.vertical.x  );
    PRINTF("plan.window.chan_grp_stride.x           = %d\n\n",      plan.window.chan_grp_stride.x        );
    PRINTF("plan.window.start_stride.y              = %d\n",        plan.window.start_stride.y           );
    PRINTF("plan.window.outer_stride.horizontal.y   = %d\n",        plan.window.outer_stride.horizontal.y);
    PRINTF("plan.window.outer_stride.vertical.y     = %d\n",        plan.window.outer_stride.vertical.y  );
    PRINTF("plan.window.chan_grp_stride.y           = %d\n\n",      plan.window.chan_grp_stride.y        );
    PRINTF("plan.window.scale                       = 0x%08X\n",    plan.scale);
    PRINTF("plan.shift                              = 0x%08X\n",    plan.shift);
#endif //DEBUG_ON



#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, casse->height * casse->width * casse->channels / 4);
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  casse->height * casse->width * casse->channels / 4);
        avgpool2d_2x2_asm((int8_t*)Y_asm, (int8_t*)X, &plan);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){

                int32_t y_base = IMG_ADDRESS_VECT(&y_params, row, col, 0);

                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = (chn&1)? 100 : -100;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = ((int8_t*)Y_c)[y_base + chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = ((int8_t*)Y_asm)[y_base + chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM        
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }

    }

}
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS
#undef DEBUG_ON





#define DEBUG_ON    (0 || TEST_DEBUG_ON)
#define CHANS       (2*VPU_INT8_ACC_PERIOD)
#define X_HEIGHT    (12)
#define X_WIDTH     (12)
#define Y_HEIGHT    (6)
#define Y_WIDTH     (6)
void test_avgpool2d_2x2_case2()
{
    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y_exp[Y_HEIGHT][Y_WIDTH][CHANS] = {{{0}}};
#if TEST_C
    int8_t WORD_ALIGNED  Y_c[Y_HEIGHT][Y_WIDTH][CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[Y_HEIGHT][Y_WIDTH][CHANS];
#endif

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {        nn_image_vect_t X;  nn_image_vect_t Y;  } start;
        struct {        nn_image_vect_t shape;                  } output;
    } test_case_t;

    const test_case_t casses[] = {
        //  X                   Y                   Out Shape    
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 6, 6, 32}}},
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 3, 3, 32}}},
        {  {{  0,  0,  0},      {  3,  3,  0}},     {{ 3, 3, 32}}},
        {  {{  0,  0,  0},      {  0,  0,  0}},     {{ 6, 6, 16}}},
        {  {{  0,  0, 16},      {  0,  0,  0}},     {{ 6, 6, 16}}},
        {  {{  0,  0,  0},      {  0,  0, 16}},     {{ 6, 6, 16}}},
        {  {{  0,  0, 16},      {  0,  0, 16}},     {{ 6, 6, 16}}},
        {  {{  2,  2,  0},      {  0,  0,  0}},     {{ 5, 5, 32}}},
        {  {{  0,  0,  0},      {  1,  1,  0}},     {{ 5, 5, 32}}},
        {  {{  3,  6,  8},      {  5,  4,  8}},     {{ 1, 2,  8}}},

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\tTest vector %u...\n", v);
            
        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS };
        nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS };
        
        nn_avgpool2d_plan_t plan;

        avgpool2d_2x2_init(&plan, &x_params, &y_params, &casse->start.X, &casse->start.Y, 
            casse->output.shape.rows, casse->output.shape.cols, casse->output.shape.channels);

        memset(Y_exp, 0xCC, sizeof(Y_exp));
        memset(X, 0xAA, sizeof(X));
        PRINTF("\t\tSetting X...\n");
        for(int wh = 0; wh < casse->output.shape.rows; wh++) {
            for(int ww = 0; ww < casse->output.shape.cols; ww++) {
                for(int ch = 0; ch < casse->output.shape.channels; ch++) {
                    int8_t y_exp = wh + 2 * ww + ch;
                    unsigned yr = casse->start.Y.rows + wh;
                    unsigned yc = casse->start.Y.cols + ww;
                    unsigned yd = casse->start.Y.channels + ch;

                    Y_exp[yr][yc][yd] = y_exp;

                    for(int r = 0; r < 2; r++) {
                        for(int c = 0; c < 2; c++) {
                            unsigned xr = casse->start.X.rows     + wh * 2 + r;
                            unsigned xc = casse->start.X.cols     + ww * 2 + c;
                            unsigned xd = casse->start.X.channels + ch;
                            X[xr][xc][xd] = y_exp;
                        }
                    }
                }
            }
        }

#if (DEBUG_ON || 0)
    PRINTF("plan.window.output.rows                 = %d\n", plan.window.output.rows              );
    PRINTF("plan.window.output.cols                 = %d\n", plan.window.output.cols              );
    PRINTF("plan.window.output.channels             = %d\n", plan.window.output.channels          );
    PRINTF("plan.window.window.rows                 = %d\n", plan.window.window.rows              );
    PRINTF("plan.window.window.cols                 = %d\n\n", plan.window.window.cols              );
    PRINTF("plan.window.start_stride.x              = %d\n", plan.window.start_stride.x           );
    PRINTF("plan.window.inner_stride.horizontal.x   = %d\n", plan.window.inner_stride.horizontal.x);
    PRINTF("plan.window.inner_stride.vertical.x     = %d\n", plan.window.inner_stride.vertical.x  );
    PRINTF("plan.window.outer_stride.horizontal.x   = %d\n", plan.window.outer_stride.horizontal.x);
    PRINTF("plan.window.outer_stride.vertical.x     = %d\n", plan.window.outer_stride.vertical.x  );
    PRINTF("plan.window.chan_grp_stride.x           = %d\n\n", plan.window.chan_grp_stride.x        );
    PRINTF("plan.window.start_stride.y              = %d\n", plan.window.start_stride.y           );
    PRINTF("plan.window.outer_stride.horizontal.y   = %d\n", plan.window.outer_stride.horizontal.y);
    PRINTF("plan.window.outer_stride.vertical.y     = %d\n", plan.window.outer_stride.vertical.y  );
    PRINTF("plan.window.chan_grp_stride.y           = %d\n\n", plan.window.chan_grp_stride.y        );
    PRINTF("plan.window.scale                       = 0x%08X\n", plan.scale);
    PRINTF("plan.shift                              = 0x%08X\n", plan.shift);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(Y_asm));
        avgpool2d_2x2_asm((int8_t*)Y_asm, (int8_t*)X, &plan);
#endif


        unsigned hadsomething = 0;
        char str_buff[200] = {0};
        PRINTF("\t\tChecking...\n");
        for(unsigned row = 0; row < Y_HEIGHT; row++){
            for(unsigned col = 0; col < Y_WIDTH; col++){
                for(unsigned y_chn = 0; y_chn < y_params.channels; y_chn++){
                    
                    int8_t y_exp = Y_exp[row][col][y_chn];
                    if(y_exp != (int8_t) 0xCC)
                        hadsomething = 1;

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = Y_c[row][col][y_chn];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = Y_asm[row][col][y_chn];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, y_chn);
                    }

#if TEST_C
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_c, str_buff);
#endif
#if TEST_ASM
                    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y_asm, str_buff);
#endif
                }
            }
        }
        TEST_ASSERT(hadsomething);
    }

}
#undef CHANS
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT
#undef DEBUG_ON

void test_avgpool2d()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_avgpool2d_case1);
    RUN_TEST(test_avgpool2d_case2);
    RUN_TEST(test_avgpool2d_case3);
    RUN_TEST(test_avgpool2d_2x2_case1);
    RUN_TEST(test_avgpool2d_2x2_case2);
}