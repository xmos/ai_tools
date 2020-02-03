
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_avgpool2d)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)


static unsigned seed = 4321234;


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

    PRINTF("test_avgpool2d_case1()...\n");

    
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

        printf("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->X.height, casse->X.width, casse->channels };
        nn_image_params_t y_params = { casse->Y.height, casse->Y.width, casse->channels };

        nn_window_map_t win_map = NN_WINDOW_MAP_DEFAULT();
        win_map.window.height   = casse->W.height;
        win_map.window.width    = casse->W.width;
        
        win_map.window.vstride  = casse->W.vstride;
        win_map.window.hstride  = casse->W.hstride;

        win_map.window.vcount   = casse->Y.height;
        win_map.window.hcount   = casse->Y.width;

        nn_avgpool_params_t params;

        avgpool2d_init(&params, &x_params, &y_params, &win_map);

        for(int r = 0; r < x_params.height; r++){
            for(int c = 0; c < x_params.width; c++){
                for(int ch = 0; ch < x_params.channels; ch++){
                    ((int8_t*)X)[IMG_ADDRESS_VECT(&x_params, r, c, ch)] = (ch&1)? 120 : -120;
                }
            }
        }

#if DEBUG_ON
    PRINTF("params.Y_h             = %d\n",     params.out_rows);
    PRINTF("params.Y_w             = %d\n",     params.out_cols);   
    PRINTF("params.Y_chans         = %d\n",     params.out_chans);
    PRINTF("params.W_h             = %d\n",     params.W_h);
    PRINTF("params.W_w             = %d\n",     params.W_w);
    PRINTF("params.scale           = 0x%08X\n", params.scale);
    PRINTF("params.shift           = 0x%08X\n", params.shift);
    PRINTF("params.hstride_incr_x  = %d\n",     params.hstride_incr_x);
    PRINTF("params.vstride_incr_x  = %d\n",     params.vstride_incr_x);
    PRINTF("params.vstride_incr_y  = %d\n",     params.vstride_incr_y);
    PRINTF("params.chan_incr_x     = %d\n",     params.chan_incr_x);
    PRINTF("params.win_col_incr_x  = %d\n",     params.win_col_incr_x);
    PRINTF("params.win_row_incr_x  = %d\n",     params.win_row_incr_x);
    PRINTF("params.start_incr_x    = %d\n",     params.start_incr_x);
    PRINTF("params.start_incr_y    = %d\n",     params.start_incr_y);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\tC...\n");
        memset(Y_c, 0xCC, casse->Y.height * casse->Y.width * casse->channels);    //too expensive to write the whole image, so just do the part that's in play
        avgpool2d_c((int8_t*)Y_c, (int8_t*)X, &params);
#endif
#if TEST_ASM
        PRINTF("\t\tASM...\n");
        memset(Y_asm, 0xCC,  casse->Y.height * casse->Y.width * casse->channels);
        avgpool2d_asm((int8_t*)Y_asm, (int8_t*)X, &params);
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
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED Y_exp[HEIGHT][WIDTH][CHANS];

#if TEST_C
    int8_t WORD_ALIGNED  Y_c[HEIGHT][WIDTH][CHANS];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[HEIGHT][WIDTH][CHANS];
#endif

    PRINTF("test_avgpool2d_case2()...\n");

    
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
        {       {   2,  2 },        {   4,  4 },        {   0,  0 }         },
        {       {   2,  2 },        {   0,  0 },        {   8,  8 }         }, // 15
        {       {   3,  3 },        {   5,  1 },        {   8,  9 }         },
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        printf("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, CHANS };
        nn_image_params_t y_params = { HEIGHT, WIDTH, CHANS };

        nn_window_map_t win_map = NN_WINDOW_MAP_DEFAULT();
        win_map.window.height   = casse->window.height;
        win_map.window.width    = casse->window.width;
        
        win_map.window.vstride  = casse->window.height;
        win_map.window.hstride  = casse->window.width;

        
        win_map.window.vcount   = (HEIGHT - casse->x_start.row) / casse->window.height;
        win_map.window.hcount   = (WIDTH  - casse->x_start.col) / casse->window.width;
        if((HEIGHT - casse->y_start.row) < win_map.window.vcount) 
            win_map.window.vcount = (HEIGHT - casse->y_start.row);
        if((WIDTH  - casse->y_start.col) < win_map.window.hcount) 
            win_map.window.hcount = (WIDTH  - casse->y_start.col);

        win_map.start.X.rows = casse->x_start.row;
        win_map.start.X.cols = casse->x_start.col;
        win_map.start.X.channels = 0;

        win_map.start.Y.rows = casse->y_start.row;
        win_map.start.Y.cols = casse->y_start.col;
        win_map.start.Y.channels = 0;

        nn_avgpool_params_t params;

        avgpool2d_init(&params, &x_params, &y_params, &win_map);

        PRINTF("\t\tSetting X...\n");
        memset(Y_exp, 0xCC, sizeof(Y_exp));
        for(int vpos = 0; vpos < win_map.window.vcount; vpos++){
            for(int hpos = 0; hpos < win_map.window.hcount; hpos++){

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
    PRINTF("params.Y_h             = %d\n",     params.out_rows);
    PRINTF("params.Y_w             = %d\n",     params.out_cols);   
    PRINTF("params.Y_chans         = %d\n",     params.out_chans);
    PRINTF("params.W_h             = %d\n",     params.W_h);
    PRINTF("params.W_w             = %d\n",     params.W_w);
    PRINTF("params.scale           = 0x%08X\n", params.scale);
    PRINTF("params.shift           = 0x%08X\n", params.shift);
    PRINTF("params.hstride_incr_x  = %d\n",     params.hstride_incr_x);
    PRINTF("params.vstride_incr_x  = %d\n",     params.vstride_incr_x);
    PRINTF("params.vstride_incr_y  = %d\n",     params.vstride_incr_y);
    PRINTF("params.chan_incr_x     = %d\n",     params.chan_incr_x);
    PRINTF("params.win_col_incr_x  = %d\n",     params.win_col_incr_x);
    PRINTF("params.win_row_incr_x  = %d\n",     params.win_row_incr_x);
    PRINTF("params.start_incr_x    = %d\n",     params.start_incr_x);
    PRINTF("params.start_incr_y    = %d\n",     params.start_incr_y);
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

