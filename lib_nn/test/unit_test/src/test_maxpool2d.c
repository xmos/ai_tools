
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#define GET_IMAGE_VAL(IMG, PRMS, ROW, COL, CHAN)         (((nn_image_t*)IMG)[ ROW*(PRMS.width * PRMS.channels) + COL*(PRMS.channels) + CHAN ])
#define SET_IMAGE_VAL(IMG, PRMS, ROW, COL, CHAN, VAL)    ((nn_image_t*)IMG)[ ROW*(PRMS.width * PRMS.channels) + COL*(PRMS.channels) + CHAN ] = VAL

static void Check_Y(
    const nn_image_t y_exp,
    const nn_image_t* Y,
    const nn_image_params_t* y_params,
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line)
{
    char str_buff[200];

    nn_image_t y = Y[IMG_ADDRESS_VECT(y_params, row, col, chn)];

    if(y != y_exp){
        sprintf(str_buff, "Y[%u][%u][%u] was wrong [line %u]", 
                row, col, chn, line);
    }

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}



#define MAX_CHANS   (2*VPU_INT8_EPV)
#define MAX_HEIGHT  (32)
#define MAX_WIDTH   (32)
#define REPS        (3)
void test_maxpool2d_case0()
{
    srand(66535);

    int8_t WORD_ALIGNED  X[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[MAX_HEIGHT][MAX_WIDTH][MAX_CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        struct {    uint32_t height;    uint32_t width;     } X;
        struct {    uint32_t height;    uint32_t width;     } Y;
        struct {    uint32_t height;    uint32_t width;
                    int32_t vstride;    int32_t hstride;    } W;
        unsigned channels;
        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        //  X               // Y            // W                      //Chans
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        32            , __LINE__ },  //0
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        16            , __LINE__ },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         8            , __LINE__ },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        48            , __LINE__ },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},         4            , __LINE__ },
        {   { 1,  1},       { 1,  1},       {  1,  1,  1,  1},        MAX_CHANS     , __LINE__ },
        
        {   { 1,  2},       { 1,  1},       {  1,  2,  1,  1},        32            , __LINE__ },  //6
        {   { 2,  1},       { 1,  1},       {  2,  1,  1,  1},        32            , __LINE__ },
        {   { 2,  2},       { 1,  1},       {  2,  2,  1,  1},        32            , __LINE__ },
        {   { 1,  4},       { 1,  1},       {  1,  4,  1,  1},        32            , __LINE__ },
        {   { 4,  1},       { 1,  1},       {  4,  1,  1,  1},        32            , __LINE__ },
        {   { 4,  4},       { 1,  1},       {  4,  4,  1,  1},        32            , __LINE__ },

        {   { 1,  3},       { 1,  1},       {  1,  3,  1,  1},        32            , __LINE__ },  //12
        {   { 3,  1},       { 1,  1},       {  3,  1,  1,  1},        32            , __LINE__ },
        {   { 3,  3},       { 1,  1},       {  3,  3,  1,  1},        32            , __LINE__ },
        {   { 5,  3},       { 1,  1},       {  5,  3,  1,  1},        32            , __LINE__ },  
        {   { 9,  1},       { 1,  1},       {  9,  1,  1,  1},        32            , __LINE__ },
        {   { 3, 13},       { 1,  1},       {  3, 13,  1,  1},        32            , __LINE__ },

        {   { 1,  2},       { 1,  2},       {  1,  1,  1,  1},        32            , __LINE__ },  //18
        {   { 2,  1},       { 2,  1},       {  1,  1,  1,  1},        32            , __LINE__ },
        {   { 2,  2},       { 2,  2},       {  1,  1,  1,  1},        32            , __LINE__ },
        {   { 1,  3},       { 1,  3},       {  1,  1,  1,  1},        32            , __LINE__ },
        {   { 3,  3},       { 3,  3},       {  1,  1,  1,  1},        32            , __LINE__ },
        {   { 4,  1},       { 4,  1},       {  1,  1,  1,  1},        32            , __LINE__ },
        {   { 5,  7},       { 5,  7},       {  1,  1,  1,  1},        32            , __LINE__ },

        {   { 1,  1},       { 1,  1},       {  1,  1,  2,  2},        32            , __LINE__ },  //25
        {   { 4,  2},       { 2,  2},       {  1,  1,  2,  1},        32            , __LINE__ },
        {   { 2,  4},       { 2,  2},       {  1,  1,  1,  2},        32            , __LINE__ },
        {   { 4,  4},       { 2,  2},       {  1,  1,  2,  2},        32            , __LINE__ },
        {   { 9,  9},       { 3,  3},       {  1,  1,  3,  3},        32            , __LINE__ },
        
        {   { 4,  4},       { 2,  2},       {  2,  2,  2,  2},        32            , __LINE__ },  //30
        {   { 4,  4},       { 3,  3},       {  2,  2,  1,  1},        32            , __LINE__ },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32            , __LINE__ },
        {   { 9,  9},       { 3,  3},       {  3,  3,  3,  3},        32            , __LINE__ },
        {   {16, 16},       { 4,  4},       {  4,  4,  4,  4},        MAX_CHANS     , __LINE__ },
        {   {25, 25},       { 5,  5},       {  5,  5,  5,  5},         8            , __LINE__ },
        {   {32, 32},       { 4,  8},       {  8,  4,  8,  4},        24            , __LINE__ },

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];
 
        PRINTF("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { casse->X.height, casse->X.width, casse->channels };
        nn_image_params_t y_params = { casse->Y.height, casse->Y.width, casse->channels };

        nn_window_params_t window_params;

        window_params.shape.height = casse->W.height;
        window_params.shape.width  = casse->W.width;

        window_params.start.row = 0;
        window_params.start.column = 0;

        window_params.stride.vertical = casse->W.vstride;
        window_params.stride.horizontal = casse->W.hstride;


        const unsigned x_bytes = x_params.height * x_params.width * x_params.channels;
        const unsigned y_bytes = y_params.height * y_params.width * y_params.channels;

        pseudo_rand_bytes((char*)X, x_bytes);

        for(unsigned rep = 0; rep < REPS; rep++){
            PRINTF("\t\tRep %d...\n", rep);

            memset(Y, 0xCC, y_bytes);

            maxpool2d((int8_t*)Y, (int8_t*)X, &x_params, &y_params, &window_params);

            char str_buff[200] = {0};
            for(unsigned row = 0; row < y_params.height; row++){
                for(unsigned col = 0; col < y_params.width; col++){
                    for(unsigned chn = 0; chn < y_params.channels; chn++){
                        
                        int8_t mx = -128;

                        for(int wr = 0; wr < window_params.shape.height; wr++){
                            for(int wc = 0; wc < window_params.shape.width; wc++){
                                int32_t x_offset = IMG_ADDRESS_VECT(&x_params, 
                                                                    (casse->W.vstride * row + wr), 
                                                                    (casse->W.hstride * col + wc), chn);
                                int8_t x_val = ((int8_t*)X)[x_offset];
                                mx = (x_val > mx)? x_val : mx;
                            }   
                        }

                        int8_t y_exp = mx;

                        Check_Y(y_exp, (nn_image_t*)Y, &y_params, row, col, chn, casse->line);
                    }
                }
            }
        }

    }

}
#undef REPS
#undef MAX_WIDTH
#undef MAX_HEIGHT
#undef MAX_CHANS
#undef DEBUG_ON











#define CHANS        (2*VPU_INT8_EPV)
#define X_HEIGHT     (12)
#define X_WIDTH      (12)
#define Y_HEIGHT     (6)
#define Y_WIDTH      (6)
#define REPS         (4)
void test_maxpool2d_case1()
{
    srand(3523466);

    int8_t WORD_ALIGNED  X[X_HEIGHT][X_WIDTH][CHANS] = {{{0}}};
    int8_t WORD_ALIGNED  Y_exp[Y_HEIGHT][Y_WIDTH][CHANS] = {{{0}}};
    
    int8_t WORD_ALIGNED  Y[Y_HEIGHT][Y_WIDTH][CHANS];

    PRINTF("%s...\n", __func__);

    
    typedef struct {
        // struct {        nn_image_vect_t X;  nn_image_vect_t Y;  } start;   //choose start locations randomly
        struct {        nn_image_vect_t shape;                  } output;
        struct {
            struct {    uint32_t height;    uint32_t width;     } shape;
            struct {    int32_t vertical;   int32_t horizontal; } stride;
        } window;

        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        //    Out Shape          Win shape    stride               
        {     {{ 6, 6, 32}},     {{ 2, 2},    { 2, 2}},     __LINE__},  // 0
        {     {{ 3, 3, 32}},     {{ 4, 4},    { 4, 4}},     __LINE__},
        {     {{ 3, 3, 32}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 6, 6, 16}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}},     __LINE__},
        {     {{ 5, 5, 32}},     {{ 2, 2},    { 2, 2}},     __LINE__}, // 8 
        {     {{ 1, 2,  8}},     {{ 3, 2},    { 3, 2}},     __LINE__},

    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];

        PRINTF("\tTest vector %u...\n", v);

        nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, casse->output.shape.channels };
        nn_image_params_t y_params = { casse->output.shape.rows, casse->output.shape.cols, casse->output.shape.channels };

        nn_window_params_t window_params;
        nn_window_op_job_params_t job_params;

        window_params.shape.height = casse->window.shape.height;
        window_params.shape.width  = casse->window.shape.width;
    
        window_params.stride.vertical = casse->window.stride.vertical;
        window_params.stride.horizontal = casse->window.stride.horizontal;

        window_params.start.row = 0;
        window_params.start.column = 0;

        for(unsigned rep = 0; rep < REPS; rep++){

            PRINTF("\t\tRep %d...\n", rep);

            // (minimum rows, columns, channels is 1, 1, 4)
            job_params.size.rows     = y_params.height == 1? 1 : (pseudo_rand_uint16() % (y_params.height - 1)) + 1;
            job_params.size.cols     = y_params.width  == 1? 1 : (pseudo_rand_uint16() % (y_params.width  - 1)) + 1; 
            job_params.size.channels = 4*((pseudo_rand_uint16() % (y_params.channels/4 - 1)) + 1);

            // Start indices must be low enough that the job doesn't go outside the image
            const uint32_t max_row  = y_params.height   - job_params.size.rows + 1;
            const uint32_t max_col  = y_params.width    - job_params.size.cols + 1;
            const uint32_t max_chan = y_params.channels - job_params.size.channels + 4;

            job_params.start.rows = pseudo_rand_uint16() % max_row;
            job_params.start.cols = pseudo_rand_uint16() % max_col;
            job_params.start.channels = 4*(pseudo_rand_uint16() % (max_chan/4));
            
            memset(Y_exp, 0xCC, sizeof(Y_exp));
            memset(Y, 0xCC, sizeof(Y));
            memset(X, 0xAA, sizeof(X));
            PRINTF("\t\t\tSetting X...\n");

            for(int out_row = 0; out_row < job_params.size.rows; out_row++) {
                for(int out_col = 0; out_col < job_params.size.cols; out_col++) {
                    for(int out_chan = 0; out_chan < job_params.size.channels; out_chan++) {
                        int8_t y_exp = out_row + 2 * out_col + out_chan;
                        unsigned yr = job_params.start.rows + out_row;
                        unsigned yc = job_params.start.cols + out_col;
                        unsigned yd = job_params.start.channels + out_chan;

                        SET_IMAGE_VAL(Y_exp, y_params, yr, yc, yd, y_exp);
                        // Y_exp[yr][yc][yd] = y_exp;

                        struct { unsigned row, col; } win_start = {
                            job_params.start.rows * window_params.shape.height,
                            job_params.start.cols * window_params.shape.width };

                        // The X row is the window start row, plus the output row index (NOT Y row index) times
                        // the vertical stride of the window, plus the window row for that output.
                        for(int r = 0; r < window_params.shape.height; r++) {
                            for(int c = 0; c < window_params.shape.width; c++) {
                                struct { unsigned row, col, chn; } xdex = {
                                    win_start.row + (out_row * window_params.stride.vertical) + r,
                                    win_start.col + (out_col * window_params.stride.horizontal) + c,
                                    job_params.start.channels + out_chan };
                                
                                SET_IMAGE_VAL(X, x_params, xdex.row, xdex.col, xdex.chn, y_exp);
                            }
                        }
                    }
                }
            }

            PRINTF("\t\t\tRunning...\n");

            maxpool2d_ext((int8_t*)Y, (int8_t*)X, &x_params, &y_params, &window_params, 
                            &job_params, MAXPOOL2D_FLAG_NONE);


            char str_buff[200] = {0};
            PRINTF("\t\t\tChecking...\n");
            for(unsigned row = 0; row < y_params.height; row++){
                for(unsigned col = 0; col < y_params.width; col++){
                    for(unsigned y_chn = 0; y_chn < y_params.channels; y_chn++){
                        
                        // int8_t y_exp = Y_exp[row][col][y_chn];
                        int8_t y_exp = GET_IMAGE_VAL(Y_exp, y_params, row, col, y_chn);

                        Check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, y_chn, casse->line);
                    }
                }
            }
        }
    }

}
#undef REPS
#undef CHANS
#undef Y_WIDTH
#undef Y_HEIGHT
#undef X_WIDTH
#undef X_HEIGHT

void test_maxpool2d()
{

    UNITY_SET_FILE();
    
    RUN_TEST(test_maxpool2d_case0);
    RUN_TEST(test_maxpool2d_case1);
}