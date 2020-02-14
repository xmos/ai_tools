
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "xs3_vpu.h"

#include "Unity.h"

#ifdef __XC__
#define WORD_ALIGNED [[aligned(4)]]
#else
#define WORD_ALIGNED
#endif

#if (defined(__XS3A__) && USE_ASM_maxpool2d)
 #define HAS_ASM (1)
#else
 #define HAS_ASM (0)
#endif

#define TEST_ASM ((HAS_ASM)     && 1)
#define TEST_C ((TEST_C_GLOBAL) && 1)

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 1)

#define PRINTF(...)     do{if (DO_PRINT_EXTRA) {printf(__VA_ARGS__);}} while(0)


static unsigned seed = 4321434;


#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define CHANS_IN        (VPU_INT8_EPV)
#define CHANS_OUT       (VPU_INT8_ACC_PERIOD)
#define HEIGHT          (2)
#define WIDTH           (2)
#define CHANS_OUT_CEIL  ((CHANS_OUT + (VPU_INT8_ACC_PERIOD-1)) & (0xFFFFFFFF << VPU_INT8_ACC_PERIOD_LOG2))
void test_conv2d_1x1_case0()
{
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][CHANS_IN];

    int8_t WORD_ALIGNED  K[CHANS_OUT][CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT_CEIL];
        int16_t shift1[CHANS_OUT_CEIL];
        int16_t scale[CHANS_OUT_CEIL];
        int16_t shift2[CHANS_OUT_CEIL];
    } BSS;


#if TEST_C
    int8_t WORD_ALIGNED  Y_c[HEIGHT][WIDTH][CHANS_OUT];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[HEIGHT][WIDTH][CHANS_OUT];
#endif

    PRINTF("test_conv2d_1x1_case0()...\n");

    
    typedef struct {
        int8_t x;
        int8_t k;
        int32_t bias;
        int16_t shift1;
        int16_t scale;
        int16_t shift2;

        int8_t y;
        unsigned line;
    } test_case_t;


    //   Y[i] = C_in * (x * k)

    const test_case_t casses[] = {
        //  X       K           bias            shift1      scale       shift2      Y
        {   0x00,   0x00,       0x00000000,     0,          0x0000,     0,          0x00,       __LINE__}, 
        {   0x00,   0x00,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__}, 
        {   0x00,   0x00,       0x00100000,     0,          0x0000,     0,          0x00,       __LINE__}, 
        {   0xCC,   0x00,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__},
        {   0xCC,   0xCC,       0x00000000,     0,          0x0000,     0,          0x00,       __LINE__},
        {   0x00,   0xCC,       0x00000000,     0,          0x0001,     0,          0x00,       __LINE__},
        {   0xAA,   0xBB,       0x0ABCDABC,     0,          0x0000,     0,          0x00,       __LINE__},

        {   0x00,   0x00,       0x00000001,     0,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000011,     0,          0x0001,     0,          0x11,       __LINE__}, 
        {   0x00,   0x00,       0x0000007F,     0,          0x0001,     0,          0x7F,       __LINE__}, 
        {   0x00,   0x00,      -0x0000007F,     0,          0x0001,     0,         -0x7F,       __LINE__}, 
        {   0x00,   0x00,       0x00000080,     0,          0x0001,     0,          0x7F,       __LINE__}, 
        {   0x00,   0x00,      -0x00000080,     0,          0x0001,     0,         -0x7F,       __LINE__}, 
        {   0x00,   0x00,       0x0FFFFFFF,     0,          0x0001,     0,          0x7F,       __LINE__}, 
        {   0x00,   0x00,      -0x0FFFFFFF,     0,          0x0001,     0,         -0x7F,       __LINE__}, 

        {   0x00,   0x00,       0x00000002,     0,          0x0001,     1,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000004,     0,          0x0001,     1,          0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000004,     0,          0x0001,     2,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,          0x0001,     1,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000003,     0,          0x0001,     1,          0x02,       __LINE__}, 
        {   0x00,   0x00,      -0x00000002,     0,          0x0001,     1,         -0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000002,     0,          0x0001,     1,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000100,     0,          0x0001,     7,          0x02,       __LINE__},

        {   0x00,   0x00,       0x00000001,     0,          0x0002,     0,          0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,          0x0003,     0,          0x03,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     0,         -0x0002,     0,         -0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000010,     0,          0x0002,     0,          0x20,       __LINE__}, 
        {   0x00,   0x00,       0x00000010,     0,          0x0002,     0,          0x20,       __LINE__}, 
        {   0x00,   0x00,       0x00004000,     0,          0x0100,    21,          0x02,       __LINE__},
        {   0x00,   0x00,       0x00004000,     0,         -0x0100,    21,         -0x02,       __LINE__},
        {   0x00,   0x00,       0x00004000,     0,          0x0100,    18,          0x10,       __LINE__},

        {   0x00,   0x00,       0x00000002,     1,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000004,     1,          0x0001,     0,          0x02,       __LINE__}, 
        {   0x00,   0x00,       0x00000004,     2,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000001,     1,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000003,     1,          0x0001,     0,          0x02,       __LINE__}, 
        {   0x00,   0x00,      -0x00000002,     1,          0x0001,     0,         -0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000002,     1,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x00,   0x00,       0x00000100,     7,          0x0001,     0,          0x02,       __LINE__},
        {   0x00,   0x00,       0x00010000,     0,          0x0001,    10,          0x20,       __LINE__},
        {   0x00,   0x00,       0x00010000,    10,          0x0001,     0,          0x40,       __LINE__},
        {   0x00,   0x00,       0x00010000,     0,          0x0002,    10,          0x40,       __LINE__},
        {   0x00,   0x00,       0x00010000,     4,          0x0001,     6,          0x40,       __LINE__},
        {   0x00,   0x00,      -0x00010000,     4,          0x0001,     6,         -0x40,       __LINE__},
        {   0x00,   0x00,       0x00010000,     4,          0x0004,     8,          0x40,       __LINE__},

        {   0x01,   0x01,       0x00000000,     0,          0x0001,     0,          0x20,       __LINE__}, 
        {   0x01,   0x02,       0x00000000,     0,          0x0001,     0,          0x40,       __LINE__}, 
        {   0x02,   0x01,       0x00000000,     0,          0x0001,     0,          0x40,       __LINE__}, 
        {   0x01,  -0x01,       0x00000000,     0,          0x0001,     0,         -0x20,       __LINE__}, 
        {  -0x01,   0x01,       0x00000000,     0,          0x0001,     0,         -0x20,       __LINE__}, 
        {  -0x01,  -0x01,       0x00000000,     0,          0x0001,     0,          0x20,       __LINE__}, 
        {   0x01,   0x01,      -0x00000010,     0,          0x0001,     0,          0x10,       __LINE__}, 
        {   0x02,   0x02,       0x00000000,     0,          0x0001,     1,          0x40,       __LINE__}, 
        {   0x10,   0x02,       0x00000000,     0,          0x0001,     5,          0x20,       __LINE__}, 
        {   0x10,   0x10,      -0x00002020,     0,          0x0001,     0,         -0x20,       __LINE__}, 
        {   0x10,   0x02,       0x00000000,     0,          0x0010,     8,          0x40,       __LINE__}, 

        {   0x40,   0x40,       0x00000000,    16,          0x0001,     0,          0x02,       __LINE__}, 
        {   0x40,   0x40,       0x00000000,    17,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x40,   0x40,       0x00000000,    18,          0x0001,     0,          0x01,       __LINE__}, 
        {   0x40,   0x40,       0x00000000,    19,          0x0001,     0,          0x00,       __LINE__}, 
        {   0x40,  -0x40,       0x00000000,    16,          0x0001,     0,         -0x02,       __LINE__}, 
        {  -0x40,   0x40,       0x00000000,    17,          0x0001,     0,         -0x01,       __LINE__}, 
        {  -0x40,  -0x40,       0x00000000,    18,         -0x0001,     0,         -0x01,       __LINE__}, 
        {   0x40,   0x40,       0x00000000,    19,         -0x0001,     0,          0x00,       __LINE__}, 

        //SPECIAL CASES  
        //  The case logic will look for this case (by x==0xED) and handle it differently.
        //  Point of the special case is to not have all X and all K be identical.
        //  That way it can make sure that, for example, it's not just always using the same input data

        // X[i][j][k] = k,   K[u][v] = 1        ->   Y[i][j][k] = (0 + 1 + ... + 31) >> 4 =  (2^4 * 31) >> 4 = 31
        {   0xED,   0x00,       0x00000000,     0,          0x0001,     4,          0x00,       __LINE__},
        // X[i][j][k] = 1,   K[u][v] = u        ->   Y[i][j][k] = 32 * k >> 4  = 2*k
        {   0xED,   0x01,       0x00000000,     0,          0x0001,     4,          0x00,       __LINE__},
        // X[i][j][k] = 1,   K[u][v] = v        ->   Y[i][j][k] =  (0 + 1 + ... + 31) >> 3 =  (2^4 * 31) >> 3 = 62
        {   0xED,   0x02,       0x00000000,     0,          0x0001,     3,          0x00,       __LINE__},
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;

    print_warns(start_case, TEST_C, TEST_ASM);

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){

        const test_case_t* casse = (const test_case_t*) &casses[v];

        printf("\ttest vector %u...\n", v);
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, CHANS_IN };
        nn_image_params_t y_params = { HEIGHT, WIDTH, CHANS_OUT };

        if(((uint8_t)casse->x) != 0xED){
            memset(X, casse->x, sizeof(X));
            memset(K, casse->k, sizeof(K));
        } else {
            switch(casse->k){
                case 0:
                    memset(K, 1, sizeof(K));
                    for(int r = 0; r < HEIGHT; r++)
                        for(int c = 0; c < WIDTH; c++)
                            for(int i = 0; i < CHANS_IN; i++)
                                X[r][c][i] = i;
                    break;
                case 1:
                    memset(X, 1, sizeof(X));
                    for(int i = 0; i < CHANS_OUT; i++)
                        for(int j = 0; j < CHANS_IN; j++)
                            K[i][j] = i;
                    break;
                case 2:
                    memset(X, 1, sizeof(X));
                    for(int i = 0; i < CHANS_OUT; i++)
                        for(int j = 0; j < CHANS_IN; j++)
                            K[i][j] = j;
                    break;
                default:
                    TEST_FAIL();
                    break;
            }
        }   

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k]     = casse->bias;
            BSS.shift1[k]   = casse->shift1;
            BSS.scale[k]    = casse->scale;
            BSS.shift2[k]   = casse->shift2;
        }

        fc_boggle_BSS((data16_t*) &BSS, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                      (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, CHANS_OUT);

        nn_conv2d_1x1_plan_t plan;

        conv2d_1x1_init(&plan, &x_params, &y_params);

#if (DEBUG_ON || 0)
        PRINTF("plan.start_stride.X     = %ld\n", plan.start_stride.X);
        PRINTF("plan.start_stride.Y     = %ld\n", plan.start_stride.Y);
        PRINTF("plan.start_stride.K     = %ld\n", plan.start_stride.K);
        PRINTF("plan.cog_stride.Y       = %ld\n", plan.cog_stride.Y);
        PRINTF("plan.cog_stride.K       = %ld\n", plan.cog_stride.K);
        PRINTF("plan.cig_stride.body    = %ld\n", plan.cig_stride.body);
        PRINTF("plan.cig_stride.tail    = %ld\n", plan.cig_stride.tail);
        PRINTF("plan.pix_count          = %ld\n", plan.pix_count);
        PRINTF("plan.C_in               = %ld\n", plan.C_in);
        PRINTF("plan.C_out              = %ld\n", plan.C_out);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(Y_c));    //too expensive to write the whole image, so just do the part that's in play
        conv2d_1x1_c((int8_t*)Y_c, (int8_t*)X, (int8_t*)K, (data16_t*) &BSS, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(Y_asm));
        conv2d_1x1_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (data16_t*) &BSS, &plan);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = casse->y;

                    if(((uint8_t)casse->x)==0xED){
                        int8_t exps[] = {0x1F, 2*chn, 0x3E};
                        y_exp = exps[casse->k];
                    }

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
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)  [test vector @ line %u]", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn, casse->line);
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
#undef DEBUG_ON      
#undef CHANS_IN      
#undef CHANS_OUT     
#undef HEIGHT        
#undef WIDTH         
#undef CHANS_OUT_CEIL








#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define MAX_CHANS_IN    (3*VPU_INT8_EPV)
#define MAX_CHANS_OUT   (3*VPU_INT8_ACC_PERIOD)
#define HEIGHT          (1)
#define WIDTH           (2)
#define CHANS_OUT_CEIL  ((MAX_CHANS_OUT + (VPU_INT8_ACC_PERIOD-1)) & (0xFFFFFFFF << VPU_INT8_ACC_PERIOD_LOG2))
void test_conv2d_1x1_case1()
{
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][MAX_CHANS_IN];

    int8_t WORD_ALIGNED  K[MAX_CHANS_OUT][MAX_CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT_CEIL];
        int16_t shift1[CHANS_OUT_CEIL];
        int16_t scale[CHANS_OUT_CEIL];
        int16_t shift2[CHANS_OUT_CEIL];
    } BSS;

    int8_t Y_exp[MAX_CHANS_OUT];

#if TEST_C
    int8_t WORD_ALIGNED  Y_c[HEIGHT][WIDTH][MAX_CHANS_OUT];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[HEIGHT][WIDTH][MAX_CHANS_OUT];
#endif
    PRINTF("test_conv2d_1x1_case1()...\n");

    int8_t* K_flat  = (int8_t*)K;
    int8_t* X_flat  = (int8_t*)X;
    data16_t* BSS_flat = (data16_t*) &BSS;
#if TEST_C
    int8_t* Y_c_flat = (int8_t*) Y_c;
#endif
#if TEST_ASM
    int8_t* Y_asm_flat = (int8_t*) Y_asm;
#endif

    typedef struct {
        unsigned C_in;
        unsigned C_out;
        unsigned line;
    } test_case_t;

    const test_case_t casses[] = {
        //      C_in    C_out
        {       32,     16,         __LINE__}, 
        {       32,     32,         __LINE__}, 
        {       32,     48,         __LINE__}, 
        {       64,     16,         __LINE__}, 
        {       96,     16,         __LINE__}, 
        {       64,     32,         __LINE__}, 
        {       16,     16,         __LINE__}, 
        {        8,     16,         __LINE__}, 
        {        4,     16,         __LINE__}, 
        {       32,      8,         __LINE__}, 
        {       32,     12,         __LINE__}, 
        {       32,      4,         __LINE__}, 
        {       36,     16,         __LINE__}, 
        {       48,     16,         __LINE__},
        {       80,     16,         __LINE__}, 
        {       32,     24,         __LINE__}, 
        {       32,     20,         __LINE__}, 
        {       32,     36,         __LINE__}, 
        {       16,      8,         __LINE__}, 
        {        8,      8,         __LINE__}, 
        {       48,     24,         __LINE__}, 
        {       40,     12,         __LINE__}, 
        
    };

    const unsigned N_casses = sizeof(casses)/sizeof(test_case_t);
    const unsigned start_case =  0;
    const unsigned stop_case  = -1;
    print_warns(start_case, TEST_C, TEST_ASM);

    const int8_t x_val = 0x01;
    memset(X, x_val, sizeof(X));

    for(unsigned v = start_case; v < N_casses && v < stop_case; v++){
        const test_case_t* casse = (const test_case_t*) &casses[v];
        printf("\ttest vector %u...\n", v);

        const unsigned C_out = casse->C_out;
        const unsigned C_in = casse->C_in;

        for(int k = 0; k < C_out; k++){

            int a = k % 2;
            // int b = (k>>1) % 2;
            // int c = (k % 5) - 2;
            int c = a;
            int d = 0;


            memset(&K_flat[k*C_in], c, sizeof(int8_t) * C_out * C_in);

            int32_t bias = 0x1 << d;
            int16_t shift1 = d;
            int16_t scale = 1;
            int16_t shift2 = 0;

            BSS.bias[k] = bias;
            BSS_flat[2 * C_out + k] = shift1;
            BSS_flat[3 * C_out + k] = scale;
            BSS_flat[4 * C_out + k] = shift2;

            Y_exp[k] = ((x_val * c * C_in) >> d) + 1;
        }
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, C_in };
        nn_image_params_t y_params = { HEIGHT, WIDTH, C_out };

        fc_boggle_BSS((data16_t*) &BSS, (int32_t*) (void*)BSS_flat, (int16_t*) &BSS_flat[2*C_out], 
                      (int16_t*) &BSS_flat[3*C_out], (int16_t*) &BSS_flat[4*C_out], NULL, C_out);

        if(C_out % VPU_INT8_ACC_PERIOD){
            int a = C_out % VPU_INT8_ACC_PERIOD;
            int base = VPU_INT8_ACC_PERIOD * 5 * (C_out>>VPU_INT8_ACC_PERIOD_LOG2);
            for(int k = a; k < VPU_INT8_ACC_PERIOD; k++){
                BSS_flat[base + 0 * VPU_INT8_ACC_PERIOD + k] = 0xBEEF;
                BSS_flat[base + 1 * VPU_INT8_ACC_PERIOD + k] = 0xFEED;
                BSS_flat[base + 2 * VPU_INT8_ACC_PERIOD + k] = 0xDEAD;
                BSS_flat[base + 3 * VPU_INT8_ACC_PERIOD + k] = 0xACED;
                BSS_flat[base + 4 * VPU_INT8_ACC_PERIOD + k] = 0xFABB;
            }
        }

        nn_conv2d_1x1_plan_t plan;

        conv2d_1x1_init(&plan, &x_params, &y_params);

#if (DEBUG_ON || 0)
        PRINTF("plan.start_stride.X     = %ld\n", plan.start_stride.X);
        PRINTF("plan.start_stride.Y     = %ld\n", plan.start_stride.Y);
        PRINTF("plan.start_stride.K     = %ld\n", plan.start_stride.K);
        PRINTF("plan.cog_stride.Y       = %ld\n", plan.cog_stride.Y);
        PRINTF("plan.cog_stride.K       = %ld\n", plan.cog_stride.K);
        PRINTF("plan.cig_stride.body    = %ld\n", plan.cig_stride.body);
        PRINTF("plan.cig_stride.tail    = %ld\n", plan.cig_stride.tail);
        PRINTF("plan.pix_count          = %ld\n", plan.pix_count);
        PRINTF("plan.C_in               = %ld\n", plan.C_in);
        PRINTF("plan.C_out              = %ld\n", plan.C_out);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(int8_t) * y_params.height * y_params.width);
        conv2d_1x1_c((int8_t*)Y_c, (int8_t*)X, (int8_t*)K, BSS_flat, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(int8_t) * y_params.height * y_params.width);
        conv2d_1x1_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (data16_t*) &BSS, &plan);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = Y_exp[chn];

                    unsigned offset = IMG_ADDRESS_VECT(&y_params, row, col, chn);

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = ((int8_t*)Y_c)[offset];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = ((int8_t*)Y_asm)[offset];
                    flg |= (y_asm == y_exp)? 0x00 : 0x02;
#endif
                    if(flg){
                        sprintf(str_buff, "%s%s%s failed. (row, col, chn) = (%u, %u, %u)  [test vector @ line %u]", 
                                (flg&0x01)? "C" : "", (flg==0x03)? " and " : "", (flg&0x02)? "ASM" : "",
                                row, col, chn, casse->line);
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
#undef DEBUG_ON      
#undef CHANS_IN      
#undef CHANS_OUT     
#undef HEIGHT        
#undef WIDTH         
#undef CHANS_OUT_CEIL
#undef REPS









#define DEBUG_ON        (0 || TEST_DEBUG_ON)
#define MAX_CHANS_IN    (3*VPU_INT8_EPV)
#define MAX_CHANS_OUT   (3*VPU_INT8_ACC_PERIOD)
#define HEIGHT          (2)
#define WIDTH           (1)
#define CHANS_OUT_CEIL  ((MAX_CHANS_OUT + (VPU_INT8_ACC_PERIOD-1)) & (0xFFFFFFFF << VPU_INT8_ACC_PERIOD_LOG2))
#define REPS            (50)
void test_conv2d_1x1_case2()
{
    int8_t WORD_ALIGNED  X[HEIGHT][WIDTH][MAX_CHANS_IN];

    int8_t WORD_ALIGNED  K[MAX_CHANS_OUT][MAX_CHANS_IN];

    struct {
        int32_t bias[CHANS_OUT_CEIL];
        int16_t shift1[CHANS_OUT_CEIL];
        int16_t scale[CHANS_OUT_CEIL];
        int16_t shift2[CHANS_OUT_CEIL];
    } BSS;
    data16_t BSS2[CHANS_OUT_CEIL*5];

    int8_t Y_exp[MAX_CHANS_OUT];

#if TEST_C
    int8_t WORD_ALIGNED  Y_c[HEIGHT][WIDTH][MAX_CHANS_OUT];
#endif
#if TEST_ASM
    int8_t WORD_ALIGNED  Y_asm[HEIGHT][WIDTH][MAX_CHANS_OUT];
#endif

    int8_t* K_flat  = (int8_t*)K;
    int8_t* X_flat  = (int8_t*)X;
    data16_t* BSS_flat = (data16_t*) &BSS;
#if TEST_C
    int8_t* Y_c_flat = (int8_t*) Y_c;
#endif
#if TEST_ASM
    int8_t* Y_asm_flat = (int8_t*) Y_asm;
#endif

    PRINTF("test_conv2d_1x1_case2()...\n");

    print_warns(-1, TEST_C, TEST_ASM);

    const int8_t x_val = 1;

    memset(X, x_val, sizeof(X));

    //For debugging purposes, burn a number of pseudo_rands equal to this many 
    //  test cases. This skips the earlier tests without changing the result of the one
    //  that was bugged.
#define SKIP_REPS  (0)
    unsigned skip_reps = SKIP_REPS;
    

    for(unsigned rep = SKIP_REPS; rep < REPS; rep++){

        unsigned C_in = 4 * (pseudo_rand_uint32(&seed) % ((MAX_CHANS_IN>>2)-1)) + 4;
        unsigned C_out = 4 * (pseudo_rand_uint32(&seed) % ((MAX_CHANS_OUT>>2)-1)) + 4;
        if(rep == 0){
            C_in = VPU_INT8_EPV;
            C_out = VPU_INT8_ACC_PERIOD;
        }
        
        printf("\trep %u...(%u, %u)\n", rep, C_in, C_out);


        for(int k = 0; k < C_out; k++){
            int8_t k_val = pseudo_rand_uint32(&seed);

            memset(&K_flat[k*C_in], k_val, sizeof(int8_t) * C_in);
            
            const int32_t dot_prod = k_val * x_val * C_in;

            int32_t bias = pseudo_rand_int32(&seed) >> 8;

            const int32_t acc32 = bias + dot_prod;

            int l2_ub = ceil_log2(acc32);

            int16_t shift1 = 0;
            if(l2_ub > 15)
                shift1 = (pseudo_rand_uint32(&seed) % 4) + (l2_ub - 15);

            int16_t prescale = vlsat_single_s16(acc32, shift1);

            int16_t scale = pseudo_rand_uint32(&seed);
            scale = 2;

            int32_t postscale = prescale * ((int32_t)scale);
            l2_ub = ceil_log2(postscale < 0? -postscale : postscale);

            
            int16_t shift2 = 0;
            if(l2_ub > 7 && !(postscale*postscale == 1<<14)){
                 shift2 = l2_ub - 7 + pseudo_rand_uint32(&seed) % (3);
            }

            int8_t output = vlsat_single_s8(postscale, shift2);
            
            BSS.bias[k] = bias;
            BSS_flat[2*C_out + k] = shift1;
            BSS_flat[3*C_out + k] = scale;
            BSS_flat[4*C_out + k] = shift2;

            // if(!skip_reps)
            //     printf("%d:\t% 4d\t0x%08X\t0x%08X\n", k, k_val, bias, acc32);

            Y_exp[k] = output;
#if (DEBUG_ON || 0) // in case you need to convince yourself not all outputs are 0, -127 or 127
            PRINTF("Y_exp[%d] = %d\n", k, output);
#endif
        }

        //Here we've done all our pseudo_rand's, so we can continue without changing the behavior.
        if(skip_reps){
            PRINTF("\t\t(skipped)\n");
            skip_reps--;   
            continue;
        }
            
        nn_image_params_t x_params = { HEIGHT, WIDTH, C_in };
        nn_image_params_t y_params = { HEIGHT, WIDTH, C_out };

        fc_boggle_BSS((data16_t*) &BSS, (int32_t*) (void*) BSS_flat, (int16_t*) &BSS_flat[2*C_out], 
                      (int16_t*) &BSS_flat[3*C_out], (int16_t*) &BSS_flat[4*C_out], NULL, C_out);

        nn_conv2d_1x1_plan_t plan;

        conv2d_1x1_init(&plan, &x_params, &y_params);

#if (DEBUG_ON || 0)
        PRINTF("plan.start_stride.X     = %ld\n", plan.start_stride.X);
        PRINTF("plan.start_stride.Y     = %ld\n", plan.start_stride.Y);
        PRINTF("plan.start_stride.K     = %ld\n", plan.start_stride.K);
        PRINTF("plan.cog_stride.Y       = %ld\n", plan.cog_stride.Y);
        PRINTF("plan.cog_stride.K       = %ld\n", plan.cog_stride.K);
        PRINTF("plan.cig_stride.body    = %ld\n", plan.cig_stride.body);
        PRINTF("plan.cig_stride.tail    = %ld\n", plan.cig_stride.tail);
        PRINTF("plan.pix_count          = %ld\n", plan.pix_count);
        PRINTF("plan.C_in               = %ld\n", plan.C_in);
        PRINTF("plan.C_out              = %ld\n", plan.C_out);
#endif //DEBUG_ON

#if TEST_C
        PRINTF("\t\t\tC...\n");
        memset(Y_c, 0xCC, sizeof(int8_t) * y_params.height * y_params.width);
        conv2d_1x1_c((int8_t*)Y_c, (int8_t*)X, (int8_t*)K, BSS_flat, &plan);
#endif
#if TEST_ASM
        PRINTF("\t\t\tASM...\n");
        memset(Y_asm, 0xCC,  sizeof(int8_t) * y_params.height * y_params.width);
        conv2d_1x1_asm((int8_t*)Y_asm, (int8_t*)X, (int8_t*)K, (data16_t*) &BSS, &plan);
#endif

        char str_buff[200] = {0};
        PRINTF("\t\t\tChecking...\n");
        for(unsigned row = 0; row < y_params.height; row++){
            for(unsigned col = 0; col < y_params.width; col++){
                for(unsigned chn = 0; chn < y_params.channels; chn++){
                    
                    int8_t y_exp = Y_exp[chn];

                    unsigned offset = IMG_ADDRESS_VECT(&y_params, row, col, chn);

                    int flg = 0;     //Annoying, but avoids unnecessary calls to sprintf().
#if TEST_C      
                    int8_t y_c = ((int8_t*)Y_c)[offset];
                    flg |= (y_c == y_exp)? 0x00 : 0x01;
#endif
#if TEST_ASM
                    int8_t y_asm = ((int8_t*)Y_asm)[offset];
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
#undef DEBUG_ON      
#undef CHANS_IN      
#undef CHANS_OUT     
#undef HEIGHT        
#undef WIDTH         
#undef CHANS_OUT_CEIL
#undef REPS