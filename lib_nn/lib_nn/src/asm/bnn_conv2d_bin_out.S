
#if defined(__XS3A__)


.issue_mode  dual

//Plan layout
#define PLAN_X_V_STRIDE 0
#define PLAN_X_H_STRIDE 1
#define PLAN_K_HEIGHT_LOOP_COUNTER 2
#define PLAN_K_WIDTH_LOOP_COUNTER 3
#define PLAN_IC_LOOP_COUNTER 4
#define PLAN_OC_LOOP_COUNTER 5
#define PLAN_X_HEIGHT_LOOP_COUNTER 6
#define PLAN_X_WIDTH_LOOP_COUNTER 7
#define PLAN_Y_V_STRIDE 8
#define PLAN_OUTER_X_H_STRIDE 9
#define PLAN_OUTER_X_V_STRIDE 10

//Stack Usage

#define S_0       0
#define S_1       1

#define S_SAVE_R4 2
#define S_SAVE_R5 3

#define S_SAVE_R6 4
#define S_SAVE_R7 5

#define S_SAVE_R8 6
#define S_SAVE_R9 7

#define S_SAVE_R10 8
#define S_K_p      9  

#define S_X_p 10
#define S_Y_p 11

#define S_THRESHOLDS_0_15 12
#define S_THRESHOLDS_16_31 13

#define S_K_HEIGHT_LOOP_COUNTER 14
#define S_K_WIDTH_LOOP_COUNTER 15

#define S_X_HEIGHT_LOOP_COUNTER 16
#define S_X_WIDTH_LOOP_COUNTER 17

#define S_IC_LOOP_COUNTER 18
#define S_OC_LOOP_COUNTER 19

#define S_ZERO_0 20
#define S_ZERO_1 21 
#define S_ZERO_2 22
#define S_ZERO_3 23
#define S_ZERO_4 24
#define S_ZERO_5 25
#define S_ZERO_6 26
#define S_ZERO_7 27

#define S_THRESHOLD_P 28
#define S_Y_V_STRIDE 29

#define S_X_V_STRIDE 30
#define S_X_H_STRIDE 31

#define S_OUTER_X_H_STRIDE 32
#define S_OUTER_X_V_STRIDE 33


//Helpers that reference the above
#define S_SAVE_R4_R5 (S_SAVE_R4/2)
#define S_SAVE_R6_R7 (S_SAVE_R6/2)
#define S_SAVE_R8_R9 (S_SAVE_R8/2)
#define S_SAVE_R10_K_p (S_SAVE_R10/2)
#define S_X_p_Y_p (S_X_p/2)
#define S_THRESHOLDS (S_THRESHOLDS_0_15/2)

#define S_ZERO_256 S_ZERO_0

//Registers

#define Y_p       r0
#define X_p       r1
#define K_p       r2
#define threshold_current r3

#define X_current r1
#define K_current r2

#define k_height_loop r4
#define k_width_loop r5
#define ic_loop_counter r6
#define oc_loop_counter r7
#define r_256_bits r8
#define x_width_loop r9
#define s r10
#define t r11

make_16_bit_output:

    //init R to threshold_current
    mov t, threshold_current
    {vldr t[0];nop}
    ldaw threshold_current, threshold_current[8] //advance the threshold_current pointer by 16 half words
    
    {vldd threshold_current[0];nop}
    ldaw threshold_current, threshold_current[8] //advance the threshold_current pointer by 16 half words

    //X_current points to X[h + 0][w + 0][ic]

    ldw k_height_loop, sp[S_K_HEIGHT_LOOP_COUNTER]
    kh_loop:
        
        ldw k_width_loop, sp[S_K_WIDTH_LOOP_COUNTER]
        kw_loop:

            //get the X data for X[h + kh][w + kw][ic]
            vldc X_current[0]
            
            ldw ic_loop_counter, sp[S_IC_LOOP_COUNTER]
            ic_inner_loop:

                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 

                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 

                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 

                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 
                {vlmaccr1 K_current[0] ; add K_current, K_current, r_256_bits}; 

                //advance X by 256 bits
                {add X_current, X_current, r_256_bits; nop} //schedule this to avoid an fnop

            {bt ic_loop_counter, ic_inner_loop; sub ic_loop_counter, ic_loop_counter, 1}

            //advance X by one(stride) right, dont forget we've on the next pixel
            ldw t, sp[S_X_H_STRIDE]
            {add X_current, X_current, t; nop}

        {bt k_width_loop, kw_loop; sub k_width_loop, k_width_loop, 1}

        //advance X by one(stride) down
        ldw t, sp[S_X_V_STRIDE]
        {add X_current, X_current, t; nop}

    {bt k_height_loop, kh_loop; sub k_height_loop, k_height_loop, 1}

    retsp 0


/*
void bnn_conv2d_bin_out_asm(bnn_b32_t* Y_p, const bnn_b256_t* X_p,
                        const bnn_b256_t* K_p,
                        int32_t* thresholds,  //[out_channel];
                        const nn_bnn_conv2d_bin_out_plan_t* plan)
*/

#define FUNCTION_NAME bnn_conv2d_bin_out_asm

#define NSTACKWORDS  34
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

//There must be multiples of 32 output channels
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[S_SAVE_R4_R5]
    std r6, r7, sp[S_SAVE_R6_R7]
    std r8, r9, sp[S_SAVE_R8_R9]
    std r2, r10, sp[S_SAVE_R10_K_p]

    std r0, r1, sp[S_X_p_Y_p]

    stw r3, sp[S_THRESHOLD_P]
    
	ldc r11, 256 //set int16 mode - 32 bit accumulators	
    vsetc r11;

    ldaw t, sp[NSTACKWORDS+1] //load the plan pointer
    ldw t, t[0]
    ldw s, t[PLAN_X_V_STRIDE]
    stw s, sp[S_X_V_STRIDE]

    ldw s, t[PLAN_X_H_STRIDE]
    stw s, sp[S_X_H_STRIDE]

    ldw s, t[PLAN_K_HEIGHT_LOOP_COUNTER]
    stw s, sp[S_K_HEIGHT_LOOP_COUNTER]

    ldw s, t[PLAN_K_WIDTH_LOOP_COUNTER]
    stw s, sp[S_K_WIDTH_LOOP_COUNTER]

    ldw s, t[PLAN_IC_LOOP_COUNTER]
    stw s, sp[S_IC_LOOP_COUNTER]

    ldw s, t[PLAN_OC_LOOP_COUNTER]
    stw s, sp[S_OC_LOOP_COUNTER]

    ldw s, t[PLAN_X_WIDTH_LOOP_COUNTER]
    stw s, sp[S_X_WIDTH_LOOP_COUNTER]

    ldw s, t[PLAN_X_HEIGHT_LOOP_COUNTER]
    stw s, sp[S_X_HEIGHT_LOOP_COUNTER]

    ldw s, t[PLAN_Y_V_STRIDE]
    stw s, sp[S_Y_V_STRIDE]

    ldw s, t[PLAN_OUTER_X_H_STRIDE]
    stw s, sp[S_OUTER_X_H_STRIDE]

    ldw s, t[PLAN_OUTER_X_V_STRIDE]
    stw s, sp[S_OUTER_X_V_STRIDE]

    {ldc s, 0; nop}
    stw s, sp[S_ZERO_0]
    stw s, sp[S_ZERO_1]
    stw s, sp[S_ZERO_2]
    stw s, sp[S_ZERO_3]
    stw s, sp[S_ZERO_4]
    stw s, sp[S_ZERO_5]
    stw s, sp[S_ZERO_6]
    stw s, sp[S_ZERO_7]

    start:
    ldc r_256_bits, (256/8)

    h_loop:

        {ldw x_width_loop, sp[S_X_WIDTH_LOOP_COUNTER]; nop}
        w_loop:

            ldw threshold_current, sp[S_THRESHOLD_P]
            ldw oc_loop_counter, sp[S_OC_LOOP_COUNTER]

            //init K_current
            {ldw K_current, sp[S_K_p]; nop}

            oc_loop: //this loops over (output_channels/32)
                
                //load X_p to X_current

                ldw X_current, sp[S_X_p]
                bl make_16_bit_output 

                ldaw s, sp[S_THRESHOLDS_0_15] //this will hold the lower 16 bits of the 32 output channels
                ldaw t, sp[S_ZERO_256]
                {vlsat t[0] ; nop}
                {vdepth1; ldc t, 3}
                vstrpv s[0], t

                //load X_p to X_current
                ldw X_current, sp[S_X_p]

                bl make_16_bit_output

                ldaw s, sp[S_THRESHOLDS_16_31] //this will hold the lower 16 bits of the 32 output channels
                ldaw t, sp[S_ZERO_256]
                {vlsat t[0] ; nop}
                {vdepth1; ldc t, 3}
                vstrpv s[0], t
                
                //merge the two 16 bits together
                ldd s, t, sp[S_THRESHOLDS]
                {shl s, s, 16; zext t, 16}
                {add t, s, t; nop}

                {not t, t; nop}
                stw t, Y_p[0]
                {add Y_p, Y_p, 4 ; nop}
                //save the 32 results to memory

            {bt oc_loop_counter, oc_loop; sub oc_loop_counter, oc_loop_counter, 1}

            //advance X by one(stride) to the right
            lbl0:
            ldw t, sp[S_OUTER_X_H_STRIDE]
            ldw X_current, sp[S_X_p]
            {add X_current, X_current, t; nop }
            stw X_current, sp[S_X_p]

        {bt x_width_loop, w_loop; sub x_width_loop, x_width_loop, 1 }

        //TODO advance Y to the next line
        {ldw t, sp[S_Y_V_STRIDE]; nop}
        {add Y_p, Y_p, t; nop }

        {ldw t, sp[S_OUTER_X_V_STRIDE]; nop}
        ldw X_current, sp[S_X_p]
        {add X_current, X_current, t; nop }
        stw X_current, sp[S_X_p]

        {ldw t, sp[S_X_HEIGHT_LOOP_COUNTER]; nop}
        {sub t, t, 1; nop}
        stw t, sp[S_X_HEIGHT_LOOP_COUNTER] 
 
    {bt t, h_loop; nop}

.Lfunc_end:
    ldd r4, r5, sp[S_SAVE_R4_R5]
    ldd r6, r7, sp[S_SAVE_R6_R7]
    ldd r8, r9, sp[S_SAVE_R8_R9]
    ldd r2, r10, sp[S_SAVE_R10_K_p]
    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif



