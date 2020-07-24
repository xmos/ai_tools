
#if defined(__XS3A__)


.issue_mode  dual

//Plan layout
#define PLAN_OUTER_X_H_STEP        0
#define PLAN_OC_LOOP_COUNTER       1
#define PLAN_THRESHOLD_P           2
#define PLAN_INNER_X_V_STEP        3
#define PLAN_K_V_STEP              4
#define PLAN_INNER_X_H_STEP        5
#define PLAN_K_H_STEP              6
#define PLAN_OUTER_X_V_STEP        7
#define PLAN_Y_V_STEP              8
#define PLAN_K_HEIGHT_LOOP_COUNTER 9
#define PLAN_K_WIDTH_LOOP_COUNTER  10
#define PLAN_X_HEIGHT_LOOP_COUNTER 11
#define PLAN_X_WIDTH_LOOP_COUNTER  12
#define PLAN_IC_LOOP_COUNTER       13

//Stack Usage

#define S_0       0
#define S_1       1

#define S_SAVE_R4 2
#define S_SAVE_R5 3

#define S_SAVE_R6 4
#define S_SAVE_R7 5

#define S_SAVE_R8               6
#define S_SAVE_R9               7

#define S_SAVE_R10              8
#define S_K_p                   9 

#define S_THRESHOLDS_0_15       10
#define S_THRESHOLDS_16_31      11

#define S_X_p                   12
#define S_OUTER_X_H_STEP        13

#define S_OC_LOOP_COUNTER       14 
#define S_THRESHOLD_P           15 

#define S_INNER_X_V_STEP        16
#define S_K_V_STEP              17

#define S_INNER_X_H_STEP        18 
#define S_K_H_STEP              19

#define S_OUTER_X_V_STEP        20 
#define S_Y_V_STEP              21 

#define S_K_HEIGHT_LOOP_COUNTER 22
#define S_K_WIDTH_LOOP_COUNTER  23

#define S_X_HEIGHT_LOOP_COUNTER 24 
#define S_X_WIDTH_LOOP_COUNTER  25

#define S_IC_LOOP_COUNTER       26 

//Note: The elements 27 -> 30 will be clobbered when copying the plan to the stack
//      this might mean leaving some spare stack space

#define S_ZERO_0 28
#define S_ZERO_1 29 
#define S_ZERO_2 30
#define S_ZERO_3 31
#define S_ZERO_4 32
#define S_ZERO_5 33
#define S_ZERO_6 34
#define S_ZERO_7 35



//Helpers that reference the above
#define S_SAVE_R4_R5 (S_SAVE_R4/2) //0
#define S_SAVE_R6_R7 (S_SAVE_R6/2)
#define S_SAVE_R8_R9 (S_SAVE_R8/2)
#define S_SAVE_R10_K_p (S_SAVE_R10/2)
#define S_X_p_S_OUTER_X_H_STEP (S_X_p/2)
#define S_THRESHOLDS (S_THRESHOLDS_0_15/2)
#define S_OC_LOOP_COUNTER_THRESHOLD_P (S_OC_LOOP_COUNTER/2)
#define S_INNER_V_STEPS (S_INNER_X_V_STEP/2)
#define S_INNER_H_STEPS (S_INNER_X_H_STEP/2)
#define S_OUTER_X_V_STEP_Y_V_STEP (S_OUTER_X_V_STEP/2)

#define S_ZERO_256 S_ZERO_0

//Registers

#define Y_p       r0
#define X_p       r1
#define K_p       r2
#define threshold_current r3

#define X_p r1
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

    {vldr t[0]; add threshold_current, threshold_current, s}
    {vldd threshold_current[0]; add threshold_current, threshold_current, s}

    {ldw k_height_loop, sp[S_K_HEIGHT_LOOP_COUNTER]; nop}
    kh_loop:
        
        {ldw k_width_loop, sp[S_K_WIDTH_LOOP_COUNTER]; nop}
        kw_loop:
  
            {ldw ic_loop_counter, sp[S_IC_LOOP_COUNTER]; nop}
            ic_inner_loop:
                {vldc X_p[0];      add X_p, X_p, r_256_bits}

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

            {bt ic_loop_counter, ic_inner_loop; sub ic_loop_counter, ic_loop_counter, 1}

            ldd s, t, sp[S_INNER_H_STEPS]
            {add X_p, X_p, t; add K_current, K_current, s }

        {bt k_width_loop, kw_loop; sub k_width_loop, k_width_loop, 1}

        ldd s, t, sp[S_INNER_V_STEPS]
        {add X_p, X_p, t; add K_current, K_current, s }

    {bt k_height_loop, kh_loop; sub k_height_loop, k_height_loop, 1}

    retsp 0


/*
void bnn_conv2d_bin_out_asm(bnn_b32_t* Y_p, const bnn_b256_t* X_p,
                        const bnn_b256_t* K_p,
                        int32_t* thresholds,  //[out_channel];
                        const nn_bnn_conv2d_bin_out_plan_t* plan)
*/

#define FUNCTION_NAME bnn_conv2d_bin_out_asm

#define NSTACKWORDS  36
    
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

    stw r1, sp[S_X_p]

    ldc r_256_bits, (256/8)
    
	ldc r11, 256 //set int16 mode - 32 bit accumulators	
    {vsetc r11; nop}
    
    ldaw s, sp[NSTACKWORDS+1] //load the plan pointer
    {ldw t, s[0]; nop}

    {vldr t[0];  add t, t, r_256_bits}
    ldaw s, sp[S_OUTER_X_H_STEP]
    {vstr s[0]; add s, s, r_256_bits}

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //This could use a bit mask

    stw r3, sp[S_THRESHOLD_P]

    //Set up the zeros (and clear the over write from the vector copy)
    ldaw t, sp[S_ZERO_0]
    {ldc s, 0; nop}
    std s, s, t[0]
    std s, s, t[1]
    std s, s, t[2]
    std s, s, t[3]

    h_loop:

        {ldw x_width_loop, sp[S_X_WIDTH_LOOP_COUNTER]; nop}
        w_loop:

            ldd threshold_current, oc_loop_counter, sp[S_OC_LOOP_COUNTER_THRESHOLD_P]

            //init K_current
            {ldw K_current, sp[S_K_p]; nop}

            oc_loop: //this loops over (output_channels/32)
            
                {ldw X_p, sp[S_X_p]; ldc s, 32}
                {bl make_16_bit_output ; mov t, threshold_current }

                ldaw s, sp[S_THRESHOLDS_0_15]
                ldaw t, sp[S_ZERO_256] //TODO use the constant pool
                {vlsat t[0] ; nop}
                {vdepth1; ldc t, 3}
                vstrpv s[0], t

                {ldw X_p, sp[S_X_p]; ldc s, 32}
                {bl make_16_bit_output ; mov t, threshold_current }

                ldaw s, sp[S_THRESHOLDS_16_31] 
                ldaw t, sp[S_ZERO_256] //TODO use the constant pool
                {vlsat t[0] ; nop} 
                {vdepth1; ldc t, 3}
                vstrpv s[0], t
                
                //merge the two 16 bits together
                ldd s, t, sp[S_THRESHOLDS]
                {shl s, s, 16; zext t, 16}
                {add t, s, t; nop}

                {add Y_p, Y_p, 4 ; stw t, Y_p[0]} //save the 32 results to memory

            {bt oc_loop_counter, oc_loop; sub oc_loop_counter, oc_loop_counter, 1}

            //advance X by one(STEP) to the right
            ldd X_p, t, sp[S_X_p_S_OUTER_X_H_STEP]
            {add X_p, X_p, t; nop }
            stw X_p, sp[S_X_p]

        {bt x_width_loop, w_loop; sub x_width_loop, x_width_loop, 1 }

        //TODO this section could be scheduled to be faster
        ldd s, t, sp[S_OUTER_X_V_STEP_Y_V_STEP]
        {ldw X_p, sp[S_X_p]; nop}
        {add X_p, X_p, t; add Y_p, Y_p, s }
        stw X_p, sp[S_X_p]
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



