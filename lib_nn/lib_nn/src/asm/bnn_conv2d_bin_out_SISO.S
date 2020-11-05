#if defined(__XS3A__)

.issue_mode  dual

//Plan layout
#define PLAN_K_HEIGHT_LOOP_COUNTER 0
#define PLAN_DATA_SCRATCH          1
#define PLAN_K_WIDTH_LOOP_COUNTER  2
#define PLAN_INNER_X_V_STEP        3
#define PLAN_INNER_X_H_STEP        4 
#define PLAN_DATA_SCRATCH_ADJUST   5
#define PLAN_OC_LOOP_COUNTER       6 
#define PLAN_THRESHOLD_P           7
#define PLAN_X_p                   8
#define PLAN_OUTER_X_H_STEP        9
#define PLAN_OUTER_X_V_STEP        10 
#define PLAN_Y_V_STEP              11 
#define PLAN_PATCH_LOOP_COUNTER    12   
#define PLAN_X_WIDTH_LOOP_COUNTER  13  
#define PLAN_K_p                   14 
#define PLAN_X_HEIGHT_LOOP_COUNTER 15  
#define PLAN_IC_LOOP_COUNTER       16
#define PLAN_K_ADJUST              17  
#define PLAN_Y_p                   18

//Stack Usage

#define S_0                     0
#define S_SAVE_R10              1

#define S_SAVE_R4               2
#define S_SAVE_R5               3

#define S_SAVE_R6               4
#define S_SAVE_R7               5

#define S_SAVE_R8               6
#define S_SAVE_R9               7

#define S_THRESHOLDS_0_15       8
#define S_THRESHOLDS_16_31      9

#define S_K_HEIGHT_LOOP_COUNTER 10
#define S_DATA_SCRATCH          11

#define S_K_WIDTH_LOOP_COUNTER  12
#define S_INNER_X_V_STEP        13

#define S_INNER_X_H_STEP        14 
#define S_DATA_SCRATCH_ADJUST   15

#define S_OC_LOOP_COUNTER       16 
#define S_THRESHOLD_P           17

#define S_X_p                   18
#define S_OUTER_X_H_STEP        19
 
#define S_OUTER_X_V_STEP        20 
#define S_Y_V_STEP              21 

#define S_PATCH_LOOP_COUNTER    22   //wants to be next to S_DATA_SCRATCH
#define S_X_WIDTH_LOOP_COUNTER  23   //wants to be next to S_X_p
#define S_K_p                   24   //wants to be next to S_X_p
#define S_X_HEIGHT_LOOP_COUNTER 25  
#define S_IC_LOOP_COUNTER       26
#define S_K_ADJUST              27  
#define S_Y_p                   28

//doesn't have to be double work aligned
#define S_ZERO_0 29
#define S_ZERO_1 30 
#define S_ZERO_2 31
#define S_ZERO_3 32
#define S_ZERO_4 33
#define S_ZERO_5 34
#define S_ZERO_6 35
#define S_ZERO_7 36

//Helpers that reference the above
#define S_SAVE_R4_R5 (S_SAVE_R4/2)                  //1
#define S_SAVE_R6_R7 (S_SAVE_R6/2)                  //2
#define S_SAVE_R8_R9 (S_SAVE_R8/2)                  //3
#define S_THRESHOLDS (S_THRESHOLDS_0_15/2)          //4
#define S_K_HEIGHT_LOOP_COUNTER_DATA_SCRATCH (S_K_HEIGHT_LOOP_COUNTER/2)    //5
#define S_K_WIDTH_LOOP_COUNTER_INNER_X_V_STEP (S_K_WIDTH_LOOP_COUNTER/2)    //6
#define S_INNER_X_H_STEP_DATA_SCRATCH_ADJUST (S_INNER_X_H_STEP/2)           //7
#define S_OC_LOOP_COUNTER_THRESHOLD_P (S_OC_LOOP_COUNTER/2)                 //8
#define S_X_p_OUTER_X_H_STEP (S_X_p/2)                                      //9
#define S_OUTER_X_V_STEP_Y_V_STEP (S_OUTER_X_V_STEP/2)                      //10

#define S_ZERO_256 S_ZERO_0

//Registers

#define Y_p               r0
#define X_p               r1
#define K_p               r2
#define threshold_current r3 //threshold_current and D_p are never used at the same time
#define D_p               r3 

#define k_height_loop     r4
#define k_width_loop      r5
#define ic_loop_counter   r6
#define oc_loop_counter   r7
#define r_256_bits        r8
#define x_width_loop      r9
#define s                 r10
#define t                 r11


.align 4 //TODO align correctly
bnn_make_patch:

    ldd D_p, k_height_loop, sp[S_K_HEIGHT_LOOP_COUNTER_DATA_SCRATCH]
    bnn_kh_loop:

        //This could be hoisted and executed on the output
        {ldw k_width_loop, sp[S_K_WIDTH_LOOP_COUNTER]; nop} 

        bnn_kw_loop:
  
            {ldw ic_loop_counter, sp[S_IC_LOOP_COUNTER]; nop}
            bnn_ic_inner_loop:
                {vldd X_p[0];      add X_p, X_p, r_256_bits}
                {vstd D_p[0];      add D_p, D_p, r_256_bits} //D_p needs space for the overrun
            {bt ic_loop_counter, bnn_ic_inner_loop; sub ic_loop_counter, ic_loop_counter, 1}

            ldd s, t, sp[S_INNER_X_H_STEP_DATA_SCRATCH_ADJUST]
            {add X_p, X_p, t; add D_p, D_p, s} //this can rewind D_p up to 256 - 32 bits

        {bt k_width_loop, bnn_kw_loop; sub k_width_loop, k_width_loop, 1}

        //ldd s, t, sp[S_K_WIDTH_LOOP_COUNTER_INNER_X_V_STEP] //S_K_WIDTH_LOOP_COUNTER and S_INNER_X_V_STEP should be together
        ldw t, sp[S_INNER_X_V_STEP]
        {add X_p, X_p, t; nop }

    {bt k_height_loop, bnn_kh_loop; sub k_height_loop, k_height_loop, 1}

    vclrdr
    vstr D_p[0]   //D_p points to the final (possibly-partial) set of data //buffer needs to be at most 256 bits too big

    retsp 0

.align 4 //TODO align correctly
bnn_compute_patch:

    {ldc s, 32; nop}
    {vldr t[0]; add threshold_current, threshold_current, s}
    {vldd threshold_current[0]; add threshold_current, threshold_current, s}

    {ldw t, sp[S_PATCH_LOOP_COUNTER]; nop}
    {ldw X_p, sp[S_DATA_SCRATCH]; nop}
    {bf t, bnn_patch_tail; sub t, t, 1}

    bnn_patch_loop:
        {vldc X_p[0];      add X_p, X_p, r_256_bits}
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 

        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 

        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 

        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, r_256_bits}; 
        
    {bt t, bnn_patch_loop; sub t, t, 1}

    bnn_patch_tail: 
    //if this has different number of fnops between the fall through and branch to 
    //then there will be error in the instruction estimator(the error will be bounded to 
    //x_height*x_width inst hopefully)
    
    ldw s, sp[S_K_ADJUST]

    {vldc X_p[0];     nop }
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 

    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 

    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 

    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    
    {retsp 0; nop}

/*
void bnn_conv2d_bin_out_SISO_asm(const nn_bnn_conv2d_bin_out_asm_plan_t* plan);
*/

#define FUNCTION_NAME bnn_conv2d_bin_out_SISO_asm

#define NSTACKWORDS  40 //FIXME should be lower
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 2
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[S_SAVE_R4_R5]
    std r6, r7, sp[S_SAVE_R6_R7]
    std r8, r9, sp[S_SAVE_R8_R9]
    stw r10, sp[S_SAVE_R10]

	ldc r11, 256 //set int16 mode - 32 bit accumulators	
    {vsetc r11; shr r_256_bits, r11, 3} 
    
    {mov t, r0;  ldc s, PLAN_Y_p}
    {ldw r0, t[s]; ldc s, PLAN_X_p} //put PLAN_Y_P into r0
    {ldw r1, t[s]; ldc s, PLAN_K_p}
    {ldw r2, t[s]; nop}

    {vldr t[0];  add t, t, r_256_bits}
    ldaw s, sp[S_K_HEIGHT_LOOP_COUNTER] //this is where we will start the plan copy from
    {vstr s[0]; add s, s, r_256_bits}

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //second copy

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //third copy 

    //Set up the zeros (and clear the over write from the vector copy)
    ldaw t, sp[S_ZERO_0]
    vclrdr
    vstr t[0]

    bnn_h_loop:

        {ldw x_width_loop, sp[S_X_WIDTH_LOOP_COUNTER]; nop}
        bnn_w_loop:

            {ldw X_p, sp[S_X_p]; } //this could be hoisted
            {bl bnn_make_patch ; nop}

            //init K_p
            {ldw K_p, sp[S_K_p]; nop}

            ldd threshold_current, oc_loop_counter, sp[S_OC_LOOP_COUNTER_THRESHOLD_P]

            bnn_oc_loop: //this loops over (output_channels/32)
            
                {bl bnn_compute_patch ; mov t, threshold_current }

                ldaw s, sp[S_THRESHOLDS_0_15]
                ldaw t, sp[S_ZERO_256] //TODO use the constant pool
                {vlsat t[0] ; nop}
                {vdepth1; ldc t, 3}
                vstrpv s[0], t

                {bl bnn_compute_patch ; mov t, threshold_current }

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

            {bt oc_loop_counter, bnn_oc_loop; sub oc_loop_counter, oc_loop_counter, 1}

            //advance X by one(STEP) to the right
            ldd X_p, t, sp[S_X_p_OUTER_X_H_STEP]
            {add X_p, X_p, t; nop }
            stw X_p, sp[S_X_p]

        {bt x_width_loop, bnn_w_loop; sub x_width_loop, x_width_loop, 1 }

        //TODO this section could be scheduled to be faster
        ldd s, t, sp[S_OUTER_X_V_STEP_Y_V_STEP]
        {ldw X_p, sp[S_X_p]; nop}
        {add X_p, X_p, t; add Y_p, Y_p, s }
        stw X_p, sp[S_X_p]
        {ldw t, sp[S_X_HEIGHT_LOOP_COUNTER]; nop}
        {sub t, t, 1; nop}
        stw t, sp[S_X_HEIGHT_LOOP_COUNTER] 
 
    {bt t, bnn_h_loop; nop}

    ldd r4, r5, sp[S_SAVE_R4_R5]
    ldd r6, r7, sp[S_SAVE_R6_R7]
    ldd r8, r9, sp[S_SAVE_R8_R9]
    ldw r10, sp[S_SAVE_R10]
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



