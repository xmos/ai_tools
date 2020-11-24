#if defined(__XS3A__)

.section        .cp.const32,"aMc",@progbits,32
.cc_top VECT_INT16_MAX.data
.align 4
VECT_INT16_MAX:
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.cc_bottom VECT_INT16_MAX.data

.section        .cp.const32,"aMc",@progbits,32
.cc_top VECT_INT8_MIN.data
.align 4
VECT_INT8_MIN:
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.cc_bottom VECT_INT8_MIN.data
.text

.issue_mode  dual

//Plan layout
#define PLAN_X_P                   0
#define PLAN_OUTER_X_H_STEP        1
#define PLAN_OC_LOOP_COUNTER       2
#define PLAN_K_P                   3
#define PLAN_INNER_X_V_STEP        4
#define PLAN_K_V_STEP              5
#define PLAN_INNER_X_H_STEP        6
#define PLAN_K_H_STEP              7

#define PLAN_OUTER_X_V_STEP        8
#define PLAN_Y_V_STEP              9
#define PLAN_K_HEIGHT_LOOP_COUNTER 10
#define PLAN_K_WIDTH_LOOP_COUNTER  11
#define PLAN_X_HEIGHT_LOOP_COUNTER 12
#define PLAN_X_WIDTH_LOOP_COUNTER  13
#define PLAN_CUR_PA_MUL            14
#define PLAN_CUR_PA_BIAS           15

#define PLAN_VLSAT                 16
#define PLAN_ASHR                  17
#define PLAN_FINAL_SHR             18
#define PLAN_BIAS_MULTIPLIER       19
#define PLAN_PA_MUL                20
#define PLAN_PA_BIAS               21
#define PLAN_IC_LOOP_COUNTER       22
#define PLAN_Y_P                   23

//Stack Usage

#define S_0                     0
#define S_1                     1
////
#define S_X_p                   2
#define S_OUTER_X_H_STEP        3

#define S_OC_LOOP_COUNTER       4 
#define S_K_p                   5 

#define S_INNER_X_V_STEP        6
#define S_K_V_STEP              7

#define S_INNER_X_H_STEP        8 
#define S_K_H_STEP              9
////
#define S_OUTER_X_V_STEP        10 
#define S_Y_V_STEP              11 

#define S_K_HEIGHT_LOOP_COUNTER 12
#define S_K_WIDTH_LOOP_COUNTER  13

#define S_X_HEIGHT_LOOP_COUNTER 14 
#define S_X_WIDTH_LOOP_COUNTER  15

#define S_CUR_PA_MUL_p          16
#define S_CUR_PA_BIAS_p         17
////
#define S_VLSAT                 18
#define S_ASHR                  19

#define S_FINAL_SHR             20
#define S_BIAS_MULTIPLIER       21

#define S_PA_MUL_p              22
#define S_PA_BIAS_p             23

#define S_IC_LOOP_COUNTER       24
#define S_Y_p                   25

////
#define S_SAVE_R4               26
#define S_SAVE_R5               27

#define S_SAVE_R6               28
#define S_SAVE_R7               29

#define S_SAVE_R8               30
#define S_SAVE_R9               31

#define S_SAVE_R10              32
#define S_UNUSED_0              33


//Note: The elements 30 -> 37 will be clobbered when copying the plan to the stack
//      this might mean leaving some spare stack space

//doesn't have to be double word aligned
#define S_TEMP_0 34
#define S_TEMP_1 35 
#define S_TEMP_2 36
#define S_TEMP_3 37
#define S_TEMP_4 38
#define S_TEMP_5 39
#define S_TEMP_6 40
#define S_TEMP_7 41

//doesn't have to be double word aligned
#define S_VLSAT_0 42
#define S_VLSAT_1 43 
#define S_VLSAT_2 44
#define S_VLSAT_3 45
#define S_VLSAT_4 46
#define S_VLSAT_5 47
#define S_VLSAT_6 48
#define S_VLSAT_7 49

//doesn't have to be double word aligned
#define S_BIAS_SHIFT_0 50
#define S_BIAS_SHIFT_1 51 
#define S_BIAS_SHIFT_2 52
#define S_BIAS_SHIFT_3 53
#define S_BIAS_SHIFT_4 54
#define S_BIAS_SHIFT_5 55
#define S_BIAS_SHIFT_6 56
#define S_BIAS_SHIFT_7 57

//doesn't have to be double word aligned
#define S_FINAL_SHR_0 58
#define S_FINAL_SHR_1 59 
#define S_FINAL_SHR_2 60
#define S_FINAL_SHR_3 61
#define S_FINAL_SHR_4 62
#define S_FINAL_SHR_5 63
#define S_FINAL_SHR_6 64
#define S_FINAL_SHR_7 65

#define NSTACKWORDS  70 //50!

//Helpers that reference the above
#define S_X_p_S_OUTER_X_H_STEP (S_X_p/2)                        //1
#define S_OC_LOOP_COUNTER_K_p (S_OC_LOOP_COUNTER/2)     //2
#define S_INNER_V_STEPS (S_INNER_X_V_STEP/2)                    //3
#define S_INNER_H_STEPS (S_INNER_X_H_STEP/2)                    //4

#define S_OUTER_X_V_STEP_Y_V_STEP (S_OUTER_X_V_STEP/2)          //5
                                                                //6
                                                                //7
#define S_CUR_PA_MUL_p_CUR_PA_BIAS_p (S_CUR_PA_MUL_p/2)         //8

#define S_VLSAT_ASHR (S_VLSAT/2)                                //9
#define S_FINAL_SHR_MASK (S_FINAL_SHR/2)                        //10
#define S_PA_MUL_p_PA_BIAS_p (S_PA_MUL_p/2)                     //11

#define S_TEMP S_TEMP_0

//Registers

#define Y_p               r0
#define X_p               r1
#define K_p               r2
#define r                 r3 //apparently unused!

#define k_height_loop     r4
#define k_width_loop      r5
#define ic_loop_counter   r6
#define oc_loop_counter   r7
#define r_256_bits        r8
#define x_width_loop      r9
#define s                 r10
#define t                 r11
.align 4
bnn_make_16_channel_output:

    {vclrdr; nop}

    //TODO reorder the stack to use a double load here with S_X_p
    {ldw k_height_loop, sp[S_K_HEIGHT_LOOP_COUNTER]; nop} 
    bnn_kh_loop:
        
        {ldw k_width_loop, sp[S_K_WIDTH_LOOP_COUNTER]; nop}
        bnn_kw_loop:
  
            {ldw ic_loop_counter, sp[S_IC_LOOP_COUNTER]; nop}
            bnn_ic_inner_loop:
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

            {bt ic_loop_counter, bnn_ic_inner_loop; sub ic_loop_counter, ic_loop_counter, 1}

            ldd s, t, sp[S_INNER_H_STEPS]
            {add X_p, X_p, t; add K_p, K_p, s }

        {bt k_width_loop, bnn_kw_loop; sub k_width_loop, k_width_loop, 1}

        ldd s, t, sp[S_INNER_V_STEPS]
        {add X_p, X_p, t; add K_p, K_p, s }

    {bt k_height_loop, bnn_kh_loop; sub k_height_loop, k_height_loop, 1}

    retsp 0

/*
void bconv2d_int8_DIDO_impl(const nn_bconv2d_int8_DIDO_impl_plan_t* plan);
*/

#define FUNCTION_NAME bconv2d_int8_DIDO_impl

    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 2
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    stw r4, sp[S_SAVE_R4]
    stw r5, sp[S_SAVE_R5]
    stw r6, sp[S_SAVE_R6]
    stw r7, sp[S_SAVE_R7]
    stw r8, sp[S_SAVE_R8]
    stw r9, sp[S_SAVE_R9]
    stw r10, sp[S_SAVE_R10]

	ldc r11, 256 //set int16 mode - 32 bit accumulators	
    {vsetc r11; shr r_256_bits, r11, 3} 
    
    {mov t, r0;  ldc s, PLAN_Y_P}
    {ldw r0, t[s]; ldc s, PLAN_K_P} //put PLAN_Y_P into r0
    {ldw r2, t[s]; nop}         //put PLAN_K_P into r2

    {vldr t[0];  add t, t, r_256_bits}
    ldaw s, sp[S_X_p]
    {vstr s[0]; add s, s, r_256_bits}

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} 

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //This could use a bit mask to only store the relavent fields

    ldaw t, sp[S_VLSAT_0]
    {ldw t, sp[S_VLSAT]; mov s, t}
    { stw s, sp[S_VLSAT]; shl s, t, 16}
    { add t, t, s; nop}
    stw t, sp[S_VLSAT_0]
    stw t, sp[S_VLSAT_1]
    stw t, sp[S_VLSAT_2]
    stw t, sp[S_VLSAT_3]
    stw t, sp[S_VLSAT_4]
    stw t, sp[S_VLSAT_5]
    stw t, sp[S_VLSAT_6]
    stw t, sp[S_VLSAT_7]

    ldaw t, sp[S_BIAS_SHIFT_0]
    {ldw t, sp[S_BIAS_MULTIPLIER]; mov s, t}
    { stw s, sp[S_BIAS_MULTIPLIER]; shl s, t, 16}
    { add t, t, s; nop}
    stw t, sp[S_BIAS_SHIFT_0]
    stw t, sp[S_BIAS_SHIFT_1]
    stw t, sp[S_BIAS_SHIFT_2]
    stw t, sp[S_BIAS_SHIFT_3]
    stw t, sp[S_BIAS_SHIFT_4]
    stw t, sp[S_BIAS_SHIFT_5]
    stw t, sp[S_BIAS_SHIFT_6]
    stw t, sp[S_BIAS_SHIFT_7]

    ldaw t, sp[S_FINAL_SHR_0]
    {ldw t, sp[S_FINAL_SHR]; mov s, t}
    { stw s, sp[S_FINAL_SHR]; shl s, t, 16}
    { add t, t, s; nop}
    stw t, sp[S_FINAL_SHR_0]
    stw t, sp[S_FINAL_SHR_1]
    stw t, sp[S_FINAL_SHR_2]
    stw t, sp[S_FINAL_SHR_3]
    stw t, sp[S_FINAL_SHR_4]
    stw t, sp[S_FINAL_SHR_5]
    stw t, sp[S_FINAL_SHR_6]
    stw t, sp[S_FINAL_SHR_7]

    bnn_h_loop:

        {ldw x_width_loop, sp[S_X_WIDTH_LOOP_COUNTER]; nop}
        bnn_w_loop:

            ldd K_p, oc_loop_counter, sp[S_OC_LOOP_COUNTER_K_p]

            // Reset the current mul and bias 
            ldd t, s, sp[S_PA_MUL_p_PA_BIAS_p]
            std t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p]

            bnn_oc_loop: // This loops over (output_channels/16)
            
                {ldw X_p, sp[S_X_p]; }
                {bl bnn_make_16_channel_output ; nop }

                //TODO clamping would go here

                ldd s, t, sp[S_VLSAT_ASHR]
                {vlsat t[0]; ldaw t, sp[S_TEMP]}

                {vstr t[0]; nop}
                vlashr t[0], s

                {vstr t[0]; nop}
                {vclrdr; nop}

                //Scale and bias
                ldd t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p] //s in mul, t is bias

                {vldc t[0]; ldaw r, sp[S_BIAS_SHIFT_0]}
                {vlmacc r[0]; ldaw r, sp[S_TEMP]}
                {vldc r[0]; add t, t, r_256_bits}
                {vlmacc s[0]; add s, s, r_256_bits}

                std t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p]

                ldw t, sp[S_FINAL_SHR]
                {vlsat t[0]; ldaw s, sp[S_TEMP]}

                ldaw t, cp[VECT_INT16_MAX]
                {vstr s[0]; nop}
                {vladd t[0]; nop}
                {vdepth1; mkmsk t, 16}

                vstrpv Y_p[0], t //let's use Y_p as we have it avaliable and we're going to write to it later anyway

                ldaw t, cp[VECT_INT8_MIN]
                {ldw s, Y_p[0]; nop}
                {vldr t[0]; mkmsk t, 16}
                vstrpv Y_p[0], t
///
                {ldaw t, sp[S_TEMP]}
                {vldr t[0]; mkmsk t, 16}
                {vdepth8; andnot t, s}
                vstrpv Y_p[0], t

                ldaw Y_p, Y_p[4]

            {bt oc_loop_counter, bnn_oc_loop; sub oc_loop_counter, oc_loop_counter, 1}

            //advance X by one(STEP) to the right
            ldd X_p, t, sp[S_X_p_S_OUTER_X_H_STEP]
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


    ldw r4, sp[S_SAVE_R4]
    ldw r5, sp[S_SAVE_R5]
    ldw r6, sp[S_SAVE_R6]
    ldw r7, sp[S_SAVE_R7]
    ldw r8, sp[S_SAVE_R8]
    ldw r9, sp[S_SAVE_R9]
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



