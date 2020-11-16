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
.text

.section        .cp.const32,"aMc",@progbits,32
.cc_top VECT_ZEROS.data
.align 4
VECT_ZEROS:
.word    0x00000000
.word    0x00000000
.word    0x00000000
.word    0x00000000
.word    0x00000000
.word    0x00000000
.word    0x00000000
.word    0x00000000
.cc_bottom VECT_ZEROS.data
.text

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
#define PLAN_INNER_X_H_STEP        0
#define PLAN_DATA_SCRATCH_ADJUST   1
#define PLAN_K_HEIGHT_LOOP_COUNTER 2
#define PLAN_DATA_SCRATCH          3
#define PLAN_K_WIDTH_LOOP_COUNTER  4
#define PLAN_INNER_X_V_STEP        5
#define PLAN_OUTER_X_V_STEP        6
#define PLAN_Y_V_STEP              7

#define PLAN_OC_LOOP_COUNTER       8
#define PLAN_K_p                   9
#define PLAN_CUR_PA_MUL_p          10
#define PLAN_CUR_PA_BIAS_p         11
#define PLAN_VLSAT                 12
#define PLAN_ASHR                  13
#define PLAN_PA_MUL_p              14
#define PLAN_PA_BIAS_p             15

#define PLAN_IC_LOOP_COUNTER       16
#define PLAN_Y_p                   17
#define PLAN_X_p                   18
#define PLAN_OUTER_X_H_STEP        19
#define PLAN_FINAL_CHAN_MASK       20
#define PLAN_FINAL_CHAN_BYTES      21
#define PLAN_PATCH_LOOP_COUNTER    22
#define PLAN_FINAL_SHR             23

#define PLAN_K_ADJUST              24
#define PLAN_K_CHAN_REWIND         25
#define PLAN_X_WIDTH_LOOP_COUNTER  26
#define PLAN_X_HEIGHT_LOOP_COUNTER 27  
#define PLAN_BIAS_MULTIPLIER       28  


//Stack Usage

#define S_0                     0
#define S_1                     1

#define S_INNER_X_H_STEP        2
#define S_DATA_SCRATCH_ADJUST   3

#define S_K_HEIGHT_LOOP_COUNTER 4
#define S_DATA_SCRATCH          5

#define S_K_WIDTH_LOOP_COUNTER  6
#define S_INNER_X_V_STEP        7

#define S_OUTER_X_V_STEP        8
#define S_Y_V_STEP              9 

#define S_OC_LOOP_COUNTER       10
#define S_K_p                   11

#define S_CUR_PA_MUL_p          12
#define S_CUR_PA_BIAS_p         13

#define S_VLSAT                 14
#define S_ASHR                  15

#define S_PA_MUL_p              16
#define S_PA_BIAS_p             17

#define S_IC_LOOP_COUNTER       18
#define S_Y_p                   19

#define S_X_p                   20
#define S_OUTER_X_H_STEP        21

#define S_K_ADJUST              22
#define S_PATCH_BRANCH          23

#define S_FINAL_CHAN_MASK       24
#define S_FINAL_CHAN_BYTES      25
#define S_PATCH_LOOP_COUNTER    26
#define S_FINAL_SHR             27
#define S_K_CHAN_REWIND         28
#define S_X_WIDTH_LOOP_COUNTER  29
#define S_X_HEIGHT_LOOP_COUNTER 30 
#define S_BIAS_MULTIPLIER       31  


//doesn't have to be double work aligned
#define S_TEMP_0                32
#define S_TEMP_1                33 
#define S_TEMP_2                34
#define S_TEMP_3                35
#define S_TEMP_4                36
#define S_TEMP_5                37
#define S_TEMP_6                38
#define S_TEMP_7                39

//doesn't have to be double work aligned
#define S_VLSAT_0               40
#define S_VLSAT_1               41 
#define S_VLSAT_2               42
#define S_VLSAT_3               43
#define S_VLSAT_4               44
#define S_VLSAT_5               45
#define S_VLSAT_6               46
#define S_VLSAT_7               47

//doesn't have to be double word aligned
#define S_BIAS_SHIFT_0          48
#define S_BIAS_SHIFT_1          49 
#define S_BIAS_SHIFT_2          50
#define S_BIAS_SHIFT_3          51
#define S_BIAS_SHIFT_4          52
#define S_BIAS_SHIFT_5          53
#define S_BIAS_SHIFT_6          54
#define S_BIAS_SHIFT_7          55

//doesn't have to be double word aligned
#define S_FINAL_SHR_0           56
#define S_FINAL_SHR_1           57 
#define S_FINAL_SHR_2           58
#define S_FINAL_SHR_3           59
#define S_FINAL_SHR_4           60
#define S_FINAL_SHR_5           61
#define S_FINAL_SHR_6           62
#define S_FINAL_SHR_7           63

#define S_SAVE_R10              64
#define S_SAVE_R4               65
#define S_SAVE_R5               66
#define S_SAVE_R6               67
#define S_SAVE_R7               68
#define S_SAVE_R8               69
#define S_SAVE_R9               70

#define S_T                     71


#define NSTACKWORDS  90 //TODO


//Helpers that reference the above
#define S_INNER_X_H_STEP_DATA_SCRATCH_ADJUST    (S_INNER_X_H_STEP/2)
#define S_K_HEIGHT_LOOP_COUNTER_DATA_SCRATCH    (S_K_HEIGHT_LOOP_COUNTER/2)
#define S_K_WIDTH_LOOP_COUNTER_INNER_X_V_STEP   (S_K_WIDTH_LOOP_COUNTER/2)
#define S_OC_LOOP_COUNTER_K_p                   (S_OC_LOOP_COUNTER/2)
#define S_CUR_PA_MUL_p_CUR_PA_BIAS_p            (S_CUR_PA_MUL_p/2)
#define S_VLSAT_ASHR                            (S_VLSAT/2)                                //9
#define S_PA_MUL_p_PA_BIAS_p                    (S_PA_MUL_p/2)   
#define S_IC_LOOP_COUNTER_Y_p                   (S_IC_LOOP_COUNTER/2)
#define S_X_p_OUTER_X_H_STEP                    (S_X_p/2)
#define S_OUTER_X_V_STEP_Y_V_STEP               (S_OUTER_X_V_STEP/2)
#define S_TEMP S_TEMP_0

//Registers

#define Y_p               r0
#define X_p               r1
#define K_p               r2
#define D_p               r3 

#define k_height_loop     r4
#define k_width_loop      r5
#define ic_loop_counter   r6
#define oc_loop_counter   r7
#define r_256_bits        r8
#define x_width_loop      r9
#define s                 r10
#define t                 r11

#define r X_p //r can be used when X_p is not in use


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

    {vclrdr; nop}

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
        {vlmaccr1 K_p[0] ; add K_p, K_p, s}; 
    {bt t, bnn_patch_loop; sub t, t, 1}

    bnn_patch_tail: 

    //convert s to the tail loop count
    {vldc X_p[0];ldc t, 15 - 1 }      
    ashr s, s, 5
    {ldw t, sp[S_K_ADJUST]; add s, t, s}
    {ashr s, s, 2}

    bnn_tail_loop:

        {vlmaccr1 K_p[0] ; add K_p, K_p, t}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, t}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, t}; 
        {vlmaccr1 K_p[0] ; add K_p, K_p, t}; 
        {bt s, bnn_tail_loop; sub s, s, 1}

    //TODO clamping would go here


    ldd s, t, sp[S_VLSAT_ASHR]
    {vlsat t[0]; ldaw t, sp[S_TEMP]}

    {vstr t[0]; nop}
    vlashr t[0], s

    {vstr t[0]; nop}
    {vclrdr; nop}

    //Scale and bias
    ldd t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p]

    {vldc t[0]; ldaw r, sp[S_BIAS_SHIFT_0]}

    {vlmacc r[0]; ldaw r, sp[S_TEMP]}
    {vldc r[0]; add t, t, r_256_bits}
    {vlmacc s[0]; add s, s, r_256_bits}

    std t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p]
            
    ldw t, sp[S_FINAL_SHR]
    {vlsat t[0]; ldaw s, sp[S_TEMP]}

    {nop; ldaw t, cp[VECT_INT16_MAX]}
    {vstr s[0]; nop}
    {vladd t[0]; nop}
    {vdepth1; mkmsk t, 4}

    vstrpv Y_p[0], t //let's use Y_p as we have it avaliable and we're going to write to it later anyway

    {ldaw t, cp[VECT_INT8_MIN]}
    {ldw s, Y_p[0]; nop}

    {retsp 0; nop}

/*
void bnn_conv2d_int8_out_SISO_asm(const nn_bnn_conv2d_int8_out_asm_plan_t* plan);
*/

#define FUNCTION_NAME bnn_conv2d_int8_out_SISO_asm
    
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
    
    {mov t, r0;  ldc s, PLAN_Y_p}
    {ldw r0, t[s]; ldc s, PLAN_X_p} //put PLAN_Y_P into r0
    {ldw r1, t[s]; ldc s, PLAN_K_p}
    {ldw r2, t[s]; nop}

    {vldr t[0];  add t, t, r_256_bits}
    ldaw s, sp[S_INNER_X_H_STEP] //this is where we will start the plan copy from
    {vstr s[0]; add s, s, r_256_bits}

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //second copy

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //third copy 

    {vldr t[0]; add t, t, r_256_bits}
    {vstr s[0]; add s, s, r_256_bits} //forth copy 
    //half of the above copy will have gone over S_TEMP but it wont matter

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

            {ldw X_p, sp[S_X_p]; } //this could be hoisted
            {bl bnn_make_patch ; nop}

            //init K_p
            ldd K_p, oc_loop_counter, sp[S_OC_LOOP_COUNTER_K_p]

            ldd t, s, sp[S_PA_MUL_p_PA_BIAS_p]
            std t, s, sp[S_CUR_PA_MUL_p_CUR_PA_BIAS_p]

            {bf oc_loop_counter, bnn_final_loop; nop}

            bnn_oc_loop: 
            
                {bl bnn_compute_patch ; mov s, r_256_bits}

                {vldr t[0]; mkmsk t, 16}
                vstrpv Y_p[0], t

                {ldaw t, sp[S_TEMP]}
                {vldr t[0]; mkmsk t, 16}
                {vdepth8; andnot t, s}
                vstrpv Y_p[0], t

                ldaw Y_p, Y_p[4]

            {bt oc_loop_counter, bnn_oc_loop; sub oc_loop_counter, oc_loop_counter, 1}

            bnn_final_loop:

            ldw s, sp[S_K_CHAN_REWIND]
            {bl bnn_compute_patch ; nop }

            {vldr t[0]; nop}
            {ldw t, sp[S_FINAL_CHAN_MASK]}
            vstrpv Y_p[0], t

            {ldaw t, sp[S_TEMP]}
            {vldr t[0]; nop}
            {ldw t, sp[S_FINAL_CHAN_MASK]}

            {vdepth8; andnot t, s}
            
            vstrpv Y_p[0], t
            {ldw t, sp[S_FINAL_CHAN_BYTES]}
            {add Y_p, Y_p, t ; nop}

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



