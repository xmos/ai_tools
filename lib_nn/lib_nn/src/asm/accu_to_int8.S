#if defined(__XS3A__)

.issue_mode  dual

//Registers

#define Y_p               r0
#define X_p               r1
#define K_p               r2

#define s                 r10
#define t                 r11

#define S_VLSAT 0
#define S_ASHR_P 0 //probably constant for all output channels
#define S_BIAS_P 0 //advances per output channel batch(16)
#define S_MUL_P 0  //advances per output channel batch(16)
#define S_BIAS_SHIFT_P 0 //probably constant for all output channels
#define S_TEMP_P 0 //some temp memory to hold a vector


// This will probably end up as a macro in order to allow the caller 
// to define the registers
accu_to_int8:

    ldd s, t, sp[S_VLSAT_ASHR]
    {vlsat t[0]; ldaw t, sp[S_TEMP]}

    {vstr t[0]; nop}
    vlashr t[0], s
    {vstr t[0]; nop}

    {vclrdr; nop}
    ldw s, sp[S_BIAS_SHIFT_P]
    vldc s[0]

    ldd t, s, sp[S_BIAS_P_MUL_P]
    {vlmacc s[0] ; add t, t, r256bits }   // multiply BIAS * BIAS_SHIFT to align it correctly

    {vldc t[0]; add t, t, r256bits }      //load multiplier into C
    std s, t, sp[S_BIAS_P_MUL_P]

    ldaw t, sp[S_TEMP]
    vlmacc t[0] // multiply Accu by multiplier and add it to the shifted bias

    //now reduce it to an int8



retsp


/*
void test_accu_to_int8(const test_accu_to_int8_t* state);
*/

#define FUNCTION_NAME test_accu_to_int8

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



