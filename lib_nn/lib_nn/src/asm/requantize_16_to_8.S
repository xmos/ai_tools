
#if defined(__XS3A__)


/*
void requantize_16_to_8(
    int8_t* y,
    const int16_t* x,
    const unsigned n);
*/

#define FUNCTION_NAME requantize_16_to_8

#define NSTACKWORDS  4
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


.align 4

#define arg_y   r0
#define arg_x   r1
#define arg_n   r2

#define Y           r0
#define VBYTES      r1
#define N_vec       r2
#define tail_mask   r3
#define MASK16      r4
#define EPV         r5
#define X           r11

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[1]

    {   ldc r11, 32             ;   mkmsk r3, 4             }
    {   shl r11, r11, 3         ;   and r3, arg_n, r3       }
    {   mov X, arg_x            ;   vsetc r11               }
    {   mkmsk tail_mask, r3     ;   shr N_vec, arg_n, 4     }
    {   mkmsk MASK16, 16        ;   bf N_vec, .L_loop_end   }
    {   ldc VBYTES, 32          ;   ldc EPV, 16             }
    .align 4
    .L_loop_top:
        {   add X, X, VBYTES        ;   vldr X[0]               }
        {   sub N_vec, N_vec, 1     ;   vdepth8                 }
        vstrpv Y[0], MASK16
        {   add Y, Y, EPV           ;   bt N_vec, .L_loop_top   }
    .L_loop_end:

    {                           ;   bf tail_mask, .Lfunc_end}
    {                           ;   vldr X[0]               }
    {                           ;   vdepth8                 }
    vstrpv Y[0], tail_mask
    
    // {                           ;                           }

.Lfunc_end:
    ldd r4, r5, sp[1]
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



