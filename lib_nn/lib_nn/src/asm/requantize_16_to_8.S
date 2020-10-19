
#if defined(__XS3A__)

#include "nn_config.h"

/*
void requantize_16_to_8(
    int8_t* Y,
    const int16_t* X,
    const unsigned elm_start,
    const unsigned elm_count);
*/

#define FUNCTION_NAME requantize_16_to_8

#define NSTACKWORDS  (4+8)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_VEC_TMP   NSTACKWORDS-8

#define arg_x       r1
#define arg_start   r2

#define Y               r0
#define _32             r1
#define tail_mask       r2
#define length          r3
#define MASK16          r4
#define _16             r5
#define tmp             r6


#if !CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
  #define X           r7
#else
  #define X         r11
#endif


/*
    To allow the possibility of outputing 0x80 (-128), we have to do a little magic.

    The main problem is that the two primary ways we have of collapsing 16-bit values down to 
    8-bit values (VLSAT and VDEPTH8) both saturate values to -127 instead of -128. We need
    to somehow get the outputs into the lower 16 bytes of the vector, but still as part of
    32-bit accumulators, so that the saturation doesn't occur.

    This ultimately requires combining outputs (2k) and (2k+1) into the *same* 16-bit value,
    so that the 32-bit accumulator in 16-bit mode can be used.

    So, the procedure will be to first determine which outputs should be -128 and create 
    a correction vector with that info.

    Then we do vdepth8 on the inputs as usual, and just apply the correction vector. The
    correction vector consists of 16 8-bit values which are 1 if output should be -128 and
    0 otherwise.

    When applying the correction vector, we'll have 8-bit values in 16-bit mode. So, if we
    make sure that vD is always 0, then the 16-bit values in vR will all be interepreted
    as positive (i.e. unsigned).

    Note that the value being subtracted when applying the adjustment will be one of the
    following 4 values:  0x0000, 0x0001, 0x0100, or 0x0101

        0x0000XXXX - 0x0000 -> 0x0000XXXX
        0x0000XX81 - 0x0001 -> 0x0000XX80
        0x000081XX - 0x0100 -> 0x000080XX
        0x00008181 - 0x0101 -> 0x00008080

        As long as XX is not 0x80 (-128) the logic works. And it can't be 0x80, because
        then we wouldn't have this problem.
    

*/

.L_constA:
.short -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
.L_constB:
.short -0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01,-0x7F01

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]


    {   ldc _16, 16                             ;   add X, arg_x, arg_start                 }
    {   add Y, Y, arg_start                     ;   ldc _32, 32                             }
    {   add X, X, arg_start                     ;   shl r11, _32, 3                         }
    {   mov tail_mask, length                   ;   vsetc r11                               }
    {   zext tail_mask, 4                       ;   shr length, length, 4                   }
    {   mkmsk tail_mask, tail_mask              ;   mkmsk MASK16, 16                        }
    {                                           ;   bf length, .L_loop_end                  }


#if !CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
    {   ldap r11, .L_constA                     ;   vclrdr                                  }
    {   mov r11, X                              ;   vldc r11[0]                             }

    .L_loop_top:

        {   ldap r11, .L_constB                     ;   vldr r11[0]                             }
        {                                           ;   vlsub r11[0]                            }
        {                                           ;   vpos                                    }
        {   ldaw tmp, sp[STACK_VEC_TMP]             ;   vdepth8                                 }
        {   mov r11, X                              ;   vstr tmp[0]                             }
        {   add X, X, _32                           ;   vldr r11[0]                             }
        {   mov r11, X                              ;   vdepth8                                 }
        {   sub length, length, 1                   ;   vlmacc tmp[0]                           }
        vstrpv Y[0], MASK16
        {   add Y, Y, _16                           ;   bt length, .L_loop_top                  }

    .L_loop_end:

        {   mov r11, X                              ;   bf tail_mask, .Lfunc_end                }
        {   ldap r11, .L_constB                     ;   vldr r11[0]                             }
        {                                           ;   vlsub r11[0]                            }
        {                                           ;   vpos                                    }
        {   ldaw tmp, sp[STACK_VEC_TMP]             ;   vdepth8                                 }
        {   mov r11, X                              ;   vstr tmp[0]                             }
        {                                           ;   vldr r11[0]                             }
        {                                           ;   vdepth8                                 }
        {                                           ;   vlmacc tmp[0]                           }
        vstrpv Y[0], tail_mask

#else  // CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8

    .L_loop_top:
        {   add X, X, _32                           ;   vldr X[0]                               }
        {   sub length, length, 1                   ;   vdepth8                                 }
        vstrpv Y[0], MASK16
        {   add Y, Y, _16                           ;   bt N_vec, .L_loop_top                   }
    .L_loop_end:

        {                                           ;   bf tail_mask, .Lfunc_end                }
        {                                           ;   vldr X[0]                               }
        {                                           ;   vdepth8                                 }
        vstrpv Y[0], tail_mask

#endif // CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8


    
.Lfunc_end:
    ldd r6, r7, sp[1]
    ldd r4, r5, sp[0]
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



