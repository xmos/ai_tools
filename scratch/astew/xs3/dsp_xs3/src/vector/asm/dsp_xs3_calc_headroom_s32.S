
#if defined(__XS3A__)


/*  
unsigned dsp_xs3_calc_headroom_s32_asm(
    const int32_t* A,
    const unsigned length);
*/

#define FUNCTION_NAME	dsp_xs3_calc_headroom_s32_asm

#define NSTACKWORDS     8

#define NOOOOOOOOP      nop
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define A           r0
#define length		r1

#define vR_tmp      sp[0]

/**
 * Timing analysis (instruction fetches not included):
 * 
 *  J = (length / 16)            ---   J  >=    0
 *  K = (length / 8) & 0x01      ---   K  in   {0,1}
 *  L = (length % 8) > 0         ---   L  in   {0,1}
 * 
 *  [Total instructions] = 5*J + 2*K + 7*L + 11
 * 
 * 
 * def dsp_xs3_calc_headroom_s32_asm_timing(N):
 *     J = (N // 16)
 *     K = (N // 8) & 0x01
 *     L = (N % 8) > 0
 *     return 5*J + 2*K + 7*L + 11
 */

FUNCTION_NAME:

//Part 1:   Handle the first (length%8) elements. Rather than handle each individually,
//          this will handle them all at once using the VPU to do a partial store.
//          This should be cheaper if (length%8 is greater than 2 or 3) 
    {	dualentsp NSTACKWORDS		;	mkmsk r3, 3	            }
    {   ldc r11, 0                  ;   and r3, length, r3      }
    {   vsetc r11                   ;   shr length, length, 3   }
    {   bf r3, .L_part2             ;   shl r3, r3, 2           }
    


    {   vclrdr                      ;   ldaw r11, vR_tmp        }
    {   vstr r11[0]                 ;   mov r11, A              }
    {   vldr r11[0]                 ;   mkmsk r2, r3            }
    {   NOOOOOOOOP                  ;   ldaw r11, vR_tmp        }
    vstrpv  r11[0], r2                                                     //<-- doesn't update vHR
    {   vldr    r11[0]              ;   NOOOOOOOOP              }
    {   vstr    r11[0]              ;   add A, A, r3            }


//Part 2: Perform (at most) 1 single vector VPU load/store, so that
//        remaining load/store can be done 2 at a time.
.L_part2:
    {   mkmsk r3, 1                 ;   ldc r2, 32              }
    {   and r3, length, r3          ;   shr length, length, 1   }
    {   bf r3, .L_part3             ;   NOOOOOOOOP              }
    {   vldd A[0]                   ;   NOOOOOOOOP              }
    {   vstd A[0]                   ;   add A, A, r2            }

//Part 3: Handle the remainder using pairs of loads and stores
.L_part3:
    {   bf length, .L_loop_end      ;   NOOOOOOOOP              }
.L_loop_top:
    {   vldd A[0]                   ;   sub length, length, 1   }       //I'm told this will achieve just about peak efficiency with the VPU
    {   vstd A[0]                   ;   add A, A, r2            }       // as we can only do about 4 instructions before having to perform an
    {   vldd A[0]                   ;   NOOOOOOOOP              }       // instruction fetch
    {   vstd A[0]                   ;   add A, A, r2            }
    {   bt length, .L_loop_top      ;   NOOOOOOOOP              }   
.L_loop_end:
    {   vgetc r11                   ;   mkmsk r2, 6             }
    {   and r11, r11, r2            ;   ldc r0, 31              }
.Lfunc_end:
    {   retsp NSTACKWORDS           ;   sub r0, r0, r11         }

    .cc_bottom FUNCTION_NAME.function
    .set	FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl	FUNCTION_NAME.nstackwords
    .set	FUNCTION_NAME.maxcores,1
    .globl	FUNCTION_NAME.maxcores
    .set	FUNCTION_NAME.maxtimers,0
    .globl	FUNCTION_NAME.maxtimers
    .set	FUNCTION_NAME.maxchanends,0
    .globl	FUNCTION_NAME.maxchanends
.Ltmp0:
    .size	FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif  //__XS3A__



