
#if defined(__XS3A__)


/*
Source and dest must be word-aligned to use this.

void vpu_memcpy(
    void* dst,
    void* src,
    size_t size);
*/

#define FUNCTION_NAME vpu_memcpy

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


.align 4


FUNCTION_NAME:
    dualentsp NSTACKWORDS

    {   mkmsk r3, 5             ;   mov r11, r1             }
    {   and r3, r2, r3          ;   shr r2, r2, 5           }
    {   ldc r1, 32              ;   sub r2, r2, 1           }

    .L_loop_top:

        {   add r11, r11, r1        ;   vldr r11[0]             }
        {   add r0, r0, r1          ;   vstr r0[0]              }
        {   sub r2, r2, 1           ;   bt r2, .L_loop_top      }

    {   mkmsk r3, r3            ;   vldr r11[0]             }
    vstrpv r0[0], r3


.Lfunc_end:
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



