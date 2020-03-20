
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


#define dst         r0
#define src         r11
#define len         r2
#define _32         r1
#define tail        r3

.align 4
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    {   mkmsk r3, 5                 ;   mov src, r1                 }
    {   and tail, len , r3          ;   shr len , len , 5           }
    {   ldc _32, 32                 ;   bf len , .L_loop_end        }

    .L_loop_top:
        {   sub len , len , 1           ;   vldr src[0]                 }
        {   add src, src, _32           ;   vstr dst[0]                 }
        {   add dst, dst, _32           ;   bt len , .L_loop_top        }
//      {       FNOP        }

    .L_loop_end:
    {   mkmsk tail, tail            ;   vldr src[0]                 }
    vstrpv dst[0], tail


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



