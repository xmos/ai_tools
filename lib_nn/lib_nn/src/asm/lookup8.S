
#if defined(__XS3A__)


/*
void lookup8_asm(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length);
*/

#define FUNCTION_NAME lookup8_asm

#define NSTACKWORDS  10
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME



#define Y       r0
#define X       r1
#define LUT     r2
#define n       r3

.align 4;

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    stw r10, sp[8]

    #define data    r4
    #define i       r5
    #define k       r6
    
#define BLOCK_COUNT_LOG2 (4)
#define BLOCK_SIZE  (3)
#define BLOCK   {                               ;   ld8u data, X[n]             };       \
                {   sub n, n, 1                 ;   ld8u data, LUT[data]        };       \
                st8 data, Y[n];

    {   sub Y, Y, 1                     ;   sub X, X, 1                     }
    {   ldc i, (1<<BLOCK_COUNT_LOG2)    ;   mov r11, n                      }
    {   zext r11, BLOCK_COUNT_LOG2      ;   shr k, n, BLOCK_COUNT_LOG2      }
    {   sub r11, i, r11                 ;   ldc i, BLOCK_SIZE               }
    mul i, r11, i
    {   add Y, Y, 1                     ;   bru i                           }

    .align 4
    .L_loop_top:


        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        BLOCK
        {                               ;   bt n, .L_loop_top           }
    



.Lfunc_end:
    ldw r10, sp[8]
    ldd r8, r9, sp[3]
    ldd r6, r7, sp[2]
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



