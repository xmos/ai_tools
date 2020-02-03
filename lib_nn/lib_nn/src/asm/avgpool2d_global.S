
#if defined(__XS3A__)


/*  
void avgpool2d_global_asm(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const uint32_t shift
    const uint32_t scale);
*/

#define FUNCTION_NAME avgpool2d_global_asm

#define NSTACKWORDS  20
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_Y           r0
#define arg_X           r1
#define arg_x_height    r2
#define arg_x_width     r3

#define Y               r0
#define X               r1

#define pix_groups      r2
#define chans           r3
#define chan_grp_left   r4
#define x_chan_incr     r5
#define write_mask      r6
#define chan_tail       r7
#define pix_loop_hop    r8
#define pix_count       r9
#define _16             r10

#define STACK_X_CHANS       (NSTACKWORDS+1)
#define STACK_SHIFT         (NSTACKWORDS+2)
#define STACK_SCALE         (NSTACKWORDS+3)
#define STACK_TMP_VEC       (NSTACKWORDS-8)

.align 4
.L_shift_const:
.word 0x00010001
.L_scale_const: 
.word 0x01010101

.align 4
.L_sat_vec:  //used in VLSAT instruction
.short 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2



.align 4
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc _16, 16                             ;   stw r10, sp[8]                          }
    mul pix_count, arg_x_height, arg_x_width
    {   shl r11, _16, 5                         ;   ldw chans, sp[STACK_X_CHANS]            }
    {   shr chan_grp_left, chans, 4             ;   vsetc r11                               }

    {   ldaw r6, sp[STACK_TMP_VEC]              ;   ldw r11, sp[STACK_SCALE]                }
    std r11, r11, r6[0]
    std r11, r11, r6[1]
    std r11, r11, r6[2]
    std r11, r11, r6[3]
    {                                           ;   vldc r6[0]                              }
    {                                           ;   ldw r11, sp[STACK_SHIFT]                }
    std r11, r11, r6[0]
    std r11, r11, r6[1]
    std r11, r11, r6[2]
    std r11, r11, r6[3]

    mul x_chan_incr, pix_count, chans
    {   sub x_chan_incr, x_chan_incr, _16       ;                                           }
    {   mkmsk write_mask, 16                    ;                                           }

    {   mkmsk r11, 4                            ;                                           }
    {   and chan_tail, chans, r11               ;   and pix_loop_hop, pix_count, r11        }
    
    {   ldaw r11, sp[STACK_TMP_VEC]             ;   sub pix_loop_hop, _16, pix_loop_hop     }
    {                                           ;   bf chan_grp_left, .L_chan_loop_end      }

    //TODO: HAVE TO VLDC!!!
     
    .align 4
    .L_chan_loop:
        {   shr pix_groups, pix_count, 4            ;   vclrdr                                  }
        {   sub chan_grp_left, chan_grp_left, 1     ;   bru pix_loop_hop                        } // Do tail first
        .L_pix_loop:
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 0  -- should be a no-op
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 10
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 15
            {   sub pix_groups, pix_groups, 1           ;   bt pix_groups, .L_pix_loop              }   // 16
        .L_pix_loop_end:        

        {   sub X, X, x_chan_incr                   ;   vlsat r11[0]                            }
        vstrpv Y[0], write_mask
        {   add Y, Y, _16                           ;   bt chan_grp_left, .L_chan_loop          }

    .L_chan_loop_end:
    
    {   mkmsk r11, 16                           ;   ldc chan_grp_left, 1                }
    {   eq r11, write_mask, r11                 ;                                       }
    {   mkmsk write_mask, chan_tail             ;   bf r11, .L_img_end                  }
    {   ldaw r11, sp[STACK_TMP_VEC]             ;   bt write_mask, .L_chan_loop         }


.L_img_end:

.Lfunc_end:
    {                                           ;   ldw r10, sp[8]                      }
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



