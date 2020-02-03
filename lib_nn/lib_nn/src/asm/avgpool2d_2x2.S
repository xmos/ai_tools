
#if defined(__XS3A__)


/*  
void avgpool2d_2x2_asm(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans);
*/

#define FUNCTION_NAME avgpool2d_2x2_asm

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

#define x_vstride       r2
#define chan_grp_left   r3
#define chans           r4
#define rows_left       r5
#define cols_left       r6
#define x_row_incr      r7
#define x_chan_incr     r8
#define y_chan_incr     r9
#define write_mask      r10


#define STACK_X_CHANS       (NSTACKWORDS+1)
#define STACK_Y_ROWS        9
#define STACK_Y_COLS        10
#define STACK_TAIL_LEN      11

.align 4
.L_macc_const: //multiplies input in VLMACC instruction
.byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

.align 4
.L_sat_vec:  //used in VLSAT instruction
.short 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2



.align 4
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc r10, 16                             ;   stw r10, sp[8]                          }
    {   shl r11, r10, 5                         ;   ldw chans, sp[STACK_X_CHANS]            }
    {   mov r11, arg_x_height                   ;   vsetc r11                               }
    
    mul x_vstride, arg_x_width, chans
    mul x_chan_incr, x_vstride, r11
    {   shr arg_x_width, arg_x_width, 1         ;   shr r11, r11, 1                         }
    {   shr y_chan_incr, x_chan_incr, 2         ;   stw r11, sp[STACK_Y_ROWS]               }
    {   sub x_chan_incr, r10, x_chan_incr       ;   stw arg_x_width, sp[STACK_Y_COLS]       }
    {   sub y_chan_incr, r10, y_chan_incr       ;   ldap r11, .L_macc_const                 }
    {   sub x_row_incr, x_vstride, chans        ;   vldc r11[0]                             }    
    {   mkmsk write_mask, 4                     ;                                           }
    {   and r3, chans, write_mask               ;   ldc r11, 16                             }
    {                                           ;   sub r11, r11, r3                        }
    {   mkmsk write_mask, r3                    ;   stw r11, sp[STACK_TAIL_LEN]             }
    {   ldc r3, 1                               ;   bt r3, .L_chan_loop                     }

    {   mkmsk write_mask, 16                    ;   shr chan_grp_left, chans, 4             }

    .L_chan_loop:
        {   sub chan_grp_left, chan_grp_left, 1     ;   ldw rows_left, sp[STACK_Y_ROWS]         }
        .L_row_loop:
            {   sub rows_left, rows_left, 1        ;   ldw cols_left, sp[STACK_Y_COLS]          }
            .L_col_loop:
                {   ldap r11, .L_sat_vec                ;   vclrdr                                  }
                {   add X, X, chans                     ;   vlmacc X[0]                             }
                {   add X, X, x_row_incr                ;   vlmacc X[0]                             }
                {   add X, X, chans                     ;   vlmacc X[0]                             }
                {   sub X, X, x_row_incr                ;   vlmacc X[0]                             }
                {   sub cols_left, cols_left, 1         ;   vlsat r11[0]                            }
                vstrpv Y[0], write_mask
                {   add Y, Y, chans                     ;   bt cols_left, .L_col_loop               }
            {   add X, X, x_vstride                 ;   bt rows_left, .L_row_loop               }
        {   add X, X, x_chan_incr               ;                                           }
        {   add Y, Y, y_chan_incr               ;   bt chan_grp_left, .L_chan_loop          }

    .L_do_body:

    {   mkmsk r3, 16                            ;   ldw r11, sp[STACK_TAIL_LEN]         }
    {   eq r3, write_mask, r3                   ;   sub X, X, r11                       }
    {   sub Y, Y, r11                           ;   bt r3, .L_img_end                   }
    {   mkmsk write_mask, 16                    ;   shr chan_grp_left, chans, 4         }    
    {                                           ;   bt chan_grp_left, .L_chan_loop      }


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



