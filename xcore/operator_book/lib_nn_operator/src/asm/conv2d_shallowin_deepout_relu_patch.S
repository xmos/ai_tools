
#if defined(__XS3A__)


/*  
(returns next Y position to be written to)
int8_t* conv2d_shallowin_deepout_relu_asm_patch(
    int8_t* y,
    const int8_t* K,
    const unsigned c_out_groups,
    const uint32_t pad_mask,
    const int8_t* X,
    const unsigned rows,
    const int16_t bias_lo,
    const int16_t bias_hi,
    const int16_t shifts,
    const int16_t scales,
    const unsigned X_row_incr,
    );
*/

#define FUNCTION_NAME conv2d_shallowin_deepout_relu_asm_patch

#define NSTACKWORDS  (8 + 16)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop
    
#define arg_y               r0
#define arg_k               r1
#define arg_cout            r2
#define arg_pad_mask        r3

#define STACK_X_INITIAL         (NSTACKWORDS+1)
#define STACK_ROWS              (NSTACKWORDS+2)
#define STACK_BIAS_LO           (NSTACKWORDS+3)
#define STACK_BIAS_HI           (NSTACKWORDS+4)
#define STACK_SHIFTS            (NSTACKWORDS+5)
#define STACK_SCALES            (NSTACKWORDS+6)
#define STACK_X_ROW_INCR        (NSTACKWORDS+7)
#define STACK_KERNEL_ADVANCE    (NSTACKWORDS+8)

#define STACK_TMP_VEC1          (NSTACKWORDS-8)
#define STACK_TMP_VEC2          (NSTACKWORDS-16)

#define y                   r0
#define k                   r1
#define cout                r2
#define pad_mask            r3
#define x                   r4
#define row_iter            r5
#define bias_lo             r6
#define bias_hi             r7
#define shifts              r8
#define scales              r9
#define tmp1                r10


//Some constants
.L_zero_vec:
.int 0,0,0,0,0,0,0,0

#define ADD_VAL (-((1<<14)-1))
.L_add_vec:
.short ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL, ADD_VAL

.L_mode16:
.int 0x0100
.L_mode8:
.int 0x0200


FUNCTION_NAME:
    {   NOOOOP                                  ;   dualentsp NSTACKWORDS               }
// we need all the registers
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {   NOOOOP                                  ;   stw r10, sp[6]                      }

//initialize some of our registers
    {   NOOOOP                                  ;   ldw shifts, sp[STACK_SHIFTS]        }
    {   NOOOOP                                  ;   ldw scales, sp[STACK_SCALES]        }
    {   NOOOOP                                  ;   ldw bias_lo, sp[STACK_BIAS_LO]      }
    {   NOOOOP                                  ;   ldw bias_hi, sp[STACK_BIAS_HI]      }

.L_patch_chan_start:
// decrement remaining channels, finish if 0
    {   sub cout, cout, 1                       ;   bf cout, .Lfunc_end                 }
// set VPU mode, reset row_incr
    {   ldap r11, .L_mode8                      ;   ldc tmp1, 32                        }
    {   NOOOOP                                  ;   ldw r11, r11[0]                     }
    {   mov r11, bias_lo                        ;   vsetc r11                           }
// load biases
    {   add bias_lo, bias_lo, tmp1              ;   vldr r11[0]                         }
    {   add bias_hi, bias_hi, tmp1              ;   vldd bias_hi[0]                     }
// reset patch_x for the current channel
    {   NOOOOP                                  ;   ldw x, sp[STACK_X_INITIAL]          }
// reset row iterator for the current channel
    {   NOOOOP                                  ;   ldw row_iter, sp[STACK_ROWS]        }

.L_patch_row_start:
// store accumulator vR in TMP_VEC1
    {   ldaw tmp1, sp[STACK_TMP_VEC1]           ;   NOOOOP                              }
    {   ldap r11, .L_zero_vec                   ;   vstr tmp1[0]                        }
// zero out TMP_VEC2
    {   ldaw tmp1, sp[STACK_TMP_VEC2]           ;   vldr r11[0]                         }
    {   mov r11, x                              ;   vstr tmp1[0]                        }
// load the (entire) row of X into vR
    {   NOOOOP                                  ;   vldr r11[0]                         }
// store this into TMP_VEC2 using our padding mask (so all padding cells are zero)
    vstrpv tmp1[0], pad_mask
// load padded X into vC
    {   ldaw r11, sp[STACK_TMP_VEC1]            ;   vldc tmp1[0]                        }
// increment X and load the stored accumulator into vR
    {   NOOOOP                                  ;   ldw tmp1, sp[STACK_X_ROW_INCR]      }
    {   add x, x, tmp1                          ;   vldr r11[0]                         }

// compute the sum for this row for each of 16 channels
    {   ldc tmp1, 32                            ;   NOOOOP                              }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   vlmaccr k[0]                        }
    {   sub row_iter, row_iter, 1               ;   vlmaccr k[0]                        }
    {   add k, k, tmp1                          ;   bt row_iter, .L_patch_row_start     }

.L_patch_sum_end:
// set VPU mode to 16-bit
    {   ldap r11, .L_mode16                     ;   ldw tmp1, sp[STACK_KERNEL_ADVANCE]  }
    {   NOOOOP                                  ;   ldw r11, r11[0]                     }
    {   add k, k, tmp1                          ;   NOOOOP                              }
    {   ldc tmp1, 32                            ;   vsetc r11                           }
// saturate to 16 bits
    {   add shifts, shifts, tmp1                ;   vlsat shifts[0]                     }
// apply relu
    {   ldap r11, .L_add_vec                    ;   vpos                                }
// add -((2^14)-1)
    {   ldc r11, 16                             ;   vladd r11[0]                        }
// apply scales
    {   add scales, scales, tmp1                ;   vlmul scales[0]                     }
    {   mkmsk tmp1, 16                          ;   vdepth8                             }
    vstrpv y[0], tmp1
.L_chan_end:
// increment kernel to next group and loop to next channel group
    {   add y, y, r11                           ;   bt cout, .L_patch_chan_start        }
    
.L_patch_end:

.Lfunc_end:
    //y is already r0, so this doesn't really matter
    {   mov r0, y                               ;   ldw r10, sp[6]                      }
    ldd r8, r9, sp[2]
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



