
#if defined(__XS3A__)


/*  
(returns next Y position to be written to)
int8_t* conv2d_deepin_deepout_relu_asm_patch(
    const int8_t* y,
    const int8_t* patch_k,
    const int16_t* biases_lo,
    const int16_t* biases_hi,
    const unsigned patch_row_incr,
    const unsigned kernel_row_incr,
    const int8_t* patch_x, 
    const unsigned rows,
    const unsigned row_maccs,          
    const unsigned chan_outs,          
    const unsigned kernel_advance,
    const int16_t* shifts,
    const int16_t* scales);
*/

#define FUNCTION_NAME conv2d_deepin_deepout_relu_asm_patch

#define NSTACKWORDS  (26)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop
    
#define arg_y               r0
#define arg_patch_k         r1
#define arg_bias_lo         r2
#define arg_bias_hi         r3

#define STACK_PATCH_ROW_INCR    (NSTACKWORDS+1)
#define STACK_KERNEL_ROW_INCR   (NSTACKWORDS+2)
#define STACK_X_INITIAL         (NSTACKWORDS+3)
#define STACK_ROWS              (NSTACKWORDS+4)
#define STACK_ROW_MACCS         (NSTACKWORDS+5)
#define STACK_CHAN_OUTS         (NSTACKWORDS+6)
#define STACK_KERNEL_ADVANCE    (NSTACKWORDS+7)
#define STACK_SHIFTS            (NSTACKWORDS+8)
#define STACK_SCALES            (NSTACKWORDS+9)

#define STACK_TMP_VEC           (NSTACKWORDS-8)
#define STACK_MODE8             (7)
#define STACK_MODE16            (8)            

#define y                   r0
#define patch_k             r1
#define bias_lo             r2
#define bias_hi             r3
#define patch_x             r4
#define macc_iter           r5
#define row_iter            r6
#define shifts              r7
#define scales              r8
#define chan_outs           r9
#define tmp1                r10

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
    {   ldc r11, 1                              ;   ldw scales, sp[STACK_SCALES]        }
    {   shl r11, r11, 8                         ;   ldw chan_outs, sp[STACK_CHAN_OUTS]  }

    {   shl r11, r11, 1                         ;   stw r11, sp[STACK_MODE16]           }
    {   NOOOOP                                  ;   stw r11, sp[STACK_MODE8]            }

.L_patch_chan_start:
// decrement remaining channels, finish if 0
    {   sub chan_outs, chan_outs, 1             ;   bf chan_outs, .Lfunc_end            }
// set VPU mode, reset row_incr
    {   ldap r11, .L_mode8                      ;   ldw row_iter, sp[STACK_ROWS]        }
    {   ldc tmp1, 32                            ;   ldw r11, r11[0]                     }
    {   mov r11, bias_lo                        ;   vsetc r11                           }
// load biases
    {   add bias_lo, bias_lo, tmp1              ;   vldr r11[0]                         }
    {   add bias_hi, bias_hi, tmp1              ;   vldd bias_hi[0]                     }
// reset patch_x for the current channel
    {   NOOOOP                                  ;   ldw patch_x, sp[STACK_X_INITIAL]    }
// reset row iterator for the current channel
    {   NOOOOP                                  ;   ldw row_iter, sp[STACK_ROWS]        }

.L_patch_row_start:
// reset macc count for row
    {   NOOOOP                                  ;   ldw macc_iter, sp[STACK_ROW_MACCS]  }

.L_patch_row_loop:
    {   ldc tmp1, 32                            ;   NOOOOP                              }
    {   add patch_x, patch_x, tmp1              ;   vldc patch_x[0]                     }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   vlmaccr patch_k[0]                  }
    {   sub macc_iter, macc_iter, 1             ;   vlmaccr patch_k[0]                  }
    {   add patch_k, patch_k, tmp1              ;   bt macc_iter, .L_patch_row_loop     } //loop if more MACC groups

.L_patch_row_complete:
// move patch_x and patch_k to their next rows
    {   sub row_iter, row_iter, 1               ;   ldw tmp1, sp[STACK_PATCH_ROW_INCR]  }
    {   add patch_x, patch_x, tmp1              ;   ldw tmp1, sp[STACK_KERNEL_ROW_INCR] }
    {   add patch_k, patch_k, tmp1              ;   bt row_iter, .L_patch_row_start     } //loop if more rows

.L_patch_sum_end:
// set VPU mode to 16-bit and saturate
    {   ldap r11, .L_mode16                     ;   NOOOOP                              }
    {   NOOOOP                                  ;   ldw r11, r11[0]                     }
    {   ldc tmp1, 32                            ;   vsetc r11                           }
// saturate to 16 bits
    {   add shifts, shifts, tmp1                ;   vlsat shifts[0]                     }
    {   ldc tmp1, 32                            ;   NOOOOP                              }
    {   ldc r11, 16                             ;   NOOOOP                              }
// apply scales
    {   add scales, scales, tmp1                ;   vlmul scales[0]                     }
    {   mkmsk tmp1, 16                          ;   vdepth8                             }
    vstrpv y[0], tmp1
    {   add y, y, r11                           ;   ldw tmp1, sp[STACK_KERNEL_ADVANCE]  }

.L_chan_end:
// increment kernel to next group and loop to next channel group
    {   add patch_k, patch_k, tmp1              ;   bt chan_outs, .L_patch_chan_start   }

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



