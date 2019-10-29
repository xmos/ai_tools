
#if defined(__XS3A__)


/*  
void nn_mat_vec_mul_s8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y);
*/

#define FUNCTION_NAME nn_mat_vec_mul_s8_asm

#define NSTACKWORDS  6
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

    
#define W    r0
#define x    r1
#define N_bands         r2
#define N_chunks  r3
#define shr             r4
#define y       r5
#define y_step          r6
#define chunk_iter      r7
#define x_band          r8
#define strpv_mask      r9
#define macc_step       r11

FUNCTION_NAME:

    {   dualentsp NSTACKWORDS   ;   ldc macc_step, 32           }
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    // {   nop                     ;   ldc y_step, 16                  }

    {   ldw y, sp[NSTACKWORDS+2]    ;   sub N_bands, N_bands, 1 }
    {   ldw shr, sp[NSTACKWORDS+1]  ;   ldc y_step, 16          }

    {   vclrdr                      ;   shl r11, macc_step, 4   }
    {   vsetc   r11                 ;   ldc macc_step, 32       }

.Lcompute_band:
    {   mov chunk_iter, N_chunks    ;   mov x_band, x           }
.Lcompute_chunk:
    { vldc x_band[0] ; mkmsk strpv_mask, 16            }

    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
    { vlmaccr W[0]     ; add W, W, macc_step             }
.Lcompute_chunk_end:
    { nop                             ;   sub chunk_iter, chunk_iter, 1   }
    { bt chunk_iter, .Lcompute_chunk ;   add x_band, x_band, macc_step   }
.Lcompute_band_end:
    {   vlsat   shr[0]                  ;   add shr, shr, macc_step         }
    vstrpv  y[0], strpv_mask
    {   vclrdr                          ;   add y, y, y_step                }
    {   bt  N_bands, .Lcompute_band     ;   sub N_bands, N_bands, 1         }

.Lfunc_end:
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



