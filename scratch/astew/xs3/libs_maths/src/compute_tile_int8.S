
#if defined(__XS3A__)


/*  
void compute_tile_int8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_chunks,
    const int W_step,
    const int W_chunk_step);
*/

#define FUNCTION_NAME	compute_tile_int8_asm

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define W				r0
#define x				r1
#define	N_chunks		r2
#define macc_step       r3
// #define W_chunk_step	r11

// .Lchunk_size:
//     .int    16*32

FUNCTION_NAME:
{   dualentsp NSTACKWORDS               ;   ldc macc_step, 32        }
// std	r4, r5, sp[0]

// {	shl W_chunk_step, W_chunk_step, 4	;	mov macc_step, W_chunk_step	}

.Lcompute_chunk:
    {	vldc	x[0]		;	nop             }

    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}
    {	vlmaccr W[0]	;	add W, W, macc_step	}

    {	add x, x, macc_step             ;	sub N_chunks, N_chunks, 1	}
    {	bt N_chunks, .Lcompute_chunk	;	nop	                        }

.Lfunc_end:

// ldd r4, r5, sp[0]
retsp NSTACKWORDS

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

#endif



