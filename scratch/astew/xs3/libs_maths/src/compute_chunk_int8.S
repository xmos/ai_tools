
#if defined(__XS3A__)


/*  
void compute_chunk_int8_asm(
    int8_t* W,
    int8_t* x,
    int W_step);
*/

#define FUNCTION_NAME	compute_chunk_int8_asm

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define W       r0
#define x		r1
#define W_step  r2
#define iter	r3

FUNCTION_NAME:
    { dualentsp NSTACKWORDS ;   ldc W_step, 32      }
    { vldc	x[0]			;	ldc iter, 3         }

//I was told instruction fetches end up happening every 4 vector ops anyway,
// so may as well loop
#if (0)
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
#else

.Lloop_top:
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	vlmaccr W[0]		;	add W, W, W_step	}
    {	bt iter, .Lloop_top ;   sub iter, iter, 1   }

#endif

.Lfunc_end:
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



