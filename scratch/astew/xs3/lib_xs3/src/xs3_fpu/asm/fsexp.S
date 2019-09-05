
#if defined(__XS3A__)


/*  
void fsexp(
    int* exp,
    int* sign,
    const float a);
*/

#define FUNCTION_NAME	fsexp

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
dualentsp NSTACKWORDS

fsexp r3, r11, r2
stw r11, r0[0]
stw r3, r1[0]

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



