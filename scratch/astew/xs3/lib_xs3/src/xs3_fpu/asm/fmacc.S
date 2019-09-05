
#if defined(__XS3A__)


/*  
float fmacc(
    const float acc,
    const float a,
    const float b);
*/

#define FUNCTION_NAME	fmacc

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
dualentsp NSTACKWORDS

fmacc r0, r0, r1, r2

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



