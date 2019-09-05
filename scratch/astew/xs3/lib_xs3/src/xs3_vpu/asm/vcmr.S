
#if defined(__XS3A__)


/*  
void vcmr();
*/

#define FUNCTION_NAME	vcmr

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	
FUNCTION_NAME:
dualentsp NSTACKWORDS

vcmr

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



