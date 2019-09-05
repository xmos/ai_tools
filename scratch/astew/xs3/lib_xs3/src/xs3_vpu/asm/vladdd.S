
#if defined(__XS3A__)


/*  
void vladdd(
	const void* data);
*/

#define FUNCTION_NAME	vladdd

#define NSTACKWORDS  1
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	
#define data           r0

FUNCTION_NAME:
dualentsp NSTACKWORDS
stw r10, sp[0]

ldw r10, data[0]
vladdd r10[0]

.Lfunc_end:
ldw r10, sp[0]
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



