
#if defined(__XS3A__)


/*  
void vgetc(
    unsigned *value);
*/


#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	vgetc
.align	4
.type	vgetc,@function
.cc_top vgetc.function,vgetc
	
#define value           r0

vgetc:
dualentsp NSTACKWORDS

vgetc r11
stw r11, value[0]

.Lvsetc_end:
retsp NSTACKWORDS

	.cc_bottom vgetc.function
	.set	vgetc.nstackwords,NSTACKWORDS
	.globl	vgetc.nstackwords
	.set	vgetc.maxcores,1
	.globl	vgetc.maxcores
	.set	vgetc.maxtimers,0
	.globl	vgetc.maxtimers
	.set	vgetc.maxchanends,0
	.globl	vgetc.maxchanends
.Ltmp0:
	.size	vgetc, .Ltmp0-vgetc
    .issue_mode  single

#endif



