
#if defined(__XS3A__)


/*  
void vclrdr();
*/


#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	vclrdr
.align	4
.type	vclrdr,@function
.cc_top vclrdr.function,vclrdr

vclrdr:
dualentsp NSTACKWORDS

vclrdr 

.Lvclrdr_end:
retsp NSTACKWORDS

	.cc_bottom vclrdr.function
	.set	vclrdr.nstackwords,NSTACKWORDS
	.globl	vclrdr.nstackwords
	.set	vclrdr.maxcores,1
	.globl	vclrdr.maxcores
	.set	vclrdr.maxtimers,0
	.globl	vclrdr.maxtimers
	.set	vclrdr.maxchanends,0
	.globl	vclrdr.maxchanends
.Ltmp0:
	.size	vclrdr, .Ltmp0-vclrdr
    .issue_mode  single

#endif



