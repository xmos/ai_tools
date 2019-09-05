
#if defined(__XS3A__)


/*  
void vsetc(
    unsigned value);
*/


#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl	vsetc
.align	4
.type	vsetc,@function
.cc_top vsetc.function,vsetc
	
#define value           r0

vsetc:
dualentsp NSTACKWORDS

add r11, value, 0
vsetc r11

.Lvsetc_end:
retsp NSTACKWORDS

	.cc_bottom vsetc.function
	.set	vsetc.nstackwords,NSTACKWORDS
	.globl	vsetc.nstackwords
	.set	vsetc.maxcores,1
	.globl	vsetc.maxcores
	.set	vsetc.maxtimers,0
	.globl	vsetc.maxtimers
	.set	vsetc.maxchanends,0
	.globl	vsetc.maxchanends
.Ltmp0:
	.size	vsetc, .Ltmp0-vsetc
    .issue_mode  single

#endif



