// Copyright (c) 2016, XMOS Ltd, All rights reserved

	.section    .cp.rodata, "ac", @progbits
	g_hyp_scalar:
	.word 0x9b74eda7 // (unsigned)((double)UINT_MAX*0.607252935008881);

	.section	.dp.data,"awd",@progbits
	.text
	.cc_top hypot_i.function
	.globl	hypot_i
	.align	4
	.type	hypot_i,@function

#define ITT(N) \
	{lss r3, r0, r11; ldc r2, N};\
	{bt r3, neg_ ## N; shr r3, r1, r2};\
		{shr r2, r0, r2;sub r0, r0, r3};\
		{bu next_ ## N; add r1, r1, r2};\
	neg_ ## N:;\
		ashr r2, r0, r2;\
		{add r0, r0, r3; sub r1, r1, r2};\
	next_ ## N:

#define ITT_FINAL(N) \
	{lss r3, r0, r11; ldc r2, N};\
	{bt r3, neg_ ## N;};\
		{shr r2, r0, r2; ldc r3, 0};\
		{bu next_ ## N; add r1, r1, r2};\
	neg_ ## N:;\
		ashr r2, r0, r2;\
		{sub r1, r1, r2; ldc r3, 0};\
	next_ ## N:

hypot_i:
.align 8
.issue_mode dual
	ashr r0, r0, 2
	ldc r11, 0
	lss r2, r1, r11
	{bf r2, start; lss r2, r0, r11}
		{neg r1, r1}
	start:
	{bf r2, neg_0;shr r1, r1, 2}
		{add r0, r0, r1; sub r1, r1, r0}
		bu next_0
	neg_0:
		{sub r0, r0, r1; add r1, r1, r0}
	next_0:

	ITT(1)
	ITT(2)
	ITT(3)
	ITT(4)
	ITT(5)
	ITT(6)
	ITT(7)
	ITT(8)
	ITT(9)
	ITT(10)
	ITT(11)
	ITT(12)
	ITT(13)
	ITT(14)
	ITT(15)
	ITT(16)
	ITT(17)
	ITT_FINAL(18)

	ldw r0, cp[g_hyp_scalar]
	maccu r11, r3, r1, r0
	{retsp 0;shl r0, r11, 2}


.tmp_hypot_i:
	.size	hypot_i, .tmp_hypot_i-hypot_i
	.align	4
	.cc_bottom hypot_i.function

	.set	hypot_i.nstackwords,0
	.globl	hypot_i.nstackwords
	.set	hypot_i.maxcores,1
	.globl	hypot_i.maxcores
	.set	hypot_i.maxtimers,0
	.globl	hypot_i.maxtimers
	.set	hypot_i.maxchanends,0
	.globl	hypot_i.maxchanends

	.cc_top scale.function
	.globl	scale
	.align	4
	.type	scale,@function
scale:
.align 8
.issue_mode dual
	ldc r3, 0
	ldivu r1,r11,r3,r1,r2
	ldivu r0,r2,r11,r0,r2
	retsp 0

.tmp_scale:
	.size	scale, .tmp_scale-scale
	.align	4
	.cc_bottom scale.function

	.set	scale.nstackwords,0
	.globl	scale.nstackwords
	.set	scale.maxcores,1
	.globl	scale.maxcores
	.set	scale.maxtimers,0
	.globl	scale.maxtimers
	.set	scale.maxchanends,0
	.globl	scale.maxchanends



