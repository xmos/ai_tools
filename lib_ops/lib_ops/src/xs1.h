// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XS1_CLOCK_COMPAT_H_
#define XS1_CLOCK_COMPAT_H_
#ifndef __ASSEMBLER__
#ifndef _xc_clock_defined
#ifdef __XC__
#define clock __clock_t
#define _xc_clock_defined
#endif /* ifdef__XC__ */
#endif /* _xc_clock_defined */
#endif /* __ASSEMBLER__ */
/*
 * Prevent clock from being defaulted by xs1.h to a
 * typedef unsigned in C files to permit use of time.h
 */
#define _clock_defined
#include_next <xs1.h>
#endif /* XS1_CLOCK_COMPAT_H_ */