

PLATFORM_NAME = xcore.ai

PLATFORM_FLAGS_DEFAULT := -Os                           \
						-Wno-xcore-fptrgroup            \
						-Wno-unused-variable            \
						-report                         \
						-MMD
#                       -mcmodel=large
#                       -DXCORE
#                       -Wno-unknown-pragmas
#                       -Wno-unknown-attributes
#                       -fcmdline-buffer-bytes=1024
PLATFORM_INCLUDES :=
PLATFORM_EXE_SUFFIX = .xe

PLATFORM_FLAGS := $(PLATFORM_FLAGS_DEFAULT)

AS := xcc
CC := xcc
XCC := xcc
CXX := xcc
AR := xmosar

AS_FLAGS := -g
CC_FLAGS := -g
XCC_FLAGS := -g -O3
CXX_FLAGS := -std=c++11 -g
LD_FLAGS :=

AR_FLAGS := rc