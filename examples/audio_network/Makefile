# The TARGET variable determines what target system the application is 
# compiled for. It either refers to an XN file in the source directories
# or a valid argument for the --target option when compiling.

TARGET = xk-audio-316-mc.xn
APP_NAME =

ifeq ($(XMOS_AITOOLSLIB_PATH),)
$(error Path to XMOS AI Tools library and headers not set correctly!)
endif
include ${XMOS_AITOOLSLIB_PATH}/buildfiles/aitoolslib.make

# The USED_MODULES variable lists other module used by the application. These
# modules will extend the SOURCE_DIRS, INCLUDE_DIRS and LIB_DIRS variables. 
# Modules are expected to be in the directory above the BASE_DIR directory.
USED_MODULES = lib_xua lib_i2c lib_src lib_xcore_math

# Audio Class 2, Async, I2S Master, 8xInput, 8xOutput
XCC_FLAGS= -fcomment-asm -O3 -report -lquadflash -g -DUSB_TILE=tile[0] -DADAT_TX_USE_SHARED_BUFF=1 -DQUAD_SPI_FLASH=1 -DMIXER=0 #  -fxscope

XCC_FLAGS += $(APP_FLAGS) $(XMOS_AITOOLSLIB_DEFINITIONS) $(XMOS_AITOOLSLIB_INCLUDES)
XCC_CPP_FLAGS += $(APP_FLAGS) -std=c++14 $(XMOS_AITOOLSLIB_DEFINITIONS) $(XMOS_AITOOLSLIB_INCLUDES)
XCC_MAP_FLAGS += $(APP_FLAGS) $(XMOS_AITOOLSLIB_LIBRARIES)

#=============================================================================
# The following part of the Makefile includes the common build infrastructure
# for compiling XMOS applications. You should not need to edit below here.

XMOS_MAKE_PATH ?= ../..
include $(XMOS_MAKE_PATH)/xcommon/module_xcommon/build/Makefile.common
