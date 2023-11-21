TARGET = XCORE-AI-EXPLORER
APP_NAME =

APP_FLAGS  = -report \
-O3 -mcmodel=large -fxscope -Wno-xcore-fptrgroup \
-g \
-Xcompiler-xc -mtimer-abi=hwtimers \
-DUSB_TILE='tile[1]' -DXUD_CORE_CLOCK=600 \
-lquadflash

# The USED_MODULES variable lists other module used by the application. 
USED_MODULES = lib_xud

# *******************************************************
# Include AI Tools library and headers
# *******************************************************
# The following variables are defined for AI Tools library 
# and should be included when building the app.
# XMOS_AITOOLSLIB_DEFINITIONS
# XMOS_AITOOLSLIB_INCLUDES
# XMOS_AITOOLSLIB_LIBRARIES
#
ifeq ($(XMOS_AITOOLSLIB_PATH),)
$(error Path to XMOS AI Tools library and headers not set correctly!)
endif
include ${XMOS_AITOOLSLIB_PATH}/buildfiles/aitoolslib.make

XCC_FLAGS = $(APP_FLAGS) $(XMOS_AITOOLSLIB_DEFINITIONS) $(XMOS_AITOOLSLIB_INCLUDES)
XCC_CPP_FLAGS = $(APP_FLAGS) -std=c++14 $(XMOS_AITOOLSLIB_DEFINITIONS) $(XMOS_AITOOLSLIB_INCLUDES)
XCC_MAP_FLAGS = $(APP_FLAGS) $(XMOS_AITOOLSLIB_LIBRARIES)

#=============================================================================
# The following part of the Makefile includes the common build infrastructure
# for compiling XMOS applications. You should not need to edit below here.

XMOS_MAKE_PATH ?= ../..
include $(XMOS_MAKE_PATH)/xcommon/module_xcommon/build/Makefile.common
