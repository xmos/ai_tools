TARGET = XCORE-AI-EXPLORER
APP_NAME = app_device

APP_FLAGS = -report \
-O3 -fcomment-asm -mcmodel=large -fxscope -Wno-xcore-fptrgroup \
-Wno-unknown-pragmas \
-Wno-bidirectional-buffered-port \
-lquadflash \
-g
#-Wm,--image-size,8388608,--image-base,8388608

# Comment out the below TFLMC_XCORE_PROFILE definition to disable profiling
APP_FLAGS += -DTFLMC_XCORE_PROFILE

# Enable the below SHARED_TENSOR_ARENA definition for multiple models 
# or to manage tensor arena array yourself
#APP_FLAGS += -DSHARED_TENSOR_ARENA

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

