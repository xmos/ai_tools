

# The following variables should already be defined:
#    $(BUILD_DIR)    - The absolute path of the build directory -- where object files should be placed
#    $(PLATFORM)     - Either "x86" or "xcore" (no quotes) depending on whether it's being built for x86 or xcore
#    $(API_CHECK)    - Will contain the value 1 iff this file is being included (via the include make directive) to 
#                        gather information about API headers and such.

lib_nn_PATH ?= ./
LIB_PATH := $(abspath $(lib_nn_PATH))

# LIB_SRC_DIR := $(THIS_DIR)src

API_DIR := api
SRC_DIR := src

INCLUDES += $(LIB_PATH)/$(API_DIR) $(LIB_PATH)/$(SRC_DIR)

ifneq ($(strip $(API_CHECK)),$(strip 1))
  LIB_SOURCES += $(wildcard $(SRC_DIR)/**/*.c)
  
  ifeq ($(strip $(PLATFORM)),$(strip xcore))
    LIB_SOURCES += $(wildcard $(SRC_DIR)/**/*.S)
  endif

  SOURCE_FILES += $(LIB_SOURCES)
endif





