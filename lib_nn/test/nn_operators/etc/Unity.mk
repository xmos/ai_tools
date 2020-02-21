

# The following variables should already be defined:
#    $(BUILD_DIR)    - The absolute path of the build directory -- where object files should be placed
#    $(PLATFORM)     - Either "x86" or "xcore" (no quotes) depending on whether it's being built for x86 or xcore
#    $(API_CHECK)    - Will contain the value 1 iff this file is being included (via the include make directive) to 
#                        gather information about API headers and such.

Unity_PATH ?= ./
LIB_PATH := $(abspath $(Unity_PATH))

# LIB_SRC_DIR := $(THIS_DIR)src

SRC_DIR := src

INCLUDES += $(LIB_PATH)/$(SRC_DIR)

LIB_SOURCES += $(wildcard $(LIB_PATH)/$(SRC_DIR)/*.c)
SOURCE_FILES += $(LIB_SOURCES)