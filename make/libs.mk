LIB_NN_DIR := ${PROJECT_ROOT_DIR}/lib_nn
OPERATORS_DIR := ${PROJECT_ROOT_DIR}/operators

LIBS_INCLUDES := \
	-I$(LIB_NN_DIR) \
	-I$(LIB_NN_DIR)/lib_nn/api \
	-I$(OPERATORS_DIR)

LIBS_VPATH += \
	$(LIB_LOGGING_DIR) \
	$(LIB_NN_DIR) \
	$(OPERATORS_DIR)

LIBS_SOURCES := \
	lib_nn/src/asm/asm_constants.c \
	lib_nn/src/c/vpu_sim.c \
	lib_nn/src/c/nn_op_utils.c \
	lib_nn/src/c/nn_operator.c \
	lib_nn/src/c/nn_operator_conv.c \
	lib_nn/src/c/conv2d_deep.c \
	lib_nn/src/c/conv2d_shallowin.c \
	lib_nn/src/c/conv2d_1x1.c \
	lib_nn/src/c/conv2d_depthwise.c \
	lib_nn/src/c/avgpool2d.c \
	lib_nn/src/c/maxpool2d.c \
	lib_nn/src/c/fully_connected.c \
	lib_nn/src/c/util/deep/nn_conv2d_hstrip_deep.c \
	lib_nn/src/c/util/shallow/nn_conv2d_hstrip_shallowin.c \
	lib_nn/src/c/util/depthwise/nn_conv2d_hstrip_depthwise.c \
	operators/device_memory.c \
	operators/xcore_profiler.cpp \
	operators/xcore_reporter.cpp \
	operators/xcore_interpreter.cpp \
	operators/planning.cpp \
	operators/dispatcher.cpp \
	operators/conv2d.cpp \
	operators/fully_connected.cpp\
	operators/activations.cpp \
	operators/pooling.cpp \
	operators/arg_min_max.cpp \
	operators/type_conversions.cpp

ifneq ($(TARGET), x86)
	LIBS_SOURCES += \
		lib_nn/src/asm/conv2d_1x1.S \
		lib_nn/src/asm/fully_connected_16.S \
		lib_nn/src/asm/avgpool2d.S \
		lib_nn/src/asm/maxpool2d.S \
		lib_nn/src/asm/avgpool2d_2x2.S \
		lib_nn/src/asm/avgpool2d_global.S \
		lib_nn/src/asm/vpu_memcpy.S \
		lib_nn/src/asm/requantize_16_to_8.S \
		lib_nn/src/asm/lookup8.S \
		lib_nn/src/asm/util/shallow/nn_conv2d_hstrip_shallowin.S \
		lib_nn/src/asm/util/shallow/nn_conv2d_hstrip_shallowin_padded.S \
		lib_nn/src/asm/util/shallow/nn_conv2d_hstrip_tail_shallowin.S \
		lib_nn/src/asm/util/shallow/nn_conv2d_hstrip_tail_shallowin_padded.S \
		lib_nn/src/asm/util/deep/nn_conv2d_hstrip_deep.S \
		lib_nn/src/asm/util/deep/nn_conv2d_hstrip_deep_padded.S \
		lib_nn/src/asm/util/deep/nn_conv2d_hstrip_tail_deep.S \
		lib_nn/src/asm/util/deep/nn_conv2d_hstrip_tail_deep_padded.S \
		lib_nn/src/asm/util/depthwise/nn_conv2d_hstrip_depthwise_padded.S \
		lib_nn/src/asm/util/depthwise/nn_conv2d_hstrip_depthwise.S
endif
