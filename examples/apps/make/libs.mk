LIB_NN_DIR := ../../../lib_nn
LIB_OPS_DIR := ../../../lib_ops

LIBS_INCLUDES := \
	-I$(LIB_NN_DIR) \
	-I$(LIB_NN_DIR)/lib_nn/api \
	-I$(LIB_OPS_DIR)

LIBS_VPATH += \
	$(LIB_NN_DIR) \
	$(LIB_OPS_DIR)

LIBS_SOURCES := \
	lib_nn/src/c/nn_op_utils.c \
	lib_nn/src/c/nn_operator.c \
	lib_nn/src/c/nn_operator_conv.c \
	lib_nn/src/c/conv2d_1x1.c \
	lib_nn/src/c/conv2d_depthwise.c \
	lib_nn/src/c/avgpool2d.c \
	lib_nn/src/c/maxpool2d.c \
	lib_nn/src/c/fully_connected.c \
	lib_ops/src/operator_dispatcher.cpp \
	lib_ops/src/conv2d.cpp \
	lib_ops/src/fully_connected.cpp\
	lib_ops/src/activations.cpp \
	lib_ops/src/pooling.cpp \
	lib_ops/src/arg_min_max.cpp \
	lib_ops/src/type_conversions.cpp

ifneq ($(TARGET), x86)
	LIBS_SOURCES += \
		lib_nn/src/asm/conv2d_deepin_deepout_block.S \
		lib_nn/src/asm/conv2d_shallowin_deepout_block.S \
		lib_nn/src/asm/conv2d_1x1.S \
		lib_nn/src/asm/fully_connected_16.S \
		lib_nn/src/asm/avgpool2d.S \
		lib_nn/src/asm/maxpool2d.S \
		lib_nn/src/asm/avgpool2d_2x2.S \
		lib_nn/src/asm/avgpool2d_global.S \
		lib_nn/src/asm/vpu_memcpy.S \
		lib_nn/src/asm/requantize_16_to_8.S \
		lib_nn/src/asm/lookup8.S \
		lib_nn/src/asm/util/nn_compute_hstrip_depthwise.S \
		lib_nn/src/asm/util/nn_compute_hstrip_depthwise_bias_adj.S \
		lib_nn/src/asm/util/nn_compute_hstrip_depthwise_padded.S \
		lib_nn/src/asm/util/nn_compute_patch_depthwise.S \
		lib_nn/src/asm/util/nn_compute_patch_depthwise_padded.S
endif
