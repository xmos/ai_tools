FLATBUFFERS_DIR := ${PROJECT_ROOT_DIR}/third_party
GEMMLOWP_DIR := ${PROJECT_ROOT_DIR}/third_party/gemmlowp
TENSORFLOW_DIR := ${PROJECT_ROOT_DIR}/third_party/tensorflow

TENSORFLOW_INCLUDES := \
	-I$(FLATBUFFERS_DIR)/flatbuffers/include \
	-I$(GEMMLOWP_DIR) \
	-I$(TENSORFLOW_DIR)

# source file paths
TENSORFLOW_VPATH := \
	$(FLATBUFFERS_DIR) \
	$(TENSORFLOW_DIR)


TENSORFLOW_SOURCES := \
	tensorflow/lite/c/common.c \
	tensorflow/lite/core/api/error_reporter.cc \
	tensorflow/lite/core/api/flatbuffer_conversions.cc \
	tensorflow/lite/core/api/op_resolver.cc \
	tensorflow/lite/core/api/tensor_utils.cc \
	tensorflow/lite/micro/memory_helpers.cc \
	tensorflow/lite/micro/micro_allocator.cc \
	tensorflow/lite/micro/micro_error_reporter.cc \
	tensorflow/lite/micro/micro_interpreter.cc \
	tensorflow/lite/micro/micro_profiler.cc \
	tensorflow/lite/micro/micro_utils.cc \
	tensorflow/lite/micro/micro_string.cc \
	tensorflow/lite/micro/simple_memory_allocator.cc \
	tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	tensorflow/lite/kernels/internal/quantization_util.cc \
	tensorflow/lite/kernels/kernel_util.cc \
	tensorflow/lite/micro/kernels/pad.cc \
	tensorflow/lite/micro/kernels/softmax.cc

#TENSORFLOW_SOURCES += \
	# tensorflow/lite/micro/kernels/activations.cc \
	# tensorflow/lite/micro/kernels/add.cc \
	# tensorflow/lite/micro/kernels/all_ops_resolver.cc \
	# tensorflow/lite/micro/kernels/arg_min_max.cc \
	# tensorflow/lite/micro/kernels/ceil.cc \
	# tensorflow/lite/micro/kernels/comparisons.cc \
	# tensorflow/lite/micro/kernels/concatenation.cc \
	# tensorflow/lite/micro/kernels/conv.cc \
	# tensorflow/lite/micro/kernels/depthwise_conv.cc \
	# tensorflow/lite/micro/kernels/dequantize.cc \
	# tensorflow/lite/micro/kernels/elementwise.cc \
	# tensorflow/lite/micro/kernels/floor.cc \
	# tensorflow/lite/micro/kernels/fully_connected.cc \
	# tensorflow/lite/micro/kernels/logical.cc \
	# tensorflow/lite/micro/kernels/logistic.cc \
	# tensorflow/lite/micro/kernels/maximum_minimum.cc \
	# tensorflow/lite/micro/kernels/mul.cc \
	# tensorflow/lite/micro/kernels/neg.cc \
	# tensorflow/lite/micro/kernels/pack.cc \
	# tensorflow/lite/micro/kernels/pooling.cc \
	# tensorflow/lite/micro/kernels/prelu.cc \
	# tensorflow/lite/micro/kernels/quantize.cc \
	# tensorflow/lite/micro/kernels/reduce.cc \
	# tensorflow/lite/micro/kernels/reshape.cc \
	# tensorflow/lite/micro/kernels/round.cc \
	# tensorflow/lite/micro/kernels/softmax.cc \
	# tensorflow/lite/micro/kernels/split.cc \
	# tensorflow/lite/micro/kernels/strided_slice.cc \
	# tensorflow/lite/micro/kernels/svdf.cc \
	# tensorflow/lite/micro/kernels/unpack.cc \

ifeq ($(TARGET), x86)
	TENSORFLOW_SOURCES += \
		flatbuffers/src/util.cpp \
		tensorflow/lite/micro/debug_log.cc \
		tensorflow/lite/micro/micro_time.cc 
else # must be xcore
	TENSORFLOW_SOURCES += \
		tensorflow/lite/micro/xcore/debug_log.cc \
		tensorflow/lite/micro/xcore/micro_time.cc 
endif

#************************
# XCORE custom operators
#************************
TENSORFLOW_SOURCES += \
	tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_conv2d.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_arg_min_max.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_pooling.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_fully_connected.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_type_conversions.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_activations.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_custom_options.cc \
