TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	BUILD_TARGET := x86

	CC := gcc
	CCFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3 

	CXX := g++
	CXXFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3
	CXXFLAGS += -std=c++11 

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm

	OBJ_DIR := .build/x86

	APP = test_model
else # must be xcore
	BUILD_TARGET := xcore.ai
	
	PLATFORM_FLAGS := -target=XU316-1024-QF60-C20
	PLATFORM_FLAGS += -mcmodel=large
	PLATFORM_FLAGS += -Os
	PLATFORM_FLAGS += -DXCORE
	PLATFORM_FLAGS += -Wno-xcore-fptrgroup
	PLATFORM_FLAGS += -Wno-unknown-pragmas
	PLATFORM_FLAGS += -Wno-unknown-attributes
	PLATFORM_FLAGS += -report
	PLATFORM_FLAGS += -fcmdline-buffer-bytes=1024

	AS := xcc
	ASFLAGS := $(PLATFORM_FLAGS)

	CC := xcc
	CCFLAGS := -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
	CCFLAGS += $(PLATFORM_FLAGS)

	CXX := xcc
	CXXFLAGS := -std=c++11 -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
	CXXFLAGS += $(PLATFORM_FLAGS)

	LDFLAGS := $(PLATFORM_FLAGS)

	OBJ_DIR := .build/xcore

	APP = test_model.xe
endif

$(info Building for $(BUILD_TARGET))

#********************
# Dependencies
#********************
FLATBUFFERS_DIR := ../../../third_party/
GEMMLOWP_DIR := ../../../third_party/gemmlowp
TENSORFLOW_DIR := ../../../third_party/tensorflow
LIB_NN_DIR := ../../../lib_nn

INCLUDES := \
	-I$(FLATBUFFERS_DIR)/flatbuffers/include \
	-I$(GEMMLOWP_DIR) \
	-I$(TENSORFLOW_DIR) \
	-I$(LIB_NN_DIR)/lib_nn/api \
	-Isrc

# source file paths
VPATH := \
	$(FLATBUFFERS_DIR) \
	$(TENSORFLOW_DIR) \
	$(LIB_NN_DIR) \
	src

SOURCES := \
	tensorflow/lite/c/common.c \
	tensorflow/lite/core/api/error_reporter.cc \
	tensorflow/lite/core/api/flatbuffer_conversions.cc \
	tensorflow/lite/core/api/op_resolver.cc \
	tensorflow/lite/core/api/tensor_utils.cc \
	tensorflow/lite/kernels/kernel_util.cc \
	tensorflow/lite/kernels/internal/quantization_util.cc \
	tensorflow/lite/micro/debug_log_numbers.cc \
	tensorflow/lite/micro/memory_helpers.cc \
	tensorflow/lite/micro/micro_allocator.cc \
	tensorflow/lite/micro/micro_error_reporter.cc \
	tensorflow/lite/micro/micro_interpreter.cc \
	tensorflow/lite/micro/micro_utils.cc \
	tensorflow/lite/micro/simple_memory_allocator.cc \
	tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	tensorflow/lite/micro/kernels/activations.cc \
	tensorflow/lite/micro/kernels/add.cc \
	tensorflow/lite/micro/kernels/all_ops_resolver.cc \
	tensorflow/lite/micro/kernels/arg_min_max.cc \
	tensorflow/lite/micro/kernels/ceil.cc \
	tensorflow/lite/micro/kernels/comparisons.cc \
	tensorflow/lite/micro/kernels/concatenation.cc \
	tensorflow/lite/micro/kernels/conv.cc \
	tensorflow/lite/micro/kernels/depthwise_conv.cc \
	tensorflow/lite/micro/kernels/dequantize.cc \
	tensorflow/lite/micro/kernels/elementwise.cc \
	tensorflow/lite/micro/kernels/floor.cc \
	tensorflow/lite/micro/kernels/fully_connected.cc \
	tensorflow/lite/micro/kernels/logical.cc \
	tensorflow/lite/micro/kernels/logistic.cc \
	tensorflow/lite/micro/kernels/maximum_minimum.cc \
	tensorflow/lite/micro/kernels/mul.cc \
	tensorflow/lite/micro/kernels/neg.cc \
	tensorflow/lite/micro/kernels/pack.cc \
	tensorflow/lite/micro/kernels/pad.cc \
	tensorflow/lite/micro/kernels/pooling.cc \
	tensorflow/lite/micro/kernels/prelu.cc \
	tensorflow/lite/micro/kernels/quantize.cc \
	tensorflow/lite/micro/kernels/reshape.cc \
	tensorflow/lite/micro/kernels/round.cc \
	tensorflow/lite/micro/kernels/softmax.cc \
	tensorflow/lite/micro/kernels/split.cc \
	tensorflow/lite/micro/kernels/strided_slice.cc \
	tensorflow/lite/micro/kernels/svdf.cc \
	tensorflow/lite/micro/kernels/unpack.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_conv2d.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_arg_min_max.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_pooling.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_fully_connected.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_type_conversions.cc \
	lib_nn/src/nn_operator.c \
	lib_nn/src/nn_operator_conv.c \
	main.cc

ifeq ($(TARGET), x86)
	SOURCES += \
		flatbuffers/src/util.cpp \
		tensorflow/lite/micro/debug_log.cc 
else # must be xcore
	SOURCES += \
		lib_nn/src/asm/avgpool2d_deep.S \
		lib_nn/src/asm/conv2d_deepin_deepout_block.S \
		lib_nn/src/asm/conv2d_shallowin_deepout_block.S \
		lib_nn/src/asm/fc_deepin_shallowout_16.S \
		lib_nn/src/asm/fc_deepin_shallowout_8.S \
		lib_nn/src/asm/maxpool2d_deep.S \
		lib_nn/src/asm/vpu_memcpy.S \
		lib_nn/src/asm/requantize_16_to_8.S \
		tensorflow/lite/micro/xcore/debug_log.cc 
endif

OBJECT_FILES := $(patsubst %.cpp,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(SOURCES)))))
OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(OBJECT_FILES))
OBJECTS := $(SOURCES) $(OBJECT_FILES)

.PHONY: all clean $(APP)

all: $(BIN_DIR)/$(APP)

$(OBJ_DIR)/%.o: %.S
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) -o $@ -c $<

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
