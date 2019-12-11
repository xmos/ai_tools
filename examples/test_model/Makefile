TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	CC := gcc
	CCFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3 

	CXX := g++
	CXXFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3
	CXXFLAGS += -std=c++11 

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm

	OBJ_DIR := .build/x86

	APP = test_model
else # must be xcore
	PLATFORM_FLAGS := -target=XU316-1024-QF60-C20
	PLATFORM_FLAGS += -mcmodel=large
	PLATFORM_FLAGS += -Os
	PLATFORM_FLAGS += -DXCORE
	PLATFORM_FLAGS += -Wno-xcore-fptrgroup
	PLATFORM_FLAGS += -Wno-unknown-pragmas
	PLATFORM_FLAGS += -Wno-unknown-attributes
	PLATFORM_FLAGS += -report
	PLATFORM_FLAGS += -fcmdline-buffer-bytes=256

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

#********************
# Dependencies
#********************
FLATBUFFERS_DIR := ../../../flatbuffers
GEMMLOWP_DIR := ../../../gemmlowp
TENSORFLOW_DIR := ../../../tensorflow
LIB_NN_DIR := ../../xcore/operator_book

INCLUDES := \
	-I$(FLATBUFFERS_DIR)/include \
	-I$(GEMMLOWP_DIR) \
	-I$(TENSORFLOW_DIR) \
	-I$(LIB_NN_DIR)/lib_nn_operator/api \
	-Isrc

# source file paths
VPATH := \
	$(TENSORFLOW_DIR) \
	$(LIB_NN_DIR) \
	src

SOURCES := \
	tensorflow/lite/c/common.c \
	tensorflow/lite/core/api/error_reporter.cc \
	tensorflow/lite/core/api/flatbuffer_conversions.cc \
	tensorflow/lite/core/api/op_resolver.cc \
	tensorflow/lite/core/api/tensor_utils.cc \
	tensorflow/lite/micro/debug_log_numbers.cc \
	tensorflow/lite/micro/memory_helpers.cc \
	tensorflow/lite/micro/micro_allocator.cc \
	tensorflow/lite/micro/micro_error_reporter.cc \
	tensorflow/lite/micro/micro_interpreter.cc \
	tensorflow/lite/micro/micro_mutable_op_resolver.cc \
	tensorflow/lite/micro/micro_utils.cc \
	tensorflow/lite/micro/simple_memory_allocator.cc \
	tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.cc \
	lib_nn_operator/src/nn_operator.c \
	main.cc

ifeq ($(TARGET), x86)
	SOURCES += \
		tensorflow/lite/micro/debug_log.cc 
else # must be xcore
	SOURCES += \
		lib_nn_operator/src/asm/conv2d_deepin_deepout_relu_patch.S \
		lib_nn_operator/src/asm/conv2d_deepin_deepout_relu.S \
		lib_nn_operator/src/asm/conv2d_shallowin_deepout_relu_patch.S \
		lib_nn_operator/src/asm/conv2d_shallowin_deepout_relu.S \
		lib_nn_operator/src/asm/fc_deepin_shallowout_lin.S \
		lib_nn_operator/src/asm/maxpool2d_deep.S \
		tensorflow/lite/micro/xcore/debug_log.cc 
endif

OBJECT_FILES := $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(SOURCES))))
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

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
