TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	CC := gcc
	CCFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3 

	CXX := g++
	CXXFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY -O3
	CXXFLAGS += -std=c++11 

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm

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

	APP = test_model.xe
endif

#********************
# Dependencies
#********************
FLATBUFFERS_DIR := ../../../flatbuffers
GEMMLOWP_DIR := ../../../gemmlowp
TENSORFLOW_DIR := ../../../tensorflow
LIB_NN_DIR := ../../xcore/operator_book/lib_nn_operator

INCLUDES := \
	-I$(FLATBUFFERS_DIR)/include \
	-I$(GEMMLOWP_DIR) \
	-I$(TENSORFLOW_DIR) \
	-I$(LIB_NN_DIR)/api \
	-Isrc

SOURCES := \
	$(TENSORFLOW_DIR)/tensorflow/lite/c/common.c \
	$(TENSORFLOW_DIR)/tensorflow/lite/core/api/error_reporter.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/core/api/op_resolver.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/core/api/tensor_utils.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/debug_log_numbers.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/memory_helpers.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/micro_allocator.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/micro_error_reporter.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/micro_mutable_op_resolver.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/micro_utils.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/simple_memory_allocator.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	$(TENSORFLOW_DIR)/tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.cc \
	$(wildcard $(LIB_NN_DIR)/src/*.c) \
	$(wildcard src/*.cc)

ifeq ($(TARGET), x86)
	SOURCES += \
		$(TENSORFLOW_DIR)/tensorflow/lite/micro/debug_log.cc 
else # must be xcore
	SOURCES += \
		$(wildcard $(LIB_NN_DIR)/src/asm/*.S) \
		$(TENSORFLOW_DIR)/tensorflow/lite/micro/xcore/debug_log.cc 
endif

OBJECTS := $(SOURCES:%.S=%.o)
OBJECTS += $(SOURCES:%.c=%.o)
OBJECTS += $(SOURCES:%.cc=%.o)

OBJECT_FILES := $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(SOURCES))))

.PHONY: all clean $(APP)

all: $(BIN_DIR)/$(APP)

%.o: %.S
	$(AS) $(ASFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
