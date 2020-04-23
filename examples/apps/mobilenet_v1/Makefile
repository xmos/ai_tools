TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	BUILD_TARGET := x86

	CC := gcc
	CCFLAGS := -g -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3
	CCFLAGS += -DXCORE_TRACE -DXCORE_ERROR
	#CCFLAGS += -DNDEBUG  

	CXX := g++
	CXXFLAGS := -g -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3
	CXXFLAGS += -DXCORE_TRACE -DXCORE_ERROR
	#CXXFLAGS += -DNDEBUG  
	CXXFLAGS += -std=c++11 

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm -pthread

	OBJ_DIR := .build/x86

	APP = mobilenet
else # must be xcore
	BUILD_TARGET := xcore.ai
	
	PLATFORM_FLAGS := -target=XU316-1024-FB265-C32  # 800MHz
	#PLATFORM_FLAGS := -target=XU316-1024-FB265-I28 # 700MHz
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
	CCFLAGS := -DTF_LITE_STATIC_MEMORY
	CCFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS
	CCFLAGS += -DXCORE_TRACE -DXCORE_ERROR
	CCFLAGS += -DNDEBUG
	CCFLAGS += $(PLATFORM_FLAGS)

	CXX := xcc
	CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY
	CXXFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS
	CXXFLAGS += -DXCORE_TRACE -DXCORE_ERROR
	CXXFLAGS += -DNDEBUG
	CXXFLAGS += $(PLATFORM_FLAGS)

	LDFLAGS := $(PLATFORM_FLAGS)

	OBJ_DIR := .build/xcore

	APP = mobilenet.xe
endif

$(info Building for $(BUILD_TARGET))

#********************
# Dependencies
#********************
PROJECT_ROOT_DIR = ${CURDIR}/../../..
include ../../../make/tensorflow.mk
include ../../../make/libs.mk

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

# Add built in ops from TFLu that are needed
TENSORFLOW_SOURCES += \
	tensorflow/lite/kernels/kernel_util.cc \
	tensorflow/lite/kernels/internal/quantization_util.cc \
	tensorflow/lite/micro/kernels/conv.cc \
	tensorflow/lite/micro/kernels/pad.cc \
	tensorflow/lite/micro/kernels/softmax.cc

SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += mobilenet_ops_resolver.cc
SOURCES += mobilenet_v1.c
SOURCES += main.cc

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
