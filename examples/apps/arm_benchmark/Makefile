TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	BUILD_TARGET := x86

	CC := gcc
	CCFLAGS := -g -O3
	CCFLAGS += -DTF_LITE_STATIC_MEMORY
	CCFLAGS += -DTF_LITE_DISABLE_X86_NEON
	#CCFLAGS += -DNDEBUG  

	CXX := g++
	CXXFLAGS := -g -O3 -std=c++11 
	CXXFLAGS += -DTF_LITE_STATIC_MEMORY
	CXXFLAGS += -DTF_LITE_DISABLE_X86_NEON
	#CXXFLAGS += -DNDEBUG  

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm -pthread

	OBJ_DIR := .build/x86

	APP = cifar-10
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
	CCFLAGS += -DNDEBUG

	CCFLAGS += $(PLATFORM_FLAGS)

	CXX := xcc
	CXXFLAGS := -std=c++11 -DTF_LITE_STATIC_MEMORY
	CXXFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS
	CXXFLAGS += -DNDEBUG
	CXXFLAGS += $(PLATFORM_FLAGS)

	LDFLAGS := $(PLATFORM_FLAGS)

	OBJ_DIR := .build/xcore

	APP = cifar-10.xe
endif

$(info Building for $(BUILD_TARGET))

#********************
# Dependencies
#********************
include ../make/tensorflow.mk
include ../make/libs.mk

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += main.cc xcore_model.c

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
