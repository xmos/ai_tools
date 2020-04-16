TARGET := xcore
BIN_DIR := bin

ifeq ($(TARGET), x86)
	BUILD_TARGET := x86

	CC := gcc
	CCFLAGS := -g -O3
	CCFLAGS += -std=c99
	CXX := g++
	CXXFLAGS := -g -O3
	CXXFLAGS += -std=c++11 

	LDFLAGS  := -L/usr/local/lib -lstdc++ -lm -pthread

	OBJ_DIR := .build/x86

	APP = lib_ops_test
else # must be xcore
	BUILD_TARGET := xcore.ai
	
	PLATFORM_FLAGS := -target=XU316-1024-FB265-C32
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
	CCFLAGS += $(PLATFORM_FLAGS)

	CXX := xcc
	CXXFLAGS := -std=c++11
	CXXFLAGS += $(PLATFORM_FLAGS)

	LDFLAGS := $(PLATFORM_FLAGS)

	OBJ_DIR := .build/xcore

	APP = lib_ops_test.xe
endif

$(info Building for $(BUILD_TARGET))

#********************
# Dependencies
#********************
PROJECT_ROOT_DIR = ${CURDIR}/../..
include ../../make/libs.mk
include ../../make/unity.mk

INCLUDES := $(LIBS_INCLUDES)
INCLUDES += $(UNITY_INCLUDES)
#INCLUDES += -Isrc

# source file paths
VPATH := $(LIBS_VPATH)
VPATH += $(UNITY_VPATH)
VPATH += src

SOURCES := $(LIBS_SOURCES)
SOURCES += $(UNITY_SOURCES)
SOURCES += test_allocator.cpp
SOURCES += test_dispatcher.cpp
SOURCES += test_par_region.cpp
SOURCES += main.cpp

OBJECT_FILES := $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(SOURCES))))
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

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
