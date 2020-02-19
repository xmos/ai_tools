

APP_NAME = nn_operators_test
SRC_DIR = src
BUILD_DIR = .build
BIN_DIR = bin
# APP_BUILD_DIR = $(BUILD_DIR)/$(APP_NAME)
TARGET = xcore # XU316-1024-QF60-C20
APP_CONFIG = Default

#Windows command to create a directory if it doesn't exist.
define mkdir_cmd
	@test -d $(subst /,\,$(dir $(1))) || mkdir $(subst /,\,$(dir $(1)))
endef

ifeq ($(TARGET), x86)
	BUILD_TARGET := x86

	CC := gcc
	CCFLAGS := -g -O3

	XCC := gcc
	XCCFLAGS := -g -O3
	
	CXX := g++
	CXXFLAGS := -g -O3
	CXXFLAGS += -std=c++11

	LDFLAGS := -L/usr/local/lib -lstdc++ -lm

	OBJ_DIR := .build/x86

	APP = $(APP_NAME)
else 
	#must be xcore
	BUILD_TARGET := xcore.ai

	PLATFORM_FLAGS := -target=XU316-1024-QF60-C20
	# PLATFORM_FLAGS += -mcmodel=large
	PLATFORM_FLAGS += -Os
	# PLATFORM_FLAGS += -DXCORE
	PLATFORM_FLAGS += -Wno-xcore-fptrgroup
	# PLATFORM_FLAGS += -Wno-unknown-pragmas
	# PLATFORM_FLAGS += -Wno-unknown-attributes
	PLATFORM_FLAGS += -Wno-unused-variable
	PLATFORM_FLAGS += -report
	# PLATFORM_FLAGS += -fcmdline-buffer-bytes=1024

	AS := xcc
	ASFLAGS := $(PLATFORM_FLAGS)

	CC := xcc
	CCFLAGS := -g
	CCFLAGS += $(PLATFORM_FLAGS)
	
	XCC := xcc
	XCCFLAGS := -g -O3
	XCCFLAGS += $(PLATFORM_FLAGS)

	CXX := xcc
	CXXFLAGS := -std=c++11 -g -DTF_LITE_STATIC_MEMORY -DNDEBUG
	CXXFLAGS += $(PLATFORM_FLAGS)

	LDFLAGS := $(PLATFORM_FLAGS)

	OBJ_DIR := .build/xcore

	APP = $(APP_NAME).xe
endif

$(info Building for $(BUILD_TARGET))


DEPENDENCIES := lib_nn Unity

DEP_PATH := ../..
DEP_PATHS := $(addsuffix /lib.mk,$(addprefix $(DEP_PATH)/,$(DEPENDENCIES)))

ALL_INCLUDES :=
ALL_OBJECT_FILES := 
ALL_OBJECTS := 

include $(DEP_PATHS)

INCLUDES := -I$(SRC_DIR)

ALL_INCLUDES += $(INCLUDES)

VPATH := $(SRC_DIR)

APP_SOURCES := $(wildcard $(SRC_DIR)/*.xc)
APP_SOURCES += $(wildcard $(SRC_DIR)/*.c)

ifeq ($(TARGET), x86)

else
	APP_SOURCES += $(wildcard $(LIB_NN_DIR)/*.S)
endif

APP_OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(addsuffix .o, $(APP_SOURCES)))


# APP_OBJECT_FILES := $(patsubst %.xc,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(APP_SOURCES))))))
# APP_OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(APP_OBJECT_FILES))
APP_OBJECTS := $(APP_SOURCES) $(APP_OBJECT_FILES)
# OBJECTS := $(APP_SOURCES) $(OBJECT_FILES)

ALL_OBJECT_FILES += $(APP_OBJECT_FILES)

ALL_OBJECTS += $(APP_OBJECTS)

.PHONY: all clean $(APP_NAME)

all: $(BIN_DIR)/$(APP)


# $(OBJ_DIR)/%.o: %.S
# 	@mkdir -p $(dir $@)
# 	$(AS) $(ASFLAGS) -o $@ -c $<

# $(OBJ_DIR)/%.o: %.c
# 	@mkdir -p $(dir $@)
# 	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

# $(OBJ_DIR)/%.o: $(SRC_DIR)/%.xc
# 	@mkdir -p $(dir $@)
# 	$(XCC) $(XCCFLAGS) $(INCLUDES) -o $@ -c $<

# $(OBJ_DIR)/%.o: %.cc
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

# $(OBJ_DIR)/%.o: %.cpp
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(filter %.S.o, $(APP_OBJECT_FILES)): $(OBJ_DIR)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(AS) $(ASFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(filter %.cpp.o, $(APP_OBJECT_FILES)): $(OBJ_DIR)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(CXX) $(CXXFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(filter %.cc.o, $(APP_OBJECT_FILES)): $(OBJ_DIR)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(CC) $(CCFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(filter %.c.o, $(APP_OBJECT_FILES)): $(OBJ_DIR)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(CC) $(CCFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(filter %.xc.o, $(APP_OBJECT_FILES)): $(OBJ_DIR)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(XCC) $(XCCFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(ALL_OBJECTS)
	$(call mkdir_cmd,$@)
	$(info Building executable $@..)
	@$(XCC) $(LDFLAGS) $(ALL_OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	$(info Cleaning project...)
	rm -rvf $(BIN_DIR)
	rm -rvf $(OBJ_DIR)


