#********************
# Dependencies
#********************
PROJECT_ROOT_DIR = ${CURDIR}/../..
include ../../make/platform.mk
include ../../make/libs.mk
include ../../make/unity.mk

#********************
# App specific
#********************
ifeq ($(TARGET), x86)
	APP = lib_ops_test
else # must be xcore
	APPFLAGS += -fcmdline-buffer-bytes=1024
	APP = lib_ops_test.xe
endif

APPFLAGS := -DENABLE_TRACING
APPFLAGS += -DENABLE_BENCHMARKING
#APPFLAGS += -DNDEBUG

INCLUDES := $(LIBS_INCLUDES)
INCLUDES += $(UNITY_INCLUDES)

# source file paths
VPATH := $(LIBS_VPATH)
VPATH += $(UNITY_VPATH)
VPATH += src

SOURCES := $(UNITY_SOURCES)
SOURCES += lib_ops/src/device_memory.c
SOURCES += lib_ops/src/allocator.cpp
SOURCES += lib_ops/src/benchmarking.cpp
SOURCES += lib_ops/src/dispatcher.cpp
SOURCES += lib_ops/src/planning.cpp
SOURCES += test_allocator.cpp
SOURCES += test_dispatcher.cpp
SOURCES += test_execution_plan.cpp
SOURCES += test_stopwatch.cpp
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
	$(CC) $(CCFLAGS) $(APPFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(APPFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
