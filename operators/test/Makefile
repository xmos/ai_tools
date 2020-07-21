#********************
# Dependencies
#********************
PROJECT_ROOT_DIR = ${CURDIR}/../..
include ../../make/platform.mk
include ../../make/tensorflow.mk
include ../../make/libs.mk
include ../../make/unity.mk

#********************
# App specific
#********************
ifeq ($(TARGET), x86)
	APP = operators_test
else # must be xcore
	APPFLAGS += -fcmdline-buffer-bytes=1024
	APP = operators_test.xe
endif

APPFLAGS := 
#APPFLAGS += -DNDEBUG  # define this for remove debug and profiling 
#APPFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS # define this to remove logging

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += $(UNITY_INCLUDES)

# source file paths
VPATH := $(UNITY_VPATH)
VPATH += $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

SOURCES := $(UNITY_SOURCES)
SOURCES += $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += test_allocator.cpp
SOURCES += test_dispatcher.cpp
SOURCES += test_execution_plan.cpp
SOURCES += main.cpp

include ../../make/rules.mk

.PHONY: all clean $(APP)

all: $(BIN_DIR)/$(APP)

# $(OBJ_DIR)/%.o: %.S
# 	@mkdir -p $(dir $@)
# 	$(AS) $(ASFLAGS) -o $@ -c $<

# $(OBJ_DIR)/%.o: %.c
# 	@mkdir -p $(dir $@)
# 	$(CC) $(CCFLAGS) $(APPFLAGS) $(INCLUDES) -o $@ -c $<

# $(OBJ_DIR)/%.o: %.cpp
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CXXFLAGS) $(APPFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
