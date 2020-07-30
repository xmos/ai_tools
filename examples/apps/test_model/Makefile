#********************
# Dependencies
#********************
PROJECT_ROOT_DIR = ${CURDIR}/../../..
include ../../../make/platform.mk
include ../../../make/tensorflow.mk
include ../../../make/libs.mk

#********************
# App specific
#********************
APPFLAGS := 
#APPFLAGS += -DNDEBUG  # define this for remove debug and profiling 
#APPFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS # define this to remove logging

ifeq ($(TARGET), x86)
	APP = test_model
else # must be xcore
	APPFLAGS += -fcmdline-buffer-bytes=1024
	APP = test_model.xe
endif

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += main.cc

include ../../../make/rules.mk

.PHONY: all clean $(APP)

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
