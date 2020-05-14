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
APPFLAGS := -DNDEBUG
#APPFLAGS += -DENABLE_TRACING
APPFLAGS += -DENABLE_BENCHMARKING

ifeq ($(TARGET), x86)
	APP = mobilenet
else # must be xcore
	APP = mobilenet.xe
endif

CCFLAGS += -DNDEBUG
CXXFLAGS += -DNDEBUG

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

# sources
SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += mobilenet_ops_resolver.cc
SOURCES += mobilenet_v1.c
SOURCES += ai_engine.cc 
ifeq ($(TARGET), x86)
	SOURCES += x86_main.cc
else # must be xcore
	SOURCES += xcore_main.xc
endif

include ../../../make/rules.mk

.PHONY: all clean $(APP)

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
