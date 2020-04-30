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
ifeq ($(TARGET), x86)
	APP = mobilenet
else # must be xcore
	CCFLAGS += -fcmdline-buffer-bytes=1024
	CXXFLAGS += -fcmdline-buffer-bytes=1024
	APP = mobilenet.xe
endif

CCFLAGS += -DNDEBUG
CXXFLAGS += -DNDEBUG

# CCFLAGS += -DENABLE_TRACING
# CXXFLAGS += -DENABLE_TRACING

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)
VPATH += src

SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += mobilenet_ops_resolver.cc
SOURCES += mobilenet_v1.c
SOURCES += main.cc 

OBJECT_FILES := $(patsubst %.cpp,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.S,%.o,$(SOURCES)))))
OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(OBJECT_FILES))
OBJECTS := $(SOURCES) $(OBJECT_FILES)

.PHONY: all clean $(APP)

$(OBJ_DIR)/%.o: %.S
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) -o $@ -c $<

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)