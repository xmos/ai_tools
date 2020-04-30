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
	APP = lib_ops_test.xe
endif

#CCFLAGS += -DNDEBUG
#CXXFLAGS += -DNDEBUG

CCFLAGS += -DENABLE_TRACING
CXXFLAGS += -DENABLE_TRACING

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
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APP)
