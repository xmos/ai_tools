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
APPBASE = mobilenet
APPFLAGS :=
APPFLAGS += -DNDEBUG  # define this for remove debug and profiling 
APPFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS # define this to remove logging

ifeq ($(TARGET), x86)
	APP = $(APPBASE)
else # must be xcore
	APPFLAGS += -fxscope
	APPFLAGS += config.xscope
	APP = $(APPBASE).xe
endif

INCLUDES := $(TENSORFLOW_INCLUDES)
INCLUDES += $(LIBS_INCLUDES)
INCLUDES += -Isrc

# source file paths
VPATH := $(TENSORFLOW_VPATH)
VPATH += $(LIBS_VPATH)

MODEL_SOURCE_FILE := src/mobilenet_v1.c

# sources
SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += $(MODEL_SOURCE_FILE)
SOURCES += src/mobilenet_ops_resolver.cc
SOURCES += src/inference_engine.cc 
SOURCES += src/main.c
ifeq ($(TARGET), xcore)
	SOURCES += src/main.xc
endif

include ../../../make/rules.mk

.PHONY: all flash ddr clean $(APP)

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

$(BIN_DIR)/image_n0c0.swmem: $(MODEL_SOURCE_FILE)
	cd $(BIN_DIR) && xobjdump --strip $(APPBASE).xe
	cd $(BIN_DIR) && xobjdump --split $(APPBASE).xb
	$(FLASH) --write-all $@ --target XCORE-AI-EXPLORER

flash: APPFLAGS += -DUSE_SWMEM
flash: $(BIN_DIR)/$(APP) $(BIN_DIR)/image_n0c0.swmem

ddr: APPFLAGS += -DUSE_EXTMEM
ddr: $(BIN_DIR)/$(APP)

clean:
	-@rm -rvf $(OBJECT_FILES) $(BIN_DIR)/$(APPBASE).xe
	-@rm -rvf $(BIN_DIR)/$(APPBASE).xb
	-@rm -rvf $(BIN_DIR)/*.bin
	-@rm -rvf $(BIN_DIR)/*.xn
	-@rm -rvf $(BIN_DIR)/*.swmem
	-@rm -rvf $(BIN_DIR)/*.txt
	-@rm -rvf $(BIN_DIR)/*.xscope
