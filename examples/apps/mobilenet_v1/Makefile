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
#APPFLAGS := -DNDEBUG
APPFLAGS := -g
#APPFLAGS += -DENABLE_TRACING
APPFLAGS += -DENABLE_BENCHMARKING

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
VPATH += src

# sources
SOURCES := $(TENSORFLOW_SOURCES)
SOURCES += $(LIBS_SOURCES)
SOURCES += mobilenet_ops_resolver.cc
SOURCES += mobilenet_v1.c
SOURCES += ai_engine.cc 
SOURCES += main.c
ifeq ($(TARGET), xcore)
	SOURCES += main.xc
endif

include ../../../make/rules.mk

.PHONY: all flash ddr clean $(APP)

$(BIN_DIR)/$(APP): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $(APPFLAGS) $(OBJECT_FILES) -o $(BIN_DIR)/$(APP)

flash: APPFLAGS += -DUSE_SWMEM
flash: $(BIN_DIR)/$(APP)
	cd $(BIN_DIR) && xobjdump --strip $(APPBASE).xe
	cd $(BIN_DIR) && xobjdump --split $(APPBASE).xb
	#cd $(BIN_DIR) && xflash --write-all image_n0c0.swmem --target XCORE-AI-EXPLORER

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
