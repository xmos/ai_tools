

export PLATFORM ?= xcore

include ./etc/platform/$(strip $(PLATFORM)).mk
include ./etc/common.mk

export TARGET_DEVICE
export BUILD_DIR := .build
export BIN_DIR   := bin
export OBJ_DIR   := $(BUILD_DIR)/$(PLATFORM)

export APP_FLAGS
export XCC_FLAGS
export CC_FLAGS
export CXX_FLAGS
export AS_FLAGS

export LIB_NAME

export PLATFORM_FLAGS

LIBRARY_MAKEFILE := $(abspath etc/lib_makefile)

$(info Building for $(PLATFORM_NAME))


INCLUDES :=
SOURCE_FILES := 
VPATH :=


include app.mk
$(call check_defined, APP_NAME, Name of the application being built)
ifeq ($(strip $(PLATFORM)),$(strip xcore))
  $(call check_defined, TARGET_DEVICE, The xcore device being targeted)
  PLATFORM_FLAGS += -target=$(TARGET_DEVICE)
endif
APP_EXE = $(APP_NAME)$(PLATFORM_EXE_SUFFIX)

OBJ_FILES := $(SOURCE_FILES:%=$(OBJ_DIR)/$(APP_NAME)/%.o)

$(info Application dependencies: $(DEPENDENCIES))
DEP_MK_PATHS := $(DEPENDENCIES:%=./etc/%.mk)

API_CHECK = 1
include $(DEP_MK_PATHS)
API_CHECK = 0

DEPENDENCY_TARGETS := $(DEPENDENCIES:%=lib_target_%)

#######################################################
# MAKE TARGETS
#######################################################
.PHONY: help all build clean $(DEPENDENCY_TARGETS)

help:
	$(info ***********************)
	$(info nn_operators_test: Unit tests for the functions in lib_nn)
	$(info ***********************)

all: build

build: $(BIN_DIR)/$(APP_EXE)

lib_target_%: LIB_PATH = $($(subst lib_target_,,$@)_PATH)
lib_target_%: LIB_NAME = $(subst lib_target_,,$@)

$(DEPENDENCY_TARGETS): BUILD_DIR := $(abspath $(BUILD_DIR))
$(DEPENDENCY_TARGETS): OBJ_DIR := $(abspath $(OBJ_DIR))

$(DEPENDENCY_TARGETS): 
	@$(MAKE) -s -e --makefile=$(LIBRARY_MAKEFILE) -C $(LIB_PATH) build

MAP_xc = XCC
MAP_c = CC
MAP_cc = CC
MAP_cpp = CXX
MAP_S = AS


$(filter %.c.o, $(OBJ_FILES)):   OBJ_COMP := $(CC)
$(filter %.cc.o, $(OBJ_FILES)):  OBJ_COMP := $(CC)
$(filter %.xc.o, $(OBJ_FILES)):  OBJ_COMP := $(XCC)
$(filter %.cpp.o, $(OBJ_FILES)): OBJ_COMP := $(CXX)
$(filter %.S.o, $(OBJ_FILES)):   OBJ_COMP := $(AS)

$(filter %.c.o, $(OBJ_FILES)):   OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(c_FLAGS)   $(CC_FLAGS)
$(filter %.cc.o, $(OBJ_FILES)):  OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(cc_FLAGS)  $(CC_FLAGS)
$(filter %.xc.o, $(OBJ_FILES)):  OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(xc_FLAGS)  $(XCC_FLAGS)
$(filter %.cpp.o, $(OBJ_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(cpp_FLAGS) $(CXX_FLAGS)
$(filter %.S.o, $(OBJ_FILES)):   OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(S_FLAGS)   $(AS_FLAGS)

$(filter %.c.o, $(OBJ_FILES)):   OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(c_INCLUDES)   $(CC_INCLUDES)
$(filter %.cc.o, $(OBJ_FILES)):  OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(cc_INCLUDES)  $(CC_INCLUDES)
$(filter %.xc.o, $(OBJ_FILES)):  OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(xc_INCLUDES)  $(XCC_INCLUDES)
$(filter %.cpp.o, $(OBJ_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(cpp_INCLUDES) $(CXX_INCLUDES)
$(filter %.S.o, $(OBJ_FILES)):   OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(S_INCLUDES)   $(AS_INCLUDES)

$(OBJ_FILES): $(OBJ_DIR)/$(APP_NAME)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<)
	@$(OBJ_COMP) $(OBJ_FLAGS) $(addprefix -I,$(OBJ_INCLUDES)) -o $@ -c $<

-include $(OBJ_FILES:%.o=%.d)


# NOTE: For some reason the recursive wildcard call only sometimes
# finds the .o files for the application, even though they're
# compiled first... which is the reason for the filtering below.
# Because on the occasions where rwildcard DOES find them, there
# is a duplicate symbol error for every symbol in the app source.

# NOTE: I can't figure out how to get make to realize when the
# binary doesn't need to be regenerated, so repeated calls to
# 'make all' seem to generate it repeatedly. I suspect it is
# to do with the dependency targets being phony?

$(BIN_DIR)/$(APP_EXE): $(OBJ_FILES) $(DEPENDENCY_TARGETS)
	$(call mkdir_cmd,$@)
	$(info Linking binary $@)
	@$(XCC) $(LDFLAGS)                      \
		$(APP_FLAGS)                        \
		$(PLATFORM_FLAGS)                   \
		$(OBJ_FILES)                        \
		$(filter-out $(OBJ_DIR)/$(APP_NAME)/%,$(call rwildcard,$(OBJ_DIR),*.o))    \
		-o $(BIN_DIR)/$(APP_EXE)

clean:
	$(info Cleaning project...)
	rm -rvf $(BIN_DIR)
	rm -rvf $(BUILD_DIR)


