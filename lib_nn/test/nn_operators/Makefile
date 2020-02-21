

PLATFORM ?= xcore

include ./etc/platform/$(strip $(PLATFORM)).mk
include ./etc/common.mk

BUILD_DIR := .build
BIN_DIR   := bin
OBJ_DIR   := $(BUILD_DIR)/$(PLATFORM)

help: app_help; @echo > NUL


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
APP_EXE_TARGET = $(BIN_DIR)/$(APP_EXE)


OBJ_FILES := $(SOURCE_FILES:%=$(OBJ_DIR)/$(APP_NAME)/%.o)

APP_SOURCE_FILES := $(SOURCE_FILES)
APP_OBJECT_FILES := $(OBJ_FILES)

$(APP_EXE_TARGET): $(APP_OBJ_FILES)

#######################################################
# DEPENDENCY TARGETS
#######################################################
define newline


endef

$(foreach dep,$(DEPENDENCIES),$(eval\
$(newline)  $$(dep)_PATH_ABS:=$$(abspath $$($$(dep)_PATH))\
$(newline)  $$(dep)_MK_PATH:=./etc/$$(dep).mk\
))


# Include each dependency's .mk file. Set [lib_name]_SOURCE_FILES
$(foreach dep,$(DEPENDENCIES),$(eval                 \
$(newline)  SOURCE_FILES:=                           \
$(newline)  LIB_PATH:=                               \
$(newline)  include $($(dep)_MK_PATH)                \
$(newline)  $(dep)_SOURCE_FILES:=$$(strip $$(SOURCE_FILES))    \
))

# Set [lib_name]_OBJECT_FILES to the object files it generates
$(foreach dep,$(DEPENDENCIES), $(eval \
$(newline)  $(dep)_OBJECT_FILES:=$(patsubst $(strip $($(dep)_PATH_ABS))/%,$(OBJ_DIR)/$(dep)/%.o,$($(dep)_SOURCE_FILES)) \
))

#
# Each dependency becomes a static library which is a prerequisite for the executable
#
$(APP_EXE_TARGET): $(DEPENDENCIES:%=$(BUILD_DIR)/%.a)

#
# Library archive build targets
#
DEP_ARCHIVES = $(DEPENDENCIES:%=$(BUILD_DIR)/%.a)
ar_get_lib_name=$(basename,$(subst $(BUILD_DIR)/,,$(1)))

$(BUILD_DIR)/%.a: LIB_NAME = $(patsubst $(BUILD_DIR)/%.a,%,$@)
$(BUILD_DIR)/%.a: LIB_OBJECT_FILES = $($(LIB_NAME)_OBJECT_FILES)

$(DEP_ARCHIVES): $(BUILD_DIR)/%.a :
	$(info Generating archive $(LIB_NAME).a)
	@$(AR) $(AR_FLAGS) $@ $(LIB_OBJECT_FILES)
	
#
# This sets up a source file prerequisite for each object file in each library
#	(TODO: test whether it's working by adding something to the source file name to break it)
# $(info $(foreach dep,$(DEPENDENCIES), $$($(dep)_OBJECT_FILES): $(OBJ_DIR)/$(dep)/%.o : $(strip $($(dep)_PATH_ABS))/%$(newline)))
$(eval $(foreach dep,$(DEPENDENCIES), $$($(dep)_OBJECT_FILES): $(OBJ_DIR)/$(dep)/%.o : $(strip $($(dep)_PATH_ABS))/%$(newline)))

#
# Make each library's object files prerequisites for its static library
#
# $(info $(foreach dep,$(DEPENDENCIES),$(BUILD_DIR)/$(dep).a: $($(dep)_OBJECT_FILES) $(newline)))
$(eval $(foreach dep,$(DEPENDENCIES),$(BUILD_DIR)/$(dep).a: $($(dep)_OBJECT_FILES) $(newline)))

DEP_OBJECT_FILES := $(foreach dep,$(DEPENDENCIES),$($(dep)_OBJECT_FILES))

get_lib_name=$(firstword $(subst /, ,$(subst $(OBJ_DIR)/,,$(1))))
get_file_rel_path_from_lib=$(subst $(OBJ_DIR)/$(call get_lib_name,$(1))/,,$(1))
get_lib_abs_path=$($(call get_lib_name,$(1))_PATH_ABS)
get_file_abs_path=$(basename $(strip $(call get_lib_abs_path,$(1)))/$(call get_file_rel_path_from_lib,$(1)))
get_src_file=$(basename $(strip $($(firstword $(subst /, ,$(subst $(OBJ_DIR)/,,$@)..))_PATH_ABS))/$(subst $(OBJ_DIR)/,,$(1)))

$(DEP_OBJECT_FILES): $(OBJ_DIR)/%.o :
	$(call mkdir_cmd,$@)
	$(info Compiling $(call get_file_abs_path,$@))
	@$(OBJ_COMP) $(OBJ_FLAGS) $(addprefix -I,$(OBJ_INCLUDES)) -o $@ -c $(call get_file_abs_path,$@)




#######################################################
# APPLICATION TARGETS
#######################################################



ALL_OBJECT_FILES := $(APP_OBJECT_FILES) $(DEP_OBJECT_FILES)

$(filter %.c.o,   $(ALL_OBJECT_FILES)): OBJ_COMP := $(CC)
$(filter %.cc.o,  $(ALL_OBJECT_FILES)): OBJ_COMP := $(CC)
$(filter %.xc.o,  $(ALL_OBJECT_FILES)): OBJ_COMP := $(XCC)
$(filter %.cpp.o, $(ALL_OBJECT_FILES)): OBJ_COMP := $(CXX)
$(filter %.S.o,   $(ALL_OBJECT_FILES)): OBJ_COMP := $(AS)

$(filter %.c.o,   $(ALL_OBJECT_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(c_FLAGS)   $(CC_FLAGS)
$(filter %.cc.o,  $(ALL_OBJECT_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(cc_FLAGS)  $(CC_FLAGS)
$(filter %.xc.o,  $(ALL_OBJECT_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(xc_FLAGS)  $(XCC_FLAGS)
$(filter %.cpp.o, $(ALL_OBJECT_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(cpp_FLAGS) $(CXX_FLAGS)
$(filter %.S.o,   $(ALL_OBJECT_FILES)): OBJ_FLAGS := $(PLATFORM_FLAGS) $(APP_FLAGS) $(S_FLAGS)   $(AS_FLAGS)

$(filter %.c.o,   $(ALL_OBJECT_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(c_INCLUDES)   $(CC_INCLUDES)
$(filter %.cc.o,  $(ALL_OBJECT_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(cc_INCLUDES)  $(CC_INCLUDES)
$(filter %.xc.o,  $(ALL_OBJECT_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(xc_INCLUDES)  $(XCC_INCLUDES)
$(filter %.cpp.o, $(ALL_OBJECT_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(cpp_INCLUDES) $(CXX_INCLUDES)
$(filter %.S.o,   $(ALL_OBJECT_FILES)): OBJ_INCLUDES := $(PLATFORM_INCLUDES) $(INCLUDES) $(S_INCLUDES)   $(AS_INCLUDES)


$(APP_OBJECT_FILES): $(OBJ_DIR)/$(APP_NAME)/%.o: %
	$(call mkdir_cmd,$@)
	$(info Compiling $<)
	@$(OBJ_COMP) $(OBJ_FLAGS) $(addprefix -I,$(OBJ_INCLUDES)) -o $@ -c $<

-include $(ALL_OBJECT_FILES:%.o=%.d)


$(APP_EXE_TARGET): $(OBJ_FILES) $(DEPENDENCY_ARCHIVES)
	$(call mkdir_cmd,$@)
	$(info Linking binary $@)
	@$(XCC) $(LDFLAGS)                      \
		$(APP_FLAGS)                        \
		$(PLATFORM_FLAGS)                   \
		$(OBJ_FILES)                        \
		-o $(BIN_DIR)/$(APP_EXE)            \
		$(DEP_ARCHIVES)





#######################################################
# OTHER TARGETS
#######################################################

.PHONY: help all build clean run FORCE MSG_PLATFORM MSG_DEPENDENCIES app_help

all: build; @echo > NUL

MSG_PLATFORM:
	$(info Building for $(PLATFORM_NAME))
MSG_DEPENDENCIES:
	$(info Application dependencies: $(DEPENDENCIES))

build: MSG_PLATFORM MSG_DEPENDENCIES  $(APP_EXE_TARGET)
	$(info Done.)

clean:
	$(info Cleaning project...)
	rm -rvf $(BIN_DIR)
	rm -rvf $(BUILD_DIR)


run: build
	xsim $(APP_EXE_TARGET)