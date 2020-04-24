

PLATFORM ?= xcore

include ./etc/platform/$(strip $(PLATFORM)).mk
include ./etc/common.mk

help: app_help; @echo > NUL

#######################################################
# Variables .mk files probably shouldn't touch
#######################################################
BUILD_DIR := .build
BIN_DIR   := bin
OBJ_DIR   := $(BUILD_DIR)/$(PLATFORM)
LIB_BUILD_DIR := $(OBJ_DIR)/lib
EXE_DIR   := $(BIN_DIR)/$(PLATFORM)

#######################################################
# Variables .mk files can modify (add to, usually)
#######################################################

INCLUDES :=

# TODO: Make this per-dependency with this as a default?
SOURCE_FILE_EXTENSIONS := c cc xc cpp S




#######################################################
# CONSTANTS
#######################################################

###
# Common application config. If it exists, it is loaded
# prior to any config-specific .mk file.
APP_COMMON_MK_FILE_NAME = app.common.mk

###
# .mk file for the default config.
APP_MK_FILE_DEFAULT = app.mk

###
# Pattern for config-specific app config .mk files
APP_CONFIG_MK_PATTERN = app.%.mk

###
# Empty string, used for comparisons
EMPTY_STR :=

###
# Variables that are 'local' to an application
#  (these get turned into APP_[VARIABLE_NAME] after the
#	dependency's .mk file is loaded)
APP_LOCAL_VARIABLES := SOURCE_FILES SOURCE_DIRS SOURCE_FILE_EXTS LIBRARIES

###
# Variables that are 'local' to a dependency
#  (these get turned into [lib_name]_[VARIABLE_NAME] after the
#	dependency's .mk file is loaded)
LIB_LOCAL_VARIABLES := SOURCE_FILES SOURCE_DIRS SOURCE_FILE_EXTS LIBRARIES

###
# Path of this app. (current working directory)
#
APP_PATH := .

#######################################################
# OTHER STUFF
#######################################################

VPATH :=


#######################################################
# APPLICATION OPTIONS
#######################################################

#
# If the application makefile sets this to any value other than 1, no static 
# libraries (.a files) will be created for dependencies, and the executable
# will be linked directly against library object files.
#
BUILD_STATIC_LIBRARIES := 1

#######################################################
# APPLICATION MAKE FILE(S)
#######################################################

SOURCE_FILE_EXTS := $(SOURCE_FILE_EXTENSIONS)
APP_REQUIRE_CONFIG_SPECIFIC := 1

APP_COMMON_MK_FILE := $(wildcard $(APP_COMMON_MK_FILE_NAME))
ifneq ($(APP_COMMON_MK_FILE),$(EMPTY_STR))
  $(info Including: $(APP_COMMON_MK_FILE))
  include $(APP_COMMON_MK_FILE)
endif

ifeq ($(APP_REQUIRE_CONFIG_SPECIFIC),1)
  #
  # Check whether to load default application makefile
  #   or to load a specific configuration makefile
  #
  APP_MK_FILE_NAME := $(APP_MK_FILE_DEFAULT)
  ifdef CONFIG
    $(info Using config: $(CONFIG))
    EXE_DIR := $(EXE_DIR)/$(CONFIG)
    APP_MK_FILE_NAME = $(CONFIG:%=$(APP_CONFIG_MK_PATTERN))
  endif
  
  #
  # Load application make file. Give error if not found
  #
  APP_MK_FILE := $(wildcard $(APP_MK_FILE_NAME))
  
  ifeq ($(APP_MK_FILE),$(EMPTY_STR))
    $(error Could not find application make file: $(APP_MK_FILE_NAME))
  endif
  
  include $(APP_MK_FILE)
endif

#
# Check whether the application make files have defined the
# required variables:
#	APP_NAME: The name of the application
#	TARGET_DEVICE: The xcore device being targeted (only required if the target platform is xcore)
#
$(call check_defined, APP_NAME, Name of the application being built)
ifeq ($(strip $(PLATFORM)),$(strip xcore))
  $(call check_defined, TARGET_DEVICE, The xcore device being targeted)
  PLATFORM_FLAGS += -target=$(TARGET_DEVICE)
endif

#
# Application executable name and file
#
APP_EXE_FILE_NAME = $(APP_NAME)$(PLATFORM_EXE_SUFFIX)
APP_EXE_FILE = $(EXE_DIR)/$(APP_EXE_FILE_NAME)

#
# Rename application variables
#
$(call rename_variables,APP,$(APP_LOCAL_VARIABLES))


#
# Includes both APP and dependencies. Useful for dealing with things like
# object files
#
APP_COMPONENTS := APP $(DEPENDENCIES)


#######################################################
# LOADING DEPENDENCY MAKE FILES
#######################################################

#
# Load each dependency using the load_dependency function from
#  etc/common.mk. If [LibName]_MK_FILE is defined, then $([LibName]_MK_FILE) will
#  be included for that dependency. Otherwise ./etc/[LibName].mk will be included.
# If the dependency's make file sets any of the variables in LIB_LOCAL_VARIABLES, they
#  will be renamed to [LibName]_[VariableName]
#
$(foreach dep,$(DEPENDENCIES),$(call load_dependency,$(dep),$(LIB_LOCAL_VARIABLES),$$($(dep)_MK_FILE)))


#######################################################
# HOUSEKEEPING 
#######################################################

# Done including user makefiles, can use this area to do a bit of cleanup
#

BUILD_STATIC_LIBRARIES := $(strip $(BUILD_STATIC_LIBRARIES))

EXTERNAL_LIBRARIES := $(strip $(foreach comp,$(APP_COMPONENTS),$($(comp)_LIBRARIES)))

#######################################################
# COMPONENT SOURCE FILE SEARCH
#######################################################

#
# Search for source files using the [component]_SOURCE_DIRS, 
#  [component]_SOURCE_FILE_EXTS, and [component]_PATH variables
# Also convert to absolute paths
# 
# $(info $(foreach comp,$(APP_COMPONENTS),$(comp)_SOURCE_FILES += $(strip $(foreach src_dir,$($(comp)_SOURCE_DIRS),$(call rwildcard,$($(comp)_PATH)/$(src_dir),$($(comp)_SOURCE_FILE_EXTS:%=*.%))))$(newline)))
$(eval $(foreach comp,$(APP_COMPONENTS), \
                $(comp)_SOURCE_FILES += $(strip $(abspath $(foreach src_dir,$($(comp)_SOURCE_DIRS),\
                                      $(call rwildcard,$($(comp)_PATH)/$(src_dir),$($(comp)_SOURCE_FILE_EXTS:%=*.%)))))$(newline)))


#######################################################
# COMPONENT OBJECT FILES
#######################################################

#
# Set [lib_name]_OBJECT_FILES to the object files it generates, relative to $(OBJ_DIR)
#
# $(info $(foreach comp,$(APP_COMPONENTS), $(comp)_OBJECT_FILES := $(strip \
# 				         $(patsubst $(abspath $($(comp)_PATH))/%,$(OBJ_DIR)/$(comp)/%,$($(comp)_SOURCE_FILES:%=%.o))\
# 					) $(newline)))

$(eval $(foreach comp,$(APP_COMPONENTS), $(comp)_OBJECT_FILES := $(strip \
				         $(patsubst $(abspath $($(comp)_PATH))/%,$(OBJ_DIR)/$(comp)/%,$($(comp)_SOURCE_FILES:%=%.o))\
					) $(newline)))

DEPENDENCY_OBJECT_FILES := $(foreach dep,$(DEPENDENCIES),$($(dep)_OBJECT_FILES))
COMPONENT_OBJECT_FILES := $(APP_OBJECT_FILES) $(DEPENDENCY_OBJECT_FILES)

#######################################################
# OBJECT FILE TARGETS
#######################################################


#
# Every app object file has its source file as a prerequisite
#
$(APP_OBJECT_FILES): $(OBJ_DIR)/APP/%.o: %

#
# Use vpath to pattern out where to find source files for each dependency. e.g.
#
#   vpath SomeLibrary/%.c $(SomeLibrary_PATH)/..
#
# Uncomment this to see how it expands:
#    $(info $(foreach ext,$(SOURCE_FILE_EXTENSIONS),$(foreach dep,$(DEPENDENCIES), vpath $(dep)/%.$(ext) $($(dep)_PATH)/..$(newline))$(newline)))
$(eval $(foreach ext,$(SOURCE_FILE_EXTENSIONS),$(foreach dep,$(DEPENDENCIES), vpath $(dep)/%.$(ext) $($(dep)_PATH)/..$(newline))$(newline)))

#
# Every dependency object file has its source file as a prerequisite.
$(DEPENDENCY_OBJECT_FILES): $(OBJ_DIR)/%.o: %


#########
## Recipe-scoped variables for building objects.
#########

# OBJ_FILE_TYPE
# The source file's file type
$(eval $(foreach ext,$(SOURCE_FILE_EXTENSIONS), $(filter %.$(ext).o,$(COMPONENT_OBJECT_FILES)): OBJ_FILE_TYPE = $(ext)$(newline)))

# OBJ_COMPONENT
# The component (APP or one of the library names) from which this object file comes
$(eval $(foreach comp,$(APP_COMPONENTS), $(filter $(OBJ_DIR)/$(comp)/%.o,$(COMPONENT_OBJECT_FILES)): OBJ_COMPONENT = $(comp)$(newline)))

# OBJ_TOOL
# Maps from file extension to the tool type (not necessarily 1-to-1 mapping with
# file extension). This simplifies some of the code below.
$(COMPONENT_OBJECT_FILES): OBJ_TOOL = $(MAP_COMP_$(OBJ_FILE_TYPE))

# OBJ_COMPILER: Compilation program for this object
$(COMPONENT_OBJECT_FILES): OBJ_COMPILER = $($(OBJ_TOOL))

# $(1) - Component
# $(2) - Tool
# $(3) - File extension
ctf_combo_str = $(1)_$(2)_$(3) $(1)_$(2) $(1)_$(3) $(2)_$(3) $(1) $(2) $(3)
flags_combo_str = GLOBAL_FLAGS PLATFORM_FLAGS $(patsubst %,%_FLAGS,$(ctf_combo_str))
includes_combo_str = INCLUDES PLATFORM_INCLUDES $(patsubst %,%_INCLUDES,$(ctf_combo_str))

$(COMPONENT_OBJECT_FILES): OBJ_FLAGS = $(strip $(foreach grp,$(call flags_combo_str,$(OBJ_COMPONENT),$(OBJ_TOOL),$(OBJ_FILE_TYPE)),$($(grp))))
$(COMPONENT_OBJECT_FILES): OBJ_INCLUDES = $(strip $(foreach grp,$(call includes_combo_str,$(OBJ_COMPONENT),$(OBJ_TOOL),$(OBJ_FILE_TYPE)),$($(grp))))

###
# make target for all component object files.
#
$(COMPONENT_OBJECT_FILES):
	$(call mkdir_cmd,$@)
	$(info [$(OBJ_COMPONENT)] Compiling $<)

	@$(OBJ_COMPILER)                              \
	     $(OBJ_FLAGS)                             \
		 $(addprefix -I,$(OBJ_INCLUDES))          \
		 -o $@ -c $<

###
# If the -MMD flag is used when compiling, the .d files will contain additional header 
# file prerequisites for each object file. Otherwise it won't know to recompile if only
# header files have changed, for example.
-include $(COMPONENT_OBJECT_FILES:%.o=%.d)

#######################################################
# LIBRARY ARCHIVE FILES
#######################################################

ifeq ($(BUILD_STATIC_LIBRARIES),1)

#
# List of generated archive files. Needed for executable.
#
DEPENDENCY_ARCHIVES = $(DEPENDENCIES:%=$(LIB_BUILD_DIR)/%.a)

#
# Helper variables usable from within the archive file recipe
#
$(LIB_BUILD_DIR)/%.a: LIB_NAME = $(patsubst $(LIB_BUILD_DIR)/%.a,%,$@)
$(LIB_BUILD_DIR)/%.a: LIB_OBJECT_FILES = $($(LIB_NAME)_OBJECT_FILES)

#
# Make each dependency's object files prerequisites for its static library
#
# $(info $(foreach dep,$(DEPENDENCIES),$(BUILD_DIR)/$(dep).a: $($(dep)_OBJECT_FILES) $(newline)))
$(eval $(foreach dep,$(DEPENDENCIES),                    \
    $(LIB_BUILD_DIR)/$(dep).a: $($(dep)_OBJECT_FILES)    \
$(newline)))

#
# Recipe for building the archive files.
#   They get placed in $(LIB_BUILD_DIR)
#
$(DEPENDENCY_ARCHIVES): $(LIB_BUILD_DIR)/%.a :
	$(call mkdir_cmd,$@)
	$(info Generating archive $(LIB_NAME).a)
	$(AR) $(AR_FLAGS) $@ $(LIB_OBJECT_FILES)

endif # BUILD_STATIC_LIBRARIES


#######################################################
# APPLICATION TARGET
#######################################################
executable: external_libs
executable: app_objects

REQUIRED_LIBARIES := $(EXTERNAL_LIBRARIES)
REQUIRED_OBJECT_FILES := $(APP_OBJECT_FILES)

ifeq ($(BUILD_STATIC_LIBRARIES),1)
  executable: archives
  REQUIRED_LIBRARIES += $(DEPENDENCY_ARCHIVES)
else
  executable: lib_objects
  REQUIRED_OBJECT_FILES += $(DEPENDENCY_OBJECT_FILES)
endif

$(APP_EXE_FILE): $(REQUIRED_OBJECT_FILES) $(REQUIRED_LIBRARIES)
	$(call mkdir_cmd,$@)
	$(info Linking binary $@)
	@$(XCC) $(LD_FLAGS)                     \
		$(APP_FLAGS)                        \
		$(PLATFORM_FLAGS)                   \
		$(REQUIRED_OBJECT_FILES)            \
		-o $(APP_EXE_FILE)                  \
		$(REQUIRED_LIBRARIES)

executable: $(APP_EXE_FILE)

#######################################################
# OTHER TARGETS
#######################################################

.PHONY: help all check_required_libs build clean run FORCE MSG_PLATFORM MSG_DEPENDENCIES app_help

all: build;

MSG_PLATFORM:
	$(info Building for $(PLATFORM_NAME))
MSG_DEPENDENCIES:
	$(info Application dependencies: $(DEPENDENCIES))

build: MSG_PLATFORM MSG_DEPENDENCIES executable
	$(info Done.)

external_libs: $(EXTERNAL_LIBRARIES)

app_objects: $(APP_OBJECT_FILES)
lib_objects: $(DEPENDENCY_OBJECT_FILES)
objects: app_objects lib_objects

archives: $(DEPENDENCY_ARCHIVES)

clean:
	$(info Cleaning project...)
	rm -rf $(BIN_DIR)
	rm -rf $(BUILD_DIR)

