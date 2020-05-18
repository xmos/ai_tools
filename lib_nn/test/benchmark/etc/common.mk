

ifeq ($(OS),Windows_NT)
  ifeq ($(findstring windows32,$(shell uname -s)),windows32)
    mkdir_cmd = @test -d $(subst /,\,$(dir $(1))) || mkdir $(subst /,\,$(dir $(1)))
  else
    mkdir_cmd = @mkdir -p $(dir $(1))
  endif
else
  mkdir_cmd = @mkdir -p $(dir $(1))
endif

check_defined = $(strip $(foreach 1,$1, $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = $(if $(value $1),, $(error Undefined $1$(if $2, ($2))))

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

define newline


endef

#####
# rename_variables(prefix, var_names[])
#    Renames each of the variables X in argument $(2) to $(1)_X, and then
#    deletes X
rename_variables=$(eval $(foreach var,$(2),$(1)_$(var):=$($(var))$(newline)$(var):=$(newline)))
# rename_variables_print=$(info $(foreach var,$(2),$(1)_$(var):=$($(var))$(newline)undefine $(var)$(newline)))



###
# Load the dependency with name $(1)
# Afterwards, rename each variable X in $(2) to $(1)_X
# The optional third argument $(3) is the .mk file to be
# loaded. If $(3) is not provided, ./etc/$(1).mk will be
# loaded instead.
define load_dependency_
  MK_FILE := ./etc/$(1).mk
  ifneq ($$(strip $(3)),)
    MK_FILE := $(3)
  endif
  MK_FILE := $$(abspath $$(MK_FILE))
  include $$(MK_FILE)
  $$(call rename_variables,$(1),$(2))
endef
load_dependency=$(eval $(call load_dependency_,$(1),$(2),$(3)))



MAP_COMP_c   = CC
MAP_COMP_cc  = CC
MAP_COMP_xc  = XCC
MAP_COMP_cpp = CXX
MAP_COMP_S   = AS