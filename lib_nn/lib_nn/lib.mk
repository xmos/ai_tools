

THIS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

LIB_INCLUDES := -I$(THIS_DIR)src -I$(THIS_DIR)/api

LIB_SRC_DIR = $(THIS_DIR)src

LIB_XC_SOURCES := $(wildcard $(LIB_SRC_DIR)/c/*.xc)
LIB_C_SOURCES := $(wildcard $(LIB_SRC_DIR)/c/*.c)
LIB_ASM_SOURCES := $(wildcard $(LIB_SRC_DIR)/asm/*.S)

LIB_XC_OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(subst $(DEP_PATH)/,,$(patsubst %.xc,%.o,$(LIB_XC_SOURCES))))
LIB_C_OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(subst $(DEP_PATH)/,,$(patsubst %.c,%.o,$(LIB_C_SOURCES))))
LIB_ASM_OBJECT_FILES := $(addprefix $(OBJ_DIR)/,$(subst $(DEP_PATH)/,,$(patsubst %.S,%.o,$(LIB_ASM_SOURCES))))


LIB_SOURCES := $(LIB_C_SOURCES) $(LIB_XC_SOURCES) $(LIB_ASM_SOURCES)
LIB_OBJECT_FILES := $(LIB_XC_OBJECT_FILES) $(LIB_C_OBJECT_FILES) $(LIB_ASM_OBJECT_FILES)
LIB_OBJECTS := $(LIB_SOURCES) $(LIB_OBJECT_FILES)


ALL_OBJECT_FILES += $(LIB_OBJECT_FILES)
ALL_INCLUDES += $(LIB_INCLUDES)
ALL_OBJECTS += $(LIB_OBJECTS)

# $(info lib!! $(LIB_SOURCES))

$(LIB_ASM_OBJECT_FILES): $(OBJ_DIR)/%.o: $(DEP_PATH)/%.S
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(AS) $(ASFLAGS) -o $@ -c $<

$(LIB_C_OBJECT_FILES): $(OBJ_DIR)/%.o: $(DEP_PATH)/%.c
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(CC) $(CCFLAGS) $(ALL_INCLUDES) -o $@ -c $<

$(LIB_XC_OBJECT_FILES): $(OBJ_DIR)/%.o: $(DEP_PATH)/%.xc
	$(call mkdir_cmd,$@)
	$(info Compiling $<..)
	@$(XCC) $(XCCFLAGS) $(ALL_INCLUDES) -o $@ -c $<

	