UNITY_PATH := $(PROJECT_ROOT_DIR)/third_party/unity

UNITY_INCLUDES := \
	-I$(UNITY_PATH)/src \
	-I$(UNITY_PATH)/extras/memory/src \
	-I$(UNITY_PATH)/extras/fixture/src

UNITY_VPATH := \
	$(UNITY_PATH)/src \
	$(UNITY_PATH)/extras/fixture/src \
	$(UNITY_PATH)/extras/memory/src

UNITY_SOURCES := \
	unity.c \
	unity_fixture.c \
	unity_memory.c
