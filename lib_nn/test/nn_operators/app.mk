
#(Required) Determines the name of the generated executable
APP_NAME = nn_operators_test

SRC_DIR = src
TARGET_DEVICE = XU316-1024-QF60-C20

DEPENDENCIES := lib_nn Unity

lib_nn_PATH := ../../lib_nn
Unity_PATH := ../../Unity

VPATH += $(SRC_DIR)

SOURCE_FILES += $(wildcard $(SRC_DIR)/*.xc)
SOURCE_FILES += $(wildcard $(SRC_DIR)/*.c)
