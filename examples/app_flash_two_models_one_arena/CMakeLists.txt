cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_flash_two_models_one_arena)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)
set(APP_CXX_SRCS
  src/main.cpp
  src/model1.tflite.cpp
  src/model2.tflite.cpp
)

list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -Wall
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -DSHARED_TENSOR_ARENA
  -lquadflash
)

set(APP_DEPENDENT_MODULES "")
XMOS_REGISTER_APP()

include(${CMAKE_CURRENT_LIST_DIR}/../aitools.cmake)
