cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_flash_single_model)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET src/XCORE-AI-EXPLORER-800.xn)
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)
set(APP_XC_SRCS src/main.cpp)
set(APP_CXX_SRCS
  src/model.tflite.cpp
)

list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -Wall
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -lquadflash
)

set(APP_DEPENDENT_MODULES "")
XMOS_REGISTER_APP()

include(${CMAKE_CURRENT_LIST_DIR}/../aitools.cmake)
