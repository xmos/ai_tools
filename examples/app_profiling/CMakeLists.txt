cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_profiling)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)
set(APP_CXX_SRCS
  src/main.cpp
  src/model.tflite.cpp
)

list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -lquadflash
  -DTFLMC_XCORE_PROFILE
)

set(APP_DEPENDENT_MODULES "")
XMOS_REGISTER_APP()

include(${CMAKE_CURRENT_LIST_DIR}/../aitools.cmake)
