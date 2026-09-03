cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_no_flash)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET XK-EVK-XU316)
endif()
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
  -Wall
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
)

set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
