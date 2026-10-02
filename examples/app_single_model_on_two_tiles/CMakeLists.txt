cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_single_model_on_two_tiles)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/export.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET XK-EVK-XU316)
endif()
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)
set(APP_XC_SRCS src/main.xc)
set(APP_C_SRCS
  src/model_weights.c
)
set(APP_CXX_SRCS
  src/support.cpp
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
  -Wno-unknown-pragmas
  -Wno-bidirectional-buffered-port
  -DTFLMC_XCORE_PROFILE
)

set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
