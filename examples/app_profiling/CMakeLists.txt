cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_profiling)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET XK-EVK-XU316)
endif()
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)

list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -DTFLMC_XCORE_PROFILE
)

set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
