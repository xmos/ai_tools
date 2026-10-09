cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_no_flash)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET XK-EVK-XU316)
endif()

set(APP_COMPILER_FLAGS
  -O3
  -g
  -mcmodel=large
)

if(NOT DEFINED APP_CXX_SRCS)
  set(APP_CXX_SRCS
    src/default.cpp
  )
endif()

set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
