cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(test_sim)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET XK-EVK-XU316)
endif()

if(NOT DEFINED APP_HW_TARGET_VX AND NOT BUILD_NATIVE)
  set(APP_HW_TARGET_VX "${XMOS_SANDBOX_DIR}/ai_tools/ai_tools/XK-EVK-XU416.xml")
endif()

set(APP_COMPILER_FLAGS
  -O3
  -g
)

if(NOT XMOS_IS_RV)
  list(APPEND APP_COMPILER_FLAGS -mcmodel=large)
endif()

if(NOT DEFINED APP_CXX_SRCS)
  set(APP_CXX_SRCS
    src/default.cpp
  )
endif()

set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
