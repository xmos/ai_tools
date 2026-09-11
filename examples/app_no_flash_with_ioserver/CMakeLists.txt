cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_no_flash_with_ioserver)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES src)
set(APP_XC_SRCS src/main.xc)
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
  -Xcompiler-xc
  -mtimer-abi=hwtimers
  "-DUSB_TILE=tile[1]"
  -DXUD_CORE_CLOCK=600
  -lquadflash
)

set(APP_DEPENDENT_MODULES "lib_xud(2.3.0)")
XMOS_REGISTER_APP()

include(${CMAKE_CURRENT_LIST_DIR}/../aitools.cmake)