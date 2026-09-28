cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(device_test)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(APP_XSCOPE_SRCS src/config.xscope)
set(APP_INCLUDES src)

list(APPEND APP_COMPILER_FLAGS
  -report
  -O3
  -mcmodel=large
  -fxscope
  -Wno-xcore-fptrgroup
  -g
  -Xcompiler-xc
  -mtimer-abi=hwtimers
  -DUSB_TILE=tile[1]
  -DXUD_CORE_CLOCK=600
  -lquadflash
)

set(APP_DEPENDENT_MODULES "ai_tools"  "lib_xud(v2.4.0)" )
XMOS_REGISTER_APP()