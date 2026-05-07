cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_mobilenetv2)

set(XMOS_SANDBOX_DIR    ${CMAKE_CURRENT_LIST_DIR}/../..)

# target
set(APP_HW_TARGET XK-EVK-XU316)

# flags
list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -Wall
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -lquadflash 
  -DSHARED_TENSOR_ARENA
  # -DTFLMC_XCORE_PROFILE # enable to profile
)

# dependencies and register
set(APP_INCLUDES src)
set(APP_DEPENDENT_MODULES "ai_tools")
XMOS_REGISTER_APP()
