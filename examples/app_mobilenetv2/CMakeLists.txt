cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_mobilenetv2)

# HW
set(APP_HW_TARGET XK-EVK-XU316)

# base flags
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
  # -DTFLMC_XCORE_PROFILE
)

# dependencies and register
set(APP_INCLUDES src)
set(APP_DEPENDENT_MODULES "")
XMOS_REGISTER_APP()

# custom libs
include(aitools.cmake)
