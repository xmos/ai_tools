cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_ddr)

# hw and xscope
set(APP_HW_TARGET src/XCORE-AI-EXPLORER-800.xn)
set(APP_XSCOPE_SRCS src/config.xscope)

# export model and sources
execute_process(
  COMMAND python generate_optimized_cpp_for_xcore.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
set(APP_INCLUDES 
  src
)

set(APP_XC_SRCS "")

set(APP_C_SRCS
  src/model_weights.c
)
set(APP_CXX_SRCS
  src/model.tflite.cpp
  src/main.cpp
)

# Base flags
list(APPEND APP_COMPILER_FLAGS
  -O3
  -g
  -report
  -Wall
  -fxscope
  -mcmodel=large
  -Wno-xcore-fptrgroup
  -lquadflash 
  # -DTFLMC_XCORE_PROFILE
)

# dependencies and register
set(APP_DEPENDENT_MODULES "")
XMOS_REGISTER_APP()

# Custom libs
include(aitools.cmake)
