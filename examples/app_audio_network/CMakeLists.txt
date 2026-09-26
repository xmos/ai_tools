cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_audio_network)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/export.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

if(NOT DEFINED APP_HW_TARGET)
  set(APP_HW_TARGET src/core/xk-audio-316-mc.xn)
endif()
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES
  src
  src/core
  src/extensions
)

list(APPEND APP_COMPILER_FLAGS
  -fcomment-asm
  -O3
  -report
  -lquadflash
  -g
  -DADAT_TX_USE_SHARED_BUFF=1
  -DQUAD_SPI_FLASH=1
  -DMIXER=0
  -fxscope
)

set(APP_DEPENDENT_MODULES
  "ai_tools"
  "lib_xua(5.3.0)"
  "lib_i2c(6.4.1)"
  "lib_src(2.7.1)"
  "lib_xcore_math(2.4.1)"
)
XMOS_REGISTER_APP()
