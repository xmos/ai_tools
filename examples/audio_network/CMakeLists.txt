cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(audio_network)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_HW_TARGET src/core/xk-audio-316-mc.xn)
set(APP_XSCOPE_SRCS src/config.xscope)

set(APP_INCLUDES
  src
  src/core
  src/extensions
)
set(APP_C_SRCS
  src/data_path.c
  src/dsp.c
  src/dsp_usb.c
  src/log.c
  src/mel.c
  src/extensions/button.c
)
set(APP_CXX_SRCS
  src/support.cpp
  src/model_audioi16.tflite.cpp
)
set(APP_XC_SRCS
  src/extensions/XUD_HAL.xc
  src/extensions/audiohw.xc
  src/extensions/audiostream.xc
)
set(APP_ASM_SRCS
  src/hypot.S
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
  "lib_xua(5.3.0)"
  "lib_i2c(6.4.1)"
  "lib_src(2.7.1)"
  "lib_xcore_math(2.4.1)"
)
XMOS_REGISTER_APP()

include(${CMAKE_CURRENT_LIST_DIR}/../aitools.cmake)
