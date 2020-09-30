cmake_minimum_required(VERSION 3.10)

project(tflite2xcore VERSION 1.0.1)


# This variable is ignored on platforms other than Apple
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk) 

#**********************
# Paths
#**********************
set(GEMMLOWP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/gemmlowp")
set(RUY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/ruy")
set(FLATBUFFERS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/flatbuffers/include")
set(FLATBUFFERS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/flatbuffers/src")
set(TENSORFLOW_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/tensorflow")
set(TENSORFLOW_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/tensorflow")

set(LIB_NN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../lib_nn")
set(LIB_NN_ALT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../lib_nn/lib_nn/api")
set(LIB_NN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../lib_nn")

#**********************
# Build flags
#**********************
set(BUILD_FLAGS "-g -DTF_LITE_DISABLE_X86_NEON -O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS ${BUILD_FLAGS})
set(CMAKE_C_FLAGS "${BUILD_FLAGS} -std=c99")

#**********************
# Build shared library
#**********************

add_library(libtflite2xcore SHARED)
set_target_properties(libtflite2xcore PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(libtflite2xcore PROPERTIES PREFIX "")
target_link_libraries(libtflite2xcore stdc++ m pthread)

#**********************
# Flatbuffer sources
#**********************
set(FLATBUFFER_SOURCES
  "${FLATBUFFERS_SOURCE_DIR}/idl_parser.cpp"
  "${FLATBUFFERS_SOURCE_DIR}/idl_gen_text.cpp"
  "${FLATBUFFERS_SOURCE_DIR}/util.cpp"
)

#**********************
# TensorFlow sources
#**********************
set(TENSORFLOW_LITE_SOURCES
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/c/common.c"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/core/api/error_reporter.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/core/api/flatbuffer_conversions.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/core/api/op_resolver.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/core/api/tensor_utils.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/kernels/kernel_util.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc"
)

file (GLOB TENSORFLOW_LITE_MICRO_SOURCES 
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/*.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/kernels/*.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/*.cc"
)
list(FILTER TENSORFLOW_LITE_MICRO_SOURCES EXCLUDE REGEX ".test.")  # filter out test source files

#**********************
# XMOS sources
#**********************
file (GLOB XCORE_SOURCES 
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/kernels/xcore/*.cc"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/asm/*.c"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/c/*.c"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/c/util/*.c"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/c/util/deep/*.c"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/c/util/depthwise/*.c"
  "${LIB_NN_SOURCE_DIR}/lib_nn/src/c/util/shallow/*.c"
)

#TODO: Remove me when compilation error is fixed
list(REMOVE_ITEM XCORE_SOURCES "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/kernels/xcore/xcore_bsign.cc")

target_sources(libtflite2xcore
  PRIVATE ${FLATBUFFER_SOURCES}
  PRIVATE ${TENSORFLOW_LITE_SOURCES}
  PRIVATE ${TENSORFLOW_LITE_MICRO_SOURCES}
  PRIVATE ${XCORE_SOURCES}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/flexbuffer_helpers.cpp"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/xcore_interpreter_c.cpp"
)

target_include_directories(libtflite2xcore
  PRIVATE ${FLATBUFFERS_INCLUDE_DIR}
  PRIVATE ${GEMMLOWP_INCLUDE_DIR}
  PRIVATE ${RUY_INCLUDE_DIR}
  PRIVATE ${TENSORFLOW_INCLUDE_DIR}
  PRIVATE ${LIB_NN_ALT_INCLUDE_DIR}
  PRIVATE ${LIB_NN_INCLUDE_DIR}
  PRIVATE ${LIB_OPS_INCLUDE_DIR}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/linux")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/macos")
else() # must be Windows
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/windows")
endif()

install(TARGETS libtflite2xcore DESTINATION ${INSTALL_DIR})
