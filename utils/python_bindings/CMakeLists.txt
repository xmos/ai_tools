cmake_minimum_required(VERSION 3.14)

project(tflite2xcore VERSION 1.0.1)

# set host build
set(X86 ON)

include("${PROJECT_SOURCE_DIR}/../cmake/xcore_interpreter.cmake")

# This variable is ignored on platforms other than Apple
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk) 


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


set(ADDITIONAL_TENSORFLOW_SORCES
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/kernels/ethosu.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/all_ops_resolver.cc"
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/micro/kernels/xcore/xcore_extended_interpreter.cc"
)

target_sources(libtflite2xcore
  PRIVATE ${XCORE_INTERPRETER_SOURCES}
  PRIVATE ${ADDITIONAL_TENSORFLOW_SORCES}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/flexbuffer_helpers.cpp"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/xcore_interpreter_c.cpp"
)

target_include_directories(libtflite2xcore
  PRIVATE ${XCORE_INTERPRETER_INCLUDE_DIRS}
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
