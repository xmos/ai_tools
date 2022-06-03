cmake_minimum_required(VERSION 3.14)

set(CMAKE_BUILD_TYPE "Release")
project(xtflm_python VERSION 1.0.1)

# set host build
set(X86 ON)

# This variable is ignored on platforms other than Apple
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk) 

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "C++ Compiler Base Flags" FORCE)


#**********************
# Build flags
#**********************
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(BUILD_FLAGS "/DTF_LITE_DISABLE_X86_NEON /D__xtflm_conf_h_exists__ /O2 /DNN_USE_REF")
else()
  set(BUILD_FLAGS "-g -DTF_LITE_DISABLE_X86_NEON -D__xtflm_conf_h_exists__ -O3 -DNN_USE_REF")
endif()


if(DEFINED ENV{CMAKE_ENABLE_DARWIN_TARGET_ARM64})
  set(BUILD_FLAGS "${BUILD_FLAGS} -target arm64-apple-macos11")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS ${BUILD_FLAGS})
set(CMAKE_C_FLAGS "${BUILD_FLAGS} -std=c99")

#**********************
# Build shared library
#**********************

add_library(xtflm_python SHARED)
set_target_properties(xtflm_python PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(xtflm_python PROPERTIES PREFIX "")
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  target_link_libraries(xtflm_python stdc++ m pthread -static-libgcc)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(xtflm_python stdc++ m pthread)
endif()

set(TOP_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/lib_tflite_micro")

include(${TOP_DIR}/cmakefiles/xtflm.cmake)

target_sources(xtflm_python
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/dll_interpreter.cc"
  PRIVATE ${ALL_SOURCES}
)

target_include_directories(xtflm_python
  PRIVATE ${ALL_INCLUDES}
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/libs/linux")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/libs/macos")
else() # must be Windows
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/libs/windows")
endif()

install(TARGETS xtflm_python DESTINATION ${INSTALL_DIR})

