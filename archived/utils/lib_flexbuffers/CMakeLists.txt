cmake_minimum_required(VERSION 3.14)

project(libflexbuffers VERSION 1.0.1)

# This variable is ignored on platforms other than Apple
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk) 

#**********************
# Paths
#**********************

set(AI_TOOLS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

#**********************
# Build flags
#**********************
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-g -O0")
set(CMAKE_C_FLAGS "-std=c99 -g -O0")

#**********************
# Build shared library
#**********************

add_library(libflexbuffers SHARED)
set_target_properties(libflexbuffers PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(libflexbuffers PROPERTIES PREFIX "")
target_link_libraries(libflexbuffers stdc++ m)


target_sources(libflexbuffers
  PRIVATE "${AI_TOOLS_ROOT_DIR}/third_party/flatbuffers/src/util.cpp"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/flexbuffer_helpers.cpp"
)

target_include_directories(libflexbuffers
  PRIVATE "${AI_TOOLS_ROOT_DIR}/third_party/flatbuffers/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/linux")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/macos")
else() # must be Windows
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/../../tflite2xcore/tflite2xcore/libs/windows")
endif()

install(TARGETS libflexbuffers DESTINATION ${INSTALL_DIR})
