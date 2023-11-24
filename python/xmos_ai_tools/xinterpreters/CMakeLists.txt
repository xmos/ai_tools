cmake_minimum_required(VERSION 3.14)

set(CMAKE_BUILD_TYPE "Release")
project(xtflm_python VERSION 1.0.1)

# This variable is ignored on platforms other than Apple
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
# set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)

#**********************
# Build flags
#**********************
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(BUILD_FLAGS "/O2")
else()
  set(BUILD_FLAGS
              "-g"
              "-O2")
endif()


if(DEFINED ENV{CMAKE_ENABLE_DARWIN_TARGET_ARM64})
  set(BUILD_FLAGS ${BUILD_FLAGS} "-target" "arm64-apple-macos11")
endif()

set(TOP_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/lib_tflite_micro")

include(${TOP_DIR}/cmakefiles/xtflm.cmake)

#**********************
# Build host library
#**********************
add_library(host_xtflitemicro STATIC)
set(DEFINTIONS
    "__xtflm_conf_h_exists__"
    "NO_INTERPRETER"
    "TF_LITE_STATIC_MEMORY"
    "TF_LITE_DISABLE_X86_NEON"
    "TF_LITE_STRIP_ERROR_STRINGS"
    "NN_USE_REF"
)
list(APPEND DEFINTIONS "FLATBUFFERS_LOCALE_INDEPENDENT=0")
target_compile_features(host_xtflitemicro PUBLIC cxx_std_17)

target_sources(host_xtflitemicro
    PRIVATE ${TFLM_KERNEL_SOURCES}
    PRIVATE ${TFLITE_SOURCES}
    PRIVATE ${NN_SOURCES}
    PRIVATE ${XTFLIB_KERNEL_SOURCES}
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(BUILD_FLAGS ${BUILD_FLAGS} "-fPIC")
endif()
target_compile_options(host_xtflitemicro PRIVATE ${BUILD_FLAGS})

target_link_options(host_xtflitemicro PRIVATE ${BUILD_FLAGS})
target_compile_definitions(host_xtflitemicro PUBLIC
    ${DEFINTIONS}
)

target_include_directories(host_xtflitemicro
    PRIVATE ${ALL_INCLUDES}
)

set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/lib")
install(TARGETS host_xtflitemicro DESTINATION ${INSTALL_DIR})

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

set(DEFINTIONS
    "__xtflm_conf_h_exists__"
    "TF_LITE_DISABLE_X86_NEON"
    "NN_USE_REF"
)
target_compile_definitions(xtflm_python PUBLIC ${DEFINTIONS})
target_compile_features(xtflm_python PUBLIC cxx_std_17)
target_compile_options(xtflm_python PRIVATE ${BUILD_FLAGS})
target_link_options(xtflm_python PRIVATE ${BUILD_FLAGS})

target_sources(xtflm_python
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/dll_interpreter.cc"
  PRIVATE ${ALL_SOURCES}
)

list(APPEND ALL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/src")

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

