# Check if defined lib_name
if(NOT DEFINED LIB_NAME)
    message(FATAL_ERROR "LIB_NAME is not defined.")
endif()

# Get path of xmos ai tools
set(CMD "\
import os; \
import xmos_ai_tools.runtime as rt; \
print(os.path.dirname(rt.__file__)) \
")
execute_process(
    COMMAND python -c "${CMD}"
    OUTPUT_VARIABLE XMOS_AITOOLSLIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

# Check lib_version matches
set(CMD "\
import xmos_ai_tools; \
print(xmos_ai_tools.__version__) \
")
execute_process(
  COMMAND python -c "${CMD}"
  OUTPUT_VARIABLE PY_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
if(NOT "${PY_VERSION}" STREQUAL "${LIB_VERSION}")
  message(WARNING "Version mismatch: LIB_VERSION:${LIB_VERSION} != PY_VERSION:${PY_VERSION}")
else()
  message(VERBOSE "Version OK: ${LIB_VERSION}")
endif()

# Library definitions for xmos ai tools
set(XMOS_AITOOLSLIB_DEFINITIONS
    "TF_LITE_STATIC_MEMORY"
    "TF_LITE_STRIP_ERROR_STRINGS"
    "XCORE"
    "NO_INTERPRETER"
)

set(XMOS_AITOOLSLIB_COMPILE_OPTIONS "")
set(XMOS_AITOOLSLIB_LINK_OPTIONS "")

# Set static library path based on architecture
set(lib_path "${XMOS_AITOOLSLIB_PATH}/lib")
if(APP_BUILD_ARCH STREQUAL "xs3a" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL XCORE_XS3A)
    set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro_xs3a.a")
elseif(APP_BUILD_ARCH STREQUAL "vx4b")
    set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro_vx4b.a")
    list(APPEND XMOS_AITOOLSLIB_DEFINITIONS "__VX4A__") # TODO legacy code
    set(XMOS_AITOOLSLIB_COMPILE_OPTIONS -Wfptrgroup -ffunction-sections -fdata-sections -Os)
    set(XMOS_AITOOLSLIB_LINK_OPTIONS -lxc -Wl,--gc-sections)
else()
    set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libhost_xtflitemicro.a")
endif()

# Set the include path for the library
set(XMOS_AITOOLSLIB_INCLUDES "${XMOS_AITOOLSLIB_PATH}/include")

# Link library
if(NOT TARGET ${LIB_NAME})
add_library(${LIB_NAME} STATIC IMPORTED GLOBAL)
target_compile_definitions(${LIB_NAME} INTERFACE ${XMOS_AITOOLSLIB_DEFINITIONS})
target_compile_options(${LIB_NAME} INTERFACE ${XMOS_AITOOLSLIB_COMPILE_OPTIONS})
target_link_options(${LIB_NAME} INTERFACE ${XMOS_AITOOLSLIB_LINK_OPTIONS})
set_target_properties(${LIB_NAME} PROPERTIES
    LINKER_LANGUAGE CXX
    IMPORTED_LOCATION ${XMOS_AITOOLSLIB_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES ${XMOS_AITOOLSLIB_INCLUDES})
endif()

# Link aitools with the targets
foreach(target ${APP_BUILD_TARGETS})
    message(STATUS "Linking ${target} with ${LIB_NAME}")
    target_link_libraries(${target} PRIVATE ${LIB_NAME})
endforeach()
