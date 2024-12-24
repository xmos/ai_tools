# Get path of xmos ai tools
set(CMD "\
import os; \
import xmos_ai_tools.runtime as rt; \
print(os.path.dirname(rt.__file__)) \
")

execute_process(
    COMMAND python -c "${CMD}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE XMOS_AITOOLSLIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

# Add tflite_micro 
set(XMOS_AITOOLSLIB_PATH_CMAKE "${XMOS_AITOOLSLIB_PATH}/buildfiles/aitoolslib.cmake")

if(XMOS_AITOOLSLIB_PATH STREQUAL "")
    message(FATAL_ERROR "Path to XMOS AI tools NOT found")
elseif(NOT EXISTS  ${XMOS_AITOOLSLIB_PATH_CMAKE})
    message(FATAL_ERROR "Cmake file 'aitoolslib.cmake' NOT found in this path")
else()
    message(STATUS "\nFound python package xmos-ai-tools at: ${XMOS_AITOOLSLIB_PATH}")
    include(${XMOS_AITOOLSLIB_PATH_CMAKE})
    set(LIB_NAME tflite_micro)
    add_library(${LIB_NAME} STATIC IMPORTED GLOBAL)
    target_compile_definitions(${LIB_NAME} INTERFACE ${XMOS_AITOOLSLIB_DEFINITIONS})
    set_target_properties(${LIB_NAME}  PROPERTIES
        LINKER_LANGUAGE CXX
        IMPORTED_LOCATION ${XMOS_AITOOLSLIB_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES ${XMOS_AITOOLSLIB_INCLUDES})
endif()

# Link aitools with the targets
foreach(target ${APP_BUILD_TARGETS})
    message(STATUS "Linking ${target} with ${LIB_NAME}")
    target_link_libraries(${target} PRIVATE ${LIB_NAME})
endforeach()
