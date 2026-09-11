set(LIB_NAME ai_tools)

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

if(XMOS_AITOOLSLIB_PATH STREQUAL "")
	message(FATAL_ERROR "Path to XMOS AI Tools library and headers not found")
endif()

set(ENV{XMOS_AITOOLSLIB_PATH} "${XMOS_AITOOLSLIB_PATH}")
message(STATUS "XMOS_AITOOLSLIB_PATH=${XMOS_AITOOLSLIB_PATH}")

if(DEFINED LIB_VERSION)
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
endif()

set(XMOS_AITOOLSLIB_DEFINITIONS
	"TF_LITE_STATIC_MEMORY"
	"TF_LITE_STRIP_ERROR_STRINGS"
	"XCORE"
	"NO_INTERPRETER"
)

set(XMOS_AITOOLSLIB_COMPILE_OPTIONS "")
set(XMOS_AITOOLSLIB_LINK_OPTIONS "")

set(lib_path "${XMOS_AITOOLSLIB_PATH}/lib")
if(APP_BUILD_ARCH STREQUAL "xs3a" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL XCORE_XS3A)
	set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro_xs3a.a")
	if(NOT EXISTS "${XMOS_AITOOLSLIB_LIBRARIES}")
		set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro.a")
	endif()
elseif(APP_BUILD_ARCH STREQUAL "vx4b")
	set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro_vx4b.a")
	list(APPEND XMOS_AITOOLSLIB_DEFINITIONS "__VX4A__")
	set(XMOS_AITOOLSLIB_COMPILE_OPTIONS -Wfptrgroup -ffunction-sections -fdata-sections -Os)
	set(XMOS_AITOOLSLIB_LINK_OPTIONS -lxc -Wl,--gc-sections)
elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL XCORE_XS)
	# CMAKE_SYSTEM_PROCESSOR will be defined to be XCORE_XS
	# by xcommon_cmake regardless of the target architectire
	# hardcode to xs3a for now
	set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libxtflitemicro_xs3a.a")
else()
	set(XMOS_AITOOLSLIB_LIBRARIES "${lib_path}/libhost_xtflitemicro.a")
endif()

set(XMOS_AITOOLSLIB_INCLUDES "${XMOS_AITOOLSLIB_PATH}/include")

if(NOT EXISTS "${XMOS_AITOOLSLIB_LIBRARIES}")
	message(FATAL_ERROR "XMOS AI Tools library not found: ${XMOS_AITOOLSLIB_LIBRARIES}")
endif()
if(NOT EXISTS "${XMOS_AITOOLSLIB_INCLUDES}")
	message(FATAL_ERROR "XMOS AI Tools include directory not found: ${XMOS_AITOOLSLIB_INCLUDES}")
endif()

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

foreach(target ${APP_BUILD_TARGETS})
	message(STATUS "Linking ${target} with ${LIB_NAME}")
	target_link_libraries(${target} PRIVATE ${LIB_NAME})
endforeach()
