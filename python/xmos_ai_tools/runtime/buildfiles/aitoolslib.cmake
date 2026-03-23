set(XMOS_AITOOLSLIB_DEFINITIONS
  "TF_LITE_STATIC_MEMORY"
	"TF_LITE_STRIP_ERROR_STRINGS"
	"XCORE"
	"NO_INTERPRETER"
)

if(${APP_BUILD_ARCH} STREQUAL xs3a OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL XCORE_XS3A)
	set(XMOS_AITOOLSLIB_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../lib/libxtflitemicro_xs3a.a")
elseif(${APP_BUILD_ARCH} STREQUAL vx4b)
	set(XMOS_AITOOLSLIB_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../lib/libxtflitemicro_vx4b.a")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL XCORE_XS)
	# CMAKE_SYSTEM_PROCESSOR will be defined to be XCORE_XS
	# by xcommon_cmake regardless of the target architectire
	# hardcode to xs3a for now
	set(XMOS_AITOOLSLIB_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../lib/libxtflitemicro_xs3a.a")
else()
	if(UNIX)
		set(XMOS_AITOOLSLIB_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../lib/libhost_xtflitemicro.a")
	else()
		set(XMOS_AITOOLSLIB_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../lib/host_xtflitemicro.lib")
	endif()
endif()
set(XMOS_AITOOLSLIB_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/../include")
