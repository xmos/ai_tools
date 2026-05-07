# xcommon cmake
set(LIB_NAME ai_tools)
set(LIB_VERSION 1.4.3.dev39)
set(LIB_INCLUDES "")
XMOS_REGISTER_MODULE()

# link library
include(${CMAKE_CURRENT_LIST_DIR}/cfg_target.cmake)
