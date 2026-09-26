cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(ai_tools_examples)


add_subdirectory(app_flash_single_model)
add_subdirectory(app_flash_two_models)
add_subdirectory(app_flash_two_models_one_arena)
add_subdirectory(app_mobilenetv2)
add_subdirectory(app_no_flash)
add_subdirectory(app_profiling)
add_subdirectory(app_single_model_on_one_tile_and_DDR)
add_subdirectory(app_single_model_on_two_tiles)

# Needs special reqs
# add_subdirectory(app_audio_network)
# add_subdirectory(app_yolov8_classification)
