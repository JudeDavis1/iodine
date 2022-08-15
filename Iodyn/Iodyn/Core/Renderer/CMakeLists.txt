cmake_minimum_required (VERSION 3.8)

add_library(Iodyn "Renderer.cc" "Renderer.h")
target_include_directories(Iodyn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})