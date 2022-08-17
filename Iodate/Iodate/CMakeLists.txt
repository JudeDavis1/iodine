# CMakeList.txt : CMake project for Iodate, include source and define
# project specific logic here.
#

cmake_minimum_required (VERSION 3.8)

include ("C:\\Program Files\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")

# Add source to this project's executable.
add_executable (Iodate "Main.cc" "Application.cc" "Application.h" "Window.cc" )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Iodate PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
find_package(glad CONFIG REQUIRED)
target_link_libraries(Iodate PRIVATE glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(Iodate PRIVATE glfw)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(Iodate PRIVATE imgui::imgui)

include (${CMAKE_SOURCE_DIR}/CMake/functions.cmake)

INCLUDE_LOCAL_DIRS(Iodate)

