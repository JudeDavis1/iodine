cmake_minimum_required(VERSION 3.8)

file(GLOB APISources *.cc)
file(GLOB APIHeaders *.h)

message("Adding: ${APISources} ${APIHeaders}")


add_library(API ${APISources} ${APIHeaders})


find_package(glad CONFIG REQUIRED)
target_link_libraries(API PRIVATE glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(API PRIVATE glfw)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(API PRIVATE imgui::imgui)

find_package(glm CONFIG REQUIRED)
target_link_libraries(API PRIVATE glm::glm)

target_include_directories(API PUBLIC ../../../third_party)