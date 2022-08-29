cmake_minimum_required(VERSION 3.8)

file(GLOB UtilsSources *.cc)
file(GLOB UtilsHeaders *.h)

message("${UtilsSources} ${UtilsHeaders}")


add_library(Utils ${UtilsSources} ${UtilsHeaders})


find_package(glad CONFIG REQUIRED)
target_link_libraries(Utils PRIVATE glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(Utils PRIVATE glfw)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(Utils PRIVATE imgui::imgui)

find_package(glm CONFIG REQUIRED)
target_link_libraries(Utils PRIVATE glm::glm)