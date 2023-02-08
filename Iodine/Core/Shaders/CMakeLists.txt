cmake_minimum_required(VERSION 3.8)

file(GLOB ShaderSources *.cc)
file(GLOB ShaderHeaders *.h)

message("Adding: ${ShaderSources} ${ShaderHeaders}")


add_library(Shaders ${ShaderSources} ${ShaderHeaders})


find_package(glad CONFIG REQUIRED)
target_link_libraries(Shaders PRIVATE glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(Shaders PRIVATE glfw)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(Shaders PRIVATE imgui::imgui)

find_package(glm CONFIG REQUIRED)
target_link_libraries(Shaders PRIVATE glm::glm)

target_include_directories(Shaders PUBLIC ../../../third_party)
