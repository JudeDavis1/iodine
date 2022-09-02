cmake_minimum_required(VERSION 3.8)

file(GLOB RendererSources *.cc)
file(GLOB RendererHeaders *.h)


add_library(Renderer ${RendererSources} ${RendererHeaders} "../Shaders/Shader.h" "../Shaders/Shader.cc")


find_package(glad CONFIG REQUIRED)
target_link_libraries(Renderer PUBLIC glad::glad)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(Renderer PUBLIC glfw)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(Renderer PUBLIC imgui::imgui)

find_package(glm CONFIG REQUIRED)
target_link_libraries(Renderer PUBLIC glm::glm)

target_include_directories(Renderer PUBLIC "../../")
target_link_libraries(Renderer PUBLIC API)


target_include_directories(Renderer PUBLIC ../../../third_party)
