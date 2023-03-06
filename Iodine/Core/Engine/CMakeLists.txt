file(GLOB EngineSources *.cc)
file(GLOB EngineHeaders *.h)

message("Adding ${EngineSources} ${EngineHeaders}")
add_library(Engine ${EngineSources} ${EngineHeaders})

target_include_directories(Engine PUBLIC ../../../third_party)
target_include_directories(Engine PUBLIC ../../../Iodate)

find_package(pybind11 CONFIG REQUIRED)
target_link_libraries(Engine PUBLIC pybind11::lto pybind11::embed pybind11::module pybind11::headers)