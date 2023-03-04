file(GLOB EngineSources *.cc)
file(GLOB EngineHeaders *.h)

message("Adding ${EngineSources} ${EngineHeaders}")
add_library(Engine ${EngineSources} ${EngineHeaders})

target_include_directories(Renderer PUBLIC ../../../third_party)
target_include_directories(Renderer PUBLIC ../../../Iodate)
