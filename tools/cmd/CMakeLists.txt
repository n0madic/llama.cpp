# llama-cmd: ollama-style management subcommands (pull/run/list/ps/stop/rm/create/show)
#
# This builds only a library (no standalone executable). The commands are exposed
# exclusively through the unified `llama` binary (see app/), which links this library
# and registers the entry points in its subcommand table.

set(TARGET llama-cmd)

add_library(${TARGET} cmd.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC  llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()
