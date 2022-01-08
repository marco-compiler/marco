set(SOURCES
    marco.cpp
    marco-frontend.cpp)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend
    PRIVATE marco::frontendTool)

clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver)

install(TARGETS marco-driver RUNTIME DESTINATION bin)
