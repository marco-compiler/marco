set(SOURCES
    src/ExecuteCompilerInvocation.cpp)

marco_add_library(frontendTool SHARED ${SOURCES})

marco_link_llvm_libs(frontendTool USE_SHARED core support)

target_link_libraries(frontendTool
    PUBLIC marco::frontend)

#add_subdirectory(test)
