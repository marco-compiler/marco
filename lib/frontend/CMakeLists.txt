set(SOURCES
    src/CompilerInstance.cpp
    src/CompilerInvocation.cpp
    src/FrontendAction.cpp
    src/FrontendActions.cpp
    src/FrontendOptions.cpp
    src/TextDiagnostic.cpp
    src/TextDiagnosticBuffer.cpp
    src/TextDiagnosticPrinter.cpp)

marco_add_library_new(frontend SHARED
    ${SOURCES}

    DEPENDS

    LINK_LIBS
    clangDriver
    marco::ast
    marco::utils

    LINK_COMPONENTS
    Option
    Support)

#marco_add_library(frontend SHARED ${SOURCES})

#marco_link_llvm_libs(frontend USE_SHARED core support )

#target_link_libraries(frontend
#    PUBLIC clangBasic
#    PUBLIC clangDriver
#    PUBLIC marco::ast
#    PUBLIC marco::utils)

#add_subdirectory(test)
