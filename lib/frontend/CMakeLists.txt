set(SOURCES
    CompilerInstance.cpp
    CompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    Options.cpp
    TextDiagnostic.cpp
    TextDiagnosticBuffer.cpp
    TextDiagnosticPrinter.cpp)

marco_add_library(frontend SHARED
    ${SOURCES}

    DEPENDS
    MARCOFrontendOptions

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
