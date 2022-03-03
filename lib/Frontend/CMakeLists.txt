set(SOURCES
    CompilerInstance.cpp
    CompilerInvocation.cpp
    ExecuteCompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    Options.cpp
    SimulationOptions.cpp
    TextDiagnostic.cpp
    TextDiagnosticBuffer.cpp
    TextDiagnosticPrinter.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIROpenMPToLLVMIRTranslation)

marco_add_library(frontend SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::codegen
    marco::utils
    MARCOFrontendOptions

    LINK_LIBS
    PRIVATE clangDriver
    PRIVATE marco::ast
    PRIVATE marco::codegen
    PRIVATE marco::utils
    PRIVATE ${MLIR_LIBS}

    LINK_COMPONENTS
    Option
    Support)

mlir_check_all_link_libraries(frontend)