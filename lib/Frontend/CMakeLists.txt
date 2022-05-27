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

set(MLIR_LIBS
    ${dialect_libs}
    MLIRExecutionEngine
    MLIROpenMPToLLVMIRTranslation)

marco_add_library(frontend SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::parser
    marco::codegen
    marco::variableFilter
    marco::utils
    MARCOFrontendOptions

    LINK_LIBS
    PUBLIC marco::ast
    PUBLIC marco::parser
    PRIVATE clangDriver
    PRIVATE marco::codegen
    PRIVATE marco::variableFilter
    PRIVATE marco::utils
    PRIVATE ${MLIR_LIBS}

    LINK_COMPONENTS
    Option
    Support)

mlir_check_all_link_libraries(frontend)
