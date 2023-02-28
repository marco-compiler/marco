set(SOURCES
    CompilerInstance.cpp
    CompilerInvocation.cpp
    ExecuteCompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    Options.cpp
    SimulationOptions.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRLLVMToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
    MLIRTargetLLVMIRImport)

marco_add_library(frontend
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::parser
    marco::codegen
    marco::variableFilter
    MARCOFrontendOptions

    LINK_LIBS
    marco::ast
    marco::ASTPasses
    marco::parser
    marco::codegen
    marco::variableFilter
    ${MLIR_LIBS}

    LINK_COMPONENTS
    Option
    Support)

mlir_check_all_link_libraries(frontend)
