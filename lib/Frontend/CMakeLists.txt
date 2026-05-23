set(SOURCES
    Instrumentation/VerificationModelEmitter.cpp
    Passes/AggressiveLICM.cpp
    Passes/HeapFunctionsReplacement.cpp
    Passes/Verifier.cpp
    CompilerInstance.cpp
    CompilerInvocation.cpp
    DiagnosticHandler.cpp
    ExecuteCompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    LanguageOptions.cpp
    Passes.cpp
    SimulationOptions.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

marco_add_library(Frontend
    ${SOURCES}

    CLANG_LIBS
    clangBasic
    clangDriver

    MLIR_LIBS
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRLLVMToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
    MLIRTargetLLVMIRImport

    LINK_LIBS
    marco::ASTBaseModelica
    marco::ParserBaseModelica
    marco::IO
    marco::Lowering
    marco::variableFilter

    LINK_COMPONENTS
    Option
    Support

    DEPENDS
    MLIRFrontendPassesIncGen)
