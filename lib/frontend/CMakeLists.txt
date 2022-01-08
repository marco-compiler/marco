get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(SOURCES
    CompilerInstance.cpp
    CompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    Options.cpp
    SimulationOptions.cpp
    TextDiagnostic.cpp
    TextDiagnosticBuffer.cpp
    TextDiagnosticPrinter.cpp)

set(MLIR_LIBS
    MLIRIR
    MLIRMathToLLVM
    MLIROpenMPToLLVM
    MLIROpenMPToLLVMIRTranslation
    MLIRSCF
    MLIRSCFToStandard
    MLIRSCFToOpenMP
    MLIRStandard
    MLIRStandardToLLVM
    MLIRTransforms
    MLIRTranslation
    MLIRExecutionEngine)

marco_add_library(frontend SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::codegen
    marco::utils
    MARCOFrontendOptions

    LINK_LIBS
    clangDriver
    marco::ast
    marco::codegen
    marco::utils
    ${dialect_libs}
    ${MLIR_LIBS}

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
