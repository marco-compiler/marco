set(SOURCES
    marco.cpp
    marco-cc1.cpp
    marco-mc1.cpp)

set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Core
    MC
    MCParser
    Support)

marco_add_tool(marco ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(marco
    PRIVATE
    marco::frontend
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
    MLIRCustomTransforms
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRToLLVMIRTranslationRegistration)

clang_target_link_libraries(marco
    PRIVATE
    clangDriver
    clangBasic
    clangFrontend
    clangFrontendTool)
