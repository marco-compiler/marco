set(SOURCES
    Verifier.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

marco_add_tool(modelica-verifier ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    marco::JIT
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRToLLVMIRTranslationRegistration)

target_link_libraries(modelica-verifier PRIVATE ${LIBS})
