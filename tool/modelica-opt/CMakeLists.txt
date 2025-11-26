set(SOURCES
    modelica-opt.cpp)

marco_add_tool(modelica-opt ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    marco::codegen
    marco::frontend
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRToLLVMIRTranslationRegistration)

target_link_libraries(modelica-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(modelica-opt)
