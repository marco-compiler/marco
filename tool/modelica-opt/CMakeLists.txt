set(SOURCES
        modelica-opt.cpp)

marco_add_tool(modelica-opt ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(modelica-opt
    PRIVATE ${dialect_libs}
    PRIVATE ${extension_libs}
    PRIVATE ${conversion_libs}
    PRIVATE MLIROptLib
    PRIVATE marco::codegen
    PRIVATE marco::ast)

mlir_check_all_link_libraries(modelica-opt)
