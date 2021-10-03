set(SOURCES
        marco.cpp)

marco_add_tool(marco ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib)

target_link_libraries(marco
        PRIVATE marco::frontend
        PRIVATE marco::mlirlowerer
        PRIVATE marco::utils
        PRIVATE ${MLIR_LIBS})

mlir_check_all_link_libraries(marco)

install(TARGETS marco RUNTIME DESTINATION bin)
