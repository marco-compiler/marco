set(SOURCES
    marco-lsp.cpp)

marco_add_tool(marco-lsp ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${dialect_libs}
    ${extension_libs}
    MLIRLspServerLib
    MLIRRegisterAllDialects)

target_link_libraries(marco-lsp PRIVATE ${LIBS})
mlir_check_all_link_libraries(marco-lsp)
