set(SOURCES
    marco.cpp
    marco-frontend.cpp)

marco_add_tool(marco ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib)

target_link_libraries(marco
        PRIVATE marco::ast
        PRIVATE marco::codegen
        PRIVATE marco::frontend
        PRIVATE marco::frontendTool
        PRIVATE marco::utils
        PRIVATE ${MLIR_LIBS})

clang_target_link_libraries(marco
    PRIVATE
    clangDriver)

mlir_check_all_link_libraries(marco)

install(TARGETS marco RUNTIME DESTINATION bin)
