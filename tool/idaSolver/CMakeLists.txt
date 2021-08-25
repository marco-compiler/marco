get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        marco::frontend
        marco::mlirlowerer
        sundials)

add_llvm_executable(idaSolver idaSolver.cpp)

llvm_update_compile_flags(idaSolver)
target_link_libraries(idaSolver PRIVATE ${LIBS})
mlir_check_all_link_libraries(idaSolver)

install(TARGETS idaSolver RUNTIME DESTINATION bin)
