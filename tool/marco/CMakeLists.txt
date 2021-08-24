get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        marco::frontend
        marco::mlirlowerer)

add_llvm_executable(marco marco.cpp)

llvm_update_compile_flags(marco)
target_link_libraries(marco PRIVATE ${LIBS})
mlir_check_all_link_libraries(marco)

install(TARGETS marco RUNTIME DESTINATION bin)
