get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        modelica::frontend
        modelica::mlirlowerer)

add_llvm_executable(modelica-opt modelica-opt.cpp)

llvm_update_compile_flags(modelica-opt)
target_link_libraries(modelica-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(modelica-opt)

install(TARGETS modelica-opt RUNTIME DESTINATION bin)
