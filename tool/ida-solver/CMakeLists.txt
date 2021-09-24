get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
	${dialect_libs}
	${conversion_libs}
	MLIROptLib
	marco::frontend
	marco::ida
	marco::mlirlowerer
	sundials
)

add_llvm_executable(ida-solver ida-solver.cpp)

llvm_update_compile_flags(ida-solver)
target_link_libraries(ida-solver PRIVATE ${LIBS})
mlir_check_all_link_libraries(ida-solver)

install(TARGETS ida-solver RUNTIME DESTINATION bin)
