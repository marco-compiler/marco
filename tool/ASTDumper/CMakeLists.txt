add_executable(ASTDumper src/Main.cpp)
add_executable(modelica::ASTDumper ALIAS ASTDumper)

llvm_map_components_to_libnames(llvm_ast_dumper_libs core support ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(ASTDumper PUBLIC ${llvm_ast_dumper_libs} modelica::frontend)
target_compile_features(ASTDumper PUBLIC cxx_std_17)

include(GNUInstallDirs)
INSTALL(TARGETS ASTDumper RUNTIME DESTINATION bin)

add_subdirectory(test)
